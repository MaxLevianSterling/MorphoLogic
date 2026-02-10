# src/morphologic/core.py
from __future__ import annotations

# General imports (stdlib)
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Third-party imports
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from tqdm import tqdm

# Local imports
from .aggregate import Aggregator
from .config import Config
from .exceptions import DataNotFound
from .integration import extend_dataframe
from .io import load_image_bundle, find_rois_for_folder, discover_traces, read_swc_file
from .neurite import Neurite
from .puncta import assign_puncta
from .soma import Soma
from .structure import create_data_structure
from .topology import soma_center, split_by_neurites
from .visualization import Visualization


class TraceDiscovery:
    """Data access for trace/SWC files (discovery, optional reading)."""

    def __init__(self, cfg: Config) -> None:
        """
        Initialize the repository with a given configuration.

        Args:
            cfg (Config): Global configuration that provides directory and
                filename patterns used to locate SWC/trace files.
        """
        self.cfg = cfg

    def discover_swc(self):
        """
        Discover and group SWC/trace files under the configured directory.

        Uses the configured base extension and '.swc' to find candidate files
        and returns a nested structure compatible with `create_data_structure`.

        Returns:
            list[Group]: Groups of (image_root, image_path, path_structure, trace_paths).

        Raises:
            DataNotFound: If no SWC/trace files can be found under the
                configured directory.
        """
        groups = discover_traces(
            self.cfg.pathing.directory,
            self.cfg.pathing.image_suffix,
            ".swc",
        )
        if not groups:
            raise DataNotFound(f"No SWC/trace files under {self.cfg.pathing.directory}")
        return groups

    def discover_pkl(self):
        """
        Discover and group processed per-cell PKLs under cfg.pathing.directory/Processed.

        Uses '.pkl' base and criteria extensions to group pickled per-cell outputs.

        Returns:
            list[Group]: Groups of (image_root, image_path, path_structure, pkl_paths).

        Raises:
            DataNotFound: If no processed PKLs can be found under the
                configured directory.
        """
        processed_dir = self.cfg.pathing.directory / "Processed"
        groups = discover_traces(processed_dir, ".pkl", ".pkl")
        if not groups:
            raise DataNotFound(f"No processed PKLs under {processed_dir}")
        return groups


class Processor:
    """Encapsulates the per-cell processing logic."""

    def __init__(self, cfg: Config) -> None:
        """
        Initialize the Processor with configuration and a Plotter helper.

        Args:
            cfg (Config): Global configuration controlling parameters,
                visualization, and processing options.
        """
        self.cfg = cfg
        self.plotter = Plotter(cfg)

    def run(self, data_structure: Dict[str, Any], n_cells: int, progress_cb: Optional[Callable[[int, int, float], None]] = None,) -> None:
        """
        Process all discovered cells, grouped per image.

        Use:
            Group leaf cells by (folder, image_stem) and prepare shared per-image assets
            once per group. Each image is processed in two passes:
              1) Prepare per-cell topology and ROI identity (lightweight).
              2) Compute per-cell metrics, cache PKLs, and export visualizations.
            If puncta are enabled, an image-level puncta assignment step runs between
            prepare and compute.

        Args:
            data_structure (Dict[str, Any]): Nested structure from `create_data_structure`.
            n_cells (int): Total number of cells (for progress reporting).
            progress_cb (Optional[Callable[[int, int, float], None]]): Optional callback
                receiving (processed, total, sec_per_cell).

        Returns:
            None
        """
        # Timing and progress bookkeeping
        start_time = time.time()
        processed = 0

        with tqdm(total=n_cells, desc="Processing Cells", unit="cell", smoothing=0) as pbar:
            # Pre-group leaf cells so image assets are prepared once per (folder, image_stem)
            image_groups = self._collect_cells_by_image(data_structure)

            for (folder, image_stem), group_cells in image_groups.items():
                # Prepare shared image assets for this group and decide whether the whole group is skippable
                assets = self._prepare_image_assets(folder, image_stem, group_cells)
                skip_group = assets.get("skip_group", False) # Group fully cached
                skip_image_io = assets.get("skip_image_io", False) # No image I/O
                puncta_enabled = bool(self.cfg.processing.extract_puncta)

                # Sort cells in a deterministic order
                cells_sorted = sorted(group_cells.items(), key=lambda kv: str(kv[1]["file"]))

                # Prepare cells
                for stem, cell in cells_sorted:
                    # Processing boolean
                    cell["_needs_processing"] = True                   

                    # Update progress if skipping
                    if skip_group:
                        cell["_needs_processing"] = False
                        processed += 1
                        pbar.update(1)
                        if progress_cb is not None:
                            elapsed = time.time() - start_time
                            sec_per_cell = elapsed / processed if processed > 0 else 0.0
                            progress_cb(processed, n_cells, sec_per_cell)
                        continue

                    # Derive output names/paths and create the per-group output directory
                    file_path = Path(cell["file"])
                    rel = file_path.parent.relative_to(self.cfg.pathing.directory)

                    # Create the output directory for this cell/group under cfg.pathing.directory/Processed/
                    save_dir = Path(self.cfg.pathing.directory) / "Processed" / rel
                    save_dir.mkdir(parents=True, exist_ok=True)

                    # Use the shared image_stem + soma_id naming contract for PKL + all visualizations
                    base_name = f"{image_stem}_{cell['soma_id']}"
                    out = save_dir / f"{base_name}.pkl"

                    # Per-cell skipping / viz-repair when not overwriting
                    if not self.cfg.processing.overwrite:
                        # Cache PKL/viz status once to avoid duplicated validation work
                        pkl_ok = self._pkl_is_valid(out)
                        viz_ok = self._viz_complete(save_dir, base_name)

                        # Skip cell if cached outputs are complete
                        if (pkl_ok and viz_ok and not puncta_enabled) or (pkl_ok and viz_ok and puncta_enabled and skip_image_io):
                            cell["_needs_processing"] = False
                            processed += 1
                            pbar.update(1)
                            if progress_cb is not None:
                                elapsed = time.time() - start_time
                                sec_per_cell = elapsed / processed if processed > 0 else 0.0
                                progress_cb(processed, n_cells, sec_per_cell)
                            continue

                        # Repair figures from cached PKL when allowed
                        if (pkl_ok and (not viz_ok) and not puncta_enabled) or (pkl_ok and (not viz_ok) and puncta_enabled and skip_image_io):
                            cell["_needs_processing"] = False
                            with open(out, "rb") as fh:
                                cached_cell = pickle.load(fh)
                            self.plotter.export_visualizations(
                                cell_data=cached_cell,
                                out_dir=save_dir,
                                basename=base_name,
                            )
                            processed += 1
                            pbar.update(1)
                            if progress_cb is not None:
                                elapsed = time.time() - start_time
                                sec_per_cell = elapsed / processed if processed > 0 else 0.0
                                progress_cb(processed, n_cells, sec_per_cell)
                            continue
                    
                    # Preflight said "viz repair only"; processing here indicates inconsistent state
                    if skip_image_io:
                        raise RuntimeError("skip_image_io=True but cell requires processing; check preflight logic.")

                    # Attach per-image assets to the cell dict for downstream steps
                    cell["_save_dir"] = save_dir
                    cell["_base_name"] = base_name
                    cell["_out"] = out
                    cell["base_folder"] = assets["base_folder"]
                    cell["image_path"] = assets["image_path"]
                    cell["image_shape"] = assets["image_shape"]
                    cell["image_signals"] = assets["image_signals"]
                    cell["somatic_rois"] = assets["somatic_rois"]
                    cell["nuclear_rois"] = assets["nuclear_rois"]

                    # Prepare cell objects for downstream calculations
                    self.prepare_cell(cell)

                # Assign puncta
                if puncta_enabled and (any(cell["_needs_processing"] for _, cell in cells_sorted) or self.cfg.processing.overwrite):
                    assign_puncta(
                        cfg=self.cfg,
                        cells_sorted=cells_sorted,
                        puncta_rois=assets["puncta_rois"],
                    )

                # Process cells
                for stem, cell in cells_sorted:

                    # Skip cells that do not need processing
                    if not cell["_needs_processing"]:
                        continue

                    # Process cell
                    self.process_cell(cell)

                    # Cache per-cell results via atomic PKL
                    tmp = cell["_out"].with_suffix(cell["_out"].suffix + ".tmp")
                    to_save = {k: v for k, v in cell.items() if not k.startswith("_")}
                    with open(tmp, "wb") as fh:
                        pickle.dump(to_save, fh)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, cell["_out"])

                    # Export plots/diagnostics for this cell
                    self.plotter.export_visualizations(
                        cell_data=cell,
                        out_dir=cell["_save_dir"],
                        basename=cell["_base_name"],
                    )

                    # Advance progress and optionally notify the UI with throughput
                    processed += 1
                    pbar.update(1)
                    if progress_cb is not None:
                        elapsed = time.time() - start_time
                        sec_per_cell = elapsed / processed if processed > 0 else 0.0
                        progress_cb(processed, n_cells, sec_per_cell)


    def prepare_cell(self, cell: Dict[str, Any]) -> None:
        """
        Mutate a single cell dict in-place with topology and segment geometry.

        This is the per-cell prepare phase of the pipeline. It computes the soma center,
        splits neurites, resolves the soma ROI, and prepares neurite segment geometry
        for downstream puncta assignment and later metric computation.

        Args:
            cell (Dict[str, Any]): Cell dictionary with at least the keys:
                - 'file': Path to the SWC file.
                - 'soma_id': Node ID used as soma/root.
                - plus fields injected by `run()`:
                'image_shape', 'image_signals', 'somatic_rois', 'nuclear_rois'.

        Returns:
            None
        """
        # Load SWC into a dataframe for downstream operations
        swc_path = Path(cell["file"])
        df = read_swc_file(swc_path, self.cfg.pathing.signal_channels, self.cfg.pathing.puncta_roi_suffix)

        # Compute soma center in microns and convert to pixels using voxel size
        cell["soma_node_id"], cell["soma_center"] = soma_center(df, self.cfg.parameters.voxel_size)

        # Split the full trace into neurite subtrees rooted at the soma
        neurite_dfs = dict(enumerate(split_by_neurites(
            df,
            root=cell["soma_node_id"],
            soma_center_um=cell["soma_center"]["um"],
            max_root_offset_um=self.cfg.parameters.max_root_offset_um,
            split_branchpoints_within=self.cfg.parameters.enforce_primaries_until,
        ).values()))

        # Resolve the soma ROI for this cell based on soma position and loaded ROI set
        somatic_rois = cell["somatic_rois"]
        roi = Soma.find_containing_soma(cell["soma_center"]["px"], somatic_rois, cell["file"])
        cell["soma_roi"] = roi

        # Initialize the neurite class with image geometry and analysis thresholds
        neurite = Neurite(
            cfg=self.cfg,
            soma_id=cell["soma_id"],
            image_shape=cell["image_shape"],
            voxel_size_um=self.cfg.parameters.voxel_size,
            filename=str(swc_path),
            min_bp_distance=self.cfg.parameters.min_bp_distance,
            min_branch_length=self.cfg.parameters.min_branch_length,
            min_segment_length_um=self.cfg.parameters.min_segment_length,
            smooth_radii=self.cfg.parameters.smooth_radii,
            radii_smoothing_window_length=self.cfg.parameters.radii_smoothing_window_length,
            radii_smoothing_interval=self.cfg.parameters.radii_smoothing_interval,
            radii_smoothing_min=self.cfg.parameters.radii_smoothing_min,
            soma_center_um=cell["soma_center"]["um"]
        )

        # Prepare each neurite independently and cache prepared objects
        prepared_neurites: Dict[int, Dict[str, Any]] = {}
        for nid, ndf in neurite_dfs.items():
            df_prepared, neurite_segments = neurite.prepare(ndf)
            prepared_neurites[nid] = {
                "df": df_prepared,
                "segments": neurite_segments,
            }

        # Cache prepared objects for downstream calculations (not saved in PKL)
        cell["_neurite_processor"] = neurite
        cell["_prepared_neurites"] = prepared_neurites

        return


    def process_cell(self, cell: Dict[str, Any]) -> None:
        """
        Mutate a single cell dict in-place with morphology, Sholl and signal data.

        This is the per-cell compute phase of the pipeline. It computes the soma metrics,
        processes neurites (metrics + Sholl), computes dendritic tree area, and integrates
        all results back into the cell dict.

        Args:
            cell (Dict[str, Any]): Cell dictionary with at least the keys:
                - 'file': Path to the SWC file.
                - 'soma_id': Node ID used as soma/root.
                - 'somatic_metrics': Dict to be updated with soma/ROI metrics.
                - plus fields injected by `run()`:
                'image_shape', 'image_signals', 'somatic_rois',
                'nuclear_rois'.

        Returns:
            None
        """
        # Extract soma and ROI-derived metrics from images and store into the cell dict
        roi = cell["soma_roi"]
        cell["somatic_metrics"] = Soma.get_soma_metrics(
            soma_roi=roi[1],
            nuclear_rois=cell["nuclear_rois"],
            image_shape=cell["image_shape"],
            image_signals=cell["image_signals"],
            voxel_um=self.cfg.parameters.voxel_size,
            deduct_nuclei=self.cfg.processing.deduct_nuclei
        )

        # Process each neurite independently and append per-neurite outputs into containers
        for nid in sorted(cell["_prepared_neurites"].keys()):
            neurite = cell["_prepared_neurites"][nid]
            df_prepared = neurite["df"]
            neurite_segments = neurite["segments"]
            df, neurite_metrics, neurite_segments, sholl_bundle = cell["_neurite_processor"].process(
                nid,
                df_prepared,
                neurite_segments,
                image_signals=cell["image_signals"],
            )

            # Per-neurite appends
            cell["geometric_dataframes"].append(df)
            cell["geometric_analysis_neurite"].append(neurite_metrics)
            cell["neurite_segments"].append(neurite_segments)
            cell["sholl_analysis_neurite"].append(sholl_bundle)

        # Sum per-neurite Sholl bundles into cell-level Sholl bundle
        cell["sholl_analysis_cell"].append({
            key: {
                radius: sum(neurite[key][radius] for neurite in cell["sholl_analysis_neurite"])
                for radius in cell["sholl_analysis_neurite"][0][key]
            }
            for key in ["sholl_intersections", "sholl_segment_lengths", "sholl_branch_points", "sholl_terminal_points"]
        })

        # Sum per-neurite geometric scalars into cell-level metrics
        cell["geometric_analysis_cell"].append({
            key: sum(neur[key] for neur in cell["geometric_analysis_neurite"])
            for key in ["e_length", "length", "surface_area", "volume"]
        })

        # Find the surface area of the entire dendritic tree
        cell["dendritic_tree_area"] = ConvexHull(np.column_stack([
            pd.concat([df[key] for df in cell["geometric_dataframes"]], ignore_index=True)
            for key in ["X", "Y"]
        ])).volume  # In 2D, ConvexHull.volume is the polygon area

        # Integrate computed metrics and signal features back into unified per-neurite dataframes
        extend_dataframe(
            cell,
            signal_channels=list(self.cfg.pathing.signal_channels),
            voxel_size_um=self.cfg.parameters.voxel_size,
            puncta_enabled=bool(self.cfg.pathing.puncta_roi_suffix),
        )

        return
    

    def _collect_cells_by_image(self, data_structure: Dict[str, Any]) -> dict[tuple[Path, str], dict]:
        """
        Group leaf cell entries by image.

        Use:
            Depth-first traverse the `data_structure` (from `create_data_structure`) and
            collect leaf "cell" dicts (those containing `"file"` and `"soma_id"`).
            Cells are grouped by:
                (swc_parent_folder, image_stem)
            where `image_stem` is the SWC stem with a trailing "-<digits>" removed
            (e.g. "Snap-5848-000" -> "Snap-5848").

        Args:
            data_structure (Dict[str, Any]): Nested dict containing intermediate nodes
                and leaf cell dicts.

        Returns:
            dict[tuple[Path, str], dict]: Mapping (folder, image_stem) -> {cell_key: cell_dict}.
        """
        # (folder, image_stem) -> {cell_key: cell_dict}
        groups: dict[tuple[Path, str], dict] = defaultdict(dict)

        # DFS over nested dict; leaf cells have "file" + "soma_id"
        def visit(node: Dict[str, Any]) -> None:
            for key, value in node.items():
                if isinstance(value, dict) and "file" in value and "soma_id" in value:
                    file_path = Path(value["file"])
                    folder = file_path.parent
                    image_stem = value['path_structure'][-1]
                    groups[(folder, image_stem)][key] = value
                elif isinstance(value, dict):
                    visit(value)

        visit(data_structure)
        return groups


    def _expected_viz_paths(self, save_dir: Path, base_name: str) -> list[Path]:
        """
        Compute all visualization output paths expected for a single processed cell.

        Use:
            Centralizes the filename contract used by `Plotter.export_visualizations()`
            and the `Visualization.save_*()` methods. Given `base_name` (which must
            already include the soma_id, e.g. "{image_stem}_{soma_id}") this function
            computes the set of visualization artifacts that would be produced under
            the current config when visualization is enabled. Per-mode file extensions
            come from `cfg.visualization.<mode>.display.image_format`.

        Args:
            save_dir (Path): Directory where visualization files should be written.
            base_name (str): Basename prefix for this cell (includes soma_id).

        Returns:
            list[Path]: Absolute paths for every expected visualization artifact.
        """
        # If figures are globally disabled, nothing is expected.
        if not self.cfg.processing.visualize:
            return []
        
        # Resolve output extensions from config
        ext_recon = self.cfg.visualization.reconstruction.display.image_format
        ext_geom = self.cfg.visualization.geometry.display.image_format
        ext_sig = self.cfg.visualization.signal.display.image_format
        ext_sholl = self.cfg.visualization.sholl.display.image_format
        ext_puncta = self.cfg.visualization.puncta.display.image_format

        # Snapshot the configured signal channels so we can require one plot per channel (when enabled)
        chans = list(self.cfg.pathing.signal_channels)

        # Build the expected plots list to match Plotter.export_visualizations() enable/overwrite behavior
        paths: list[Path] = []

        # Reconstruction
        if self.cfg.visualization.reconstruction.enable:
            paths.append(save_dir / f"{base_name}_reconstruction.{ext_recon}")

        # Geometry
        if self.cfg.visualization.geometry.enable:
            paths.append(save_dir / f"{base_name}_geometry.{ext_geom}")

        # Sholl
        if self.cfg.visualization.sholl.enable:
            paths.append(save_dir / f"{base_name}_sholl.{ext_sholl}")

        # Signal(s)
        if self.cfg.processing.extract_signal and self.cfg.visualization.signal.enable:
            paths.extend(save_dir / f"{base_name}_signal_ch{ch}.{ext_sig}" for ch in chans)

        # Puncta
        if self.cfg.processing.extract_puncta and self.cfg.visualization.puncta.enable:
            paths.append(save_dir / f"{base_name}_puncta.{ext_puncta}")

        return paths


    def _pkl_is_valid(self, p: Path) -> bool:
        """
        Fast-check whether a per-cell PKL cache exists and is non-empty.

        Use:
            This is intentionally a cheap existence check used on the skip path to
            avoid paying unpickle cost for every cell. The pipeline only unpickles
            when it actually needs the cached contents (e.g. viz repair).

        Args:
            p (Path): Path to the per-cell PKL cache file.

        Returns:
            bool: True if the PKL exists and is non-empty; False otherwise.
        """
        # Fast-fail if the cache file does not exist
        if not p.exists():
            return False

        # Treat zero-length files as invalid (common symptom of interrupted writes)
        try:
            return p.stat().st_size > 0
        except OSError:
            return False


    def _viz_complete(self, save_dir: Path, base_name: str) -> bool:
        """
        Check whether all visualization artifacts for a cell have been written.

        Use:
            When visualization is disabled (`cfg.processing.visualize=False`), this returns
            True to avoid gating caching on figures that will never be produced.

            When visualization is enabled, this requires every expected visualization file
            (as determined by `_expected_viz_paths`, which mirrors the Plotter's logic for
            `cfg.visualization.<section>.enable` and `cfg.processing.overwrite`) to exist
            and be non-empty. This guards against interrupted plot writes and supports
            "repair" runs (valid PKL but missing figures).

        Args:
            save_dir (Path): Directory where visualization files should exist.
            base_name (str): Basename prefix for this cell (includes soma_id).

        Returns:
            bool: True if visualization is disabled, or all expected files exist and are
            non-empty; False otherwise.
        """
        # If not producing figures, treat visualization as complete by definition
        if not self.cfg.processing.visualize:
            return True

        # Compute expected outputs under current config (section enable flags + overwrite behavior)
        expected = self._expected_viz_paths(save_dir, base_name)

        # If nothing is expected (e.g., visualize=True but all sections disabled), treat as complete
        if not expected:
            return True

        # Require every expected figure path to exist and be non-empty
        for p in expected:
            if (not p.exists()) or p.stat().st_size == 0:
                return False

        return True


    def _prepare_image_assets(self, folder: Path, image_stem: str, group_cells: dict) -> dict:
        """
        Preflight an image group: decide skip vs. process, then load shared assets.

        Use:
            For a given (folder, image_stem) group, check whether all expected PKL
            outputs already exist (unless overwrite is enabled). If any cell needs
            work, load the image bundle and ROI assets once and return them for
            reuse across all cells in the group.

        Args:
            folder (Path): Folder containing the SWC/trace files for this image.
            image_stem (str): Image identifier shared by the group's SWC stems.
            group_cells (dict): Mapping {cell_key: cell_dict} for this image group.

        Returns:
            dict: Asset bundle with:
                - skip_group (bool)
                - base_folder (Path)
                - image_path (Path)
                - image_shape (np.ndarray)
                - image_signals (Dict[int, np.ndarray])
                - somatic_rois (Any)
                - nuclear_rois (Any | None)
                - puncta_rois (Any | None)
        """
        cfg = self.cfg

        # Check whether this entire image group can be skipped based on cached outputs
        need_load = cfg.processing.overwrite
        need_viz_repair = False

        # Skip processing if cached PKLs and figures are complete.
        if not need_load:
            for cell in group_cells.values():
                file_path = Path(cell["file"])
                rel = file_path.parent.relative_to(cfg.pathing.directory)
                save_dir = Path(cfg.pathing.directory) / "Processed" / rel
                base_name = f"{image_stem}_{cell['soma_id']}"
                out_pkl = save_dir / f"{base_name}.pkl"

                # Any missing/invalid PKL means we must fully process (requires image/ROI I/O)
                if not self._pkl_is_valid(out_pkl):
                    need_load = True
                    break

                # Track whether we only need to repair missing/incomplete figures
                if not self._viz_complete(save_dir, base_name):
                    need_viz_repair = True

        # If everything is already processed, skip image/ROI I/O entirely
        if (not need_load) and (not need_viz_repair):
            return {"skip_group": True, "skip_image_io": True}

        # If PKLs are valid but some figures are missing, avoid loading images/ROIs
        if (not need_load) and need_viz_repair:
            return {"skip_group": False, "skip_image_io": True}

        # Parse configured channels for image loading
        sig = list(cfg.pathing.signal_channels)

        # Resolve image path for this group
        image_path = folder / f"{image_stem}{cfg.pathing.image_suffix}"

        # Load annotated + signal images once per image group
        image_shape, image_signals = load_image_bundle(
            image_path=image_path,
            signal_channels=sig,
        )

        # Load ROIs associated with this image stem
        somatic_rois = find_rois_for_folder(folder, image_stem, cfg.pathing.soma_roi_suffix)

        # Optionally load nuclear ROIs when nucleus deduction is enabled
        nuclear_rois = None
        if cfg.processing.deduct_nuclei:
            nuclear_rois = find_rois_for_folder(folder, image_stem, cfg.pathing.nuclear_roi_suffix)

        # Optionally load puncta ROIs when enabled
        puncta_rois = None
        if cfg.pathing.puncta_roi_suffix:
            puncta_rois = find_rois_for_folder(folder, image_stem, cfg.pathing.puncta_roi_suffix)

        return {
            "skip_group": False,
            "base_folder": folder,
            "image_path": image_path,
            "image_shape": image_shape,
            "image_signals": image_signals,
            "somatic_rois": somatic_rois,
            "nuclear_rois": nuclear_rois,
            "puncta_rois": puncta_rois,
        }
   

class Plotter:
    """Thin helper around Visualization to export per-cell figures."""

    def __init__(self, cfg: Config):
        """
        Initialize the Plotter.

        Args:
            cfg (Config): Global configuration controlling visualization
                flags and output formats.
        """
        # Store config and build the underlying visualization backend
        self.cfg = cfg
        self.viz = Visualization(cfg)

    def export_visualizations(
        self,
        cell_data: Dict[str, Any],
        out_dir: Path,
        basename: str,
    ) -> None:
        """
        Export all enabled visualizations for a single cell.

        Use:
            Depending on the visualization configuration, this may save
            reconstruction, geometry, signal, and Sholl figures using the
            provided `basename` prefix.

        Args:
            cell_data (Dict[str, Any]): Processed cell dictionary containing
               the metrics and geometries needed for plotting.
            out_dir (Path): Output directory where figure files are written.
            basename (str): Basename prefix for saved files (without suffix).

        Returns:
            None
        """
        # Only visualize if asked
        if not self.cfg.processing.visualize:
            return

        # Reconstruction
        if self.cfg.visualization.reconstruction.enable:
            self.viz.save_reconstruction(cell_data, out_dir / basename)

        # Geometry
        if self.cfg.visualization.geometry.enable:
            self.viz.save_geometry(cell_data, out_dir / basename)

        # Signals
        if self.cfg.processing.extract_signal and self.cfg.visualization.signal.enable:
            self.viz.save_signal(cell_data, out_dir / basename)

        # Puncta
        if self.cfg.processing.extract_puncta and self.cfg.visualization.puncta.enable:
            self.viz.save_puncta(cell_data, out_dir / basename)

        # Sholl
        if self.cfg.visualization.sholl.enable:
            self.viz.save_sholl(cell_data, out_dir / basename)


class MorphologyPipeline:
    """High-level orchestrator. Minimal logic here; compose replaceable parts."""

    def __init__(
        self,
        cfg: Config,
        repo: TraceDiscovery | None = None,
        processor: Processor | None = None,
        aggregator: Aggregator | None = None,
    ):
        """
        Initialize the pipeline with configuration and pluggable components.

        Args:
            cfg (Config): Global configuration used across the pipeline.
            repo (TraceDiscovery | None): Optional custom repository for
                SWC/trace discovery; defaults to `TraceDiscovery(cfg)`.
            processor (Processor | None): Optional custom processor; defaults to
                `Processor(cfg)`.
            aggregator (Aggregator | None): Optional post-processing aggregator;
                defaults to `Aggregator(cfg)`.
        """
        self.cfg = cfg
        self.repo = repo or TraceDiscovery(cfg)
        self.processor = processor or Processor(cfg)
        self.aggregator = aggregator or Aggregator(cfg)

    def run(self, progress_cb: Optional[Callable[[int, int, float], None]] = None,) -> None:
        """
        Execute full morphology pipeline and optional post-processing aggregation.

        This method:
          1) Discovers SWC/trace files via the repository.
          2) Builds a nested data structure.
          3) Delegates cell-wise processing to the Processor.
          4) Optionally aggregates processed outputs into summary CSV tables.

        Returns:
            None

        Raises:
            DataNotFound: If no input SWC/trace files (or, when aggregation is enabled,
                no processed PKLs) can be discovered under the configured directory.
        """
        groups = self.repo.discover_swc()
        data_structure, n_cells = create_data_structure(groups)
        self.processor.run(data_structure, n_cells, progress_cb=progress_cb)
        if self.cfg.processing.aggregate:
            pkl_groups = self.repo.discover_pkl()
            self.aggregator.run(pkl_groups)