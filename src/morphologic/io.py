# src/morphologic/io.py
from __future__ import annotations

# General imports (stdlib)
import os
import posixpath
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Third-party imports
import numpy as np
import pandas as pd
import tifffile as tiff
from roifile import ImagejRoi

# Local imports
from .exceptions import DataNotFound, SWCParseError

# Group structure:(image_name, image_path, path_structure, tracing_files)
Group = Tuple[str, Path, List[str], List[Path]]
"""
Group:
    A single folder-level group of trace files discovered by discover_traces.

    Layout:
        image_name      (str)        : Folder / image name used as top-level key.
        image_path      (str)       : Path to the folder containing the trace files.
        path_structure  (List[str])  : Relative path components from the root to this folder.
        tracing_files   (List[Path]) : List of SWC/trace file paths in this folder.
"""


def discover_traces(
    directory: Path,
    base_suffix: str,
    criteria_extension: str = ".swc",
) -> List[Group]:
    """
    Discover neurons reconstruction files and group them per image based on filename roots.

    Use:
        Recursively walk `directory`. In each folder, collect base files
        matching `base_suffix` and criteria files matching
        `criteria_extension`. For each base file, derive a root name and attach
        all criteria files in the same folder whose stem matches that root.

    Args:
        directory (Path): Root directory to search.
        base_suffix (str): Base suffix to identify image files
            (e.g. ".traces", or "_8bit.tif").
        criteria_extension (str): File extension used to match reconstruction files,
            default ".swc".

    Returns:
        List[Group]: A list of groups of the form:
            (image_root, image_path, path_structure_rel_to_root, [trace_paths...])

        Where:
          - image_root: the computed root (string) used for grouping,
          - image_path: the first-seen base file path for that root,
          - path_structure_rel_to_root: folder parts relative to `directory`,
          - trace_paths: all matching criteria files (Paths) in that same folder.

    Raises:
        DataNotFound: If the root directory does not exist or if no matching
            base/criteria groupings can be discovered under it.
    """
    # Validate the root directory exists on disk
    root = Path(directory)
    if not root.exists():
        raise DataNotFound(f"Directory not found: {root}")
    
    # Require a base extension so criteria files can be grouped per image root
    if not base_suffix:
        raise DataNotFound(
            f"No base_suffix provided; cannot group '{criteria_extension}' files per image under: '{str(root)}'. "
            f"Set base_suffix to a filename suffix such as '.traces' or '_8bit.tif'."
        )
    
    # Normalize criteria extension
    if criteria_extension and not criteria_extension.startswith("."):
        criteria_extension = "." + criteria_extension

    # Accumulate one output tuple per discovered image root
    groups: List[Group] = []

    # Walk the directory tree and collect base/criteria files per folder
    for dirpath, _dirnames, filenames in os.walk(str(root)):
        base_files = []
        criteria_files = []

        # Scan this folder once and collect base and criteria files with their stems/paths
        for item in filenames:
            path = posixpath.join(dirpath, item)

            if item.endswith(base_suffix):
                rel = os.path.relpath(path, str(root))
                struct = rel.split(os.sep)[:-1]
                bf_stem = item[: -len(base_suffix)].rstrip(".")
                base_files.append((bf_stem, path, struct))

            if item.endswith(criteria_extension):
                criteria_files.append(path)

        # Group criteria files by normalized root derived from each base file
        seen_stems = set()
        for bf_stem, base_path, struct in base_files:
            # Derive bf_stem differently when base and criteria suffix are the same
            if base_suffix == criteria_extension:
                bf_stem = bf_stem.rsplit("_", 1)[0]

            # Skip stems already seen
            if bf_stem in seen_stems:
                continue

            # Match all criteria files in this folder whose stem matches
            matches = []
            for cf in criteria_files:
                cf_stem = os.path.splitext(os.path.basename(cf))[0]
                if cf_stem == bf_stem or cf_stem.startswith(bf_stem + "-") or cf_stem.startswith(bf_stem + "_"):
                    matches.append(cf)

            # Emit one group entry per root name with all matching criteria paths
            if matches:
                groups.append(
                    (
                        bf_stem,
                        str(Path(base_path)),
                        list(struct),
                        [Path(m) for m in matches],
                    )
                )
                seen_stems.add(bf_stem)

    # Fail if no groupings were discovered anywhere under the root
    if not groups:
        raise DataNotFound(f"No '*{criteria_extension}' files found under: '{str(root)}' using '*{base_suffix}' as base suffix")

    # Return all discovered image-root groupings
    return groups


def read_swc_file(
    filepath: Path,
    signal_channels: tuple[int, ...] = (),
    puncta_suffix: str = "",
) -> pd.DataFrame:
    """
    Read an SWC file into a DataFrame and initialize pipeline-required columns.

    Use:
        Load the core SWC fields into a pandas DataFrame (keeping only the
        columns used downstream), then pre-create analysis/output columns that
        later pipeline steps will fill (distances, segment metrics, and optional
        signal intensities).

        This reader also validates that core SWC columns contain numeric data
        of the expected kind (e.g., IDs/Types/Parents as integers, coordinates
        and radii as floats).

    Args:
        filepath (Path): SWC file to read.
        signal_channels (tuple[int, ...]): Optional 1-based channel indices for
            which to create 'signal_intensity_<ch>' columns.
        puncta_suffix (str): Optional puncta_suffix. If non-empty, pre-create a 
            'segment_puncta_count' integer column.

    Returns:
        pd.DataFrame: DataFrame containing at minimum:
            ID, Type, X, Y, Radius, Parent, plus initialized metric columns.

    Raises:
        SWCParseError: If the SWC cannot be read or parsed, or if it fails
            validation (non-numeric fields, invalid/dangling Parent references,
            or non-root rows where ID <= Parent).
    """
    try:
        # Standard SWC columns; we keep the subset used in the pipeline
        column_names = ["ID", "Type", "X", "Y", "Z", "Radius", "Parent"]
        usecols = ["ID", "Type", "X", "Y", "Radius", "Parent"]

        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            comment="#",
            names=column_names,
            usecols=usecols,
            engine="python",
        )

        # Validate numeric types and content for required SWC columns
        missing = [c for c in usecols if c not in df.columns]
        if missing:
            raise SWCParseError(f"Missing required SWC columns: {missing}")

        # Coerce and check integer-like columns
        int_like_cols = ["ID", "Type", "Parent"]
        for col in int_like_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            bad_mask = s.isna()
            if bad_mask.any():
                bad_rows = df.index[bad_mask].to_list()[:10]
                raise SWCParseError(
                    f"Column '{col}' contains non-numeric values. "
                    f"First bad row indices: {bad_rows}"
                )

            # Ensure values are integer-like
            frac = (s % 1).abs()
            non_int_mask = frac.gt(0) & ~s.isna()
            if non_int_mask.any():
                bad_rows = df.index[non_int_mask].to_list()[:10]
                bad_vals = s[non_int_mask].iloc[:10].to_list()
                raise SWCParseError(
                    f"Column '{col}' contains non-integer values. "
                    f"First bad rows/values: {list(zip(bad_rows, bad_vals))}"
                )

            df[col] = s.astype("int64")

        # Coerce and check float columns
        float_cols = ["X", "Y", "Radius"]
        for col in float_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            bad_mask = s.isna()
            if bad_mask.any():
                bad_rows = df.index[bad_mask].to_list()[:10]
                raise SWCParseError(
                    f"Column '{col}' contains non-numeric values. "
                    f"First bad row indices: {bad_rows}"
                )
            df[col] = s.astype("float64")

        # Additional sanity checks commonly expected for SWC
        if (df["ID"] <= 0).any():
            bad_rows = df.index[df["ID"] <= 0].to_list()[:10]
            raise SWCParseError(f"SWC 'ID' must be positive. First bad row indices: {bad_rows}")

        # Parent is typically -1 for soma/root, otherwise references an existing ID
        if (df["Parent"] == 0).any():
            bad_rows = df.index[df["Parent"] == 0].to_list()[:10]
            raise SWCParseError(
                f"SWC 'Parent' should be -1 for root or a positive ID (0 is unusual/invalid). "
                f"First bad row indices: {bad_rows}"
            )

        # Verify there are no dangling parents
        ids = set(df["ID"].to_list())
        parent_vals = set(df.loc[df["Parent"] > 0, "Parent"].to_list())
        dangling = sorted(parent_vals - ids)
        if dangling:
            raise SWCParseError(
                f"SWC contains dangling Parent references (not present in ID). "
                f"First missing parent IDs: {dangling[:10]}"
            )
        
        # Verify topological ID ordering (child IDs must be greater than their Parent IDs)
        non_root_mask = df["Parent"] != -1
        bad_mask = non_root_mask & (df["ID"] <= df["Parent"])
        if bad_mask.any():
            bad_rows = df.index[bad_mask].to_list()[:10]
            bad_pairs = list(zip(df.loc[bad_mask, "ID"].iloc[:10].to_list(),
                                 df.loc[bad_mask, "Parent"].iloc[:10].to_list()))
            raise SWCParseError(
                "SWC must satisfy ID > Parent for all non-root nodes (Parent != -1). "
                f"First bad row indices and (ID, Parent) pairs: {list(zip(bad_rows, bad_pairs))}"
            )

        # Radius must be non-negative, and non-root nodes (Parent != -1) should have Radius > 0
        if (df["Radius"] < 0).any():
            bad_rows = df.index[df["Radius"] < 0].to_list()[:10]
            raise SWCParseError(f"SWC 'Radius' must be non-negative. First bad row indices: {bad_rows}")

        non_root_mask = df["Parent"] != -1
        if (df.loc[non_root_mask, "Radius"] <= 0).any():
            bad_rows = df.index[non_root_mask & (df["Radius"] <= 0)].to_list()[:10]
            raise SWCParseError(
                f"SWC 'Radius' must be > 0 for non-root nodes (Parent != -1). "
                f"First bad row indices: {bad_rows}"
            )

        # Initialize segment-varying numeric columns
        df["e_dist_from_soma"] = np.nan
        df["dist_from_soma_um"] = np.nan
        df["segment_e_length"] = np.nan
        df["segment_length_um"] = np.nan
        df["segment_surface_um"] = np.nan
        df["segment_volume_um"] = np.nan
        df["segment_mask_area_um"] = np.nan

        # Optionally pre-create intensity columns for requested channels
        for ch in signal_channels or ():
            col = f"signal_intensity_{ch}"
            if col not in df.columns:
                df[col] = pd.Series(np.nan, index=df.index, dtype="float64")

        # Optionally pre-create a puncta count column when puncta suffixing is enabled
        if puncta_suffix:
            if "segment_puncta_count" not in df.columns:
                df["segment_puncta_count"] = pd.Series(0, index=df.index, dtype="int64")

        return df

    except Exception as e:
        raise SWCParseError(f"Failed to read SWC '{filepath}': {e}") from e


def find_rois_for_folder(folder: Path, image_stem: str, suffix: str) -> Dict[str, Any]:
    """
    Locate and load ROIs associated with a given image stem.

    Use:
        In `folder`, find the first ROI source whose name starts with
        f"{image_stem}{suffix}" and return its contents as a dict.

        Supported formats:
          - ImageJ ROI containers (.roi, .zip): returned as {roi_name: ImagejRoi}
          - CSV point tables (.csv): returned as {generated_key: ImagejRoi}

        Unnamed ROIs get generated keys like "<roi_file_stem>-0000", "<roi_file_stem>-0001", ...

    Args:
        folder (Path): Directory in which to search for the ROI file
        image_stem (str): Base stem used to match ROI filenames
        suffix (str): Suffix following the stem, including extension
            (e.g. "_somas.zip", "_nuclei.roi", "_puncta.csv")

    Returns:
        Dict[str, Any]: Mapping from ROI name (or generated key) to ROI objects

    Raises:
        FileNotFoundError: If no matching ROI file is found in the folder
        OSError: If the ROI file cannot be read from disk
        ValueError: If a CSV file is found but does not contain X and Y columns
        roifile.RoiError: If the file is not a valid ImageJ ROI container
    """
    # Find first ROI file starting with stem and suffix
    for f in folder.iterdir():
        if f.is_file() and f.name.startswith(f"{image_stem}{suffix}"):

            # Route by file extension
            ext = f.suffix.lower()

            # CSV puncta: build point ROIs from X/Y columns
            if ext == ".csv":
                # Load puncta table (one punctum per row)
                df = pd.read_csv(f)

                # Resolve x/y columns case-insensitively
                cols = {c.lower(): c for c in df.columns}
                xcol, ycol = cols.get("x"), cols.get("y")

                # Require x/y columns to be present
                if xcol is None or ycol is None:
                    raise ValueError(f"CSV puncta file must contain X and Y columns: {f}")

                # Extract coordinates as float array (pixel space)
                xy = df[[xcol, ycol]].to_numpy(dtype=float, copy=False)

                # Use file stem for generated ROI names
                stem = f.stem

                # Convert each point into a 1-vertex ImageJ ROI
                return {
                    (key := f"{stem}-{i:04d}"): ImagejRoi.frompoints(
                        np.array([[x, y]], dtype=np.float32),
                        name=key,
                    )
                    for i, (x, y) in enumerate(xy)
                }

            # ImageJ ROI container (.roi or .zip)
            rois = ImagejRoi.fromfile(str(f))

            # Normalize singleton ROI into a list
            rois = rois if isinstance(rois, list) else [rois]

            # Use file stem for generated ROI names
            stem = f.stem

            # Return ROIs keyed by ROI name (or generated fallback)
            return {
                (roi.name or f"{stem}-{i:04d}"): roi
                for i, roi in enumerate(rois)
            }

    # Fail if no matching ROI source exists in this folder
    raise FileNotFoundError(
        f"No ROI file with suffix {suffix} for base {image_stem} in {folder}"
    )


def load_image_bundle(
    image_path: Path,
    signal_channels: list[int],
) -> Tuple[Tuple[int, int], Dict[int, np.ndarray]]:
    """
    Load a TIFF image and split it into an annotation channel and signal channels.

    Use:
        Choose the annotation channel (or the whole image if it is 2D), then
        load each requested signal channel by indexing the channel axis.
        Assumes TIFF arrays are organized as (C, Y, X) when `img.ndim >= 3`.

    Args:
        image_path (Path): Path to the TIFF image on disk
        signal_channels (list[int]): Channel indices from config["signal_channels"]
            (1-based)

    Returns:
        Tuple[Tuple[int, int], Dict[int, np.ndarray]]:
            - image_shape: 2D image shape as (Y, X) in pixels
            - image_signals: Dict[int, np.ndarray] keyed by 1-based channel index
    """
    # Load image 
    img = tiff.imread(str(image_path))

    # Derive 2D image shape (Y, X)
    image_shape = img.shape[-2:]

    # Load 0-based signal channels (if any)
    image_signals: Dict[int, np.ndarray] = {}
    for ch in signal_channels:
        ch_idx = int(ch) - 1
        if img.ndim < 3:
            image_signals[ch] = img
        else:
            image_signals[ch] = img[ch_idx, :, :]

    return image_shape, image_signals