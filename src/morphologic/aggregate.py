"""
Post-processing aggregation pipeline.

Use:
    Stream pickled per-cell outputs, concatenate per-cell DataFrames, compute
    summary statistics for geometric and Sholl metrics, and export result tables
    to CSV with timing logs for each stage.
"""


# src/morphologic/aggregate.py
from __future__ import annotations

# General imports (stdlib)
import os
import pickle
import time
from contextlib import contextmanager
from typing import Any, Iterable, Callable

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from .config import Config
from .structure import Group


class Aggregator:
    """
    Post-processing aggregation for processed morphology outputs.

    Pipeline:
      1) Stream processed per-cell PKLs and load per-cell dictionaries
      2) Concatenate per-cell geometric and Sholl DataFrames
      3) Aggregate Sholl and geometric metrics
      4) Export CSV tables
    """

    def __init__(
        self,
        cfg: Config,
        *,
        processed_subdir: str = "Processed",
        output_subdir: str = "Aggregated",
    ) -> None:
        """
        Initialize an Aggregator with configuration and output conventions.

        Use:
            Construct once per run and reuse for aggregation of the processed
            dataset under `cfg.pathing.directory / processed_subdir`.

        Args:
            cfg (Config): Global configuration used for aggregation parameters.
            processed_subdir (str): Subfolder under `cfg.pathing.directory` containing
                processed per-cell PKLs.
            output_subdir (str): Subfolder under `cfg.pathing.directory` where CSV
                exports are written.
        """
        self.cfg = cfg
        self.processed_dir = self.cfg.pathing.directory / processed_subdir
        self.output_dir = self.cfg.pathing.directory / output_subdir
        self.output_path_template = str(self.output_dir / "{}.csv")


    def run(self, groups: list[Group]) -> None:
        """
        Run post-processing aggregation from processed PKLs to CSV exports.

        Use:
            Stream, with timing logs, per-cell dictionaries from processed PKLs,
            concatenate per-cell tables, compute aggregation summaries for Sholl 
            and geometric metrics, and export result tables to CSV.

        Args:
            groups (list[Group]): Discovered PKL groups (e.g., from
                `TraceDiscovery.discover_pkl()`), where each group contains a list of
                PKL paths under `tracing_files`.

        Returns:
            None
        """
        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)

        # Build an iterator over cell PKLs
        cells = self.gather_cells(groups)

        # Concatenate per-cell tables
        geom_df, sholl_df = self.step(
            "concatenate dataframes",
            self.concatenate_dataframe,
            cells,
        )

        # Aggregations
        sholl_df, sholl_data = self.step(
            "aggregate_sholl_data",
            self.aggregate_sholl_data,
            sholl_df,
            self.cfg,
        )
        geom_df, geom_data = self.step(
            "aggregate_geometric_data",
            self.aggregate_geometric_data,
            geom_df,
            self.cfg,
        )

        # CSV exports
        self.step(
            "save sholl tables",
            self.save_data_dicts,
            sholl_data,
            self.output_path_template,
            prefix_override="sholl",
        )
        self.step(
            "save geom tables",
            self.save_data_dicts,
            geom_data,
            self.output_path_template,
            prefix_override=None,
        )


    @staticmethod 
    def step(label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Run a function and print a timing log.

        Args:
            label (str): Label printed in the log line.
            fn (Callable): Function to run.
            *args: Positional args forwarded to `fn`.
            **kwargs: Keyword args forwarded to `fn`.

        Returns:
            Any: The return value of `fn(*args, **kwargs)`.
        """
        # Start wall-clock timer for this step
        t0 = time.time()

        # Execute the callable and capture its return value
        out = fn(*args, **kwargs)

        # Compute elapsed time in milliseconds
        dt_ms = (time.time() - t0) * 1000.0

        # Emit a compact timing log line for pipeline profiling
        print(f"[ok] {label} ({dt_ms:,.0f} ms)")

        # Return the wrapped function result unchanged
        return out


    @staticmethod
    @contextmanager
    def timed(label: str):
        """
        Context manager that prints a timing log when the block exits.

        Args:
            label (str): Label printed in the log line.

        Yields:
            None
        """
        # Start wall-clock timer for this timed block
        t0 = time.time()
        try:
            # Run the caller's code inside the context
            yield
        finally:
            # Always compute and log elapsed time, even if an exception occurs
            dt_ms = (time.time() - t0) * 1000.0
            print(f"[ok] {label} ({dt_ms:,.0f} ms)")


    @staticmethod
    def gather_cells(groups: list[Group]) -> Iterable[dict[str, Any]]:
        """
        Yield per-cell dictionaries loaded from processed PKL files.

        Use:
            Iterate over discovered PKL groups and lazily unpickle each per-cell cache.
            This provides a streaming source of cell dictionaries for concatenation and
            aggregation without materializing all cells in memory at once.

        Args:
            groups (list[Group]): Discovered PKL groups (e.g., from
                `TraceDiscovery.discover_pkl()`), where each group contains a list of
                PKL paths under `tracing_files`.

        Yields:
            dict[str, Any]: One unpickled per-cell dictionary per PKL file.
        """
        # Walk each discovered group and lazily load per-cell caches
        for _image_name, _image_path, _path_structure, tracing_files in groups:
            for pkl_path in tracing_files:
                # Load a single processed cell dictionary from disk and yield it
                with open(pkl_path, "rb") as f:
                    yield pickle.load(f)


    @staticmethod
    def concatenate_dataframe(cells: Iterable[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Concatenate per-cell geometric and Sholl DataFrames into two unified DataFrames.

        Use:
            Iterate over already-loaded per-cell dictionaries (from processed PKLs) and
            vertically concatenate all per-cell geometric and Sholl dataframes.

        Args:
            cells (Iterable[dict[str, Any]]): Iterable of per-cell dictionaries loaded
                from processed PKLs.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                - Concatenated geometric dataframe (may be empty).
                - Concatenated Sholl dataframe (may be empty).
        """
        # Accumulators for geometric and Sholl dataframes
        aggregated_data: list[pd.DataFrame] = []
        sholl_data: list[pd.DataFrame] = []

        # Traverse all cell dicts and gather per-cell dataframes
        for cell in cells:
            for df in cell.get("geometric_dataframes", []):
                aggregated_data.append(df)
            for df in cell.get("sholl_dataframes", []):
                sholl_data.append(df)

        # Vertically concatenate gathered frames (or return empty DataFrames)
        aggregated_df = pd.concat(aggregated_data, ignore_index=True) if aggregated_data else pd.DataFrame()
        sholl_df = pd.concat(sholl_data, ignore_index=True) if sholl_data else pd.DataFrame()

        # Return unified geometric and Sholl DataFrames
        return aggregated_df, sholl_df


    @staticmethod
    def _group_weighted_means(
        df: pd.DataFrame,
        group_cols: list[str],
        dependents: list[str],
        weight_col: str,
    ) -> pd.DataFrame:
        """
        Compute vectorized weighted means by group.

        Use:
            For each dependent column, compute sum(w*x)/sum(w) within groups and
            return NaN where total weight is zero.

        Args:
            df (pd.DataFrame): Source dataframe.
            group_cols (list[str]): Columns defining groups.
            dependents (list[str]): Numeric columns to average with weights.
            weight_col (str): Column containing non-negative weights.

        Returns:
            pd.DataFrame: Group-wise weighted means with group columns preserved.
        """
        # Minimal subset to cut memory and ensure all columns exist
        cols = list(dict.fromkeys(group_cols + dependents + [weight_col]))
        g = df[cols].copy()

        # Cast once
        w = g[weight_col].astype(float)
        X = g[dependents].astype(float)

        # Groupers must be arrays/Series aligned to X/w's index
        groupers = [g[c] for c in group_cols]

        # Numerator: sum(w * x) per group
        num = (X.mul(w, axis=0)).groupby(groupers, observed=False).sum()

        # Denominator: sum(w) per group
        den = w.groupby(groupers, observed=False).sum()

        # Weighted mean with safe division; NaN where sum(w) == 0
        out = num.div(den, axis=0)
        out = out.where(den.ne(0), np.nan)

        return out.reset_index()


    @staticmethod
    def _compute_basic_stats(
        df: pd.DataFrame,
        group_cols: list[str],
        dependents: list[str],
        subject: str | None = None,
        use_length_weight: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute per-element collapsed values and group summary stats.

        Use:
            1) Collapse segment-level rows into per-element values (cell or neurite)
            within each group.
            2) Aggregate the per-element values into group-level mean/SEM/count tables.

        Args:
            df (pd.DataFrame): Source dataframe of segment-level measures.
            group_cols (list[str]): Grouping columns for aggregation.
            dependents (list[str]): Measure columns to aggregate.
            subject (str | None): Element unit: "cell" or "neurite".
            use_length_weight (bool): If True, weight by 'segment_length_um'.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                - elements: Per-element collapsed values.
                - stats: Group-level mean/sem/count per dependent.
        """
        # Subject identifier columns
        if subject == "neurite":
            subj_cols = ["soma_id", "neurite_uid"]
            if "neurite_length" not in dependents:
                subj_cols.append("neurite_length")
            if "Type" in df.columns:
                subj_cols.append("Type")
        elif subject == "branch":
            subj_cols = ["soma_id", "neurite_uid", "neurite_length", "Type", "branch_uid", "branch_order", "branch_dist_from_soma", "branch_e_dist_from_soma"]
        else:
            subj_cols = ["soma_id"]

        # Compute per-element collapsed values either length-weighted or simple mean
        if use_length_weight:
            elements = Aggregator._group_weighted_means(df, group_cols + subj_cols, dependents, 'segment_length_um')
        else:
            elements = (
                df.groupby(group_cols + subj_cols, observed=False)[dependents]
                .mean()
                .reset_index()
            )

        # Descriptive helpers
        def q1(x: pd.Series) -> float:
            return float(x.quantile(0.25))

        def q3(x: pd.Series) -> float:
            return float(x.quantile(0.75))

        # Descriptive labels
        q1.__name__ = "q1"
        q3.__name__ = "q3"

        # Aggregate per-element values to group-level
        stats = (
            elements.groupby(group_cols, observed=False)[dependents]
                    .agg(["mean", "sem", "count", "min", "max", "median", q1, q3])
                    .reset_index()
        )

        # Flatten multi-level column index produced by aggregation
        stats.columns = ['_'.join(filter(None, col)).strip() for col in stats.columns.ravel()]

        # Return per-element table and the aggregated statistics
        return elements, stats


    @staticmethod
    def _add_norm_columns(df: pd.DataFrame, deps: list[str], group_key: str | list[str] | None) -> None:
        """
        Add distribution-aligned variants of signal columns in-place.

        Use:
            For each dependent column containing "signal", create a `{col}_norm`
            column by mapping each group's distribution to the reference group's
            distribution using a robust affine transform:
                - center: median
                - scale: IQR (q75 - q25)

            For group g and reference group r:
                x' = (x - median_g) * (IQR_r / IQR_g) + median_r

            The reference group is taken as the first group in sorted group-key order.

        Args:
            df (pd.DataFrame): Dataframe to mutate.
            deps (list[str]): Dependent column names; extended in-place with new *_norm names.
            group_key (str | list[str] | None): Grouping key(s) for robust distribution alignment.

        Returns:
            None
        """
        # Skip if no grouping key is provided
        if not group_key:
            return

        # Collect new dependent names to append after mutation
        norm_cols: list[str] = []

        # Apply robust median/IQR alignment for every signal-like dependent
        for col in deps:
            if "signal" not in col:
                continue

            # Compute per-group robust center and spread for this column
            stats = df.groupby(group_key, observed=False)[col].agg(
                median=lambda s: float(np.nanmedian(s.to_numpy())),
                q25=lambda s: float(np.nanpercentile(s.to_numpy(), 25)),
                q75=lambda s: float(np.nanpercentile(s.to_numpy(), 75)),
            )
            stats["iqr"] = stats["q75"] - stats["q25"]

            # Select the reference group from stable sorted group-key order
            ref_key = stats.index.sort_values()[0]
            ref_med = float(stats.loc[ref_key, "median"])
            ref_iqr = float(stats.loc[ref_key, "iqr"])

            # Build per-row group medians and IQRs for vectorized alignment
            med = df.groupby(group_key, observed=False)[col].transform("median").astype(float)
            q25 = df.groupby(group_key, observed=False)[col].transform(lambda s: np.nanpercentile(s.to_numpy(), 25)).astype(float)
            q75 = df.groupby(group_key, observed=False)[col].transform(lambda s: np.nanpercentile(s.to_numpy(), 75)).astype(float)
            iqr = (q75 - q25).astype(float)

            # Create the aligned column with safe handling of zero/NaN IQR
            norm_col = f"{col}_norm"
            scale = ref_iqr / iqr
            df[norm_col] = (df[col].astype(float) - med) * scale + ref_med
            df.loc[~np.isfinite(scale) | (iqr == 0), norm_col] = df.loc[~np.isfinite(scale) | (iqr == 0), col]

            # Record the new column name for downstream dependent tracking
            norm_cols.append(norm_col)

        # Extend dependent list in-place with the new normalized columns
        deps += norm_cols


    @staticmethod
    def aggregate_geometric_data(
        geom_df: pd.DataFrame,
        cfg: Config,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
        """
        Aggregate geometric metrics at segment, neurite, and cell levels.

        Use:
            Clean and enrich the concatenated geometric segments table, create
            engineered bin columns (distance, electrotonic distance, radius, and
            percent-distance bins), then execute a suite of aggregation "jobs" that
            collapse per-segment rows into per-element (neurite/cell) values and
            compute group-level mean/SEM/count summaries.

        Args:
            geom_df (pd.DataFrame): Concatenated geometric segments dataframe.
            cfg (Config): Global configuration providing aggregation parameters.

        Returns:
            tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
                - geom_df: Cleaned/enriched dataframe with bin columns added.
                - geom_data: Mapping of job name -> {"el": elements_df, "stats": stats_df}.
        """
        # Clean and annotate geometry: drop root, coerce path level, derive species
        geom_df = geom_df[geom_df['ID'] != 1].copy()

        agg = cfg.aggregation

        # Read independent grouping keys and dependent metric lists from config
        independents = [f'path_structure_{i}' for i in agg.independents]
        seg_deps     = agg.geometric['per_segment']['dependents'][:]
        branch_deps  = agg.geometric['per_branch']['dependents'][:]
        neurite_deps = agg.geometric['per_neurite']['dependents'][:]
        cell_deps    = agg.geometric['per_cell']['dependents'][:]

        # Add percentile-normalized variants of signal-like metrics (in-place)
        if agg.norm_independent is not None:
            Aggregator._add_norm_columns(geom_df, seg_deps,  independents[agg.norm_independent])
            Aggregator._add_norm_columns(geom_df, cell_deps, independents[agg.norm_independent])

        # Prepare numeric bin edges for distance, radius, and percentage bins
        dist_bins = np.arange(*agg.geometric['per_segment']['binning']['distance'])
        rad_bins  = np.arange(*agg.geometric['per_segment']['binning']['radius'])
        pct_bins  = np.arange(*agg.geometric['per_segment']['binning']['percent'])

        # Discretize continuous features into interval bins for downstream grouping
        geom_df['dist_bin']                 = pd.IntervalIndex(pd.cut(geom_df['dist_from_soma_um'],            bins=dist_bins))
        geom_df['e_dist_bin']                = pd.IntervalIndex(pd.cut(geom_df['e_dist_from_soma'],  bins=dist_bins))
        geom_df['rad_bin']                  = pd.IntervalIndex(pd.cut(geom_df['Radius'],                           bins=rad_bins))
        geom_df['dist_pct_neurite_bin']     = pd.IntervalIndex(pd.cut(geom_df['dist_pct_neurite'],                 bins=pct_bins))
        geom_df['dist_pct_max_neurite_bin'] = pd.IntervalIndex(pd.cut(geom_df['dist_pct_max_neurite'],             bins=pct_bins))
        geom_df['e_dist_pct_neurite_bin']    = pd.IntervalIndex(pd.cut(geom_df['e_pct_neurite'],              bins=pct_bins))
        geom_df['e_dist_pct_max_neurite_bin']= pd.IntervalIndex(pd.cut(geom_df['e_pct_max_neurite'],          bins=pct_bins))

        # Map each engineered bin column to a concise suffix used in job names
        bin_cols = [
            ('dist_bin',                 'dist'),
            ('dist_pct_neurite_bin',     'dist_pct_neurite'),
            ('dist_pct_max_neurite_bin', 'dist_pct_max_neurite'),
            ('e_dist_bin',                'e_dist'),
            ('e_dist_pct_neurite_bin',    'e_dist_pct_neurite'),
            ('e_dist_pct_max_neurite_bin','e_dist_pct_max_neurite'),
            ('rad_bin',                  'rad'),
            ('branch_order',             'order'),
        ]

        # Define all geometric aggregation jobs (subject, bins, dependents, weighting)
        jobs = [
            *[dict(name=f'geom_neurite_{suffix}', bins=[bcol], subject='neurite', deps=seg_deps, use_wt=True) for bcol, suffix in bin_cols],
            *[dict(name=f'geom_cell_{suffix}',    bins=[bcol], subject='cell',    deps=seg_deps, use_wt=True) for bcol, suffix in bin_cols],
            dict(name='geom_branch',  bins=[], subject='branch',  deps=branch_deps,  use_wt=False),
            dict(name='geom_neurite', bins=[], subject='neurite', deps=neurite_deps, use_wt=False),
            dict(name='geom_cell',    bins=[], subject='cell',    deps=cell_deps,    use_wt=False),
        ]

        # Execute jobs with timing, collecting per-element stats
        geom_data = {}
        for job in jobs:
            with Aggregator.timed(f"aggregate_geometric_data:{job['name']}"):
                # Build grouping columns for this job from independents and bins
                gcols = independents + job['bins']

                # Collapse to per-element values and compute group mean/SEM/count
                el, stats = Aggregator._compute_basic_stats(
                    geom_df, group_cols=gcols, dependents=job['deps'], subject=job['subject'], use_length_weight=job['use_wt']
                )

                geom_data[job['name']] = {
                    'el':      el,
                    'stats':   stats
                }

        return geom_df, geom_data


    @staticmethod
    def aggregate_sholl_data(
        sholl_df: pd.DataFrame,
        cfg: Config,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
        """
        Aggregate Sholl metrics at radius, neurite, and cell levels.

        Use:
            Build per-element collapsed values and group-level summary statistics
            for Sholl outputs across a set of fixed aggregation jobs:
            per-radius and unbinned summaries for both cell- and neurite-level subjects.

        Args:
            sholl_df (pd.DataFrame): Concatenated Sholl dataframe.
            cfg (Config): Global configuration providing aggregation parameters.

        Returns:
            tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
                - df: Copy of the input Sholl dataframe (unmodified aside from copy).
                - sholl_data: Mapping of job name -> {"el": elements_df, "stats": stats_df}.
        """
        df = sholl_df.copy()

        agg = cfg.aggregation

        # Read config: grouping keys and dependent metrics
        independents = [f'path_structure_{i}' for i in agg.independents]
        radius_deps = agg.sholl['per_radius_dependents']
        cell_deps   = agg.sholl['per_subject_dependents']

        # Compose expanded dependent names for unbinned summaries
        per_cell_deps    = [f'cell_{r}_{c}'    for r in radius_deps for c in cell_deps]
        per_neurite_deps = [f'neurite_{r}_{c}' for r in radius_deps for c in cell_deps]

        # Define aggregation jobs (bins, subject, dependents)
        jobs = [
            dict(name='cell_r',    bins=['sholl_radius'], subject='cell',    deps=radius_deps),
            dict(name='cell',      bins=[],                 subject='cell',    deps=per_cell_deps),
            dict(name='neurite_r', bins=['sholl_radius'], subject='neurite', deps=radius_deps),
            dict(name='neurite',   bins=[],                 subject='neurite', deps=per_neurite_deps),
        ]

        # Run jobs with timing; compute element stats
        sholl_data = {}
        for job in jobs:
            with Aggregator.timed(f"aggregate_sholl_data:{job['name']}"):
                group_cols = independents + job['bins']
                el, stats = Aggregator._compute_basic_stats(
                    df, group_cols=group_cols, dependents=job['deps'], subject=job['subject']
                )
                sholl_data[job['name']] = {
                    'el':      el,
                    'stats':   stats
                }

        return df, sholl_data


    @staticmethod
    def _output_subdir(domain: str, job_name: str) -> str:
        """
        Map an aggregation job name to an output subdirectory.

        Use:
            Convert a job key into a stable folder path using:
            Domain -> Subject -> (Binned|Unbinned) -> (bin_name or Radius/sholl_radius).

        Args:
            domain (str): Output domain ("Geometric" or "Sholl").
            job_name (str): Aggregation job name (e.g. "geom_neurite_dist", "cell_r").

        Returns:
            str: Relative output subdirectory (no leading/trailing slash).
        """
        # Route geometric jobs by subject and optional bin suffix
        if domain == "Geometric":
            if not job_name.startswith("geom_"):
                return "Geometric/Other/Unbinned"

            rest = job_name[len("geom_"):]  # neurite_dist, cell, branch, ...
            parts = rest.split("_", 1)
            subject = parts[0].capitalize()

            # Unbinned jobs have no suffix after the subject
            if len(parts) == 1:
                return f"Geometric/{subject}/Unbinned"

            # Binned jobs encode the bin name after the first underscore
            bin_name = parts[1]
            return f"Geometric/{subject}/Binned/{bin_name}"

        # Route sholl jobs by subject and whether they are radius-binned
        if job_name.endswith("_r"):
            subject = job_name[:-2].capitalize()
            return f"Sholl/{subject}/Binned/Radius/sholl_radius"

        subject = job_name.capitalize()
        return f"Sholl/{subject}/Unbinned"


    @staticmethod
    def _save_full_csv(df: pd.DataFrame | None, name: str, template: str) -> None:
        """
        Save a DataFrame to CSV using standard CSV formatting.

        Use:
            Write to `template.format(name)` using comma separators and dot decimals.
            Name includes subdirectories (see _output_subdir).
            Creates the output directory if needed.

        Args:
            df (pd.DataFrame | None): DataFrame to persist (no-op if None).
            name (str): Logical table name used to format the output path.
            template (str): Path template containing a single '{}' placeholder.

        Returns:
            None
        """
        # No-op on missing table
        if df is None:
            return

        # Resolve output path from the naming template
        path = template.format(name)

        # Create the parent directory if required
        outdir = os.path.dirname(path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        # Write CSV with comma separators and dot decimals
        df.to_csv(path, sep=',', decimal='.', index=False, float_format="%.6f")


    @staticmethod
    def save_data_dicts(
        data_dict: dict[str, dict[str, pd.DataFrame | None]],
        template: str,
        prefix_override: str | None = None,
    ) -> None:
        """
        Persist a nested dictionary of DataFrames to CSV files.

        Use:
            Iterate over `{prefix -> {metric -> df}}` and emit one CSV per DataFrame
            using a consistent naming scheme.

            Output folders follow:
                Domain -> Subject -> (Binned|Unbinned) -> (bin_name or Radius/sholl_radius)

            Filenames remain unchanged; only directories change.

        Args:
            data_dict (dict[str, dict[str, pd.DataFrame | None]]): Nested mapping to persist.
            template (str): Path template containing a single '{}' placeholder.
            prefix_override (str | None): If provided, force a common prefix for all files.

        Returns:
            None
        """
        # Decide output domain from the caller's prefix choice
        domain = "Sholl" if prefix_override else "Geometric"

        # Walk jobs/metrics and emit one CSV per DataFrame
        for top_prefix, tables in data_dict.items():
            # Resolve the folder path for this job
            subdir = Aggregator._output_subdir(domain, top_prefix)

            for metric, df in tables.items():
                # Skip missing tables.
                if df is None:
                    continue

                # Build the output table name (optionally forcing a shared prefix)
                if prefix_override:
                    file_stem = f"{prefix_override}_{top_prefix}_{metric}"
                else:
                    file_stem = f"{top_prefix}_{metric}"

                # Persist the table using the configured template
                name = f"{subdir}/{file_stem}"
                Aggregator._save_full_csv(df, name, template)