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

    HUMAN_CELL_LINES = {"Sendai", "WTC", "123", "125"}

    GREAT_APE_CELL_LINES = {
        "Bonobo",
        "Chimp",
        "Chimpanzee",
        "Joc",
        "Sandra",
    }

    CELL_LINE_ALIASES = {
        "Chimp": "Sandra",
        "Chimpanzee": "Sandra",
    }


    @staticmethod
    def _canonical_cell_line(value: Any) -> str | None:
        """
        Normalize cell-line labels.

        Chimp and Chimpanzee are treated as Sandra.
        """
        if pd.isna(value):
            return None

        s = str(value).strip()
        return Aggregator.CELL_LINE_ALIASES.get(s, s)


    @staticmethod
    def _species_from_cell_line(value: Any) -> str | None:
        """
        Map a cell-line label to Human or Great Ape.
        """
        if pd.isna(value):
            return None

        raw = str(value).strip()
        canonical = Aggregator._canonical_cell_line(raw)

        if raw in Aggregator.HUMAN_CELL_LINES or canonical in Aggregator.HUMAN_CELL_LINES:
            return "Human"

        if raw in Aggregator.GREAT_APE_CELL_LINES or canonical in Aggregator.GREAT_APE_CELL_LINES:
            return "Great Ape"

        return None


    @staticmethod
    def _infer_cell_line_col(
        df: pd.DataFrame,
        preferred_cols: list[str] | None = None,
    ) -> str | None:
        """
        Infer which column contains cell-line labels by inspecting values.

        This avoids assuming that the cell line is the last configured independent.
        """
        known = (
            Aggregator.HUMAN_CELL_LINES
            | Aggregator.GREAT_APE_CELL_LINES
            | set(Aggregator.CELL_LINE_ALIASES)
            | set(Aggregator.CELL_LINE_ALIASES.values())
        )

        candidates: list[str] = []

        if preferred_cols:
            candidates.extend([c for c in preferred_cols if c in df.columns])

        candidates.extend(
            c for c in df.columns
            if c.startswith("path_structure_") and c not in candidates
        )

        best_col: str | None = None
        best_hits = 0

        for col in candidates:
            vals = df[col].dropna().astype(str).str.strip()
            if vals.empty:
                continue

            canonical_vals = vals.replace(Aggregator.CELL_LINE_ALIASES)
            hits = vals.isin(known).sum() + canonical_vals.isin(known).sum()

            if hits > best_hits:
                best_hits = int(hits)
                best_col = col

        return best_col if best_hits > 0 else None


    @staticmethod
    def _infer_coverslip_col(
        df: pd.DataFrame,
        preferred_cols: list[str] | None = None,
    ) -> str | None:
        """
        Infer which column contains coverslip labels by inspecting values.

        Coverslip labels are expected to look like CS1, CS2, CS3, etc.
        """
        candidates: list[str] = []

        if preferred_cols:
            candidates.extend([c for c in preferred_cols if c in df.columns])

        candidates.extend(
            c for c in df.columns
            if c.startswith("path_structure_") and c not in candidates
        )

        best_col: str | None = None
        best_hits = 0

        for col in candidates:
            vals = df[col].dropna().astype(str).str.strip()
            if vals.empty:
                continue

            hits = vals.str.match(r"(?i)^CS\d+$").sum()

            if hits > best_hits:
                best_hits = int(hits)
                best_col = col

        return best_col if best_hits > 0 else None


    @staticmethod
    def _pooled_base_cols(
        independents: list[str],
        *,
        cell_line_col: str | None,
        coverslip_col: str | None,
    ) -> list[str]:
        """
        Return grouping columns for pooled/diff outputs.

        The pooled/diff tables always remove:
            - the inferred cell-line column, because it is replaced by species
            - the inferred coverslip column, because coverslips should be pooled
        """
        exclude = {
            cell_line_col,
            coverslip_col,
        }

        return [
            c for c in independents
            if c not in exclude
        ]


    @staticmethod
    def _annotate_species(
        df: pd.DataFrame,
        preferred_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, str | None]:
        """
        Add cell_line_raw, cell_line, and species columns.

        The cell-line source column is inferred from dataframe values.
        """
        df = df.copy()

        cell_line_col = Aggregator._infer_cell_line_col(
            df,
            preferred_cols=preferred_cols,
        )

        if cell_line_col is None:
            df["cell_line_raw"] = pd.NA
            df["cell_line"] = pd.NA
            df["species"] = pd.NA
            return df, None

        df["cell_line_raw"] = df[cell_line_col].astype(str).str.strip()
        df["cell_line"] = df["cell_line_raw"].map(Aggregator._canonical_cell_line)
        df["species"] = df["cell_line_raw"].map(Aggregator._species_from_cell_line)

        return df, cell_line_col

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
        subject: str | None,
        sum_dependents: list[str],
        use_length_weight: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute per-element collapsed values and group summary stats.

        Use:
            1) Collapse row-level values into one value per biological element
               using the aggregation rule declared by the calling job.
            2) Aggregate those per-element values into group-level descriptive stats.

        Args:
            df (pd.DataFrame): Source dataframe.
            group_cols (list[str]): Grouping columns for aggregation.
            dependents (list[str]): Measure columns to aggregate.
            subject (str | None): Element unit: "cell", "neurite", or "branch".
            sum_dependents (list[str]): Dependents that represent additive row-level
                contributions and must be summed during the per-element collapse.
                All other dependents are averaged or length-weighted.
            use_length_weight (bool): If True, length-weight non-additive dependents
                by 'segment_length_um'.

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

        # Build grouping columns for the first-stage per-element collapse
        gcols = group_cols + subj_cols

        # The calling job explicitly declares which dependents are additive.
        # Everything else is treated as a repeated/continuous value and averaged.
        sum_deps = [d for d in dependents if d in sum_dependents]
        mean_deps = [d for d in dependents if d not in sum_dependents]

        # Collapse non-additive dependents by weighted or simple mean
        if mean_deps:
            if use_length_weight:
                elements = Aggregator._group_weighted_means(
                    df,
                    gcols,
                    mean_deps,
                    "segment_length_um",
                )
            else:
                elements = (
                    df.groupby(gcols, observed=False)[mean_deps]
                    .mean()
                    .reset_index()
                )
        else:
            # All dependents are additive; initialize one row per element group
            elements = (
                df[gcols]
                .drop_duplicates()
                .reset_index(drop=True)
            )

        # Collapse additive dependents by sum
        if sum_deps:
            sums = (
                df.groupby(gcols, observed=False)[sum_deps]
                .sum()
                .reset_index()
            )
            elements = elements.merge(sums, on=gcols, how="left")

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
    def _compute_pooled_stats(
        df: pd.DataFrame,
        base_cols: list[str],
        pooled_group_cols: list[str],
        dependents: list[str],
        subject: str | None,
        sum_dependents: list[str],
        use_length_weight: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Compute pooled species-level element values, species stats, and Human - Great Ape diff.

        Pooled grouping is:
            base_cols + ["species"] + pooled_group_cols

        where base_cols are the normal independent variables except the inferred
        cell-line column.
        """
        base_cols = list(base_cols or [])
        pooled_group_cols = list(pooled_group_cols or [])

        valid = df[df["species"].isin(["Human", "Great Ape"])].copy()

        if valid.empty:
            empty = pd.DataFrame()
            return empty, empty, empty

        pooled_group_cols_full = base_cols + ["species"] + pooled_group_cols

        pooled_el, pooled_stats = Aggregator._compute_basic_stats(
            valid,
            group_cols=pooled_group_cols_full,
            dependents=dependents,
            subject=subject,
            sum_dependents=sum_dependents,
            use_length_weight=use_length_weight,
        )

        diff_idx = base_cols + pooled_group_cols

        human = pooled_stats[pooled_stats["species"] == "Human"]
        great_ape = pooled_stats[pooled_stats["species"] == "Great Ape"]

        if human.empty or great_ape.empty:
            diff = pd.DataFrame(columns=diff_idx + ["species"])
            return pooled_el, pooled_stats, diff

        human = human.set_index(diff_idx).drop(columns=["species"], errors="ignore")
        great_ape = great_ape.set_index(diff_idx).drop(columns=["species"], errors="ignore")

        human_numeric = human.select_dtypes(include=["number"])
        great_ape_numeric = great_ape.select_dtypes(include=["number"])

        numeric_cols = human_numeric.columns.intersection(great_ape_numeric.columns)

        diff = human_numeric[numeric_cols].sub(
            great_ape_numeric[numeric_cols],
            fill_value=np.nan,
        )

        diff = diff.reset_index()
        diff["species"] = "Human - Great Ape"

        return pooled_el, pooled_stats, diff


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
        # Clean geometry by dropping the root segment
        geom_df = geom_df[geom_df['ID'] != 1].copy()

        agg = cfg.aggregation

        # Read independent grouping keys from config
        independents = [f'path_structure_{i}' for i in agg.independents]

        # Infer cell-line/species annotations on a separate dataframe so normal
        # el/stats outputs remain on the original aggregation path.
        geom_df_pooled, cell_line_col = Aggregator._annotate_species(
            geom_df,
            preferred_cols=independents,
        )

        # Infer coverslip column and always remove it from pooled/diff grouping.
        coverslip_col = Aggregator._infer_coverslip_col(
            geom_df,
            preferred_cols=independents,
        )

        pooled_base_cols = Aggregator._pooled_base_cols(
            independents,
            cell_line_col=cell_line_col,
            coverslip_col=coverslip_col,
        )

        # Read dependent metric lists from config
        seg_deps     = agg.geometric['per_segment']['dependents'][:]
        branch_deps  = agg.geometric['per_branch']['dependents'][:]
        neurite_deps = agg.geometric['per_neurite']['dependents'][:]
        cell_deps    = agg.geometric['per_cell']['dependents'][:]

        # Add percentile-normalized variants of signal-like metrics (in-place).
        # Keep separate dependent lists for normal and pooled paths.
        seg_deps_pooled = seg_deps[:]
        cell_deps_pooled = cell_deps[:]

        if agg.norm_independent is not None:
            Aggregator._add_norm_columns(geom_df, seg_deps,  independents[agg.norm_independent])
            Aggregator._add_norm_columns(geom_df, cell_deps, independents[agg.norm_independent])

            Aggregator._add_norm_columns(geom_df_pooled, seg_deps_pooled,  independents[agg.norm_independent])
            Aggregator._add_norm_columns(geom_df_pooled, cell_deps_pooled, independents[agg.norm_independent])

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

        # Add the same engineered bins to the pooled dataframe for pooled/diff outputs.
        geom_df_pooled['dist_bin']                 = pd.IntervalIndex(pd.cut(geom_df_pooled['dist_from_soma_um'],       bins=dist_bins))
        geom_df_pooled['e_dist_bin']                = pd.IntervalIndex(pd.cut(geom_df_pooled['e_dist_from_soma'],       bins=dist_bins))
        geom_df_pooled['rad_bin']                  = pd.IntervalIndex(pd.cut(geom_df_pooled['Radius'],                  bins=rad_bins))
        geom_df_pooled['dist_pct_neurite_bin']     = pd.IntervalIndex(pd.cut(geom_df_pooled['dist_pct_neurite'],        bins=pct_bins))
        geom_df_pooled['dist_pct_max_neurite_bin'] = pd.IntervalIndex(pd.cut(geom_df_pooled['dist_pct_max_neurite'],    bins=pct_bins))
        geom_df_pooled['e_dist_pct_neurite_bin']   = pd.IntervalIndex(pd.cut(geom_df_pooled['e_pct_neurite'],           bins=pct_bins))
        geom_df_pooled['e_dist_pct_max_neurite_bin']= pd.IntervalIndex(pd.cut(geom_df_pooled['e_pct_max_neurite'],      bins=pct_bins))

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

        # Explicit first-stage collapse rules for geometric metrics.
        # Per-segment length and puncta are additive contributions, so they sum when
        # segment rows are collapsed into a neurite/cell/bin. Branch-, neurite-, and
        # cell-level totals are already repeated scalars and therefore do not sum again.
        seg_sum_deps = [
            dep for dep in ["segment_length_um", "segment_puncta_count"]
            if dep in seg_deps
        ]
        seg_sum_deps_pooled = [
            dep for dep in ["segment_length_um", "segment_puncta_count"]
            if dep in seg_deps_pooled
        ]

        # Define all geometric aggregation jobs and their explicit collapse rules
        jobs = [
            *[dict(name=f'geom_neurite_{suffix}', bins=[bcol], subject='neurite', deps=seg_deps, pooled_deps=seg_deps_pooled, sum_deps=seg_sum_deps, pooled_sum_deps=seg_sum_deps_pooled, use_wt=True) for bcol, suffix in bin_cols],
            *[dict(name=f'geom_cell_{suffix}',    bins=[bcol], subject='cell',    deps=seg_deps, pooled_deps=seg_deps_pooled, sum_deps=seg_sum_deps, pooled_sum_deps=seg_sum_deps_pooled, use_wt=True) for bcol, suffix in bin_cols],
            dict(name='geom_branch',  bins=[], subject='branch',  deps=branch_deps,  pooled_deps=branch_deps,       sum_deps=[], pooled_sum_deps=[], use_wt=False),
            dict(name='geom_neurite', bins=[], subject='neurite', deps=neurite_deps, pooled_deps=neurite_deps,      sum_deps=[], pooled_sum_deps=[], use_wt=False),
            dict(name='geom_cell',    bins=[], subject='cell',    deps=cell_deps,    pooled_deps=cell_deps_pooled,  sum_deps=[], pooled_sum_deps=[], use_wt=False),
        ]

        # Execute jobs with timing, collecting per-element stats
        geom_data = {}
        for job in jobs:
            with Aggregator.timed(f"aggregate_geometric_data:{job['name']}"):
                # Build grouping columns for this job from independents and bins
                gcols = independents + job['bins']

                # Collapse to per-element values and compute group mean/SEM/count
                el, stats = Aggregator._compute_basic_stats(
                    geom_df,
                    group_cols=gcols,
                    dependents=job['deps'],
                    subject=job['subject'],
                    sum_dependents=job['sum_deps'],
                    use_length_weight=job['use_wt'],
                )

                # Collapse to pooled Human / Great Ape values and compute diff
                el_p, stats_p, diff = Aggregator._compute_pooled_stats(
                    geom_df_pooled,
                    base_cols=pooled_base_cols,
                    pooled_group_cols=job['bins'],
                    dependents=job['pooled_deps'],
                    subject=job['subject'],
                    sum_dependents=job['pooled_sum_deps'],
                    use_length_weight=job['use_wt'],
                )

                geom_data[job['name']] = {
                    'el':      el,
                    'stats':   stats,
                    'el_p':    el_p,
                    'stats_p': stats_p,
                    'diff':    diff,
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

        # Read independent grouping keys from config
        independents = [f'path_structure_{i}' for i in agg.independents]

        # Infer cell-line/species annotations on a separate dataframe so normal
        # el/stats outputs remain on the original aggregation path.
        df_pooled, cell_line_col = Aggregator._annotate_species(
            df,
            preferred_cols=independents,
        )

        # Infer coverslip column and always remove it from pooled/diff grouping.
        coverslip_col = Aggregator._infer_coverslip_col(
            df,
            preferred_cols=independents,
        )

        pooled_base_cols = Aggregator._pooled_base_cols(
            independents,
            cell_line_col=cell_line_col,
            coverslip_col=coverslip_col,
        )

        # Read dependent metrics from config
        radius_deps = agg.sholl['per_radius_dependents']
        cell_deps   = agg.sholl['per_subject_dependents']

        # Compose expanded dependent names for unbinned summaries
        per_cell_deps    = [f'cell_{r}_{c}'    for r in radius_deps for c in cell_deps]
        per_neurite_deps = [f'neurite_{r}_{c}' for r in radius_deps for c in cell_deps]

        # Define Sholl jobs and their explicit first-stage collapse rules.
        # cell_r combines multiple neurites at the same radius, so all per-radius
        # quantities sum. The other jobs operate on values already defined at the
        # target biological level, so they are not summed during the first collapse.
        jobs = [
            dict(name='cell_r',    bins=['sholl_radius'], subject='cell',    deps=radius_deps,       sum_deps=radius_deps),
            dict(name='cell',      bins=[],               subject='cell',    deps=per_cell_deps,     sum_deps=[]),
            dict(name='neurite_r', bins=['sholl_radius'], subject='neurite', deps=radius_deps,       sum_deps=[]),
            dict(name='neurite',   bins=[],               subject='neurite', deps=per_neurite_deps, sum_deps=[]),
        ]

        # Run jobs with timing; compute element stats
        sholl_data = {}
        for job in jobs:
            with Aggregator.timed(f"aggregate_sholl_data:{job['name']}"):
                group_cols = independents + job['bins']

                el, stats = Aggregator._compute_basic_stats(
                    df,
                    group_cols=group_cols,
                    dependents=job['deps'],
                    subject=job['subject'],
                    sum_dependents=job['sum_deps'],
                )

                el_p, stats_p, diff = Aggregator._compute_pooled_stats(
                    df_pooled,
                    base_cols=pooled_base_cols,
                    pooled_group_cols=job['bins'],
                    dependents=job['deps'],
                    subject=job['subject'],
                    sum_dependents=job['sum_deps'],
                    use_length_weight=False,
                )

                sholl_data[job['name']] = {
                    'el':      el,
                    'stats':   stats,
                    'el_p':    el_p,
                    'stats_p': stats_p,
                    'diff':    diff,
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