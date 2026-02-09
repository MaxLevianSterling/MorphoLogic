# src/morphologic/integration.py
from __future__ import annotations

# General imports (stdlib)
from typing import Any, Dict, List

# Third-party imports
import numpy as np
import pandas as pd
from scipy.stats import linregress


def _compute_sholl_stats(metric_radii_dict: Dict[float, float]) -> Dict[str, float]:
    """
    Compute summary statistics for a Sholl metric over radius.

    Use:
        Takes a mapping from Sholl radius (µm) to a metric value (e.g.
        intersections, segment lengths) and returns a set of summary
        statistics: AUC, critical radius, total, slope and correlation.

    Args:
        metric_radii_dict (Dict[float, float]): Mapping of radius (µm) to
            metric value at that radius.

    Returns:
        Dict[str, float]: Dictionary with the keys:
            - "auc": Area under the curve of value vs radius.
            - "crit": Sholl radius at which the metric reaches its maximum.
            - "total": Sum of all metric values across radii.
            - "slope": Slope of the linear fit (value vs radius).
            - "r": Pearson correlation coefficient of that linear fit.
    """
    # Split dict into parallel lists for sorting and numeric operations
    metric_radii = list(metric_radii_dict.keys())
    metric_values = list(metric_radii_dict.values())

    # Sort by radius so integration and regression use an ordered domain
    sorted_idx = np.argsort(metric_radii)
    radii_sorted = [metric_radii[i] for i in sorted_idx]
    values_sorted = [metric_values[i] for i in sorted_idx]

    # Generate sholl stats
    slope, _, r_value, _, _ = linregress(radii_sorted, values_sorted)
    auc_val = float(np.trapezoid(values_sorted, x=radii_sorted))
    crit_val = float(radii_sorted[int(np.argmax(values_sorted))])
    total_val = float(sum(values_sorted))

    return {
        "auc": auc_val,
        "crit": crit_val,
        "total": total_val,
        "slope": slope,
        "r": r_value,
    }


def build_sholl_dataframe(cell: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a long-form Sholl dataframe for a single cell.

    Use:
        Creates one row per (soma_id, neurite_id, sholl_radius) containing
        per-radius Sholl metrics, per-neurite Sholl summary stats, and
        cell-level Sholl summary stats.

    Args:
        cell (Dict[str, Any]): Processed cell dictionary containing:
            - "sholl_analysis_neurite": List of per-neurite Sholl dict bundles.
            - "sholl_analysis_cell": List containing a single cell-level Sholl dict bundle.
            - "geometric_analysis_neurite": List of per-neurite geometry metrics (used for neurite_length).
            - "path_structure": Optional list of hierarchical path tokens.
            - "file", "soma_id": Identity fields for this cell.

    Returns:
        pd.DataFrame: Long-form dataframe where each row corresponds to a single
        Sholl radius within a neurite and includes:
            - identity columns: soma_id, neurite_id, neurite_uid, neurite_length, file
            - per-radius metrics: intersections, segment_lengths, branch_points, terminal_points
            - per-neurite summary stats: neurite_*_{auc,crit,total,slope,r}
            - cell-level summary stats: cell_*_{auc,crit,total,slope,r}
    """
    # Compute cell-level summary stats once and reuse for every row
    cell_sholl = cell["sholl_analysis_cell"][0]  # single cell-level entry
    cell_summary_cols = {}
    for metric_key in ["intersections", "segment_lengths", "branch_points", "terminal_points"]:
        stats = _compute_sholl_stats(cell_sholl[f"sholl_{metric_key}"])
        cell_summary_cols[f"cell_{metric_key}_auc"]   = stats["auc"]
        cell_summary_cols[f"cell_{metric_key}_crit"]  = stats["crit"]
        cell_summary_cols[f"cell_{metric_key}_total"] = stats["total"]
        cell_summary_cols[f"cell_{metric_key}_slope"] = stats["slope"]
        cell_summary_cols[f"cell_{metric_key}_r"]     = stats["r"]

    # Accumulate row dicts and materialize a dataframe at the end
    rows: List[Dict[str, Any]] = []

    # Emit one row per radius for each neurite while attaching summaries as repeated columns
    for neurite_id, sholl_dicts in enumerate(cell["sholl_analysis_neurite"]):
        # Compute neurite-level summary stats over a canonical set of radii
        neurite_summary_cols: Dict[str, float] = {}

        # Canonical radii are taken from intersections to keep metrics aligned by radius
        canonical_radii = sorted(sholl_dicts["sholl_intersections"].keys())
        for metric_key in ["intersections", "segment_lengths", "branch_points", "terminal_points"]:
            base_dict = sholl_dicts[f"sholl_{metric_key}"]
            metric_dict = {
                r: float(base_dict[r]) for r in canonical_radii
            }
            stats = _compute_sholl_stats(metric_dict)
            neurite_summary_cols[f"neurite_{metric_key}_auc"]   = stats["auc"]
            neurite_summary_cols[f"neurite_{metric_key}_crit"]  = stats["crit"]
            neurite_summary_cols[f"neurite_{metric_key}_total"] = stats["total"]
            neurite_summary_cols[f"neurite_{metric_key}_slope"] = stats["slope"]
            neurite_summary_cols[f"neurite_{metric_key}_r"]     = stats["r"]

        # Derive neurite identity fields that stay constant across radii
        radii_sorted = sorted(sholl_dicts["sholl_intersections"].keys())
        neurite_length = cell["geometric_analysis_neurite"][neurite_id]["length"]
        neurite_uid = f'{cell["soma_id"]}_{neurite_id}'

        # Create per-radius rows for this neurite
        for r in radii_sorted:
            row: Dict[str, Any] = {
                "soma_id":         cell["soma_id"],
                "neurite_id":      neurite_id,
                "neurite_uid":     neurite_uid,
                "neurite_length":  neurite_length,
                "file":            cell["file"],
                "sholl_radius":    r,
                "intersections":   sholl_dicts["sholl_intersections"][r],
                "segment_lengths": sholl_dicts["sholl_segment_lengths"][r],
                "branch_points":   sholl_dicts["sholl_branch_points"][r],
                "terminal_points": sholl_dicts["sholl_terminal_points"][r],
            }

            # Expand hierarchical path tokens into stable numbered columns
            for iV, value in enumerate(cell["path_structure"]):
                row[f"path_structure_{iV+1}"] = value

            # Attach cell-level and neurite-level summary columns to each per-radius row
            row.update(cell_summary_cols)
            row.update(neurite_summary_cols)

            rows.append(row)

    return pd.DataFrame(rows)


def extend_dataframe(
    cell_data: Dict[str, Any],
    signal_channels: List[int],
    voxel_size_um: float,
    puncta_enabled: bool,
) -> None:
    """
    Enrich neurite DataFrames with additional neurite, cell-level, and Sholl metrics.

    Use:
        Mutates each neurite geometry DataFrame in-place by attaching identity fields,
        path structure columns, signal density features, soma metrics, neurite aggregates,
        distance percentages, then appends a unified Sholl
        DataFrame into cell_data["sholl_dataframes"]

    Args:
        cell_data (Dict[str, Any]): Data structure containing geometric, Sholl, and morphological
            data for a specific cell including:
            - "geometric_dataframes": List of per-neurite geometry DataFrames
            - "geometric_analysis_cell": List with a single dict of cell-level totals
            - "geometric_analysis_neurite": List of per-neurite geometry totals
            - "neurite_segments": List of per-neurite segment bundles containing mask_area_px
            - "somatic_metrics": Dict of soma geometry and intensity metrics
            - "dendritic_tree_area": 2D convex hull area for the cell
            - "path_structure": Hierarchical path labels
            - "file", "soma_id": Identity fields for this cell
        signal_channels (List[int]): Channel indices used (1-based) for intensity-derived features
        voxel_size_um (float): Voxel size in micrometers used for area-related conversions
        puncta_enabled (bool): Whether puncta metrics are present for this run.

    Returns:
        None
    """
    # Local aliases for frequently accessed cell-level containers and IDs
    gdfs = cell_data["geometric_dataframes"]
    n_primaries = len(gdfs)
    file_path = cell_data["file"]
    soma_id = cell_data["soma_id"]
    path_structure = cell_data["path_structure"]
    dendritic_tree_area = cell_data["dendritic_tree_area"]

    # Cell-wide totals reused across all neurites
    geo_cell = cell_data["geometric_analysis_cell"][0]
    total_neurite_e_length = geo_cell["e_length"]
    total_neurite_length = geo_cell["length"]
    total_neurite_area_3d = geo_cell["surface_area"]
    total_neurite_volume = geo_cell["volume"]

    # Total 2D neurite area derived from segment mask areas (in µm²)
    total_neurite_area_2d = (
        sum(value for neurite in cell_data["neurite_segments"] for value in neurite["mask_area_px"].values())
        * voxel_size_um ** 2
    )

    # Total puncta counts derived from per-node puncta counts
    if puncta_enabled:
        soma_puncta_count = cell_data.get("soma_puncta_count", 0)
        total_neurite_puncta_count = sum(df["segment_puncta_count"].sum() for df in gdfs)
        total_puncta_count = soma_puncta_count + total_neurite_puncta_count

    # Global maxima across neurites used for percent-of-max distance features
    global_max_dist = max(df["dist_from_soma_um"].max() for df in gdfs)
    global_max_e_dist = max(df["e_dist_from_soma"].max() for df in gdfs)

    # Soma geometry scalars reused across all neurites
    soma_metrics = cell_data["somatic_metrics"]
    soma_max_diameter = soma_metrics["max_diameter"]
    soma_area_2d = soma_metrics["area_2d"]

    # Precompute soma intensities / densities per channel
    soma_signal_density_vals = {}
    for channel in signal_channels:
        soma_signal_density_vals[channel] = soma_metrics[f"signal_intensity_{channel}"] / 2 # This uses a rough approximation of soma 3D surface area (2x 2D area)
        # Be careful with comparing somatic density to dendritic density, they rely on different surface area estimates

    # Enrich each neurite dataframe with identity fields and metrics
    for i, df in enumerate(gdfs):
        # Per-neurite identity fields for downstream aggregation
        df["neurite_id"] = i
        df["neurite_uid"] = f"{soma_id}_{i}"
        df["soma_id"] = soma_id
        df["file"] = file_path

        # Expand hierarchical path tokens into stable numbered columns
        for iV, value in enumerate(path_structure):
            df[f"path_structure_{iV+1}"] = value

        # Segment-level signal densities per channel
        for channel in signal_channels:
            df[f"signal_density_{channel}"] = (
                df[f"signal_intensity_{channel}"] * df["segment_mask_area_um"] / df["segment_surface_um"]
            )

        # Cell-wide totals repeated for every row
        df["n_primaries"] = n_primaries
        df["dendritic_tree_area"] = dendritic_tree_area
        df["total_neurite_e_length"] = total_neurite_e_length
        df["total_neurite_length"] = total_neurite_length
        df["total_neurite_area_2d"] = total_neurite_area_2d
        df["total_neurite_area_3d"] = total_neurite_area_3d
        df["total_neurite_volume"] = total_neurite_volume

        # Puncta scalars repeated for every row
        if puncta_enabled:
            df["soma_puncta_count"] = soma_puncta_count
            df["total_neurite_puncta_count"] = total_neurite_puncta_count
            df["total_puncta_count"] = total_puncta_count

        # Soma geometry and channel features repeated for every row
        df["soma_max_diameter"] = soma_max_diameter
        df["soma_area_2d"] = soma_area_2d          
        for channel in signal_channels:
            df[f"soma_signal_density_{channel}"] = soma_signal_density_vals[channel]

        # Neurite-level aggregates repeated for every row in this neurite dataframe
        df["neurite_e_length"] = cell_data["geometric_analysis_neurite"][i]["e_length"]
        df["neurite_length"] = cell_data["geometric_analysis_neurite"][i]["length"]
        df["neurite_surface_area"] = cell_data["geometric_analysis_neurite"][i]["surface_area"]
        df["neurite_volume"] = cell_data["geometric_analysis_neurite"][i]["volume"]
        if puncta_enabled:
            df["neurite_puncta_count"] = df["segment_puncta_count"].sum()

        # Distance percentiles relative to this neurite and to the global maximum across neurites
        df["dist_pct_neurite"] = df["dist_from_soma_um"] / df["dist_from_soma_um"].max()
        df["e_pct_neurite"] = df["e_dist_from_soma"] / df["e_dist_from_soma"].max()
        df["dist_pct_max_neurite"] = df["dist_from_soma_um"] / global_max_dist
        df["e_pct_max_neurite"] = df["e_dist_from_soma"] / global_max_e_dist

    # Build and append a unified Sholl dataframe
    sholl_df = build_sholl_dataframe(cell_data)
    cell_data["sholl_dataframes"].append(sholl_df)

    return