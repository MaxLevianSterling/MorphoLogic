# src/morphologic/puncta.py
from __future__ import annotations

# General imports (stdlib)
from typing import Any, Dict

# Third-party imports
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

# Local imports
from .config import Config
from .soma import Soma


def assign_puncta(
    *,
    cfg: Config,
    cells_sorted: list[tuple[str, dict]],
    puncta_rois: Dict[str, Any],
) -> None:
    """
    Assign puncta ROIs to the closest soma ROI or neurite segment (edge-to-edge).

    Use:
        Build a set of target polygons from prepared cells (soma ROI polygon edges and
        neurite segment polygons). For each punctum point ROI, find the closest target
        by Shapely point-to-geometry distance (0 if overlapping). Distances are
        computed in pixel space; the configured max distance (µm) is converted to
        pixels via the voxel size. Puncta beyond that threshold from all targets
        are ignored.

    Args:
        cfg (Config): Global configuration providing voxel size and puncta distance threshold.
        cells_sorted (list[tuple[str, dict]]): Prepared cells for this image group.
        puncta_rois (Dict[str, Any]): Mapping roi_name -> punctum ROI object.

    Mutates:
        cells_sorted (list[tuple[str, dict]]):
            Updates each cell dict in-place by:
              - Appending puncta in pixel space to:
                    cell["soma_puncta_px"] and cell["neurite_puncta_px"]
              - Incrementing per-segment puncta counts in each prepared neurite dataframe:
                    cell["_prepared_neurites"][neurite_id]["df"]["segment_puncta_count"]
              - Incrementing per-soma puncta counts in:
                    cell["soma_puncta_count"]
    """
    # Compute maximum allowed puncta distance in pixels
    voxel_um = cfg.parameters.voxel_size
    max_dist_px = cfg.parameters.puncta_max_distance_um / voxel_um

    # Collect target polygons and stable metadata for assignments
    geoms: list = []
    meta_by_idx: list[dict] = []

    # Cache lookup from (soma_id, neurite_id, node_id) -> neurite df row index
    node_row_idx: dict[tuple[int, int, int], int] = {}

    # Cache lookup from soma_id -> cell dict
    cell_by_soma_id: dict[int, dict] = {c["soma_id"]: c for _, c in cells_sorted}

    # Accumulate per-node puncta increments to apply to dataframes in one pass
    puncta_increments: dict[tuple[int, int, int], int] = {}

    # Add soma ROI polygons and neurite segment polygons as assignment targets
    for _, cell in cells_sorted:
        # Convert soma ROI to a Shapely polygon in pixel space and append metadata
        soma_poly = Polygon(Soma.roi_points(cell["soma_roi"][1]))
        geoms.append(soma_poly)
        meta_by_idx.append({"kind": "soma", "soma_id": cell["soma_id"]})

        # Add each neurite segment polygon in pixel space
        for neurite_id, prepared in cell["_prepared_neurites"].items():
            # Grab the prepared neurite dataframe once per neurite so we can cache row indices by node ID
            id_to_row = dict(zip(prepared["df"]["ID"].to_numpy(), prepared["df"].index.to_numpy()))
            # Index each segment polygon and record which soma/neurite/node/index it belongs to for assignment
            for node_id, seg_poly in prepared["segments"]["polygon_px"].items():
                node_row_idx[(cell["soma_id"], neurite_id, node_id)] = id_to_row[node_id]
                geoms.append(seg_poly)
                meta_by_idx.append({
                    "kind": "segment",
                    "soma_id": cell["soma_id"],
                    "neurite_id": neurite_id,
                    "node_id": node_id,
                })

    # Build a spatial index for fast candidate retrieval within a distance envelope
    tree = STRtree(geoms)

    # Assign each punctum ROI to its closest target geometry
    for _, roi in puncta_rois.items():
        # Convert punctum ROI to a Shapely point in pixel space
        (cx, cy) = Soma.roi_points(roi)[0]
        puncta_pt = Point(float(cx), float(cy))

        # Query the index for nearby candidates using a buffered search region
        candidates = tree.query(puncta_pt.buffer(max_dist_px))

        # Track closest candidate
        best_dist = np.inf
        best_meta = None

        # Evaluate distances and keep the closest candidate (prefer soma on ties)
        for idx in candidates:
            g = geoms[idx]
            d = g.distance(puncta_pt)
            m = meta_by_idx[idx]

            if (d < best_dist) or (d == best_dist and m["kind"] == "soma"):
                best_dist = d
                best_meta = m

        # Drop puncta that have no nearby candidates or exceed the max distance threshold
        if best_meta is None or best_dist > max_dist_px:
            continue

        # Record the assignment either to the soma or to a specific neurite segment
        if best_meta["kind"] == "soma":
            # Cache the cell dict
            cell = cell_by_soma_id[best_meta["soma_id"]]

            # Store soma-assigned puncta for visualization
            cell.setdefault("soma_puncta_px", []).append((cx, cy))

            # Increment per-soma puncta count
            cell["soma_puncta_count"] = cell.get("soma_puncta_count", 0) + 1
        else:
            # Cache the cell dict
            cell = cell_by_soma_id[best_meta["soma_id"]]

            # Store segment-assigned puncta for visualization
            cell.setdefault("neurite_puncta_px", []).append((cx, cy))

            # Accumulate per-node puncta increments for a single df write pass later
            key = (best_meta["soma_id"], best_meta["neurite_id"], best_meta["node_id"])
            puncta_increments[key] = puncta_increments.get(key, 0) + 1
        
    # Apply accumulated per-node puncta increments to each neurite dataframe in one pass
    for (soma_id, neurite_id, node_id), inc in puncta_increments.items():
        cell = cell_by_soma_id[soma_id]
        df = cell["_prepared_neurites"][neurite_id]["df"]
        ridx = node_row_idx[(soma_id, neurite_id, node_id)]
        df.at[ridx, "segment_puncta_count"] += inc

    return