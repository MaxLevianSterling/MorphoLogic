# src/morphologic/soma.py
from __future__ import annotations

# General imports (stdlib)
from typing import Any, Dict, Optional, Tuple

# Third-party imports
import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon, Point

# Local imports
from .exceptions import MetricComputationError
from .topology import polygon_to_pixels, weighted_mean_intensity


class Soma:
    """
    Utilities for working with soma ROIs.

    All coordinates:
      - ROI polygons are in pixel coordinates (image space).
      - SWC soma position is in micrometers (μm) and converted using voxel_size_um.
    """

    @staticmethod
    def roi_points(roi: Any) -> list[tuple[float, float]]:
        """
        Extract ROI polygon vertices as (x, y) points in pixel coordinates.

        Use:
            Convert an ROI object into a standard vertex list suitable for
            point-in-polygon tests and centroid calculations. Supports ROIs
            that provide absolute subpixel vertices or integer vertices with
            an (left, top) offset.

        Args:
            roi (Any): ROI-like object with either:
                - subpixel_coordinates (np.ndarray): Absolute (x, y) vertices, or
                - integer_coordinates (np.ndarray) plus left/top offsets.

        Returns:
            list[tuple[float, float]]: Polygon vertices as (x_px, y_px) float tuples.
        """
        # Prefer absolute subpixel vertices when available, otherwise offset integer vertices
        if isinstance(roi.subpixel_coordinates, np.ndarray):
            coords = roi.subpixel_coordinates.copy()
        else:
            coords = roi.integer_coordinates.copy()
            coords[:, 0] += roi.left
            coords[:, 1] += roi.top

        # Normalize to plain python float tuples for downstream geometry utilities
        return [tuple(map(float, p[:2])) for p in coords]
    

    @classmethod
    def find_containing_soma(
        cls,
        soma_xy_px: Tuple[float, float],
        somatic_rois: Dict[str, Any],
        file: str,
    ) -> tuple[str, Any, np.ndarray]:
        """
        Locate the somatic ROI that contains the SWC soma center (in pixels).

        Use:
            Given a soma position in pixels and a mapping of ROI objects, find the
            unique ROI whose polygon contains the soma point.

        Args:
            soma_xy_px (Tuple[float, float]): Soma center in pixels as (x_px, y_px).
            somatic_rois (Dict[str, Any]): Mapping roi_name → ROI object.
            file (str): Soma filename.

        Returns:
            tuple[str, Any, np.ndarray]:
                (roi_name, roi_object, roi_center_px), where roi_center_px is a 2-element
                array [cx_px, cy_px] computed as the mean of ROI vertices.

        Raises:
            ValueError:
                If no ROIs contain the soma, or if multiple ROIs contain the soma.
        """
        # Use soma location in pixel space
        px, py = soma_xy_px

        # Collect all ROIs whose polygon contains the soma point
        containing: list[tuple[str, Any, np.ndarray]] = []
        for roi_name, roi in (somatic_rois or {}).items():
            pts = cls.roi_points(roi)  # list[(x_px, y_px)]
            if not pts:
                continue

            # Containment test in pixel space
            if Path(pts).contains_point((px, py)):
                pts_arr = np.asarray(pts, dtype=float)
                center_px = pts_arr.mean(axis=0)  # (cx_px, cy_px)
                containing.append((roi_name, roi, center_px))

        # Enforce a single containing ROI for unambiguous assignment
        if len(containing) == 0:
            raise ValueError(
                f"Filename: {file}\n"
                f"No soma ROI contains ({px:.1f},{py:.1f}) px; "
                f"available: {list((somatic_rois or {}).keys())}"
            )
        if len(containing) > 1:
            raise ValueError(
                f"Filename: {file}\n"
                f"Multiple soma ROIs contain ({px:.1f},{py:.1f}) px: {[c[0] for c in containing]}"
            )

        # Return the unique match: (roi_name, roi_obj, roi_center_px)
        return containing[0]
    

    @classmethod
    def get_soma_metrics(
        cls,
        soma_roi,
        nuclear_rois: Optional[Dict[str, object]],
        image_shape: np.ndarray,
        image_signals: Optional[Dict[int, np.ndarray]],
        voxel_um: float,
        deduct_nuclei: bool,
    ) -> Dict[str, float]:
        """
        Compute soma geometry and per-channel intensities with full compatibility.

        Use:
            Convert the soma ROI into a polygon, compute area and maximum diameter,
            then compute weighted mean intensities for each signal channel over the
            soma polygon. Optionally subtract nuclear-overlap contributions from
            the soma intensities.

        Args:
            soma_roi: Soma ROI object providing either subpixel_coordinates (absolute)
                or integer_coordinates with (left, top) offset.
            nuclear_rois (Optional[Dict[str, object]]): Mapping roi_name → nuclear ROI
                objects. Only used when deduct_nuclei is True.
            image_shape (np.ndarray): Reference image shape in pixels (Y, X).
            image_signals (Optional[Dict[int, np.ndarray]]): Mapping channel ID (1-based)
                to 2D signal image. If None or empty, only geometry metrics are returned.
            voxel_um (float): µm per pixel conversion factor.
            deduct_nuclei (bool): If True, subtract intensities over nuclear intersection
                polygons from soma intensities using area-weighted means.

        Returns:
            Dict[str, float]: Flat metrics dictionary containing:
                - "area_2d": soma area in µm²
                - "max_diameter": maximum soma diameter in µm
                - "signal_intensity_<channel>": mean intensity for each channel (unitless)

        Raises:
            MetricComputationError:
                If deduct_nuclei is True and no nuclear ROI overlaps the soma, if the soma
                is fully contained by a nuclear ROI, or if nuclear subtraction yields a
                non-positive remaining weight.
        """
        # Build an absolute-coordinate vertex array for the soma ROI polygon
        if isinstance(soma_roi.subpixel_coordinates, np.ndarray):
            coordinates = soma_roi.subpixel_coordinates.copy()
        else:
            coordinates = soma_roi.integer_coordinates.copy()
            coordinates[:, 0] += soma_roi.left
            coordinates[:, 1] += soma_roi.top
        
        # Convert the vertex array into a Shapely polygon in pixel coordinates
        vertices_x, vertices_y = coordinates.T
        soma_polygon = Polygon(zip(vertices_x, vertices_y)).buffer(0)

        # Compute soma area (µm²) and maximum convex-hull diameter (µm)
        mask_area = soma_polygon.area * (voxel_um ** 2)
        hull = soma_polygon.convex_hull
        hull_coords = list(hull.exterior.coords)
        max_diameter = 0.0
        for i in range(len(hull_coords)):
            for j in range(i + 1, len(hull_coords)):
                dist = Point(hull_coords[i]).distance(Point(hull_coords[j]))
                if dist > max_diameter:
                    max_diameter = dist
        max_diameter *= voxel_um

        # Initialize the output metrics dict with geometry scalars
        out = {
            "area_2d": float(mask_area),
            "max_diameter": float(max_diameter),
        }

        # Rasterize the soma polygon to candidate pixels for weighted intensity computation
        rr, cc = polygon_to_pixels(soma_polygon, image_shape)

        # Compute channel intensities over the soma polygon, with optional nuclear subtraction
        if image_signals:
            nuc_intersections: list[tuple[str, Polygon, np.ndarray, np.ndarray]] = []

            # Precompute soma ∩ nucleus intersection polygons (and their pixel masks) when enabled
            if deduct_nuclei:
                # Nuclear subtraction requires a nuclear ROI set
                if not nuclear_rois:
                    raise MetricComputationError("Nuclear deduction requested but no nuclear ROIs were provided.")
                
                # For each nuclear ROI, build a polygon in absolute pixel coordinates and intersect with the soma polygon.
                for nuc_roi_name, nuc_roi in nuclear_rois.items():
                    # Prefer absolute subpixel vertices when available; otherwise offset integer vertices by ROI bbox
                    if isinstance(nuc_roi.subpixel_coordinates, np.ndarray):
                        nuc_coords = nuc_roi.subpixel_coordinates.copy()
                    else:
                        nuc_coords = nuc_roi.integer_coordinates.copy()
                        nuc_coords[:, 0] += nuc_roi.left
                        nuc_coords[:, 1] += nuc_roi.top

                    # Construct a valid nuclear polygon in pixel coordinates
                    nuc_x, nuc_y = nuc_coords.T
                    nuclear_polygon = Polygon(zip(nuc_x, nuc_y)).buffer(0)
                    
                    # Compute the overlap region between soma and nucleus
                    intersection_polygon = soma_polygon.intersection(nuclear_polygon)
                    intersection_area = intersection_polygon.area

                    # If the nucleus fully covers the soma, subtraction is ill-defined (no remaining soma area)
                    if np.isclose(intersection_area, soma_polygon.area):
                        raise MetricComputationError(
                            f"Soma ROI is fully contained by nuclear ROI '{nuc_roi_name}'."
                        )
                    
                    # Keep only true overlaps; rasterize to candidate pixels for weighted intensity
                    if intersection_area > 0:
                        rr_nuc, cc_nuc = polygon_to_pixels(intersection_polygon, image_shape)
                        nuc_intersections.append((nuc_roi_name, intersection_polygon, rr_nuc, cc_nuc))
                
                # Require at least one overlapping nucleus if subtraction was requested
                if len(nuc_intersections) == 0:
                    raise MetricComputationError("No nuclear ROI overlaps with the soma; cannot deduct nuclei.")

            # Cache soma weights across channels (stay with soma)
            soma_wts = None

            # Cache nuclear weights across channels, keyed by nucleus ROI name (stay with their nucleus)
            nuc_wts: Dict[str, np.ndarray] = {}

            # Compute the weighted mean soma intensity per channel, subtracting nuclear overlap if requested
            for ch_name, ch_img in image_signals.items():
                # Weighted mean over soma polygon for this channel; reuse soma_wts after first channel
                avg, w, soma_wts = weighted_mean_intensity(soma_polygon, rr, cc, ch_img, weights=soma_wts)
                
                # Optional nucleus subtraction: subtract the weighted contribution of soma∩nucleus polygons
                if deduct_nuclei and nuc_intersections:
                    for nuc_roi_name, intersection_polygon, rr_nuc, cc_nuc in nuc_intersections:
                        # Reuse weights for this nucleus only (keyed by nuc_roi_name)
                        nuc_avg, nuc_w, nuc_wts[nuc_roi_name] = weighted_mean_intensity(
                            intersection_polygon,
                            rr_nuc,
                            cc_nuc,
                            ch_img,
                            weights=nuc_wts.get(nuc_roi_name),
                        )

                        # Subtract nucleus contribution in weighted-sum space, then renormalize by remaining weight
                        avg = ((avg * w) - (nuc_avg * nuc_w)) / (w - nuc_w)
                        w -= nuc_w

                        # Guard against full subtraction (or numeric issues) leaving no remaining soma weight
                        if w <= 0:
                            raise MetricComputationError(
                                f"After deducting nuclear ROI '{nuc_roi_name}', soma weight is zero/negative."
                            )
                
                # Store final per-channel soma intensity and approximate signal density (after optional nucleus subtraction)
                out[f"signal_intensity_{ch_name}"] = float(avg)
                out[f"signal_density_{ch_name}"] = float(avg) / 2 # This uses a rough approximation of soma 3D surface area (2x 2D area)

        return out