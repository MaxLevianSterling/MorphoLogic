# src/morphologic/topology.py
from __future__ import annotations

# General imports (stdlib)
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# General imports (third-party)
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import rasterize
from scipy.spatial import ConvexHull
from shapely import area, box, intersection
from shapely.geometry import Polygon

# Local imports
from .exceptions import (
    MetricComputationError,
    SWCParseError,
    ValidationError,
)


@dataclass
class Tree:
    """
    Minimal tree representation derived from an SWC DataFrame.

    Attributes:
        soma_id (int): Node ID chosen as the soma/root.
        parents (Dict[int, int]): Mapping child_id → parent_id
            (parent_id == -1 for roots).
        children (Dict[int, List[int]]): Mapping parent_id → list of child IDs.
        pos (Dict[int, np.ndarray]): Mapping node_id → np.array([X, Y]) positions.
    """
    soma_id: int
    parents: Dict[int, int]
    children: Dict[int, List[int]]
    pos: Dict[int, np.ndarray]


def build_tree(df: pd.DataFrame, soma_id: int) -> Tree:
    """
    Build a Tree object from an SWC DataFrame.

    Use:
        Convert SWC-style parent pointers into a convenient structure
        with parents, children, and per-node (X, Y) positions.

    Args:
        df (pd.DataFrame): SWC DataFrame with 'ID' and 'Parent' columns.
        soma_id (Optional[int]): Soma/root node ID. This must be provided; this helper
            does not infer the soma from the dataframe.

    Returns:
        Tree: A Tree instance with:
            - soma_id
            - parents
            - children
            - pos: node_id → np.array([X, Y]) positions.
    """
    # Decide which node ID is treated as the soma/root for traversal
    soma_id = int(soma_id)
    parents = {}
    children = defaultdict(list)

    # Convert SWC parent pointers into parent/child lookup tables
    for r in df.itertuples(index=False):
        nid = int(r.ID); par = int(r.Parent)
        parents[nid] = par
        if par > 0:
            children[par].append(nid)

    # Build node_id → (X, Y) position lookup for downstream geometry checks
    pos = node_positions(df)

    # Package topology + positions into the Tree dataclass
    return Tree(soma_id=soma_id, parents=parents, children=dict(children), pos=pos)


def node_positions(df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Map SWC node IDs to 2D coordinate vectors.

    Use:
        Build a lookup dict mapping each node's integer SWC ID to its (X, Y)
        coordinate vector for geometric operations and distance checks.

    Args:
        df (pd.DataFrame): SWC DataFrame with an 'ID' column and 'X', 'Y' columns.

    Returns:
        Dict[int, np.ndarray]: Mapping node_id → np.array([X, Y]).

    Raises:
        SWCParseError: If required coordinate columns are missing.
    """
    # Validate required coordinate columns exist before building the position map
    for c in ("X", "Y"):
        if c not in df:
            raise SWCParseError(f"SWC dataframe missing coordinate column {c!r}")

    # Build node_id → (X, Y) vector mapping for fast geometric lookups
    pos: Dict[int, np.ndarray] = {}
    for r in df.itertuples(index=False):
        nid = int(r.ID)
        pos[nid] = np.array([float(r.X), float(r.Y)], dtype=float)

    # Return the completed position lookup
    return pos


def soma_center(df: pd.DataFrame, voxel_size: float) -> Tuple[int, Dict[str, Tuple[float, float]]]:
    """
    Return soma node id and soma center in µm and pixels.

    Use:
        Find the soma row as the unique row with Parent == -1, then return:
          - soma_node_id (int)
          - soma_center: {"um": (x,y), "px": (x/voxel, y/voxel)}

    Args:
        df (pd.DataFrame): SWC DataFrame with at least 'ID', 'Parent', 'X', 'Y'.
        voxel_size (float): µm per pixel conversion factor.

    Returns:
        Tuple[int, Dict[str, Tuple[float, float]]]:
            (soma_node_id, {"um": (x, y), "px": (x_px, y_px)})

    Raises:
        SWCParseError: If the soma row (Parent == -1) is missing or non-unique,
            or required columns are missing.
    """
    # Validate required SWC columns are present before any processing
    required = {"ID", "Parent", "X", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise SWCParseError(f"SWC missing required columns: {sorted(missing)}")

    # Identify the somatic node
    soma_rows = df.loc[df["Parent"].astype(int) == -1]
    if len(soma_rows) != 1:
        raise SWCParseError(f"Expected exactly 1 soma row with Parent == -1, found {len(soma_rows)}")

    # Extract soma node id and coordinates from the soma row
    soma_row = soma_rows.iloc[0]
    soma_node_id = int(soma_row["ID"])

    x = float(soma_row["X"])
    y = float(soma_row["Y"])

    # Compute soma center in microns and convert to pixel units using voxel_size
    um = (x, y)
    px = (x / voxel_size, y / voxel_size)

    # Return the soma node id and both unit representations of the soma center
    return soma_node_id, {"um": um, "px": px}


def split_by_neurites(
    df: pd.DataFrame,
    root: int,
    soma_center_um: Optional[Tuple[float, float]] = None,
    max_root_offset_um: Optional[float] = None,
    split_branchpoints_within: int = 1,
) -> Dict[int, pd.DataFrame]:
    """
    Split an SWC tree into per-neurite subtrees, one per primary branch.

    Use:
        Additionally split neurites at branchpoints within N nodes of the soma
        (node-distance along edges). Each resulting neurite includes the full
        shared path from the soma-child through the split point, plus the unique
        subtree after the split.

        Depth convention:
            - A direct soma-child has depth 1
            - Its child has depth 2
            - Etc.

    Args:
        df (pd.DataFrame): SWC DataFrame with 'ID', 'Parent', and coordinate columns.
        root (int): Explicit soma/root ID passed through to `build_tree`.
        soma_center_um (Optional[Tuple[float, float]]): Soma center
            (x, y) in µm. Required if max_root_offset_um is set (to avoid
            recomputing the soma center).
        max_root_offset_um (Optional[float]): If given, validate that each direct
            soma-child is within this distance of the soma center (in µm).
        split_branchpoints_within (int): (min. 1) Split at any branchpoint node whose distance
            from soma (in nodes/edges) is <= this value.

    Returns:
        Dict[int, pd.DataFrame]:
            Mapping "branch key" → neurite DataFrame.

    Raises:
        ValidationError:
            If a direct soma-child is farther than `max_root_offset_um` from the soma center.
        ValidationError:
            If the user is trying to let a secondary dendrite start at the soma.
    """
    # Build the parent/children structure (soma-rooted) for traversal
    tree = build_tree(df, root)

    # Prepare soma center vector only if we need distance validation
    if max_root_offset_um is not None:
        sx, sy = soma_center_um
        sc = np.array([sx, sy], dtype=float)

    # Helper: collect all descendants of a node (inclusive) using the children map
    def _descendants(start: int) -> List[int]:
        out: List[int] = []
        q = deque([start])
        while q:
            n = q.popleft()
            out.append(n)
            q.extend(tree.children.get(n, []))
        return out
    
    # Helper: remap IDs inside a neurite DF to 1..N and set external parents to -1
    def _remap_ids(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        id_list = sub["ID"].astype(int).tolist()
        id_map = {old: i for i, old in enumerate(id_list, start=1)}
        sub["Parent"] = sub["Parent"].map(lambda x: id_map.get(int(x), -1)).astype(int)
        sub["ID"] = sub["ID"].map(id_map).astype(int)
        sub.reset_index(drop=True, inplace=True)
        return sub

    # Build neurite sub-DFs (either one per soma-child, or split within depth)
    result: Dict[int, pd.DataFrame] = {}
    df_id = df["ID"].astype(int)

    for soma_child in tree.children.get(root, []):
        # Optionally validate the soma-child proximity to soma center
        if max_root_offset_um is not None:
            p = tree.pos[soma_child]
            d = float(np.linalg.norm(np.asarray((p[0], p[1]), dtype=float) - sc))
            if d > float(max_root_offset_um):
                raise ValidationError(
                    f"Neurite root ID {soma_child} is {d:.2f} µm from soma "
                    f"(threshold {max_root_offset_um} µm)."
                )

        # Splitting mode: split at any branchpoint within `split_branchpoints_within` nodes
        split_depth = int(split_branchpoints_within)
        if split_depth < 1:
            raise ValidationError(
                "Split_branchpoints_within must be >= 1; secondary dendrites cannot start at the soma."
            )

        def _walk(
            node: int,
            depth_from_soma: int,
            prefix: List[int],
            branch_key: int,
        ) -> List[Tuple[int, set]]:
            """
            Traverse from a direct soma-child down the "trunk" and optionally split early.

            Splitting rule:
                If a node within `split_depth` edges from the soma has 2+ children,
                produce one output per child. Each output contains:
                    - the shared path from the soma (root) to the branchpoint, and
                    - the full descendant subtree of the chosen child.

            Depth convention:
                depth_from_soma = 1 at the direct soma-child.

            Returns:
                List of (branch_key, node_id_set), where node_id_set are original SWC IDs.
                The `branch_key` is set to the ID of the first unique node after the split
                (or remains the soma-child ID if no split occurs).
            """
            # Extend the shared path prefix with the current node
            prefix2 = prefix + [int(node)]

            # If we've gone beyond the split depth, stop splitting and take the full subtree
            if depth_from_soma > split_depth:
                return [(int(branch_key), set(prefix) | set(_descendants(node)))]

            # If this node is a branchpoint within depth, fork into one output per child
            kids = tree.children.get(int(node), [])
            if len(kids) >= 2:
                out: List[Tuple[int, set]] = []
                for k in kids:
                    out.extend(_walk(int(k), depth_from_soma + 1, prefix2, branch_key=int(k)))
                return out

            # If there is exactly one child, keep walking down the trunk
            if len(kids) == 1:
                return _walk(int(kids[0]), depth_from_soma + 1, prefix2, branch_key=branch_key)

            # Leaf within split depth: this branch is just the accumulated path
            return [(int(branch_key), set(prefix2))]

        # Generate (key, ids) sets for this soma_child, starting at depth 1 from soma
        branches = _walk(
            node=int(soma_child),
            depth_from_soma=1,
            prefix=[],
            branch_key=int(soma_child),
        )

        # Materialize each branch into a filtered DataFrame (and always remap IDs)
        for key, ids_set in branches:
            sub = df[df_id.isin(ids_set)].copy()
            result[int(key)] = _remap_ids(sub)

    return result


def polygon_to_pixels(polygon: Polygon, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterize a Shapely polygon onto a pixel grid.

    Use:
        Given a Shapely polygon and an image shape (H, W), return the row
        and column indices of all pixels whose unit square intersects the polygon.
        This is used to build masks for neurite segments.

    Args:
        polygon (shapely.Polygon): Polygon in (x, y) space.
        image_shape (Tuple[int, int]): Target image shape as (height, width).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - rr: row indices (y) of pixels that intersect the polygon.
            - cc: column indices (x) of the same pixels.
    """
    # Restrict the scan to the polygon's bounding box, clipped to image bounds
    min_x, min_y, max_x, max_y = polygon.bounds

    # Convert bounds to integer pixel ranges (x→cols, y→rows)
    h = int(image_shape[0])
    w = int(image_shape[1])
    min_col = max(0, int(np.floor(min_x)))
    max_col = min(w, int(np.ceil(max_x)))
    min_row = max(0, int(np.floor(min_y)))
    max_row = min(h, int(np.ceil(max_y)))
    
    # Compute local bounding window size in pixels
    win_w = max_col - min_col
    win_h = max_row - min_row

    # Build an affine transform mapping the local window back to global pixel coordinates
    transform = Affine.translation(min_col, min_row)

    # Rasterize within the local bounding window
    mask = rasterize(
        [(polygon, 1)],
        out_shape=(win_h, win_w),
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=True,
        dtype=np.uint8,
    )

    # Extract row/col indices of pixels set in the local mask
    rr_rel, cc_rel = np.nonzero(mask)

    # Convert local indices back to absolute image indices
    rr_array = (rr_rel + min_row).astype(int)
    cc_array = (cc_rel + min_col).astype(int)

    # Return absolute (row, col) index arrays
    return rr_array, cc_array


def approximate_frustum_convexhull(
    x1: float, y1: float, x2: float, y2: float,
    x3: float, y3: float, x4: float, y4: float,
    n_segments: int = 20,
) -> Tuple[float, float]:
    """
    Approximate lateral surface area and volume of a generalized frustum via 3D convex hull.

    Use:
        Given two line segments representing diameters of circles (endpoints
        in the XY plane), construct two edge-on circles in 3D, build the
        convex hull of their union, and interpret:
          - hull.area minus the two circular caps as lateral surface area,
          - hull.volume as frustum volume.

    Args:
        x1, y1, x2, y2 (float): Diameter endpoints of the first circle.
        x3, y3, x4, y4 (float): Diameter endpoints of the second circle.
        n_segments (int): Number of points used to discretize each circle.

    Returns:
        Tuple[float, float]:
            - lateral_area: Lateral surface area of the frustum.
            - volume: Volume enclosed by the convex hull.

    Raises:
        MetricComputationError:
            If the diameter is degenerate (points too close) or a valid
            perpendicular cannot be constructed.
    """

    # Precompute base circle angles for vectorized point generation
    thetas = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    def param_edge_on_circle(
        x1: float, y1: float, x2: float, y2: float,
        start_x: float, start_y: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate a set of 3D coordinates representing a circle lying edge-on in the XY-plane.

        Args:
            x1, y1, x2, y2 (float): Coordinates defining the diameter of the circle.
            start_x, start_y (float): Starting point for parametric ordering.

        Returns:
            tuple:
                - np.ndarray: Array of shape (N, 3) containing 3D coordinates of the circle.
                - float: The radius of the circle.
        """
        # Compute the circle's center and radius
        center_2d = 0.5 * np.array([x1 + x2, y1 + y2])
        radius = 0.5 * np.hypot(x2 - x1, y2 - y1)

        # Define the local coordinate system for the circle
        d = np.array([x2 - x1, y2 - y1, 0.0])
        zhat = np.array([0, 0, 1.0])
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-12:
            raise MetricComputationError("Diameter is degenerate (points too close) when building frustum circle.")

        # Build an orthonormal basis for the circle's plane in 3D.
        u = d / norm_d
        n = np.cross(u, zhat)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            raise MetricComputationError("Could not build perpendicular direction in XY plane for frustum circle.")
        n /= norm_n
        v = np.cross(n, u)

        # Embed the 2D center into 3D (z = 0), since the frustum geometry is constructed in 3D.
        center_3d = np.array([center_2d[0], center_2d[1], 0.0])

        # Determine starting angle (Solve θ0) to align parametric ordering with (start_x, start_y)
        desired_pt = np.array([start_x, start_y, 0.0])
        VR = (desired_pt - center_3d) / radius
        cos0 = np.dot(VR, u)
        sin0 = np.dot(VR, v)
        theta0 = np.arctan2(sin0, cos0)

        # Generate points along the perimeter of the circle
        c0 = np.cos(theta0)
        s0 = np.sin(theta0)
        cos_a = c0 * cos_t - s0 * sin_t
        sin_a = s0 * cos_t + c0 * sin_t
        circle_pts = center_3d + radius * (cos_a[:, None] * u + sin_a[:, None] * v)

        return circle_pts, radius

    # Generate discretized points for Circle A
    circleA_pts, rA = param_edge_on_circle(
        x1, y1, x2, y2,
        start_x=x1, start_y=y1
    )

    # Generate discretized points for Circle B
    circleB_pts, rB = param_edge_on_circle(
        x3, y3, x4, y4,
        start_x=x4, start_y=y4
    )

    # Collect all 3D points to construct the convex hull
    points_3d = np.vstack((circleA_pts, circleB_pts))

    # Compute the convex hull of the collected points
    hull = ConvexHull(points_3d)

    # Extract total surface area and volume from the hull
    total_area = hull.area
    volume     = hull.volume

    # Compute and subtract the area of the two circular caps
    area_capA = np.pi * rA**2
    area_capB = np.pi * rB**2
    lateral_area = total_area - (area_capA + area_capB)

    return lateral_area, volume


def weighted_mean_intensity(
    polygon: Polygon,
    rr: np.ndarray,
    cc: np.ndarray,
    img: np.ndarray,
    weights: np.ndarray | None = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Compute an area-weighted mean intensity over a polygon footprint.

    Use:
        Given a polygon and a candidate pixel mask (rr/cc), compute the
        polygon–pixel intersection area for each pixel and return the
        intensity weighted by those areas.

    Args:
        polygon: Shapely polygon in pixel coordinates.
        rr (np.ndarray): Row indices (y) of candidate pixels.
        cc (np.ndarray): Column indices (x) of candidate pixels.
        img (np.ndarray): 2D image array.
        weights (np.ndarray | None): Optional cached polygon–pixel intersection areas
            computed for this exact (polygon, rr, cc) after in-bounds filtering.

    Returns:
        Tuple[float, float, np.ndarray]:
            (mean_intensity, total_weight, weights), where total_weight is the sum of
            polygon–pixel intersection areas and weights are the per-pixel intersection
            areas (cached for reuse across channels).
    """
    # Filter indices to pixels that fall inside the image bounds
    h, w = img.shape[:2]
    m = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rr = rr[m]
    cc = cc[m]

    # Sample image intensities at the candidate pixels
    intensities = img[rr, cc].astype(np.float64, copy=False)

    # Compute polygon–pixel intersection areas to use as per-pixel weights
    if weights is None:
        boxes = box(cc, rr, cc + 1, rr + 1)
        weights = area(intersection(polygon, boxes)).astype(np.float64, copy=False)

    # Combine intensities with weights to form the area-weighted mean
    weighted_sum = float(np.sum(intensities * weights))
    total_weight = float(np.sum(weights))

    return weighted_sum / total_weight, total_weight, weights