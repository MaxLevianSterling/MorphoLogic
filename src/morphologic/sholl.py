# src/morphologic/sholl.py
from __future__ import annotations

# General imports (stdlib)
import bisect

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from .exceptions import ValidationError


def analyze_sholl_segments(
    neurite_line_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    center_um: tuple[float, float],
    radii_um: list[float],
) -> tuple[dict[float, int], dict[float, float]]:
    """
    Compute Sholl intersections and in-shell segment lengths for neurite segments.

    Use:
        Given a list of neurite line segments in µm, compute:
          - how many times segments cross each Sholl circle,
          - how much segment length lies in each radial shell [r_i, r_{i+1}).

        This is the per-neurite Sholl engine used by Neurite.

    Args:
        neurite_line_segments (list[tuple[tuple[float, float], tuple[float, float]]]):
            Sequence of segment endpoints in µm, as returned by
            `neurite_segments['line_segments_um'].values()`.
        center_um (tuple[float, float]): Soma center in µm (x_um, y_um).
        radii_um (list[float]): Sholl radii in micrometers.

    Returns:
        Tuple[Dict[float, int], Dict[float, float]]:
            - intersections: dict[radius_um -> int] number of circle crossings.
            - segment_lengths: dict[radius_um -> float] total segment length
              inside each shell [r_in, r_out), keyed by r_out.
    
    Raises:
        ValidationError: If the smallest Sholl radius is not 0 µm.
    """
    # Validate Sholl range starts at 0 µm
    if not radii_um or float(radii_um[0]) != 0.0:
        raise ValidationError(
            f"Sholl radii must start at 0 µm; got first radius {radii_um[0] if radii_um else None}."
        )

    # Soma center is provided in micrometers (µm)
    center_x_um, center_y_um = center_um

    # Sort radii and define circle and shell boundaries in micrometers
    circle_radii_sq = [r ** 2 for r in radii_um]

    # Initialize outputs keyed by Sholl circle radii (and shell outer edges)
    intersections = {r: 0 for r in radii_um}
    segment_lengths = {r: 0.0 for r in radii_um}

    def find_intersection_tvals(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        cx: float,
        cy: float,
        radius: float,
    ) -> list[float]:
        # Solve the parametric line-circle intersection for t in [0, 1]
        dx, dy = x2 - x1, y2 - y1
        fx, fy = x1 - cx, y1 - cy
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius
        disc = b * b - 4 * a * c
        if disc < 0:
            return []
        root = np.sqrt(disc)
        t1 = (-b - root) / (2 * a)
        t2 = (-b + root) / (2 * a)
        return [t for t in (t1, t2) if 0 <= t <= 1]

    def clip_length_in_shell(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        r_in: float,
        r_out: float,
        d_min: float,
        d_max: float,
    ) -> float:
        """
        Compute the length of a segment inside the shell [r_in, r_out).

        Args:
            x1, y1, x2, y2 (float): Segment endpoints in µm.
            r_in (float): Inner radius of the shell (µm).
            r_out (float): Outer radius of the shell (µm).
            d_min (float): Min distance² of endpoints to soma.
            d_max (float): Max distance² of endpoints to soma.

        Returns:
            float: Segment length in µm that lies inside [r_in, r_out).
        """
        # Quickly reject segments that lie entirely inside or outside the radial band
        if d_max < r_in ** 2 or d_min >= r_out ** 2:
            return 0.0

        # Evaluate segment point P(t) for t in [0, 1]
        def pt(t: float) -> tuple[float, float]:
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        # Test whether a point lies in the radial shell [r_in, r_out)
        def inside_shell(xx: float, yy: float) -> bool:
            d2 = (xx - center_x_um) ** 2 + (yy - center_y_um) ** 2
            return (d2 >= r_in ** 2) and (d2 < r_out ** 2)

        # Collect candidate t-values from boundary intersections and in-shell endpoints
        pts: list[tuple[float, str]] = []
        for t in find_intersection_tvals(x1, y1, x2, y2, center_x_um, center_y_um, r_in):
            pts.append((t, "inner"))
        for t in find_intersection_tvals(x1, y1, x2, y2, center_x_um, center_y_um, r_out):
            pts.append((t, "outer"))
        if inside_shell(x1, y1):
            pts.append((0.0, "endpoint_in"))
        if inside_shell(x2, y2):
            pts.append((1.0, "endpoint_in"))

        if not pts:
            return 0.0

        # Sort candidates and integrate sub-intervals whose midpoint lies in the shell
        pts.sort(key=lambda z: z[0])

        total_len = 0.0
        for (ta, _), (tb, _) in zip(pts[:-1], pts[1:]):
            if tb <= ta:
                continue
            tm = 0.5 * (ta + tb)
            xm, ym = pt(tm)
            if inside_shell(xm, ym):
                xa, ya = pt(ta)
                xb, yb = pt(tb)
                total_len += np.hypot(xb - xa, yb - ya)

        return total_len

    # Walk each segment once and accumulate circle crossings and shell lengths
    for (x1, y1), (x2, y2) in neurite_line_segments:
        d1_sq = (x1 - center_x_um) ** 2 + (y1 - center_y_um) ** 2
        d2_sq = (x2 - center_x_um) ** 2 + (y2 - center_y_um) ** 2
        d_min = min(d1_sq, d2_sq)
        d_max = max(d1_sq, d2_sq)

        # Count circle straddles using segment endpoints with d_min <= r_sq < d_max by bisecting to the slice [lo, hi).
        lo = bisect.bisect_left(circle_radii_sq, d_min)
        hi = bisect.bisect_left(circle_radii_sq, d_max)
        for i in range(lo, hi):
            intersections[radii_um[i]] += 1

        # Accumulate length contribution within each radial shell [r_in, r_out)
        for i in range(len(radii_um) - 1):
            r_in = radii_um[i]
            r_out = radii_um[i + 1]
            seg_len_part = clip_length_in_shell(x1, y1, x2, y2, r_in, r_out, d_min, d_max)
            if seg_len_part > 0:
                segment_lengths[r_out] += seg_len_part

    return intersections, segment_lengths


def analyze_sholl_nodes(
    df: pd.DataFrame,
    branch_point_ids: list[int],
    terminal_point_ids: list[int],
    radii_um: list[float],
    center_um: tuple[float, float],
) -> tuple[dict[float, int], dict[float, int]]:
    """
    Count branch and terminal points in each Sholl shell [r_i, r_{i+1}).

    Use:
        Given the positions of branch and terminal nodes, assign each node
        to the shell [r_i, r_{i+1}) based on its distance from the soma and
        count how many fall into each shell.

    Args:
        df (pd.DataFrame): DataFrame with columns 'ID', 'X', 'Y' (µm).
        branch_point_ids (list[int]): Node IDs of branch points.
        terminal_point_ids (list[int]): Node IDs of terminal tips.
        radii_um (list[float]): Sholl radii in µm.
        center_um (tuple[float, float]): Soma center in µm (x_um, y_um).

    Returns:
        tuple[dict[float, int], dict[float, int]]:
            - sholl_branch_counts: dict[radius_um -> int] branch points in
              each shell [r_i, r_{i+1}).
            - sholl_terminal_counts: dict[radius_um -> int] terminal points
              in each shell.

    Raises:
        ValidationError: If a branch or terminal point lies outside the
            Sholl range [0, max(radii_um)], i.e. cannot be assigned to
            any shell.
        ValidationError: If the smallest Sholl radius is not 0 µm.
    """
    # Validate Sholl range starts at 0 µm
    if not radii_um or float(radii_um[0]) != 0.0:
        raise ValidationError(
            f"Sholl radii must start at 0 µm; got first radius {radii_um[0] if radii_um else None}."
        )

    # Soma center is provided in micrometers (µm)
    center_x_um, center_y_um = center_um

    # Define shell edges [0,r1), [r1,r2), ..., using squared edges for fast binning
    radii_um_sq = [r ** 2 for r in radii_um]

    # Initialize counts keyed by the outer edge of each shell (r1, r2, ..., rN)
    sholl_branch_counts = {r: 0 for r in radii_um}
    sholl_terminal_counts = {r: 0 for r in radii_um}

    def assign_point_to_shell(point_id: int, counts_dict: dict[float, int], point_type: str) -> None:
        """
        Assign a single point to a radial shell and increment the corresponding count.

        Args:
            point_id (int): Node ID of the point to assign.
            counts_dict (dict[float, int]): Shell counts keyed by outer radius (µm).
            point_type (str): Label used in error messages ("branch" or "terminal").

        Raises:
            ValidationError: If the point lies outside all shells.
        """
        # Look up the node position in micrometers and compute squared distance to the soma
        x_um, y_um = df.loc[df["ID"] == point_id, ["X", "Y"]].values[0]
        dist_sq = (x_um - center_x_um) ** 2 + (y_um - center_y_um) ** 2

        # Find the shell index i such that radii_um[i] <= dist < radii_um[i+1]
        idx = bisect.bisect_right(radii_um_sq, dist_sq) - 1

        # Increment the count for the shell's outer edge; error if the point falls outside the range
        if 0 <= idx < len(radii_um) - 1:
            r_out = radii_um[idx + 1]
            counts_dict[r_out] += 1
        else:
            max_r = radii_um[-1]
            raise ValidationError(
                f"{point_type.capitalize()} point ID {point_id} at "
                f"{np.sqrt(dist_sq):.2f} µm is outside the Sholl range "
                f"(max shell edge {max_r} µm)."
            )

    # Assign all branch points to shells
    for pid in branch_point_ids:
        assign_point_to_shell(pid, sholl_branch_counts, "branch")

    # Assign all terminal points to shells
    for pid in terminal_point_ids:
        assign_point_to_shell(pid, sholl_terminal_counts, "terminal")

    return sholl_branch_counts, sholl_terminal_counts