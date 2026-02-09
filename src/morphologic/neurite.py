# src/morphologic/neurite.py
from __future__ import annotations

# General imports (stdlib)
import itertools
from collections import defaultdict, deque
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

# Local imports
from .config import Config
from .exceptions import ValidationError
from .sholl import analyze_sholl_nodes, analyze_sholl_segments
from .topology import (
    approximate_frustum_convexhull,
    weighted_mean_intensity,
    polygon_to_pixels,
)


class Neurite:
    """
    Pipeline:
      1) Pre-clean SWC (merge close branch points, remove short branches, drop OOB nodes, enforce min segment length)
      2) (Optional) Smooth radii (if provided/specified)
      3) Branch order
      4) Segment construction (edges, lengths; optional per-pixel paths for intensities)
      5) Branch- & neurite-level metrics
      6) Sholl analysis (segments + nodes)
    """

    def __init__(
        self,
        cfg: Config,
        soma_id: int,
        image_shape: Tuple[int, int],
        voxel_size_um: float,
        *,
        filename: str,
        min_bp_distance: int,
        min_branch_length: int,
        min_segment_length_um: float,
        smooth_radii: bool,
        radii_smoothing_window_length: int,
        radii_smoothing_interval: int,
        radii_smoothing_min: float,
        soma_center_um: Optional[Tuple[float, float]],
    ):
        """
        Initialize a Neurite class with geometry, smoothing and Sholl settings.

        Use:
            Construct once per cell (or per configuration) and reuse it for
            each neurite DataFrame belonging to that cell.

        Args:
            cfg (Config): Global configuration providing Sholl radii and other defaults.
            soma_id (int): Globally unique soma identifier for this cell.
            image_shape (Tuple[int, int]): Image size as (height_px, width_px).
            voxel_size_um (float): Voxel size in micrometers (µm per pixel),
                used to convert between pixel and physical units.
            filename (str): File context used in error messages.
            min_bp_distance (int): Minimum allowed node spacing, in *graph hops* when
                merging nearby branch points.
            min_branch_length (int): Minimum allowed branch length in nodes; shorter
                terminal twigs are removed.
            min_segment_length_um (float): Minimum segment length in µm; short segments
                are collapsed by rewiring nodes.
            smooth_radii (bool): If True, apply sliding-window regression to smooth
                radii along neurite paths.
            radii_smoothing_window_length (int): Sliding window length (nodes) used
                for radius smoothing.
            radii_smoothing_interval (int): Step between consecutive windows along a path.
            radii_smoothing_min (float): Minimum allowed radius after smoothing.
            soma_center_um (Optional[Tuple[float, float]]): Soma center in µm as
                (x_um, y_um). Required when running Sholl analysis.
        """
        self.cfg = cfg
        self.soma_id = soma_id
        self.image_shape = tuple(image_shape)
        self.voxel = float(voxel_size_um)
        self.filename = str(filename)
        self.min_bp_distance = int(min_bp_distance)
        self.min_branch_length = int(min_branch_length)
        self.min_segment_length_um = float(min_segment_length_um)
        self.smooth_radii = bool(smooth_radii)
        self.radii_window = int(radii_smoothing_window_length)
        self.radii_interval = int(radii_smoothing_interval)
        self.radii_min = float(radii_smoothing_min)
        self.soma_center_um = tuple(soma_center_um)


    def prepare(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Prepare a single neurite DataFrame for downstream computation.

        Use:
            Applies all pre-cleaning steps, optional radius smoothing, branch order
            computation, and segment construction. Prepared results are returned
            as a bundle for later computation (metrics + Sholl).

        Args:
            df (pd.DataFrame): Neurite SWC DataFrame containing at least the
                columns ['ID', 'Parent', 'X', 'Y', 'Radius'].

        Returns:
            tuple:
                - df (pd.DataFrame): Cleaned neurite DataFrame with branch order filled.
                - neurite_segments (dict): Segment bundle with keys such as
                'line_segments_um', 'cone_vertices_um', 'mask_area_px',
                'rr', 'cc', 'polygon_px'.
        """
        # Quality control
        df = self.merge_false_branch_points(df, self.min_bp_distance)
        df = self.remove_short_branches(df, self.min_branch_length)
        df = self.remove_extraneous_nodes(df, self.image_shape, self.voxel)
        df = self.enforce_segment_length(df, self.min_segment_length_um)

        # Smooth radii
        if self.smooth_radii and "Radius" in df.columns:
            df = self.smooth_radii_with_regression(
                df,
                filename=self.filename,
                window_length=self.radii_window,
                interval=self.radii_interval,
                min_radius=self.radii_min,
            )

        # Branch order
        df = self.calculate_branch_order(df)

        # Segments
        neurite_segments = self.construct_neurite_segments(df, self.image_shape, self.voxel, self.filename)

        return df, neurite_segments


    def process(
        self,
        nid: int,
        df: pd.DataFrame,
        neurite_segments: dict,
        image_signals: Optional[Dict[int, np.ndarray]],
    ) -> tuple[pd.DataFrame, dict[str, float], dict, dict[str, dict[float, float] | dict[float, int]]]:
        """
        Compute neurite geometry, signals and Sholl from prepared neurite inputs.

        Use:
            Applies branch/neurite metrics and Sholl analysis. All results are returned
            as a bundle.

        Args:
            nid (int): Locally unique neurite identifier.
            df (pd.DataFrame): Prepared neurite SWC DataFrame containing at least the
                columns ['ID', 'Parent', 'X', 'Y', 'Radius'].
            neurite_segments (dict): Segment bundle with keys such as
                'line_segments_um', 'cone_vertices_um', 'mask_area_px',
                'rr', 'cc', 'polygon_px'.
            image_signals (Optional[Dict[int, np.ndarray]]): Mapping from channel index
                (1-based) to 2D signal image arrays.

        Returns:
            tuple:
                - df (pd.DataFrame): Neurite DataFrame with per-node metrics filled.
                - neurite_metrics (dict[str, float]): Aggregated neurite-level totals
                  (length, surface_area, volume, e_length).
                - neurite_segments (dict): Segment bundle (passed through).
                - sholl_bundle (dict[str, Dict[float, float]]): Per-radius Sholl metrics:
                  'sholl_intersections', 'sholl_segment_lengths', 'sholl_branch_points',
                  'sholl_terminal_points'.
        """
        # Geometric
        (
            df,
            neurite_metrics,
            branch_points,
            terminal_points,
        ) = self.calculate_geometrics(
            df, nid, self.voxel, image_signals, neurite_segments
        )

        # Sholl segments
        line_segments_um = list(neurite_segments["line_segments_um"].values())
        sholl_intersections_neurite, sholl_segment_lengths_neurite = analyze_sholl_segments(
            line_segments_um,
            center_um=self.soma_center_um,
            radii_um=np.arange(*self.cfg.parameters.sholl_range).tolist(),
        )

        # Sholl nodes
        sholl_branch_points_neurite, sholl_terminal_points_neurite = analyze_sholl_nodes(
            df,
            branch_point_ids=branch_points,
            terminal_point_ids=terminal_points,
            radii_um=np.arange(*self.cfg.parameters.sholl_range).tolist(),
            center_um=self.soma_center_um,
        )

        # Bundle Sholl data
        sholl_bundle = {
            "sholl_intersections": sholl_intersections_neurite,
            "sholl_segment_lengths": sholl_segment_lengths_neurite,
            "sholl_branch_points": sholl_branch_points_neurite,
            "sholl_terminal_points": sholl_terminal_points_neurite,
        }

        return df, neurite_metrics, neurite_segments, sholl_bundle


    @staticmethod
    def build_adjacency(df: pd.DataFrame) -> Dict[int, List[int]]:
        """
        Build an undirected adjacency list from SWC IDs and Parents.

        Use:
            Treat each edge between a node and its parent as an undirected
            connection, and return a mapping node_id → list of neighbor IDs.

        Args:
            df (pd.DataFrame): SWC DataFrame with at least 'ID' and 'Parent'
                integer columns.

        Returns:
            Dict[int, List[int]]: Undirected adjacency list including the
            root node (Parent == -1) even if it has no children.
        """
        adj: Dict[int, List[int]] = defaultdict(list)
        
        # Add an undirected edge for each SWC (Parent, ID) pair
        for parent, child in df[["Parent", "ID"]].itertuples(index=False):
            parent = int(parent)
            child = int(child)

            # Link parent ↔ child for all non-root edges
            if parent != -1:
                adj[parent].append(child)
                adj[child].append(parent)
                continue

            # Ensure the root node appears in the adjacency map even if isolated
            adj.setdefault(child, [])

        # Return the neighbor map for graph traversal utilities
        return adj


    @staticmethod
    def bfs_distance(adj: Dict[int, List[int]], start: int, max_depth: int) -> Set[int]:
        """
        Compute nodes within a maximum number of hops from a start node.

        Use:
            Breadth-first search on an undirected adjacency list to find all
            nodes at graph distance ≤ `max_depth` from `start`.

        Args:
            adj (Dict[int, List[int]]): Undirected adjacency list.
            start (int): Start node ID.
            max_depth (int): Maximum number of hops allowed.

        Returns:
            Set[int]: Set of node IDs within the requested hop distance.
        """
        if max_depth <= 0:
            return set()
        seen, out = {start}, set()
        q = deque([(start, 0)])
        while q:
            nid, d = q.popleft()
            if d == max_depth:
                continue
            for nb in adj.get(nid, []):
                if nb in seen:
                    continue
                seen.add(nb)
                out.add(nb)
                q.append((nb, d + 1))
        return out


    def merge_false_branch_points(self, df: pd.DataFrame, min_bp_distance: int) -> pd.DataFrame:
        """
        Merge nearby branch points within a hop-distance threshold.

        Use:
            Identify branch points as nodes with degree > 2 in the undirected SWC graph.
            For each branch point in ascending ID order, find other branch points within
            ≤ min_bp_distance hops and merge them into the current node by rewiring
            Parent references. Nodes are not deleted; only Parent IDs are redirected.

        Args:
            df (pd.DataFrame): SWC DataFrame with at least 'ID' and 'Parent' columns.
            min_bp_distance (int): Maximum hop distance for considering two branch points
                close enough to merge (nodes).

        Returns:
            pd.DataFrame: DataFrame with Parent IDs updated to reflect merges.
        """
        # Build an undirected adjacency mapping for hop-distance queries
        adj = self.build_adjacency(df)

        # Collect branch points as nodes with degree > 2, then process in ascending ID order
        branch_points = sorted(list({node for node, neighbors in adj.items() if len(neighbors) > 2}))
        branch_point_set = set(branch_points)

        merged: Set[int] = set() # Track nodes that have been merged
        merge_map: Dict[int, int] = {} # Map nodes to their merged counterparts

        # For each branch point, merge other branch points found within the hop threshold into it
        for bp in branch_points:
            if bp in merged:
                continue # Skip already merged nodes

            # Identify branch points within the defined node distance
            close_nodes = self.bfs_distance(adj, bp, min_bp_distance) & branch_point_set

            # Merge close branch points into a single representative node
            for close_node in sorted(close_nodes):
                if close_node == bp or close_node in merged:
                    continue
                merge_map[close_node] = bp
                merged.add(close_node)

        # Update the parent references in the DataFrame to reflect the merging process
        for old_node, new_node in merge_map.items():
            df.loc[df['Parent'] == old_node, 'Parent'] = new_node

        return df


    def remove_short_branches(self, df: pd.DataFrame, min_branch_length: int) -> pd.DataFrame:
        """
        Remove short terminal branches based on node count.

        Use:
            Identify terminal nodes (tips) and trace each tip backward through Parent
            pointers until reaching either a branch point or the root. If the trace
            reaches a branch point in fewer than `min_branch_length` steps, drop the
            traced nodes as a short terminal twig. After pruning, reindex IDs to 1..N
            and update Parent references accordingly.

        Args:
            df (pd.DataFrame): SWC DataFrame with at least 'ID' and 'Parent' columns.
            min_branch_length (int): Terminal twig length threshold in node count. Terminal
                paths shorter than this are removed when they attach to a branch point.

        Returns:
            pd.DataFrame: Pruned SWC DataFrame with IDs remapped to 1..N and Parent pointers
                updated (missing parents become -1).
        """
        # Build an undirected adjacency list to detect branch points by degree
        adjacency_list = self.build_adjacency(df)
        branch_points = {node for node, neighbors in adjacency_list.items() if len(neighbors) > 2}

        # Terminal nodes are IDs that never appear as a Parent
        terminal_points = set(df["ID"]) - set(df["Parent"])

        # Build a fast ID → Parent lookup for repeated traceback steps
        parent_of = df.set_index("ID")["Parent"].to_dict()

        # Walk from a terminal node toward the root, collecting nodes along the way
        nodes_to_remove: set[int] = set()
        def trace_back(node: int, path: list[int]) -> bool:
            length = 0
            while node not in branch_points and length < min_branch_length:
                path.append(node)
                parent = int(parent_of[node])
                if parent == -1:
                    break
                node = parent
                length += 1

            # Mark for removal only if a branch point was reached within the threshold
            return (length < min_branch_length) and (node in branch_points)

        # Collect all nodes belonging to short terminal twigs
        for terminal in terminal_points:
            path: list[int] = []
            if trace_back(terminal, path):
                nodes_to_remove.update(path)

        # Drop all marked nodes in one pass
        df = df[~df["ID"].isin(nodes_to_remove)].copy()

        # Remap IDs to 1..N and update Parent references to match the new IDs
        id_map = {old_id: new_id for new_id, old_id in enumerate(df["ID"], start=1)}
        df["ID"] = df["ID"].map(id_map)
        df["Parent"] = df["Parent"].map(id_map).fillna(-1).astype(int)

        return df


    @staticmethod
    def remove_extraneous_nodes(
        df: pd.DataFrame,
        image_shape: Tuple[int, int],
        voxel_um: float,
    ) -> pd.DataFrame:
        """
        Remove nodes outside the image bounds and all of their descendants.

        Use:
            Convert image bounds from pixels to micrometers using `voxel_um`
            and drop nodes whose X,Y coordinates fall outside these limits.
            All descendants of those nodes are also removed.

        Args:
            df (pd.DataFrame): SWC DataFrame with 'ID', 'Parent', 'X', 'Y'.
            image_shape (Tuple[int, int]): Image size (height_px, width_px).
            voxel_um (float): Voxel size in µm per pixel.

        Returns:
            pd.DataFrame: Updated DataFrame with extraneous nodes and their
            descendants removed.
        """
        # Extract valid X and Y ranges based on the image shape
        x_max = image_shape[1] * voxel_um
        y_max = image_shape[0] * voxel_um

        # Identify nodes with X, Y coordinates outside the valid range
        outside_nodes = df[
            (df["X"] < 0) | (df["X"] >= x_max) |
            (df["Y"] < 0) | (df["Y"] >= y_max)
        ]["ID"].tolist()

        if not outside_nodes:
            return df

        # Initialize a set to collect all nodes to be removed
        nodes_to_remove = set(outside_nodes)

        # Build a parent -> children lookup to traverse descendants efficiently
        children: DefaultDict[object, List[object]] = defaultdict(list)
        for parent_id, node_id in df[["Parent", "ID"]].itertuples(index=False, name=None):
            if parent_id != -1:
                children[parent_id].append(node_id)

        # Iteratively collect all descendants of outside nodes
        stack = list(outside_nodes)
        while stack:
            node_id = stack.pop()
            for child_id in children.get(node_id, []):
                if child_id not in nodes_to_remove:
                    nodes_to_remove.add(child_id)
                    stack.append(child_id)

        # Remove all nodes in nodes_to_remove from the DataFrame
        updated_df = df[~df["ID"].isin(nodes_to_remove)].copy()

        return updated_df


    def enforce_segment_length(self, df: pd.DataFrame, min_segment_length: float) -> pd.DataFrame:
        """
        Remove nodes where the segment length between adjacent nodes is less than
        `min_segment_length` (in µm), ensuring newly created segments also meet the
        length criterion.

        Use:
            Build an undirected adjacency graph from the SWC structure, compute edge
            lengths, and iteratively eliminate short edges by deleting a node and
            rewiring its neighbors. Preserve roots, branch points, and terminal
            nodes whenever possible, and continue processing if rewiring creates
            new short edges.

        Args:
            df (pd.DataFrame): SWC DataFrame with 'ID', 'Parent', 'X', 'Y'.
            min_segment_length (float): Minimum allowed edge length in µm.

        Returns:
            pd.DataFrame: Updated DataFrame with short segments removed and parent
            relationships updated.
        """

        # Cache node coordinates and parent pointers for constant-time lookup
        node_coords = df.set_index('ID')[['X', 'Y']].to_dict('index')

        # Compute Euclidean distance between two nodes in the XY plane
        def calculate_segment_length(node1, node2):
            coords1 = np.array([node_coords[node1]['X'], node_coords[node1]['Y']])
            coords2 = np.array([node_coords[node2]['X'], node_coords[node2]['Y']])
            return np.sqrt(np.sum((coords1 - coords2) ** 2))

        def process_short_segment(node1, node2):
            """
            Recursively resolve a short edge by deleting one endpoint and rewiring
            its remaining neighbor to the other endpoint.
            """
            if node1 in nodes_to_delete or node2 in nodes_to_delete:
                return

            def is_special(node):
                return node in root_nodes or node in branch_points or node in terminal_points

            def handle_deletion(node_to_delete, keep_node):
                # Mark node for removal and identify the remaining neighbor to splice through
                nodes_to_delete.add(node_to_delete)
                new_neighbor = [n for n in adjacency_list[node_to_delete] if n != keep_node][0]

                # Drop the two edges incident to the deleted node from the length cache
                segment_lengths.pop((min(node_to_delete, new_neighbor), max(node_to_delete, new_neighbor)), None)
                segment_lengths.pop((min(node_to_delete, keep_node), max(node_to_delete, keep_node)), None)

                # Add the new spliced edge and its length
                new_segment = (min(keep_node, new_neighbor), max(keep_node, new_neighbor))
                segment_lengths[new_segment] = calculate_segment_length(keep_node, new_neighbor)

                # Rewire adjacency to bypass the deleted node
                adjacency_list[new_neighbor].remove(node_to_delete)
                adjacency_list[new_neighbor].append(keep_node)
                adjacency_list[keep_node].append(new_neighbor)
                adjacency_list[keep_node].remove(node_to_delete)
                del adjacency_list[node_to_delete]

                return keep_node, new_neighbor, segment_lengths[new_segment]

            def decide_deletion_by_length():
                # Prefer deleting the endpoint whose alternative adjacent edge is shorter
                neighbor1 = [n for n in adjacency_list[node1] if n != node2][0]
                neighbor2 = [n for n in adjacency_list[node2] if n != node1][0]
                length1 = segment_lengths[(min(node1, neighbor1), max(node1, neighbor1))]
                length2 = segment_lengths[(min(node2, neighbor2), max(node2, neighbor2))]
                if length1 <= length2:
                    return handle_deletion(node1, node2)
                else:
                    return handle_deletion(node2, node1)

            # Preserve connectivity when both endpoints are structurally important
            if is_special(node1) and is_special(node2):
                return
            elif is_special(node1):
                keep_node, new_neighbor, new_length = handle_deletion(node2, node1)
            elif is_special(node2):
                keep_node, new_neighbor, new_length = handle_deletion(node1, node2)
            else:
                keep_node, new_neighbor, new_length = decide_deletion_by_length()

            # Continue collapsing if the newly formed edge is still too short
            if new_length < min_segment_length:
                process_short_segment(keep_node, new_neighbor)

        # Build an undirected adjacency list from the SWC parent pointers
        adjacency_list = self.build_adjacency(df)

        # Identify structurally important nodes from the current topology
        branch_points = {node for node, neighbors in adjacency_list.items() if len(neighbors) > 2}
        terminal_points = set(df['ID']) - set(df['Parent'])
        root_nodes = set(df[df['Parent'] == -1]['ID'])

        # Cache all edge lengths to enable shortest-first processing
        segment_lengths = {
            (min(node, neighbor), max(node, neighbor)): calculate_segment_length(node, neighbor)
            for node, neighbors in adjacency_list.items()
            for neighbor in neighbors if node < neighbor
        }

        # Sweep edges from shortest to longest and resolve those below the threshold
        nodes_to_delete = set()
        for (node1, node2), length in sorted(segment_lengths.items(), key=lambda x: x[1]):
            if length >= min_segment_length:
                break
            process_short_segment(node1, node2)

        # Redirect surviving downstream nodes past deleted nodes, then drop deleted nodes
        for node_to_delete in nodes_to_delete:
            parent_of_deleted = df.loc[df['ID'] == node_to_delete, 'Parent'].values[0]
            df.loc[df['Parent'] == node_to_delete, 'Parent'] = parent_of_deleted
        df = df[~df['ID'].isin(nodes_to_delete)].copy()

        return df


    @staticmethod
    def smooth_radii_with_regression(
        df: pd.DataFrame,
        filename: str,
        window_length: int,
        interval: int,
        min_radius: float,
    ) -> pd.DataFrame:
        """
        Smooth radii along neurite paths via sliding-window linear regression.

        Use:
            For each path from soma to terminal, build sliding windows of
            length `window_length` with step `interval`, fit a simple linear
            model per window, and use fitted values to smooth the radius at
            each node. Nodes appearing in multiple windows receive the mean
            of their window-wise fitted radii. Values are clipped at
            `min_radius`.

        Args:
            df (pd.DataFrame): SWC DataFrame with 'Radius', 'ID', 'Parent'.
            filename (str): Filename used in error messages for context.
            window_length (int): Sliding window size in node count.
            interval (int): Step between window starts along the path.
            min_radius (float): Minimum allowed radius after smoothing.

        Returns:
            pd.DataFrame: DataFrame with 'Radius' updated with smoothed values.

        Raises:
            ValidationError: If window_length is 1 and regression is undefined.
            ValidationError: If all original radii are zero and smoothing
                cannot be meaningfully performed.
        """
        # Validate window length for regression-based smoothing
        if int(window_length) == 1:
            raise ValidationError(
                f"window_length must be >= 2 in {filename}; cannot fit regression with window_length=1."
            )

        # Extract radii, node IDs, and parent IDs into arrays for fast lookup
        original_radii = df["Radius"].values
        original_ids = df["ID"].values
        parent_ids = df["Parent"].values

        # Identify terminal nodes as those that never appear as a Parent
        terminal_points = set(original_ids) - set(parent_ids)

        # Validate that radii contain nonzero values for meaningful smoothing
        if np.all(original_radii == 0):
            raise ValidationError(f"All original radii are 0 in {filename}; cannot compute padding/fit.")

        # Build an ID -> index lookup for O(1) node access along traced paths
        id_to_index = {int(node_id): i for i, node_id in enumerate(original_ids)}

        # Trace each terminal node back to the root to form root-to-terminal paths
        def trace_back(node):
            path = []
            while node != -1:
                idx = id_to_index[int(node)]
                path.append((node, original_radii[idx]))
                node = parent_ids[idx]
            return path[::-1]

        # Build root-to-terminal paths for all terminals
        paths = [trace_back(terminal) for terminal in terminal_points]

        # Create sliding windows of nodes and radii along each path
        path_data = []

        # Build windowed node/radius sequences for each root-to-terminal path
        for path in paths:
            path_nodes = [node for node, _ in path]
            path_radii = [radius for _, radius in path]

            # Compute the number of windows needed to cover the path with the given stride
            n_windows = max(1, (len(path_nodes) - window_length + interval - 1) // interval + 1)

            # Initialize per-path window containers
            node_windows = []
            radius_windows = []

            # Generate fixed-length windows (shifting the final window to end-aligned)
            for i in range(n_windows):
                start_idx = i * interval
                end_idx = start_idx + window_length

                # Adjust indices for the final window to keep window sizes consistent
                if end_idx > len(path_nodes):
                    end_idx = len(path_nodes)
                    start_idx = max(0, end_idx - window_length)

                node_windows.append(path_nodes[start_idx:end_idx])
                radius_windows.append(path_radii[start_idx:end_idx])

            # Store windows for later regression and aggregation
            path_data.append((node_windows, radius_windows))

        # Fit a line in each window and collect fitted values for each node across overlapping windows
        smoothed_radii_by_node = defaultdict(list)

        # Accumulate per-node fitted radii across all windows (averaged later)
        for node_windows, radius_windows in path_data:
            for nodes, radii in zip(node_windows, radius_windows):

                # Build the regression axis for node-count-based smoothing
                x = np.arange(len(radii), dtype=float)
                x_mean = np.mean(x)
                x_var = np.var(x)

                # Center radii values and compute the least-squares slope
                y = np.asarray(radii, dtype=float)
                y_mean = np.mean(y)
                cov_xy = np.mean((x - x_mean) * (y - y_mean))
                slope = cov_xy / x_var
                smoothed_radius = y_mean + slope * (x - x_mean)

                # Add fitted radii to each node's accumulator
                for node, radius in zip(nodes, smoothed_radius):
                    smoothed_radii_by_node[node].append(radius)

        # Average fitted radii for nodes that appear in multiple windows
        final_smoothed_radii = {node: np.mean(radii) for node, radii in smoothed_radii_by_node.items()}

        # Update radii and enforce a minimum radius
        df["Radius"] = (
            df["ID"]
            .map(final_smoothed_radii)
            .fillna(df["Radius"])
            .apply(lambda r: max(r, min_radius))
        )

        # Return the DataFrame with smoothed radii
        return df


    @staticmethod
    def calculate_branch_order(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute branch order for each node in an SWC tree.

        Use:
            Assign branch order based on tree hierarchy:
            - Root nodes (Parent == -1) have branch order 0.
            - Non-root nodes start at 1.
            - When a node has more than one child, branch order increments
                for the children and is propagated through their subtrees.

        Args:
            df (pd.DataFrame): SWC DataFrame with 'ID' and 'Parent' columns.

        Returns:
            pd.DataFrame: DataFrame with an added 'branch_order' column.
        """
        # Initialize the branch order column and a lookup for computed orders
        df["branch_order"] = 1
        branch_order_dict = {node: 1 for node in df["ID"]}

        # Build a parent -> children lookup for fast child access
        children_map = defaultdict(list)
        for parent_id, node_id in df[["Parent", "ID"]].itertuples(index=False, name=None):
            children_map[parent_id].append(node_id)

        def set_branch_order(node_id, current_order) -> None:
            # Fetch children without scanning the full DataFrame
            children = children_map.get(node_id, [])

            # Increase order at a divergence and pass it down the subtree
            if len(children) > 1:
                current_order += 1

            for child in children:
                branch_order_dict[child] = current_order
                set_branch_order(child, current_order)

        # Assign branch order starting from all roots
        root_nodes = df[df["Parent"] == -1]["ID"]
        for root in root_nodes:
            set_branch_order(root, 1)

        # Map computed branch orders back to the DataFrame
        df["branch_order"] = df["ID"].map(branch_order_dict)

        # Roots use 0 by definition
        df.loc[df["Parent"] == -1, "branch_order"] = 0

        return df


    def construct_neurite_segments(
        self,
        df: pd.DataFrame,
        image_shape: Tuple[int, int],
        voxel_um: float,
        filepath: str,
    ) -> dict:
        """
        Construct neurite segment geometries and pixel masks from an SWC tree.

        Use:
            Build an undirected adjacency graph from the SWC structure, then for
            each unique edge construct a 2D frustum polygon using the radii at the
            endpoints. Polygons are clipped to the image bounds, converted to pixel
            space, and rasterized to (rr, cc) coordinates for downstream intensity
            extraction and geometry metrics.

        Args:
            df (pd.DataFrame): SWC DataFrame with columns ['ID', 'X', 'Y', 'Radius', 'Parent'].
            image_shape (Tuple[int, int]): Image size as (height_px, width_px).
            voxel_um (float): Conversion factor from pixels to micrometers (µm per pixel).
            filepath (str): File context used in error messages.

        Returns:
            dict: Dictionary containing:
                - 'line_segments_um': neighbor_id -> ((x1_um, y1_um), (x2_um, y2_um))
                - 'cone_vertices_um': neighbor_id -> tuple of four (x_um, y_um) vertices
                - 'rr', 'cc': neighbor_id -> row/column indices of the rasterized polygon
                - 'mask_area_px': neighbor_id -> polygon area in pixel space
                - 'polygon_px': neighbor_id -> shapely Polygon in pixel space

        Raises:
            ValueError: If a segment has zero radius at both endpoints, or if a valid
                polygon cannot be formed from the frustum vertices.
        """
        # Build adjacency list for neurite connections
        adjacency_list = self.build_adjacency(df)

        # Extract relevant columns as numpy arrays for fast access
        ids = df["ID"].values
        x_values = df["X"].values
        y_values = df["Y"].values
        r_values = df["Radius"].values
        root_node = df[df["Parent"] == -1]["ID"].values[0]

        # Create lookup dictionaries for efficient access
        id_to_index = {id_val: idx for idx, id_val in enumerate(ids)}
        points = {id_val: (x_values[idx], y_values[idx]) for id_val, idx in id_to_index.items()}
        radii = {id_val: r_values[idx] for id_val, idx in id_to_index.items()}

        # Initialize perpendicular line point storage
        perp_line_points_x = {id_val: (None, None) for id_val, _ in id_to_index.items()}
        perp_line_points_y = {id_val: (None, None) for id_val, _ in id_to_index.items()}

        # Dictionary to store computed segment properties
        segments = {
            "line_segments_um": {},
            "cone_vertices_um": {},
            "rr": {},
            "cc": {},
            "mask_area_px": {},
            "polygon_px": {},
        }

        # Iterate over adjacency list to construct segment data
        for node, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                if node < neighbor:
                    # Fetch endpoint coordinates and store the µm line segment for this edge
                    point1, point2 = points[node], points[neighbor]
                    segments["line_segments_um"][neighbor] = (point1, point2)

                    # Unpack endpoints and radii, and reject degenerate segments with zero radius
                    x1, y1 = point1
                    x2, y2 = point2
                    r1, r2 = radii[node], radii[neighbor]
                    if not (r1 and r2):
                        raise ValueError(f"The line segment between node ID {node} and {neighbor} has no radius in {filepath}")

                    # Compute vector components
                    dx, dy = x2 - x1, y2 - y1
                    length_squared = dx * dx + dy * dy

                    # Avoid division by zero for overlapping points
                    length = np.sqrt(length_squared) if length_squared != 0 else np.finfo(float).eps

                    # Compute normalized perpendicular vector
                    perp_dx = -dy / length
                    perp_dy = dx / length

                    # Compute polygon (cone) vertices using radii
                    vertices_x = np.array([
                        x1 + perp_dx * r1, x1 - perp_dx * r1,
                        x2 - perp_dx * r2, x2 + perp_dx * r2,
                    ])
                    vertices_y = np.array([
                        y1 + perp_dy * r1, y1 - perp_dy * r1,
                        y2 - perp_dy * r2, y2 + perp_dy * r2,
                    ])

                    # Clip vertices to image boundaries
                    vertices_x = np.clip(vertices_x, 0.0, image_shape[1] * voxel_um - 1e-6)
                    vertices_y = np.clip(vertices_y, 0.0, image_shape[0] * voxel_um - 1e-6)

                    # Store perpendicular line segment endpoints
                    perp_line_points_x[neighbor] = (vertices_x[2:4])
                    perp_line_points_y[neighbor] = (vertices_y[2:4])

                    # Handle root node separately
                    if node == root_node:
                        perp_line_points_x[node] = (vertices_x[0:2])
                        perp_line_points_y[node] = (vertices_y[0:2])

                    # Concatenate to form full polygon vertices
                    cone_vertices_x_um = np.concatenate((perp_line_points_x[node], perp_line_points_x[neighbor]))
                    cone_vertices_y_um = np.concatenate((perp_line_points_y[node], perp_line_points_y[neighbor]))

                    # Convert concatenated vertices from µm to pixel coordinates
                    cone_vertices_um = list(zip(cone_vertices_x_um, cone_vertices_y_um))
                    cone_vertices_px = [(x / voxel_um, y / voxel_um) for x, y in cone_vertices_um]

                    # Label node/nbr vertices to preserve endpoint identity through permutation search
                    labeled_pts = [
                        ("node", 0, cone_vertices_px[0]),
                        ("node", 1, cone_vertices_px[1]),
                        ("nbr", 0, cone_vertices_px[2]),
                        ("nbr", 1, cone_vertices_px[3]),
                    ]

                    # Find best polygon arrangement
                    max_area = 0.0
                    best_perm, best_polygon = None, None

                    # Search all vertex orderings and keep the valid polygon with maximum area
                    for perm in itertools.permutations(labeled_pts, 4):
                        coords = [p[2] for p in perm]
                        poly = Polygon(coords)
                        if poly.is_valid and poly.area > max_area:
                            max_area, best_perm, best_polygon = poly.area, perm, poly

                    # Fail fast if no valid polygon can be formed from the frustum vertices
                    if best_polygon is None or not best_polygon.is_valid:
                        raise ValueError("No valid polygon could be formed from the points.")

                    # Store polygon data in a stable node/nbr order
                    node_pts = [p[2] for p in best_perm if p[0] == "node"]
                    nbr_pts = [p[2] for p in best_perm if p[0] == "nbr"]

                    # Reorder vertices as (node endpoints, neighbor endpoints) in pixel and µm space
                    ordered_px = node_pts + nbr_pts
                    ordered_um = tuple((x * voxel_um, y * voxel_um) for x, y in ordered_px)

                    # Persist per-segment polygon geometry and area for downstream metrics
                    segments["mask_area_px"][neighbor] = max_area
                    segments["cone_vertices_um"][neighbor] = ordered_um
                    segments["polygon_px"][neighbor] = best_polygon

                    # Convert polygon to pixel coordinates
                    rr, cc = polygon_to_pixels(best_polygon, image_shape)
                    segments["rr"][neighbor] = rr
                    segments["cc"][neighbor] = cc

        return segments


    def calculate_geometrics(
        self,
        df: pd.DataFrame,
        nid: int,
        voxel_size: float,
        image_signals: Optional[Dict[int, np.ndarray]],
        neurite_segments: Dict,
    ) -> Tuple[pd.DataFrame, Dict[str, float], Set[int], Set[int]]:
        """
        Compute branch- and neurite-level metrics and annotate per-node distances.

        Use:
            Traverse the neurite tree branch by branch and compute per-segment
            length, surface area, volume, and an electrotonic-length proxy. Optionally
            compute mean intensity per segment from provided signal images. Per-node
            cumulative distances from the soma are written back into `df`. Branch totals 
            are computed during traversal and written per-node. Neurite totals 
            are derived by summing per-segment maps.

        Args:
            df (pd.DataFrame): Neurite SWC DataFrame with at least
                ['ID', 'Parent', 'branch_order'] and metric columns to be filled
                (e.g. 'dist_from_soma_um', 'segment_length_um', etc.).
            nid (int): Locally unique neurite identifier.
            voxel_size (float): Voxel size in µm per pixel; used to convert mask area
                from pixels to µm².
            image_signals (Optional[Dict[int, np.ndarray]]): Mapping channel -> 2D
                signal image used for intensity extraction. If None or empty, no
                intensity values are computed (1-based keys).
            neurite_segments (Dict): Segment geometry bundle containing at least
                'cone_vertices_um', 'line_segments_um', 'polygon_px', 'rr', 'cc',
                and 'mask_area_px'.

        Returns:
            Tuple:
                - df (pd.DataFrame): DataFrame updated in place with per-node metrics.
                - neurite_metrics (Dict[str, float]): Aggregated neurite-level totals.
                - branch_points (Set[int]): Node IDs of branch points (degree > 2).
                - terminal_points (Set[int]): Node IDs of terminal tips.
        """

        def calculate_segment_metrics(
            end_node: int,
            current_dist: float,
            current_e_dist: float,
        ) -> Tuple[float, float, float, Dict[str, float], float, float, float]:
            """
            Compute metrics for a single segment ending at `end_node`.

            Use:
                Given a segment's geometric frustum (from `neurite_segments`)
                and optional signal images, compute segment length, surface,
                volume, mean intensity per channel, and update cumulative
                distances.

            Args:
                end_node (int): Node ID at the distal end of the segment.
                current_dist (float): Current distance from soma in µm.
                current_e_dist (float): Current electrotonic distance from soma
                    (arbitrary units).

            Returns:
                Tuple:
                    - segment_length_um (float)
                    - segment_surface_area_um (float)
                    - segment_volume_um (float)
                    - mean_intensity (Dict[Any, float])
                    - current_dist (float): updated distance from soma
                    - segment_e_length (float): electrotonic length for segment
                    - current_e_dist (float): updated electrotonic distance
            """
            # Compute frustum surface area and volume from the segment polygon vertices (µm space)
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = list(neurite_segments["cone_vertices_um"][end_node])
            segment_surface_area_um, segment_volume_um = approximate_frustum_convexhull(
                x1, y1, x2, y2, x3, y3, x4, y4
            )

            # Compute segment length from the line endpoints (µm space)
            (x1l, y1l), (x2l, y2l) = list(neurite_segments["line_segments_um"][end_node])
            segment_length_um = np.hypot(x2l - x1l, y2l - y1l)

            # Compute mean intensity per channel within the segment polygon mask (pixel space)
            mean_intensity: Dict[str, float] = {}
            if image_signals:
                # Cache weights for this exact segment across channels (stay local to this segment)
                seg_wts = None

                # Resolve the segment polygon and pixel mask once (constant across channels)
                poly = neurite_segments["polygon_px"][end_node]
                rr_seg = neurite_segments["rr"][end_node]
                cc_seg = neurite_segments["cc"][end_node]

                # Compute per-channel intensities while reusing the same per-pixel intersection weights
                for channel, ch_img in image_signals.items():
                    mean_intensity[channel], _, seg_wts = weighted_mean_intensity(
                        poly,
                        rr_seg,
                        cc_seg,
                        ch_img,
                        weights=seg_wts,
                    )

            # Compute an electrotonic-length proxy using segment length and a mean diameter estimate
            d_start = np.hypot(x2 - x1, y2 - y1)
            d_end = np.hypot(x4 - x3, y4 - y3)
            mean_d = np.nanmean([d_start, d_end])
            segment_e_length = segment_length_um / np.sqrt(mean_d)
            current_e_dist += segment_e_length

            # Update cumulative path length from the soma
            current_dist += segment_length_um

            return (
                segment_length_um,
                segment_surface_area_um,
                segment_volume_um,
                mean_intensity,
                current_dist,
                segment_e_length,
                current_e_dist,
            )

        def traverse_and_assign(
            start_node: int,
            direction_node: int,
            start_dist: float,
            start_e_dist: float,
        ) -> None:
            """
            Traverse from `start_node` toward `direction_node` along a branch.

            Use:
                Follow the tree until a branch point, tip, or root is
                encountered, accumulating segment metrics along the way and
                updating per-node metrics in `df`.

            Args:
                start_node (int): Node where this branch starts (e.g. root or
                    branch point).
                direction_node (int): First node to move to from start_node.
                start_dist (float): Initial distance from soma in µm.
                start_e_dist (float): Initial electrotonic distance from soma.

            Returns:
                None
            """
            # Accumulate branch totals while advancing node by node
            branch_e_length = 0
            branch_length = 0
            branch_surface_area = 0
            branch_volume = 0
            current_node = start_node
            current_dist = start_dist
            current_e_dist = start_e_dist
            first_iteration = True

            # Track nodes belonging to this branch
            branch_nodes: list[int] = []
            branch_puncta_count = 0

            while True:
                if current_node not in branch_points and not first_iteration:
                    visited.add(current_node)

                # Advance to the next node along the branch, excluding visited nodes and the branch start
                if first_iteration:
                    next_node = direction_node
                    first_iteration = False
                else:
                    neighbors = adjacency_list[current_node]
                    next_node = next(
                        neighbor
                        for neighbor in neighbors
                        if neighbor not in visited and neighbor != start_node
                    )

                # Compute segment metrics for the edge ending at the next node
                (
                    length,
                    surface_area,
                    volume,
                    intensity,
                    current_dist,
                    e_length,
                    current_e_dist,
                ) = calculate_segment_metrics(next_node, current_dist, current_e_dist)

                branch_e_length += e_length
                branch_length += length
                branch_surface_area += surface_area
                branch_volume += volume

                # Store intensity values per node when signals are present
                if image_signals:
                    for channel in image_signals:
                        signal_intensity_maps[channel][next_node] = intensity[channel]

                # Update per-node metric dicts for the newly visited node
                e_dist_from_soma[next_node] = current_e_dist
                dist_from_soma[next_node] = current_dist
                segment_e_length[next_node] = e_length
                segment_length[next_node] = length
                segment_surface[next_node] = surface_area
                segment_volume[next_node] = volume
                segment_mask_area[next_node] = neurite_segments["mask_area_px"][next_node] * voxel_size ** 2

                # Record membership of this node in the current branch and accumulate puncta if available
                branch_nodes.append(next_node)
                if puncta_by_id:
                    branch_puncta_count += puncta_by_id.get(next_node, 0)

                # Stop traversal at the root, a branch point, or a terminal tip
                current_node = next_node
                parent_id = parent_map[next_node]
                if parent_id == -1 or next_node in branch_points or next_node in terminal_points:
                    break

            # Stamp per-branch constants onto every node in this branch (local branch_uid is start_end).
            end_node = current_node
            branch_uid = f"{self.soma_id}_{nid}_{start_node}_{end_node}"
            for node_id in branch_nodes:
                branch_uid_map[node_id] = branch_uid
                branch_e_length_map[node_id] = branch_e_length
                branch_length_map[node_id] = branch_length
                branch_surface_area_map[node_id] = branch_surface_area
                branch_volume_map[node_id] = branch_volume
                branch_dist_from_soma_map[node_id] = start_dist
                branch_e_dist_from_soma_map[node_id] = start_e_dist
                if puncta_by_id:
                    branch_puncta_count_map[node_id] = branch_puncta_count

            return

        # Build adjacency and determine branch points and terminal tips
        adjacency_list = self.build_adjacency(df)
        branch_points = {node for node, neighbors in adjacency_list.items() if len(neighbors) > 2}
        terminal_points = set(df["ID"]) - set(df["Parent"])

        # Traversal bookkeeping
        visited = set()

        # Enable puncta branch totals when available
        puncta_enabled = bool(self.cfg.processing.extract_puncta) and ("segment_puncta_count" in df.columns)
        puncta_by_id = df.set_index("ID")["segment_puncta_count"].to_dict() if puncta_enabled else {}

        # Branch annotation maps
        branch_uid_map: dict[int, str] = {}
        branch_e_length_map: dict[int, float] = {}
        branch_length_map: dict[int, float] = {}
        branch_surface_area_map: dict[int, float] = {}
        branch_volume_map: dict[int, float] = {}
        branch_dist_from_soma_map: dict[int, float] = {}
        branch_e_dist_from_soma_map: dict[int, float] = {}
        branch_puncta_count_map: dict[int, int] = {}

        # Cache per-node maps once (mutated during traversal, written back once).
        id_index = df.set_index("ID", drop=False)
        parent_map = id_index["Parent"].to_dict()
        e_dist_from_soma = id_index.get("e_dist_from_soma", pd.Series(dtype=float)).to_dict()
        dist_from_soma = id_index.get("dist_from_soma_um", pd.Series(dtype=float)).to_dict()
        segment_e_length = id_index.get("segment_e_length", pd.Series(dtype=float)).to_dict()
        segment_length = id_index.get("segment_length_um", pd.Series(dtype=float)).to_dict()
        segment_surface = id_index.get("segment_surface_um", pd.Series(dtype=float)).to_dict()
        segment_volume = id_index.get("segment_volume_um", pd.Series(dtype=float)).to_dict()
        segment_mask_area = id_index.get("segment_mask_area_um", pd.Series(dtype=float)).to_dict()
        signal_intensity_maps = (
            {channel: id_index.get(f"signal_intensity_{channel}", pd.Series(dtype=float)).to_dict() for channel in image_signals}
            if image_signals
            else {}
        )

        # Use the soma/root node (Parent == -1) as the distance origin
        root_node = df[df["Parent"] == -1]["ID"].values[0]
        dist_from_soma[root_node] = 0.0
        e_dist_from_soma[root_node] = 0.0
        
        # Choose a deterministic set of branch entry edges to traverse
        starting_points = [(root_node, max(adjacency_list[root_node]))]
        if branch_points:
            for node in branch_points:
                for child in adjacency_list[node]:
                    if child > node:
                        starting_points.append((node, child))
        starting_points.sort()

        # Traverse from each start edge and record branch-level metrics
        for start_node, direction_node in starting_points:
            if direction_node not in visited:
                initial_dist = float(dist_from_soma.get(start_node, 0.0))
                initial_e_dist = float(e_dist_from_soma.get(start_node, 0.0))
                traverse_and_assign(start_node, direction_node, initial_dist, initial_e_dist)

        # Write per-node metrics/annotations back once (index-aligned by ID).
        df.set_index("ID", inplace=True, drop=False)
        if image_signals:
            for channel in image_signals:
                df[f"signal_intensity_{channel}"] = pd.Series(signal_intensity_maps[channel])
        if puncta_by_id:
            df["branch_puncta_count"] = pd.Series(branch_puncta_count_map)
        df["e_dist_from_soma"] = pd.Series(e_dist_from_soma)
        df["dist_from_soma_um"] = pd.Series(dist_from_soma)
        df["segment_e_length"] = pd.Series(segment_e_length)
        df["segment_length_um"] = pd.Series(segment_length)
        df["segment_surface_um"] = pd.Series(segment_surface)
        df["segment_volume_um"] = pd.Series(segment_volume)
        df["segment_mask_area_um"] = pd.Series(segment_mask_area)
        df["branch_uid"] = pd.Series(branch_uid_map)
        df["branch_e_length"] = pd.Series(branch_e_length_map)
        df["branch_length"] = pd.Series(branch_length_map)
        df["branch_surface_area"] = pd.Series(branch_surface_area_map)
        df["branch_volume"] = pd.Series(branch_volume_map)
        df["branch_dist_from_soma"] = pd.Series(branch_dist_from_soma_map)
        df["branch_e_dist_from_soma"] = pd.Series(branch_e_dist_from_soma_map)
        df.reset_index(drop=True, inplace=True)

        # Aggregate neurite totals by summing per-segment contributions
        neurite_metrics = {
            "e_length": float(np.nansum(list(segment_e_length.values()))),
            "length": float(np.nansum(list(segment_length.values()))),
            "surface_area": float(np.nansum(list(segment_surface.values()))),
            "volume": float(np.nansum(list(segment_volume.values()))),
        }

        return df, neurite_metrics, branch_points, terminal_points