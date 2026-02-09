# src/morphologic/visualization.py
from __future__ import annotations

# General imports (stdlib)
from pathlib import Path
from typing import Any, Dict, List

# General imports (third-party)
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for CLI usage
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from shapely import area, box, intersection
from shapely.geometry import Polygon

# Local imports
from .config import Config
from .exceptions import VisualizationError
from .topology import polygon_to_pixels


def add_legend(
    ax: Any,
    vis_config: Any,
    mode: str,
    unique_colors: List[np.ndarray],
    soma_color: np.ndarray,
    width: int,
    height: int,
    voxel_size: float,
) -> None:
    """
    Draw the on-image legend for reconstruction, geometry, or signal views.

    Use:
        Places a compact legend in image coordinates (µm) using the configured
        relative legend placement and sizing. Called by the visualization
        routines after the main render.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw into.
        vis_config: Visualization section config for the current mode.
        mode (str): One of "reconstruction", "geometry", "signal", or "puncta".
        unique_colors (list[np.ndarray]): RGB colors used for neurite order bins (0–255).
        soma_color (np.ndarray): RGB color used for soma (0–255).
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        voxel_size (float): Microns per pixel used to convert pixel extents to µm.

    Returns:
        None
    """
    disp = vis_config.display

    # Convert image extents from pixels to microns so legend placement matches plot coordinates
    total_height = height * voxel_size
    total_width = width * voxel_size

    # Resolve legend box position and size from config fractions of the image extent
    legend_y = vis_config.legend.position_y * total_height
    legend_x = vis_config.legend.position_x * total_width
    legend_height = vis_config.legend.height * total_height
    bar_width = vis_config.legend.width * total_width

    # Scale label spacing relative to a reference image height used during tuning
    aspect_scale = max(1.0, total_width / total_height)
    text_offset = disp.medium_font_size * 1.5 * (total_height / (1104 * .227)) * ((aspect_scale + 1) / 2)

    if mode == "signal":
        # Render a vertical colormap bar in the same direction as the signal normalization
        gradient = np.linspace(0, 1, 256).reshape(256, 1)
        gradient_image = plt.cm.jet_r(gradient)[..., :3] * 255

        ax.imshow(
            gradient_image.astype(np.uint8),
            extent=[legend_x, legend_x + bar_width, legend_y + legend_height, legend_y],
        )

        # Label the low end and high end of the signal range
        ax.text(
            legend_x + bar_width / 2,
            legend_y + legend_height + text_offset,
            "0",
            va='bottom',
            ha='center',
            fontsize=disp.medium_font_size,
            color="black",
        )
        ax.text(
            legend_x + bar_width / 2,
            legend_y - text_offset,
            "max (I)",
            va='top',
            ha='center',
            fontsize=disp.medium_font_size,
            color="black",
        )

    elif mode == "geometry":
        # Stack one rectangle per Sholl/branch order color, including soma as order 0
        num_colors = len(unique_colors) + 1  # Unique colors + soma
        compartment_height = legend_height / num_colors

        # Header for the order legend
        ax.text(
            legend_x + bar_width / 2,
            legend_y - text_offset,
            "Order",
            va='top',
            ha='center',
            fontsize=disp.medium_font_size,
            fontweight="bold",
            color="black",
        )

        # Draw colored compartments and label their order index
        for i, color in enumerate([soma_color] + unique_colors):  # Soma first
            rect = Rectangle(
                (legend_x, legend_y + i * compartment_height),
                bar_width,
                compartment_height,
                facecolor=np.array(color) / 255,
                edgecolor='black',
            )
            ax.add_patch(rect)

            ax.text(
                legend_x + bar_width * 1.5,
                legend_y + (i + 0.5) * compartment_height,
                f"{i}",
                va='center',
                fontsize=disp.medium_font_size,
                color="black",
            )

    elif mode in ("reconstruction", "puncta"):
        # Simple two-bin legend showing neurites vs soma
        compartment_height = legend_height / 2
        soma_rect = Rectangle(
            (legend_x, legend_y),
            bar_width,
            compartment_height,
            facecolor=np.array(soma_color) / 255,
            edgecolor='black',
        )
        neurite_rect = Rectangle(
            (legend_x, legend_y + compartment_height),
            bar_width,
            compartment_height,
            facecolor='black',
            edgecolor='black',
        )

        ax.add_patch(neurite_rect)
        ax.add_patch(soma_rect)

        # Labels above/below the bar to match the rendered compartments
        ax.text(
            legend_x + bar_width / 2,
            legend_y + legend_height + text_offset,
            "Neurites",
            va='bottom',
            ha='center',
            fontsize=disp.medium_font_size,
            color="black",
        )
        ax.text(
            legend_x + bar_width / 2,
            legend_y - text_offset,
            "Soma",
            va='top',
            ha='center',
            fontsize=disp.medium_font_size,
            color="black",
        )


def visualize_neurite_segments(
    combined_image: np.ndarray,
    cell_data: Dict[str, Any],
    vis_config: Any,
    mode: str,
    colors: np.ndarray | None = None,
    signal_channel: int | None = None,
    max_signal: float | None = None,
) -> tuple[np.ndarray, List[np.ndarray]]:
    """
    Rasterize neurite segment masks into an RGB canvas.

    Use:
        Fills each precomputed neurite segment mask (rr/cc pixels) into
        `combined_image` using a mode-dependent color rule:
          - reconstruction: constant neurite color from config
          - geometry: branch-order color from the provided colormap
          - signal: jet colormap based on per-segment intensity normalized by max_signal

    Args:
        combined_image (np.ndarray): H×W×3 RGB canvas modified in place.
        cell_data (dict): Per-cell output dictionary containing:
            - "geometric_dataframes": Per-neurite DataFrames with node metrics
            - "neurite_segments": Per-neurite segment bundles with rr/cc masks
        vis_config: Visualization config for the active view.
        mode (str): One of "reconstruction", "geometry", "signal", or "puncta".
        colors (np.ndarray | None): Colormap lookup indexed by branch order (geometry mode).
        signal_channel (int | None): Channel index used for signal_intensity_{signal_channel} (signal mode).
        max_signal (float | None): Global max intensity used to normalize signal values (signal mode).

    Returns:
        tuple[np.ndarray, list[np.ndarray]]:
            - combined_image: The updated canvas with neurite segments painted.
            - unique_colors: Unique RGB colors actually used across all painted segments.
    """
    # Track unique colors that were actually painted so callers can build legends
    unique_colors: List[np.ndarray] = []

    # Each neurite has a dataframe with metrics and a parallel segment bundle with pixel masks
    for df, neurite in zip(cell_data["geometric_dataframes"], cell_data["neurite_segments"]):
        # Cache per-node metrics for O(1) lookup during segment painting
        order_by_id = dict(zip(df["ID"].to_numpy(), df["branch_order"].to_numpy()))
        density_by_id = (
            dict(zip(df["ID"].to_numpy(), df[f"signal_density_{signal_channel}"].to_numpy()))
            if mode == "signal"
            else {}
        )

        # Iterate over segment ids that have precomputed pixel coordinates
        for child_node_id, _ in neurite["line_segments_um"].items():
            # Choose a color for this segment based on the active visualization mode
            if mode == "geometry":
                order = order_by_id[child_node_id]
                final_colors = 255 * (1 - (1 - np.array(colors[order][:3])))
            elif mode == "signal":
                segment_intensity = density_by_id[child_node_id]
                fraction = segment_intensity / max_signal if max_signal else 0.0
                color_rgba = np.array(plt.cm.jet(fraction))
                final_colors = np.array(color_rgba[:3] * 255, dtype=int)
            else:
                final_colors = np.array(vis_config.color_neurites)

            # Paint the segment mask pixels directly into the RGB canvas
            cc = neurite["cc"][child_node_id]
            rr = neurite["rr"][child_node_id]
            combined_image[rr, cc, :] = final_colors

            # Record this color once so legends can reflect what was used
            if not any(np.array_equal(final_colors, existing_color) for existing_color in unique_colors):
                unique_colors.append(final_colors)

    return combined_image, unique_colors


def visualize_soma(
    combined_image: np.ndarray,
    cell_data: Dict[str, Any],
    vis_config: Any,
    mode: str,
    colors: np.ndarray | None = None,
    signal_channel: int | None = None,
    max_signal: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rasterize the soma ROI into an RGB canvas.

    Use:
        Fills the soma polygon into `combined_image` after neurites have been drawn.
        In geometry and reconstruction modes, fractional pixel coverage is used to
        anti-alias the soma boundary. In signal mode, the soma is filled with a
        uniform colormap-derived color from the soma intensity.

        In geometry mode, the soma is colored using the branch-order colormap (order 0).

    Args:
        combined_image (np.ndarray): H×W×3 RGB canvas modified in place.
        cell_data (dict): Per-cell dictionary containing:
            - "soma_roi": ROI tuple where index 1 is the ImageJ ROI object
            - "somatic_metrics": Soma metric dict including per-channel intensity values
        vis_config: Visualization config for the active view.
        mode (str): One of "reconstruction", "geometry", "signal", or "puncta".
        colors (np.ndarray | None): Colormap lookup indexed by branch order (geometry mode).
        signal_channel (int | None): Channel index used for signal_intensity_{signal_channel} (signal mode).
        max_signal (float | None): Global max intensity used to normalize signal values (signal mode).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - combined_image: Updated canvas with soma painted.
            - soma_color: RGB soma color used (0–255).
    """
    # Reuse cached soma rasterization across visualization modes for the same cell
    soma_cache = cell_data.setdefault("_soma_cache", {})

    if not soma_cache:
        # Extract the ROI object and choose subpixel vertices when available for better boundaries
        soma_roi = cell_data["soma_roi"][1]
        coordinates = (
            soma_roi.subpixel_coordinates.copy()
            if isinstance(soma_roi.subpixel_coordinates, np.ndarray)
            else soma_roi.integer_coordinates.copy()
        )

        # Convert integer ROI coordinates into image coordinates by applying the ROI bounding-box offset
        if not isinstance(soma_roi.subpixel_coordinates, np.ndarray):
            coordinates[:, 0] += soma_roi.left
            coordinates[:, 1] += soma_roi.top
        
        # Build a polygon from ROI vertices and rasterize it into pixel coordinates
        vertices_x, vertices_y = coordinates.T
        polygon = Polygon(zip(vertices_x, vertices_y)).buffer(0)
        rr, cc = polygon_to_pixels(polygon, combined_image.shape[:2])
        
        # Compute per-pixel intersection area for fractional coverage (anti-aliasing)
        boxes = box(cc, rr, cc + 1, rr + 1)
        intersection_areas = area(intersection(polygon, boxes)).astype(np.float64, copy=False)

        # Cache costly soma variables
        soma_cache["rr"] = rr
        soma_cache["cc"] = cc
        soma_cache["intersection_areas"] = intersection_areas

    else:
        # Reuse cached soma variables (computed once per cell)
        rr = soma_cache["rr"]
        cc = soma_cache["cc"]
        intersection_areas = soma_cache["intersection_areas"]

    # Choose soma color and paint into the canvas according to the active mode
    if mode == "geometry" and colors is not None:
        soma_color = 255 * (1 - (1 - np.array(colors[0][:3])))
        combined_image[rr, cc, :] = soma_color * intersection_areas[:, None]
    elif mode == "signal":
        soma_intensity = cell_data["somatic_metrics"][f"signal_density_{signal_channel}"]
        fraction = soma_intensity / max_signal if max_signal else 0.0
        color_rgba = np.array(plt.cm.jet(fraction))
        soma_color = np.array(color_rgba[:3] * 255, dtype=int)
        combined_image[rr, cc, :] = soma_color
    else:
        soma_color = np.array(vis_config.color_soma)
        combined_image[rr, cc, :] = soma_color * intersection_areas[:, None]

    return combined_image, soma_color


def visualize_puncta(
    combined_image: np.ndarray,
    cell_data: Dict[str, Any],
    vis_config: Any,
) -> np.ndarray:
    """
    Rasterize puncta into an RGB canvas.

    Use:
        Paint puncta as small filled circles centered at each punctum ROI.
        Soma-assigned puncta use the soma color with neurite-colored outline.
        Neurite-assigned puncta use the neurite color with soma-colored outline.

    Args:
        combined_image (np.ndarray): H×W×3 RGB canvas modified in place.
        cell_data (dict): Per-cell dictionary containing:
            - "soma_puncta_px": list[(x_px, y_px)]
            - "neurite_puncta_px": list[(x_px, y_px)]
        vis_config: Visualization config for puncta mode.

    Returns:
        np.ndarray: Updated canvas with puncta dots painted.
    """
    # Resolve dot radius and colors from config
    dot_r = int(vis_config.dot_radius_px)
    soma_fill = np.array(vis_config.color_soma, dtype=np.float32)
    neurite_fill = np.array(vis_config.color_neurites, dtype=np.float32)

    # Outline is the opposite compartment color
    soma_edge = neurite_fill
    neurite_edge = soma_fill

    h, w = combined_image.shape[:2]

    def paint_dot(cx_px: float, cy_px: float, fill: np.ndarray, edge: np.ndarray) -> None:
        # Convert centroid to integer pixel coordinates
        cx = int(round(cx_px))
        cy = int(round(cy_px))

        # Bound the dot drawing region to the image extent
        x0 = max(0, cx - dot_r - 1)
        x1 = min(w, cx + dot_r + 2)
        y0 = max(0, cy - dot_r - 1)
        y1 = min(h, cy + dot_r + 2)

        # Compute a distance mask in the dot's bounding box
        xs = np.arange(x0, x1, dtype=int)
        ys = np.arange(y0, y1, dtype=int)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2

        # Fill the interior circle
        fill_mask = d2 <= dot_r * dot_r
        combined_image[yy[fill_mask], xx[fill_mask], :] = fill

        # Draw a 2-pixel outline ring outside the filled circle
        r_out = dot_r + 2
        edge_mask = (d2 <= r_out * r_out) & (d2 > dot_r * dot_r)
        combined_image[yy[edge_mask], xx[edge_mask], :] = edge

    # Paint soma-assigned puncta
    for (x_px, y_px) in cell_data.get("soma_puncta_px", []):
        paint_dot(x_px, y_px, soma_fill, soma_edge)

    # Paint neurite-assigned puncta
    for (x_px, y_px) in cell_data.get("neurite_puncta_px", []):
        paint_dot(x_px, y_px, neurite_fill, neurite_edge)

    return combined_image


def visualize_cell(
    cell_data: Dict[str, Any],
    image_shape: np.ndarray,
    voxel_size: float,
    config: Config,
    save_file_path: str,
    mode: str,
    signal_channel: int | None = None,
) -> None:
    """
    Render and save a composite cell figure for one visualization mode.

    Use:
        Builds an RGB canvas, paints neurite segments and the soma ROI, overlays
        a legend and optional scale bar, and optionally annotates branch and
        cell-level metrics depending on the visualization config. The figure is
        saved to disk using `save_file_path` as a prefix.

    Args:
        cell_data (dict): Per-cell output dict containing neurite dataframes,
            soma metrics, branch metrics, and Sholl/geometry aggregates.
        image_shape (np.ndarray): 2D reference image shape in pixels (Y, X).
        voxel_size (float): Microns per pixel used to set plot extents and scale bar.
        config (Config): Global config object containing visualization settings.
        save_file_path (str): Output path prefix without mode-specific suffix.
        mode (str): One of "reconstruction", "geometry", "signal", or "puncta".
        signal_channel (int | None): Channel ID (1-based) used for signal mode rendering.

    Returns:
        None

    Raises:
        VisualizationError: If saving the figure fails.
    """
    # Select the visualization sub-config for the active mode and its scale bar settings
    vis_config = getattr(config.visualization, mode)
    disp = vis_config.display
    scale_bar_config = vis_config.scale_bar

    # Geometry mode uses a branch-order colormap sized to the max branch_order present
    colors = None
    if mode == "geometry":
        max_branch_order = max(df["branch_order"].max() for df in cell_data["geometric_dataframes"])
        colors = plt.cm.rainbow(np.linspace(0, 1, max_branch_order + 1))[::-1]

    # Signal mode normalizes intensities by the maximum signal_intensity_{signal_channel} across neurites and soma
    max_signal = 0.0
    if mode == "signal":
        ch_label = signal_channel
        for df in cell_data["geometric_dataframes"]:
            curr_max = df[f"signal_density_{signal_channel}"].max()
            if curr_max > max_signal:
                max_signal = curr_max
        curr_max = cell_data["somatic_metrics"][f"signal_density_{signal_channel}"]
        if curr_max > max_signal:
            max_signal = curr_max
    else:
        ch_label = ""

    # Initialize an RGB canvas using the configured background color
    combined_image = (
        np.ones((*image_shape, 3), dtype=np.float32) * np.array(disp.background_color)
    )

    # Paint neurite segments first so the soma can sit "on top"
    combined_image, unique_colors = visualize_neurite_segments(
        combined_image, cell_data, vis_config, mode, colors, signal_channel, max_signal
    )

    # Paint soma ROI and optionally return soma center for geometry label placement
    combined_image, soma_color = visualize_soma(
        combined_image, cell_data, vis_config, mode, colors, signal_channel, max_signal
    )

    # Paint puncta dots after soma/neurites so markers sit on top
    if mode == "puncta":
        combined_image = visualize_puncta(
            combined_image=combined_image,
            cell_data=cell_data,
            vis_config=vis_config
        )

    # Create the figure and display the composed image in micron coordinates
    fig, ax1 = plt.subplots(figsize=(6, 6))
    height, width = combined_image.shape[:2]
    aspect_scale = max(1.0, width / height)
    ax1.imshow(
        combined_image.astype(np.uint8),
        extent=[0, width * voxel_size, height * voxel_size, 0],
        origin="upper",
    )

    # Add a compact legend whose content depends on the mode and colors used
    add_legend(ax1, vis_config, mode, unique_colors, soma_color, width, height, voxel_size)

    # Optionally show axes/title, otherwise remove all frame and tick adornments
    if vis_config.show_axes_and_title:
        if mode == "reconstruction":
            title = "Neuronal Reconstruction"
        elif mode == "geometry":
            title = "Neuronal Geometry"
        elif mode == "puncta":
            title = "Neuronal Reconstruction with Puncta"
        elif mode == "signal":
            ch_name = config.visualization.signal.channel_names[config.pathing.signal_channels.index(ch_label)]
            title = f"Neuronal Reconstruction with Signal Density (Channel {ch_label}: {ch_name})"
        else:
            title = "Neuronal Reconstruction"

        ax1.set_title(title, fontsize=disp.big_font_size, fontweight="bold")
        ax1.set_xlabel("X (µm)", fontsize=disp.big_font_size)
        ax1.set_ylabel("Y (µm)", fontsize=disp.big_font_size)
        ax1.tick_params(axis="both", which="major", labelsize=disp.tick_label_size)
    else:
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_frame_on(False)

    # Draw a scale bar in the configured corner using pixel->micron conversion
    if vis_config.show_scale_bar:
        scale_length_px = scale_bar_config.length_um / voxel_size
        scale_thickness = scale_bar_config.thickness
        scale_color = np.array(scale_bar_config.color) / 255

        x_offset = width * 0.1
        y_offset = height * 0.1
        positions = {
            "bottom_right": (width - scale_length_px - x_offset, width - x_offset, height - y_offset),
            "bottom_left": (x_offset, x_offset + scale_length_px, height - y_offset),
            "top_left": (x_offset, x_offset + scale_length_px, y_offset),
            "top_right": (width - scale_length_px - x_offset, width - x_offset, y_offset),
        }

        scale_x_start, scale_x_end, scale_y = positions.get(
            scale_bar_config.location, positions["bottom_right"]
        )

        ax1.plot(
            [scale_x_start * voxel_size, scale_x_end * voxel_size],
            [scale_y * voxel_size, scale_y * voxel_size],
            color=scale_color,
            linewidth=scale_thickness,
        )

        ax1.text(
            (scale_x_start + scale_x_end) / 2 * voxel_size,
            (scale_y + 12 * (height / 1104) * ((1 + aspect_scale) / 2)) * voxel_size,
            f"{scale_bar_config.length_um} µm",
            fontsize=disp.medium_font_size,
            color=scale_color,
            ha="center",
            va="top",
        )

    # Geometry mode can also show a summary block of cell-level metrics in the top-right
    if mode == "geometry" and vis_config.show_cell_metrics:
        total_neurite_length = cell_data["geometric_analysis_cell"][0]["length"]
        total_neurite_surface_area_2D = sum(
            value for neurite in cell_data["neurite_segments"] for value in neurite["mask_area_px"].values()
        ) * voxel_size ** 2
        total_neurite_surface_area_3D = cell_data["geometric_analysis_cell"][0]["surface_area"]
        total_neurite_volume = cell_data["geometric_analysis_cell"][0]["volume"]
        tree_area_2D = cell_data["dendritic_tree_area"]
        max_soma_diameter = cell_data["somatic_metrics"]["max_diameter"]
        soma_surface_area_2D = cell_data["somatic_metrics"]["area_2d"]
        metrics_text = (
            f"Total neurite length: {round(total_neurite_length):,} um\n"
            f"Total neurite surface area (2D): {round(total_neurite_surface_area_2D):,} um²\n"
            f"Total neurite surface area (3D)*: {round(total_neurite_surface_area_3D):,} um²\n"
            f"Total neurite volume*: {round(total_neurite_volume):,} um³\n"
            f"Tree area (2D): {round(tree_area_2D):,} um²\n"
            f"Max soma diameter: {round(max_soma_diameter):,} um\n"
            f"Soma surface area (2D): {round(soma_surface_area_2D):,} um²\n"
            f"* Heuristic approximation"
        )
        text_x = width * 0.95
        text_y = height * 0.75
        ax1.text(
            text_x * voxel_size,
            text_y * voxel_size,
            metrics_text,
            fontsize=disp.small_font_size,
            color="black",
            weight="regular",
            va="top",
            ha="right",
        )

    # Build output filename, adding the channel suffix only for signal mode
    if mode == "signal":
        output_file = f"{save_file_path}_{mode}_ch{ch_label}.{disp.image_format}"
    else:
        output_file = f"{save_file_path}_{mode}.{disp.image_format}"

    # Save and always close the figure to avoid leaking matplotlib state
    try:
        plt.savefig(
            output_file,
            format=disp.image_format,
            dpi=disp.dpi,
            bbox_inches="tight",
        )
    except Exception as e:  # pragma: no cover - safety net
        plt.close(fig)
        raise VisualizationError(f"Failed to save visualization to {output_file}: {e}") from e
    else:
        plt.close(fig)


def visualize_sholl(
    cell_data: Dict[str, Any],
    config: Config,
    save_file_path: str,
) -> None:
    """
    Render and save a 2×2 grid of Sholl curves for one cell.

    Use:
        Plots the cell-level Sholl curves for:
          - intersections
          - segment lengths
          - branch points
          - terminal points
        and annotates each panel with summary stats (AUC, crit, R, slope, total)
        when those columns exist in the unified Sholl dataframe.

    Args:
        cell_data (dict): Per-cell data dict containing:
            - "sholl_analysis_cell": List with a single dict of per-radius metrics
            - "sholl_dataframes": List containing the long-form Sholl dataframe
              that includes cell_* summary columns
        config (Config): Global config object containing visualization settings.
        save_file_path (str): Output path prefix without the "_sholl" suffix.

    Returns:
        None

    Raises:
        VisualizationError: If the Sholl figure cannot be saved.
    """
    # Load the visualization config for Sholl formatting and output settings
    vis_config = config.visualization.sholl
    disp = vis_config.display

    def get_metric_val(sholl_metric, stat):
        # Read a single scalar stat from the first Sholl dataframe if present
        col_name = f"cell_{sholl_metric}_{stat}"
        if col_name in cell_data["sholl_dataframes"][0].columns:
            return cell_data["sholl_dataframes"][0][col_name].iloc[0]
        return None

    # Define the four Sholl metrics and their panel labels
    metrics = [
        ("intersections", "Intersections", "Count"),
        ("segment_lengths", "Segment Lengths", "Length (um)"),
        ("branch_points", "Branch Points", "Count"),
        ("terminal_points", "Terminal Points", "Count"),
    ]

    # Create a 2x2 figure layout with modest spacing between panels
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    axes_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Choose a max x-limit based on the largest radius with any non-zero value
    candidate_radii = []
    for metric_dict in cell_data["sholl_analysis_cell"][0].values():
        for r, val in metric_dict.items():
            if val != 0:
                candidate_radii.append(r)
    max_radius = max(candidate_radii) if candidate_radii else 0

    # Render one Sholl curve per panel and optionally annotate summary statistics
    for (metric_key, metric_title, metric_ylabel), (row, col) in zip(metrics, axes_positions):
        ax = fig.add_subplot(gs[row, col])

        # Pull the per-radius dict for this metric and plot radius vs value
        values_dict = cell_data["sholl_analysis_cell"][0].get(f"sholl_{metric_key}", {})
        x_vals = list(values_dict.keys())
        y_vals = list(values_dict.values())
        ax.plot(x_vals, y_vals, "b-o", markersize=3)

        # Apply consistent axis limits and labeling across panels
        ax.set_xlim(0, max_radius)
        ax.set_title(f"Sholl {metric_title}", fontsize=disp.big_font_size, fontweight="bold")
        ax.set_xlabel("Radius (um)", fontsize=disp.big_font_size)
        ax.set_ylabel(metric_ylabel, fontsize=disp.big_font_size)
        ax.tick_params(axis="both", which="major", labelsize=disp.big_font_size)

        # Ensure integer-only y-axis tick marks for count-based panels
        if metric_key in ("branch_points", "terminal_points"):
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Fetch summary stats from the unified Sholl dataframe when available
        auc_val = get_metric_val(metric_key, "auc")
        crit_val = get_metric_val(metric_key, "crit")
        r_val = get_metric_val(metric_key, "r")
        slope_val = get_metric_val(metric_key, "slope")
        total_val = get_metric_val(metric_key, "total")

        # Build a compact multi-line annotation only for stats that exist
        text_lines = []
        if auc_val is not None:
            text_lines.append(f"AUC: {auc_val:.2f}")
        if crit_val is not None:
            text_lines.append(f"Crit: {crit_val:.2f}")
        if r_val is not None:
            text_lines.append(f"R: {r_val:.2f}")
        if slope_val is not None:
            text_lines.append(f"Slope: {slope_val:.2e}")
        if total_val is not None:
            text_lines.append(f"Total: {total_val:.2f}")

        # Place the annotation in the top-right of the panel in data coordinates
        if text_lines:
            metrics_text = "\n".join(text_lines)
            x_lim = ax.get_xlim()[1]
            y_lim = ax.get_ylim()[1]
            ax.text(
                0.95 * x_lim,
                0.95 * y_lim,
                metrics_text,
                fontsize=disp.medium_font_size,
                color="black",
                weight="regular",
                va="top",
                ha="right",
            )

    # Save the figure and always close it to avoid leaking matplotlib state
    output_file = f"{save_file_path}_sholl.{disp.image_format}"
    try:
        plt.savefig(
            output_file,
            format=disp.image_format,
            dpi=disp.dpi,
            bbox_inches="tight",
        )
    except Exception as e:
        plt.close(fig)
        raise VisualizationError(f"Failed to save Sholl figure to {output_file}: {e}") from e
    else:
        plt.close(fig)


class Visualization:
    """
    Adapter around the visualization helpers.

    Use:
        Construct this with a `Config` instance and call the `save_*`
        methods to export reconstruction / geometry / signal / Sholl
        figures for each processed cell.
    """

    def __init__(self, cfg: Config) -> None:
        """
        Initialize the visualization adapter.

        Args:
            cfg (Config): Global configuration with visualization and
                processing sections.
        """
        self.cfg = cfg

    def save_reconstruction(self, cell_data: Dict[str, Any], out_base: Path) -> None:
        """
        Save a reconstruction overlay figure (soma + neurites).

        Args:
            cell_data (dict): Per-cell data dict from the pipeline.
            out_base (Path): Base output path (without suffix).

        Returns:
            None
        """
        voxel_size = self.cfg.parameters.voxel_size
        image_shape = cell_data["image_shape"]

        visualize_cell(
            cell_data=cell_data,
            image_shape=image_shape,
            voxel_size=voxel_size,
            config=self.cfg,
            save_file_path=str(out_base),
            mode="reconstruction",
        )

    def save_puncta(self, cell_data: Dict[str, Any], out_base: Path) -> None:
        """
        Save a puncta overlay figure (reconstruction + puncta dots).

        Args:
            cell_data (dict): Per-cell data dict from the pipeline.
            out_base (Path): Base output path (without suffix).

        Returns:
            None
        """
        voxel_size = self.cfg.parameters.voxel_size
        image_shape = cell_data["image_shape"]

        visualize_cell(
            cell_data=cell_data,
            image_shape=image_shape,
            voxel_size=voxel_size,
            config=self.cfg,
            save_file_path=str(out_base),
            mode="puncta",
        )

    def save_geometry(self, cell_data: Dict[str, Any], out_base: Path) -> None:
        """
        Save a geometry-colored overlay (branch order + metrics).

        Args:
            cell_data (dict): Per-cell data dict from the pipeline.
            out_base (Path): Base output path (without suffix).

        Returns:
            None
        """
        voxel_size = self.cfg.parameters.voxel_size
        image_shape = cell_data["image_shape"]

        visualize_cell(
            cell_data=cell_data,
            image_shape=image_shape,
            voxel_size=voxel_size,
            config=self.cfg,
            save_file_path=str(out_base),
            mode="geometry",
        )

    def save_signal(self, cell_data: Dict[str, Any], out_base: Path) -> None:
        """
        Save signal-density overlays for each configured channel.

        Args:
            cell_data (dict): Per-cell data dict from the pipeline.
            out_base (Path): Base output path (without suffix).

        Returns:
            None
        """
        voxel_size = self.cfg.parameters.voxel_size
        image_shape = cell_data["image_shape"]

        channels = self.cfg.pathing.signal_channels
        for ch in channels:
            visualize_cell(
                cell_data=cell_data,
                image_shape=image_shape,
                voxel_size=voxel_size,
                config=self.cfg,
                save_file_path=str(out_base),
                mode="signal",
                signal_channel=ch,
            )

    def save_sholl(self, cell_data: Dict[str, Any], out_base: Path) -> None:
        """
        Save a 2×2 Sholl summary figure for the cell.

        Args:
            cell_data (dict): Per-cell data dict from the pipeline.
            out_base (Path): Base output path (without suffix).

        Returns:
            None
        """
        visualize_sholl(
            cell_data=cell_data,
            config=self.cfg,
            save_file_path=str(out_base),
        )