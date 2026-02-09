# src/morphologic/config.py
from __future__ import annotations

# General imports (stdlib)
import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Local imports
from .exceptions import ConfigError, DataNotFound


@dataclass(frozen=True)
class Pathing:
    directory: Path = Path("/scratch2/msterling/NaV_staining")                   # Parent directory containing all data
    image_suffix: str = "_8bit"                                           # Image file suffix
    soma_roi_suffix: str = "_somas"                                              # File suffix for soma ROIs
    puncta_roi_suffix: str = "_czi_corrected_colocResults_0"                 # Optional: File suffix for puncta ROIs
    signal_channels: Tuple[int, ...] = (2, 4)                                    # Optional: 1-based channel indices of signal channels in image
    nuclear_roi_suffix: str = "_nuclei"                                          # Optional: File suffix for nuclear ROIs 


@dataclass(frozen=True)
class Processing:
    overwrite: bool = True                                                       # Recompute even if cell outputs already exist
    aggregate: bool = True                                                       # Run post-processing aggregation of data
    visualize: bool = True                                                       # Toggle per-cell figure generation
    extract_puncta: bool = False                                                 # Toggle morphology-aware puncta mapping
    extract_signal: bool = True                                                  # Toggle morphology-aware signal mapping
    deduct_nuclei: bool = True                                                   # Subtract nuclear signal from somatic signal


@dataclass(frozen=True)
class Parameters:
    # General
    recursion_limit: int = 5000                                                  # Python recursion limit
    #voxel_size: float = 0.0720635                                                # µm per pixel (µm)
    voxel_size: float = 0.227                                                # µm per pixel (µm)
    sholl_range: Tuple[float, float, float] = (0.0, 801.0, 10.0)                 # Sholl radii definition: [start, stop, step] (µm)
    puncta_max_distance_um: float = 3.0                                          # Ignore puncta farther than this from any soma/segment (µm)

    # Neurite radius smoothing
    smooth_radii: bool = True                                                    # Smooth radii along neurites with linear regression
    radii_smoothing_window_length: int = 60                                      # Window size for smoothing (nodes)
    radii_smoothing_interval: int = 5                                            # Stride between smoothing fits (nodes)
    radii_smoothing_min: float = 0.05                                            # Minimum node radius after smoothing (µm)

    # Quality control
    enforce_primaries_until: int = 1                                             # Force first # nodes from soma to be primary (nodes)
    min_bp_distance: int = 3                                                     # Enforce minimum node spacing between branch points (nodes)
    min_branch_length: int = 3                                                   # Drop small branches (nodes)
    min_segment_length: float = 0.2                                              # Enforce minimum segment length (µm)
    max_root_offset_um: float = 50                                               # Raise overly long first segments from soma (µm)


@dataclass(frozen=True)
class Display:
    big_font_size: int = 6                                                       # Big label font size
    medium_font_size: float = 4.5                                                # Medium caption/annotation font size
    small_font_size: Optional[float] = 3                                         # Extra-small (used by some views)
    tick_label_size: int = 6                                                     # Axis tick label font size
    background_color: Tuple[int, int, int] = (255, 255, 255)                       # Canvas background RGB (0–255)
    dpi: int = 300                                                               # Output resolution in dots-per-inch
    image_format: str = "png"                                                    # File format for saved figure


@dataclass(frozen=True)
class ScaleBar:
    length_um: int = 50                                                          # Scale bar length in µm
    color: Tuple[int, int, int] = (0, 0, 0)                                      # Scale bar color in RGB
    thickness: int = 2                                                           # Scale bar thickness in pixels
    location: str = "bottom_left"                                                # bottom_left | bottom_right | top_left | top_right


@dataclass(frozen=True)
class Legend:
    position_y: float = 0.1                                                      # Relative legend Y position in axes coords (0–1)
    position_x: float = 0.92                                                     # Relative legend X position in axes coords (0–1)
    height: float = 0.2                                                          # Relative legend height in axes coords (0–1)
    width: float = 0.01                                                          # Relative legend bar width in axes coords (0–1)


@dataclass(frozen=True)
class Reconstruction:
    enable: bool = True                                                          # Enable rendering reconstruction figure
    show_axes_and_title: bool = True                                             # Draw axes and a title
    show_scale_bar: bool = True                                                  # Draw a scale bar on the figure
    color_soma: Tuple[int, int, int] = (254, 57, 66)                             # RGB color for soma
    color_neurites: Tuple[int, int, int] = (0, 0, 0)                             # RGB color for neurites
    display: Display = Display()                                                 # General display settings
    scale_bar: ScaleBar = ScaleBar()                                             # Scale bar styling/placement
    legend: Legend = Legend()                                                    # Legend placement and geometry


@dataclass(frozen=True)
class Geometry:
    enable: bool = True                                                          # Enable rendering geometry figure
    show_axes_and_title: bool = True                                             # Draw axes and a title
    show_scale_bar: bool = True                                                  # Draw a scale bar on the figure
    show_cell_metrics: bool = True                                               # Print cell-level geometry metrics on plot
    display: Display = Display()                                                 # General display settings
    scale_bar: ScaleBar = ScaleBar()                                             # Scale bar styling/placement
    legend: Legend = Legend()                                                    # Legend placement and geometry


@dataclass(frozen=True)
class Sholl:
    enable: bool = True                                                          # Enable rendering sholl figure
    show_axes_and_title: bool = True                                             # Draw axes and a title
    display: Display = Display()                                                 # General display settings


@dataclass(frozen=True)
class Puncta:
    enable: bool = True                                                          # Enable rendering puncta overlay figure
    show_axes_and_title: bool = True                                             # Draw axes and a title
    show_scale_bar: bool = True                                                  # Draw a scale bar on the figure
    dot_radius_px: int = 5                                                       # Puncta dot radius (pixels)
    color_soma: Tuple[int, int, int] = (254, 57, 66)                             # RGB color for soma
    color_neurites: Tuple[int, int, int] = (0, 0, 0)                             # RGB color for neurites
    display: Display = Display()                                                 # General display settings
    scale_bar: ScaleBar = ScaleBar()                                             # Scale bar styling/placement
    legend: Legend = Legend()                                                    # Legend placement and geometry


@dataclass(frozen=True)
class Signal:
    enable: bool = True                                                          # Enable rendering signal figures
    show_axes_and_title: bool = True                                             # Draw axes and a title
    show_scale_bar: bool = True                                                  # Draw a scale bar on the figure
    channel_names: Tuple[str, ...] = ("SCN2A", "SCN8A")                          # Signal channel names (match General - Signal Channels)
    display: Display = Display()                                                 # General display settings
    scale_bar: ScaleBar = ScaleBar()                                             # Scale bar styling/placement
    legend: Legend = Legend()                                                    # Legend placement and geometry


@dataclass(frozen=True)
class Visualization:
    reconstruction: Reconstruction = Reconstruction()                            # Reconstruction view
    geometry: Geometry = Geometry()                                              # Geometry overlays
    signal: Signal = Signal()                                                    # Signal/intensity maps
    sholl: Sholl = Sholl()                                                       # Sholl analysis plots
    puncta: Puncta = Puncta()                                                    # Puncta overlay view   


@dataclass(frozen=True)
class Aggregate:
    # General
    independents: List[int] = field(default_factory=lambda: [1,2,3])             # Subfolder levels treated as independent variables (under pathing.directory)
    norm_independent: Optional[int] = 1                                       # (Optional: None) Index of independents to create normalized signal data for (e.g. batch)

    # Sholl analysis
    sholl: Dict[str, Any] = field(default_factory=lambda: {                      # Sholl dependents
        "per_radius_dependents": [                                               # Dependent variables computed per Sholl radius
            "intersections",                                                     # Number of intersections at each radius
            "segment_lengths",                                                   # Total segment length crossing each radius
            "branch_points",                                                     # Number of branch points encountered at each radius
            "terminal_points",                                                   # Number of terminal tips (endpoints) at each radius
        ],
        "per_subject_dependents": [                                                 # Dependent variables computed once per subject
            "auc",                                                               # Area under the curve across all radii
            "crit",                                                              # Critical radius (maximum value point)
            "r",                                                                 # Correlation coefficient for the radial fit
            "slope",                                                             # Slope of the linear fit for the radial distribution
            "total",                                                             # Sum of values across all radii
        ],
    })

    # Geometric analysis
    geometric: Dict[str, Any] = field(default_factory=lambda: {                  # Geometric dependents; augmented in _add_dependents
        "per_segment": {                                                         # Per-segment (node-to-node) aggregation settings
            "binning": {                                                         # Histogram bin definitions for stratified analyses
                "distance": [0, 800, 10],                                        # [start, stop, step] for (electrotonic) distance from soma (unitless/µm)
                "radius": [0, 2, 0.05],                                          # [start, stop, step] for dendritic radius (µm)
                "percent": [0, 1, 0.01],                                         # [start, stop, step] for fractional/percent metrics (0–1)
            },
            "dependents": [                                                      # Metrics aggregated per segment ()
                "Radius",                                                        # Segment radius (µm)
            ],
        },
        "per_branch": {                                                          # Per-branch aggregation settings
            "dependents": [                                                      # Metrics aggregated per branch (all segments combined)
                "branch_e_length",                                               # Electrotonic branch length (dendritic radius approximation)(unitless)
                "branch_length",                                                 # Branch length (µm) 
                "branch_surface_area",                                           # Branch 3D surface area (circular approximation) (µm²)
                "branch_volume",                                                 # Branch volume (µm³)
            ],
        },
        "per_neurite": {                                                         # Per-neurite aggregation settings
            "dependents": [                                                      # Metrics aggregated per neurite (all segments combined)
                "neurite_e_length",                                              # Electrotonic neurite length (dendritic radius approximation)(unitless)
                "neurite_length",                                                # Neurite length (µm) 
                "neurite_surface_area",                                          # Neurite 3D surface area (circular approximation) (µm²)
                "neurite_volume",                                                # Neurite volume (µm³)
            ],
        },
        "per_cell": {                                                            # Per-cell aggregation settings
            "dependents": [                                                      # Metrics aggregated once per cell (all neurites combined)
                "n_primaries",                                                   # Number of primary neurites emanating from the soma (count)
                "total_neurite_e_length",                                        # Total electrotonic neurite length (dendritic radius heuristic)(unitless)
                "total_neurite_length",                                          # Total neurite length (µm)
                "total_neurite_area_2d",                                         # Total 2D projected neurite area (µm²)
                "total_neurite_area_3d",                                         # Total 3D neurite surface area (µm²)
                "total_neurite_volume",                                          # Total neurite volume (µm³)
                "dendritic_tree_area",                                           # Total dendritic arbor area (convex hull) (µm²)
                "soma_max_diameter",                                             # Maximum soma diameter (µm)
                "soma_area_2d",                                                  # Soma area in 2D (µm²)
            ],
        },
    })

    def with_derived_dependents(self, signal_channels: Tuple[int, ...], puncta_roi_suffix: str) -> "Aggregate":
        """
        Return a copy of this Aggregate with signal- and puncta-dependent metric names injected.

        Use:
            Derive signal density metric names from `signal_channels` and append them to the
            geometric aggregation spec. If `puncta_roi_suffix` is non-empty, also append
            puncta count metrics.

            Args:
                signal_channels: 1-based image channel indices used to derive signal metrics.
                puncta_roi_suffix: Non-empty enables puncta-dependent metric names; empty disables.

            Returns:
                Aggregate: New Aggregate with updated `geometric` dependents reflecting the
                provided channel and puncta settings.
        """
        # Build channel-derived dependent names for per-segment and per-cell aggregation
        segment_signal = [f"signal_density_{ch}" for ch in signal_channels]
        soma_signal    = [f"soma_signal_density_{ch}" for ch in signal_channels]

        # Deep-copy the geometry spec so this method is purely functional (no mutation of self)
        geometric = copy.deepcopy(self.geometric)

        # Extend per-segment dependents with signal densities and optional puncta counts
        per_segment = geometric["per_segment"]["dependents"]
        geometric["per_segment"]["dependents"] = (
            per_segment
            + segment_signal
            + (["segment_puncta_count"] if puncta_roi_suffix else [])
        )

        # Extend per-branch dependents with optional puncta counts
        per_branch = geometric["per_branch"]["dependents"]
        geometric["per_branch"]["dependents"] = (
            per_branch
            + (["branch_puncta_count"] if puncta_roi_suffix else [])
        )

        # Extend per-neurite dependents with optional puncta counts
        per_neurite = geometric["per_neurite"]["dependents"]
        geometric["per_neurite"]["dependents"] = (
            per_neurite
            + (["neurite_puncta_count"] if puncta_roi_suffix else [])
        )

        # Extend per-cell dependents with soma signal densities and optional puncta summary counts
        per_cell = geometric["per_cell"]["dependents"]
        geometric["per_cell"]["dependents"] = (
            per_cell
            + soma_signal
            + (["soma_puncta_count", "total_neurite_puncta_count", "total_puncta_count"] if puncta_roi_suffix else [])
        )

        # Return a new Aggregate instance with updated geometry while preserving other fields
        return Aggregate(
            independents=self.independents,
            norm_independent=self.norm_independent,
            sholl=self.sholl,
            geometric=geometric,
        )


@dataclass(frozen=True)
class Config:
    pathing: Pathing = Pathing()                                                 # File/directory locations and naming conventions
    processing: Processing = Processing()                                        # High-level processing toggles and runtime behavior
    parameters: Parameters = Parameters()                                        # Quantitative analysis parameters (voxel size, thresholds, etc.)
    visualization: Visualization = Visualization()                               # Visualization settings for plots/figures
    aggregation: Aggregate = Aggregate()                                         # Post-processing aggregation configuration


def make_config() -> Config:
    """
    Build and validate a Config object for a morphology analysis run.

    Use:
        Construct a Config with defaults and validate that the configured data
        directory exists on disk.

    Returns:
        Config: Fully-initialized configuration with a validated cfg.pathing.directory.

    Raises:
        ConfigError: If cfg.pathing.directory is empty.
        DataNotFound: If cfg.pathing.directory does not exist or is not a directory.
        ConfigError: If signal or puncta extraction is enabled but its required Pathing field is empty.
        ConfigError: If nuclear deduction is enabled but pathing.nuclear_roi_suffix is empty.
    """
    # Instantiate configuration using defaults from the Config dataclass/module
    cfg = Config()

    # Validate that a directory is configured
    if not cfg.pathing.directory:
        raise ConfigError(
            "Config: 'pathing.directory' is empty. Set it in the Config defaults."
        )

    # Validate that the configured directory exists on disk
    if not Path(cfg.pathing.directory).is_dir():
        raise DataNotFound(f"directory not found: {cfg.pathing.directory}")

    # Normalize image suffix so ".tif" is always present
    img_suf = str(cfg.pathing.image_suffix).strip()
    if not img_suf.lower().endswith((".tif", ".tiff")):
        img_suf += ".tif"
    cfg = replace(cfg, pathing=replace(cfg.pathing, image_suffix=img_suf))

    # Validate toggle-dependent configuration for signals
    if cfg.processing.extract_signal and not cfg.pathing.signal_channels:
        raise ConfigError(
            "Config: 'processing.extract_signal' is True but 'pathing.signal_channels' is empty. "
            "Either set signal_channels (e.g. (1, 4)) or disable extract_signal."
        )

    # Validate visualization channel names when signal rendering is enabled
    if cfg.processing.extract_signal and cfg.pathing.signal_channels:
        names = tuple(cfg.visualization.signal.channel_names)
        chans = tuple(cfg.pathing.signal_channels)

        if len(names) != len(chans):
            raise ConfigError(
                "Config: 'visualization.signal.channel_names' must have the same length as "
                "'pathing.signal_channels'."
            )

        if any(not str(n).strip() for n in names):
            raise ConfigError(
                "Config: 'visualization.signal.channel_names' contains empty names. "
                "Provide a non-empty name for each entry in 'pathing.signal_channels'."
            )

    # Validate toggle-dependent configuration for puncta
    if cfg.processing.extract_puncta and not str(cfg.pathing.puncta_roi_suffix).strip():
        raise ConfigError(
            "Config: 'processing.extract_puncta' is True but 'pathing.puncta_roi_suffix' is empty. "
            "Either set puncta_roi_suffix (e.g. '_puncta') or disable extract_puncta."
        )

    # Normalize dependent toggles
    if not cfg.processing.extract_signal and cfg.processing.deduct_nuclei:
        cfg = replace(cfg, processing=replace(cfg.processing, deduct_nuclei=False))

    # Validate toggle-dependent configuration for nuclear deduction
    if cfg.processing.deduct_nuclei and not str(cfg.pathing.nuclear_roi_suffix).strip():
        raise ConfigError(
            "Config: 'processing.deduct_nuclei' is True but 'pathing.nuclear_roi_suffix' is empty. "
            "Either set nuclear_roi_suffix (e.g. '_nuclei') or disable deduct_nuclei."
        )

    # Normalize toggle-disabled pathing fields
    if not cfg.processing.extract_signal and cfg.pathing.signal_channels:
        cfg = replace(cfg, pathing=replace(cfg.pathing, signal_channels=()))

    if not cfg.processing.extract_puncta and str(cfg.pathing.puncta_roi_suffix).strip():
        cfg = replace(cfg, pathing=replace(cfg.pathing, puncta_roi_suffix=""))

    # Extend aggregation dependents for signals / puncta
    agg = cfg.aggregation.with_derived_dependents(
        signal_channels=cfg.pathing.signal_channels,
        puncta_roi_suffix=cfg.pathing.puncta_roi_suffix,
    )

    # Instantiate edited configuration with added aggregation dependents
    cfg = Config(
        pathing=cfg.pathing,
        processing=cfg.processing,
        parameters=cfg.parameters,
        visualization=cfg.visualization,
        aggregation=agg,
    )

    # Return the validated configuration
    return cfg