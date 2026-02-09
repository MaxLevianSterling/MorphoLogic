CONFIG_HELP = {

    # Pathing
    "pathing.directory": "Parent directory containing all data",
    "pathing.image_suffix": "Image file suffix/extension",
    "pathing.soma_roi_suffix": "File suffix for soma ROIs",
    "pathing.puncta_roi_suffix": "Optional: File suffix for puncta ROIs",
    "pathing.signal_channels": "Optional: 1-based channel indices of signal channels in image",
    "pathing.nuclear_roi_suffix": "Optional: File suffix for nuclear ROIs",

    # Processing
    "processing.overwrite": "Recompute even if cell outputs already exist",
    "processing.aggregate": "Run post-processing aggregation of data",
    "processing.visualize": "Toggle per-cell figure generation",
    "processing.extract_puncta": "Toggle morphology-aware puncta mapping",
    "processing.extract_signal": "Toggle morphology-aware signal mapping",
    "processing.deduct_nuclei": "Subtract nuclear signal from somatic signal",

    # Parameters
    "parameters.recursion_limit": "Python recursion limit",
    "parameters.voxel_size": "µm per pixel (µm)",
    "parameters.puncta_max_distance_um": "Ignore puncta farther than this from any soma/segment (µm)",
    "parameters.smooth_radii": "Smooth radii along neurites with linear regression",
    "parameters.radii_smoothing_window_length": "Window size for smoothing (nodes)",
    "parameters.radii_smoothing_interval": "Stride between smoothing fits (nodes)",
    "parameters.radii_smoothing_min": "Minimum node radius after smoothing (µm)",
    "parameters.sholl_range": "Sholl radii definition: [start, stop, step] (µm)",
    "parameters.enforce_primaries_until": "Force first # nodes from soma to be primary (nodes)",
    "parameters.min_bp_distance": "Enforce minimum node spacing between branch points (nodes)",
    "parameters.min_branch_length": "Drop small branches (nodes)",
    "parameters.min_segment_length": "Enforce minimum segment length (µm)",
    "parameters.max_root_offset_um": "Raise overly long first segments from soma (µm)",

    # Display
    "visualization.display.big_font_size": "Big label font size",
    "visualization.display.medium_font_size": "Medium caption/annotation font size",
    "visualization.display.small_font_size": "Extra-small (used by some views)",
    "visualization.display.tick_label_size": "Axis tick label font size",
    "visualization.display.background_color": "Canvas background RGB (0–255)",
    "visualization.display.dpi": "Output resolution in dots-per-inch",
    "visualization.display.image_format": "File format for saved figure",

    # ScaleBar
    "visualization.scale_bar.length_um": "Scale bar length in µm",
    "visualization.scale_bar.color": "Scale bar color in RGB",
    "visualization.scale_bar.thickness": "Scale bar thickness in pixels",
    "visualization.scale_bar.location": "bottom_left | bottom_right | top_left | top_right",

    # Legend
    "visualization.legend.position_y": "Relative legend Y position in axes coords (0–1)",
    "visualization.legend.position_x": "Relative legend X position in axes coords (0–1)",
    "visualization.legend.height": "Relative legend height in axes coords (0–1)",
    "visualization.legend.width": "Relative legend bar width in axes coords (0–1)",

    # Reconstruction
    "visualization.reconstruction.enable": "Enable rendering reconstruction figure",
    "visualization.reconstruction.show_axes_and_title": "Draw axes and a title",
    "visualization.reconstruction.show_scale_bar": "Draw a scale bar on the figure",
    "visualization.reconstruction.color_soma": "RGB color for soma",
    "visualization.reconstruction.color_neurites": "RGB color for neurites",

    # Geometry
    "visualization.geometry.enable": "Enable rendering geometry figure",
    "visualization.geometry.show_axes_and_title": "Draw axes and a title",
    "visualization.geometry.show_scale_bar": "Draw a scale bar on the figure",
    "visualization.geometry.show_cell_metrics": "Print cell-level geometry metrics on plot",

    # Signal
    "visualization.signal.enable": "Enable rendering signal figures",
    "visualization.signal.show_axes_and_title": "Draw axes and a title",
    "visualization.signal.show_scale_bar": "Draw a scale bar on the figure",
    "visualization.signal.channel_names": "Signal channel names (match General - Signal Channels)",

    # Puncta
    "visualization.puncta.enable": "Enable rendering puncta overlay figure",
    "visualization.puncta.show_axes_and_title": "Draw axes and a title",
    "visualization.puncta.show_scale_bar": "Draw a scale bar on the figure",
    "visualization.puncta.dot_radius_px": "Puncta dot radius (pixels)",
    "visualization.puncta.color_soma": "RGB color for soma",
    "visualization.puncta.color_neurites": "RGB color for neurites",

    # Sholl
    "visualization.sholl.enable": "Enable rendering sholl figure",
    "visualization.sholl.show_axes_and_title": "Draw axes and a title",

    # Aggregation
    "aggregation.independents": "Subfolder levels treated as independent variables (under pathing.directory)",
    "aggregation.norm_independent": "(Optional: None) Index of independents to create normalized signal data for (e.g. batch)",
}
