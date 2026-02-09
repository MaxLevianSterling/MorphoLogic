# src/morphologic/gui.py
from __future__ import annotations

# General imports (stdlib)
import sys
import threading
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Third-party imports
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Local imports
from .config import (
    Config,
    Parameters,
    Processing,
    Pathing,
    Aggregate,
    Visualization as VizConfig,
    Display,
    ScaleBar,
    Legend,
)
from .core import MorphologyPipeline
from .exceptions import (
    ConfigError,
    DataNotFound,
    ImageIOError,
    MetricComputationError,
    MorphologicError,
    SWCParseError,
    ValidationError,
    VisualizationError,
)
from .tooltips import CONFIG_HELP


class ConfigGUI(tk.Tk):
    """
    Tkinter front-end for the Morphology pipeline.

    Use:
        Provide a GUI for editing a `Config` dataclass and running the
        `MorphologyPipeline` with progress reporting. The GUI surfaces all
        configuration fields across notebook tabs.

    Notes:
        - `self.field_vars` maps dotted config paths (e.g. "pathing.directory")
          to a tuple of (Tk variable, original default value). Tk variables
          store user input as strings/bools; parsing is performed when building
          the final `Config` object in `_build_config_from_form`.
    """

    def __init__(self) -> None:
        """
        Initialize the GUI, defaults, and widget state.

        Use:
            Create the main window, load default `Config` instances for initial
            widget values, initialize bookkeeping (worker thread, status, and
            field variable mappings), and build the full UI layout.
        """
        # Initialize the Tk root window and basic window metadata.
        super().__init__()
        self.title("MorphoLogic – Config & Run")

        # Apply global styling (theme, colors, fonts)
        self._init_styles()

        # Track the background worker thread to prevent concurrent runs and manage shutdown behavior
        self.worker_thread: Optional[threading.Thread] = None

        # Ensure window-close events go through our controlled close handler
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Default config snapshot used for widget defaults and type-guided parsing
        self._default_cfg = Config()
        self._default_pathing = self._default_cfg.pathing
        self._default_proc = self._default_cfg.processing
        self._default_params = self._default_cfg.parameters
        self._default_viz = self._default_cfg.visualization
        self._default_agg = self._default_cfg.aggregation

        # Hold references to top-level UI containers that are created during layout construction
        self.main_notebook: Optional[ttk.Notebook] = None
        self.visualization_tab: Optional[ttk.Frame] = None
        self.aggregation_tab: Optional[ttk.Frame] = None
        self._pathing_entries: Dict[str, Any] = {}
        self._processing_vars: Dict[str, tk.Variable] = {}
        self._processing_widgets: Dict[str, Any] = {}

        # Aggregation widgets that need to be enabled/disabled by processing toggles
        self._aggregation_widgets: Dict[str, Any] = {}

        # Widgets and tabs controlled by extract_signal / extract_puncta
        self._parameter_widgets: Dict[str, Any] = {}
        self._viz_nb: Optional[ttk.Notebook] = None
        self._viz_tabs: Dict[str, ttk.Frame] = {}

        # Store Tk variables and their original defaults for later reconstruction into typed dataclasses
        self.field_vars: Dict[str, Tuple[tk.Variable, Any]] = {}

        # Backing variable for the status label shown at the bottom of the window
        self.status_var = tk.StringVar(value="Idle")

        # Widget handles created during layout construction
        self.progress = None
        self.run_btn = None

        # Build the full UI (tabs, inputs, status/progress row)
        self._build_layout()


    def _init_styles(self) -> None:
        """
        Initialize ttk theme and shared styles.

        Use:
            Select a platform-appropriate ttk theme, define a consistent color
            palette, and configure a small set of widget styles used across tabs.
        """
        # Define a muted scientific palette (4 colors)
        self._ui = {
            "bg": "#F3F5F8",         # white-ish
            "panel": "#EEF1F6",      # light grey-blue
            "text": "#1F2937",       # dark grey
            "accent": "#4A607A",     # grey-ish blue
            "tooltip": "#4A607A",    # tooltip tone (grey-blue)
            "stripe": "#E4E8F0",     # faint stripe
        }

        # Select a ttk theme with platform fallbacks
        style = ttk.Style(self)
        themes = set(style.theme_names())
        for name in ("vista", "aqua", "clam", "alt", "default"):
            if name in themes:
                style.theme_use(name)
                break

        # Apply window background
        self.configure(bg=self._ui["bg"])

        # Base widget styling
        style.configure(".", background=self._ui["bg"], foreground=self._ui["text"])
        style.configure("TFrame", background=self._ui["panel"])
        style.configure("TLabel", background=self._ui["panel"], foreground=self._ui["text"])

        # Notebook tabs
        style.configure("TNotebook", background=self._ui["bg"])
        style.configure("TNotebook.Tab", padding=(10, 4))
        style.map(
            "TNotebook.Tab",
            foreground=[("selected", self._ui["text"]), ("!selected", self._ui["text"])],
        )

        # LabelFrame styling
        style.configure("TLabelframe", background=self._ui["panel"])
        style.configure("TLabelframe.Label", background=self._ui["panel"], foreground=self._ui["text"])

        # Action button styling for the bottom control row
        style.configure("Accent.TButton", padding=(10, 4))
        style.map(
            "Accent.TButton",
            foreground=[("disabled", "#9CA3AF"), ("!disabled", self._ui["text"])],
            background=[("active", self._ui["stripe"]), ("!disabled", self._ui["panel"])],
        )

        # Tooltips and stripes
        style.configure("Stripe0.TLabel", background=self._ui["panel"], foreground=self._ui["text"])
        style.configure("Stripe1.TLabel", background=self._ui["stripe"], foreground=self._ui["text"])
        style.configure("HelpStripe0.TLabel", background=self._ui["panel"], foreground=self._ui["tooltip"])
        style.configure("HelpStripe1.TLabel", background=self._ui["stripe"], foreground=self._ui["tooltip"])
        style.configure("Stripe0.TCheckbutton", background=self._ui["panel"], foreground=self._ui["text"])
        style.configure("Stripe1.TCheckbutton", background=self._ui["stripe"], foreground=self._ui["text"])

        # Stripe-matching frames (needed because Pathing rows use inner frames)
        style.configure("Stripe0.TFrame", background=self._ui["panel"])
        style.configure("Stripe1.TFrame", background=self._ui["stripe"])


    def _stripe_row(self, parent: ttk.Frame, row: int, col_span: int) -> None:
        """
        Render a faint background stripe behind a grid row.

        Use:
            Improve readability by alternating row backgrounds in simple forms.
        """
        # Create the stripe widget and place it behind row content
        style = "Stripe1.TLabel" if row % 2 else "Stripe0.TLabel"
        stripe = ttk.Label(parent, text="", style=style)
        stripe.grid(row=row, column=0, columnspan=col_span, sticky="nsew")
        parent.grid_rowconfigure(row, minsize=24)
        stripe.lower()


    def _build_layout(self) -> None:
        """
        Build the main window layout and top-level notebook tabs.

        Use:
            Construct the primary `ttk.Notebook`; build each tab’s
            controls; and create the bottom status/progress/button row.
        """
        # Configure the root grid so the notebook expands with the window
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Create the top-level notebook that hosts the main configuration tabs
        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.main_notebook = notebook

        # Create one frame per top-level tab and add them to the notebook
        general_frame = ttk.Frame(notebook)
        params_frame = ttk.Frame(notebook)
        viz_frame = ttk.Frame(notebook)
        agg_frame = ttk.Frame(notebook)

        notebook.add(general_frame, text="General")
        notebook.add(params_frame, text="Parameters")
        notebook.add(viz_frame, text="Visualization")
        self.visualization_tab = viz_frame
        notebook.add(agg_frame, text="Aggregation")
        self.aggregation_tab = agg_frame

        # Populate each tab with its widgets and register all field variables in `self.field_vars`
        self._build_general_tab(general_frame)
        self._build_parameters_tab(params_frame)
        self._build_visualization_tab(viz_frame)
        self._build_aggregation_tab(agg_frame)

        # Apply the initial enable/disable state of the Aggregation tab based on default Processing settings
        self._set_aggregation_tab_state(self._default_proc.aggregate)
        self._set_visualization_tab_state(self._default_proc.visualize)

        # Apply cross-tab enable/disable rules now that all widgets have been created
        self._update_feature_dependent_states()

        # Create the bottom row shared across all tabs (status, progress, run/quit controls)
        bottom = ttk.Frame(self)
        bottom.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        bottom.columnconfigure(0, weight=0)
        bottom.columnconfigure(1, weight=1)

        # Status label driven by `self.status_var`
        ttk.Label(bottom, textvariable=self.status_var, style="Help.TLabel").grid(
            row=0, column=0, sticky="w"
        )

        # Determinate progress bar updated by `_update_progress` during pipeline execution
        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=5)
        self.progress["value"] = 0

        # Run button starts the pipeline; disabled during a run
        self.run_btn = ttk.Button(
            bottom,
            text="Run",
            style="Accent.TButton",
            command=self._on_run_clicked,
        )
        self.run_btn.grid(row=0, column=2, padx=5)

        # Quit button triggers the close handler (prompts if a run is in progress)
        ttk.Button(bottom, text="Quit", style="Accent.TButton", command=self._on_close).grid(row=0, column=3, padx=5)


    def _set_visualization_tab_state(self, enabled: bool) -> None:
        """
        Enable or disable the Visualization tab in the main notebook.

        Use:
            Apply the Tk notebook tab state for the Visualization tab based on the
            Processing.visualize toggle. When disabling the tab while it is currently
            selected, the UI first switches to a safe fallback tab to avoid leaving
            the notebook on a disabled page.

        Args:
            enabled (bool): True to enable the Visualization tab, False to disable it.
        """
        # Bail out if the notebook or visualization tab has not been constructed yet
        if self.main_notebook is None or self.visualization_tab is None:
            return

        # Use the Visualization tab frame handle when applying per-tab state
        tab_id = self.visualization_tab

        # Enable the tab when requested
        if enabled:
            self.main_notebook.tab(tab_id, state="normal")
            return

        # Switch away if the Visualization tab is currently selected
        try:
            current = self.main_notebook.select()
            if current == str(tab_id):
                self.main_notebook.select(0)
        except Exception:
            pass

        # Disable the tab so it cannot be selected
        self.main_notebook.tab(tab_id, state="disabled")


    def _set_viz_subtab_state(self, name: str, enabled: bool) -> None:
        """
        Enable or disable a Visualization sub-tab in the nested notebook.

        Use:
            Apply per-tab state based on Processing feature toggles (signal/puncta).
            When disabling a currently-selected sub-tab, switch to the first tab.
        """
        # Bail out if the nested notebook or tab registry is not ready
        if self._viz_nb is None or name not in self._viz_tabs:
            return

        # Use the frame handle when applying per-tab state
        tab_id = self._viz_tabs[name]

        # Enable the tab when requested
        if enabled:
            self._viz_nb.tab(tab_id, state="normal")
            return

        # Switch away if this sub-tab is currently selected
        try:
            current = self._viz_nb.select()
            if current == str(tab_id):
                self._viz_nb.select(0)
        except Exception:
            pass

        # Disable the tab so it cannot be selected
        self._viz_nb.tab(tab_id, state="disabled")


    def _update_feature_dependent_states(self) -> None:
        """
        Apply enable/disable state to feature-dependent widgets across tabs.

        Use:
            Disable feature-only controls when the corresponding processing step is off.
            This keeps puncta distance inputs and the Puncta view inactive when puncta
            extraction is off, and keeps the Signal view inactive when signal extraction
            is off.
        """
        # Read controlling Processing toggles from cached variables
        var_extract_puncta = self._processing_vars.get("extract_puncta")
        var_extract_signal = self._processing_vars.get("extract_signal")

        extract_puncta = bool(var_extract_puncta.get()) if var_extract_puncta is not None else True
        extract_signal = bool(var_extract_signal.get()) if var_extract_signal is not None else True

        # Apply a unified enable/disable state across ttk and non-ttk widgets
        def set_state(widget: Any, enabled: bool) -> None:
            try:
                widget.state(["!disabled"] if enabled else ["disabled"])
            except Exception:
                try:
                    widget.configure(state="normal" if enabled else "disabled")
                except Exception:
                    pass

        # Gate puncta distance parameter on puncta extraction
        w = self._parameter_widgets.get("puncta_max_distance_um")
        if w is not None:
            set_state(w, extract_puncta)

        # Gate visualization subtabs on feature toggles
        self._set_viz_subtab_state("puncta", extract_puncta)
        self._set_viz_subtab_state("signal", extract_signal)


    def _update_progress(self, current: int, total: int, sec_per_cell: float) -> None:
        """
        Update the progress bar and status label from a pipeline callback.

        Use:
            Receive progress events from a worker thread and safely schedule UI
            updates on the Tk main thread using `self.after(0, ...)`.

        Args:
            current (int): Number of completed units (e.g., processed cells).
            total (int): Total number of units expected for the run.
            sec_per_cell (float): Measured seconds per unit, used for display.
                If <= 0, a simpler status message is shown.
        """

        # Apply the progress update on the Tk event loop thread
        def update():
            if self.progress is None or total <= 0:
                return

            # Configure the progress bar domain on first update and update the current value.
            self.progress["maximum"] = total
            self.progress["value"] = current

            percent = (current / total) * 100 if total else 0.0

            # Status string includes throughput when available.
            if sec_per_cell > 0:
                self.status_var.set(
                    f"Processing cells: {percent:.0f}% ({current}/{total}) – {sec_per_cell:.3f} s/cell"
                )
            else:
                self.status_var.set(
                    f"Processing cells: {percent:.0f}% ({current}/{total})"
                )

        # UI updates must run on the Tk thread; `after` safely schedules the closure.
        self.after(0, update)


    def _help_label(
        self,
        parent: ttk.Frame,
        row: int,
        key: str,
        help_col: int = 3,
        wraplength: int = 340,
    ) -> None:
        """
        Render permanent helper text for a config field in the GUI.

        Use:
            Look up descriptive help text from `CONFIG_HELP` using `key` and
            display it as a right-aligned, non-interactive label in the same
            grid row as the corresponding input widget.

        Args:
            parent (ttk.Frame):
                The container frame that owns the grid layout.
            row (int):
                The grid row index where the helper text should be placed,
                typically matching the row of the associated label/entry.
            key (str):
                Configuration path used to look up help text in `CONFIG_HELP`
                (e.g. `"processing.aggregate"` or `"parameters.voxel_size"`).
            help_col (int, optional):
                Grid column index where the helper label should be placed.
                Defaults to column 3, allowing a consistent right-hand help column
                across tabs.
            wraplength (int, optional):
                Pixel width used to wrap helper text.
        """
        # Pull the configured helper string for this field (empty string if missing)
        text = CONFIG_HELP.get(key, "")

        # Match the help-label background to the striped row it sits on
        help_style = "HelpStripe1.TLabel" if (row % 2) else "HelpStripe0.TLabel"

        # Render the helper text as a wrapped, left-justified label in the help column
        lbl = ttk.Label(
            parent,
            text=text,
            style=help_style,
            justify="left",
            wraplength=wraplength,
        )
        lbl.grid(row=row, column=help_col, sticky="w", padx=(10, 6), pady=3)
    

    def _update_general_field_states(self) -> None:
        """
        Apply enable/disable state to General-tab fields based on Processing toggles.

        Use:
            Grey out Pathing and Processing controls whose meaning is disabled by higher-level
            Processing booleans.
        """
        # Read the controlling Processing toggles from cached Tk variables
        var_extract_signal = self._processing_vars.get("extract_signal")
        var_extract_puncta = self._processing_vars.get("extract_puncta")
        var_deduct_nuclei = self._processing_vars.get("deduct_nuclei")

        # Snapshot toggle values with safe defaults for early/partial widget construction
        extract_signal = bool(var_extract_signal.get()) if var_extract_signal is not None else True
        extract_puncta = bool(var_extract_puncta.get()) if var_extract_puncta is not None else True
        deduct_nuclei = bool(var_deduct_nuclei.get()) if var_deduct_nuclei is not None else False

        # Nuclear deduction is only meaningful when signal extraction is enabled
        if not extract_signal and var_deduct_nuclei is not None and deduct_nuclei:
            var_deduct_nuclei.set(False)
            deduct_nuclei = False

        # Apply a unified enable/disable state across ttk and non-ttk widgets
        def set_state(widget: Any, enabled: bool) -> None:
            try:
                widget.state(["!disabled"] if enabled else ["disabled"])
            except Exception:
                try:
                    widget.configure(state="normal" if enabled else "disabled")
                except Exception:
                    pass

        # Signal channels are relevant only when signal extraction is enabled
        sig_entry = self._pathing_entries.get("pathing.signal_channels")
        if sig_entry is not None:
            set_state(sig_entry, extract_signal)

        # Puncta ROI suffix is relevant only when puncta extraction is enabled
        punc_entry = self._pathing_entries.get("pathing.puncta_roi_suffix")
        if punc_entry is not None:
            set_state(punc_entry, extract_puncta)

        # Nuclear ROI suffix is relevant only when signal extraction and nuclear deduction are enabled
        nuc_entry = self._pathing_entries.get("pathing.nuclear_roi_suffix")
        if nuc_entry is not None:
            set_state(nuc_entry, extract_signal and deduct_nuclei)

        # The deduct_nuclei toggle itself is only meaningful when signal extraction is enabled
        deduct_widget = self._processing_widgets.get("deduct_nuclei")
        if deduct_widget is not None:
            set_state(deduct_widget, extract_signal)


    def _update_aggregation_field_states(self) -> None:
        """
        Apply enable/disable state to Aggregation-tab fields based on Processing toggles.

        Use:
            Grey out Aggregation controls whose meaning is disabled by higher-level
            Processing booleans.
        """
        # Read the controlling Processing toggles from cached Tk variables
        var_extract_signal = self._processing_vars.get("extract_signal")
        extract_signal = bool(var_extract_signal.get()) if var_extract_signal is not None else True

        # Apply a unified enable/disable state across ttk and non-ttk widgets
        def set_state(widget: Any, enabled: bool) -> None:
            try:
                widget.state(["!disabled"] if enabled else ["disabled"])
            except Exception:
                try:
                    widget.configure(state="normal" if enabled else "disabled")
                except Exception:
                    pass

        # Norm independent is only meaningful when signal extraction is enabled
        norm_entry = self._aggregation_widgets.get("aggregation.norm_independent")
        if norm_entry is not None:
            set_state(norm_entry, extract_signal)


    def _build_general_tab(self, frame: ttk.Frame) -> None:
        """
        Build the General tab controls.

        Use:
            Render Pathing and Processing controls in a 4-column layout:
                [Processing toggles] [Processing help] [Pathing fields] [Pathing help]

            Processing toggles are bottom-aligned against the Pathing rows. The
            unused top-left area is used to render a MorphoLogic label.

        Args:
            frame (ttk.Frame): The notebook tab frame into which widgets are placed.
        """
        # Common grid padding for widgets in this tab
        padding = {"padx": 8, "pady": 3}

        # Configure a fixed grid: Processing (columns 0-1), Pathing (columns 2-3)
        frame.grid_columnconfigure(0, weight=0, minsize=170)
        frame.grid_columnconfigure(1, weight=1, minsize=260)
        frame.grid_columnconfigure(2, weight=0, minsize=360)
        frame.grid_columnconfigure(3, weight=1, minsize=340)

        # Pathing rows define the vertical rhythm for both sections
        total_rows = 7
        for r in range(total_rows):
            frame.grid_rowconfigure(r, minsize=28)
            self._stripe_row(frame, r, col_span=4)

        # Render the MorphoLogic label in the unused top-left area (row 0, columns 0-1)
        logo = tk.Canvas(
            frame,
            height=34,
            highlightthickness=0,
            borderwidth=0,
            bg=self._ui["panel"],
        )
        logo.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            padx=(padding["padx"], 0),
            pady=(padding["pady"], padding["pady"]),
        )

        # Define logo colors and font (two-tone with opposing outlines)
        c1 = self._ui["text"]
        c2 = self._ui["accent"]
        font = ("TkDefaultFont", 18, "bold")

        # Helper to draw outlined text by painting offsets around the main glyphs
        def outlined_text(x: int, y: int, text: str, fill: str, outline: str) -> int:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                logo.create_text(x + dx, y + dy, text=text, fill=outline, font=font, anchor="w")
            return logo.create_text(x, y, text=text, fill=fill, font=font, anchor="w")

        # Draw "Morpho" and "Logic" in opposing fill/outline colors
        t1 = outlined_text(2, 18, "Morpho", fill=c2, outline=c1)
        bbox = logo.bbox(t1) or (0, 0, 80, 20)
        outlined_text(bbox[2], 18, "Logic", fill=c1, outline=c2)

        # Configure Pathing field widths (entry width reduced vs previous layout)
        ENTRY_W_DIR = 25
        ENTRY_W_PATH = 39
        LABEL_W = 28

        # Helper to render one Pathing row inside a compact label+entry sub-frame
        def add_path_row(row: int, label: str, key: str, default_value: Any) -> None:
            # Field container placed in the Pathing field column
            field_style = "Stripe1.TFrame" if (row % 2) else "Stripe0.TFrame"
            field = ttk.Frame(frame, style=field_style)
            field.grid(
                row=row,
                column=2,
                sticky="w",
                padx=(padding["padx"], 0),
                pady=(padding["pady"], padding["pady"]),
            )

            # Field label
            lbl_style = "Stripe1.TLabel" if (row % 2) else "Stripe0.TLabel"
            ttk.Label(field, text=label, width=LABEL_W, style=lbl_style).grid(row=0, column=0, sticky="w")

            # Format sequence defaults as comma-separated text; scalars as plain strings
            if isinstance(default_value, (tuple, list)):
                text = ", ".join(str(v) for v in default_value)
            else:
                text = str(default_value)

            # Field entry
            var = tk.StringVar(value=text)
            field.grid_columnconfigure(1, weight=1)
            entry = ttk.Entry(field, textvariable=var, width=ENTRY_W_PATH)
            entry.grid(row=0, column=1, sticky="we")

            # Attach help text and register the Tk variable for later Config reconstruction
            self.field_vars[key] = (var, default_value)
            self._pathing_entries[key] = entry
            self._help_label(frame, row, key, help_col=3, wraplength=360)

        # Render the data directory selector (Entry + Browse button) in the Pathing field column
        dir_row = 1
        dir_field_style = "Stripe1.TFrame" if (dir_row % 2) else "Stripe0.TFrame"
        dir_field = ttk.Frame(frame, style=dir_field_style)
        dir_field.grid(
            row=dir_row,
            column=2,
            sticky="w",
            padx=(padding["padx"], 0),
            pady=(padding["pady"], padding["pady"]),
        )

        # Directory label + entry + browse button
        dir_lbl_style = "Stripe1.TLabel" if (dir_row % 2) else "Stripe0.TLabel"
        ttk.Label(dir_field, text="Data directory", width=LABEL_W, style=dir_lbl_style).grid(row=0, column=0, sticky="w")
        dir_var = tk.StringVar(value=str(self._default_pathing.directory))
        ttk.Entry(dir_field, textvariable=dir_var, width=ENTRY_W_DIR).grid(
            row=0, column=1, sticky="w", padx=(0, 6)
        )
        ttk.Button(dir_field, text="Browse…", command=lambda: self._browse_directory(dir_var)).grid(row=0, column=2, sticky="w")

        # Attach help text and register the Tk variable for later Config reconstruction
        self.field_vars["pathing.directory"] = (dir_var, self._default_pathing.directory)
        self._help_label(frame, dir_row, "pathing.directory", help_col=3, wraplength=360)

        # Render remaining Pathing fields in the configured order
        add_path_row(2, "Image suffix", "pathing.image_suffix", self._default_pathing.image_suffix)
        add_path_row(3, "Soma ROI suffix", "pathing.soma_roi_suffix", self._default_pathing.soma_roi_suffix)
        add_path_row(4, "Puncta ROI suffix", "pathing.puncta_roi_suffix", self._default_pathing.puncta_roi_suffix)
        add_path_row(5, "Signal extraction channels", "pathing.signal_channels", self._default_pathing.signal_channels)
        add_path_row(6, "Nuclear ROI suffix", "pathing.nuclear_roi_suffix", self._default_pathing.nuclear_roi_suffix)

        # Render Processing toggles in the left columns and bottom-align them to the Pathing rows
        ordered = ["overwrite", "aggregate", "visualize", "extract_puncta", "extract_signal", "deduct_nuclei"]
        start_row = total_rows - len(ordered)

        for i, field_name in enumerate(ordered):
            # Compute the target row so the toggle list aligns to the bottom of the Pathing section
            row = start_row + i
            value = getattr(self._default_proc, field_name)
            key = f"processing.{field_name}"

            # Toggle checkbox
            var = tk.BooleanVar(value=bool(value))
            cb_style = "Stripe1.TCheckbutton" if (row % 2) else "Stripe0.TCheckbutton"
            cb = ttk.Checkbutton(frame, text=_pretty_label(field_name), variable=var, style=cb_style)
            cb.grid(
                row=row,
                column=0,
                sticky="w",
                padx=(padding["padx"], 0),
                pady=(padding["pady"], padding["pady"]),
            )

            # Attach help text and register the Tk variable for later Config reconstruction
            self.field_vars[key] = (var, value)
            self._help_label(frame, row, key, help_col=1, wraplength=260)

            # Cache Processing variables and widgets for cross-field UI state control
            self._processing_vars[field_name] = var
            self._processing_widgets[field_name] = cb

            # Toggling aggregation enables or disables the Aggregation notebook tab
            if field_name == "aggregate":
                var.trace_add(
                    "write",
                    lambda *_: self._set_aggregation_tab_state(bool(self._processing_vars["aggregate"].get())),
                )

            # Toggling visualization enables or disables the Visualization notebook tab
            if field_name == "visualize":
                var.trace_add(
                    "write",
                    lambda *_: self._set_visualization_tab_state(bool(self._processing_vars["visualize"].get())),
                )

            # Toggling extraction/deduction gates Pathing fields and feature-dependent widgets across tabs
            if field_name in {"extract_signal", "extract_puncta", "deduct_nuclei"}:
                var.trace_add(
                    "write",
                    lambda *_: (
                        self._update_general_field_states(),
                        self._update_feature_dependent_states(),
                        self._update_aggregation_field_states(),
                    ),
                )

        # Apply the initial enable/disable state of toggle-dependent General-tab fields
        self._update_general_field_states()


    def _build_parameters_tab(self, frame: ttk.Frame) -> None:
        """
        Build the Parameters tab UI from the Parameters dataclass.

        Use:
            Render editable controls for every field in Parameters and register each
            Tk variable in self.field_vars so _build_config_from_form can rebuild a
            typed Parameters instance later.

            Parameters are presented in three groups:
            - General
            - Neurite radius smoothing
            - Quality control

        Args:
            frame (ttk.Frame): Parent notebook tab frame that receives the widgets.
        """
        # Common grid padding used by all widgets in this tab
        padding = {"padx": 8, "pady": 3}
        params = self._default_params

        # Configure the tab so its single column expands
        frame.columnconfigure(0, weight=1)

        # Allocate extra width for tooltips so they tend to fit on one line
        HELP_COL_MINSIZE = 560

        # Define parameter groups in the desired order
        general_keys = [
            "recursion_limit",
            "voxel_size",
            "sholl_range",
            "puncta_max_distance_um",
        ]
        smoothing_keys = [
            "smooth_radii",
            "radii_smoothing_window_length",
            "radii_smoothing_interval",
            "radii_smoothing_min",
        ]
        qc_keys = [
            "enforce_primaries_until",
            "min_bp_distance",
            "min_branch_length",
            "min_segment_length",
            "max_root_offset_um",
        ]

        # Helper to render a labeled group box
        def build_group(parent: ttk.Frame, title: str, keys: list[str], start_row: int) -> int:
            box = ttk.LabelFrame(parent, text=title)
            box.grid(row=start_row, column=0, sticky="we", padx=8, pady=(8 if start_row == 0 else 6, 6))
            box.columnconfigure(0, weight=0)
            box.columnconfigure(1, weight=1)
            box.columnconfigure(2, weight=0, minsize=HELP_COL_MINSIZE)

            row = 0
            for k in keys:
                value = getattr(params, k)
                path = f"parameters.{k}"

                # Row stripe behind controls
                self._stripe_row(box, row, col_span=3)

                # Render booleans as checkbuttons
                if isinstance(value, bool):
                    var = tk.BooleanVar(value=value)
                    cb_style = "Stripe1.TCheckbutton" if (row % 2) else "Stripe0.TCheckbutton"
                    cb = ttk.Checkbutton(box, text=_pretty_label(k), variable=var, style=cb_style)
                    cb.grid(row=row, column=0, columnspan=2, sticky="w", **padding)

                    self.field_vars[path] = (var, value)
                    self._help_label(box, row, path, help_col=2, wraplength=HELP_COL_MINSIZE)
                    row += 1
                    continue

                # Render scalar and sequence parameters as label + expanding entry
                lbl_style = "Stripe1.TLabel" if (row % 2) else "Stripe0.TLabel"
                ttk.Label(box, text=_pretty_label(k), style=lbl_style).grid(row=row, column=0, sticky="w", **padding)

                if isinstance(value, (tuple, list)):
                    text = ", ".join(str(v) for v in value)
                elif value is None:
                    text = ""
                else:
                    text = str(value)

                var = tk.StringVar(value=text)
                entry = ttk.Entry(box, textvariable=var)
                entry.grid(row=row, column=1, sticky="we", **padding)

                # Keep a handle to entries that are disabled by processing toggles
                self._parameter_widgets[k] = entry

                self.field_vars[path] = (var, value)
                self._help_label(box, row, path, help_col=2, wraplength=HELP_COL_MINSIZE)
                row += 1

            return start_row + 1

        # Build the three groups
        next_row = 0
        next_row = build_group(frame, "General", general_keys, next_row)
        next_row = build_group(frame, "Neurite radius smoothing", smoothing_keys, next_row)
        build_group(frame, "Quality control", qc_keys, next_row)


    def _make_scrollable_inner(self, container: ttk.Frame) -> ttk.Frame:
        """
        Create a scrollable content frame inside a tab container.

        Use:
            Build a Tk Canvas with a vertical scrollbar and embed an inner ttk.Frame
            as a canvas window. The returned inner frame is where callers should add
            widgets. The canvas scrollregion is kept in sync with inner frame size,
            and mousewheel scrolling is only active while the pointer is over the
            canvas to avoid global scroll conflicts.

        Args:
            container (ttk.Frame): Parent frame that hosts the canvas and scrollbar.

        Returns:
            ttk.Frame: The embedded inner frame where child widgets should be placed.
        """
        # Create the scrolling canvas and its vertical scrollbar
        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)

        # Pack scrollbar and canvas so the canvas expands to fill available space
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Create an inner frame and embed it as a window item inside the canvas
        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        # Update the canvas scrollregion whenever the inner frame size changes
        def _on_frame_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _on_frame_configure)

        # Keep the embedded window width matched to the visible canvas width
        def _on_canvas_configure(event):
            canvas.itemconfig(inner_id, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)

        # Handle mousewheel scrolling for the active canvas only
        def _on_mousewheel(event):
            if event.delta:
                direction = -1 if event.delta > 0 else 1
                canvas.yview_scroll(direction, "units")

        # Bind mousewheel events when the pointer enters the canvas area
        def _bind_wheel(_event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Unbind mousewheel events when the pointer leaves the canvas area
        def _unbind_wheel(_event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        # Return the inner frame so callers can populate it with widgets
        return inner


    def _build_visualization_tab(self, container: ttk.Frame) -> None:
        """
        Build the Visualization tab UI.

        Use:
            Create a nested notebook under the main Visualization tab. The nested
            notebook splits configuration into:
            - Global styling tabs shared by multiple views (Display, ScaleBar, Legend).
            - Per-view tabs that contain only section-level toggles and colors
                (Reconstruction, Geometry, Signal, Puncta, Sholl).

            Each sub-tab uses a scrollable inner frame so long forms remain usable
            in a fixed-size window. All widgets register Tk variables into
            self.field_vars so _build_config_from_form can rebuild a typed
            Visualization config later.

        Args:
            container (ttk.Frame): The main Visualization tab frame.
        """
        # Read default visualization values from the default Config snapshot
        viz = self._default_viz

        # Create the nested notebook that holds visualization sub-tabs
        vis_nb = ttk.Notebook(container)
        vis_nb.pack(fill="both", expand=True)

        # Keep the nested notebook handle so subtabs can be enabled/disabled
        self._viz_nb = vis_nb

        # Create one tab frame per sub-section of the visualization configuration
        display_tab = ttk.Frame(vis_nb)
        scalebar_tab = ttk.Frame(vis_nb)
        legend_tab = ttk.Frame(vis_nb)
        recon_tab = ttk.Frame(vis_nb)
        geom_tab = ttk.Frame(vis_nb)
        sholl_tab = ttk.Frame(vis_nb)
        puncta_tab = ttk.Frame(vis_nb)
        sig_tab = ttk.Frame(vis_nb)

        # Register each tab with a user-facing label (desired order)
        vis_nb.add(display_tab, text="Display")
        vis_nb.add(scalebar_tab, text="ScaleBar")
        vis_nb.add(legend_tab, text="Legend")
        vis_nb.add(recon_tab, text="Reconstruction")
        vis_nb.add(geom_tab, text="Geometry")
        vis_nb.add(sholl_tab, text="Sholl")
        vis_nb.add(puncta_tab, text="Puncta")
        vis_nb.add(sig_tab, text="Signal")

        # Cache tab frames for enable/disable control
        self._viz_tabs = {
            "signal": sig_tab,
            "puncta": puncta_tab,
        }

        # Create scrollable content frames so each tab can host a long form layout
        display_inner = self._make_scrollable_inner(display_tab)
        scalebar_inner = self._make_scrollable_inner(scalebar_tab)
        legend_inner = self._make_scrollable_inner(legend_tab)
        recon_inner = self._make_scrollable_inner(recon_tab)
        geom_inner = self._make_scrollable_inner(geom_tab)
        sig_inner = self._make_scrollable_inner(sig_tab)
        puncta_inner = self._make_scrollable_inner(puncta_tab)
        sholl_inner = self._make_scrollable_inner(sholl_tab)

        # Build the global Display controls using a representative default instance
        disp_default = viz.reconstruction.display
        disp_frame = ttk.LabelFrame(display_inner, text="Display (global)")
        disp_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=3)
        self._build_simple_dataclass(
            disp_frame,
            prefix="visualization.display",
            instance=disp_default,
        )
        display_inner.columnconfigure(0, weight=1)

        # Build the global ScaleBar controls using a representative default instance
        sb_default = viz.reconstruction.scale_bar
        sb_frame = ttk.LabelFrame(scalebar_inner, text="ScaleBar (global)")
        sb_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=4)
        self._build_simple_dataclass(
            sb_frame,
            prefix="visualization.scale_bar",
            instance=sb_default,
        )
        scalebar_inner.columnconfigure(0, weight=1)

        # Build the global Legend controls using a representative default instance
        lg_default = viz.reconstruction.legend
        lg_frame = ttk.LabelFrame(legend_inner, text="Legend (global)")
        lg_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=4)
        self._build_simple_dataclass(
            lg_frame,
            prefix="visualization.legend",
            instance=lg_default,
        )
        legend_inner.columnconfigure(0, weight=1)

        # Build section-level controls for each visualization mode
        sec_frame = ttk.LabelFrame(recon_inner, text="Reconstruction")
        sec_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=4)
        self._build_viz_section(sec_frame, "reconstruction", viz.reconstruction)
        recon_inner.columnconfigure(0, weight=1)

        sec_frame = ttk.LabelFrame(geom_inner, text="Geometry")
        sec_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=4)
        self._build_viz_section(sec_frame, "geometry", viz.geometry)
        geom_inner.columnconfigure(0, weight=1)

        sec_frame = ttk.LabelFrame(sig_inner, text="Signal")
        sec_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=4)
        self._build_viz_section(sec_frame, "signal", viz.signal)
        sig_inner.columnconfigure(0, weight=1)

        sec_frame = ttk.LabelFrame(puncta_inner, text="Puncta")
        sec_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=4)
        self._build_viz_section(sec_frame, "puncta", viz.puncta)
        puncta_inner.columnconfigure(0, weight=1)

        sec_frame = ttk.LabelFrame(sholl_inner, text="Sholl")
        sec_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=4)
        self._build_viz_section(sec_frame, "sholl", viz.sholl)
        sholl_inner.columnconfigure(0, weight=1)


    def _build_viz_section(self, frame: ttk.Frame, sec_name: str, sec: Any) -> None:
        """
        Build the per-section visualization controls.

        Use:
            Render a compact form for one visualization section (for example
            reconstruction, geometry, signal, puncta, or sholl). Only section-level
            fields are handled here. Shared styling fields (Display, ScaleBar,
            Legend) are edited in the global visualization tabs and are injected
            later when rebuilding the final Config.

            Widgets created here register their Tk variables into self.field_vars
            using the key pattern "visualization.<section>.<field>" so
            _build_config_from_form can reconstruct the section dataclass.

        Args:
            frame (ttk.Frame): Container that receives the section widgets.
            sec_name (str): Section name used to namespace field keys.
            sec (Any): Section config instance that supplies default values.
        """
        # Use a small, consistent padding so each row reads like a tidy form
        padding = {"padx": 6, "pady": 3}
        row = 0

        # Track the section-wide enable toggle so it can gate all other controls
        enable_var: Optional[tk.BooleanVar] = None
        other_widgets: list[Any] = []

        # Let the entry column expand while keeping the help column anchored
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)

        # Limit the UI to the handful of section-level fields we actually expose
        section_fields = [
            "enable",
            "show_axes_and_title",
            "show_scale_bar",
            "channel_names",
            "show_cell_metrics",
            "dot_radius_px",
            "color_soma",
            "color_neurites",
        ]

        # Override a few labels so the UI reads naturally (especially multiword flags)
        LABEL_OVERRIDES = {
            "enable": "Enable",
            "show_axes_and_title": "Show axes and title",
            "show_scale_bar": "Show scale bar",
            "channel_names": "Channel names (comma-separated strings)",
            "show_cell_metrics": "Show cell metrics",
            "dot_radius_px": "Dot radius (px)",
            "color_soma": "Soma color",
            "color_neurites": "Neurite color",
        }

        # Walk the supported fields and draw only those that exist on this section type
        for name in section_fields:
            if not hasattr(sec, name):
                continue

            # Pull the default value so we can choose the right widget type and seed the input
            value = getattr(sec, name)
            path = f"visualization.{sec_name}.{name}"

            # Stripe rows to keep wide forms readable at a glance
            self._stripe_row(frame, row, col_span=3)

            # Booleans become checkboxes; everything else becomes a labeled entry
            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                cb_style = "Stripe1.TCheckbutton" if (row % 2) else "Stripe0.TCheckbutton"
                cb = ttk.Checkbutton(
                    frame,
                    text=LABEL_OVERRIDES.get(name, _pretty_label(name)),
                    variable=var,
                    style=cb_style,
                )
                cb.grid(row=row, column=0, columnspan=2, sticky="w", **padding)

                # Keep a handle on the main enable toggle and collect all other controls for gating
                if name == "enable":
                    enable_var = var
                else:
                    other_widgets.append(cb)

                # Store the Tk variable + default for typed rebuild later
                self.field_vars[path] = (var, value)
                self._help_label(frame, row, path, help_col=2)
                row += 1
                continue

            # Label column uses the same stripe style so labels blend into the row background
            lbl_style = "Stripe1.TLabel" if (row % 2) else "Stripe0.TLabel"
            ttk.Label(
                frame,
                text=LABEL_OVERRIDES.get(name, _pretty_label(name)),
                style=lbl_style,
            ).grid(row=row, column=0, sticky="w", **padding)

            # Represent lists/tuples as comma-separated text to match the parser used on submit
            text = ", ".join(str(v) for v in value) if isinstance(value, (tuple, list)) else str(value)
            var = tk.StringVar(value=text)

            # Entries stretch to fill the row, but keep a reasonable minimum width for readability
            entry = ttk.Entry(frame, textvariable=var, width=30)
            entry.grid(row=row, column=1, sticky="we", **padding)
            other_widgets.append(entry)

            # Register the variable and add the per-field help string
            self.field_vars[path] = (var, value)
            self._help_label(frame, row, path, help_col=2)
            row += 1

        # When a section is disabled, make the rest of its controls read-only to match runtime behavior
        if enable_var is not None:

            def apply_enable_state() -> None:
                enabled = bool(enable_var.get())
                for w in other_widgets:
                    try:
                        w.state(["!disabled"] if enabled else ["disabled"])
                    except Exception:
                        try:
                            w.configure(state="normal" if enabled else "disabled")
                        except Exception:
                            pass

            enable_var.trace_add("write", lambda *_: apply_enable_state())
            apply_enable_state()


    def _build_aggregation_tab(self, frame: ttk.Frame) -> None:
        """
        Build the Aggregation tab controls.

        Use:
            Render the small set of user-editable aggregation inputs that control how
            post-processing tables are grouped and labeled. Each widget registers its
            Tk variable into self.field_vars under the key pattern "aggregation.<field>"
            so _build_config_from_form can rebuild a typed Aggregate instance.

            This tab may be disabled elsewhere when Processing.aggregate is False.

        Args:
            frame (ttk.Frame): Container for the Aggregation tab widgets.
        """
        # Common grid padding for widgets in this tab
        padding = {"padx": 8, "pady": 3}
        agg = self._default_agg

        # Wrap aggregation inputs in a labeled box for consistency with other tabs
        box = ttk.LabelFrame(frame, text="General")
        box.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)

        # Let the box expand with the tab
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # Configure a 3-column layout: label, entry, tooltip (inside the box)
        box.columnconfigure(0, weight=0)
        box.columnconfigure(1, weight=1)
        box.columnconfigure(2, weight=0, minsize=680)

        # Independents row
        row = 0
        self._stripe_row(box, row, col_span=3)

        lbl_style = "Stripe1.TLabel" if (row % 2) else "Stripe0.TLabel"
        ttk.Label(box, text="Independents (comma-separated ints)", style=lbl_style).grid(row=row, column=0, sticky="w", **padding)
        indep_var = tk.StringVar(value=", ".join(str(i) for i in agg.independents))
        ttk.Entry(box, textvariable=indep_var).grid(row=row, column=1, sticky="we", **padding)
        self.field_vars["aggregation.independents"] = (indep_var, agg.independents)
        self._help_label(box, row, "aggregation.independents", help_col=2, wraplength=680)
        row += 1

        # Norm independent row
        self._stripe_row(box, row, col_span=3)

        # Disable this field when signal extraction is off
        lbl_style = "Stripe1.TLabel" if (row % 2) else "Stripe0.TLabel"
        ttk.Label(box, text="Norm independent", style=lbl_style).grid(row=row, column=0, sticky="w", **padding)
        norm_var = tk.StringVar(value="" if agg.norm_independent is None else str(agg.norm_independent))
        norm_entry = ttk.Entry(box, textvariable=norm_var)
        norm_entry.grid(row=row, column=1, sticky="we", **padding)

        # Cache widget so Processing toggles can disable it
        self._aggregation_widgets["aggregation.norm_independent"] = norm_entry

        self.field_vars["aggregation.norm_independent"] = (norm_var, agg.norm_independent)
        self._help_label(box, row, "aggregation.norm_independent", help_col=2, wraplength=680)

        # Apply initial enable/disable state for Aggregation controls
        self._update_aggregation_field_states()


    def _set_aggregation_tab_state(self, enabled: bool) -> None:
        """
        Enable or disable the Aggregation tab in the main notebook.

        Use:
            Apply the Tk notebook tab state for the Aggregation tab based on the
            Processing.aggregate toggle. When disabling the tab while it is currently
            selected, the UI first switches to a safe fallback tab to avoid leaving
            the notebook on a disabled page.

        Args:
            enabled (bool): True to enable the Aggregation tab, False to disable it.
        """
        # Bail out if the notebook or aggregation tab has not been constructed yet
        if self.main_notebook is None or self.aggregation_tab is None:
            return

        # Use the Aggregation tab frame handle when applying per-tab state
        tab_id = self.aggregation_tab

        # Enable the tab when requested
        if enabled:
            self.main_notebook.tab(tab_id, state="normal")
            return

        # Switch away if the Aggregation tab is currently selected
        try:
            current = self.main_notebook.select()
            if current == str(tab_id):
                self.main_notebook.select(0)
        except Exception:
            pass

        # Disable the tab so it cannot be selected
        self.main_notebook.tab(tab_id, state="disabled")


    def _build_simple_dataclass(self, frame: ttk.Frame, prefix: str, instance: Any) -> None:
        """
        Render a small dataclass as a simple form inside a parent frame.

        Use:
            Iterate the dataclass fields on `instance` and create one row per field.
            Boolean fields are rendered as checkbuttons. All other fields are rendered
            as text entries, using a comma-separated representation for tuple/list
            defaults and an empty string for None.

            Each widget's Tk variable is registered in `self.field_vars` under a
            dotted key of the form "{prefix}.{field_name}". That registry is later
            consumed by `_build_config_from_form` to reconstruct typed dataclass
            instances.

            The "image_format" field is forced to "png" and made read-only when present.

        Args:
            frame (ttk.Frame): Container frame receiving the generated widgets.
            prefix (str): Dotted key prefix used when registering fields in `self.field_vars`.
            instance (Any): Dataclass instance whose fields should be rendered.
        """
        # Consistent spacing for all widgets in this block
        padding = {"padx": 6, "pady": 2}
        row = 0

        # Configure grid columns for label/input/help layout
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)

        # Emit one row per dataclass field
        for f in fields(type(instance)):
            value = getattr(instance, f.name)
            path = f"{prefix}.{f.name}"

            # Stripe the row for readability
            self._stripe_row(frame, row, col_span=3)

            # Render boolean fields as checkbuttons
            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                cb_style = "Stripe1.TCheckbutton" if (row % 2) else "Stripe0.TCheckbutton"
                cb = ttk.Checkbutton(frame, text=_pretty_label(f.name), variable=var, style=cb_style)
                cb.grid(row=row, column=0, columnspan=2, sticky="w", **padding)

                self.field_vars[path] = (var, value)
                self._help_label(frame, row, path, help_col=2)
                row += 1
                continue

            # Render non-boolean fields as a label + entry pair
            lbl_style = "Stripe1.TLabel" if (row % 2) else "Stripe0.TLabel"
            ttk.Label(frame, text=_pretty_label(f.name), style=lbl_style).grid(
                row=row, column=0, sticky="w", **padding
            )

            # Turn the default value into the same string format the parser expects on submit
            if value is None:
                text = ""
            elif isinstance(value, (tuple, list)):
                text = ", ".join(str(v) for v in value)
            else:
                text = str(value)

            # Lock image_format to png so saved figures stay consistent across the pipeline
            if f.name == "image_format":
                text = "png"

            # Bind a StringVar so edits can be collected later when rebuilding the dataclass
            var = tk.StringVar(value=text)

            # Make image_format read-only, but keep it in the form so it’s recorded in field_vars
            if f.name == "image_format":
                entry = ttk.Entry(frame, textvariable=var)
                entry.state(["readonly"])
                entry.grid(row=row, column=1, sticky="we", **padding)
            else:
                ttk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="we", **padding)

            # Register the Tk variable alongside the original default for typed parsing downstream
            self.field_vars[path] = (var, value)
            self._help_label(frame, row, path, help_col=2)

            # Advance to the next dataclass field row
            row += 1


    def _browse_directory(self, dir_var: tk.StringVar) -> None:
        """
        Prompt for a directory and write the selection into a bound Tk variable.

        Use:
            Let the user choose a filesystem folder via a standard dialog and copy
            the selected path into `dir_var`. This is used by the Pathing directory
            entry so the GUI can stay string-based until config construction.

        Args:
            dir_var (tk.StringVar): Tk variable bound to the directory Entry widget.
        """
        # Open a directory chooser dialog and update the bound variable if the user selects a folder
        directory = filedialog.askdirectory(title="Select data directory")
        if directory:
            dir_var.set(directory)


    def _on_run_clicked(self) -> None:
        """
        Start a pipeline run from the current GUI state.

        Use:
            Prevent concurrent runs, rebuild a typed Config from the form, then run
            the pipeline in a background thread. Progress updates are forwarded to
            the UI through `_update_progress`, and successful completion schedules
            `_on_pipeline_done` on the Tk thread.

        Raises:
            None. Errors are reported to the user via message boxes and the status label.
        """
        # Block re-entry if a previous run is still active
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo(
                "Pipeline running",
                "The pipeline is already running. Please wait for it to finish.",
                parent=self,
            )
            return

        # Build a validated Config from the current widget values
        try:
            cfg = self._build_config_from_form()
        except MorphologicError as e:
            messagebox.showerror("Configuration error", str(e), parent=self)
            return
        except Exception as e:
            messagebox.showerror("Unexpected error", f"Failed to build config:\n{e}", parent=self)
            return

        # Update UI state to reflect that a run has started
        self.status_var.set("Running pipeline…")
        self._set_running_state(True)

        # Run the pipeline in a daemon thread so the UI remains responsive
        def worker():
            try:
                sys.setrecursionlimit(cfg.parameters.recursion_limit)
                pipeline = MorphologyPipeline(cfg)
                pipeline.run(progress_cb=self._update_progress)

                # Schedule completion UI updates on the Tk thread
                self.after(0, self._on_pipeline_done)

            except (
                ConfigError,
                DataNotFound,
                SWCParseError,
                ImageIOError,
                ValidationError,
                MetricComputationError,
                VisualizationError,
            ) as e:
                self._set_status_safe("Pipeline failed with a domain error.")
                messagebox.showerror("Pipeline error", str(e), parent=self)

            except MorphologicError as e:
                self._set_status_safe("Pipeline failed with a morphological error.")
                messagebox.showerror("Morphological error", str(e), parent=self)

            except Exception as e:
                self._set_status_safe("Pipeline failed with an unexpected error.")
                messagebox.showerror("Unexpected error", f"{type(e).__name__}: {e}", parent=self)

            finally:
                # Clear the worker handle and restore UI controls regardless of outcome
                self.worker_thread = None
                self._set_running_state(False)

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()


    def _on_close(self) -> None:
        """
        Close the GUI window safely.

        Use:
            If a run is currently in progress, ask the user whether to quit anyway.
            If no run is active, or the user confirms, destroy the window and end
            the Tk mainloop.

        Raises:
            None.
        """
        # Confirm before exiting if a background run is active
        if self.worker_thread is not None and self.worker_thread.is_alive():
            ok = messagebox.askyesno(
                "Quit",
                "The pipeline is still running.\n\nQuit anyway?",
                parent=self,
            )
            if not ok:
                return

        # Destroy the root window and exit the Tk event loop
        self.destroy()


    def _set_running_state(self, running: bool) -> None:
        """
        Toggle run-related widgets between idle and running states.

        Use:
            Disable the Run button during execution and reset the progress bar at
            run start. Re-enable the Run button when the run ends. Updates are
            scheduled onto the Tk thread using `after`.

        Args:
            running (bool): True while a pipeline run is active, False otherwise.
        """
        # Apply widget state changes on the Tk thread
        def update():
            if self.progress is None or self.run_btn is None:
                return

            if running:
                self.progress["value"] = 0
                self.run_btn.config(state="disabled")
            else:
                self.run_btn.config(state="normal")

        self.after(0, update)


    def _set_status_safe(self, text: str) -> None:
        """
        Update the status label from any thread.

        Use:
            Schedule a `status_var.set(...)` call on the Tk thread so worker code
            can safely report status changes.

        Args:
            text (str): Status message to display in the GUI.
        """
        # Schedule a simple status update on the Tk thread
        self.after(0, lambda: self.status_var.set(text))


    def _on_pipeline_done(self) -> None:
        """
        Finalize UI state after a successful pipeline run.

        Use:
            Set the progress bar to its maximum, flush pending UI updates, then
            present a completion dialog and update the status label.

        Raises:
            None.
        """
        # Mark the progress bar complete if it exists
        if hasattr(self, "progress") and isinstance(self.progress, ttk.Progressbar):
            try:
                self.progress["value"] = self.progress["maximum"]
            except Exception:
                pass

        # Flush pending redraws before presenting modal UI
        self.update_idletasks()

        # Report completion in the status label and via a modal dialog
        self.status_var.set("Pipeline finished successfully.")
        messagebox.showinfo("Done", "Pipeline finished successfully.", parent=self)


    def _build_config_from_form(self) -> Config:
        """
        Rebuild a typed Config instance from the current GUI state.

        Use:
            Read Tkinter variables stored in `self.field_vars`, parse them into the
            appropriate Python types, and construct a new Config dataclass. This is
            the single point where UI strings and checkboxes become typed
            configuration objects.

            This method also performs minimal validation that is required before the
            pipeline can run, most notably that `pathing.directory` exists and is a
            directory.

        Returns:
            Config: A fully populated configuration instance reflecting the current
            contents of the GUI.

        Raises:
            ConfigError: If `pathing.directory` is missing, empty, or does not exist.
            ValueError: If a numeric field cannot be parsed (raised by parsing helpers).
            KeyError: If required widget paths are missing from `self.field_vars`.
        """
        # Read and validate the input directory early because later parsing depends on it
        dir_var, _dir_orig = self.field_vars["pathing.directory"]
        directory = Path(dir_var.get()).expanduser()
        if not directory:
            raise ConfigError("Please choose a data directory.")
        if not directory.exists() or not directory.is_dir():
            raise ConfigError(f"Directory does not exist: {directory}")

        # Helper for scalar pathing fields that are stored as strings in the UI
        def read_str(path: str, orig: Any) -> Any:
            var, _ = self.field_vars[path]
            return _parse_scalar(orig, var.get())

        # Parse Pathing fields from the General tab and allow explicit empty overrides
        image_suffix = read_str("pathing.image_suffix", self._default_pathing.image_suffix)
        soma_roi_suffix = read_str("pathing.soma_roi_suffix", self._default_pathing.soma_roi_suffix)

        var_punc, _orig_punc = self.field_vars["pathing.puncta_roi_suffix"]
        raw_punc = var_punc.get().strip()
        puncta_roi_suffix = "" if raw_punc == "" or raw_punc.lower() == "none" else raw_punc

        # Parse signal channels from comma-separated text into the configured sequence type
        var_sig, orig_sig = self.field_vars["pathing.signal_channels"]
        raw_sig = var_sig.get().strip()
        sig_tuple = () if raw_sig == "" or raw_sig.lower() == "none" else tuple(_parse_seq(orig_sig, raw_sig))

        var_nuc, _orig_nuc = self.field_vars["pathing.nuclear_roi_suffix"]
        raw_nuc = var_nuc.get().strip()
        nuclear_roi_suffix = "" if raw_nuc == "" or raw_nuc.lower() == "none" else raw_nuc

        # Normalize image suffix so ".tif" is always present
        image_suffix = str(image_suffix).strip()
        if not image_suffix.lower().endswith((".tif", ".tiff")):
            image_suffix += ".tif"

        # Construct the typed Pathing dataclass
        pathing = Pathing(
            directory=directory,
            image_suffix=image_suffix,
            soma_roi_suffix=soma_roi_suffix,
            puncta_roi_suffix=puncta_roi_suffix,
            signal_channels=tuple(sig_tuple),
            nuclear_roi_suffix=nuclear_roi_suffix,
        )

        # Rebuild Processing by iterating the dataclass fields and parsing values against defaults
        proc_kwargs: Dict[str, Any] = {}
        for f in fields(Processing):
            path = f"processing.{f.name}"
            default_val = getattr(self._default_proc, f.name)

            if path not in self.field_vars:
                proc_kwargs[f.name] = default_val
                continue

            var, _ = self.field_vars[path]
            if isinstance(default_val, bool):
                proc_kwargs[f.name] = bool(var.get())
            else:
                proc_kwargs[f.name] = _parse_scalar(default_val, var.get())

        processing = Processing(**proc_kwargs)

        # Extract_signal requires one or more signal channels
        if processing.extract_signal and len(pathing.signal_channels) == 0:
            raise ConfigError(
                "processing.extract_signal is enabled, but pathing.signal_channels is empty. "
                "Provide one or more channels (e.g. '1, 4') or disable extract_signal."
            )

        # Extract_puncta requires a non-empty puncta ROI suffix
        if processing.extract_puncta and not str(pathing.puncta_roi_suffix).strip():
            raise ConfigError(
                "processing.extract_puncta is enabled, but pathing.puncta_roi_suffix is empty. "
                "Provide a suffix (e.g. '_puncta') or disable extract_puncta."
            )

        # Disable deduct_nuclei if extract_signal is disabled
        if not processing.extract_signal and processing.deduct_nuclei:
            processing = replace(processing, deduct_nuclei=False)

        # Deduct_nuclei requires a non-empty nuclear ROI suffix
        if processing.deduct_nuclei and not str(pathing.nuclear_roi_suffix).strip():
            raise ConfigError(
                "processing.deduct_nuclei is enabled, but pathing.nuclear_roi_suffix is empty. "
                "Provide a suffix (e.g. '_nuclei') or disable deduct_nuclei."
            )

        # Clear signal_channels if extract_signal is disabled
        if not processing.extract_signal and pathing.signal_channels:
            pathing = replace(pathing, signal_channels=())

        # Clear puncta_roi_suffix if extract_puncta is disabled
        if not processing.extract_puncta and str(pathing.puncta_roi_suffix).strip():
            pathing = replace(pathing, puncta_roi_suffix="")

        # Rebuild Parameters with scalar vs sequence parsing based on the default value type
        params_kwargs: Dict[str, Any] = {}
        for f in fields(Parameters):
            path = f"parameters.{f.name}"
            default_val = getattr(self._default_params, f.name)
            var, _ = self.field_vars[path]

            if isinstance(default_val, bool):
                params_kwargs[f.name] = bool(var.get())
                continue

            raw = var.get()
            if isinstance(default_val, (tuple, list)):
                params_kwargs[f.name] = _parse_seq(default_val, raw)
            else:
                params_kwargs[f.name] = _parse_scalar(default_val, raw)

        parameters = Parameters(**params_kwargs)

        # Rebuild a shared visualization dataclass (Display / ScaleBar / Legend) from the global tabs
        def build_global_dataclass(prefix: str, cls: Any, default_instance: Any) -> Any:
            """
            Rebuild a visualization sub-dataclass (Display, ScaleBar, or Legend) from GUI fields.

            Use:
                Read `{prefix}.{field}` from `self.field_vars`, parse each value using
                the corresponding field in `default_instance` to infer type, then
                construct a new instance of `cls`.

            Args:
                prefix (str): Key prefix used in `self.field_vars`.
                cls (Any): Dataclass type to construct.
                default_instance (Any): Default instance used to infer parsing types.

            Returns:
                Any: A new `cls` instance populated from GUI inputs.
            """
            # Collect parsed field values into kwargs for the dataclass constructor
            kw: Dict[str, Any] = {}
            for f in fields(cls):
                # Look up the Tk variable for this field and keep the default to drive parsing
                path = f"{prefix}.{f.name}"
                var, _ = self.field_vars[path]
                raw = var.get()
                default_val = getattr(default_instance, f.name)

                # Parse sequences as comma-separated lists; parse scalars by default type
                if isinstance(default_val, (tuple, list)):
                    kw[f.name] = _parse_seq(default_val, raw)
                else:
                    kw[f.name] = _parse_scalar(default_val, raw)

            # Construct the typed dataclass instance from the collected values
            return cls(**kw)

        # Rebuild global Display, ScaleBar, and Legend from their tabs
        viz_default = self._default_viz
        global_display = build_global_dataclass("visualization.display", Display, viz_default.reconstruction.display)
        global_scalebar = build_global_dataclass("visualization.scale_bar", ScaleBar, viz_default.reconstruction.scale_bar)
        global_legend = build_global_dataclass("visualization.legend", Legend, viz_default.reconstruction.legend)

        # Helper to rebuild a visualization section using section-only widgets plus injected globals
        def build_section_only(name: str, orig_section: Any) -> Any:
            sec_kwargs: Dict[str, Any] = {}

            # Read only non-nested section fields from the section tab
            for f_sec in fields(type(orig_section)):
                if f_sec.name in {"display", "scale_bar", "legend"}:
                    continue

                path = f"visualization.{name}.{f_sec.name}"
                if path not in self.field_vars:
                    continue

                var, _ = self.field_vars[path]
                raw = var.get()
                default_val = getattr(orig_section, f_sec.name)

                if isinstance(default_val, bool):
                    sec_kwargs[f_sec.name] = bool(var.get())
                elif isinstance(default_val, (tuple, list)):
                    sec_kwargs[f_sec.name] = _parse_seq(default_val, raw)
                else:
                    sec_kwargs[f_sec.name] = _parse_scalar(default_val, raw)

            # Inject the global visualization dataclasses
            sec_kwargs["display"] = global_display
            if hasattr(orig_section, "scale_bar"):
                sec_kwargs["scale_bar"] = global_scalebar
            if hasattr(orig_section, "legend"):
                sec_kwargs["legend"] = global_legend

            return type(orig_section)(**sec_kwargs)

        # Rebuild each visualization section and assemble the Visualization config
        viz_recon = build_section_only("reconstruction", viz_default.reconstruction)
        viz_geom = build_section_only("geometry", viz_default.geometry)
        viz_signal = build_section_only("signal", viz_default.signal)
        viz_puncta = build_section_only("puncta", viz_default.puncta)
        viz_sholl = build_section_only("sholl", viz_default.sholl)

        visualization = VizConfig(
            reconstruction=viz_recon,
            geometry=viz_geom,
            signal=viz_signal,
            puncta=viz_puncta,
            sholl=viz_sholl,
        )
        
        # Validate visualization channel names when signal rendering is enabled
        if processing.extract_signal and pathing.signal_channels:
            names = tuple(visualization.signal.channel_names)
            chans = tuple(pathing.signal_channels)

            if len(names) != len(chans):
                raise ConfigError(
                    "visualization.signal.channel_names must have the same length as "
                    "pathing.signal_channels."
                )

            if any(not str(n).strip() for n in names):
                raise ConfigError(
                    "visualization.signal.channel_names contains empty names. "
                    "Provide a non-empty name for each entry in pathing.signal_channels."
                )

        # Start Aggregation from defaults and apply UI overrides for fields that are exposed in the GUI
        agg_default = self._default_agg
        agg_kwargs: Dict[str, Any] = {f.name: getattr(agg_default, f.name) for f in fields(Aggregate)}

        if "aggregation.independents" in self.field_vars:
            var, orig = self.field_vars["aggregation.independents"]
            agg_kwargs["independents"] = _parse_seq(orig, var.get())

        if "aggregation.norm_independent" in self.field_vars:
            var, _orig = self.field_vars["aggregation.norm_independent"]
            raw = var.get().strip()
            agg_kwargs["norm_independent"] = None if raw == "" or raw.lower() == "none" else int(raw)

        aggregation = Aggregate(**agg_kwargs)

        # Derive channel- and puncta-dependent aggregation fields from current Pathing settings
        aggregation = aggregation.with_derived_dependents(
            signal_channels=pathing.signal_channels,
            puncta_roi_suffix=pathing.puncta_roi_suffix,
        )

        # Return the fully typed configuration
        return Config(
            pathing=pathing,
            processing=processing,
            parameters=parameters,
            visualization=visualization,
            aggregation=aggregation,
        )


def _parse_scalar(orig: Any, raw: str) -> Any:
    """
    Parse a UI string into a scalar compatible with a default value.

    Use:
        Convert a string read from a Tkinter Entry back into the expected Python
        type by inspecting `orig` (the default/original value). This allows the
        GUI to store user input as text while still producing a strongly typed
        configuration object.

        Supported conversions (based on the type of `orig`):
            - bool  : case-insensitive membership in {"1", "true", "yes", "on"}
            - int   : int(raw)
            - float : float(raw)
            - Path  : Path(raw)
            - str   : raw

        Special handling for Optional fields:
            - If raw is blank or "none" and orig is None, return None
            - If raw is blank or "none" and orig is not None, return orig unchanged
            - If orig is None and raw is non-empty, infer int then float, else keep string

    Args:
        orig (Any): Default/original value whose type controls conversion.
        raw (str): Raw string from the UI.

    Returns:
        Any: Parsed value compatible with `orig`.

    Raises:
        ValueError: If conversion to int/float is inferred and parsing fails.
    """
    # Normalize UI text input
    raw = raw.strip()

    # Treat blank or "none" as "no override"
    if raw == "" or raw.lower() == "none":
        return None if orig is None else orig

    # When the default is None, infer a reasonable scalar type from the input
    if orig is None:
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw

    # Convert based on the type of the default value
    if isinstance(orig, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(orig, int):
        return int(raw)
    if isinstance(orig, float):
        return float(raw)
    if isinstance(orig, Path):
        return Path(raw)
    if isinstance(orig, str):
        return raw

    # Unknown scalar type: return the raw string unchanged
    return raw


def _parse_seq(orig: Any, raw: str) -> Any:
    """
    Parse comma-separated text into a sequence compatible with a default value.

    Use:
        Convert a comma-separated UI string (e.g. "2, 4") into a tuple/list whose
        container type matches `orig` and whose element type is inferred from the
        first element of `orig` when available.

        Behavior:
            - If `raw` is blank/whitespace, return `orig` unchanged.
            - Split on commas and strip whitespace.
            - Infer element conversion from `orig[0]` when `orig` is a non-empty
              tuple/list:
                * float -> float(part)
                * int   -> int(part)
                * else  -> keep as string
            - Preserve container type:
                * tuple -> tuple(converted)
                * list  -> list(converted)
            - If `orig` is not a tuple/list, return the converted values as a list.

    Args:
        orig (Any): Default/original value used to infer container and element types.
        raw (str): Comma-separated string from the UI.

    Returns:
        Any: Parsed values as the same container type as `orig`, or `orig` if `raw` is empty.

    Raises:
        ValueError: If element conversion is inferred as int/float and conversion fails.
    """
    # Treat blank input as "leave default unchanged"
    if raw.strip() == "":
        return orig

    # Split on commas and trim whitespace around each token
    parts = [p.strip() for p in raw.split(",")]

    # Infer element type from the first element of the default sequence when available
    elem_example = None
    if isinstance(orig, (tuple, list)) and orig:
        elem_example = orig[0]

    def conv(p: str) -> Any:
        if isinstance(elem_example, float):
            return float(p)
        if isinstance(elem_example, int):
            return int(p)
        return p

    # Convert all tokens into typed values
    converted = [conv(p) for p in parts]

    # Preserve the container type of the default value when possible
    if isinstance(orig, tuple):
        return tuple(converted)
    if isinstance(orig, list):
        return converted
    return converted


def _pretty_label(name: str) -> str:
    """
    Convert a snake_case identifier into a human-friendly label.

    Use:
        Format dataclass field names for display in Tkinter widgets by replacing
        underscores with spaces and using sentence-style capitalization. Units
        like "um" are omitted from the title label.

    Example:
        "signal_channels" -> "Signal channels"
        "puncta_max_distance_um" -> "Puncta max distance"
    """
    # Split into tokens and drop the unit suffix used in field names
    parts = [p for p in name.split("_") if p and p.lower() != "um"]

    # Avoid producing junk labels for empty or fully-filtered names
    if not parts:
        return ""

    # Preserve acronyms while normalizing ordinary words
    def keep_word(w: str) -> str:
        if w.isupper():
            return w
        return w.lower()

    # Capitalize the first word and normalize the remainder for a sentence-style label
    first = parts[0].capitalize() if not parts[0].isupper() else parts[0]
    rest = [keep_word(w) for w in parts[1:]]
    return " ".join([first] + rest)


def main() -> None:
    """
    Launch the configuration GUI and start the Tk event loop.

    Use:
        Construct the top-level `ConfigGUI` window and enter Tkinter's
        main event loop. This function is the single entry point used by the
        `if __name__ == "__main__":` guard.
    """
    # Construct the top-level application window
    app = ConfigGUI()

    # Hand control to Tkinter until the window is closed
    app.mainloop()


if __name__ == "__main__":
    main()