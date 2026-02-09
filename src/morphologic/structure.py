# src/morphologic/structure.py
from __future__ import annotations

# General imports (stdlib)
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Group structure:(image_name, image_path, path_structure, tracing_files)
Group = Tuple[str, Path, List[str], List[Path]]
"""
Group:
    A single folder-level group of trace files discovered by io.discover_traces.

    Layout:
        image_name      (str)        : Folder / image name used as top-level key.
        image_path      (Path)       : Path to the folder containing the trace files.
        path_structure  (List[str])  : Relative path components from the root to this folder.
        tracing_files   (List[Path]) : List of SWC/trace file paths in this folder.
"""


def create_data_structure(files_with_paths: List[Group]) -> tuple[Dict[str, Any], int]:
    """
    Build the nested data structure used by the morphology pipeline.

    Use:
        Convert a flat list of `Group` tuples into a hierarchical dictionary keyed
        by folder structure and file stems. Each leaf corresponds to one “cell”
        (one SWC/trace file) and is initialized with empty containers ready for
        pipeline outputs.

    Args:
        files_with_paths (List[Group]):
            Output from `io_swc.discover_traces` (or equivalent), where each entry is:
              (image_name, image_path, path_structure, tracing_files)

    Returns:
        tuple[Dict[str, Any], int]:
            - data_structure: Nested dictionary representing folders and cells.
            - n_cells: Total number of cells (leaf entries) added.
    """
    data_structure: Dict[str, Any] = {}     # Root dictionary for hierarchical data storage
    n_cells = 0                             # Counter for unique cell instances

    for image_name, _, path_structure, tracing_files in files_with_paths:
        # Append image name to the directory structure
        path_structure.append(image_name)

        # Traverse and create necessary nested levels in the dictionary
        current = data_structure
        for level in path_structure:
            current = current.setdefault(level, {})  # drill down / create levels

        # Populate the data structure with tracing file details
        for file_path in tracing_files:
            key = file_path.stem
            current[key] = {
                'geometric_dataframes': [],
                'geometric_analysis_cell': [],
                'geometric_analysis_neurite': [],
                'neurite_segments': [],
                'dendritic_tree_area': None,
                'sholl_dataframes': [],
                'sholl_analysis_cell': [],
                'sholl_analysis_neurite': [],
                'soma_roi': None,
                'somatic_metrics': dict(),
                'soma_id': n_cells,
                'file': str(file_path),
                'path_structure': path_structure
            }
            n_cells += 1

    return data_structure, n_cells