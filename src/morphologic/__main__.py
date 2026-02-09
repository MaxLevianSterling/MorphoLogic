# src/morphologic/__main__.py
from __future__ import annotations

# General imports (stdlib)
import sys

# Local imports
from .config import make_config
from .core import MorphologyPipeline


def main() -> None:
    cfg = make_config()

    sys.setrecursionlimit(cfg.parameters.recursion_limit)

    MorphologyPipeline(cfg).run()


if __name__ == "__main__":
    main()