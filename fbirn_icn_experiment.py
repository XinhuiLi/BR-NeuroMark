#!/usr/bin/env python3
"""
Launcher for the FBIRN ICN experiment. Implementation: ``fbirn_experiment`` package.

Examples:
  python fbirn_icn_experiment.py
  python fbirn_icn_experiment.py --synthetic --out results/synth_demo
  python -m fbirn_experiment.cli --help
"""

from fbirn_experiment.cli import main

if __name__ == "__main__":
    main()
