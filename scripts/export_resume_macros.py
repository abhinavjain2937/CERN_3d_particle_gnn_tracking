"""Write LaTeX macros (\\trkAUC, \\trkNodes, ...) from a finished run, for the resume.

Usage:  python export_resume_macros.py path/to/runs/default trackml_metrics.tex
"""
import sys
from pathlib import Path

# Requires the repo to be installed:  pip install -e path/to/trackml-edge-classifier
from trackml_edge.report import resume_macros
from trackml_edge.utils import load_json

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    Path(sys.argv[2]).write_text(resume_macros(load_json(Path(sys.argv[1]) / "metrics.json")), encoding="utf-8")
    print("wrote", sys.argv[2])
