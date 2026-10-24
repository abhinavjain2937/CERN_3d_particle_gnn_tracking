"""Insert the results table of a finished run into README.md.

Usage:  python scripts/update_readme.py runs/default
The table goes between the <!-- RESULTS:START --> and <!-- RESULTS:END --> markers.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trackml_edge.report import metrics_table_md  # noqa: E402
from trackml_edge.utils import load_json  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def main(run_dir: str) -> None:
    metrics = load_json(Path(run_dir) / "metrics.json")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit("README.md has no RESULTS markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    block = f"{START}\n{metrics_table_md(metrics)}\n{END}"
    readme.write_text(head + block + tail, encoding="utf-8")
    print("README.md results table updated from", run_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
