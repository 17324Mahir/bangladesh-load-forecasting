"""Run the archived Colab preparation and point-forecast code on local CSVs.

The original notebook cells are the source of all data and model transformations.
Only Drive mounting, interactive file upload, and plotting are omitted.
"""
import argparse
import hashlib
import json
from pathlib import Path


NOTEBOOKS = {
    "01": ("01_PGCB_Data_Cleaning.ipynb", "# 43. FIGURE 1:"),
    "02": ("02_Advanced_Data_Cleaning.ipynb", "# 39. FIGURE 1"),
    "03": ("03_Forecasting_Dataset_and_EDA.ipynb", "# 60. FIGURE 1"),
    "07": ("07_Adaptive_Conformal_Framework.ipynb", "# 27. TEST WINDOWS"),
}
RAW_SHA256 = "05c2350f5f3eb398d0b5ca9e9fcdcad31509fa6e1340f8fcef15ec64cbee4dc3"
DRIVE_ROOT = "/content/drive/MyDrive/Load_Forecasting_Paper"


def notebook_source(notebook_dir, stage):
    filename, end_marker = NOTEBOOKS[stage]
    path = Path(notebook_dir) / filename
    notebook = json.loads(path.read_text())
    code = "\n\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    # These pinned notebooks have one code cell split into named sections.
    # Start after mounting Drive. End once all downstream data / functions exist.
    before_mount, remaining = code.split("# 1. MOUNT GOOGLE DRIVE", 1)
    start = remaining.index("# 2. ")
    after_mount = remaining[start:]
    if end_marker not in after_mount:
        raise ValueError(f"{filename}: expected notebook section {end_marker!r}")
    return (before_mount + after_mount.split(end_marker, 1)[0]).replace(
        "import matplotlib.pyplot as plt", ""
    )


def run_notebook(notebook_dir, project_root, stage, raw_csv=None):
    """Return globals from the original notebook up to the requested output stage."""
    root = Path(project_root).resolve()
    source = notebook_source(notebook_dir, stage)
    source = source.replace(f'"{DRIVE_ROOT}"', repr(str(root)))
    if stage == "01":
        raw = Path(raw_csv or root / "data/raw/pgcb_V1_raw.csv").resolve()
        if hashlib.sha256(raw.read_bytes()).hexdigest() != RAW_SHA256:
            raise ValueError("Input differs from the archived PGCB V1 CSV.")
        before_upload, rest = source.split("# 3. UPLOAD THE HOURLY DATASET", 1)
        _, after_upload = rest.split("# 7. LOAD DATA", 1)
        source = before_upload + f"pgcb_path = Path({str(raw)!r})\n" + after_upload
    namespace = {"__name__": "__main__", "display": lambda *_: None}
    exec(compile(source, str(Path(notebook_dir) / NOTEBOOKS[stage][0]), "exec"), namespace)
    return namespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--stages", nargs="+", choices=NOTEBOOKS, default=["01", "02", "03"])
    args = parser.parse_args()
    for stage in args.stages:
        run_notebook(args.notebooks, args.project, stage, args.raw)
