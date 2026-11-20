"""Download the versioned PGCB V1 raw input from its Mendeley Data release."""
import argparse
import hashlib
import io
from pathlib import Path
import subprocess
from zipfile import ZipFile

from original_notebooks import RAW_SHA256


def fetch(project: Path) -> None:
    destination = project / "data/raw/pgcb_V1_raw.csv"
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != RAW_SHA256:
            raise ValueError(f"Existing file differs from PGCB V1: {destination}")
    else:
        url = "https://data.mendeley.com/public-api/zip/vpk8spw2mm/download/1"
        download = subprocess.run(["curl", "-fsSL", "--max-time", "90", url],
                                  check=True, stdout=subprocess.PIPE)
        with ZipFile(io.BytesIO(download.stdout)) as archive:
            matches = [name for name in archive.namelist() if Path(name).name == destination.name]
            if len(matches) != 1:
                raise ValueError(f"Expected one {destination.name} in the version 1 archive.")
            data = archive.read(matches[0])
        if hashlib.sha256(data).hexdigest() != RAW_SHA256:
            raise ValueError("Downloaded data does not match the published experiment input.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print(f"{RAW_SHA256}  {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    fetch(parser.parse_args().project)
