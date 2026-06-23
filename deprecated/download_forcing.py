#!/usr/bin/env python3
"""Download ACE2-ERA5 forcing files from HuggingFace.

Usage:
    python download_forcing.py --years 2002,2003,...,2017 --out-dir /path/to/forcing_data_ace2era5
    python download_forcing.py --years 2002,2003,...,2017  # writes to ./data/lag_data/forcing_data_ace2era5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "allenai/ACE2-ERA5"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--years", required=True,
                   help="Comma-separated forcing years, e.g. 2002,2003,...,2017")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <cwd>/data/lag_data/forcing_data_ace2era5)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    years = [int(y) for y in args.years.split(",")]
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "data/lag_data/forcing_data_ace2era5"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {len(years)} forcing files to {out_dir}", flush=True)

    failed = []
    for year in years:
        dest = out_dir / f"forcing_{year}.nc"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [{year}] skip (exists)", flush=True)
            continue

        hf_path = f"forcing_data/forcing_{year}.nc"
        print(f"  [{year}] downloading {hf_path} ...", flush=True)

        if args.dry_run:
            print(f"  DRY: hf_hub_download({REPO_ID!r}, {hf_path!r}) -> {dest}")
            continue

        try:
            tmp = hf_hub_download(
                repo_id=REPO_ID,
                filename=hf_path,
                repo_type="model",
                local_dir=str(out_dir),
            )
            # hf_hub_download writes to a cache subpath; if it didn't land at dest, move it
            tmp_path = Path(tmp)
            if tmp_path != dest and tmp_path.exists():
                tmp_path.rename(dest)
            print(f"  [{year}] done -> {dest}", flush=True)
        except Exception as e:
            print(f"  [{year}] FAILED: {e}", flush=True)
            failed.append(year)

    if failed:
        print(f"\nFailed years: {failed}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll forcing files downloaded to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
