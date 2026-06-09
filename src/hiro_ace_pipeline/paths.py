from __future__ import annotations
from pathlib import Path


def ensure_layout(output_root: Path) -> dict[str, Path]:
    paths = {
        "processed": output_root / "processed_targetdate",
        "figures": output_root / "figures_targetdate",
        "cache": output_root / "cache" / "era5_daily_tmax",
        "plans": output_root / "plans",
        "metrics": output_root / "metrics",
        "synthetic_predictions": output_root / "synthetic_predictions",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
