from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass
class PipelineConfig:
    project_root: Path
    output_root: Path
    target_dates: list[str]
    lead_days: list[int]
    bbox: tuple[float, float, float, float]
    expected_members: int
    baseline_start_year: int
    baseline_end_year: int | None
    era5_zarr: str
    prediction_pattern: str
    figure_formats: list[str]
    max_map_members: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path).resolve()
        project_root = path.parents[1]
        raw: dict[str, Any] = yaml.safe_load(path.read_text())

        def resolve_path(value: str) -> str:
            p = Path(value)
            return str(p if p.is_absolute() else project_root / p)

        return cls(
            project_root=project_root,
            output_root=Path(resolve_path(str(raw["output_root"]))),
            target_dates=list(raw["target_dates"]),
            lead_days=[int(x) for x in raw["lead_days"]],
            bbox=tuple(float(x) for x in raw["bbox"]),
            expected_members=int(raw["expected_members"]),
            baseline_start_year=int(raw["baseline_start_year"]),
            baseline_end_year=None if raw.get("baseline_end_year") is None else int(raw["baseline_end_year"]),
            era5_zarr=str(raw["era5_zarr"]),
            prediction_pattern=resolve_path(str(raw["prediction_pattern"])),
            figure_formats=list(raw.get("figure_formats", ["png"])),
            max_map_members=int(raw.get("max_map_members", 6)),
        )
