from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class TargetDateCase:
    target_date: pd.Timestamp
    lead_days: int
    init_time: pd.Timestamp
    required_horizon_days: int
    required_6hour_steps: int
    prediction_path: Path

    @property
    def target_label(self) -> str:
        return self.target_date.strftime("%Y%m%d")

    @property
    def init_label(self) -> str:
        return self.init_time.strftime("%Y%m%d")


def build_cases(
    target_dates: list[str],
    lead_days: list[int],
    prediction_pattern: str,
) -> list[TargetDateCase]:
    cases: list[TargetDateCase] = []

    for raw_target in target_dates:
        target_date = pd.Timestamp(raw_target).normalize()

        for lead in lead_days:
            lead = int(lead)
            init_time = target_date - pd.Timedelta(days=lead)

            # Correct target-date hindcast design:
            # the target is a date, so a 30/60/90-day lead means a 30/60/90-day rollout.
            required_horizon_days = lead
            required_6hour_steps = lead * 4

            pred = prediction_pattern.format(
                target_yyyymmdd=target_date.strftime("%Y%m%d"),
                init_yyyymmdd=init_time.strftime("%Y%m%d"),
                lead=lead,
                horizon=required_horizon_days,
            )

            cases.append(
                TargetDateCase(
                    target_date=target_date,
                    lead_days=lead,
                    init_time=init_time,
                    required_horizon_days=required_horizon_days,
                    required_6hour_steps=required_6hour_steps,
                    prediction_path=Path(pred),
                )
            )

    return cases


def cases_to_frame(cases: list[TargetDateCase]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_date": c.target_date.strftime("%Y-%m-%d"),
                "lead_days": c.lead_days,
                "init_time": c.init_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "required_horizon_days": c.required_horizon_days,
                "required_6hour_steps": c.required_6hour_steps,
                "prediction_path": str(c.prediction_path),
                "prediction_exists": c.prediction_path.exists(),
            }
            for c in cases
        ]
    )
