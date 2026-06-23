#!/usr/bin/env python3
"""Run ACE2S inference with a runtime fix for pre-expanded ensemble forcing.

FME's inference loader can return forcing data whose sample dimension has already
been expanded to match the ensemble-expanded initial condition while the
``n_ensemble`` metadata still says 1. In that case ``Stepper.predict_paired``
expands the forcing a second time, producing ``n_ensemble ** 2`` samples. This
wrapper preserves the normal behavior except when the forcing and initial
condition batch sizes already match.
"""

from __future__ import annotations

import argparse
import dataclasses
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.ace.inference.inference import main as inference_main
from fme.ace.stepper.single_module import Stepper
from fme.core.distributed.distributed import Distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config")
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotlist override passed through to fme.ace.inference.inference.main.",
    )
    return parser.parse_args()


def batch_size(batch_data: BatchData) -> int:
    for value in batch_data.data.values():
        return int(value.shape[0])
    return 0


def patch_predict_paired() -> None:
    def predict_paired(
        self: Stepper,
        initial_condition: PrognosticState,
        forcing: BatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[PairedData, PrognosticState]:
        forcing = self.forcing_deriver(forcing)
        ic_data = initial_condition.as_batch_data()
        ic_ensemble = ic_data.n_ensemble
        if forcing.n_ensemble == 1 and ic_ensemble > 1:
            if batch_size(forcing) == batch_size(ic_data):
                forcing = dataclasses.replace(forcing, n_ensemble=ic_ensemble)
            else:
                forcing = forcing.broadcast_ensemble(n_ensemble=ic_ensemble)

        prediction, new_initial_condition = self.predict(
            initial_condition,
            forcing,
            False,
            compute_derived_forcings=False,
        )
        forward_data = self.get_forward_data(
            forcing,
            compute_derived_variables=False,
        )
        return (
            PairedData.from_batch_data(
                prediction=prediction,
                reference=BatchData.new_on_device(
                    data=forward_data.data,
                    time=forward_data.time,
                    horizontal_dims=forward_data.horizontal_dims,
                    labels=forward_data.labels,
                    n_ensemble=forward_data.n_ensemble,
                ),
            ),
            new_initial_condition,
        )

    Stepper.predict_paired = predict_paired


def main() -> None:
    args = parse_args()
    patch_predict_paired()
    with Distributed.context():
        inference_main(
            args.yaml_config,
            segments=args.segments,
            override_dotlist=args.override,
        )


if __name__ == "__main__":
    main()
