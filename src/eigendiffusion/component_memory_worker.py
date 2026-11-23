"""Fresh-process memory worker for the full-field component benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import sys

import numpy as np

from .benchmarking import deep_numpy_nbytes
from .component_benchmark import (
    STANDARD_COMPARATOR_METHODS,
    _apply_readout,
    _run_standard_comparator,
    _standard_transient_core_array_bytes,
    build_full_field_component_pipeline,
    validate_full_field_output,
)
from .config import DiffusionConfig


def _memory_status() -> tuple[int, int]:
    values: dict[str, int] = {}
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _ = line.split()
                values[key.rstrip(":")] = int(value) * 1024
    return values.get("VmRSS", 0), values.get("VmHWM", 0)


def _emit(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, separators=(",", ":")), flush=True)


def _wait_for_parent() -> None:
    if not sys.stdin.readline():
        raise RuntimeError("memory worker lost its parent acknowledgement")


def run_worker(spec: dict[str, object]) -> None:
    config = DiffusionConfig(**spec["config"])
    method = str(spec.get("method", "completed_modal"))
    gc.collect()
    rss, hwm = _memory_status()
    _emit("ready", rss_bytes=rss, hwm_bytes=hwm, method=method)
    _wait_for_parent()

    _emit("setup_start")
    _wait_for_parent()
    if method in STANDARD_COMPARATOR_METHODS:
        pipeline = None
        setup_array_bytes = 0
        handoff_time = None
    elif method == "completed_modal":
        pipeline = build_full_field_component_pipeline(
            config,
            retained_modes=int(spec["retained_modes"]),
            handoff_time=float(spec["handoff_time"]),
            completion_rank=int(spec["completion_rank"]),
            completion_ridge=float(spec["completion_ridge"]),
        )
        setup_array_bytes = deep_numpy_nbytes(pipeline)
        handoff_time = pipeline.simulator.handoff_time
    else:
        raise ValueError(f"unknown memory-worker method: {method}")
    gc.collect()
    rss, hwm = _memory_status()
    _emit(
        "setup_end",
        rss_bytes=rss,
        hwm_bytes=hwm,
        handoff_time=handoff_time,
        setup_array_bytes=setup_array_bytes,
        setup_applicable=method not in STANDARD_COMPARATOR_METHODS,
    )
    _wait_for_parent()

    _emit("trajectory_start")
    _wait_for_parent()
    if method in STANDARD_COMPARATOR_METHODS:
        counts = _run_standard_comparator(
            method,
            config,
            np.random.default_rng(int(spec["seed"])),
        )
        resident_array_bytes = counts.nbytes
        output_array_bytes = counts.nbytes
        returned_result_array_bytes = counts.nbytes
        transient_core_array_bytes = _standard_transient_core_array_bytes(
            method,
            config,
        )
        output_dtype = str(counts.dtype)
    else:
        if pipeline is None:  # pragma: no cover - guarded by method branch
            raise RuntimeError("completed modal worker omitted its pipeline")
        readout = str(spec["readout"])
        simulation_sequence, completion_sequence = np.random.SeedSequence(
            int(spec["seed"])
        ).spawn(2)
        raw = pipeline.simulator.run(rng=np.random.default_rng(simulation_sequence))
        completed, additions = pipeline.completer.complete(
            raw.spatial_counts,
            rng=np.random.default_rng(completion_sequence),
        )
        physical = _apply_readout(completed, config.n_particles, readout)
        counts = physical.counts
        resident_array_bytes = deep_numpy_nbytes(
            (pipeline, raw, completed, additions, physical)
        )
        output_array_bytes = physical.counts.nbytes
        returned_result_array_bytes = deep_numpy_nbytes(physical)
        transient_core_array_bytes = 0
        output_dtype = str(physical.counts.dtype)
    rss, hwm = _memory_status()
    _emit(
        "trajectory_end",
        rss_bytes=rss,
        hwm_bytes=hwm,
        resident_array_bytes=resident_array_bytes,
        output_array_bytes=output_array_bytes,
        returned_result_array_bytes=returned_result_array_bytes,
        transient_core_array_bytes=transient_core_array_bytes,
        output_dtype=output_dtype,
    )
    _wait_for_parent()

    validate_full_field_output(
        counts,
        config,
        require_integer=(
            method in STANDARD_COMPARATOR_METHODS
            or str(spec["readout"]) == "delta_sigma_temporal"
        ),
    )
    _emit("done")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    try:
        run_worker(json.loads(args.spec))
    except BaseException as error:  # pragma: no cover - parent reports child failure
        _emit("error", message=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
