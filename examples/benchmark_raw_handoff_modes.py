"""Reproduce the reduced-mode report's fixed 10 us raw-handoff benchmark.

The modal rows use ``HandoffCorrelatedModalDiffusion.run_minimal``: every draw
returns all 101 spatial fields plus its retained modal history, with no
completion or physical readout. The exact row uses the vectorized integer
count walk. Memory is measured in one fresh process per method while the
parent samples RSS during synchronized setup and draw phases.

Run from the repository root::

    python3 examples/benchmark_raw_handoff_modes.py
"""

from __future__ import annotations

import os

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import csv
import gc
import json
import selectors
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from eigendiffusion.baselines import vectorized_multinomial_random_walk_diffusion
from eigendiffusion.benchmarking import deep_numpy_nbytes
from eigendiffusion.config import DiffusionConfig
from eigendiffusion.correlated_modal import HandoffCorrelatedModalDiffusion

OUTPUT_DIR = Path("outputs/reduced_mode_report")
SAMPLES_CSV = "raw_handoff_performance_samples.csv"
SUMMARY_CSV = "raw_handoff_performance_summary.csv"
MEMORY_CSV = "raw_handoff_memory.csv"
SUMMARY_JSON = "raw_handoff_performance_summary.json"
SEED = 20261125
MEMORY_SAMPLE_MS = 0.5
MEMORY_TIMEOUT_S = 60.0


@dataclass(frozen=True, slots=True)
class Method:
    name: str
    label: str
    modes: int | None

    @property
    def exact(self) -> bool:
        return self.modes is None


METHODS = (
    Method("raw_101_to_50", "Raw handoff 101->50", 50),
    Method("raw_101_to_75", "Raw handoff 101->75", 75),
    Method("raw_101_to_101", "Raw handoff 101->101", 101),
    Method("exact_vectorized_count_walk", "Exact vectorized count walk", None),
)
CONFIG = DiffusionConfig(
    n_particles=5_275,
    n_nodes=101,
    n_steps=101,
    impulse_index=59,  # 2.36 um on a 4 um line.
    dt=1.0,
    length=4.0,
    diffusion_coefficient=2.20e-4,
)


def _method(name: str) -> Method:
    return next(method for method in METHODS if method.name == name)


def _prepare(method: Method) -> HandoffCorrelatedModalDiffusion | None:
    if method.exact:
        return None
    return HandoffCorrelatedModalDiffusion(
        CONFIG,
        initial_n_modes=101,
        final_n_modes=int(method.modes),
        handoff_time=10.0,
    )


def _draw(method: Method, prepared: object | None, seed: int) -> object:
    rng = np.random.default_rng(seed)
    if method.exact:
        return vectorized_multinomial_random_walk_diffusion(CONFIG, rng=rng)
    if prepared is None:  # pragma: no cover - internal contract
        raise RuntimeError("modal draw is missing its prepared simulator")
    return prepared.run_minimal(rng=rng)


def _counts(method: Method, result: object) -> np.ndarray:
    return np.asarray(result if method.exact else result.spatial_counts)


def _validate(method: Method, result: object) -> None:
    counts = _counts(method, result)
    if counts.shape != (101, 101) or not np.all(np.isfinite(counts)):
        raise RuntimeError(f"invalid full-field output from {method.name}")
    if not np.allclose(counts.sum(axis=1), 5_275, atol=1.0e-7, rtol=0.0):
        raise RuntimeError(f"mass conservation failed for {method.name}")
    if method.exact and (counts.dtype.kind not in "iu" or np.any(counts < 0)):
        raise RuntimeError("exact walk did not return nonnegative integers")
    if not method.exact:
        retained = np.asarray(result.retained_modal_amplitudes)
        if retained.shape != (101, int(method.modes)):
            raise RuntimeError(f"invalid retained history from {method.name}")


def _order(repeat: int) -> tuple[Method, ...]:
    items = list(METHODS)
    offset = repeat % len(items)
    items = items[offset:] + items[:offset]
    if (repeat // len(items)) % 2:
        items.reverse()
    return tuple(items)


def _sample(
    phase: str,
    repeat: int,
    order_index: int,
    method: Method,
    elapsed_ns: int,
    seed: int | None,
) -> dict[str, object]:
    return {
        "phase": phase,
        "repeat": repeat,
        "order_index": order_index,
        "method": method.name,
        "label": method.label,
        "initial_modes": "" if method.exact else 101,
        "final_modes": "" if method.exact else method.modes,
        "handoff_time_us": "" if method.exact else 10.0,
        "seed": "" if seed is None else seed,
        "elapsed_seconds": elapsed_ns / 1.0e9,
    }


def _timings(
    setup_repeats: int,
    simulation_repeats: int,
    combined_repeats: int,
    warmups: int,
) -> list[dict[str, object]]:
    for warmup in range(warmups):
        for method in METHODS:
            prepared = _prepare(method)
            result = _draw(method, prepared, SEED - 10_000 - warmup)
            _validate(method, result)
            del result, prepared
        gc.collect()

    rows: list[dict[str, object]] = []
    for repeat in range(setup_repeats):
        for order_index, method in enumerate(_order(repeat)):
            gc.collect()
            start = perf_counter_ns()
            prepared = _prepare(method)
            elapsed = 0 if method.exact else perf_counter_ns() - start
            rows.append(_sample("setup", repeat, order_index, method, elapsed, None))
            del prepared

    prepared = {method.name: _prepare(method) for method in METHODS}
    gc.collect()
    for repeat in range(simulation_repeats):
        seed = SEED + repeat
        for order_index, method in enumerate(_order(repeat)):
            start = perf_counter_ns()
            result = _draw(method, prepared[method.name], seed)
            elapsed = perf_counter_ns() - start
            _validate(method, result)
            rows.append(
                _sample(
                    "prepared_full_field_simulation",
                    repeat,
                    order_index,
                    method,
                    elapsed,
                    seed,
                )
            )
            del result
    del prepared
    gc.collect()

    for repeat in range(combined_repeats):
        seed = SEED + 100_000 + repeat
        for order_index, method in enumerate(_order(repeat)):
            gc.collect()
            start = perf_counter_ns()
            prepared_method = _prepare(method)
            result = _draw(method, prepared_method, seed)
            elapsed = perf_counter_ns() - start
            _validate(method, result)
            rows.append(
                _sample(
                    "setup_plus_full_field_simulation",
                    repeat,
                    order_index,
                    method,
                    elapsed,
                    seed,
                )
            )
            del result, prepared_method
    return rows


def _summaries(samples: list[dict[str, object]]) -> list[dict[str, object]]:
    phases = {
        "setup": "setup",
        "prepared_full_field_simulation": "prepared_simulation",
        "setup_plus_full_field_simulation": "setup_plus_simulation",
    }
    rows: list[dict[str, object]] = []
    for method in METHODS:
        row: dict[str, object] = {
            "method": method.name,
            "label": method.label,
            "initial_modes": "" if method.exact else 101,
            "final_modes": "" if method.exact else method.modes,
            "handoff_time_us": "" if method.exact else 10.0,
            "n_particles": 5_275,
            "n_nodes": 101,
            "n_steps": 101,
            "dt_us": 1.0,
            "total_time_us": 100.0,
            "impulse_index": 59,
            "impulse_position_um": 2.36,
        }
        for phase, prefix in phases.items():
            values = [
                float(item["elapsed_seconds"])
                for item in samples
                if item["phase"] == phase and item["method"] == method.name
            ]
            row.update(
                {
                    f"{prefix}_repeats": len(values),
                    f"{prefix}_median_seconds": statistics.median(values),
                    f"{prefix}_mean_seconds": statistics.fmean(values),
                    f"{prefix}_std_seconds": statistics.pstdev(values),
                    f"{prefix}_min_seconds": min(values),
                    f"{prefix}_max_seconds": max(values),
                }
            )
        row["independent_median_setup_plus_prepared_seconds"] = (
            float(row["setup_median_seconds"])
            + float(row["prepared_simulation_median_seconds"])
        )
        rows.append(row)
    return rows


def _memory_status() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, _unit = line.split()
            values[key.rstrip(":")] = int(value) * 1024
    return values.get("VmRSS", 0), values.get("VmHWM", 0)


def _emit(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, separators=(",", ":")), flush=True)


def _memory_worker(method_name: str, seed: int) -> int:
    """Child half of the synchronized setup/draw RSS measurement."""

    try:
        method = _method(method_name)
        gc.collect()
        rss, hwm = _memory_status()
        _emit("ready", rss_bytes=rss, hwm_bytes=hwm)
        input()
        _emit("setup_start")
        input()
        prepared = _prepare(method)
        gc.collect()
        rss, hwm = _memory_status()
        _emit(
            "setup_end",
            rss_bytes=rss,
            hwm_bytes=hwm,
            logical_setup_array_bytes=deep_numpy_nbytes(prepared),
        )
        input()
        _emit("draw_start")
        input()
        result = _draw(method, prepared, seed)
        _validate(method, result)
        counts = _counts(method, result)
        gc.collect()
        rss, hwm = _memory_status()
        _emit(
            "draw_end",
            rss_bytes=rss,
            hwm_bytes=hwm,
            logical_output_array_bytes=counts.nbytes,
            logical_returned_result_array_bytes=deep_numpy_nbytes(result),
            logical_setup_plus_result_array_bytes=deep_numpy_nbytes((prepared, result)),
            output_dtype=str(counts.dtype),
        )
        input()
        _emit("done")
        return 0
    except BaseException as error:  # pragma: no cover - parent reports worker error
        _emit("error", message=f"{type(error).__name__}: {error}")
        return 1


def _process_rss(pid: int) -> int:
    try:
        pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return 0


def _memory_row(method: Method, seed: int) -> dict[str, object]:
    """Sample setup and draw RSS in one fresh worker for this method."""

    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--memory-worker", method.name, str(seed)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    if process.stdin is None or process.stdout is None:  # pragma: no cover
        raise RuntimeError("failed to open memory-worker pipes")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    messages: dict[str, dict[str, object]] = {}
    peaks = {"setup": 0, "draw": 0}
    sample_counts = {"setup": 0, "draw": 0}
    phase = ""
    deadline = perf_counter() + MEMORY_TIMEOUT_S
    try:
        while "done" not in messages:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError("memory worker timed out")
            events = selector.select(min(MEMORY_SAMPLE_MS / 1_000.0, remaining))
            if not events:
                if phase:
                    peaks[phase] = max(peaks[phase], _process_rss(process.pid))
                    sample_counts[phase] += 1
                continue
            message = json.loads(process.stdout.readline())
            event = str(message["event"])
            if event == "error":
                raise RuntimeError(str(message["message"]))
            messages[event] = message
            if event.endswith("_start"):
                phase = event.removesuffix("_start")
                peaks[phase] = max(peaks[phase], _process_rss(process.pid))
            elif event in {"setup_end", "draw_end"}:
                ended_phase = event.removesuffix("_end")
                peaks[ended_phase] = max(peaks[ended_phase], int(message["rss_bytes"]))
                phase = ""
            if event != "done":
                process.stdin.write("continue\n")
                process.stdin.flush()
        return_code = process.wait(timeout=max(0.1, deadline - perf_counter()))
        if return_code:
            stderr = "" if process.stderr is None else process.stderr.read()
            raise RuntimeError(f"memory worker exited {return_code}: {stderr}")
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        selector.close()
        process.stdin.close()
        process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    ready, setup, draw = messages["ready"], messages["setup_end"], messages["draw_end"]
    baseline = int(ready["rss_bytes"])
    steady_setup = int(setup["rss_bytes"])
    combined_peak = max(peaks.values())
    return {
        "method": method.name,
        "label": method.label,
        "initial_modes": "" if method.exact else 101,
        "final_modes": "" if method.exact else method.modes,
        "handoff_time_us": "" if method.exact else 10.0,
        "baseline_rss_bytes": baseline,
        "baseline_peak_rss_bytes": int(ready["hwm_bytes"]),
        "isolated_setup_phase_peak_rss_bytes": peaks["setup"],
        "steady_setup_rss_bytes": steady_setup,
        "isolated_setup_peak_increase_bytes": max(0, peaks["setup"] - baseline),
        "process_peak_after_setup_rss_bytes": int(setup["hwm_bytes"]),
        "isolated_draw_phase_peak_rss_bytes": peaks["draw"],
        "final_rss_bytes": int(draw["rss_bytes"]),
        "isolated_draw_peak_increase_over_setup_bytes": max(0, peaks["draw"] - steady_setup),
        "isolated_combined_peak_rss_bytes": combined_peak,
        "isolated_combined_peak_increase_bytes": max(0, combined_peak - baseline),
        "process_peak_after_draw_rss_bytes": int(draw["hwm_bytes"]),
        "logical_setup_array_bytes": int(setup["logical_setup_array_bytes"]),
        "logical_output_array_bytes": int(draw["logical_output_array_bytes"]),
        "logical_returned_result_array_bytes": int(draw["logical_returned_result_array_bytes"]),
        "logical_setup_plus_result_array_bytes": int(draw["logical_setup_plus_result_array_bytes"]),
        "output_dtype": str(draw["output_dtype"]),
        "setup_phase_samples": sample_counts["setup"],
        "draw_phase_samples": sample_counts["draw"],
        "sampler_interval_ms": MEMORY_SAMPLE_MS,
    }


def _comparisons(rows: list[dict[str, object]]) -> dict[str, object]:
    by_name = {str(row["method"]): row for row in rows}

    def compare(candidate: str, baseline: str) -> dict[str, float | None]:
        values: dict[str, float | None] = {}
        for metric in (
            "setup_median_seconds",
            "prepared_simulation_median_seconds",
            "setup_plus_simulation_median_seconds",
        ):
            candidate_value = float(by_name[candidate][metric])
            baseline_value = float(by_name[baseline][metric])
            key = metric.removesuffix("_seconds")
            values[f"{key}_difference_seconds"] = candidate_value - baseline_value
            ratio = None if baseline_value == 0 else candidate_value / baseline_value
            values[f"{key}_ratio"] = ratio
            values[f"{key}_percent_change"] = None if ratio is None else 100 * (ratio - 1)
        return values

    pairs = {
        "raw_101_to_75_vs_raw_101_to_50": ("raw_101_to_75", "raw_101_to_50"),
        "raw_101_to_50_vs_raw_101_to_101": ("raw_101_to_50", "raw_101_to_101"),
        "raw_101_to_75_vs_raw_101_to_101": ("raw_101_to_75", "raw_101_to_101"),
        "raw_101_to_50_vs_exact": ("raw_101_to_50", "exact_vectorized_count_walk"),
        "raw_101_to_75_vs_exact": ("raw_101_to_75", "exact_vectorized_count_walk"),
    }
    return {name: compare(*pair) for name, pair in pairs.items()}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--memory-worker":
        return _memory_worker(sys.argv[2], int(sys.argv[3]))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--setup-repeats", type=int, default=15)
    parser.add_argument("--simulation-repeats", type=int, default=30)
    parser.add_argument("--combined-repeats", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.setup_repeats, args.simulation_repeats, args.combined_repeats) <= 0:
        parser.error("repeat counts must be positive")
    if args.warmups < 0:
        parser.error("warmups must be nonnegative")

    samples = _timings(
        args.setup_repeats,
        args.simulation_repeats,
        args.combined_repeats,
        args.warmups,
    )
    summaries = _summaries(samples)
    memory = [
        _memory_row(method, SEED + 200_000 + index)
        for index, method in enumerate(METHODS)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / SAMPLES_CSV, samples)
    _write_csv(args.output_dir / SUMMARY_CSV, summaries)
    _write_csv(args.output_dir / MEMORY_CSV, memory)
    payload = {
        "schema_version": 1,
        "benchmark": "raw_handoff_fixed_10us_101_nodes",
        "config": {
            **asdict(CONFIG),
            "total_time_us": 100.0,
            "impulse_position_um": 2.36,
        },
        "protocol": {
            "modal_api": "HandoffCorrelatedModalDiffusion.run_minimal",
            "exact_api": "vectorized_multinomial_random_walk_diffusion",
            "handoff_time_us": 10.0,
            "setup_repeats": args.setup_repeats,
            "prepared_simulation_repeats": args.simulation_repeats,
            "setup_plus_simulation_repeats": args.combined_repeats,
            "warmups": args.warmups,
            "seed": SEED,
            "method_order": "cyclic rotation, reversed every four repeats",
            "numerical_threads": 1,
            "memory_method": (
                "one fresh worker per method; parent samples /proc/<pid>/statm "
                "during synchronized setup and draw phases"
            ),
            "memory_sampler_interval_ms": MEMORY_SAMPLE_MS,
        },
        "timing_summary": summaries,
        "comparisons": _comparisons(summaries),
        "memory": memory,
        "artifacts": {
            "timing_samples_csv": SAMPLES_CSV,
            "timing_summary_csv": SUMMARY_CSV,
            "memory_csv": MEMORY_CSV,
        },
    }
    (args.output_dir / SUMMARY_JSON).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for row in summaries:
        print(
            f"{row['method']}: setup={1e3 * float(row['setup_median_seconds']):.3f} ms, "
            f"prepared={1e3 * float(row['prepared_simulation_median_seconds']):.3f} ms, "
            f"combined={1e3 * float(row['setup_plus_simulation_median_seconds']):.3f} ms"
        )
    print(f"wrote {args.output_dir / SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
