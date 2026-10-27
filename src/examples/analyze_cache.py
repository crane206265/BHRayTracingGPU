import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

from src.cache.reader import RayHistoryReader


STATUS_NAMES = {0: "ACTIVE", 1: "CAPTURED", 2: "ESCAPED", 3: "MAX_STEPS", 4: "NONFINITE", 5: "INVALID"}
REPO_ROOT = Path(__file__).resolve().parents[2]


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def length_summary(values):
    return {"min": int(values.min()), "median": float(np.median(values)), "max": int(values.max()), "total": int(values.sum())}


def analyze_cache(path, *, chunk_samples=65536, progress_seconds=10.0, progress=None):
    if chunk_samples < 1:
        raise ValueError("chunk_samples must be positive.")
    start_time = time.perf_counter()
    with RayHistoryReader(path) as reader:
        initial, final = reader.read_initial(), reader.read_final()
        B = reader.batch_size
        declared = reader.saved_lengths.astype(np.int64)
        initial_steps = initial.valid_length.astype(np.int64)
        final_steps = final.valid_length.astype(np.int64)
        integrated = final_steps - initial_steps
        counts = np.zeros(B, dtype=np.int64)
        last_steps = initial_steps.copy()
        last_lam = initial.lam.copy()
        last_y = initial.y.copy()
        finite_ok = np.ones(B, dtype=bool)
        order_ok = np.ones(B, dtype=bool)
        bounds_ok = np.ones(B, dtype=bool)
        side = math.isqrt(B)
        representative = (side // 2) * side + side // 2 if side * side == B else B // 2
        center_y = [initial.y[representative:representative + 1]]
        center_lam = [initial.lam[representative:representative + 1]]
        center_steps = [initial_steps[representative:representative + 1]]
        scanned = 0
        next_progress = time.perf_counter() + progress_seconds
        history = reader.file["history"]

        def consume(y, lam, steps, ids):
            nonlocal scanned, next_progress
            if not len(ids):
                return
            starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
            ends = np.r_[starts[1:] - 1, len(ids) - 1]
            rays = ids[starts]
            counts[rays] += ends - starts + 1
            finite = np.isfinite(y).all(axis=1) & np.isfinite(lam)
            finite_ok[np.unique(ids[~finite])] = False
            within = (steps > initial_steps[ids]) & (steps <= final_steps[ids])
            bounds_ok[np.unique(ids[~within])] = False
            first_ok = (steps[starts] > last_steps[rays]) & (lam[starts] < last_lam[rays])
            order_ok[rays[~first_ok]] = False
            same_ray = ids[1:] == ids[:-1]
            pair_ok = (steps[1:] > steps[:-1]) & (lam[1:] < lam[:-1])
            order_ok[np.unique(ids[1:][same_ray & ~pair_ok])] = False
            last_steps[rays] = steps[ends]
            last_lam[rays] = lam[ends]
            last_y[rays] = y[ends]
            selected = ids == representative
            if np.any(selected):
                center_y.append(y[selected].copy())
                center_lam.append(lam[selected].copy())
                center_steps.append(steps[selected].copy())
            scanned += len(ids)
            if progress is not None and time.perf_counter() >= next_progress:
                progress(f"Scanned {scanned:,} saved samples in {time.perf_counter() - start_time:.1f}s")
                next_progress = time.perf_counter() + progress_seconds

        if reader.format_version == 2:
            offsets_dataset = history["block_offsets"]
            for block in range(offsets_dataset.shape[0]):
                offsets = offsets_dataset[block]
                if not np.issubdtype(offsets.dtype, np.integer) or np.any(np.diff(offsets) < 0):
                    raise ValueError(f"Invalid ray offsets in block {block}.")
                for start in range(int(offsets[0]), int(offsets[-1]), chunk_samples):
                    stop = min(start + chunk_samples, int(offsets[-1]))
                    ids = np.searchsorted(offsets, np.arange(start, stop), side="right") - 1
                    steps = history["step"][start:stop]
                    if not np.issubdtype(steps.dtype, np.integer):
                        raise ValueError("Saved step indices must be integers.")
                    consume(history["y"][start:stop], history["lam"][start:stop], steps, ids)
        else:
            scan_steps = max(1, chunk_samples // B)
            accepted = initial_steps.copy()
            for start in range(0, reader.n_steps, scan_steps):
                stop = min(start + scan_steps, reader.n_steps)
                valid = history["valid"][start:stop]
                if valid.dtype != np.dtype("bool"):
                    raise ValueError("Legacy valid mask must be boolean.")
                y = history["y"][start:stop]
                lam = history["lam"][start:stop]
                step_grid = accepted[None, :] + np.cumsum(valid, axis=0, dtype=np.int64)
                rows, ids = np.nonzero(valid)
                order = np.argsort(ids, kind="stable")
                consume(y[rows, ids][order], lam[rows, ids][order], step_grid[rows, ids][order], ids[order])
                accepted += valid.sum(axis=0, dtype=np.int64)

        endpoint_ok = np.ones(B, dtype=bool)
        has_steps = integrated > 0
        endpoint_ok[has_steps] = (
            (counts[has_steps] > 0)
            & (last_steps[has_steps] == final_steps[has_steps])
            & (last_lam[has_steps] == final.lam[has_steps])
            & np.all(last_y[has_steps] == final.y[has_steps], axis=1)
        )
        count_ok = counts == declared
        limits_ok = (counts >= 0) & (counts <= integrated)
        checks = {
            "saved_samples_finite": bool(finite_ok.all()),
            "saved_counts_match": bool(count_ok.all()),
            "sample_order": bool(order_ok.all()),
            "step_indices_in_range": bool(bounds_ok.all()),
            "endpoints_match": bool(endpoint_ok.all()),
            "saved_not_more_than_integrated": bool(limits_ok.all()),
            "final_states_finite": bool(np.isfinite(final.y).all() and np.isfinite(final.lam).all()),
            "no_remaining_active": bool(not np.any(final.active)),
            "known_statuses": bool(np.isin(final.status, list(STATUS_NAMES)).all()),
            "no_numerical_failures": bool(not np.isin(final.status, [4, 5]).any()),
        }
        status_counts = {STATUS_NAMES.get(int(c), str(c)): int(n) for c, n in zip(*np.unique(final.status, return_counts=True))}
        params = dict(reader.file["parameters"].attrs) if "parameters" in reader.file else {}
        y = np.concatenate(center_y)
        lam = np.concatenate(center_lam)
        steps = np.concatenate(center_steps)
        R2 = np.sum(y[:, 1:4] ** 2, axis=1)
        radius = np.sqrt(R2)
        radius_kind, horizon, capture_boundary = "Euclidean", None, None
        if "mass" in params and "spin" in params:
            M, a = float(params["mass"]), float(params["spin"])
            if np.isfinite([M, a]).all() and M > 0 and abs(a) <= M:
                radius = np.sqrt(np.maximum(0.5 * (R2 - a*a + np.hypot(R2 - a*a, 2*a*y[:, 3])), 0.0))
                radius_kind = "Kerr-Schild"
                horizon = M + np.sqrt(M*M - a*a)
                if "epsilon" in params:
                    capture_boundary = horizon + float(params["epsilon"]) * M
        finite_radius = np.where(np.isfinite(radius), radius, np.inf)
        minimum = int(np.argmin(finite_radius))
        indices = sorted(set(range(max(0, minimum - 3), min(len(radius), minimum + 4))) | set(range(max(0, len(radius) - 5), len(radius))))
        samples = [{
            "integration_step": int(steps[i]), "lambda": float(lam[i]),
            "saved_delta_lambda": float(lam[i] - lam[i - 1]) if i else 0.0,
            "radius": float(radius[i]), "max_abs_momentum": float(np.max(np.abs(y[i, 4:]))),
        } for i in indices]
        representative_report = {
            "ray_id": representative,
            "selection": "nearest center pixel (no exact center for an even image width)" if side * side == B else "middle ray index",
            "status": STATUS_NAMES.get(int(final.status[representative]), str(final.status[representative])),
            "radius_kind": radius_kind, "horizon": horizon, "capture_boundary": capture_boundary,
            "minimum_radius": float(np.min(radius)), "final_radius": float(radius[-1]),
            "final_lambda": float(lam[-1]), "samples": samples,
        }
        raw_dense_bytes = reader.n_steps * B * (8 * history["y"].dtype.itemsize + history["lam"].dtype.itemsize + 1)
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cache_path": str(Path(path).resolve()), "format_version": reader.format_version,
            "file_size_bytes": Path(path).stat().st_size, "parameters": params,
            "batch_size": B, "scan_steps": reader.n_steps, "saved_samples": scanned,
            "saved_lengths": counts, "integrated_lengths": integrated,
            "saved_length_summary": length_summary(counts), "integrated_length_summary": length_summary(integrated),
            "estimated_dense_history_bytes": raw_dense_bytes,
            "status_counts": status_counts, "remaining_active": int(np.count_nonzero(final.active)),
            "checks": checks, "all_checks_passed": all(checks.values()),
            "failed_ray_ids": {
                "nonfinite_samples": np.flatnonzero(~finite_ok), "counts": np.flatnonzero(~count_ok),
                "order": np.flatnonzero(~order_ok), "step_bounds": np.flatnonzero(~bounds_ok),
                "endpoints": np.flatnonzero(~endpoint_ok), "length_limits": np.flatnonzero(~limits_ok),
            },
            "representative_ray": representative_report,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        return json_value(report)


def format_report(report):
    if "error" in report:
        return f"Cache analysis failed\nCache: {report['cache_path']}\nError: {report['error']}\n"
    lines = [
        f"Cache: {report['cache_path']}", f"Format: {report['format_version']}",
        f"File size (MB): {report['file_size_bytes'] / 1e6:.3f}",
        f"Rays: {report['batch_size']}", f"Saved samples: {report['saved_samples']:,}",
        f"Saved lengths: {report['saved_length_summary']}",
        f"Integrated lengths: {report['integrated_length_summary']}",
        f"Parameters: {json.dumps(report['parameters'], ensure_ascii=False, sort_keys=True)}",
        "", "Status counts:",
    ]
    lines.extend(f"  {name}: {count}" for name, count in report["status_counts"].items())
    lines.extend(["", "Checks:"])
    lines.extend(f"  {name}: {passed}" for name, passed in report["checks"].items())
    ray = report["representative_ray"]
    lines.extend([
        f"All checks passed: {report['all_checks_passed']}",
        f"Remaining active: {report['remaining_active']}", "",
        f"Representative ray: {ray['ray_id']} ({ray['selection']}) {ray['status']}",
        f"Radius kind: {ray['radius_kind']}",
        f"Horizon: {ray['horizon']}  Capture boundary: {ray['capture_boundary']}",
        f"Minimum/final radius: {ray['minimum_radius']} / {ray['final_radius']}",
        f"Final affine parameter: {ray['final_lambda']}",
        "step | lambda | saved_delta_lambda | radius | max_abs_momentum",
    ])
    lines.extend(" | ".join(str(sample[key]) for key in ("integration_step", "lambda", "saved_delta_lambda", "radius", "max_abs_momentum")) for sample in ray["samples"])
    lines.append(f"Analysis elapsed seconds: {report['elapsed_seconds']:.3f}")
    lines.append("Scope: cache consistency and finiteness; null constraint and conserved quantities are not tested.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Stream a ray cache and save consistency reports without GPU computation.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache", help="Cache path relative to the repository root, or an absolute path.")
    source.add_argument("--cache-output", help="Original writer OUTPUT: relative to repository cache/, or an absolute path.")
    parser.add_argument("--report-prefix", help="Output prefix for .json and .txt reports, relative to the repository root.")
    parser.add_argument("--chunk-samples", type=int, default=65536)
    parser.add_argument("--progress-seconds", type=float, default=10.)
    args = parser.parse_args()
    if args.chunk_samples < 1 or not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0:
        parser.error("chunk-samples and progress-seconds must be positive.")
    path = Path(args.cache or args.cache_output).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / (Path("cache") / path if args.cache_output is not None else path)
    prefix = Path(args.report_prefix).expanduser() if args.report_prefix else path.parent / "reports" / (path.stem + "_analysis")
    if not prefix.is_absolute():
        prefix = REPO_ROOT / prefix
    print(f"Analyzing cache: {path}", flush=True)
    try:
        report = analyze_cache(path, chunk_samples=args.chunk_samples, progress_seconds=args.progress_seconds, progress=lambda text: print(text, flush=True))
        code = 0 if report["all_checks_passed"] else 1
    except Exception as exc:
        traceback.print_exc()
        report = {"cache_path": str(path), "error": f"{type(exc).__name__}: {exc}", "all_checks_passed": False}
        code = 2
    report["analysis_slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path, text_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".txt")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = format_report(report)
    text_path.write_text(summary, encoding="utf-8")
    print(summary, end="", flush=True)
    print(f"Reports: {json_path} | {text_path}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
