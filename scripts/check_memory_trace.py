#!/usr/bin/env python3
"""Synthetic RSS trace check using fixed-size allocations at fixed intervals.

This is a sanity check for our profiling pipeline. It uses the same basic
sampling approach as collect_phase6_cold_memory_trace.py, but the worker does
something much simpler and more predictable:

- allocate a fixed number of MiB every fixed number of seconds
- optionally hold
- optionally release one chunk every fixed number of seconds

The output JSON is compatible with plot_phase6_cold_memory_trace.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import select
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
MIB = float(1 << 20)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument(
        "--time-origin",
        choices=("ready", "process"),
        default="ready",
        help="Whether t=0 starts from worker ready state or process start.",
    )
    parser.add_argument("--step-mib", type=float, default=128.0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--step-seconds", type=float, default=0.5)
    parser.add_argument("--initial-delay-seconds", type=float, default=0.0)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument(
        "--release",
        action="store_true",
        help="Release one chunk per step after the hold interval.",
    )
    parser.add_argument("--label", default="Synthetic fixed-step allocation")
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def run_worker(args):
    # Tell the parent we have finished process startup and imports, then wait
    # for the start signal so the trace can begin from a ready-state baseline.
    print(json.dumps({"status": "ready"}), flush=True)
    sys.stdin.readline()

    step_bytes = int(args.step_mib * MIB)
    allocations = []
    stage_times_s = []
    stage_sizes_mib = []

    t0 = time.perf_counter()
    if args.initial_delay_seconds > 0:
        time.sleep(args.initial_delay_seconds)
    for step in range(args.num_steps):
        block = np.empty(step_bytes, dtype=np.uint8)
        # Touch every page so the allocation is reflected in RSS.
        block[:] = (step % 251) + 1
        allocations.append(block)
        stage_times_s.append(time.perf_counter() - t0)
        stage_sizes_mib.append((step + 1) * args.step_mib)
        time.sleep(args.step_seconds)

    if args.hold_seconds > 0:
        time.sleep(args.hold_seconds)

    if args.release:
        while allocations:
            allocations.pop()
            gc.collect()
            stage_times_s.append(time.perf_counter() - t0)
            stage_sizes_mib.append(len(allocations) * args.step_mib)
            time.sleep(args.step_seconds)

    elapsed = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "status": "done",
                "call_elapsed_s": elapsed,
                "actual_device_kind": "host",
                "expected_schedule": {
                    "step_mib": args.step_mib,
                    "num_steps": args.num_steps,
                    "step_seconds": args.step_seconds,
                    "initial_delay_seconds": args.initial_delay_seconds,
                    "hold_seconds": args.hold_seconds,
                    "release": bool(args.release),
                    "stage_times_s": stage_times_s,
                    "stage_sizes_mib": stage_sizes_mib,
                    "expected_peak_mib": args.step_mib * args.num_steps,
                },
            }
        ),
        flush=True,
    )


def trace_case(args):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--time-origin",
        args.time_origin,
        "--step-mib",
        str(args.step_mib),
        "--num-steps",
        str(args.num_steps),
        "--step-seconds",
        str(args.step_seconds),
        "--initial-delay-seconds",
        str(args.initial_delay_seconds),
        "--hold-seconds",
        str(args.hold_seconds),
        "--label",
        args.label,
    ]
    if args.release:
        cmd.append("--release")

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ps_proc = psutil.Process(proc.pid)
    t0_process = time.perf_counter()
    ready_payload = None
    t_ready = None
    pre_t_s = []
    pre_rss_mb = []
    pre_uss_mb = []
    while True:
        try:
            rss = ps_proc.memory_info().rss / MIB
        except psutil.Error:
            rss = pre_rss_mb[-1] if pre_rss_mb else 0.0
        try:
            uss = ps_proc.memory_full_info().uss / MIB
        except (psutil.Error, AttributeError):
            uss = pre_uss_mb[-1] if pre_uss_mb else rss
        pre_t_s.append(time.perf_counter() - t0_process)
        pre_rss_mb.append(rss)
        pre_uss_mb.append(uss)

        if proc.stdout and select.select([proc.stdout], [], [], args.sample_interval)[0]:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "ready":
                ready_payload = payload
                t_ready = time.perf_counter()
                break

        if proc.poll() is not None:
            break
    if ready_payload is None:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(stderr or "synthetic worker did not emit ready payload")

    if args.time_origin == "process":
        t0 = t0_process
        t_s = pre_t_s
        rss_mb = pre_rss_mb
        uss_mb = pre_uss_mb
    else:
        t0 = time.perf_counter()
        t_s = []
        rss_mb = []
        uss_mb = []

        try:
            rss0 = ps_proc.memory_info().rss / MIB
        except psutil.Error:
            rss0 = 0.0
        try:
            uss0 = ps_proc.memory_full_info().uss / MIB
        except (psutil.Error, AttributeError):
            uss0 = rss0
        t_s.append(0.0)
        rss_mb.append(rss0)
        uss_mb.append(uss0)

    if proc.stdin:
        proc.stdin.write("go\n")
        proc.stdin.flush()
        proc.stdin.close()

    while proc.poll() is None:
        try:
            rss = ps_proc.memory_info().rss / MIB
        except psutil.Error:
            rss = rss_mb[-1] if rss_mb else 0.0
        try:
            uss = ps_proc.memory_full_info().uss / MIB
        except (psutil.Error, AttributeError):
            uss = uss_mb[-1] if uss_mb else rss
        t_s.append(time.perf_counter() - t0)
        rss_mb.append(rss)
        uss_mb.append(uss)
        time.sleep(args.sample_interval)

    try:
        rss = ps_proc.memory_info().rss / MIB
    except psutil.Error:
        rss = rss_mb[-1] if rss_mb else 0.0
    try:
        uss = ps_proc.memory_full_info().uss / MIB
    except (psutil.Error, AttributeError):
        uss = uss_mb[-1] if uss_mb else rss
    t_s.append(time.perf_counter() - t0)
    rss_mb.append(rss)
    uss_mb.append(uss)

    stdout = proc.stdout.read() if proc.stdout else ""
    stderr = proc.stderr.read() if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(stderr or "synthetic worker exited non-zero")

    payload = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
    if payload is None or payload.get("status") != "done":
        raise RuntimeError("synthetic worker did not emit completion payload")

    return {
        "backend": "synthetic",
        "scenario": "synthetic_memory_check",
        "size": int(args.step_mib * args.num_steps),
        "label": args.label,
        "call_elapsed_s": payload["call_elapsed_s"],
        "process_elapsed_s": t_s[-1] if t_s else 0.0,
        "peak_rss_mb": max(rss_mb) if rss_mb else 0.0,
        "peak_uss_mb": max(uss_mb) if uss_mb else 0.0,
        "device_kind": "host",
        "expected_schedule": payload["expected_schedule"],
        "expected_offset_s": (t_ready - t0_process) if (args.time_origin == "process" and t_ready is not None) else 0.0,
        "samples": {
            "t_s": t_s,
            "rss_mb": rss_mb,
            "uss_mb": uss_mb,
        },
    }


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
        return
    if args.output_json is None:
        raise SystemExit("--output-json is required unless --worker is used")

    trace = trace_case(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps([trace], indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
