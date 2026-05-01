"""Lightweight profiling helpers for ExperimentDesigner workflows."""

import csv as csv_module
import json
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

MIB = float(1 << 20)


def _require_psutil():
    try:
        import psutil  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Benchmarking requires `psutil`. Install with `pip install bayesdesign[benchmark]`."
        ) from exc
    return psutil


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Plotting benchmark traces requires `matplotlib`. "
            "Install with `pip install bayesdesign[benchmark]`."
        ) from exc
    return plt


def _init_nvml():
    try:
        import pynvml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "GPU memory sampling requires `pynvml`. "
            "Install with `pip install bayesdesign[benchmark-gpu]`."
        ) from exc
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def _gpu_mem_for_pid(handle, pid):
    if handle is None:
        return 0.0
    import pynvml  # type: ignore

    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        if proc.pid == pid:
            return (proc.usedGpuMemory or 0) / MIB
    return 0.0


def _jax_bytes_in_use_mb():
    if "jax" not in sys.modules:
        return 0.0
    try:
        import jax  # type: ignore

        stats = jax.devices()[0].memory_stats() or {}
        return float(stats.get("bytes_in_use", 0)) / MIB
    except Exception:
        return 0.0


def _sync_result(result, sync):
    if sync is not None:
        sync(result)
        return
    block = getattr(result, "block_until_ready", None)
    if callable(block):
        block()


def _plot_trace(trace, output):
    plt = _require_matplotlib()
    samples = trace["samples"]
    t_s = samples["t_s"]
    rss_mb = samples["rss_mb"]
    if not t_s or not rss_mb:
        raise RuntimeError("No benchmark samples were collected.")

    base_rss = rss_mb[0]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t_s, [rss - base_rss for rss in rss_mb], linewidth=2.0, label="RSS")
    if max(samples.get("gpu_mb", [0.0])) > 0:
        base_gpu = samples["gpu_mb"][0]
        ax.plot(
            t_s,
            [gpu - base_gpu for gpu in samples["gpu_mb"]],
            linewidth=2.0,
            label="NVML GPU",
        )
    if max(samples.get("jax_gpu_mb", [0.0])) > 0:
        base_jax = samples["jax_gpu_mb"][0]
        ax.plot(
            t_s,
            [gpu - base_jax for gpu in samples["jax_gpu_mb"]],
            linewidth=2.0,
            label="JAX device",
        )
    ax.set_title(trace["label"])
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Memory delta (MiB)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def _write_summary(trace, output):
    metadata = trace.get("metadata", {}) or {}
    row = {
        "label": trace["label"],
        "call_elapsed_s": trace["call_elapsed_s"],
        "process_elapsed_s": trace["process_elapsed_s"],
        "peak_rss_mb": trace["peak_rss_mb"],
        "peak_gpu_mb": trace["peak_gpu_mb"],
        "jax_peak_gpu_mb": trace["jax_peak_gpu_mb"],
    }
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row["metadata:%s" % key] = value

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return output


def _normalize_csv_inputs(csv_paths, labels):
    paths = [Path(path) for path in csv_paths]
    labels = list(labels)
    if len(paths) != len(labels):
        raise ValueError("csv_paths and labels must have the same length.")
    if not paths:
        raise ValueError("At least one CSV path is required.")
    return paths, labels


def _style_by_label(values, labels, name):
    if values is None:
        return {}
    if isinstance(values, dict):
        return dict(values)
    values = list(values)
    if len(values) != len(labels):
        raise ValueError(f"{name} and labels must have the same length.")
    return dict(zip(labels, values))


def _alpha_by_label(alpha, labels):
    if alpha is None:
        return {}
    if isinstance(alpha, dict):
        return dict(alpha)
    if isinstance(alpha, (int, float)):
        return {label: float(alpha) for label in labels}
    alpha = list(alpha)
    if len(alpha) != len(labels):
        raise ValueError("alpha and labels must have the same length.")
    return dict(zip(labels, alpha))


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv_module.DictReader(handle))


def _as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_columns(rows, exclude):
    if not rows:
        return []
    columns = []
    for column in rows[0]:
        if column in exclude:
            continue
        if any(_as_float(row.get(column)) is not None for row in rows):
            columns.append(column)
    return columns


def _selected_columns(value_cols, label, rows, exclude):
    if value_cols is None:
        return _numeric_columns(rows, exclude)
    if isinstance(value_cols, dict):
        columns = value_cols.get(label, [])
        if isinstance(columns, str):
            return [columns]
        return list(columns)
    if isinstance(value_cols, str):
        return [value_cols]
    return list(value_cols)


def _finish_plot(fig, ax, output_path, title, ylabel):
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.clear()
    return output_path


def plot_timeseries(
    csv_paths,
    labels,
    output_path,
    *,
    time_col="time_bin_s",
    value_cols=None,
    colors=None,
    linestyles=None,
    alpha=None,
    ylabel="RSS delta (MiB)",
    title=None,
):
    """Plot one or more benchmark memory time-series CSVs.

    Parameters
    ----------
    csv_paths : sequence of path-like
        CSV files containing a time column and one or more numeric value columns.
    labels : sequence of str
        Legend labels corresponding to ``csv_paths``.
    output_path : path-like
        Destination image path.
    time_col : str
        Name of the time axis column.
    value_cols : sequence of str, optional
        Value columns to plot. Defaults to all numeric columns except ``time_col``.
    colors : sequence of str or dict, optional
        Matplotlib colors corresponding to ``labels``. A dict maps labels to colors.
    linestyles : sequence of str or dict, optional
        Matplotlib line styles corresponding to ``labels``.
    alpha : float or sequence of float or dict, optional
        Line opacity. A scalar applies to every line; a sequence or dict maps to labels.
    ylabel, title : str, optional
        Plot labels.
    """
    paths, labels = _normalize_csv_inputs(csv_paths, labels)
    colors_by_label = _style_by_label(colors, labels, "colors")
    linestyles_by_label = _style_by_label(linestyles, labels, "linestyles")
    alpha_by_label = _alpha_by_label(alpha, labels)
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4.8))

    for path, label in zip(paths, labels):
        rows = _read_csv(path)
        columns = _selected_columns(value_cols, label, rows, {time_col})
        for column in columns:
            points = [
                (_as_float(row.get(time_col)), _as_float(row.get(column)))
                for row in rows
            ]
            points = [(x, y) for x, y in points if x is not None and y is not None]
            if not points:
                continue
            x_values, y_values = zip(*points)
            series_label = label if len(columns) == 1 else f"{label}: {column}"
            ax.plot(
                x_values,
                y_values,
                color=colors_by_label.get(label),
                linestyle=linestyles_by_label.get(label, "-"),
                alpha=alpha_by_label.get(label),
                linewidth=2.0,
                label=series_label,
            )

    ax.set_xlabel("Elapsed time (s)")
    ax.axvline(0.0, color="0.5", linestyle="--", linewidth=1.2)
    return _finish_plot(fig, ax, output_path, title, ylabel)


def plot_sweep(
    csv_paths,
    labels,
    output_path,
    *,
    x_col="n_param_axis",
    y_col="call_elapsed_s",
    group_col="backend",
    colors=None,
    linestyles=None,
    alpha=None,
    ylabel=None,
    title=None,
):
    """Plot benchmark sweep CSVs grouped by backend or another column."""
    paths, labels = _normalize_csv_inputs(csv_paths, labels)
    colors_by_label = _style_by_label(colors, labels, "colors")
    linestyles_by_label = _style_by_label(linestyles, labels, "linestyles")
    alpha_by_label = _alpha_by_label(alpha, labels)
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    grouped = defaultdict(list)
    for path, label in zip(paths, labels):
        rows = _read_csv(path)
        y_columns = _selected_columns(y_col, label, rows, {x_col, group_col})
        for row in _read_csv(path):
            for y_column in y_columns:
                x_value = _as_float(row.get(x_col))
                y_value = _as_float(row.get(y_column))
                if x_value is None or y_value is None:
                    continue
                group = row.get(group_col, "") if group_col else ""
                if label and group:
                    series_label = f"{label}: {group}"
                elif label and len(y_columns) > 1:
                    series_label = f"{label}: {y_column}"
                else:
                    series_label = group or label or y_column
                grouped[series_label].append((x_value, y_value, label, group or y_column))

    for series_label, points in grouped.items():
        points = sorted(points)
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        label = points[0][2]
        group = points[0][3]
        ax.plot(
            x_values,
            y_values,
            color=(
                colors_by_label.get(series_label)
                or colors_by_label.get(group)
                or colors_by_label.get(label)
            ),
            linestyle=(
                linestyles_by_label.get(series_label)
                or linestyles_by_label.get(group)
                or linestyles_by_label.get(label)
                or "-"
            ),
            alpha=(
                alpha_by_label.get(series_label)
                or alpha_by_label.get(group)
                or alpha_by_label.get(label)
            ),
            linewidth=2.0,
            label=series_label,
        )

    ax.set_xlabel(x_col)
    return _finish_plot(fig, ax, output_path, title, ylabel or y_col)


def profile(
    func,
    *,
    out_dir=None,
    label="benchmark",
    sample_interval=0.01,
    plot=True,
    csv=True,
    gpu=False,
    sync=None,
    metadata=None,
):
    """Profile a callable in the current Python process.

    Parameters
    ----------
    func : callable
        Work to benchmark. The elapsed time includes this function and any
        optional synchronization.
    out_dir : path-like, optional
        If provided, write trace.json and optionally trace.png / summary.csv.
    label : str
        Human-readable benchmark label.
    sample_interval : float
        Memory sampling interval in seconds.
    plot : bool
        Write trace.png when out_dir is provided.
    csv : bool
        Write summary.csv when out_dir is provided.
    gpu : bool
        Also sample process GPU memory via NVML. Requires benchmark-gpu extras.
    sync : callable, optional
        Called as sync(result) after func() returns. Use this to include
        asynchronous JAX work, e.g. sync=lambda _: jax.block_until_ready(designer.EIG).
    metadata : dict, optional
        JSON-serializable metadata copied into trace.json and summary.csv.
    """
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive.")

    psutil = _require_psutil()
    process = psutil.Process(os.getpid())
    gpu_handle = _init_nvml() if gpu else None
    metadata = dict(metadata or {})

    samples = {"t_s": [], "rss_mb": [], "gpu_mb": [], "jax_gpu_mb": []}
    stop_event = threading.Event()
    start_time = time.perf_counter()

    def sample_once():
        try:
            rss = process.memory_info().rss / MIB
        except psutil.Error:
            rss = samples["rss_mb"][-1] if samples["rss_mb"] else 0.0
        samples["t_s"].append(float(time.perf_counter() - start_time))
        samples["rss_mb"].append(float(rss))
        samples["gpu_mb"].append(float(_gpu_mem_for_pid(gpu_handle, process.pid)))
        samples["jax_gpu_mb"].append(float(_jax_bytes_in_use_mb()))

    def sample_loop():
        while not stop_event.is_set():
            sample_once()
            time.sleep(sample_interval)

    sample_once()
    sampler = threading.Thread(target=sample_loop)
    sampler.daemon = True
    sampler.start()

    call_start = time.perf_counter()
    try:
        result = func()
        _sync_result(result, sync)
    finally:
        call_elapsed_s = time.perf_counter() - call_start
        stop_event.set()
        sampler.join(timeout=1.0)
        sample_once()

    trace = {
        "label": label,
        "metadata": metadata,
        "call_elapsed_s": float(call_elapsed_s),
        "process_elapsed_s": float(samples["t_s"][-1] if samples["t_s"] else 0.0),
        "peak_rss_mb": float(max(samples["rss_mb"]) if samples["rss_mb"] else 0.0),
        "peak_gpu_mb": float(max(samples["gpu_mb"]) if samples["gpu_mb"] else 0.0),
        "jax_peak_gpu_mb": float(max(samples["jax_gpu_mb"]) if samples["jax_gpu_mb"] else 0.0),
        "samples": samples,
    }

    outputs = {"trace_json": None, "trace_png": None, "summary_csv": None}
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        trace_json = out_path / "trace.json"
        trace_json.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        outputs["trace_json"] = trace_json
        if plot:
            outputs["trace_png"] = _plot_trace(trace, out_path / "trace.png")
        if csv:
            outputs["summary_csv"] = _write_summary(trace, out_path / "summary.csv")

    return {"trace": trace, "result": result, **outputs}
