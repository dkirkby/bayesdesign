"""Lightweight profiling helpers for ExperimentDesigner workflows."""

import csv as csv_module
import json
import os
import re
import sys
import threading
import time
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


def _safe_column_label(label):
    label = re.sub(r"[^0-9A-Za-z]+", "_", str(label)).strip("_").lower()
    return label or "trace"


def _labeled_traces(traces):
    counts = {}
    for item in traces:
        if isinstance(item, tuple) and len(item) == 2:
            label, trace = item
        else:
            trace = item
            label = trace.get("label", "trace")
        label = str(label)
        safe_label = _safe_column_label(label)
        counts[safe_label] = counts.get(safe_label, 0) + 1
        if counts[safe_label] > 1:
            safe_label = f"{safe_label}_{counts[safe_label]}"
        yield safe_label, trace


def combine_memory_traces(
    traces,
    *,
    time_bin_s=0.1,
    sample_key="rss_mb",
    relative_to="start",
):
    """Combine multiple trace memory series into one wide, binned table.

    The result is a list of dictionaries with ``time_bin_s`` plus one
    ``<label>_<metric>_delta_mb`` column per trace. Pass ``[(label, trace), ...]``
    to control column names; otherwise each trace's ``label`` is used.
    """
    if time_bin_s <= 0:
        raise ValueError("time_bin_s must be positive.")
    if relative_to not in {"start", "ready"}:
        raise ValueError("relative_to must be 'start' or 'ready'.")

    series_by_column = {}
    for safe_label, trace in _labeled_traces(traces):
        samples = trace["samples"]
        t_values = samples["t_s"]
        sample_values = samples[sample_key]
        if not t_values or not sample_values:
            continue

        if relative_to == "ready":
            origin_s = float(trace.get("ready_elapsed_s", t_values[0]))
            origin_index = next(
                (index for index, t_s in enumerate(t_values) if float(t_s) >= origin_s),
                len(t_values) - 1,
            )
        else:
            origin_index = 0
        origin_t_s = float(t_values[origin_index])
        origin_value = float(sample_values[origin_index])

        binned = {}
        for t_s, value in zip(t_values, sample_values):
            relative_t_s = float(t_s) - origin_t_s
            bin_index = int(relative_t_s // time_bin_s)
            delta = float(value) - origin_value
            if bin_index not in binned:
                binned[bin_index] = delta
            elif bin_index < 0:
                binned[bin_index] = min(binned[bin_index], delta)
            else:
                binned[bin_index] = max(binned[bin_index], delta)

        column = f"{safe_label}_{sample_key.removesuffix('_mb')}_delta_mb"
        series_by_column[column] = binned

    if not series_by_column:
        return []

    columns = list(series_by_column)
    min_bin = min(min(series) for series in series_by_column.values())
    max_bin = max(max(series) for series in series_by_column.values())
    rows = []
    for bin_index in range(min_bin, max_bin + 1):
        row = {"time_bin_s": bin_index * time_bin_s}
        for column in columns:
            row[column] = series_by_column[column].get(bin_index, "")
        rows.append(row)
    return rows


def _display_or_save(fig, output_path):
    if output_path is None:
        fig.tight_layout()
        plt = _require_matplotlib()
        plt.show()
        plt.close(fig)
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt = _require_matplotlib()
    plt.close(fig)
    return output_path


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
    return _display_or_save(fig, output_path)


def _is_row_table(data):
    return isinstance(data, (list, tuple)) and (
        not data or isinstance(data[0], dict)
    )


def _plot_timeseries_rows(
    ax,
    rows,
    *,
    label,
    time_col,
    value_cols,
    colors_by_label,
    linestyles_by_label,
    alpha_by_label,
):
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
        series_label = label if label and len(columns) == 1 else column
        if label and len(columns) > 1:
            series_label = f"{label}: {column}"
        ax.plot(
            x_values,
            y_values,
            color=colors_by_label.get(label) or colors_by_label.get(series_label),
            linestyle=(
                linestyles_by_label.get(label)
                or linestyles_by_label.get(series_label)
                or "-"
            ),
            marker="o" if len(points) == 1 else None,
            alpha=alpha_by_label.get(label) or alpha_by_label.get(series_label),
            linewidth=2.0,
            label=series_label,
        )


def plot_timeseries(
    data,
    labels=None,
    output_path=None,
    *,
    time_col="time_bin_s",
    value_cols=None,
    colors=None,
    linestyles=None,
    alpha=None,
    ylabel="RSS delta (MiB)",
    title=None,
):
    """Plot benchmark memory time series from CSVs or in-memory rows.

    Parameters
    ----------
    data : sequence
        Either CSV paths or an in-memory list of row dictionaries, such as the
        output of ``combine_memory_traces(...)``.
    labels : sequence of str, optional
        For CSV inputs, legend labels corresponding to paths. For in-memory
        rows, optional labels corresponding to selected value columns.
    output_path : path-like
        Destination image path. If omitted, display inline with ``plt.show()``.
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
    if _is_row_table(data):
        rows = list(data)
        labels = list(labels or [])
        columns = _selected_columns(value_cols, "", rows, {time_col})
        if labels and len(labels) != len(columns):
            raise ValueError("labels and value columns must have the same length.")
        if labels:
            value_cols = dict(zip(labels, columns))
    else:
        paths, labels = _normalize_csv_inputs(data, labels or [])

    colors_by_label = _style_by_label(colors, labels, "colors")
    linestyles_by_label = _style_by_label(linestyles, labels, "linestyles")
    alpha_by_label = _alpha_by_label(alpha, labels)
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4.8))

    if _is_row_table(data):
        rows = list(data)
        if labels:
            for label in labels:
                _plot_timeseries_rows(
                    ax,
                    rows,
                    label=label,
                    time_col=time_col,
                    value_cols=value_cols,
                    colors_by_label=colors_by_label,
                    linestyles_by_label=linestyles_by_label,
                    alpha_by_label=alpha_by_label,
                )
        else:
            _plot_timeseries_rows(
                ax,
                rows,
                label="",
                time_col=time_col,
                value_cols=value_cols,
                colors_by_label=colors_by_label,
                linestyles_by_label=linestyles_by_label,
                alpha_by_label=alpha_by_label,
            )
    else:
        for path, label in zip(paths, labels):
            _plot_timeseries_rows(
                ax,
                _read_csv(path),
                label=label,
                time_col=time_col,
                value_cols=value_cols,
                colors_by_label=colors_by_label,
                linestyles_by_label=linestyles_by_label,
                alpha_by_label=alpha_by_label,
            )

    ax.set_xlabel("Elapsed time (s)")
    ax.axvline(0.0, color="0.5", linestyle="--", linewidth=1.2)
    return _finish_plot(fig, ax, output_path, title, ylabel)


def profile(
    func,
    *,
    out_dir=None,
    label="benchmark",
    sample_interval=0.01,
    plot=True,
    csv=True,
    gpu=False,
    metadata=None,
):
    """Profile a callable in the current Python process.

    Parameters
    ----------
    func : callable
        Work to benchmark.
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
