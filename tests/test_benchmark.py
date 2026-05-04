import time

import pytest

psutil = pytest.importorskip("psutil")


def test_profile_synthetic_allocation_writes_outputs(tmp_path):
    pytest.importorskip("matplotlib")

    import numpy as np

    from bed.benchmark import profile

    def allocate():
        blocks = []
        for index in range(2):
            block = np.empty(int(4 * (1 << 20)), dtype=np.uint8)
            block[:] = index + 1
            blocks.append(block)
            time.sleep(0.02)
        return len(blocks)

    out = profile(
        allocate,
        out_dir=tmp_path,
        label="tiny allocation",
        sample_interval=0.01,
        metadata={"size_mib": 8},
    )

    trace = out["trace"]
    samples = trace["samples"]
    assert out["result"] == 2
    assert len(samples["t_s"]) == len(samples["rss_mb"])
    assert len(samples["t_s"]) >= 2
    assert all(t2 >= t1 for t1, t2 in zip(samples["t_s"], samples["t_s"][1:]))
    assert trace["peak_rss_mb"] > 0.0
    assert out["trace_json"].exists()
    assert out["trace_png"].exists()
    assert out["summary_csv"].exists()


def test_plot_timeseries_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from bed.benchmark import plot_timeseries

    csv_path = tmp_path / "timeseries.csv"
    csv_path.write_text(
        "time_bin_s,baseline_rss_delta_mb,candidate_rss_delta_mb\n"
        "0.0,0.0,0.0\n"
        "0.1,2.0,1.0\n",
        encoding="utf-8",
    )

    output_path = plot_timeseries(
        [csv_path],
        ["macbook"],
        tmp_path / "timeseries.png",
        value_cols=["candidate_rss_delta_mb"],
        colors=["tab:blue"],
        linestyles=["--"],
        alpha=0.6,
    )

    assert output_path.exists()


def test_plot_timeseries_accepts_in_memory_rows(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from bed.benchmark import plot_timeseries

    monkeypatch.setattr(plt, "show", lambda: None)

    rows = [
        {"time_bin_s": 0.0, "run_a_rss_delta_mb": 0.0},
        {"time_bin_s": 0.1, "run_a_rss_delta_mb": 2.0},
    ]

    output_path = plot_timeseries(
        rows,
        ["run A"],
        tmp_path / "timeseries_rows.png",
        value_cols=["run_a_rss_delta_mb"],
        colors=["tab:blue"],
    )

    assert output_path.exists()
    assert plot_timeseries(rows, value_cols=["run_a_rss_delta_mb"]) is None

    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: None)
    one_point_fig = plot_timeseries([{"time_bin_s": 0.0, "run_a_rss_delta_mb": 0.0}])
    assert one_point_fig is None
    fig = plt.gcf()
    line = fig.axes[0].lines[0]
    assert line.get_marker() == "o"
    original_close(fig)


def test_in_memory_plot_helpers_write_png(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from bed.benchmark import (
        combine_memory_traces
    )

    monkeypatch.setattr(plt, "show", lambda: None)

    trace = {
        "label": "inline",
        "metadata": {"grid_size": 10},
        "call_elapsed_s": 0.2,
        "process_elapsed_s": 0.3,
        "ready_elapsed_s": 0.0,
        "peak_rss_mb": 12.0,
        "samples": {"t_s": [0.0, 0.1], "rss_mb": [10.0, 12.0]},
    }
    timeseries_rows = combine_memory_traces([("n10", trace)])
    assert timeseries_rows[0]["time_bin_s"] == pytest.approx(0.0)
    assert timeseries_rows[-1]["n10_rss_delta_mb"] == pytest.approx(2.0)

    ready_relative_rows = combine_memory_traces([("n10", trace)], relative_to="ready")
    assert ready_relative_rows[0]["time_bin_s"] == pytest.approx(0.0)


def test_plot_helpers_require_matching_labels(tmp_path):
    from bed.benchmark import combine_memory_traces, plot_timeseries

    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="same length"):
        plot_timeseries([csv_path], [], tmp_path / "timeseries.png")
    with pytest.raises(ValueError, match="same length"):
        plot_timeseries([csv_path], ["one"], tmp_path / "timeseries.png", colors=[])
    with pytest.raises(ValueError, match="same length"):
        plot_timeseries([csv_path], ["one"], tmp_path / "timeseries.png", linestyles=[])
    with pytest.raises(ValueError, match="same length"):
        plot_timeseries([csv_path], ["one"], tmp_path / "timeseries.png", alpha=[])
    with pytest.raises(ValueError, match="relative_to"):
        combine_memory_traces([], relative_to="cold_start")
