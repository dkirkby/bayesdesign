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


def test_profile_sync_callback_is_called(tmp_path):
    from bed.benchmark import profile

    called = []

    def work():
        return "result"

    def sync(result):
        called.append(result)

    out = profile(work, out_dir=tmp_path, plot=False, sync=sync)

    assert out["result"] == "result"
    assert called == ["result"]
    assert out["trace_json"].exists()
    assert out["trace_png"] is None


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


def test_plot_sweep_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from bed.benchmark import plot_sweep

    csv_path = tmp_path / "sweep.csv"
    csv_path.write_text(
        "variant,n_param_axis,call_elapsed_s,peak_rss_mb\n"
        "baseline,10,1.0,100.0\n"
        "baseline,20,2.5,140.0\n"
        "candidate,10,0.8,120.0\n"
        "candidate,20,1.7,160.0\n",
        encoding="utf-8",
    )

    output_path = plot_sweep(
        [csv_path],
        ["macbook"],
        tmp_path / "sweep.png",
        group_col="variant",
        colors={"baseline": "tab:blue", "candidate": "tab:orange"},
        linestyles={"baseline": "--", "candidate": "-"},
        alpha=0.6,
        ylabel="Time (s)",
    )

    assert output_path.exists()


def test_plot_helpers_require_matching_labels(tmp_path):
    from bed.benchmark import plot_sweep, plot_timeseries

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
    with pytest.raises(ValueError, match="same length"):
        plot_sweep([csv_path], [], tmp_path / "sweep.png")
