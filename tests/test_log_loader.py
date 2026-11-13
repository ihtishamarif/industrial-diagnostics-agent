import pandas as pd
import pytest

from diagagent.io.log_loader import DatasetStore, LogFormatError, load_log, summarize


def test_csv_numeric_time_in_ms(tmp_path):
    p = tmp_path / "log.csv"
    pd.DataFrame({"time_ms": [1000, 1010, 1020, 1005], "temp": [20, 21, 22, 20.5],
                  "status": ["ok", "ok", "ok", "ok"]}).to_csv(p, index=False)
    df = load_log(p, time_col="time_ms", time_unit="ms")
    assert list(df.columns) == ["t", "temp"]  # non-numeric column dropped
    assert df["t"].tolist() == pytest.approx([0.0, 0.005, 0.01, 0.02])  # sorted, zero-based


def test_csv_datetime_time(tmp_path):
    p = tmp_path / "log.csv"
    pd.DataFrame({"ts": ["2026-01-01 00:00:00", "2026-01-01 00:00:01", "2026-01-01 00:00:03"],
                  "v": [1, 2, 3]}).to_csv(p, index=False)
    df = load_log(p, time_col="ts")
    assert df["t"].tolist() == [0.0, 1.0, 3.0]
    s = summarize(df)
    assert s["n_samples"] == 3 and s["duration_s"] == 3.0


def test_rename_and_duplicates(tmp_path):
    p = tmp_path / "log.csv"
    pd.DataFrame({"t": [0, 0, 1], "PV_1": [1, 9, 2]}).to_csv(p, index=False)
    df = load_log(p, rename={"PV_1": "pv"})
    assert df["pv"].tolist() == [1, 2]


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_log(tmp_path / "nope.csv")
    p = tmp_path / "log.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(LogFormatError):
        load_log(p, time_col="t")


def test_store():
    s = DatasetStore()
    i = s.add(pd.DataFrame({"t": [0.0, 1.0], "a": [1, 2]}), "test")
    assert s.get(i)["a"].tolist() == [1, 2]
    assert s.list()[0]["dataset_id"] == i
    with pytest.raises(KeyError):
        s.get("ds999")
