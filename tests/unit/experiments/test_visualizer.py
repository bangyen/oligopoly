"""Tests for the experiment result visualizer."""

import csv
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from sim.experiments.visualizer import ExperimentVisualizer

pytest.importorskip("pandas")
pytest.importorskip("plotly")


def _write_results(path: Path, with_cartel: bool) -> None:
    rows = [
        {
            "config_id": cid,
            "model": "cournot",
            "num_firms": 2,
            "mean_profit_per_firm": profit,
            "avg_hhi": 5000.0,
            "avg_price": price,
            "cartel_duration": 3 if with_cartel else 0,
            "total_defections": 1,
        }
        for cid, profit, price in [("a", 100.0, 40.0), ("b", 80.0, 35.0)]
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(("with_cartel", "expected"), [(False, 2), (True, 3)])
def test_generate_all_plots(tmp_path: Path, with_cartel: bool, expected: int) -> None:
    csv_path = tmp_path / "results.csv"
    _write_results(csv_path, with_cartel)

    paths = ExperimentVisualizer(str(tmp_path / "plots")).generate_all_plots(
        str(csv_path)
    )

    assert len(paths) == expected
    for path in paths:
        assert Path(path).read_text().lstrip().lower().startswith("<html")


def test_missing_viz_extra_gives_install_hint(tmp_path: Path) -> None:
    with patch.dict(sys.modules, {"plotly.express": None}):
        with pytest.raises(ImportError, match=r"oligopoly\[viz\]"):
            ExperimentVisualizer(str(tmp_path))
