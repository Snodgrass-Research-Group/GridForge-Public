"""io/writers.py — PowerWorld CSV layouts (format-critical) and
Excel reports."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridforge.io.writers import (BRANCHES_REPORT_COLUMNS,
                                  POWERWORLD_LINE_COLUMNS, bus_list_report,
                                  write_network_workbooks,
                                  write_partition_workbooks,
                                  write_powerworld_lines,
                                  write_powerworld_transformers)

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data_store"


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


@pytest.fixture
def xf_model(toy_model):
    """Toy with transformers (skipped when exports are absent)."""
    if not (STORE / "exports" / "XFData.parquet").exists():
        pytest.skip("XFData exports not migrated")
    from gridforge.scale.transformers import add_transformers, load_xf_tables

    xf, xr = load_xf_tables(STORE)
    res = add_transformers(toy_model, xf, xr, np.random.default_rng(7))
    return res.model, res.transformers


class TestPowerWorldLines:
    def test_layout_and_values(self, toy_model, tmp_path):
        p = write_powerworld_lines(toy_model, tmp_path / "lines.csv")
        rows = read_rows(p)
        assert rows[0][0] == "Branch"
        assert all(c == "" for c in rows[0][1:])
        assert rows[1] == POWERWORLD_LINE_COLUMNS
        assert len(rows) == 2 + toy_model.n_branches

        r = rows[2]                                   # branch 0: buses 0→1
        assert [r[0], r[1]] == ["101", "102"]         # NATIONAL numbers
        assert r[3:6] == ["Closed", "Line", "No"]
        cap = float(r[9])
        assert float(r[10]) == pytest.approx(1.05 * cap)
        assert float(r[11]) == pytest.approx(1.1 * cap)
        assert r[12] == "1"                           # tap ratio
        assert float(r[13]) == pytest.approx(round(45.0 / 1.61, 1))  # miles
        assert r[14] == "SizeKcmil" and r[16] == "GMDFt"

    def test_minimal_form_has_14_columns(self, toy_model, tmp_path):
        p = write_powerworld_lines(toy_model, tmp_path / "l.csv", full=False)
        rows = read_rows(p)
        assert rows[1] == POWERWORLD_LINE_COLUMNS[:14]
        assert len(rows[2]) == 14

    def test_transformer_rows_excluded(self, xf_model, tmp_path):
        model, _ = xf_model
        p = write_powerworld_lines(model, tmp_path / "l.csv")
        rows = read_rows(p)
        n_lines = int((~model.branches["is_transformer"]).sum())
        assert len(rows) == 2 + n_lines
        assert all(r[4] == "Line" for r in rows[2:])


class TestPowerWorldTransformers:
    def test_layout_and_ratings(self, xf_model, tmp_path):
        model, xf = xf_model
        p = write_powerworld_transformers(model, xf, tmp_path / "xf.csv")
        rows = read_rows(p)
        assert rows[0][0] == "Branch"
        assert rows[1] == POWERWORLD_LINE_COLUMNS[:14]
        assert len(rows) == 2 + len(xf)
        r = rows[2]
        assert r[4:6] == ["Transformer", "Yes"]
        cap = float(r[9])
        assert float(r[10]) == pytest.approx(1.25 * cap)
        assert float(r[11]) == pytest.approx(1.5 * cap)
        assert float(r[13]) == 0.01                   # nominal length
        # national ids: new buses numbered past the toy's max (110)
        assert max(int(rows[i][1]) for i in range(2, len(rows))) > 110


class TestWorkbooks:
    def test_network_workbooks(self, xf_model, tmp_path):
        model, xf = xf_model
        written = write_network_workbooks(model, tmp_path, transformers=xf)
        names = {p.name for p in written}
        assert names == {"FinalGarverResults.xlsx", "NetworkLines.xlsx",
                         "NetworkTransformers.xlsx",
                         "GeneratorBusMapping.xlsx", "BusList.xlsx"}

        lines = pd.read_excel(tmp_path / "NetworkLines.xlsx")
        assert list(lines.columns) == BRANCHES_REPORT_COLUMNS
        assert len(lines) == int((~model.branches["is_transformer"]).sum())
        assert lines["FromBus"].min() >= 101          # national ids

        xfbook = pd.read_excel(tmp_path / "NetworkTransformers.xlsx")
        assert len(xfbook) == len(xf)
        assert (xfbook["Capacity"] % 50 == 0).all()

        gmap = pd.read_excel(tmp_path / "GeneratorBusMapping.xlsx")
        assert len(gmap) == len(model.generators)

    def test_bus_list_roster(self, toy_model):
        roster = bus_list_report(toy_model)
        # toy: 8 buses touch 138 kV lines, 8 touch 69 kV lines
        assert (roster["Voltage"] == 138).sum() == 8
        assert (roster["Voltage"] == 69).sum() == 8
        assert set(roster.columns) == {"Voltage", "National Bus Num",
                                       "Bus Name", "Lat", "Lon"}

    def test_partition_workbooks(self, toy_model, tmp_path):
        parts = [(np.arange(6), np.arange(5)),
                 (np.arange(6, 12), np.arange(5, 10))]
        written = write_partition_workbooks(toy_model, 138, parts, tmp_path)
        for p in written:
            assert p.exists()
        sheets = pd.read_excel(tmp_path / "LinePartitionResults138.xlsx",
                               sheet_name=None)
        assert set(sheets) == {"Part1", "Part2"}
        assert list(sheets["Part1"].columns) == ["From Bus", "To Bus",
                                                 "Line Length", "Line Limit"]
