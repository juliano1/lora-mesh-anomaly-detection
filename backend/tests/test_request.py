"""Parâmetros e grade temporal."""
from datetime import datetime, timedelta

import pytest

NOW = datetime(2026, 10, 8, 12, 0, 0)


def _req(**over):
    from dataset_export.request import ExportRequest
    base = dict(source_id="lab-ufxx", network_id=1, node_ids=(1, 2),
                start=datetime(2026, 9, 1, 0, 0), end=datetime(2026, 9, 1, 2, 0),
                window_seconds=900, step_seconds=420)
    base.update(over)
    return ExportRequest(**base)


def test_parse_utc_requires_offset_and_normalizes():
    from dataset_export.request import parse_utc, iso_z
    assert parse_utc("2026-09-01T03:00:00-03:00") == datetime(2026, 9, 1, 6, 0)
    assert iso_z(datetime(2026, 9, 1, 6, 0, 0, 120)) == "2026-09-01T06:00:00.000120Z"
    with pytest.raises(ValueError, match="fuso"):
        parse_utc("2026-09-01T03:00:00")


def test_grid_is_anchored_on_origin_and_half_open():
    from dataset_export.request import EPOCH
    r = _req()
    grid = r.grid()
    step = timedelta(seconds=420)
    assert all((t - EPOCH) % step == timedelta(0) for t in grid)
    assert grid[0] > r.start and grid[-1] <= r.end
    assert grid[0] - step <= r.start
    assert all(b - a == step for a, b in zip(grid, grid[1:]))


def test_start_on_grid_is_excluded():
    from dataset_export.request import EPOCH
    on_grid = EPOCH + timedelta(seconds=420 * 4257669)
    r = _req(start=on_grid, end=on_grid + timedelta(seconds=840))
    assert r.grid() == [on_grid + timedelta(seconds=420), on_grid + timedelta(seconds=840)]


def test_adjacent_requests_have_disjoint_grids():
    a = _req(end=datetime(2026, 9, 1, 1, 0)).grid()
    b = _req(start=datetime(2026, 9, 1, 1, 0)).grid()
    full = _req().grid()
    assert not set(a) & set(b) and a + b == full


def test_data_start_includes_context_of_first_window():
    r = _req()
    from ai.feature_engineering import BATTERY_LOOKBACK
    assert r.data_start() == r.grid()[0] - timedelta(seconds=900) - BATTERY_LOOKBACK
    assert r.data_start() < r.start
    tiny = _req(start=datetime(2026, 9, 1, 0, 0), end=datetime(2026, 9, 1, 0, 1))
    if not tiny.grid():
        assert tiny.data_start() == tiny.start


@pytest.mark.parametrize("over,msg", [
    ({"source_id": ""}, "source_id"),
    ({"end": datetime(2026, 9, 1, 0, 0)}, "início"),
    ({"end": NOW + timedelta(minutes=1)}, "futuro"),
    ({"window_seconds": 0}, "window_seconds"),
    ({"step_seconds": -1}, "step_seconds"),
    ({"end": datetime(2026, 10, 5)}, "partes"),
    ({"export_kind": "xyz"}, "export_kind"),
])
def test_validate_rejects(over, msg):
    with pytest.raises(ValueError, match=msg):
        _req(**over).validate(now=NOW, max_days=31)


def test_validate_accepts_valid_raw():
    _req().validate(now=NOW, max_days=31)


def test_settings_have_dataset_keys():
    from config import settings
    assert hasattr(settings, "DATASET_SOURCE_ID")
    assert settings.DATA_ORIGIN_TRACKED_SINCE is None or isinstance(settings.DATA_ORIGIN_TRACKED_SINCE, str)
    assert settings.DATASET_EXPORT_MAX_DAYS > 0


def test_all_export_kinds_accepted():
    for kind in ("raw", "features", "full"):
        _req(export_kind=kind).validate(now=NOW, max_days=31)

