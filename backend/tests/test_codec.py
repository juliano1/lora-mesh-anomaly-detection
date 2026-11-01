"""Serialização sem perda e RFC 4180."""
import csv
import io
from datetime import datetime

import pytest


def _col(table, name):
    import models  # noqa: F401
    from sqlmodel import SQLModel
    return SQLModel.metadata.tables[table].c[name]


@pytest.mark.parametrize("table,col,value,text", [
    ("telemetry", "snr", 7.1, "7.1"),
    ("telemetry", "snr", 1 / 3, repr(1 / 3)),
    ("telemetry", "rssi", -83, "-83"),
    ("neighbor", "is_parent", True, "1"),
    ("neighbor", "is_parent", False, "0"),
    ("telemetry", "timestamp", datetime(2026, 9, 1, 0, 0, 20, 123456), "2026-09-01T00:00:20.123456Z"),
    ("telemetry", "payload_raw", 'a,"b"\nc', 'a,"b"\nc'),
])
def test_encode_decode_roundtrip(table, col, value, text):
    from dataset_export.codec import encode, decode
    c = _col(table, col)
    assert encode(value, c) == text
    assert decode(text, c) == value


def test_null_handling():
    from dataset_export.codec import encode, decode
    assert encode(None, _col("telemetry", "temperature")) is None
    assert decode(None, _col("telemetry", "temperature")) is None
    with pytest.raises(ValueError, match="obrigatória"):
        decode(None, _col("telemetry", "rssi"))


def test_non_finite_float_rejected():
    from dataset_export.codec import encode
    with pytest.raises(ValueError, match="não finito"):
        encode(float("nan"), _col("telemetry", "snr"))


def test_bad_boolean_and_naive_datetime_rejected():
    from dataset_export.codec import decode
    with pytest.raises(ValueError):
        decode("true", _col("neighbor", "is_parent"))
    with pytest.raises(ValueError):
        decode("2026-09-01T00:00:00", _col("neighbor", "timestamp"))


def test_quote_notnull_distinguishes_null_from_empty_string():
    from dataset_export.codec import csv_writer, read_csv
    buf = io.StringIO()
    csv_writer(buf).writerow(["", None, 'x,"y"\nz', "1"])
    rows = list(read_csv(io.StringIO(buf.getvalue(), newline="")))
    assert rows == [["", None, 'x,"y"\nz', "1"]]
