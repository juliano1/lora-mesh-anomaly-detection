"""Roteiro E2E: partes que rodam sem Docker, broker nem backend."""
import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "e2e_sintetico.py"


@pytest.fixture(scope="module")
def e2e():
    spec = importlib.util.spec_from_file_location("e2e_sintetico", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_payload_is_accepted_by_the_backend_schema(e2e):
    sys.path.insert(0, str(ROOT / "backend"))
    from mqtt_service_sync import LoRaPacket
    rnd = random.Random(1)
    for degraded in (False, True):
        for battery in (None, 80):
            msg = e2e.payload("N3", "0003", "0002", 1234, rnd, battery=battery, degraded=degraded)
            packet = LoRaPacket(**msg)
            assert packet.q == 1234 and packet.b == battery and len(packet.n) == 2
    assert "b" not in e2e.payload("N2", "0002", "0003", 1, rnd)


def test_battery_follows_the_firmware_rule(e2e):
    gate = e2e.BatteryGate()
    beacons_per_heartbeat = e2e.BATTERY_HEARTBEAT // e2e.BEACON          # 15
    sent = [k for k in range(40) if gate.include("N2", k, 90)]
    assert sent == [0, beacons_per_heartbeat, 2 * beacons_per_heartbeat]
    assert gate.include("N2", 31, 88)                                    # mudou 2 p.p.
    assert not gate.include("N2", 32, 87)                                # só 1 p.p.


def test_broker_image_comes_pinned_from_the_compose(e2e):
    image = e2e.broker_image()
    assert image.startswith("eclipse-mosquitto:2@sha256:") and len(image.split("sha256:")[1]) == 64


def test_refuses_a_non_empty_work_folder(e2e, tmp_path, capsys):
    (tmp_path / "algo.txt").write_text("x")
    assert e2e.main(["--trabalho", str(tmp_path)]) == 2
    assert "não está vazia" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["algo.txt"]
