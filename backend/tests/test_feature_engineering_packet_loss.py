"""Regressões do cálculo de packet_loss_rate usado por treino e inferência."""

from types import SimpleNamespace

import pytest

from ai.feature_engineering import _packet_loss_rate_from_telemetries


def _telemetry(sequence, uptime):
    return SimpleNamespace(sequence=sequence, uptime=uptime)


def test_packet_loss_uses_only_samples_after_node_reset():
    before_reset = [_telemetry(sequence, 486) for sequence in range(1452, 1459)]
    after_reset = [_telemetry(sequence, sequence // 3) for sequence in range(35)]

    rate = _packet_loss_rate_from_telemetries(before_reset + after_reset)

    assert rate == 0.0


def test_packet_loss_keeps_real_gaps_after_reset():
    samples = [
        _telemetry(1457, 486),
        _telemetry(1458, 486),
        _telemetry(0, 0),
        _telemetry(1, 0),
        _telemetry(3, 1),
        _telemetry(4, 1),
    ]

    rate = _packet_loss_rate_from_telemetries(samples)

    assert rate == pytest.approx(0.25)


def test_packet_loss_deduplicates_broadcast_and_unicast():
    samples = [
        _telemetry(10, 3),
        _telemetry(10, 3),
        _telemetry(11, 3),
        _telemetry(11, 3),
        _telemetry(13, 4),
        _telemetry(13, 4),
    ]

    rate = _packet_loss_rate_from_telemetries(samples)

    assert rate == pytest.approx(1 / 3)


def test_packet_loss_detects_fast_reset_while_uptime_is_still_zero():
    samples = [
        _telemetry(4, 0),
        _telemetry(0, 0),
        _telemetry(1, 0),
        _telemetry(2, 0),
    ]

    rate = _packet_loss_rate_from_telemetries(samples)

    assert rate == 0.0


def test_packet_loss_ignores_arrival_inversion_across_minute_boundary():
    # 12 chega depois de 13 e o uptime (em minutos) recua de 6 para 5 sem reset;
    # nenhum pacote foi perdido.
    samples = [
        _telemetry(10, 5),
        _telemetry(11, 5),
        _telemetry(13, 6),
        _telemetry(12, 5),
        _telemetry(14, 6),
        _telemetry(15, 6),
    ]

    rate = _packet_loss_rate_from_telemetries(samples)

    assert rate == 0.0
