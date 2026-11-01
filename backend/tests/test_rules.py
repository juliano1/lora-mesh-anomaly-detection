"""Critérios A1–A7 em lógica de Kleene."""
import pytest

NEUTRAL = {
    "rssi_mean": -80.0, "rssi_std": 1.0, "snr_mean": 8.0, "snr_std": 1.0, "pdr_mean": 9.0,
    "battery_delta": 0.0, "packet_loss_rate": 0.0, "rssi_snr_divergence": 0.0,
    "beacon_interval_std": 0.0, "route_change_rate": 0.0, "link_asymmetry": 0.0,
}


def _ev(**over):
    from ai.anomaly_rules import evaluate_criteria
    f = dict(NEUTRAL); f.update(over)
    return evaluate_criteria(f)


def test_neutral_all_miss():
    ev = _ev()
    assert all(ev.criteria[c].state == "miss" for c in ev.criteria)
    assert ev.rules_version == "rules-v1"


def test_a2_uses_only_sequence_loss():
    assert _ev(pdr_mean=0.0).criteria["A2"].state == "miss"
    assert _ev(packet_loss_rate=0.4, pdr_mean=9.0).criteria["A2"].state == "hit"
    ev = _ev(pdr_mean=0.0, _unavailable={"packet_loss_rate": "insufficient_distinct_counters"})
    assert ev.criteria["A2"].state == "not_evaluable"
    assert ev.criteria["A2"].reasons == ["packet_loss_rate:insufficient_distinct_counters"]


def test_a1_low_rssi_confirms_without_std():
    ev = _ev(rssi_mean=-105.0, rssi_std=None, snr_std=None)
    assert ev.criteria["A1"].state == "hit"
    # SNR médio fora do corte descarta o ramo de variabilidade mesmo sem desvio → miss
    assert _ev(rssi_mean=-80.0, rssi_std=None, snr_std=None).criteria["A1"].state == "miss"
    # SNR médio < 0 e desvio ausente: o ramo fica indeterminado → not_evaluable
    ev = _ev(rssi_mean=-80.0, snr_mean=-1.0, rssi_std=None, snr_std=None)
    assert ev.criteria["A1"].state == "not_evaluable"
    assert ev.criteria["A1"].reasons == ["rssi_std:missing"]


def test_a4_index_discards_without_std():
    assert _ev(rssi_std=None, snr_std=None, pdr_mean=6.0).criteria["A4"].state == "miss"
    assert _ev(rssi_std=None, snr_std=None, pdr_mean=2.0).criteria["A4"].state == "not_evaluable"
    assert _ev(rssi_std=9.0, snr_std=6.0, pdr_mean=3.0).criteria["A4"].state == "hit"


def test_unavailable_defaults_are_not_evaluable():
    ev = _ev(_unavailable={"battery_delta": "insufficient_battery_readings",
                           "link_asymmetry": "no_reverse_link",
                           "route_change_rate": "insufficient_parent_records"})
    assert ev.criteria["A3"].state == ev.criteria["A6"].state == ev.criteria["A7"].state == "not_evaluable"


def test_a5_branches_and_borders():
    assert _ev(rssi_snr_divergence=6.0, snr_mean=2.0, pdr_mean=2.0).criteria["A5"].state == "hit"
    assert _ev(rssi_mean=-80.0, snr_std=6.0, pdr_mean=1.5).criteria["A5"].state == "hit"
    assert _ev(battery_delta=-20.0).criteria["A3"].state == "miss"
    assert _ev(link_asymmetry=15.0).criteria["A6"].state == "miss"
    assert _ev(route_change_rate=0.30).criteria["A7"].state == "miss"


@pytest.mark.parametrize("mode", ["absolute", "contextual_fallback", None])
def test_residual_never_enables_a1_in_this_version(mode):
    # caso de regressão: RSSI −80, desvios 1, resíduo −20 → A1 deve ser miss
    for extra in ({}, {"_distance_source": "independent"}):
        ev = _ev(rssi_mean=-80.0, rssi_residual_mean=-20.0, _classification_mode=mode, **extra)
        assert ev.criteria["A1"].state == "miss"
        assert ev.context_mode == mode


def test_non_finite_is_not_evaluable():
    assert _ev(rssi_mean=float("nan"), rssi_std=None).criteria["A1"].state == "not_evaluable"


def test_evidence_is_json_serializable():
    import json
    from datetime import datetime
    ev = _ev(_unavailable={"battery_delta": "insufficient_battery_readings"},
             _classification_mode="contextual_fallback", _context_reason="distance_not_independent",
             _battery_first_at=datetime(2026, 9, 1, 11), _battery_last_at=datetime(2026, 9, 1, 12))
    payload = ev.to_evidence()
    json.dumps(payload)
    assert payload["context_mode"] == "contextual_fallback"
    assert payload["criteria"]["A3"]["state"] == "not_evaluable"
    assert payload["unavailable_features"] == {"battery_delta": "insufficient_battery_readings"}
