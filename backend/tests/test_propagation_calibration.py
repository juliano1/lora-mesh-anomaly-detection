"""
Calibração de RSSI de referência por ambiente (log-distance).

Contexto: a rede "outdoor 1" estimava ~3 m para enlaces reais de ~40 m.
Causa: RSSI_REF_1M era um único valor global calibrado indoor (-52 dBm).
Fix: referência por environment_type (indoor mantém -52; outdoor calibrado
em campo). Estes testes travam a calibração e a compatibilidade retroativa.
"""
from types import SimpleNamespace

import pytest

from ai.propagation import (
    RSSI_REF_1M,
    estimate_distance_contextual,
    get_propagation_params,
    rssi_expected,
)


def _network(**over):
    """Network mínimo (só os atributos lidos por get_propagation_params)."""
    base = dict(
        environment_type="outdoor",
        building_type="open_field",
        obstacle_density="low",
        typical_link_distance_class="medium",
        availability_criticality="low",
        predominant_obstacle_material=None,
        avg_walls_per_link=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ─── Referência por ambiente ────────────────────────────────────────────────

def test_indoor_reference_unchanged():
    """Indoor mantém a calibração histórica (-52 dBm) — não regride."""
    params = get_propagation_params(_network(environment_type="indoor",
                                             building_type="residential"))
    assert params["reference_rssi_1m"] == -52


def test_outdoor_uses_field_calibrated_reference():
    """Outdoor usa referência calibrada em campo, distinta do indoor."""
    params = get_propagation_params(_network())
    assert params["reference_rssi_1m"] != -52
    assert -45 <= params["reference_rssi_1m"] <= -25


def test_reference_follows_environment_even_without_full_context():
    """A referência depende só de environment_type, não dos 5 campos de contexto.

    availability_criticality ausente NÃO deve forçar a referência indoor numa
    rede outdoor (foi exatamente o que quebrou a rede outdoor 1).
    """
    params = get_propagation_params(_network(availability_criticality=None))
    assert params["has_context"] is False
    assert params["reference_rssi_1m"] < -25
    assert params["reference_rssi_1m"] != -52


# ─── Distância física plausível ─────────────────────────────────────────────

def test_outdoor_gateway_links_estimate_near_40m():
    """Enlaces do gateway outdoor 1 (~40 m) a -69/-73 dBm, n=2.1 open_field.

    Com a referência indoor antiga (-52) isso dava ~4-6 m. Calibrado, os
    enlaces de ~40 m devem cair na faixa 30-55 m (RSSI de amostra única varia).
    """
    params = get_propagation_params(_network())
    ref = params["reference_rssi_1m"]
    n = params["n"]
    assert n == pytest.approx(2.1, abs=0.01)  # open_field
    for rssi in (-69, -73):
        d = estimate_distance_contextual(rssi, n=n, reference_rssi_1m=ref)
        assert 25 <= d <= 60, f"rssi={rssi} -> {d} m fora do esperado p/ ~40 m"


# ─── Consistência e compatibilidade ─────────────────────────────────────────

def test_estimate_and_expected_are_inverse():
    """estimate_distance_contextual e rssi_expected são inversos exatos."""
    ref, n = -37, 2.1
    for d in (5.0, 20.0, 40.0):
        r = rssi_expected(d, n=n, reference_rssi_1m=ref)
        back = estimate_distance_contextual(round(r), n=n, reference_rssi_1m=ref)
        assert back == pytest.approx(d, rel=0.15)


def test_default_reference_is_backward_compatible():
    """Sem reference_rssi_1m explícito, cai no global histórico (-52)."""
    assert RSSI_REF_1M == -52
    d_default = estimate_distance_contextual(-72, n=2.7)
    d_explicit = estimate_distance_contextual(-72, n=2.7, reference_rssi_1m=-52)
    assert d_default == d_explicit
