"""
Testes unitários para SequenceBuilder.

Cobre:
- Construção de sequência de inferência (shape correto, None quando insuficiente)
- Imputação forward-fill até max_impute_gap
- Descarte de sequências com gap > max_impute_gap
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai.sequence_builder import SequenceBuilder, N_FEATURES, FEATURE_NAMES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_feature_dict(value: float = 1.0) -> dict:
    """Dicionário de features com valor constante para todos os campos."""
    return {k: value for k in FEATURE_NAMES}


def _mock_extractor(return_seq: list):
    """
    Cria mock de FeatureExtractor.extract_for_window() com retornos em sequência.

    Args:
        return_seq: lista de dict|None — retornados em ordem pelo mock.
    """
    extractor = MagicMock()
    extractor.extract_for_window = MagicMock(side_effect=return_seq)
    return extractor


# ── Testes de inferência ──────────────────────────────────────────────────────


class TestBuildInferenceSequence:
    def test_returns_correct_shape_when_all_windows_available(self, session):
        """Sequência completa deve ter shape (W, N_FEATURES)."""
        W = 6
        builder = SequenceBuilder(window_size=W, window_minutes=5)
        features = [_make_feature_dict(float(i)) for i in range(W)]
        extractor = _mock_extractor(features)

        result = builder.build_inference_sequence(session, node_id=1, feature_extractor=extractor)

        assert result is not None
        assert result.shape == (W, N_FEATURES)
        assert result.dtype == np.float32

    def test_returns_none_when_too_many_missing(self, session):
        """Mais de max_impute_gap faltantes consecutivos → None."""
        W = 6
        builder = SequenceBuilder(window_size=W, window_minutes=5, max_impute_gap=1)
        # Janelas 0-1 OK, 2-3 None (gap=2 > max_impute_gap=1)
        features = [_make_feature_dict(1.0), _make_feature_dict(1.0), None, None, _make_feature_dict(1.0), _make_feature_dict(1.0)]
        extractor = _mock_extractor(features)

        result = builder.build_inference_sequence(session, node_id=1, feature_extractor=extractor)

        assert result is None

    def test_forward_fill_within_gap_limit(self, session):
        """Forward-fill deve copiar a janela anterior quando gap ≤ max_impute_gap."""
        W = 4
        builder = SequenceBuilder(window_size=W, window_minutes=5, max_impute_gap=2)
        # Janela 0: valor 5.0, 1: None (forward-fill com 5.0), 2-3: OK
        features = [_make_feature_dict(5.0), None, _make_feature_dict(7.0), _make_feature_dict(8.0)]
        extractor = _mock_extractor(features)

        result = builder.build_inference_sequence(session, node_id=1, feature_extractor=extractor)

        assert result is not None
        assert result.shape == (W, N_FEATURES)
        # Linha 1 deve ser forward-fill da linha 0 (valor 5.0)
        np.testing.assert_allclose(result[1], result[0])

    def test_first_window_missing_rejects_sequence(self, session):
        """Lacuna inicial não é preenchida com zeros — sequência rejeitada."""
        W = 3
        builder = SequenceBuilder(window_size=W, window_minutes=5, max_impute_gap=2)
        features = [None, _make_feature_dict(3.0), _make_feature_dict(4.0)]
        extractor = _mock_extractor(features)

        result = builder.build_inference_sequence(session, node_id=1, feature_extractor=extractor)

        assert result is None

    def test_excludes_sim_node(self, session):
        """Se exclude_sim=True e nó está em sim_mode, retorna None."""
        from models import Node, Network
        # Criar network e node em sim_mode
        net = Network(name="test_net_sim_seq")
        session.add(net)
        session.flush()
        node = Node(node_id=9901, network_id=net.id, sim_mode=True, status="online")
        session.add(node)
        session.commit()

        W = 4
        builder = SequenceBuilder(window_size=W, window_minutes=5)
        extractor = _mock_extractor([_make_feature_dict()] * W)

        result = builder.build_inference_sequence(
            session, node_id=9901, feature_extractor=extractor, exclude_sim=True
        )
        assert result is None
