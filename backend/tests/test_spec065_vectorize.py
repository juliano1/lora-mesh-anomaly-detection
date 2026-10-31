"""Vetorização preserva a identidade das amostras retidas."""
from ai.sequence_builder import FEATURE_NAMES


def _f(**over):
    base = {name: 1.0 for name in FEATURE_NAMES}
    base.update(over)
    return base


def test_kept_records_stay_aligned_with_rows():
    from ai.sequence_builder import vectorize_windows
    records = [(1, _f(rssi_mean=-70.0)), (2, _f(rssi_std=None)), (3, _f(rssi_mean=-90.0))]
    X, kept, exclusions = vectorize_windows(records)
    assert X.shape == (2, len(FEATURE_NAMES))
    assert [r[0] for r in kept] == [1, 3]
    assert X[1, FEATURE_NAMES.index("rssi_mean")] == -90.0
    assert exclusions == {"invalid_feature:missing:rssi_std": 1}


def test_custom_accessor_and_empty_input():
    from ai.sequence_builder import vectorize_windows
    X, kept, exclusions = vectorize_windows([], features_of=lambda r: r)
    assert X.shape == (0, len(FEATURE_NAMES)) and kept == [] and not exclusions



def test_unobserved_feature_rejects_the_vector():
    """Valor de apresentação (0 na perda, 5 no índice de qualidade) nunca entra como medida."""
    from ai.sequence_builder import feature_dict_to_vector, vectorize_windows
    unobserved = _f(packet_loss_rate=0.0, _unavailable={"packet_loss_rate": "insufficient_distinct_counters"})
    assert feature_dict_to_vector(unobserved) == (
        None, "missing_observation:packet_loss_rate:insufficient_distinct_counters")
    X, kept, exclusions = vectorize_windows([(1, unobserved), (2, _f())])
    assert [r[0] for r in kept] == [2]
    assert exclusions == {"missing_observation:packet_loss_rate:insufficient_distinct_counters": 1}
