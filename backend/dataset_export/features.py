"""
`features.csv` e `coverage.csv`.

Uma linha por nó solicitado × instante da grade, inclusive sem dados. As 11
features vêm de `FeatureExtractor.extract_for_window` (fonte canônica), na escala
original, sem normalização. Linha não válida tem as 11 entradas nulas — nunca
zeros. Origem: `DATA_ORIGIN_TRACKED_SINCE` do manifesto separa `real` de
`unknown`; sem esse valor, registros marcados como reais ficam `unknown`.
Elegibilidade = features válidas E aprovação da `ReferencePolicy`;
não é rótulo de saúde.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import timedelta

from ai.sequence_builder import FEATURE_NAMES, feature_dict_to_vector

# Identidade da definição das features: a mesma constante do extrator canônico.
from ai.feature_engineering import FEATURE_SCHEMA_VERSION  # noqa: E402,F401
EXTRACTION_POLICY = "all_origins_v1"
FEATURE_STATUSES = ("valid", "insufficient_data", "invalid_numeric", "error")
ORIGIN_STATUSES = ("real", "synthetic", "mixed", "unknown", "no_data")
EXTRACTOR_SOURCES = ("backend/ai/feature_engineering.py", "backend/ai/reference_policy.py",
                     "backend/ai/sequence_builder.py")

FEATURE_COLUMNS = [
    "source_id", "node_id", "node_name", "network_id",
    "window_start_utc", "window_end_utc", "available_at_utc",
    "window_seconds", "step_seconds", "grid_origin_utc",
    *FEATURE_NAMES,
    "feature_status", "invalid_reason",
    "neighbor_record_count", "distinct_neighbor_count", "telemetry_record_count",
    "reverse_record_count", "first_observation_utc", "last_observation_utc",
    "defaulted_features",
    "unavailable_features",
    "origin_status", "real_record_count", "synthetic_record_count", "unknown_origin_record_count",
    "source_node_ids",
    "simulation_overlap", "simulation_run_ids",
    "reference_eligible_by_policy", "reference_exclusion_reason",
    "feature_schema_version", "extractor_revision", "extraction_policy",
]

COVERAGE_COLUMNS = [
    "source_id", "node_id", "node_name", "requested_start_utc", "requested_end_utc",
    "first_instant_utc", "last_instant_utc", "expected_windows", "written_windows",
    *[f"status_{s}" for s in FEATURE_STATUSES],
    *[f"origin_{o}" for o in ORIGIN_STATUSES],
    "reference_eligible", "reference_excluded", "windows_with_defaults",
    "simulation_overlap_windows",
]


def extractor_revision() -> str:
    """Identidade verificável do extrator e da política (hash dos fontes)."""
    from dataset_export.provenance import REPO_ROOT, sha256_file
    joined = "".join(sha256_file(REPO_ROOT / rel) for rel in EXTRACTOR_SOURCES)
    return "sha256:" + hashlib.sha256(joined.encode()).hexdigest()[:16]


def make_origin_classifier(tracked_since):
    def classify(row) -> str:
        if getattr(row, "data_origin", "real") != "real":
            return "synthetic"
        if tracked_since is None or row.timestamp < tracked_since:
            return "unknown"
        return "real"
    return classify


def origin_status(counts: dict) -> str:
    present = [k for k in ("real", "synthetic", "unknown") if counts.get(k)]
    if not present:
        return "no_data"
    return present[0] if len(present) == 1 else "mixed"


def _status(features):
    if features is None:
        return "insufficient_data", "neighbor_records_lt_2", None
    vector, reason = feature_dict_to_vector(features)
    if vector is None:
        if reason.startswith("invalid_feature:non_finite"):
            return "invalid_numeric", reason, None
        if reason.startswith(("invalid_feature:missing", "missing_observation")):
            return "insufficient_data", reason, None      # Estimativa ausente ou não observada
        return "error", reason, None
    return "valid", None, vector


def iter_feature_rows(session, request, interpretation, node_rows: dict,
                      extractor=None, policy=None):
    """Linhas de `features.csv`, em ordem (node_id, window_end_utc)."""
    from ai.feature_engineering import FeatureExtractor
    from ai.reference_policy import ReferencePolicy
    extractor = extractor or FeatureExtractor()
    policy = policy or ReferencePolicy.from_session(session, interpretation.reference_margin_minutes)
    classify = make_origin_classifier(interpretation.data_origin_tracked_since)
    revision = extractor_revision()
    window = timedelta(seconds=request.window_seconds)
    grid = request.grid()
    for node_id in sorted(node_rows):
        node = node_rows[node_id]
        for end in grid:
            start = end - window
            prov: dict = {}
            features = None
            try:
                features = extractor.extract_for_window(
                    session, node_id, start, end, provenance=prov, origin_classifier=classify)
                status, reason, vector = _status(features)
            except Exception as exc:  # erro explícito e contabilizado (status "error")
                status, reason, vector = "error", f"{type(exc).__name__}: {exc}", None
            sources = sorted(set(prov.get("source_node_ids", ())) | {node_id})
            runs = policy.overlapping_runs(sources, start, end)
            policy_reason = policy.check_window(node_id, start, end, prov) if prov else None
            if status != "valid":
                eligible, exclusion = False, "features_not_valid"
            elif policy_reason:
                eligible, exclusion = False, policy_reason
            else:
                eligible, exclusion = True, None
            counts = prov.get("origin_counts", {})
            row = {
                "source_id": request.source_id,
                "node_id": node_id,
                "node_name": node["name"],
                "network_id": node["network_id"],
                "window_start_utc": start,
                "window_end_utc": end,
                "available_at_utc": end,
                "window_seconds": request.window_seconds,
                "step_seconds": request.step_seconds,
                "grid_origin_utc": request.grid_origin,
            }
            row.update({name: (vector[i] if vector else None) for i, name in enumerate(FEATURE_NAMES)})
            row.update({
                "feature_status": status,
                "invalid_reason": reason,
                "neighbor_record_count": prov.get("neighbor_record_count", 0),
                "distinct_neighbor_count": prov.get("distinct_neighbor_count", 0),
                "telemetry_record_count": prov.get("telemetry_record_count", 0),
                "reverse_record_count": prov.get("reverse_record_count", 0),
                "first_observation_utc": prov.get("first_observation"),
                "last_observation_utc": prov.get("last_observation"),
                "defaulted_features": json.dumps(sorted(prov.get("defaulted_features", []))),
                "unavailable_features": json.dumps(
                    dict(sorted(((features or {}).get("_unavailable") or {}).items()))),
                "origin_status": origin_status(counts),
                "real_record_count": counts.get("real", 0),
                "synthetic_record_count": counts.get("synthetic", 0),
                "unknown_origin_record_count": counts.get("unknown", 0),
                "source_node_ids": json.dumps(sources),
                "simulation_overlap": bool(runs),
                "simulation_run_ids": json.dumps(runs),
                "reference_eligible_by_policy": eligible,
                "reference_exclusion_reason": exclusion,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "extractor_revision": revision,
                "extraction_policy": EXTRACTION_POLICY,
            })
            yield row


class CoverageAccumulator:
    """Agrega `coverage.csv` por nó solicitado."""

    def __init__(self, request, node_rows: dict):
        self.request = request
        self.node_rows = node_rows
        self.expected, self.first, self.last = request.grid_bounds()
        self.counts = {n: defaultdict(int) for n in node_rows}

    def add(self, row: dict) -> None:
        c = self.counts[row["node_id"]]
        c["written_windows"] += 1
        c[f"status_{row['feature_status']}"] += 1
        c[f"origin_{row['origin_status']}"] += 1
        c["reference_eligible" if row["reference_eligible_by_policy"] else "reference_excluded"] += 1
        if json.loads(row["defaulted_features"]):
            c["windows_with_defaults"] += 1
        if row["simulation_overlap"]:
            c["simulation_overlap_windows"] += 1

    def rows(self) -> list:
        out = []
        for node_id in sorted(self.node_rows):
            c = self.counts[node_id]
            row = {
                "source_id": self.request.source_id,
                "node_id": node_id,
                "node_name": self.node_rows[node_id]["name"],
                "requested_start_utc": self.request.start,
                "requested_end_utc": self.request.end,
                "first_instant_utc": self.first,
                "last_instant_utc": self.last,
                "expected_windows": self.expected,
            }
            for column in COVERAGE_COLUMNS[len(row):]:
                row[column] = c.get(column, 0)
            out.append(row)
        return out
