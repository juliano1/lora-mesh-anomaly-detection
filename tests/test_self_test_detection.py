"""Autoteste: estados distintos, IF e LSTM-AE do backend, treino/calibração/teste separados."""
import csv
import importlib.util
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "self_test_detection.py"


@pytest.fixture(scope="module")
def selftest():
    spec = importlib.util.spec_from_file_location("self_test_detection", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolvem o módulo pelo nome
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data_dir(selftest, tmp_path_factory):
    directory = tmp_path_factory.mktemp("autoteste")
    selftest.generate(directory, seed=7, episodes=3)
    return directory


@pytest.fixture(scope="module")
def outcome(selftest, data_dir):
    pytest.importorskip("torch")
    return selftest.run(data_dir)


@pytest.fixture(scope="module")
def report(outcome):
    return {d: {r["cenario"]: r for r in verdicts} for d, verdicts in outcome[0].items()}


def test_individual_models_and_combinations_are_reported_separately(report):
    assert set(report) == {"isolation_forest", "lstm_ae", "cascata", "fusao"}


@pytest.mark.parametrize("key", ["esparsa", "nao_finito", "sem_modelo"])
def test_non_decidable_instants_are_never_normal(report, key):
    for detector, verdicts in report.items():
        assert verdicts[key]["aprovado"], (detector, verdicts[key])
        assert "normal" not in verdicts[key]["obtido"]


def test_missed_anomalies_and_false_positives_are_counted(report):
    for verdicts in report.values():
        for key in ("rssi_snr", "perda", "bateria"):
            r = verdicts[key]
            assert r["nao_detectadas"] == r["instantes"] - r["obtido"].get("anomalia", 0)
        assert verdicts["controle"]["falsos_positivos"] == verdicts["controle"]["obtido"].get("anomalia", 0)


def test_detected_signal_degradation_is_classified_a1(report):
    # RSSI ≈ −106 dBm fica sempre abaixo do corte de A1 (−100 dBm): toda detecção é A1.
    for verdicts in report.values():
        typed = [c for c in verdicts["rssi_snr"]["verificacoes"] if c["criterio"] == "tipo A1"][0]
        assert typed["valor"] == 1.0


def test_training_calibration_and_test_do_not_overlap(outcome, data_dir, selftest):
    summary = outcome[1]["resumo"]
    lstm = summary["treino"]["lstm_ae"]
    val_cut = datetime.fromisoformat(lstm["cutoff"])
    calib_cut = datetime.fromisoformat(lstm["calib_cutoff"])
    # Treino < validação (parada antecipada) < calibração (limiar p99 e τ) < teste.
    assert val_cut < calib_cut and lstm["threshold_source"] == "calibration"
    # O IF treina até o início da calibração; ela fica fora dos dois treinos.
    assert datetime.fromisoformat(summary["treino"]["isolation_forest"]["end_time"]) == calib_cut
    assert datetime.fromisoformat(summary["calibracao"]["inicio"]) == calib_cut
    assert lstm["n_train"] >= 10 and lstm["n_val"] >= 1 and lstm["n_calib"] >= 1
    calibration_end = datetime.fromisoformat(summary["calibracao"]["fim"])
    assert calib_cut < calibration_end
    first_test = min(datetime.fromisoformat(r["timestamp"]) for r in outcome[1]["resultados"])
    assert calibration_end < first_test


def test_evaluation_rejects_overlap_with_training(selftest, data_dir, tmp_path):
    shutil.copy(data_dir / selftest.TRAIN_FILE, tmp_path / selftest.TRAIN_FILE)
    shutil.copy(data_dir / selftest.TRAIN_FILE, tmp_path / selftest.EVAL_FILE)
    shutil.copy(data_dir / selftest.INSTANTS_FILE, tmp_path / selftest.INSTANTS_FILE)
    with pytest.raises(ValueError, match="posterior"):
        selftest.evaluate(tmp_path)


def test_instant_inside_reference_is_rejected(selftest, data_dir, tmp_path):
    for name in (selftest.TRAIN_FILE, selftest.EVAL_FILE):
        shutil.copy(data_dir / name, tmp_path / name)
    rows = list(csv.reader((data_dir / selftest.INSTANTS_FILE).open(encoding="utf-8")))
    rows[1][3] = "2026-01-01T12:00:00"
    with (tmp_path / selftest.INSTANTS_FILE).open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)
    with pytest.raises(ValueError, match="dentro da referência"):
        selftest.evaluate(tmp_path)


def test_if_only_mode_without_lstm(selftest, data_dir):
    verdicts, _ = selftest.run(data_dir, with_lstm=False)
    assert set(verdicts) == {"isolation_forest"}
    by = {r["cenario"]: r for r in verdicts["isolation_forest"]}
    assert by["sem_modelo"]["obtido"] == {"sem_modelo": by["sem_modelo"]["instantes"]}


def test_cli_reads_saved_csv_and_uses_pass_fail_exit_codes(data_dir, tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "decisoes.csv"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--entrada", str(data_dir), "--csv", str(out)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode in (0, 1), proc.stderr
    assert ("Resultado: APROVADO" in proc.stdout) == (proc.returncode == 0)
    assert "falsos positivos:" in proc.stdout and "Calibração" in proc.stdout
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["detector"] for r in rows} == {"isolation_forest", "lstm_ae", "cascata", "fusao"}
    assert {r["estado"] for r in rows} >= {"normal", "anomalia", "dados_insuficientes",
                                           "entrada_invalida", "sem_modelo"}


def test_cli_reports_execution_error_with_exit_code_2(tmp_path):
    proc = subprocess.run([sys.executable, str(SCRIPT), "--entrada", str(tmp_path)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2 and "ERRO de execução" in proc.stderr
