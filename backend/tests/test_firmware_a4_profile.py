"""Execute the real firmware profiles before evaluating the shared backend rules.

This checks synthetic pipeline inputs, not physical fault detection accuracy.
No radio, running service or pilot database is accessed.
"""
import csv
import io
from datetime import datetime, timedelta
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest


def _function(source, name):
    start = re.search(rf"\b(?:void|uint8_t) {name}\([^)]*\)\s*\{{", source)
    assert start is not None, name
    depth = 1
    end = start.end()
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start.start():end]


@pytest.fixture(scope="module")
def firmware_rows(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required to execute the firmware simulation profiles")
    source = (Path(__file__).resolve().parents[2] / "firmware/src/main.cpp").read_text()
    neighbor = re.search(r"struct NeighborInfo\s*\{.*?\};", source, re.S).group()
    quality = re.findall(r'^\s*neighbor\["l"\]\s*=\s*([^;]+);', source, re.M)
    assert len(quality) == 3  # A2/A5 override, primary beacon, shortened beacon
    harness = """
#include <algorithm>
#include <cstdint>
#include <iostream>
using std::max;
bool simModeActive;
char simAnomalyType;
int simStep, sequenceNumber;
int8_t simBatteryPct = 80;
struct { void println(const char*) {} } Serial;
""" + neighbor + _function(source, "calculateLinkQualityIndex") + _function(
        source, "applySimulationOverrides"
    ) + """
int main() {
    std::cout << "profile,path,step,address,rssi,snr,quality\\n";
    for (char profile : {'0', '1', '4'}) {
        simModeActive = profile != '0';
        simAnomalyType = profile == '0' ? '4' : profile;
        for (simStep = 0; simStep < 12; ++simStep) {
            NeighborInfo neighbors[] = {
                {1, -80, 8.0f, true, 1, 0.0f},
                {2, -85, 8.0f, false, 1, 0.0f}
            };
            if (simModeActive) applySimulationOverrides(neighbors, 2);
            for (int i = 0; i < 2; ++i) {
                int values[] = {PRIMARY_QUALITY, RETRY_QUALITY};
                for (int path = 0; path < 2; ++path) {
                    std::cout << profile << ',' << path << ',' << simStep << ','
                        << neighbors[i].address << ',' << neighbors[i].rssi << ','
                        << neighbors[i].snr << ',' << values[path] << '\\n';
                }
            }
        }
    }
}
""".replace("PRIMARY_QUALITY", quality[1]).replace("RETRY_QUALITY", quality[2])
    work = tmp_path_factory.mktemp("firmware_profiles")
    cpp, executable = work / "profiles.cpp", work / "profiles"
    cpp.write_text(harness)
    subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(executable)],
                   check=True, capture_output=True, text=True, timeout=30)
    output = subprocess.run([str(executable)], check=True, capture_output=True,
                            text=True, timeout=10)
    return list(csv.DictReader(io.StringIO(output.stdout)))


def _features(rows, profile, path):
    from ai.feature_engineering import compute_window_features

    selected = [r for r in rows if r["profile"] == profile and r["path"] == path]
    neighbors = [SimpleNamespace(
        neighbor_address=r["address"], rssi=float(r["rssi"]), snr=float(r["snr"]),
        link_quality=int(r["quality"]), is_parent=r["address"] == "1",
        timestamp=datetime(2026, 10, 16) + timedelta(seconds=20 * int(r["step"])),
    ) for r in selected]
    return selected, compute_window_features(neighbors, [], {})


@pytest.mark.parametrize("path", ["0", "1"], ids=["primary", "shortened"])
def test_a4_profile_satisfies_a4_without_changing_rules(firmware_rows, path):
    from ai.anomaly_rules import evaluate_criteria

    rows, features = _features(firmware_rows, "4", path)
    assert {int(r["quality"]) for r in rows} == {3}
    assert features["rssi_std"] == pytest.approx(20.0)
    assert features["snr_std"] == pytest.approx(5.5)
    assert features["pdr_mean"] == pytest.approx(3.0)
    criteria = evaluate_criteria(features).criteria
    assert [code for code, result in criteria.items() if result.state == "hit"] == ["A4"]


@pytest.mark.parametrize("path", ["0", "1"], ids=["primary", "shortened"])
def test_a1_profile_and_calculated_quality_are_preserved(firmware_rows, path):
    from ai.anomaly_rules import evaluate_criteria

    rows, features = _features(firmware_rows, "1", path)
    first_link = [r for r in rows if r["address"] == "1"]
    assert [int(r["rssi"]) for r in first_link] == list(range(-112, -124, -1))
    assert [int(r["quality"]) for r in first_link] == [2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0]
    assert evaluate_criteria(features).criteria["A1"].state == "hit"


@pytest.mark.parametrize("path", ["0", "1"], ids=["primary", "shortened"])
def test_inactive_simulation_keeps_normal_quality(firmware_rows, path):
    from ai.anomaly_rules import evaluate_criteria

    rows, features = _features(firmware_rows, "0", path)
    assert {int(r["quality"]) for r in rows} == {10}
    assert features["rssi_std"] == features["snr_std"] == 0.0
    assert evaluate_criteria(features).criteria["A4"].state == "miss"
