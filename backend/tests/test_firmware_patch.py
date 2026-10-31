"""O build recusa patches parciais e não duplica alterações em builds incrementais."""
import difflib
import runpy
import shutil
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "firmware/apply_patch.py"
FILES = (
    "src/BuildOptions.h", "src/LoraMesher.cpp", "src/LoraMesher.h",
    "src/entities/routingTable/RouteNode.h", "src/services/RoutingTableService.cpp",
    "src/services/RoutingTableService.h",
)


@pytest.fixture
def project(tmp_path):
    if not shutil.which("patch"):
        pytest.skip("GNU patch não disponível")
    library = tmp_path / ".pio/libdeps/test/LoRaMesher"
    for name in FILES:
        path = library / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old\r\n")
    patch = tmp_path / "patches/loramesher_rssi_fix.patch"
    patch.parent.mkdir()
    patch.write_text("".join(
        "".join(difflib.unified_diff(["old\n"], ["new\n"], fromfile=f"a/{name}", tofile=f"b/{name}"))
        for name in FILES[:2]
    ))

    class Environment:
        def subst(self, value):
            return {"$PROJECT_LIBDEPS_DIR": str(tmp_path / ".pio/libdeps"),
                    "$PIOENV": "test", "$PROJECT_DIR": str(tmp_path)}[value]

    def execute():
        runpy.run_path(str(SCRIPT), init_globals={"Import": lambda *args: None, "env": Environment()})

    return library, execute


def test_applies_complete_patch_and_is_idempotent(project):
    library, execute = project
    execute()
    before = {name: (library / name).read_bytes() for name in FILES}
    assert all(before[name] == b"new\n" for name in FILES[:2])
    execute()
    assert before == {name: (library / name).read_bytes() for name in FILES}


def test_partial_patch_is_refused_before_modifying_sources(project):
    library, execute = project
    (library / FILES[0]).write_text("new\n")
    with pytest.raises(RuntimeError, match="parcial"):
        execute()
    assert (library / FILES[1]).read_text() == "old\n"


def test_missing_dependency_is_refused(project):
    library, execute = project
    shutil.rmtree(library)
    with pytest.raises(RuntimeError, match="única biblioteca"):
        execute()
