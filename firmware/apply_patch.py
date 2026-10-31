"""Aplica e verifica o patch RSSI antes da compilação da biblioteca."""
import subprocess
from pathlib import Path

Import("env")  # noqa: F821 - fornecido pelo SCons/PlatformIO


def apply_loramesher_patch(build_env):
    libdeps = Path(build_env.subst("$PROJECT_LIBDEPS_DIR")) / build_env.subst("$PIOENV")
    libraries = [p for p in libdeps.glob("LoRaMesher*") if (p / "src/LoraMesher.cpp").is_file()]
    if len(libraries) != 1:
        raise RuntimeError("Esperada uma única biblioteca LoRaMesher nas dependências do ambiente")
    library = libraries[0]
    patch = Path(build_env.subst("$PROJECT_DIR")) / "patches/loramesher_rssi_fix.patch"
    if not patch.is_file():
        raise RuntimeError("Patch RSSI obrigatório não encontrado")

    # --force impede que GNU patch inverta automaticamente a direção no dry-run.
    command = ["patch", "-p1", "--force", "--batch", "--fuzz=0", "-l", "-i", str(patch)]
    # O commit fixado mistura LF e CRLF; normalizar só os arquivos do patch.
    for relative in (
        "src/BuildOptions.h", "src/LoraMesher.cpp", "src/LoraMesher.h",
        "src/entities/routingTable/RouteNode.h", "src/services/RoutingTableService.cpp",
        "src/services/RoutingTableService.h",
    ):
        source = library / relative
        content = source.read_bytes()
        if b"\r\n" in content:
            source.write_bytes(content.replace(b"\r\n", b"\n"))

    def run(*flags):
        return subprocess.run(command + list(flags), cwd=library, capture_output=True, text=True, check=False)

    try:
        if run("--reverse", "--dry-run").returncode == 0:
            print("[PATCH] RSSI verificado; já aplicado.")
            return
        if run("--forward", "--dry-run").returncode != 0:
            raise RuntimeError("Biblioteca com patch parcial ou incompatível; remova a dependência local e compile novamente")
        result = run("--forward")
        if run("--reverse", "--dry-run").returncode != 0:
            raise RuntimeError(f"Patch RSSI incompleto; compilação interrompida. {result.stdout} {result.stderr}")
    except FileNotFoundError as exc:
        raise RuntimeError("GNU patch é necessário para compilar o firmware") from exc
    print("[PATCH] RSSI aplicado e verificado.")


apply_loramesher_patch(env)  # noqa: F821 - executar antes de compilar qualquer fonte
