#!/usr/bin/env python3
"""Triagem de segredos, dados pessoais e arquivos proibidos. Não substitui revisão manual ou secret scanner.

Mostra apenas caminho, linha e regra; nunca imprime o conteúdo suspeito.
Por padrão, lê arquivos versionados e novos não ignorados da árvore atual.
Na árvore atual, também recusa referências numeradas a documentos internos.
--history também examina blobs textuais de até 5 MB alcançáveis por referências locais.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERNAL_REFERENCE = re.compile(r"(?<![A-Za-z])spec[-_ ]?[0-9]{2,3}", re.IGNORECASE)
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "credential_pattern": re.compile(
        r"(?<![A-Za-z0-9_])(?:AKIA|ASIA)[A-Z0-9]{16}"
        r"|(?<![A-Za-z0-9_])AIza[0-9A-Za-z_-]{35}"
        r"|(?<![A-Za-z0-9_])gh[pousr]_[0-9A-Za-z]{30,}"
        r"|(?<![A-Za-z0-9_])github_pat_[0-9A-Za-z_]{50,}"
        r"|(?<![A-Za-z0-9_])sk-(?:proj-|ant-)?[A-Za-z0-9_-]{40,}"
        r"|(?<![A-Za-z0-9_])[0-9]{8,12}:[A-Za-z0-9_-]{35}"
    ),
    "mac_address": re.compile(r"\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b"),
    "personal_path": re.compile(r"/(?:home|Users)/[^\s/]+/|[A-Za-z]:\\Users\\[^\\\s]+"),
    # Credencial embutida em URL (usuario:senha@host); placeholders <...> são ignorados.
    "url_with_credentials": re.compile(r"[a-z][a-z0-9+.-]*://(?!<)[^\s:/@\"'<>]+:(?!<)[^\s@/\"'<>]+@"),
    "email_address": re.compile(r"(?<![\w.+-])[\w.+-]+@(?!example\.(?:com|org|net|invalid)\b)"
                                r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"),
}
# IPv4 fora de loopback, "qualquer interface" e das faixas de documentação (RFC 5737).
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
ALLOWED_IPV4 = re.compile(r"^(?:127\.\d+\.\d+\.\d+|0\.0\.0\.0|192\.0\.2\.\d+|198\.51\.100\.\d+|203\.0\.113\.\d+)$")
# Nome de variável que sugere segredo (qualquer prefixo/sufixo), sem diferenciar maiúsculas.
_SECRET_NAME = r"[A-Za-z0-9_]*(?:SECRET|PASSWORD|PASSWD|_PASS|TOKEN|API_KEY|PRIVATE_KEY)[A-Za-z0-9_]*"
# Valor entre aspas, em qualquer arquivo: NOME = "valor", NOME: str = "valor", #define NOME "valor".
SECRET_ASSIGNMENT = re.compile(
    rf"(?i:{_SECRET_NAME})\s*(?::\s*str\s*)?(?:[=:]\s*|\s+)[\"']([^\"'\r\n]+)[\"']"
)
# Valor sem aspas no início da linha, só em arquivos de configuração (.env, YAML, INI...).
SECRET_ASSIGNMENT_BARE = re.compile(
    rf"^[ \t]*(?:export[ \t]+)?(?i:{_SECRET_NAME})[ \t]*[=:][ \t]*([^\s\"'#$<{{][^\s#]*)", re.MULTILINE
)
CONFIG_SUFFIXES = {".env", ".example", ".yml", ".yaml", ".ini", ".cfg", ".conf", ".toml", ".properties"}
PLACEHOLDERS = {"bearer", "CHANGE_ME", "CHANGE-ME", "YourPassword", "your_wifi_password", "your_mqtt_password"}
GENERATED_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pt", ".pth", ".pkl", ".pickle", ".joblib",
                      ".pem", ".key", ".p12", ".pfx", ".bin", ".parquet", ".zip"}


def git(*args, data=None):
    return subprocess.check_output(["git", *args], cwd=ROOT, input=data)


def scan_text(path, text):
    findings = []
    for rule, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"path": path, "line": text.count("\n", 0, match.start()) + 1, "rule": rule})
    for match in IPV4.finditer(text):
        if all(int(octet) <= 255 for octet in match[0].split(".")) and not ALLOWED_IPV4.match(match[0]):
            findings.append({"path": path, "line": text.count("\n", 0, match.start()) + 1,
                             "rule": "ip_address"})
    # Fixtures e exemplos podem conter strings deliberadamente inválidas/sintéticas.
    synthetic = any(part in Path(path).parts for part in ("tests", "testes")) or Path(path).name == "conftest.py"
    if not synthetic:
        p = Path(path)
        config_like = p.suffix in CONFIG_SUFFIXES or p.name.startswith(".env")
        matches = list(SECRET_ASSIGNMENT.finditer(text))
        if config_like:
            matches += list(SECRET_ASSIGNMENT_BARE.finditer(text))
        for match in matches:
            value = match[1]
            if value in PLACEHOLDERS:
                continue
            findings.append({"path": path, "line": text.count("\n", 0, match.start()) + 1,
                             "rule": "literal_secret_candidate"})
    return findings


def current_tree(inventory_path=None):
    names = sorted(set(filter(None, git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0"))))
    rows, findings = [], []
    for name in names:
        path = ROOT / name
        if inventory_path and path.resolve() == inventory_path.resolve():
            continue
        if not path.is_file():
            rows.append({"arquivo": name, "estado": "removido da árvore de trabalho"})
            continue
        if INTERNAL_REFERENCE.search(name):
            findings.append({"path": name, "line": 0, "rule": "internal_design_reference"})
        data = path.read_bytes()
        row = {"arquivo": name, "estado": "triagem automática; revisão manual pendente",
               "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if path.suffix.lower() in GENERATED_SUFFIXES:
            findings.append({"path": name, "line": 0, "rule": "generated_or_private_artifact"})
        if path.name == ".env" or path.name == "credentials.h":
            findings.append({"path": name, "line": 0, "rule": "local_credentials_file"})
        try:
            text = data.decode("utf-8")
        except UnicodeError:
            row["estado"] = "binário; verificar origem e autorização"
        else:
            row["linhas"] = len(text.splitlines())
            findings.extend(scan_text(name, text))
            for match in INTERNAL_REFERENCE.finditer(text):
                findings.append({"path": name, "line": text.count("\n", 0, match.start()) + 1,
                                 "rule": "internal_design_reference"})
            if path.suffix == ".py":
                try:
                    ast.parse(text, filename=name)
                except SyntaxError as exc:
                    findings.append({"path": name, "line": exc.lineno, "rule": "python_syntax"})
        rows.append(row)
    return rows, findings


def history():
    objects = {}
    for entry in git("rev-list", "--objects", "--all").splitlines():
        fields = entry.split(b" ", 1)
        if len(fields) == 2:
            objects[fields[0].decode()] = fields[1].decode(errors="replace")
    findings = []
    counts = {"text_blobs": 0, "binary_blobs_skipped": 0, "large_blobs_skipped": 0}
    if not objects:  # repositório sem commits: não há histórico a examinar
        return counts, findings
    info = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
               data=("\n".join(objects) + "\n").encode()).decode().splitlines()
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE) as process:
        try:
            for entry in info:
                oid, kind, size = entry.split()
                if kind != "blob":
                    continue
                if int(size) > 5_000_000:
                    counts["large_blobs_skipped"] += 1
                    continue
                process.stdin.write((oid + "\n").encode())
                process.stdin.flush()
                length = int(process.stdout.readline().split()[2])
                data = process.stdout.read(length)
                process.stdout.read(1)
                try:
                    if b"\0" in data:
                        raise UnicodeError
                    text = data.decode("utf-8")
                except UnicodeError:
                    counts["binary_blobs_skipped"] += 1
                    continue
                counts["text_blobs"] += 1
                for finding in scan_text(objects[oid], text):
                    findings.append({**finding, "blob": oid[:12]})
        finally:
            process.stdin.close()
    return counts, findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--inventory", type=Path, help="CSV de cobertura por arquivo")
    args = parser.parse_args()
    rows, findings = current_tree(args.inventory)
    result = {"scope": "working_tree", "files": len(rows), "findings": findings}
    if args.inventory:
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        fields = ["arquivo", "estado", "bytes", "linhas", "sha256"]
        if args.inventory.exists():
            # Colunas extras preenchidas à mão (decisões, responsáveis) sobrevivem à regeneração.
            with args.inventory.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                extra = [key for key in (reader.fieldnames or []) if key not in fields]
                previous = {row["arquivo"]: row for row in reader}
            fields.extend(extra)
            for row in rows:
                row.update({key: previous.get(row["arquivo"], {}).get(key, "") for key in extra})
            # Arquivos retirados continuam no inventário, com a justificativa já registrada.
            present = {row["arquivo"] for row in rows}
            for name, old in previous.items():
                if name not in present:
                    rows.append({**{key: old.get(key, "") for key in fields},
                                 "estado": "ausente da árvore", "bytes": "", "linhas": "", "sha256": ""})
            rows.sort(key=lambda row: row["arquivo"])
        with args.inventory.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    if args.history:
        counts, historic = history()
        result["history"] = {**counts, "findings": historic}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(bool(findings or result.get("history", {}).get("findings")))


if __name__ == "__main__":
    raise SystemExit(main())
