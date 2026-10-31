"""Isola a suíte das credenciais e do banco usados na operação local."""
import os
import tempfile
from pathlib import Path


_test_data = tempfile.TemporaryDirectory(prefix="lora-tests-")
os.environ.update({
    "DATABASE_URL": f"sqlite:///{Path(_test_data.name) / 'tests.db'}",
    "JWT_SECRET_KEY": "tests-only-signing-key-never-use-in-a-deployment",  # pragma: allowlist secret
    "ADMIN100_PASSWORD": "",
    "ADMIN200_PASSWORD": "",
    "MQTT_BROKER": "127.0.0.1",
    "MQTT_USER": "",
    "MQTT_PASS": "",
})
