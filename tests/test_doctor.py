from __future__ import annotations

import json
from urllib.error import URLError

from habitus_ai import doctor


def _by_name(checks):
    return {item.name: item for item in checks}


def test_doctor_accepts_configured_model_from_local_api(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "_ollama_models",
        lambda _url, _timeout: ("another:latest", "test-model:1b"),
    )
    checks = _by_name(doctor.run_checks(model="test-model:1b"))

    assert checks["python"].ok is True
    assert checks["platform"].ok is True
    assert checks["habitus-package"].ok is True
    assert checks["torch"].ok is True
    assert checks["ollama-api"].ok is True
    assert checks["speech-model"].ok is True


def test_doctor_reports_server_and_model_failure(monkeypatch) -> None:
    def unavailable(_url, _timeout):
        raise URLError("offline")

    monkeypatch.setattr(doctor, "_ollama_models", unavailable)
    checks = _by_name(doctor.run_checks(model="test-model:1b"))

    assert checks["ollama-api"].ok is False
    assert checks["ollama-api"].required is True
    assert checks["speech-model"].ok is False


def test_doctor_json_exit_status_is_machine_readable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda **_kwargs: (
            doctor.DoctorCheck("required", True, "available"),
            doctor.DoctorCheck(
                "optional",
                False,
                "not installed",
                required=False,
                fix="Optional only.",
            ),
        ),
    )

    assert doctor.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["checks"][1]["required"] is False
