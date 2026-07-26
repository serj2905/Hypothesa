from hypothesa import config


def test_validation_reports_multiple_errors_without_secret_values(monkeypatch) -> None:
    monkeypatch.setattr(config, "TELEGRAM_TOKEN", "")
    monkeypatch.setattr(config, "PARTICIPANT_SALT", "short")
    monkeypatch.setattr(config, "ADAPTIVE_SHARE", 2.0)

    errors = config.validation_errors(require_bot_secrets=True)

    assert any("TELEGRAM_TOKEN" in error for error in errors)
    assert any("PARTICIPANT_SALT" in error for error in errors)
    assert any("ADAPTIVE_SHARE" in error for error in errors)
    assert "short" not in " ".join(errors)
