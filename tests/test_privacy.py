from hypothesa.privacy import redact_pii


def test_redact_pii_hides_contacts_and_identifiers_but_keeps_amounts() -> None:
    text = (
        "Пишите ivan.petrov@example.ru или +7 (999) 123-45-67. "
        "Карта 4276 1234 5678 9012, комиссия 50000 рублей."
    )

    redacted = redact_pii(text)

    assert "example.ru" not in redacted
    assert "999" not in redacted
    assert "4276" not in redacted
    assert "50000" in redacted
    assert "[EMAIL]" in redacted
    assert "[PHONE]" in redacted
    assert "[LONG_NUMBER]" in redacted
