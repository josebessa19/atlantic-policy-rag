"""Transient Gemini errors include TLS/connect blips, not only 429."""

from src.generation.gemini import _is_transient_gemini_error


def test_ssl_eof_is_transient() -> None:
    err = ConnectionError(
        "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol"
    )
    assert _is_transient_gemini_error(err) is True


def test_connect_error_name_is_transient() -> None:
    class ConnectError(Exception):
        pass

    assert _is_transient_gemini_error(ConnectError("boom")) is True


def test_auth_error_is_not_transient() -> None:
    assert _is_transient_gemini_error(RuntimeError("API key invalid")) is False
