import pytest
import truststore

from utils import tls


@pytest.fixture(autouse=True)
def _reset_injection_flag():
    tls._injected = False
    yield
    tls._injected = False


def test_injects_into_ssl_once(monkeypatch):
    calls = []
    monkeypatch.setattr(truststore, "inject_into_ssl", lambda: calls.append(1))

    tls.enable_system_trust_store()
    tls.enable_system_trust_store()

    assert calls == [1]


def test_opt_out_env_var_skips_injection(monkeypatch):
    calls = []
    monkeypatch.setattr(truststore, "inject_into_ssl", lambda: calls.append(1))
    monkeypatch.setenv(tls.DISABLE_ENV_VAR, "1")

    tls.enable_system_trust_store()

    assert calls == []


def test_injection_failure_does_not_raise(monkeypatch):
    def _boom():
        raise RuntimeError("no system trust store")

    monkeypatch.setattr(truststore, "inject_into_ssl", _boom)

    tls.enable_system_trust_store()

    assert tls._injected is False
