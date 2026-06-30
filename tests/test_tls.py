import ssl

import pytest
import truststore

from utils import tls


class _NotInjected:
    pass


@pytest.fixture(autouse=True)
def _baseline_uninjected_ssl(monkeypatch):
    monkeypatch.setattr(ssl, "SSLContext", _NotInjected, raising=False)


def test_injects_when_not_already_active(monkeypatch):
    calls = []
    monkeypatch.setattr(truststore, "inject_into_ssl", lambda: calls.append(1))

    tls.enable_system_trust_store()

    assert calls == [1]


def test_injection_is_idempotent_across_calls(monkeypatch):
    calls = []

    def fake_inject():
        calls.append(1)
        monkeypatch.setattr(ssl, "SSLContext", truststore.SSLContext)

    monkeypatch.setattr(truststore, "inject_into_ssl", fake_inject)

    tls.enable_system_trust_store()
    tls.enable_system_trust_store()

    assert calls == [1]


def test_skips_when_already_active(monkeypatch):
    calls = []
    monkeypatch.setattr(ssl, "SSLContext", truststore.SSLContext)
    monkeypatch.setattr(truststore, "inject_into_ssl", lambda: calls.append(1))

    tls.enable_system_trust_store()

    assert calls == []


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
