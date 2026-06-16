"""Tests for :mod:`llm.exceptions`."""

from __future__ import annotations

import importlib

import pytest

from llm import error_codes
from llm.exceptions import (
    CODE_TO_EXCEPTION,
    LLMCancelled,
    LLMConnectionLost,
    LLMDaemonSpawnFailed,
    LLMDaemonUnreachable,
    LLMError,
    LLMGenerationFailed,
    LLMInferenceTimeout,
    LLMMalformedResponse,
    LLMModelCapacityExceeded,
    LLMModelError,
    LLMModelLoadFailed,
    LLMModelNotFound,
    LLMProtocolError,
    LLMRequestConflict,
    LLMRequestNotFound,
    LLMTransportError,
    LLMVersionMismatch,
)

_CONCRETE_WIRE_CLASSES = [
    LLMVersionMismatch,
    LLMMalformedResponse,
    LLMRequestConflict,
    LLMRequestNotFound,
    LLMModelNotFound,
    LLMModelLoadFailed,
    LLMGenerationFailed,
    LLMCancelled,
    LLMModelCapacityExceeded,
]

_CONCRETE_CLIENT_ONLY_CLASSES = [
    LLMDaemonUnreachable,
    LLMConnectionLost,
    LLMDaemonSpawnFailed,
    LLMInferenceTimeout,
]

_ALL_CONCRETE = _CONCRETE_WIRE_CLASSES + _CONCRETE_CLIENT_ONLY_CLASSES


@pytest.mark.parametrize("cls", _ALL_CONCRETE)
def test_constructable_with_message_only(cls: type[LLMError]) -> None:
    err = cls("boom")
    assert err.message == "boom"
    assert err.details == {}
    assert str(err) == "boom"


@pytest.mark.parametrize("cls", _ALL_CONCRETE)
def test_constructable_with_message_and_details(cls: type[LLMError]) -> None:
    err = cls("boom", {"why": "fuse"})
    assert err.message == "boom"
    assert err.details == {"why": "fuse"}


@pytest.mark.parametrize("cls", _ALL_CONCRETE)
def test_isinstance_llm_error(cls: type[LLMError]) -> None:
    assert isinstance(cls("x"), LLMError)


def test_transport_subtree() -> None:
    assert issubclass(LLMTransportError, LLMError)
    assert issubclass(LLMDaemonUnreachable, LLMTransportError)
    assert issubclass(LLMConnectionLost, LLMTransportError)
    assert issubclass(LLMDaemonSpawnFailed, LLMTransportError)
    # AC-1: the inference timeout lives in the transport subtree (client-local).
    assert issubclass(LLMInferenceTimeout, LLMTransportError)


def test_protocol_subtree() -> None:
    assert issubclass(LLMProtocolError, LLMError)
    assert issubclass(LLMVersionMismatch, LLMProtocolError)
    assert issubclass(LLMMalformedResponse, LLMProtocolError)
    assert issubclass(LLMRequestConflict, LLMProtocolError)
    assert issubclass(LLMRequestNotFound, LLMProtocolError)


def test_model_subtree() -> None:
    assert issubclass(LLMModelError, LLMError)
    assert issubclass(LLMModelNotFound, LLMModelError)
    assert issubclass(LLMModelLoadFailed, LLMModelError)
    assert issubclass(LLMGenerationFailed, LLMModelError)
    assert issubclass(LLMCancelled, LLMModelError)
    assert issubclass(LLMModelCapacityExceeded, LLMModelError)


def test_inference_timeout_has_no_wire_code() -> None:
    # It never travels the wire, so it carries no code and is not in the table.
    assert LLMInferenceTimeout.code is None
    assert LLMInferenceTimeout not in CODE_TO_EXCEPTION.values()


@pytest.mark.parametrize("cls", _CONCRETE_WIRE_CLASSES)
def test_concrete_wire_class_code_attribute(cls: type[LLMError]) -> None:
    assert cls.code is not None
    assert cls.code in CODE_TO_EXCEPTION
    assert CODE_TO_EXCEPTION[cls.code] is cls


@pytest.mark.parametrize("cls", _CONCRETE_CLIENT_ONLY_CLASSES)
def test_client_only_class_has_no_wire_code(cls: type[LLMError]) -> None:
    assert cls.code is None


def test_base_class_code_is_none() -> None:
    assert LLMError.code is None
    assert LLMTransportError.code is None
    assert LLMProtocolError.code is None
    assert LLMModelError.code is None


def test_code_round_trip_via_code_to_exception() -> None:
    for code, cls in CODE_TO_EXCEPTION.items():
        assert cls.code == code


def test_every_wire_constant_maps_to_a_class() -> None:
    assert set(CODE_TO_EXCEPTION) == set(error_codes.ALL_CODES)


def test_re_exports_from_llm_package() -> None:
    for name in (
        "LLMError",
        "LLMTransportError",
        "LLMProtocolError",
        "LLMModelError",
        "LLMDaemonUnreachable",
        "LLMConnectionLost",
        "LLMDaemonSpawnFailed",
        "LLMInferenceTimeout",
        "LLMVersionMismatch",
        "LLMMalformedResponse",
        "LLMRequestConflict",
        "LLMRequestNotFound",
        "LLMModelNotFound",
        "LLMModelLoadFailed",
        "LLMGenerationFailed",
        "LLMCancelled",
        "LLMModelCapacityExceeded",
    ):
        llm_package = importlib.import_module("llm")
        assert hasattr(llm_package, name), f"llm package should re-export {name}"
