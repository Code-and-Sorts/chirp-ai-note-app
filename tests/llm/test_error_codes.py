"""Tests for :mod:`llm.error_codes`."""

from __future__ import annotations

import re

import pytest

from llm import error_codes
from llm.exceptions import CODE_TO_EXCEPTION, LLMError

_SCREAMING_SNAKE_RE = re.compile(r"^[A-Z][A-Z_]*$")

_DECLARED_CODES = {
    "PROTOCOL_VERSION_MISMATCH": error_codes.PROTOCOL_VERSION_MISMATCH,
    "PROTOCOL_MALFORMED": error_codes.PROTOCOL_MALFORMED,
    "MODEL_NOT_FOUND": error_codes.MODEL_NOT_FOUND,
    "MODEL_LOAD_FAILED": error_codes.MODEL_LOAD_FAILED,
    "MODEL_GENERATION_FAILED": error_codes.MODEL_GENERATION_FAILED,
    "MODEL_CANCELLED": error_codes.MODEL_CANCELLED,
}


@pytest.mark.parametrize("name,value", sorted(_DECLARED_CODES.items()))
def test_declared_codes_are_screaming_snake_case(name: str, value: str) -> None:
    assert isinstance(value, str)
    assert value, "code constant must be a non-empty string"
    assert _SCREAMING_SNAKE_RE.match(value), f"{name}={value!r} not SCREAMING_SNAKE"
    assert value == name, "constant value should match its name"


def test_all_codes_frozenset_matches_constants() -> None:
    assert error_codes.ALL_CODES == set(_DECLARED_CODES.values())


def test_code_to_exception_has_no_missing_entries() -> None:
    assert set(CODE_TO_EXCEPTION.keys()) == set(_DECLARED_CODES.values())


def test_code_to_exception_has_no_orphan_entries() -> None:
    for code in CODE_TO_EXCEPTION:
        assert code in error_codes.ALL_CODES


def test_code_to_exception_values_are_llm_error_subclasses() -> None:
    for cls in CODE_TO_EXCEPTION.values():
        assert issubclass(cls, LLMError)


def test_re_exported_via_error_codes_module() -> None:
    assert error_codes.CODE_TO_EXCEPTION is CODE_TO_EXCEPTION


def test_re_exported_missing_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        error_codes.does_not_exist  # noqa: B018
