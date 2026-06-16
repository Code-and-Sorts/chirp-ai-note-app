"""Tests for :mod:`llm.protocol`."""

from __future__ import annotations

import json

import pytest

from llm import protocol
from llm.exceptions import LLMMalformedResponse

_OP_CONSTANTS = [
    protocol.OP_HELLO,
    protocol.OP_HEALTH,
    protocol.OP_CHAT,
    protocol.OP_EMBED,
    protocol.OP_CANCEL,
    protocol.OP_MODEL_LIST,
    protocol.OP_MODEL_LOAD,
    protocol.OP_MODEL_UNLOAD,
    protocol.OP_MODEL_STATUS,
]

_EVENT_CONSTANTS = [
    protocol.EVENT_READY,
    protocol.EVENT_LOADING,
    protocol.EVENT_DELTA,
    protocol.EVENT_DONE,
    protocol.EVENT_ERROR,
    protocol.EVENT_VERSION_MISMATCH,
    protocol.EVENT_STATUS,
]


def _legal_id() -> str:
    return protocol.new_request_id()


@pytest.mark.parametrize("op", _OP_CONSTANTS)
def test_round_trip_request_envelope(op: str) -> None:
    envelope = {"id": _legal_id(), "op": op, "payload": {"k": "v"}}
    if op == protocol.OP_HELLO:
        envelope.pop("id")
    encoded = protocol.encode_request(envelope)
    assert encoded.endswith(b"\n")
    assert protocol.decode_line(encoded.rstrip(b"\n")) == envelope


@pytest.mark.parametrize("event", _EVENT_CONSTANTS)
def test_round_trip_event_envelope(event: str) -> None:
    envelope = {"id": _legal_id(), "event": event, "detail": "x"}
    encoded = protocol.encode_event(envelope)
    assert encoded.endswith(b"\n")
    assert protocol.decode_line(encoded.rstrip(b"\n")) == envelope


def test_encoded_payload_is_utf8_bytes() -> None:
    encoded = protocol.encode_request({"op": protocol.OP_HELLO})
    assert isinstance(encoded, bytes)
    assert encoded.decode("utf-8")


def test_decode_rejects_non_utf8_bytes() -> None:
    bad = b"\xff\xfe not utf-8"
    with pytest.raises(LLMMalformedResponse):
        protocol.decode_line(bad)


def test_decode_rejects_invalid_json() -> None:
    with pytest.raises(LLMMalformedResponse):
        protocol.decode_line(b"{not json")


@pytest.mark.parametrize(
    "payload", [b"[1,2,3]", b'"a string"', b"42", b"null", b"true"]
)
def test_decode_rejects_non_object_root(payload: bytes) -> None:
    with pytest.raises(LLMMalformedResponse):
        protocol.decode_line(payload)


def test_decode_rejects_line_over_max_bytes() -> None:
    too_big = b"a" * (protocol.MAX_LINE_BYTES + 1)
    with pytest.raises(LLMMalformedResponse):
        protocol.decode_line(too_big)


def test_decode_accepts_line_at_exact_max_bytes_boundary() -> None:
    overhead = len(b'{"k":""}')
    padding_length = protocol.MAX_LINE_BYTES - overhead
    payload = b'{"k":"' + b"a" * padding_length + b'"}'
    assert len(payload) == protocol.MAX_LINE_BYTES
    assert protocol.decode_line(payload) == {"k": "a" * padding_length}


def test_decode_rejects_line_one_byte_over_max_bytes_boundary() -> None:
    overhead = len(b'{"k":""}')
    padding_length = protocol.MAX_LINE_BYTES - overhead + 1
    payload = b'{"k":"' + b"a" * padding_length + b'"}'
    assert len(payload) == protocol.MAX_LINE_BYTES + 1
    with pytest.raises(LLMMalformedResponse):
        protocol.decode_line(payload)


def test_encode_rejects_envelope_over_max_bytes() -> None:
    envelope = {"op": protocol.OP_CHAT, "blob": "a" * (protocol.MAX_LINE_BYTES + 16)}
    with pytest.raises(LLMMalformedResponse):
        protocol.encode_request(envelope)


def test_new_request_id_format_for_many_trials() -> None:
    seen = set()
    for _ in range(1000):
        rid = protocol.new_request_id()
        assert protocol.REQUEST_ID_RE.match(rid), rid
        seen.add(rid)
    assert len(seen) == 1000


def test_request_id_regex_rejects_uppercase_hex() -> None:
    assert protocol.REQUEST_ID_RE.match("r-ABCDEF012345") is None


def test_request_id_regex_rejects_wrong_length() -> None:
    assert protocol.REQUEST_ID_RE.match("r-abc") is None
    assert protocol.REQUEST_ID_RE.match("r-abcdef012345aa") is None


def test_request_id_regex_rejects_missing_prefix() -> None:
    assert protocol.REQUEST_ID_RE.match("abcdef012345") is None


@pytest.mark.parametrize("op", _OP_CONSTANTS)
def test_validate_request_accepts_legal_op(op: str) -> None:
    envelope: dict[str, object] = {"op": op}
    if op != protocol.OP_HELLO:
        envelope["id"] = _legal_id()
    protocol.validate_request(envelope)


def test_validate_request_rejects_unknown_op() -> None:
    with pytest.raises(LLMMalformedResponse):
        protocol.validate_request({"id": _legal_id(), "op": "nope.unknown"})


def test_validate_request_rejects_non_string_op() -> None:
    with pytest.raises(LLMMalformedResponse):
        protocol.validate_request({"id": _legal_id(), "op": 123})


def test_validate_request_rejects_missing_op() -> None:
    with pytest.raises(LLMMalformedResponse):
        protocol.validate_request({"id": _legal_id()})


def test_validate_request_accepts_hello_without_id() -> None:
    protocol.validate_request({"op": protocol.OP_HELLO})


def test_validate_request_requires_id_for_non_hello_ops() -> None:
    with pytest.raises(LLMMalformedResponse):
        protocol.validate_request({"op": protocol.OP_CHAT})


def test_validate_request_accepts_legal_id() -> None:
    protocol.validate_request({"id": "r-abcdef012345", "op": protocol.OP_HEALTH})


@pytest.mark.parametrize(
    "bad_id",
    [
        "r-ABCDEF012345",
        "r-abcdef0123",
        "r-abcdef0123456",
        "abcdef012345",
        "x-abcdef012345",
        "r_abcdef012345",
        "",
    ],
)
def test_validate_request_rejects_malformed_id(bad_id: str) -> None:
    with pytest.raises(LLMMalformedResponse):
        protocol.validate_request({"id": bad_id, "op": protocol.OP_HEALTH})


def test_validate_request_rejects_non_string_id() -> None:
    with pytest.raises(LLMMalformedResponse):
        protocol.validate_request({"id": 12345, "op": protocol.OP_HEALTH})


@pytest.mark.parametrize("not_a_dict", ["hello", 42, None, ["op", "hello"], b"hello"])
def test_validate_request_rejects_non_dict_envelope(not_a_dict: object) -> None:
    with pytest.raises(LLMMalformedResponse):
        protocol.validate_request(not_a_dict)  # type: ignore[arg-type]


def test_decoded_line_round_trip_preserves_order_insensitive_structure() -> None:
    payload = json.dumps({"id": "r-abcdef012345", "op": "chat"}).encode("utf-8")
    assert protocol.decode_line(payload) == {"id": "r-abcdef012345", "op": "chat"}


def test_protocol_version_is_a_small_integer_seed() -> None:
    # AC-2: the wire-format contract version is its own constant, seeded at 1,
    # and is NOT the package/marketing version.
    assert protocol.PROTOCOL_VERSION == 1
    assert isinstance(protocol.PROTOCOL_VERSION, int)


def test_protocol_version_is_not_derived_from_package_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A cosmetic package bump must NOT move PROTOCOL_VERSION — they are
    # independent. Patch package_version to a different string and assert the
    # protocol constant is unaffected (i.e. it is a fixed int, not derived).
    monkeypatch.setattr(protocol, "package_version", lambda: "9.9.9-cosmetic")
    assert protocol.PROTOCOL_VERSION == 1
    assert isinstance(protocol.PROTOCOL_VERSION, int)
    # And it is stable across calls regardless of the package version.
    assert protocol.PROTOCOL_VERSION == 1
