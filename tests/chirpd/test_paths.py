"""Layout invariants for the chirpd runtime filesystem constants."""

from __future__ import annotations

from chirpd.paths import (
    APP_SUPPORT_DIR,
    LOCK_PATH,
    LOG_DIR,
    LOG_FILE,
    MODELS_TOML_PATH,
    RUNTIME_DIR_MODE,
    SOCKET_PATH,
)


def test_log_file_lives_inside_log_dir() -> None:
    assert LOG_FILE.parent == LOG_DIR


def test_lock_path_lives_inside_app_support_dir() -> None:
    assert LOCK_PATH.parent == APP_SUPPORT_DIR


def test_socket_path_lives_inside_app_support_dir() -> None:
    assert SOCKET_PATH.parent == APP_SUPPORT_DIR


def test_runtime_dir_mode_is_owner_only() -> None:
    assert RUNTIME_DIR_MODE == 0o700


def test_log_dir_is_under_macos_library_logs() -> None:
    assert LOG_DIR.parts[-3:] == ("Library", "Logs", "chirp")


def test_app_support_dir_is_under_macos_application_support() -> None:
    assert APP_SUPPORT_DIR.parts[-2:] == ("Application Support", "chirp")


def test_models_toml_path_lives_inside_app_support_dir() -> None:
    assert MODELS_TOML_PATH.parent == APP_SUPPORT_DIR
    assert MODELS_TOML_PATH.name == "models.toml"
