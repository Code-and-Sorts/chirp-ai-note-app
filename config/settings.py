import os
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import tomli_w
from platformdirs import user_documents_dir
from pydantic import BaseModel, Field, ValidationError, field_validator
from rich.console import Console

SUPPORTED_CONFIG_SCHEMA_VERSION = 1

_MISSING = object()

CHIRP_DAEMON_SOCKET_ENV = "CHIRP_DAEMON_SOCKET"
CHIRP_MODEL_IDLE_TIMEOUT_ENV = "CHIRP_MODEL_IDLE_TIMEOUT"
CHIRP_INFERENCE_TIMEOUT_ENV = "CHIRP_INFERENCE_TIMEOUT"
CHIRP_MAX_RESIDENT_CHAT_ENV = "CHIRP_MAX_RESIDENT_CHAT"
CHIRP_MAX_RESIDENT_EMBED_ENV = "CHIRP_MAX_RESIDENT_EMBED"

DEFAULT_IDLE_TIMEOUT_SECONDS = 300.0
# Per inter-event read budget, not per request, so it only fires on a wedged
# daemon. The first event spans cold load + first token, hence the larger budget.
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 60.0
DEFAULT_FIRST_EVENT_TIMEOUT_SECONDS = 120.0
# Per-role resident cap; mirrors chirpd.state's defaults so config and state agree.
DEFAULT_MAX_RESIDENT_CHAT = 1
DEFAULT_MAX_RESIDENT_EMBED = 1


def get_daemon_socket_override() -> Path | None:
    override = os.environ.get(CHIRP_DAEMON_SOCKET_ENV)
    return Path(override) if override else None


def get_idle_timeout_override() -> float | None:
    override = os.environ.get(CHIRP_MODEL_IDLE_TIMEOUT_ENV)
    if not override:
        return None
    try:
        return float(override)
    except ValueError:
        return None


def resolve_idle_timeout_seconds() -> float:
    override = get_idle_timeout_override()
    if override is not None:
        return override
    try:
        return get_settings().llm.idle_timeout_seconds
    except Exception:  # noqa: BLE001 — config failures must not block daemon start
        return DEFAULT_IDLE_TIMEOUT_SECONDS


def get_inference_timeout_override() -> float | None:
    override = os.environ.get(CHIRP_INFERENCE_TIMEOUT_ENV)
    if not override:
        return None
    try:
        return float(override)
    except ValueError:
        return None


def resolve_inference_timeout_seconds() -> float:
    override = get_inference_timeout_override()
    if override is not None:
        return override
    try:
        return get_settings().llm.inference_timeout_seconds
    except Exception:  # noqa: BLE001 — config failures must not block daemon start
        return DEFAULT_INFERENCE_TIMEOUT_SECONDS


def _resolve_resident_cap(env_var: str, config_value: int) -> int:
    override = os.environ.get(env_var)
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    return config_value


def resolve_max_resident_chat() -> int:
    try:
        configured = get_settings().llm.max_resident_chat
    except Exception:  # noqa: BLE001 — config failures must not block daemon start
        configured = DEFAULT_MAX_RESIDENT_CHAT
    return _resolve_resident_cap(CHIRP_MAX_RESIDENT_CHAT_ENV, configured)


def resolve_max_resident_embed() -> int:
    try:
        configured = get_settings().llm.max_resident_embed
    except Exception:  # noqa: BLE001 — config failures must not block daemon start
        configured = DEFAULT_MAX_RESIDENT_EMBED
    return _resolve_resident_cap(CHIRP_MAX_RESIDENT_EMBED_ENV, configured)


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""


def default_notes_root() -> Path:
    return Path(user_documents_dir()) / "chirp"


def default_chirp_home() -> Path:
    return Path.home() / ".chirp"


def default_chroma_dir() -> Path:
    return default_chirp_home() / "chroma"


class DirectoriesConfig(BaseModel):
    notes_root: Path = Field(default_factory=default_notes_root)

    @field_validator("notes_root", mode="before")
    @classmethod
    def convert_to_path(cls, value):
        return Path(value) if isinstance(value, str) else value


class ModelsConfig(BaseModel):
    whisper: str = "large-v3-turbo"
    llm: str = "llama3.1:8b"
    num_predict: int = 4096
    context_window: int = 32768


class VadParameters(BaseModel):
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 1000
    max_speech_duration_s: int = 30
    speech_pad_ms: int = 300


class AudioConfig(BaseModel):
    # The bundled Chirp.app helper produces 16 kHz mono float32
    # frames; the recorder paths write 16 kHz mono int16 WAVs. The fields
    # here describe the recorder's effective output format so the value
    # `chirp config` displays matches what's actually written to disk.
    # Overrides are not honored — both AudioRecorder.start_recording and
    # LiveAudioStream.start log a warning when a non-default value is
    # configured.
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: str = "wav"
    vad_enabled: bool = True
    vad_parameters: VadParameters = Field(default_factory=VadParameters)


class MonitoringConfig(BaseModel):
    warning_minutes: int = 60
    warning_interval: int = 15
    max_recording_hours: int = 8


class LLMSettings(BaseModel):
    backend: Literal["chirpd"] = "chirpd"
    daemon_socket: Path | None = None
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS
    inference_timeout_seconds: float = DEFAULT_INFERENCE_TIMEOUT_SECONDS
    max_resident_chat: int = DEFAULT_MAX_RESIDENT_CHAT
    max_resident_embed: int = DEFAULT_MAX_RESIDENT_EMBED

    @field_validator("daemon_socket", mode="before")
    @classmethod
    def convert_to_path(cls, value):
        if value is None or value == "":
            return None
        return Path(value) if isinstance(value, str) else value


class NotesChatConfig(BaseModel):
    semantic_enabled: bool = False
    recommended_embed_model: str = "bge-small-en-v1.5-bf16"
    chunk_size: int = 1000
    overlap: int = 200
    k: int = 10
    ctx_char_budget: int = 8000
    index_dir: Path = Field(
        default_factory=default_chirp_home,
        description=(
            "Root directory for search-index artifacts. The Chroma vector store "
            "lives at index_dir/chroma; manifest.json and bm25.json sit directly "
            "under index_dir, and per-query caches under index_dir/cache."
        ),
    )
    auto_index: bool = True

    @field_validator("index_dir", mode="before")
    @classmethod
    def convert_to_path(cls, value):
        return Path(value) if isinstance(value, str) else value


class InitConfig(BaseModel):
    launch_agent_prompted_at: datetime | None = None


class ChirpSettings(BaseModel):
    schema_version: int = SUPPORTED_CONFIG_SCHEMA_VERSION
    directories: DirectoriesConfig = Field(default_factory=DirectoriesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    notes_chat: NotesChatConfig = Field(default_factory=NotesChatConfig)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    init: InitConfig = Field(default_factory=InitConfig)

    @field_validator("init", mode="before")
    @classmethod
    def _coerce_non_table_init(cls, value):
        # config.toml is user-editable; a hand-written non-table `init` value
        # (e.g. ``init = "..."``) must not fail validation and block every CLI
        # command at load time. Fall back to defaults instead.
        if isinstance(value, dict | InitConfig):
            return value
        return {}

    @classmethod
    def get_config_path(cls) -> Path:
        return default_chirp_home() / "config.toml"

    @classmethod
    def create_default_config(cls) -> "ChirpSettings":
        return cls()

    @classmethod
    def load_from_file(cls, config_path: Path | None = None) -> "ChirpSettings":
        if config_path is None:
            config_path = cls.get_config_path()

        if not config_path.exists():
            console = Console()
            console.print(f"[blue]Creating default config at {config_path}[/blue]")

            settings = cls.create_default_config()
            settings.save_to_file(config_path)

            console.print("[green]Default config created.[/green]")
            console.print(f"[dim]Edit {config_path} to customize settings[/dim]")
            return settings

        warnings = Console(stderr=True)
        try:
            with config_path.open("rb") as config_file:
                config_data = tomllib.load(config_file)
        except tomllib.TOMLDecodeError as error:
            cls._emit_warning(
                warnings,
                f"Warning: {config_path} is not valid TOML ({error}); falling "
                "back to default settings. Fix the file or run `chirp init` to "
                "reset it.",
            )
            return cls()

        cls._warn_on_schema_version(config_data, config_path, warnings)
        cls._warn_on_unknown_top_level_keys(config_data, config_path, warnings)
        return cls._validate_tolerantly(config_data, config_path, warnings)

    @staticmethod
    def _emit_warning(console: Console, message: str) -> None:
        # markup=False + soft_wrap keep paths and bracketed section names intact:
        # Rich would treat ``[monitoring]`` as a style tag and wrap long paths.
        console.print(message, style="yellow", markup=False, soft_wrap=True)

    @classmethod
    def _warn_on_schema_version(
        cls, config_data: dict[str, Any], config_path: Path, warnings: Console
    ) -> None:
        version = config_data.get("schema_version")
        # Only warn on a well-formed but unsupported version; a non-int is left
        # to the per-field recovery path so a bad type does not double-warn.
        if isinstance(version, int) and version != SUPPORTED_CONFIG_SCHEMA_VERSION:
            cls._emit_warning(
                warnings,
                f"Warning: {config_path} has schema_version {version!r}, which "
                f"this chirp does not recognise "
                f"(supported: {SUPPORTED_CONFIG_SCHEMA_VERSION}). Continuing with "
                "best-effort defaults; delete the file or run `chirp init` to "
                "reset it.",
            )

    @classmethod
    def _warn_on_unknown_top_level_keys(
        cls, config_data: dict[str, Any], config_path: Path, warnings: Console
    ) -> None:
        unknown_keys = set(config_data) - set(cls.model_fields)
        for key in sorted(unknown_keys):
            cls._emit_warning(
                warnings,
                f"Warning: unknown config section [{key}] in {config_path} was "
                "ignored. Check for a typo or a renamed section.",
            )

    @classmethod
    def _validate_tolerantly(
        cls, config_data: dict[str, Any], config_path: Path, warnings: Console
    ) -> "ChirpSettings":
        recovered = dict(config_data)
        for _ in range(len(recovered) + 1):
            try:
                return cls(**recovered)
            except ValidationError as error:
                dropped = cls._drop_invalid_fields(
                    recovered, error, config_path, warnings
                )
                if not dropped:
                    break

        cls._emit_warning(
            warnings,
            f"Warning: {config_path} could not be fully parsed; falling back to "
            "default settings.",
        )
        return cls()

    @classmethod
    def _drop_invalid_fields(
        cls,
        recovered: dict[str, Any],
        error: ValidationError,
        config_path: Path,
        warnings: Console,
    ) -> bool:
        dropped = False
        for entry in error.errors():
            location = entry.get("loc", ())
            if not location:
                continue
            field_path = ".".join(str(part) for part in location)
            section = location[0]
            if section not in recovered:
                continue
            if cls._drop_field(recovered, location):
                dropped = True
                cls._emit_warning(
                    warnings,
                    f"Warning: config field '{field_path}' in {config_path} is "
                    f"invalid (value {entry.get('input')!r}); using the default "
                    "instead.",
                )
        return dropped

    @classmethod
    def _drop_field(cls, recovered: dict[str, Any], location: tuple) -> bool:
        if len(location) == 1:
            return recovered.pop(location[0], _MISSING) is not _MISSING
        parent = recovered.get(location[0])
        if not isinstance(parent, dict):
            return recovered.pop(location[0], _MISSING) is not _MISSING
        return parent.pop(location[1], _MISSING) is not _MISSING

    def save_to_file(self, config_path: Path):
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.model_dump()
        _stringify_paths(config_dict)

        with config_path.open("wb") as config_file:
            tomli_w.dump(config_dict, config_file)

    def ensure_directories_exist(self):
        chirp_home = default_chirp_home()
        directories = [
            self.directories.notes_root,
            chirp_home,
            self.notes_chat.index_dir,
        ]
        if self.notes_chat.semantic_enabled:
            directories.append(self.notes_chat.index_dir / "chroma")
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def _stringify_paths(value):
    if isinstance(value, dict):
        for key, nested in list(value.items()):
            if nested is None:
                del value[key]
            elif isinstance(nested, Path):
                value[key] = str(nested)
            else:
                _stringify_paths(nested)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, Path):
                value[index] = str(item)
            else:
                _stringify_paths(item)


def get_settings() -> ChirpSettings:
    return ChirpSettings.load_from_file()
