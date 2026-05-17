import os
import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from platformdirs import user_documents_dir
from pydantic import BaseModel, Field, field_validator
from rich.console import Console

CHIRP_DAEMON_SOCKET_ENV = "CHIRP_DAEMON_SOCKET"
CHIRP_MODEL_IDLE_TIMEOUT_ENV = "CHIRP_MODEL_IDLE_TIMEOUT"

DEFAULT_DAEMON_SOCKET = (
    Path.home() / "Library" / "Application Support" / "chirp" / "chirpd.sock"
)
DEFAULT_IDLE_TIMEOUT_SECONDS = 300.0


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


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""

    pass


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
    ollama_url: str = "http://localhost:11434"
    num_predict: int = 4096


class VadParameters(BaseModel):
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 1000
    max_speech_duration_s: int = 30
    speech_pad_ms: int = 300


class AudioConfig(BaseModel):
    # The bundled CaptureAudio.app helper produces 16 kHz mono float32
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

    @field_validator("daemon_socket", mode="before")
    @classmethod
    def convert_to_path(cls, value):
        if value is None or value == "":
            return None
        return Path(value) if isinstance(value, str) else value


class NotesChatConfig(BaseModel):
    emb_model: str = "nomic-embed-text"
    chunk_size: int = 1000
    overlap: int = 200
    k: int = 10
    ctx_char_budget: int = 8000
    index_dir: Path = Field(default_factory=default_chirp_home)
    auto_index: bool = True

    @field_validator("index_dir", mode="before")
    @classmethod
    def convert_to_path(cls, value):
        return Path(value) if isinstance(value, str) else value


class ChirpSettings(BaseModel):
    directories: DirectoriesConfig = Field(default_factory=DirectoriesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    notes_chat: NotesChatConfig = Field(default_factory=NotesChatConfig)
    llm: LLMSettings = Field(default_factory=LLMSettings)

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

        with open(config_path, "rb") as config_file:
            config_data = tomllib.load(config_file)

        return cls(**config_data)

    def save_to_file(self, config_path: Path):
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.model_dump()
        _stringify_paths(config_dict)

        with open(config_path, "wb") as config_file:
            tomli_w.dump(config_dict, config_file)

    def ensure_directories_exist(self):
        chirp_home = default_chirp_home()
        for directory in [
            self.directories.notes_root,
            chirp_home,
            self.notes_chat.index_dir,
            self.notes_chat.index_dir / "chroma",
        ]:
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
