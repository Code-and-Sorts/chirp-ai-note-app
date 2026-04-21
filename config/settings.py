from pathlib import Path
from typing import Optional

import tomli_w
import tomllib
from platformdirs import user_documents_dir
from pydantic import BaseModel, Field, field_validator
from rich.console import Console


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
    sample_rate: int = 16000
    channels: int = 2
    chunk_size: int = 1024
    format: str = "wav"
    vad_enabled: bool = True
    vad_parameters: VadParameters = Field(default_factory=VadParameters)


class MonitoringConfig(BaseModel):
    warning_minutes: int = 60
    warning_interval: int = 15
    max_recording_hours: int = 8


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

    @classmethod
    def get_config_path(cls) -> Path:
        return default_chirp_home() / "config.toml"

    @classmethod
    def create_default_config(cls) -> "ChirpSettings":
        return cls()

    @classmethod
    def load_from_file(cls, config_path: Optional[Path] = None) -> "ChirpSettings":
        if config_path is None:
            config_path = cls.get_config_path()

        if not config_path.exists():
            console = Console()
            console.print(f"[blue]Creating default config at {config_path}[/blue]")

            settings = cls.create_default_config()
            settings.save_to_file(config_path)

            console.print("[green]✅ Default config created![/green]")
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
            if isinstance(nested, Path):
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
