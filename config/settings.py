from pathlib import Path
from typing import Optional

import yaml
from platformdirs import user_config_dir, user_documents_dir
from pydantic import BaseModel, Field, field_validator
from rich.console import Console


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""

    pass


class DirectoriesConfig(BaseModel):
    raw_audio: Path = Field(
        default_factory=lambda: Path(user_documents_dir()) / "Chirp" / "recordings"
    )
    transcriptions: Path = Field(
        default_factory=lambda: Path(user_documents_dir()) / "Chirp" / "transcripts"
    )
    notes: Path = Field(
        default_factory=lambda: Path(user_documents_dir()) / "Chirp" / "notes"
    )
    templates: Path = Field(
        default_factory=lambda: Path(user_documents_dir()) / "Chirp" / "templates"
    )

    @field_validator("*", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class ModelsConfig(BaseModel):
    whisper: str = "base"
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
    input_device: Optional[str] = None
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
    index_dir: Path = Field(default_factory=lambda: Path(".notes_index"))
    auto_index: bool = True

    @field_validator("index_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class ChirpSettings(BaseModel):
    directories: DirectoriesConfig = Field(default_factory=DirectoriesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    notes_chat: NotesChatConfig = Field(default_factory=NotesChatConfig)

    @classmethod
    def get_config_path(cls) -> Path:
        """Get the platform-specific config file path."""
        config_dir = Path(user_config_dir("chirp"))
        return config_dir / "config.yaml"

    @classmethod
    def create_default_config(cls) -> "ChirpSettings":
        """Create a default config with user-friendly defaults."""

        data_dir = Path(user_documents_dir()) / "Chirp"

        settings = cls()
        settings.directories.raw_audio = data_dir / "recordings"
        settings.directories.transcriptions = data_dir / "transcripts"
        settings.directories.notes = data_dir / "notes"
        settings.directories.templates = data_dir / "templates"
        settings.notes_chat.index_dir = data_dir / "notes_index"

        return settings

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

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def save_to_file(self, config_path: Path):
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.dict()
        for key, value in config_dict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, Path):
                        config_dict[key][subkey] = str(subvalue)

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def ensure_directories_exist(self):
        for directory in [
            self.directories.raw_audio,
            self.directories.transcriptions,
            self.directories.notes,
            self.directories.templates,
            self.notes_chat.index_dir,
            self.notes_chat.index_dir / "chroma",
            self.notes_chat.index_dir / "cache",
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        from notes.template_engine import TemplateEngine

        template_engine = TemplateEngine(self)
        template_engine.create_default_templates()


def get_settings() -> ChirpSettings:
    return ChirpSettings.load_from_file()
