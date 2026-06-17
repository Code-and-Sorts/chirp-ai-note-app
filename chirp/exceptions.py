class ChirpException(Exception):
    pass


class AudioDeviceError(ChirpException):
    pass


class RecordingError(ChirpException):
    pass


class TranscriptionError(ChirpException):
    pass


class WhisperModelLoadError(TranscriptionError):
    """The Whisper model could not be downloaded or loaded.

    Raised when ``faster_whisper.WhisperModel(...)`` construction fails — for
    example on first run with no network, a Hugging Face rate-limit, a full
    disk, or a misconfigured model name. The message names the failure as a
    Whisper model download/load problem and suggests a remedy so the user can
    act rather than read a raw traceback.
    """


class NoteGenerationError(ChirpException):
    pass


class ConfigurationError(ChirpException):
    pass


class TemplateError(ChirpException):
    pass


class EmbedModelChanged(ChirpException):
    """The active embed model no longer matches the indexed vectors.

    Raised when an incremental index build detects that the embed alias stored
    on the Chroma collection differs from the one now configured; the caller
    should rebuild with ``chirp index --force`` rather than mix dimensions.
    """
