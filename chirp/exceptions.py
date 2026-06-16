class ChirpException(Exception):
    pass


class AudioDeviceError(ChirpException):
    pass


class RecordingError(ChirpException):
    pass


class TranscriptionError(ChirpException):
    pass


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
