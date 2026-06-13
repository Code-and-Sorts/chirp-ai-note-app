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
