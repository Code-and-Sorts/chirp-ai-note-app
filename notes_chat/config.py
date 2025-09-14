from config.settings import ChirpSettings


def get_notes_config():
    return ChirpSettings.load_from_file()
