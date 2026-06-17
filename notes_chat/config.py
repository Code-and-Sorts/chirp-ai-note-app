from config.settings import ChirpSettings

# Bump on any prompt change so the answer cache treats older-prompt answers as
# stale. Lives in this leaf module (not notes_chat.prompting) to avoid a
# cache<->prompting import cycle.
PROMPT_VERSION = "2"


def get_notes_config():
    return ChirpSettings.load_from_file()
