from config.settings import ChirpSettings

# Bump whenever SYSTEM_PROMPT / the grounded-answer prompt changes so the
# answer cache treats answers from an older prompt as stale. Lives here (a
# leaf module) rather than in notes_chat.prompting to keep notes_chat.cache
# free of a cache<->prompting import cycle.
PROMPT_VERSION = "2"


def get_notes_config():
    return ChirpSettings.load_from_file()
