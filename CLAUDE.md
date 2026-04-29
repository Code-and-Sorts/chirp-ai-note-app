# Claude Compatibility Notes

Use `AGENTS.md` as the canonical contributor guide for this repository.

## Working Agreement

- Read `AGENTS.md` first for project structure, commands, style, and testing rules.
- If `AGENTS.md` and this file ever disagree, follow `AGENTS.md`.
- When shared contributor guidance changes, update `AGENTS.md` and keep this file thin.

## Claude-Specific Reminders

- Prefer editing existing files over creating new ones.
- Keep comments rare and only for genuinely non-obvious intent.
- Validate doc updates against live CLI help before finishing.

## Current CLI Surface

- Visible commands: `record`, `transcribe`, `notes`, `ask`, `search`, `init`, `about`
- Hidden maintenance commands: `config`, `devices`, `index`
