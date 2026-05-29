# `llm` test notes

The automated suite under `tests/llm/` mocks every I/O boundary — HuggingFace
(`llm.hf`), the daemon client (`llm.client.LLMClient`), and the registry path
(redirected with the `CHIRP_REGISTRY_PATH` env var) — so it runs on Linux CI
with no MLX, network, or daemon subprocess.

A few behaviors need a real Mac to confirm. Run these by hand before moving a
story to Review.

## `chirp models add` (story 4.3)

These require an Apple-silicon Mac with `huggingface_hub` + `mlx-lm` installed
and network access. Until story 4.6 wires `models` into the top-level CLI,
invoke the sub-app directly:

```bash
run_add() { uv run python -c "import sys; from llm.cli.models import app; sys.argv=['models','add',*sys.argv[1:]]; app()" "$@"; }
```

For the **`--no-warm`** check, point `CHIRP_REGISTRY_PATH` at a scratch file so
you don't touch your real registry:

```bash
export CHIRP_REGISTRY_PATH="$(mktemp -d)/models.toml"
```

For the **auto-warm** check, leave `CHIRP_REGISTRY_PATH` unset: the daemon reads
the default registry path (`~/Library/Application Support/chirp/models.toml`)
and the env override is only honored by the CLI, so a scratch path would make
the daemon's `model.load` miss the alias. Remove the temp `models.toml` and stop
`chirpd` afterward if you were starting from an empty registry.

1. **First-add happy path (auto-default + auto-warm).**
   ```bash
   run_add mlx-community/Qwen2.5-0.5B-Instruct-4bit   # CHIRP_REGISTRY_PATH unset
   ```
   Expect: a Rich progress bar during download (status lines off-TTY), weights
   cached under `~/.cache/huggingface/hub/`, `models.toml` written with the
   alias as `default_chat`, and the model warmed on a lazy-spawned daemon
   (`Ready.`). Verified ≈11s on a warm cache / fast connection (≤ 30s target).

2. **`--no-warm` skips the daemon.**
   ```bash
   run_add mlx-community/bge-small-en-v1.5-bf16 --no-warm
   ```
   Expect: download + registry write with the alias as `default_embed`, **no**
   daemon spawn, no `Warming…`/`Ready.` lines. `chirp models list` (story 4.4)
   will later show the model registered but not loaded.

All progress, status, and error output goes to **stderr**; stdout stays empty,
so `chirp models add … && …` pipelines stay clean.
