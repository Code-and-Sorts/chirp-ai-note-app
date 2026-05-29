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

## `chirp models show` / `default` / `remove` / `pull` (story 4.5)

Same harness as `add` — invoke the sub-app directly until story 4.6 wires
`models` into the top-level CLI:

```bash
run_models() { uv run python -c "import sys; from llm.cli.models import app; sys.argv=['models',*sys.argv[1:]]; app()" "$@"; }
export CHIRP_REGISTRY_PATH="$(mktemp -d)/models.toml"
```

End-to-end sequence (AC-19), run on an Apple-silicon Mac with network access:

```bash
run_models add mlx-community/bge-small-en-v1.5 --no-warm   # register an embed model
run_models show bge-small-en-v1.5                          # panel: role embed, default yes, cache_path set
run_models add mlx-community/gemma-4-4b-it-4bit --no-warm  # register a chat model
run_models default bge-small-en-v1.5                       # flip embed default (no-op if already set)
run_models pull bge-small-en-v1.5 --no-warm                # cache-hit fast path: "Pulled … (cache hit)."
run_models remove bge-small-en-v1.5 --purge                # "Removed … and purged cache (<path>)."
run_models list                                            # bge gone; cache directory gone
```

JSON contract spot-check:

```bash
run_models show gemma-4-4b-it-4bit --json | jq -e .default   # -> true
```

`show` writes its panel (TTY) or JSON document to **stdout**; `default`,
`remove`, and `pull` print only to **stderr** under every condition.
