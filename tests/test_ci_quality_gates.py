"""CI/release quality-gate invariants (story 8.6).

Both gating workflows must measure the same coverage surface (the full
``[tool.coverage.run] source`` set), the coverage floor must live in a single
source of truth (``pyproject.toml``), and Renovate must not auto-merge the
exact-pinned MLX deps whose real inference path is never exercised by CI.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _pytest_cov_flags(command: str) -> list[str]:
    return re.findall(r"--cov(?:=[\w./-]+)?(?![\w-])", command)


def _run_test_commands(workflow_path: Path) -> list[str]:
    document = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    commands: list[str] = []
    for job in document["jobs"].values():
        for step in job.get("steps", []):
            run = step.get("run")
            if run and "pytest" in run and "--cov" in run:
                commands.append(run)
    return commands


def test_fail_under_is_single_source_of_truth():
    report = _pyproject()["tool"]["coverage"]["report"]
    assert "fail_under" in report
    assert isinstance(report["fail_under"], int)
    assert report["fail_under"] <= 84


def test_both_gating_workflows_use_bare_cov():
    for workflow in ("main-build.yml", "shared-build-and-test.yaml"):
        commands = _run_test_commands(WORKFLOWS / workflow)
        assert commands, f"no pytest --cov step found in {workflow}"
        for command in commands:
            flags = _pytest_cov_flags(command)
            assert flags, f"{workflow} pytest step is missing --cov"
            assert flags == ["--cov"], (
                f"{workflow} must use bare --cov (full pyproject source set), "
                f"not a hand-listed package set: {flags}"
            )


def test_mlx_deps_excluded_from_renovate_automerge():
    with (REPO_ROOT / "renovate.json").open(encoding="utf-8") as handle:
        renovate = json.load(handle)

    mlx_rules = [
        rule
        for rule in renovate.get("packageRules", [])
        if {"mlx-lm", "mlx-embeddings"} <= set(rule.get("matchPackageNames", []))
    ]
    assert mlx_rules, "renovate.json must have a rule covering the MLX deps"
    assert all(rule.get("automerge") is False for rule in mlx_rules)


def test_mypy_stage_one_flags_present_without_strict_flip():
    mypy = _pyproject()["tool"]["mypy"]
    assert mypy["warn_unused_ignores"] is True
    assert mypy["no_implicit_optional"] is True
    assert mypy["disallow_untyped_defs"] is False
    assert mypy["ignore_missing_imports"] is True
