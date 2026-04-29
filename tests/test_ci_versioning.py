import importlib.util
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


version_utils = _load_module("version_utils", "scripts/version_utils.py")


def test_normalize_release_tag_strips_refs_prefix_and_v_prefix():
    assert version_utils.normalize_release_tag("refs/tags/v1.2.3") == "1.2.3"


def test_build_pr_version_drops_existing_dev_and_local_suffixes():
    assert version_utils.build_pr_version("1.2.3.dev9+local", "42") == "1.2.3.dev42"


def test_write_project_version_updates_pyproject_and_version_artifact(tmp_path):
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        "\n".join(
            [
                "[project]",
                'name = "chirp-notes-ai"',
                'version = "0.1.0"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    previous_version, version_file_path = version_utils.write_project_version(
        tmp_path, "2.0.0"
    )

    assert previous_version == "0.1.0"
    assert 'version = "2.0.0"' in pyproject_path.read_text(encoding="utf-8")
    assert version_file_path.read_text(encoding="utf-8") == "2.0.0"
