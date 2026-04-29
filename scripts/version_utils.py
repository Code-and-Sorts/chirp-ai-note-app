import pathlib
import re
import tomllib


def normalize_release_tag(tag: str) -> str:
    normalized_tag = tag.strip()
    if not normalized_tag:
        raise ValueError("Release tag is empty.")

    normalized_tag = normalized_tag.removeprefix("refs/tags/")
    if (
        normalized_tag.startswith("v")
        and len(normalized_tag) > 1
        and normalized_tag[1].isdigit()
    ):
        normalized_tag = normalized_tag[1:]

    if not normalized_tag:
        raise ValueError("Release tag did not contain a version.")

    return normalized_tag


def build_pr_version(base_version: str, run_number: str) -> str:
    normalized_base_version = base_version

    if "+" in normalized_base_version:
        normalized_base_version = normalized_base_version.split("+", 1)[0]
    if ".dev" in normalized_base_version:
        parts = normalized_base_version.split(".dev")
        if len(parts) > 1 and parts[-1].isdigit():
            normalized_base_version = parts[0]

    return f"{normalized_base_version}.dev{run_number}"


def read_project_version(pyproject_path: pathlib.Path) -> str:
    with pyproject_path.open("rb") as file_handle:
        data = tomllib.load(file_handle)

    project_data = data.get("project")
    if not isinstance(project_data, dict) or "version" not in project_data:
        raise ValueError(f"Could not find ['project']['version'] in {pyproject_path}")

    return str(project_data["version"])


def write_project_version(
    root_dir: pathlib.Path, new_version: str
) -> tuple[str, pathlib.Path]:
    pyproject_path = root_dir / "pyproject.toml"
    current_version = read_project_version(pyproject_path)

    with pyproject_path.open(encoding="utf-8") as file_handle:
        content = file_handle.read()

    version_pattern = re.compile(
        rf'(?m)^(version\s*=\s*"){re.escape(current_version)}(")$'
    )
    updated_content, replacements = version_pattern.subn(
        rf"\g<1>{new_version}\g<2>", content, count=1
    )

    if replacements != 1:
        raise ValueError(f"Could not replace version line in {pyproject_path}")

    with pyproject_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write(updated_content)

    dist_dir = root_dir / "dist"
    dist_dir.mkdir(exist_ok=True)
    version_file_path = dist_dir / "VERSION.txt"
    with version_file_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write(new_version)

    return current_version, version_file_path
