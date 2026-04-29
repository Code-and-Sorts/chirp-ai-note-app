import os
import pathlib

from version_utils import normalize_release_tag, write_project_version


def main():
    release_tag = os.environ.get("RELEASE_TAG")
    if not release_tag:
        print("Error: RELEASE_TAG environment variable not set.")
        raise SystemExit(1)

    root_dir = pathlib.Path(__file__).parent.parent
    normalized_version = normalize_release_tag(release_tag)
    current_version, version_file_path = write_project_version(
        root_dir, normalized_version
    )
    pyproject_path = root_dir / "pyproject.toml"

    print(f"Attempting to update version in {pyproject_path} (from {os.getcwd()})")
    print(f"Original full version in {pyproject_path}: {current_version}")
    print(f"Updating to release version: {normalized_version} (tag: {release_tag})")
    print(f"Successfully updated {pyproject_path} to version {normalized_version}")
    print(f"Successfully wrote version {normalized_version} to {version_file_path}")


if __name__ == "__main__":
    main()
