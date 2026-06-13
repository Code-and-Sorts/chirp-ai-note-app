import os
import pathlib

from version_utils import build_pr_version, read_project_version, write_project_version


def main():
    pr_number = os.environ.get("PR_NUMBER")
    run_number = os.environ.get("RUN_NUMBER")

    if run_number is None:
        print("Error: RUN_NUMBER environment variable not set.")
        raise SystemExit(1)

    root_dir = pathlib.Path(__file__).parent.parent
    pyproject_path = root_dir / "pyproject.toml"

    print(
        f"Attempting to update version in {pyproject_path} (from {pathlib.Path.cwd()})"
    )

    current_full_version = read_project_version(pyproject_path)
    new_version = build_pr_version(current_full_version, run_number)

    expected_dev_suffix = f".dev{run_number}"
    if current_full_version.endswith(expected_dev_suffix):
        print(
            f"Version {current_full_version} already ends with {expected_dev_suffix}. Assuming already updated."
        )
        new_version = current_full_version

    print(f"Original full version in {pyproject_path}: {current_full_version}")
    print(
        f"Base version for new construction: {current_full_version.split('+', 1)[0].split('.dev', 1)[0]}"
    )
    print(f"Updating to PR dev version: {new_version} (PR: {pr_number})")

    _, version_file_path = write_project_version(root_dir, new_version)
    print(f"Successfully updated {pyproject_path} to version {new_version}")
    print(f"Successfully wrote version {new_version} to {version_file_path}")


if __name__ == "__main__":
    main()
