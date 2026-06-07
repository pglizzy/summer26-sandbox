from pathlib import Path
import os


def find_project_root(start: Path | None = None) -> Path:
    """
    Walk upward until we find a project marker.
    This makes scripts work even if they are inside subfolders.
    """
    if start is None:
        start = Path(__file__).resolve()

    if start.is_file():
        start = start.parent

    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    raise FileNotFoundError("Could not find project root.")


PROJECT_ROOT = find_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = Path(os.getenv("POWER_MODEL_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()


def find_data_file(filename: str) -> Path:
    """
    Searches the data directory recursively for a file.
    Useful if the CSVs may move one or two folders deep.
    """
    matches = list(DATA_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find '{filename}' inside {DATA_DIR}"
        )

    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple files named '{filename}' found:\n"
            + "\n".join(str(p) for p in matches)
        )

    return matches[0]