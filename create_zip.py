
from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent
ZIP_NAME = "flood_risk_project.zip"

# Folders/files to include
INCLUDE = [
    "src",
    "dashboard",
    "outputs",
    "models",
    "docs",
    "config.py",
    "utils.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
]

# Directories and files to exclude
EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
    "raw_data",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
}

zip_path = PROJECT_ROOT / ZIP_NAME

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for item in INCLUDE:
        path = PROJECT_ROOT / item

        if not path.exists():
            print(f"Skipping missing: {item}")
            continue

        paths = [path] if path.is_file() else path.rglob("*")

        for file in paths:
            if not file.is_file():
                continue

            relative = file.relative_to(PROJECT_ROOT)

            if any(part in EXCLUDE_DIRS for part in relative.parts):
                continue

            if file.suffix in EXCLUDE_SUFFIXES:
                continue

            if file == zip_path:
                continue

            zipf.write(file, relative)

print(f"\nZIP created successfully: {zip_path}")
print(f"Size: {zip_path.stat().st_size / (1024**2):.2f} MB")
