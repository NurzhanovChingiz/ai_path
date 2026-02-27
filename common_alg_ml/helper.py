"""Helper utilities for ML project structure."""
from pathlib import Path


def save_py_files_structure(base_folder: Path, output_file: Path) -> None:
    """Traverse base_folder and its subdirectories, save the directory structure of all .py files into a single file, indicating the file name as comments.

    Args:
        base_folder (Path): The root directory to start the search.
        output_file (Path): The file to save the structure to.
    """
    with output_file.open("w", encoding="utf-8") as outfile:
        for root, _, files in base_folder.walk():
            if ".git" in root.parts:
                continue  # Skip .git directories
            level = len(root.relative_to(base_folder).parts)
            indent = "│   " * (level - 1) + ("├── " if level > 0 else "")
            if level == 0:
                outfile.write(f"{root.name}/\n")
            else:
                outfile.write(f"{indent}{root.name}/\n")

            sub_indent = "│   " * level + "├── "
            py_files = [file for file in files if file.endswith(".py")]
            outfile.writelines(f"{sub_indent}{file}\n" for file in py_files)
