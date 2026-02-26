"""Helper utilities for ML project structure."""
import os


def save_py_files_structure(base_folder: str, output_file: str) -> None:
    """Traverse base_folder and its subdirectories, save the directory structure of all .py files into a single file, indicating the file name as comments.

    Args:
        base_folder (str): The root directory to start the search.
        output_file (str): The file to save the structure to.
    """
    with open(output_file, "w", encoding="utf-8") as outfile:
        for root, _, files in os.walk(base_folder):
            if ".git" in root:
                continue  # Skip .git directories
            level = root.replace(base_folder, "").count(os.sep)
            indent = "│   " * (level - 1) + ("├── " if level > 0 else "")
            if level == 0:
                outfile.write(f"{os.path.basename(root)}/\n")
            else:
                outfile.write(f"{indent}{os.path.basename(root)}/\n")

            sub_indent = "│   " * level + "├── "
            py_files = [file for file in files if file.endswith(".py")]
            for file in py_files:
                outfile.write(f"{sub_indent}{file}\n")
