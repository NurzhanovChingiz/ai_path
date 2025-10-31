import os

def save_py_files_structure(base_folder, output_file):
    """
    Traverse base_folder and its subdirectories, save the directory structure
    of all .py files into a single file, indicating the file name as comments.

    :param base_folder: Directory to start traversal from.
    :param output_file: Path to the output file.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(base_folder):
            if '.git' in root: 
                continue  # Skip .git directories
            level = root.replace(base_folder, '').count(os.sep)
            indent = '│   ' * (level - 1) + ('├── ' if level > 0 else '')
            if level == 0:
                outfile.write(f"{os.path.basename(root)}/\n")
            else:
                outfile.write(f"{indent}{os.path.basename(root)}/\n")

            sub_indent = '│   ' * level + '├── '
            py_files = [file for file in files if file.endswith('.py')]
            for file in py_files:
                outfile.write(f"{sub_indent}{file}\n")
if __name__ == "__main__":
# Example usage:
    base_folder = 'C:\\code\\cv\\ai_path\\ai_path'
    output_file = 'structure.txt'
    save_py_files_structure(base_folder, output_file)
