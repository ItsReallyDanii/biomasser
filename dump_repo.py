import os
import sys

# --- Configuration ---
# Folders to skip (e.g., large data, results, environment files)
SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", "data", "results"}

def dump_repository(root_dir, output_file="goodhope_dump.txt"):
    """Walks the directory and compiles all file content into a single output file."""
    
    # 1. Gather all file paths
    files_to_dump = []
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to prune directories we don't want to visit next
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for file in files:
            full_path = os.path.join(root, file)
            files_to_dump.append(full_path)

    # Sort files for consistent output ordering
    files_to_dump.sort()

    # 2. Compile content
    out = []
    root_abs = os.path.abspath(root_dir)

    for file_path in files_to_dump:
        # Get the relative path for the label
        rel_path = os.path.relpath(file_path, root_dir)
        
        # Skip output files that might be in the root directory
        if rel_path == output_file:
            continue

        text = ""
        try:
            # Attempt to read as UTF-8 (standard for code files)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Fallback for binary or non-UTF8 files (will include garbage characters, but dumps content)
            print(f"Warning: Non-UTF8 content in {rel_path}. Reading as bytes.")
            try:
                with open(file_path, 'rb') as f:
                    text = f.read().decode('latin-1') # Use a simple, lossy decode for the dump
            except Exception as e:
                text = f"--- ERROR READING FILE: {e} ---"

        # Format the entry with clear labels
        out.append(f"========== FILE: {rel_path} ==========\n\n{text}\n\n")

    # 3. Write to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write("".join(out))
        print(f"Successfully compiled all files into {output_file}!")
    except Exception as e:
        print(f"ERROR WRITING OUTPUT FILE: {e}")

# --- Execution ---
if __name__ == "__main__":
    # Allows passing arguments from the command line interface
    # e.g., python dump_repo.py . my_code_dump.txt
    
    # Default to current directory and default output name
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_name = sys.argv[2] if len(sys.argv) > 2 else "goodhope_dump.txt"

    dump_repository(target_dir, output_name)