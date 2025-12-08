#!/usr/bin/env python3
import os
import sys
import base64

# Extensions we'll treat as text and dump directly
TEXT_EXTS = {
    ".txt", ".py", ".md", ".rst", ".csv",
    ".json", ".yaml", ".yml", ".ini", ".cfg",
    ".toml", ".xml", ".html", ".htm", ".css",
    ".js", ".ts", ".ipynb"  # ipynb is JSON anyway
}

# Directories to skip
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".idea", ".DS_Store"}

def is_text_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in TEXT_EXTS

def main():
    if len(sys.argv) < 2:
        print("Usage: python bundle_repo.py <root_dir> [output_file]")
        sys.exit(1)

    root_dir = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "repo_bundle.txt"

    with open(out_file, "w", encoding="utf-8") as out:
        out.write(f"# Repo bundle for: {os.path.abspath(root_dir)}\n\n")

        for dirpath, dirnames, filenames in os.walk(root_dir):
            # prune dirs
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

            for fname in sorted(filenames):
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root_dir)
                ext = os.path.splitext(fname)[1].lower() or "<no-ext>"

                out.write("\n" + "=" * 80 + "\n")
                out.write(f"FILE: {rel_path}\n")
                out.write(f"EXT:  {ext}\n")
                out.write("=" * 80 + "\n")

                try:
                    if is_text_file(full_path):
                        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                            out.write(f.read())
                    else:
                        with open(full_path, "rb") as f:
                            data = f.read()
                        b64 = base64.b64encode(data).decode("ascii")
                        out.write("\n[BINARY FILE – base64 encoded below]\n\n")
                        out.write(b64)
                        out.write("\n")
                except Exception as e:
                    out.write(f"\n[ERROR reading file: {e}]\n")

    print(f"Done. Bundle written to {out_file}")

if __name__ == "__main__":
    main()