"""Create a deterministic anonymous source ZIP with no local filesystem metadata."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
from check_release import scan
ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {"README.md", "LICENSE", "NOTICE", "pyproject.toml", "MANIFEST.in", ".gitignore"}
SOURCE_DIRS = {"sim_eval", "tests", "scripts", "tools", "docs", "configs", "examples", "third_party"}
EXCLUDED = {"__pycache__", ".cache", ".git", ".venv", "tau-bench"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/indieval-anonymous.zip")
    parser.add_argument("--deny-file", type=Path)
    args = parser.parse_args()
    deny = [x.strip() for x in args.deny_file.read_text().splitlines() if x.strip()] if args.deny_file else []
    audit = scan(ROOT, deny)
    if audit["status"] != "passed":
        print(json.dumps(audit, indent=2))
        return 1
    paths = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file() or path.is_symlink(): continue
        if any(x in EXCLUDED or x.endswith(".egg-info") for x in rel.parts): continue
        if path.name == ".DS_Store": continue
        if (len(rel.parts) == 1 and rel.name in ROOT_FILES) or rel.parts[0] in SOURCE_DIRS:
            paths.append(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            name = "indieval/" + path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".zip.sha256").write_text(digest + "  " + args.output.name + "\n")
    print(json.dumps({"status":"passed", "source_files":len(paths), "sha256":digest}, indent=2))
    return 0
if __name__ == "__main__": raise SystemExit(main())
