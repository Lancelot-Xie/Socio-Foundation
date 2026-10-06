"""Scan the public source tree for accidental private resources and metadata."""
from pathlib import Path
import argparse
import json
import re
ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", "__pycache__", ".cache", "build", "dist", "artifacts", "reports", "local_data", "models"}
PATTERNS = {
    "personal_home_path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "private_cluster_path": re.compile(r"/(?:inspire|mnt|scratch)/(?:hdd|global_user|users)/"),
    "private_key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "credential_token": re.compile(r"(?<![A-Za-z0-9_])(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,})"),
}
def scan(root, deny=()):
    findings=[];count=0
    for p in sorted(root.rglob("*")):
        rel=p.relative_to(root)
        if any(part in IGNORED or part.endswith(".egg-info") for part in rel.parts):continue
        if p.is_symlink():findings.append({"file":str(rel),"kind":"symlink"});continue
        if not p.is_file() or p.suffix in {".zip", ".whl", ".gz"}:continue
        count+=1
        if p.name.startswith(".env") and p.name!=".env.example":findings.append({"file":str(rel),"kind":"environment_file"})
        try:text=p.read_text()
        except UnicodeError:
            findings.append({"file":str(rel),"kind":"unexpected_binary"});continue
        for name,pattern in PATTERNS.items():
            if pattern.search(text):findings.append({"file":str(rel),"kind":name})
        lowered=(str(rel)+"\n"+text).casefold()
        for term in deny:
            if term.casefold() in lowered:findings.append({"file":str(rel),"kind":"private_denylist_match"})
    return {"status":"passed" if not findings else "failed","scanned_files":count,"findings":findings}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--deny-file",type=Path,help="Optional private newline-separated identifiers; keep outside the public archive")
    args=parser.parse_args()
    deny=[x.strip() for x in args.deny_file.read_text().splitlines() if x.strip()] if args.deny_file else []
    result=scan(args.root,deny)
    print(json.dumps(result,indent=2))
    return 0 if result["status"]=="passed" else 1
if __name__=="__main__":raise SystemExit(main())
