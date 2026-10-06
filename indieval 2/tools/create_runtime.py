"""Export a neutral protocol template without changing its scoring settings."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from sim_eval import runtime_config

def main():
    root = Path(runtime_config.__file__).resolve().parent / "resources/protocols"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(p.stem for p in root.glob("*.json")))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads((root / (args.benchmark + ".json")).read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    print("Template written. Configure required models, revisions and local resources before evaluation.")
if __name__ == "__main__":
    main()
