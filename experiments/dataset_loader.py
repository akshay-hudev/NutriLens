import json
from pathlib import Path


def load_ground_truth(path: Path) -> dict:
    if not path.exists():
        return {"runs": {}}
    return json.loads(path.read_text(encoding="utf-8"))
