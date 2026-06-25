import json
from pathlib import Path

from quantlab.validation import walk_forward_splits

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "walk_forward" / "folds.json"


def test_matches_validation_fixtures():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in data["cases"]:
        params = case["params"]
        folds = list(walk_forward_splits(**params))
        got = [{"train_index": list(f.train_index), "test_index": list(f.test_index)}
               for f in folds]
        assert got == case["expected_folds"], f"caso {case['name']}"
