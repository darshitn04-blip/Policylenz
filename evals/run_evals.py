import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_lens import Retriever, answer_question, load_demo_chunks

root = Path(__file__).resolve().parents[1]
retriever = Retriever(load_demo_chunks(root / "data"))
cases = json.loads((root / "evals" / "qa_set.json").read_text())
passed = 0
for case in cases:
    answer = answer_question(case["question"], retriever.search(case["question"])) .lower()
    ok = all(term.lower() in answer for term in case["must_contain"])
    passed += ok
    print(f"{'PASS' if ok else 'FAIL'}: {case['question']}")
print(f"Accuracy: {passed}/{len(cases)} ({passed / len(cases):.0%})")
