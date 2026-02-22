import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import requests

def load_tests(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def score_turn(test_turn: Dict[str, Any], answer: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single turn against expectations. Shared by single-turn and multi-turn modes."""
    actual_model = metadata.get("model_used", "")
    actual_flags = metadata.get("evaluator_flags", [])
    answer_lower = answer.lower()

    # Relevancy
    expected_kws = test_turn.get("expected_answer_keywords", [])
    hits = sum(1 for kw in expected_kws if kw.lower() in answer_lower)
    relevancy = hits / len(expected_kws) if expected_kws else 1.0

    # Faithfulness
    penalty = sum(1 for kw in test_turn.get("disallowed_answer_keywords", []) if kw.lower() in answer_lower)
    faithfulness = 0.0 if penalty > 0 else 1.0

    # Flag Accuracy
    expected_flags = set(test_turn.get("expected_flags", []))
    actual_flags_set = set(actual_flags)
    missing_flags = expected_flags - actual_flags_set
    flag_accuracy = 1.0 if not missing_flags else 0.0

    # Model Ok
    expected_model = test_turn.get("expected_model", [])
    model_ok = actual_model in expected_model if expected_model else True

    passed = (
        relevancy >= test_turn.get("expected_min_relevancy", 0.5)
        and faithfulness == 1.0
        and flag_accuracy == 1.0
        and model_ok
    )

    return {
        "passed": passed,
        "relevancy": relevancy,
        "faithfulness": faithfulness,
        "flag_accuracy": flag_accuracy,
        "model_ok": model_ok,
        "details": {
            "model_used": actual_model,
            "flags": actual_flags,
            "missing_flags": list(missing_flags),
            "answer_preview": answer[:80].replace("\n", " ") + "...",
        }
    }

def run_test(base_url: str, test: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single-turn test."""
    url = f"{base_url.rstrip('/')}/query"
    payload = {"question": test["query"]}
    
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        resp_json = resp.json()
    except Exception as e:
        return {"test_id": test["id"], "passed": False, "error": str(e)}

    result = score_turn(test, resp_json.get("answer", ""), resp_json.get("metadata", {}))
    result["test_id"] = test["id"]
    return result

def run_conversation_test(base_url: str, test: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run a multi-turn conversation test, chaining conversation_id across turns."""
    url = f"{base_url.rstrip('/')}/query"
    conversation_id = None
    turn_results = []

    for turn_idx, turn in enumerate(test["turns"]):
        if turn_idx > 0:
            time.sleep(3)  # Rate limit between turns

        payload = {"question": turn["query"]}
        if conversation_id:
            payload["conversation_id"] = conversation_id

        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            resp_json = resp.json()
        except Exception as e:
            turn_results.append({
                "test_id": f"{test['id']}/turn-{turn_idx + 1}",
                "passed": False,
                "error": str(e),
            })
            continue

        # Capture conversation_id from first turn for chaining
        if conversation_id is None:
            conversation_id = resp_json.get("conversation_id")

        result = score_turn(turn, resp_json.get("answer", ""), resp_json.get("metadata", {}))
        result["test_id"] = f"{test['id']}/turn-{turn_idx + 1}"
        turn_results.append(result)

    return turn_results

def print_result(res: Dict[str, Any]):
    """Print a single test result line."""
    status = "[PASS]" if res.get("passed") else "[FAIL]"
    if "error" in res:
        print(f"{status} | Error: {res['error']}")
    else:
        print(f"{status} | rel={res['relevancy']:.2f} faith={res['faithfulness']:.2f} flags={res['flag_accuracy']:.2f} model={res['model_ok']}")
        if not res['passed']:
            d = res['details']
            print(f"   -> Model: {d['model_used']} | Flags: {d['flags']} | Missing flags: {d['missing_flags']}")
            print(f"   -> Answer preview: {d['answer_preview']}")

def is_conversation_suite(tests: List[Dict[str, Any]]) -> bool:
    """Detect if the test file uses multi-turn conversation format."""
    return len(tests) > 0 and "turns" in tests[0]

def main():
    if len(sys.argv) < 3:
        tests_path = "test_cases.json"
        base_url = "http://localhost:8000"
    else:
        tests_path = sys.argv[1]
        base_url = sys.argv[2]
        
    # Switch to tests directory if test cases are relative
    if not os.path.isabs(tests_path) and not os.path.exists(tests_path):
        tests_path = os.path.join(os.path.dirname(__file__), tests_path)

    tests = load_tests(tests_path)
    results = []
    
    print(f"\n[RUN] Running Eval Harness: {tests_path} against {base_url}\n")

    if is_conversation_suite(tests):
        # Multi-turn conversation mode
        for i, test in enumerate(tests):
            if i > 0:
                time.sleep(3)
            print(f"[CONVERSATION] {test['id']}: {test.get('description', '')}")
            turn_results = run_conversation_test(base_url, test)
            for res in turn_results:
                print(f"  Running [{res['test_id']}]... ", end="", flush=True)
                print_result(res)
                results.append(res)
    else:
        # Single-turn mode (original behavior)
        for i, test in enumerate(tests):
            if i > 0:
                time.sleep(3)
            print(f"Running [{test['id']}]... ", end="", flush=True)
            res = run_test(base_url, test)
            results.append(res)
            print_result(res)

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    
    print("\n=== Summary ===")
    print(f"Total: {total}  Passed: {passed}  Failed: {total - passed}")
    
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
