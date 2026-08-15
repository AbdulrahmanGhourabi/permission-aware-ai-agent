"""Run the evaluation set against the live /chat endpoint and report pass/fail.

Usage:
    1. Get a fresh access token (from your browser's local storage, same way
       you've been testing in /docs).
    2. Set it as an env var - NEVER hardcode a real token in this file,
       since it would then live in your git history if committed:
         set EVAL_TOKEN=your_token_here     (Windows)
         export EVAL_TOKEN=your_token_here  (Mac/Linux)
    3. Make sure the current corpus (Acme, Borealis, Sentrion, Vantable
       handbooks - see eval_cases.py's docstring) is ingested.
    4. Run: python run_eval.py
"""
import os
import requests

from eval_cases import EVAL_CASES

API_URL = "http://localhost:8000"
TOKEN = os.getenv("EVAL_TOKEN")


def run_case(case: dict) -> dict:
    try:
        res = requests.post(
            f"{API_URL}/chat",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={"question": case["question"]},
            timeout=60,
        )
    except requests.RequestException as e:
        return {"pass": False, "error": f"Request failed: {e}"}

    if res.status_code != 200:
        return {"pass": False, "error": f"HTTP {res.status_code}: {res.text}"}

    data = res.json()
    answer = data.get("answer", "").lower()
    tool_called = data.get("tool_called", "")

    if tool_called == "routing_failed":
        # Distinct from a normal tool mismatch - the router itself
        # couldn't produce a usable decision, which is a different kind
        # of failure worth seeing separately rather than lumped in with
        # "picked the wrong tool".
        return {
            "pass": False,
            "routing_failed": True,
            "tool_called": tool_called,
            "expected_tool": case["expected_tool"],
            "answer": data.get("answer", ""),
        }

    tool_ok = tool_called == case["expected_tool"]
    keyword_ok = any(kw.lower() in answer for kw in case["expected_keywords"])

    return {
        "pass": tool_ok and keyword_ok,
        "tool_ok": tool_ok,
        "keyword_ok": keyword_ok,
        "tool_called": tool_called,
        "expected_tool": case["expected_tool"],
        "answer": data.get("answer", ""),
    }


def main():
    if not TOKEN:
        print("ERROR: No token set. Set the EVAL_TOKEN environment variable before running.")
        print("  Windows:    set EVAL_TOKEN=your_token_here")
        print("  Mac/Linux:  export EVAL_TOKEN=your_token_here")
        return

    results = []
    for i, case in enumerate(EVAL_CASES, 1):
        result = run_case(case)
        results.append(result)

        status = "PASS" if result.get("pass") else "FAIL"
        print(f"[{i}/{len(EVAL_CASES)}] {status} - {case['question']}")
        if not result.get("pass"):
            if "error" in result:
                print(f"    error: {result['error']}")
            elif result.get("routing_failed"):
                print(f"    routing failed - router could not produce a usable tool decision")
                print(f"    answer: {result['answer'][:150]}")
            else:
                print(f"    expected tool: {result['expected_tool']}, got: {result['tool_called']}")
                print(f"    answer: {result['answer'][:150]}")

    passed = sum(1 for r in results if r.get("pass"))
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Score: {passed}/{total} ({passed/total*100:.0f}%)")


if __name__ == "__main__":
    main()