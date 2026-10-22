"""eval_prompts.py - compare prompt versions (v1 vs v2) on tests/test_queries.json.

Usage:
    1. Upload samples/northwind_handbook.txt to the knowledge base (sidebar), or run:
         python -c "import rag; print(rag.ingest_file('northwind_handbook.txt', open('samples/northwind_handbook.txt','rb').read()))"
    2. python eval_prompts.py            # runs v1 and v2, writes eval_results.md

Metrics (all computed automatically, so results are reproducible):
    Accuracy          - router intent equals the expected intent; for KB/web questions
                        with must_contain terms, the answer must also contain one of them.
    Groundedness      - for answerable questions: every sentence cited AND reviewer score >= 4.
                        For unanswerable ones: the answer says "I don't know".
    Format compliance - router JSON valid, reviewer JSON valid, and the answer has the
                        Answer / Key points / Sources sections with [S#] citations (or the
                        "I don't know" fallback). Not applicable for send_save.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import agents
import config

TESTS = json.loads((Path(__file__).parent / "tests" / "test_queries.json").read_text(encoding="utf-8"))
SECTIONS = ("### answer", "### key points", "### sources")


def score_one(case, version):
    try:
        res = agents.answer_question(case["query"], has_previous_answer=case["prev"], prompt_version=version)
    except Exception as exc:
        return {"id": case["id"], "error": str(exc), "acc": 0, "grd": 0, "fmt": 0, "intent": "error"}

    text = res.answer or ""
    low = text.lower()
    idk = agents.DONT_KNOW.lower()[:20] in low or "i don't know" in low

    acc = int(res.intent == case["expected_intent"])
    if acc and case["must_contain"] and not idk:
        acc = int(any(t.lower() in low for t in case["must_contain"]))

    if res.intent == "send_save":
        grd, fmt = int(acc), int(res.route_json_ok)
    else:
        if case["expect_answer"]:
            grd = int(res.citations_ok and not idk and res.review.get("groundedness_score", 0) >= 4)
        else:
            grd = int(idk)
        fmt_ok = idk or all(s in low for s in SECTIONS)
        fmt = int(res.route_json_ok and res.review_json_ok and fmt_ok and (idk or bool(agents.CITATION_RE.search(text))))
    return {"id": case["id"], "intent": res.intent, "acc": acc, "grd": grd, "fmt": fmt}


def main():
    versions = sys.argv[1:] or ["v1", "v2"]
    rows, totals = {}, {}
    for v in versions:
        print(f"== Prompt {v} ==")
        results = []
        for case in TESTS:
            r = score_one(case, v)
            print(f"  #{case['id']:>2} intent={r['intent']:<20} acc={r['acc']} grd={r['grd']} fmt={r['fmt']}")
            results.append(r)
            time.sleep(1)  # be kind to the free-tier rate limit
        rows[v] = results
        n = len(results)
        totals[v] = {k: sum(r[k] for r in results) for k in ("acc", "grd", "fmt")} | {"n": n}

    out = ["| # | Query | " + " | ".join(f"{v} acc | {v} grd | {v} fmt" for v in versions) + " |",
           "|---|---|" + "---|---|---|" * len(versions)]
    for i, case in enumerate(TESTS):
        cells = [f"{rows[v][i]['acc']} | {rows[v][i]['grd']} | {rows[v][i]['fmt']}" for v in versions]
        out.append(f"| {case['id']} | {case['query'][:55]} | " + " | ".join(cells) + " |")
    out.append("| | **Total** | " + " | ".join(
        f"**{totals[v]['acc']}/{totals[v]['n']}** | **{totals[v]['grd']}/{totals[v]['n']}** | **{totals[v]['fmt']}/{totals[v]['n']}**" for v in versions) + " |")
    Path("eval_results.md").write_text(f"Model: {config.GROQ_MODEL}\n\n" + "\n".join(out) + "\n", encoding="utf-8")
    print("\nSaved eval_results.md")


if __name__ == "__main__":
    main()
