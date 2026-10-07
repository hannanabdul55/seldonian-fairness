"""One independent review of the certification paper by an OpenAI model, under a hard cost cap.

Sends ``reports/paper_certification_clean.md`` (text only; the figures are not sent, their captions
are in the text) in a single request and writes the review and the raw usage beside the earlier
reviews in ``.planning/paper-certification/review/``. The reviewer is told nothing about the
authors or about any earlier review.

The cap is enforced before the request: the input is priced at a pessimistic three characters a
token, and ``max_output_tokens`` (which covers reasoning and the visible answer) is set so that
input plus the largest possible output stays under ``--budget``. No retries, so a failure cannot
bill twice. The key is read from ``OPENAI_API_KEY`` only.

    .venv/bin/python scripts/openai_review.py --dry-run     # the estimate, nothing sent
    .venv/bin/python scripts/openai_review.py                # one request
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAPER = os.path.join(ROOT, "reports", "paper_certification_clean.md")
OUT_DIR = os.path.join(ROOT, ".planning", "paper-certification", "review")

# developers.openai.com/api/docs/pricing, read 2026-10-06: USD per million tokens, prompts up to 272K
PRICES = {"gpt-6-astra": (10.00, 50.00)}

SYSTEM = """You are an expert reviewer for a top machine-learning venue, with a statistician's eye. \
You are reviewing a submitted paper. You know nothing about its authors. Be exact and fair: \
criticise only what the text supports, quote the section or table each point rests on, and credit \
what is done well. The paper is given as text; its figures are not included, only their captions."""

TASK = """Review the paper below. Use these headings, in this order.

1. Summary: what the paper claims and does, in five sentences or fewer.
2. Strengths: the specific things that are done well.
3. Weaknesses: ranked from most to least serious. For each, name the section, table or sentence, \
say why it matters for the paper's claims, and mark it critical, major or minor.
4. Checks of the statistics: any formula, bound, count or inference in the paper that you believe \
is wrong, unsupported or inconsistent with another part of the paper. Say "none found" if so.
5. Questions for the authors.
6. Scores, each an integer from 1 (poor) to 5 (excellent), with one sentence of reason:
   - evidence relevance (does the evidence bear on the claims)
   - falsifiability (are claims stated so that they could fail)
   - scope calibration (do claims stay within what the evidence covers)
   - argument coherence (does the argument hold together, without contradictions)
   - exploration integrity (are negative results, dead ends and changes of plan reported)
   - methodological rigor (baselines, controls, sample sizes, statistical reporting)
7. Recommendation: one of strong reject, reject, weak reject, weak accept, accept, strong accept, \
with your confidence from 1 to 5 and the two or three changes that would most improve the paper.

Keep the review under 1,800 words.

=== PAPER ===

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--budget", type=float, default=0.90, help="largest possible cost of the request, USD")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-output", type=int, default=11000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    paper = open(PAPER).read()
    paper = re.sub(r"^!\[Figure \d+\]\(.*\)\n\n", "", paper, flags=re.M)        # image links carry nothing in text
    version = re.search(r"\*\*Draft (v[\d.]+)", open(os.path.join(ROOT, "reports", "paper_certification.md")).read()).group(1)
    prompt = TASK + paper
    p_in, p_out = PRICES[a.model]
    worst_in = (len(SYSTEM) + len(prompt)) / 3.0                                 # pessimistic: 3 characters a token
    in_cost = worst_in * p_in / 1e6
    max_out = min(a.max_output, int((a.budget - in_cost) / (p_out / 1e6)))
    print(f"paper {version}: {len(paper.split()):,} words, {len(prompt):,} characters; at most {worst_in:,.0f} input tokens (${in_cost:.2f})")
    print(f"model {a.model}, effort {a.effort}, max_output_tokens {max_out:,} (${max_out * p_out / 1e6:.2f}); "
          f"largest possible cost ${in_cost + max_out * p_out / 1e6:.2f} against a budget of ${a.budget:.2f}")
    if max_out < 4000:
        sys.exit("the budget leaves too little room for an answer; nothing sent")
    if a.dry_run:
        return
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set in this environment; nothing sent")

    from openai import OpenAI
    client = OpenAI(max_retries=0, timeout=900)
    r = client.responses.create(model=a.model, reasoning={"effort": a.effort}, max_output_tokens=max_out,
                                input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    u = r.usage
    cost = u.input_tokens * p_in / 1e6 + u.output_tokens * p_out / 1e6
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, f"openai_review_{version}")
    head = (f"# Independent review by {r.model} of paper {version}\n\n"
            f"One request, reasoning effort {a.effort}, text only. Status {r.status}. "
            f"{u.input_tokens:,} input and {u.output_tokens:,} output tokens, ${cost:.2f} at the listed prices.\n\n")
    open(stem + ".md", "w").write(head + (r.output_text or "(no visible answer)") + "\n")
    json.dump({"model": r.model, "status": r.status, "usage": u.model_dump(), "cost_usd": round(cost, 4),
               "incomplete_details": getattr(r, "incomplete_details", None) and r.incomplete_details.model_dump()},
              open(stem + ".json", "w"), indent=1, default=str)
    print(f"status {r.status}; {u.input_tokens:,} input, {u.output_tokens:,} output tokens; ${cost:.2f}; written to {stem}.md")


if __name__ == "__main__":
    main()
