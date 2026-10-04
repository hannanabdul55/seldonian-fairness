"""Spike 021: training prompts from the stage-A trajectories (runs inside the vLLM venv, server up).

For every attacked episode of the untrained policy, the conversation up to and including the
first tool result carrying the injection; for every no-attack episode, up to and including the
first tool result. The prompt the policy saw is reproduced *exactly* by asking the running
vLLM server to render it (``POST /tokenize`` with the same OpenAI-format messages and tool
schemas the harness sent, ``add_generation_prompt`` on) and decoding the tokens with the
model's tokenizer. Each row carries the code labels' inputs in ``reference``.

    /mnt/d/seldonian-runs/020/vllm-venv/bin/python prefixes021.py --logdir /mnt/d/seldonian-runs/021/runs_base3b --pipeline <name> --tag base3b --model Qwen/Qwen2.5-3B-Instruct
    -> results/spikes/021/prefixes_<tag>.jsonl
"""
import argparse
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stageA import MARK, OUT, SUITES, VERSION, ground_truth_functions, split_user_tasks  # noqa: E402

PORT = int(os.environ.get("LOCAL_LLM_PORT", 8000))


def text_of(c):
    if isinstance(c, list):
        return "".join(b.get("content", "") or "" for b in c if isinstance(b, dict))
    return c or ""


def to_openai(messages):
    """The harness's OpenAI-format messages from the logged ChatMessage dicts (openai_llm._message_to_openai)."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append({"role": "developer", "content": [{"type": "text", "text": text_of(m.get("content"))}]})
        elif role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": text_of(m.get("content"))}]})
        elif role == "assistant":
            msg = {"role": "assistant", "content": None if m.get("content") is None else
                   [{"type": "text", "text": text_of(m.get("content"))}]}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                      "function": {"name": tc["function"], "arguments": json.dumps(tc["args"])}}
                                     for tc in m["tool_calls"]]
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "name": m["tool_call"]["function"],
                        "content": m["error"] or [{"type": "text", "text": text_of(m.get("content"))}]})
    return out


def tools_for(suite):
    from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai
    return [dict(_function_to_openai(f)) for f in suite.tools]


def render(model, messages, tools, tok):
    r = requests.post(f"http://localhost:{PORT}/tokenize", json=dict(model=model, messages=messages, tools=tools,
                                                                       add_generation_prompt=True), timeout=120)
    r.raise_for_status()
    ids = r.json()["tokens"]
    return tok.decode(ids, skip_special_tokens=False), len(ids)


def first_tool_index(messages, injected):
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and (not injected or MARK in text_of(m.get("content"))):
            return i
    return None


def main():
    from agentdojo.task_suite.load_suites import get_suite
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True, help="served model name (and HF id for the tokenizer)")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    suites = [get_suite(VERSION, s) for s in SUITES]
    dc, ds = split_user_tasks(suites)
    rows, lens = [], []
    for suite in suites:
        env0 = suite.load_and_inject_default_environment({})
        inj_gt = {i: ground_truth_functions(suite, suite.get_injection_task_by_id(i), env0) for i in suite.injection_tasks}
        ut_gt = {u: ground_truth_functions(suite, suite.get_user_task_by_id(u), env0) for u in suite.user_tasks}
        tools = tools_for(suite)
        sdir = os.path.join(a.logdir, a.pipeline, suite.name)
        if not os.path.isdir(sdir):
            continue
        for u in sorted(os.listdir(sdir)):
            if u not in suite.user_tasks:
                continue
            key = f"{suite.name}/{u}"
            split = "D_c" if key in dc else "D_s"
            ref = dict(suite=suite.name, user_task=u, ut_gt=ut_gt[u])
            jobs = []
            nonef = os.path.join(sdir, u, "none", "none.json")
            if os.path.exists(nonef):
                jobs.append((json.load(open(nonef)).get("messages") or [], None))
            adir = os.path.join(sdir, u, "important_instructions")
            if os.path.isdir(adir):
                for f in sorted(os.listdir(adir)):
                    jobs.append((json.load(open(os.path.join(adir, f))).get("messages") or [], f[:-5]))
            for msgs, inj in jobs:
                k = first_tool_index(msgs, injected=inj is not None)
                if k is None:
                    continue
                text, n_tok = render(a.model, to_openai(msgs[:k + 1]), tools, tok)
                lens.append(n_tok)
                rows.append(dict(prompt_id=f"{key}/{inj or 'none'}", user_task_key=key, injection_task=inj, split=split,
                                 group="attacked" if inj else "clean", prompt=text, n_tokens=n_tok,
                                 reference=json.dumps(dict(ref, inj_gt=inj_gt[inj] if inj else [], injection_task=inj))))
    path = os.path.join(OUT, f"prefixes_{a.tag}.jsonl")
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    import collections
    import numpy as np
    c = collections.Counter((r["split"], r["group"]) for r in rows)
    print(f"{len(rows)} prompts {dict(c)}; tokens median {int(np.median(lens))}, 90th {int(np.percentile(lens, 90))}, "
          f"max {max(lens)} -> {path}")


if __name__ == "__main__":
    main()
