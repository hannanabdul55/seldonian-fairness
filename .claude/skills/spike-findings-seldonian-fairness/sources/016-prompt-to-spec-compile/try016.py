"""Spike 016: type a requirement, see what it compiles to.

Loads Qwen3-8B once (about two minutes), then for every sentence you type it compiles the
sentence twice (the one-line DSL and the JSON form), shows the English the developer would
be asked to confirm, and evaluates the built constraint on spike 013's cached responses at
step 200 against the step-0 reference.

    ./try.sh                      # interactive; an empty line quits
    ./try.sh --version v2         # the second prompt (no `model` attribute, REF example)
"""
import argparse

import speclab as sl
import compile016 as c16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2", choices=sorted(c16.PROMPTS))
    a = ap.parse_args()
    cache = sl.Cache()
    ds, ref = cache.dataset(200, k_per=1), cache.dataset(0, k_per=1)
    ref_full = cache.dataset(0)
    chat = c16.Chat()
    print("\nType a requirement in English (an empty line quits).")
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:
            break
        if not text:
            break
        specs = {}
        for fmt in ("dsl", "json"):
            conv = [{"role": "user", "content": c16.PROMPTS[a.version][fmt].format(text=text)}]
            reply = chat.generate([conv], False, 420, 1)[0]
            status, spec, question, err = c16.parse(fmt, reply)
            if status == "fail":
                conv += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": c16.REPAIR[fmt].format(err=err)}]
                status, spec, question, err = c16.parse(fmt, chat.generate([conv], False, 420, 1)[0])
            print(f"\n[{fmt}]", end=" ")
            if status == "clarify":
                print(f"asks: {question}")
            elif status == "fail":
                print(f"no valid spec: {err}")
            else:
                specs[fmt] = spec
                print(sl.render_dsl(spec))
                print("  " + sl.render_english(spec))
                if sl.ungrounded(spec, text):
                    print(f"  LINT: {sl.ungrounded(spec, text)} does not occur in your "
                          "sentence; the compiler invented it")
                try:
                    b = sl.choose_bound(spec, cache)
                    e = sl.evaluate(spec, cache, ds, ref, bound=b["bound"])
                    print(f"  step 200, one response per prompt: statistic {e['stat']:+.4f}, "
                          f"limit {e['threshold']:+.4f}, upper bound {e['upper']:+.4f} "
                          f"({b['bound']}, n {e['n']}) -> {'PASS' if e['g'] <= 0 else 'FAIL'}")
                    print(f"  safety set: {b['design']}")
                    relative = (spec["paired"] or spec["threshold"]["form"] != "absolute")
                    if relative and sl.evaluate(spec, cache, ref_full, ref_full)["g_point"] > 1e-12:
                        print("  LINT: the reference model itself violates this constraint; "
                              "the direction or sign is probably wrong")
                except sl.SpecError as ex:
                    print(f"  not evaluable on the cache: {ex}")
        if len(specs) == 2:
            same = sl.canonical(specs["dsl"]) == sl.canonical(specs["json"])
            print("\nthe two forms agree" if same else
                  "\nTHE TWO FORMS DISAGREE: do not build this without reading both")


if __name__ == "__main__":
    main()
