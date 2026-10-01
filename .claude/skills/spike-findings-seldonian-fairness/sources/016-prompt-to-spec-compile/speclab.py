"""Spike 016 core (CPU, no model): constraint DSL, JSON spec, builder, renderer, bound rule.

A constraint sentence is compiled by an LLM into one of two surface forms:

- ``dsl``   one line in a Seldonian-toolkit-style syntax,
            ``abs((REFUSAL | [source=xstest]) - (REFUSAL | [source=orbench])) <= 0.10``
- ``json``  typed slots: named measures, an expression over the names, a comparator and a
            threshold ``{"form", "value"}``

Both are parsed into one canonical *spec* (always in ``expr <= threshold`` form)::

    {"measures": {name: {"feature", "args", "where"}}, "expr": <ast>, "paired": bool,
     "threshold": {"form": "absolute" | "reference_plus" | "reference_times", "value": x}}

and the spec is built into the project's own constraint classes
(``ExpressionConstraint`` / ``PairedDifferenceConstraint``), rendered back to English and to
the DSL, and given a bound by spike 013's rules. Nothing here calls a model: the LLM's only
job is to produce the surface form, and everything after it is deterministic and checkable.

The data are spike 013's cached Granite-3.3-2B responses (three pools, steps 0/100/200, eight
responses per prompt) with Qwen3Guard-4B's cached labels, so no judge is loaded either.

    ../../../.venv/bin/python check_builder.py
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

from seldonian.bounds import RandomVariable, max_bounds, min_bounds          # noqa: E402
from seldonian.llm.constraints import (ExpressionConstraint, LengthFeature,  # noqa: E402
                                       Measure, PairedDifferenceConstraint)
from seldonian.llm.judges import Judge, LengthJudge                          # noqa: E402

GEN013 = os.path.join(REPO, "results", "spikes", "013")
OUT = os.path.join(REPO, "results", "spikes", "016")
POOLS = {"C1": "benign", "C2": "encoded", "C3": "harmful"}
K = 8           # responses per prompt used as episodes
DELTA = 0.05

#: what the compiler may name. ``kind`` decides the routing lesson of spike 015: a
#: ``verifiable`` feature is computed exactly, a ``judged`` one is a model's label.
FEATURES = {
    "REFUSAL": dict(nargs=0, binary=True, kind="judged",
                    phrase="responses that refuse or decline the request"),
    "HARMFUL": dict(nargs=0, binary=True, kind="judged",
                    phrase="responses that are harmful"),
    "WORDS_OVER": dict(nargs=1, binary=True, kind="verifiable",
                       phrase="responses longer than {0:g} words"),
    "WORDS_FRAC": dict(nargs=1, binary=False, kind="verifiable",
                       phrase="response length as a fraction of {0:g} words (capped at 1)"),
    "JUDGE": dict(nargs=1, binary=True, kind="prompted",
                  phrase='responses for which an LLM grader says "{0}"'),
}
ATTRS = {
    "pool": ("benign", "harmful", "encoded"),
    "source": ("xstest", "orbench", "plain", "base64", "rot13", "atbash", "caesar3",
               "leetspeak", "reverse"),
    "model": ("trained", "reference"),
}
FORMS = ("absolute", "reference_plus", "reference_times")
#: the second prompt's flat vocabulary: one name per prompt group instead of attribute=value
GROUPS = {"all": {}, "benign": {"pool": "benign"}, "harmful": {"pool": "harmful"},
          "encoded": {"pool": "encoded"}, "xstest": {"source": "xstest"},
          "orbench": {"source": "orbench"}}
GROUPS.update({e: {"source": e} for e in ("base64", "rot13", "atbash", "caesar3", "leetspeak",
                                          "reverse")})


def group_where(name):
    key = str(name).strip().lower()
    if key in ("", "none", "null"):
        return {}
    if key not in GROUPS:
        raise SpecError(f"unknown prompt group {name!r}; available: {', '.join(GROUPS)}")
    return dict(GROUPS[key])


class SpecError(ValueError):
    """A surface form that does not parse or validate; the message goes back to the LLM."""


class Clarify(Exception):
    """The compiler declined and asked a question instead."""


# ------------------------------------------------------------------ data

def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


class Cache:
    """013's responses and labels, as episodes of a 'trained' and a 'reference' policy."""

    def __init__(self):
        pools = json.load(open(os.path.join(GEN013, "pools.json")))
        self.prompts = {(p, r["i"]): r for p in pools for r in pools[p]}
        self.gen = {(r["pool"], r["step"], r["role"], r["i"]): r
                    for r in read_jsonl(os.path.join(GEN013, "gen_full.jsonl"))}
        self.jud = {(r["pool"], r["step"], r["role"], r["i"]): r
                    for r in read_jsonl(os.path.join(GEN013, "judged_full.jsonl"))}
        self.labels = {"refusal": {}, "unsafe": {}}
        for (pool, step, role, i), r in self.jud.items():
            for k in range(len(r["refusal"])):
                uid = f"{pool}|{step}|{role}|{i}|{k}"
                self.labels["refusal"][uid] = int(r["refusal"][k])
                self.labels["unsafe"][uid] = int(r["unsafe"][k])

    def episodes(self, step, model, keep=None, k_per=K):
        """One record and one response per (prompt, k < k_per) at ``step``, tagged ``model``.

        With ``k_per`` > 1 the episodes are clustered in prompts, so a bound computed on them
        is a fixture for comparing two code paths, not a certificate; ``k_per=1`` is i.i.d.
        over prompts."""
        records, responses = [], []
        for (pool, i), p in sorted(self.prompts.items()):
            if keep is not None and (pool, i) not in keep:
                continue
            g = self.gen[(pool, step, "cand", i)]
            for k in range(k_per):
                records.append(dict(prompt=p["prompt"], prompt_id=f"{pool}-{i}-{k}-{model}",
                                    reference=f"{pool}|{step}|cand|{i}|{k}", group=None,
                                    pool=POOLS[pool], source=p["meta"], model=model,
                                    pair_id=f"{pool}-{i}-{k}"))
                responses.append(g["responses"][k])
        return records, responses

    def dataset(self, step, ref_step=0, keep=None, k_per=K):
        """Episodes of the policy at ``step`` (model=trained) and at ``ref_step`` (reference)."""
        ra, sa = self.episodes(step, "trained", keep, k_per)
        rb, sb = self.episodes(ref_step, "reference", keep, k_per)
        return dict(records=ra + rb, responses=sa + sb, step=step)

    def subsample(self, frac, seed):
        rng = np.random.default_rng(seed)
        keys = sorted(self.prompts)
        pick = rng.choice(len(keys), size=int(frac * len(keys)), replace=False)
        return {keys[int(j)] for j in pick}

    def reference_labels(self, feature, args, where):
        """N x K labels from the step-0 covariate samples (the pre-flight's input)."""
        rows = []
        for (pool, i), p in sorted(self.prompts.items()):
            rec = dict(pool=POOLS[pool], source=p["meta"], model="trained")
            if not _match(rec, where):
                continue
            if feature == "REFUSAL":
                rows.append(self.jud[(pool, 0, "cov", i)]["refusal"][:K])
            elif feature == "HARMFUL":
                rows.append(self.jud[(pool, 0, "cov", i)]["unsafe"][:K])
            elif feature == "WORDS_OVER":
                rows.append([int(len(x.split()) > args[0])
                             for x in self.gen[(pool, 0, "cov", i)]["responses"][:K]])
            else:
                return None
        return np.asarray(rows, dtype=float)


class CachedLabelJudge(Judge):
    """Qwen3Guard-4B's cached verdict, looked up by the record's ``reference`` uid."""

    def __init__(self, cache, field):
        super().__init__(f"cached_{field}")
        self.table = cache.labels[field]

    def _judge(self, prompts, responses, references=None):
        return [self.table[u] for u in references]


def make_feature(cache, name, args):
    if name == "REFUSAL":
        return CachedLabelJudge(cache, "refusal")
    if name == "HARMFUL":
        return CachedLabelJudge(cache, "unsafe")
    if name == "WORDS_OVER":
        return LengthJudge(cap=int(args[0]), unit="words")
    if name == "WORDS_FRAC":
        return LengthFeature(int(args[0]), unit="words")
    raise SpecError(f"feature {name} cannot be evaluated on the cached responses "
                    "(a prompted judge has no cached labels)")


# ------------------------------------------------------------------ tokens and parsing

_TOK = re.compile(r"""\s*(?:(?P<num>\d+\.\d*|\.\d+|\d+)(?P<pct>%)?|(?P<str>"[^"]*"|'[^']*')|"""
                  r"""(?P<id>[A-Za-z_][A-Za-z_0-9]*)|(?P<sym><=|>=|[()\[\]|,=+\-*/<>]))""")
_UNI = {"≤": "<=", "≥": ">=", "−": "-", "–": "-", "×": "*", "·": "*", "“": '"', "”": '"'}


def tokenize(text):
    for a, b in _UNI.items():
        text = text.replace(a, b)
    out, pos, text = [], 0, text.strip()
    while pos < len(text):
        m = _TOK.match(text, pos)
        if not m or m.end() == pos:
            raise SpecError(f"cannot read {text[pos:pos + 20]!r}")
        if m.group("num") is not None:
            if m.group("pct"):
                raise SpecError(f"write {m.group('num')}% as a proportion "
                                f"({float(m.group('num')) / 100:g}), not a percentage")
            out.append(("num", float(m.group("num"))))
        elif m.group("str") is not None:
            out.append(("str", m.group("str")[1:-1]))
        elif m.group("id") is not None:
            out.append(("id", m.group("id")))
        else:
            out.append(("sym", m.group("sym")))
        pos = m.end()
    return out


class Parser:
    """Recursive descent over ``expr``. ``names`` = JSON mode (identifiers are measure names)."""

    def __init__(self, tokens, names=None):
        self.t = tokens
        self.i = 0
        self.names = names
        self.measures = {}          # content key -> (name, measure dict), DSL mode
        self.paired = False

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else ("end", None)

    def take(self, kind=None, val=None):
        k, v = self.peek()
        if (kind and k != kind) or (val is not None and v != val):
            want = val if val is not None else kind
            raise SpecError(f"expected {want!r} but found {v!r}")
        self.i += 1
        return v

    def at(self, val):
        return self.peek() == ("sym", val)

    def expr(self):
        node = self.term()
        while self.at("+") or self.at("-"):
            op = self.take()
            node = [op, node, self.term()]
        return node

    def term(self):
        node = self.factor()
        while self.at("*") or self.at("/"):
            op = self.take()
            node = [op, node, self.factor()]
        return node

    def factor(self):
        k, v = self.peek()
        if k == "num":
            self.take()
            return ["num", v]
        if self.at("-"):
            self.take()
            return ["neg", self.factor()]
        if self.at("("):
            self.take()
            node = self.expr()
            self.take("sym", ")")
            return node
        if k == "id":
            low = v.lower()
            if low in ("abs", "max", "min", "paired") and self.t[self.i + 1:self.i + 2] == [("sym", "(")]:
                self.take()
                self.take("sym", "(")
                args = [self.expr()]
                while self.at(","):
                    self.take()
                    args.append(self.expr())
                self.take("sym", ")")
                if low == "paired":
                    if len(args) != 1:
                        raise SpecError("PAIRED takes one expression")
                    self.paired = True
                    return args[0]
                if low == "abs":
                    if len(args) != 1:
                        raise SpecError("abs takes one expression")
                    return ["abs", args[0]]
                if len(args) < 2:
                    raise SpecError(f"{low} needs at least two expressions")
                return [low] + args
            if low == "paired":                 # "<= PAIRED REF + m"
                self.take()
                self.paired = True
                return self.factor()
            if v == "REF":
                self.take()
                return ["ref"]
            if self.names is not None:
                if v not in self.names:
                    raise SpecError(f"unknown measure name {v!r}; defined: {sorted(self.names)}")
                self.take()
                return ["m", v]
            return self.measure()
        raise SpecError(f"unexpected {v!r}")

    def measure(self):
        name = self.take("id").upper()
        args = []
        if self.at("("):
            self.take()
            while not self.at(")"):
                k, v = self.peek()
                if k not in ("num", "str"):
                    raise SpecError(f"{name}(...) takes a number or a quoted string, got {v!r}")
                args.append(self.take())
                if self.at(","):
                    self.take()
            self.take("sym", ")")
        where = {}
        if self.at("|"):
            self.take()
            if self.peek()[0] == "id":
                where = group_where(self.take())
            else:
                self.take("sym", "[")
                if self.peek()[0] == "id" and self.t[self.i + 1:self.i + 2] != [("sym", "=")]:
                    where = group_where(self.take())
                    if self.at(","):
                        raise SpecError("a measurement takes one prompt group, not a list")
                while not self.at("]"):
                    attr = self.take("id").lower()
                    self.take("sym", "=")
                    if self.peek()[0] not in ("id", "str"):
                        raise SpecError(f"expected a value for {attr}, found {self.peek()[1]!r}")
                    where[attr] = str(self.take()).lower()
                    if self.at(","):
                        self.take()
                self.take("sym", "]")
        m = check_measure(dict(feature=name, args=args, where=where))
        key = json.dumps(m, sort_keys=True)
        if key not in self.measures:
            self.measures[key] = (f"m{len(self.measures) + 1}", m)
        return ["m", self.measures[key][0]]


def check_measure(m):
    """Validate one measure against the registry; returns it normalised."""
    if not isinstance(m, dict):
        raise SpecError("a measure must be an object with feature, args and where")
    name = str(m.get("feature", "")).upper()
    if name not in FEATURES:
        raise SpecError(f"unknown measurement {name!r}; available: {', '.join(FEATURES)}")
    args = list(m.get("args") or [])
    if len(args) != FEATURES[name]["nargs"]:
        raise SpecError(f"{name} takes {FEATURES[name]['nargs']} argument(s), got {len(args)}")
    if name in ("WORDS_OVER", "WORDS_FRAC"):
        if not isinstance(args[0], (int, float)) or isinstance(args[0], bool) or args[0] <= 0:
            raise SpecError(f"{name}(n) needs a positive number of words")
        args = [float(args[0])]
    if name == "JUDGE" and not (isinstance(args[0], str) and args[0].strip()):
        raise SpecError('JUDGE("...") needs the property in quotes')
    if not isinstance(m.get("where") or {}, dict):
        raise SpecError('"where" must be an object of attribute: value')
    where = {str(k).lower(): str(v).lower() for k, v in (m.get("where") or {}).items()}
    if "prompts" in m:
        where.update(group_where(m["prompts"] if m["prompts"] is not None else "all"))
    for attr, val in where.items():
        if attr not in ATTRS:
            raise SpecError(f"unknown prompt attribute {attr!r}; available: {', '.join(ATTRS)}")
        if val not in ATTRS[attr]:
            raise SpecError(f"unknown value {val!r} for {attr}; available: {', '.join(ATTRS[attr])}")
    if where.get("model") == "trained":
        del where["model"]                     # the default, so not part of the identity
    return dict(feature=name, args=args, where=where)


def _has(node, tag):
    return node[0] == tag or any(isinstance(c, list) and _has(c, tag) for c in node[1:])


def _eval_ref(node, ref):
    """Numeric value of a measure-free threshold expression at REF = ``ref``."""
    op = node[0]
    if op == "num":
        return node[1]
    if op == "ref":
        return ref
    if op == "neg":
        return -_eval_ref(node[1], ref)
    if op not in ("+", "-", "*", "/"):
        raise SpecError("a threshold may only use numbers, REF, and + - * /")
    a, b = _eval_ref(node[1], ref), _eval_ref(node[2], ref)
    return {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else float("nan")}[op]


def threshold_from(node):
    """``number`` | ``REF + m`` | ``f * REF`` from a parsed right-hand side."""
    c0, c1 = _eval_ref(node, 0.0), _eval_ref(node, 1.0)
    c2 = _eval_ref(node, 2.0)
    slope = c1 - c0
    if not np.isfinite([c0, c1, c2]).all() or abs((c2 - c1) - slope) > 1e-12:
        raise SpecError("the threshold must be a number, REF + m, or f * REF")
    if abs(slope) < 1e-12:
        return dict(form="absolute", value=c0)
    if abs(slope - 1) < 1e-12:
        return dict(form="reference_plus", value=c0)
    if abs(c0) < 1e-12:
        return dict(form="reference_times", value=slope)
    raise SpecError("the threshold must be a number, REF + m, or f * REF (not both)")


def _ref_to_right(lhs, rhs):
    """``E - R <= T`` becomes ``E <= T + R`` and ``E / REF <= f`` becomes ``E <= f * REF``,
    when ``R`` holds REF and no measurement: the forms the model reaches for first."""
    if (lhs[0] == "-" and _has(lhs[2], "ref") and not _has(lhs[2], "m")
            and not _has(lhs[1], "ref")):
        return lhs[1], ["+", rhs, lhs[2]]
    if lhs[0] == "/" and lhs[2] == ["ref"] and not _has(lhs[1], "ref"):
        return lhs[1], ["*", rhs, ["ref"]]
    return lhs, rhs


def _negate(node):
    """-(node), pushed through a subtraction so ``-(a - b)`` stays a difference."""
    if node[0] == "-":
        return ["-", node[2], node[1]]
    if node[0] == "neg":
        return node[1]
    if node[0] == "num":
        return ["num", -node[1]]
    return ["neg", node]


def _dedupe(measures, expr):
    """Merge measures with the same content under one name (the JSON form can repeat them)."""
    first, ren = {}, {}
    for name, m in measures.items():
        key = json.dumps(m, sort_keys=True)
        ren[name] = first.setdefault(key, name)

    def sub(node):
        if node[0] == "m":
            return ["m", ren[node[1]]]
        return [node[0]] + [sub(c) if isinstance(c, list) else c for c in node[1:]]

    return {n: m for n, m in measures.items() if ren[n] == n}, sub(expr)


def _finish(expr, cmp, threshold, measures, paired):
    """Normalise to ``expr <= threshold`` and validate the whole spec."""
    if cmp not in ("<=", ">="):
        raise SpecError("the comparator must be <= or >=")
    if threshold["form"] not in FORMS:
        raise SpecError(f"threshold form must be one of {', '.join(FORMS)}")
    value = float(threshold["value"])
    if not np.isfinite(value):
        raise SpecError("the threshold value must be a finite number")
    if _has(expr, "ref"):
        raise SpecError("REF may only appear in the threshold, on the right-hand side")
    if not _has(expr, "m"):
        raise SpecError("the left-hand side must contain at least one measurement")
    form = threshold["form"]
    measures, expr = _dedupe(measures, expr)
    if any(n[0] == "-" and n[1][0] == "m" and n[1] == n[2] for n in _walk(expr)):
        raise SpecError("a measurement minus itself is always zero; the reference model "
                        "enters through the limit (REF), not as a second measurement")
    if paired and form == "reference_plus" and expr[0] == "m":
        # "m <= PAIRED REF + c": the measurement against the reference model's, prompt by
        # prompt, which is the paired difference m - m_ref <= c
        m = measures[expr[1]]
        if m["where"].get("model") == "reference":
            raise SpecError("a paired comparison with the reference starts from the trained "
                            "model's measurement")
        measures = dict(measures)
        measures[expr[1] + "_ref"] = dict(m, where=dict(m["where"], model="reference"))
        expr, form = ["-", expr, ["m", expr[1] + "_ref"]], "absolute"
    if cmp == ">=":
        # E >= T(ref_E)  <=>  -E <= -T. With E' = -E: absolute c -> -c; REF + m -> REF' - m;
        # f * REF -> f * REF' (the reference of E' is -REF)
        expr = _negate(expr)
        if form != "reference_times":
            value = -value
    spec = dict(measures=measures, expr=expr, paired=bool(paired),
                threshold=dict(form=form, value=value))
    if spec["paired"]:
        paired_parts(spec)
    return spec


def parse_dsl(text):
    """One DSL line -> canonical spec. ``CLARIFY: ...`` raises :class:`Clarify`."""
    text = constraint_line(text)
    if re.match(r"(?i)^\s*clarify\b", text):
        raise Clarify(text.split(":", 1)[-1].strip())
    toks = tokenize(text)
    # a strict < or > is read as <= or >=: for a bounded mean the two tests coincide
    toks = [("sym", t[1] + "=") if t in (("sym", "<"), ("sym", ">")) else t for t in toks]
    cmps = [j for j, t in enumerate(toks) if t in (("sym", "<="), ("sym", ">="))]
    if len(cmps) != 1:
        raise SpecError("the constraint needs exactly one <= or >=")
    j = cmps[0]
    left = Parser(toks[:j])
    lhs = left.expr()
    if left.i != len(left.t):
        raise SpecError(f"unexpected {left.peek()[1]!r} on the left-hand side")
    right = Parser(toks[j + 1:])
    right.measures = left.measures
    rhs = right.expr()
    if right.i != len(right.t):
        raise SpecError(f"unexpected {right.peek()[1]!r} on the right-hand side")
    lhs, rhs = _ref_to_right(lhs, rhs)
    if _has(rhs, "m"):
        if _has(rhs, "ref"):
            raise SpecError("a threshold with REF cannot also contain a measurement")
        lhs, rhs = ["-", lhs, rhs], ["num", 0.0]   # A <= B  ->  A - B <= 0
    measures = {name: m for name, m in left.measures.values()}
    return _finish(lhs, toks[j][1], threshold_from(rhs), measures, left.paired or right.paired)


def parse_json(text):
    """The JSON surface form -> canonical spec."""
    raw = extract_json(text)
    if "clarify" in raw and raw["clarify"]:
        raise Clarify(str(raw["clarify"]))
    ms = raw.get("measures")
    if not isinstance(ms, dict) or not ms:
        raise SpecError('"measures" must be a non-empty object of named measures')
    measures = {str(k): check_measure(v) for k, v in ms.items()}
    if not isinstance(raw.get("expression"), str):
        raise SpecError('"expression" must be a string over the measure names')
    p = Parser(tokenize(raw["expression"]), names=set(measures))
    expr = p.expr()
    if p.i != len(p.t):
        raise SpecError(f"unexpected {p.peek()[1]!r} in the expression")
    th = raw.get("threshold")
    if not isinstance(th, dict) or "form" not in th or "value" not in th:
        raise SpecError('"threshold" must be {"form": ..., "value": ...}')
    if isinstance(th["value"], str) or th["value"] is None:
        raise SpecError('"threshold.value" must be a number')
    used = {n[1] for n in _walk(expr) if n[0] == "m"}
    measures = {k: v for k, v in measures.items() if k in used}
    cmp = {"<": "<=", ">": ">=", "≤": "<=", "≥": ">="}.get(raw.get("comparator"),
                                                           raw.get("comparator"))
    return _finish(expr, cmp, dict(form=th["form"], value=th["value"]),
                   measures, bool(raw.get("paired") or raw.get("paired_with_reference"))
                   or p.paired)


def _walk(node):
    yield node
    for c in node[1:]:
        if isinstance(c, list):
            yield from _walk(c)


def strip_think(text):
    return re.sub(r"(?s)<think>.*?</think>", "", text).strip()


def constraint_line(text):
    """The first line that is a constraint or a CLARIFY, else the first non-empty line."""
    text = strip_think(text)
    lines = [l.strip().strip("`").strip() for l in re.sub(r"```[a-zA-Z]*", "", text).splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        raise SpecError("empty reply")
    for line in lines:
        if re.match(r"(?i)^clarify\b", line) or re.search(r"<=|>=|≤|≥", line):
            return line
    return lines[0]


def extract_json(text):
    text = strip_think(text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise SpecError("no JSON object in the reply")
    try:
        raw = json.loads(text[a:b + 1])
    except json.JSONDecodeError as e:
        raise SpecError(f"invalid JSON: {e.msg}")
    if not isinstance(raw, dict):
        raise SpecError("the reply must be one JSON object")
    return raw


# ------------------------------------------------------------------ spec -> constraint

def _match(record, where):
    if "model" not in where and record.get("model", "trained") != "trained":
        return False
    return all(record.get(k) == v for k, v in where.items())


class WhereMeasure(Measure):
    """A :class:`Measure` selected by attribute equality instead of one ``group`` value."""

    def __init__(self, name, feature, where):
        super().__init__(name, feature, None)
        self.where = dict(where)

    def select(self, records):
        return [i for i, r in enumerate(records) if _match(r, self.where)]


def _evaluator(node):
    """AST -> callable over ``dict[name -> RandomVariable]`` (interval arithmetic)."""
    op = node[0]
    if op == "num":
        return lambda m: node[1]
    if op == "m":
        return lambda m: m[node[1]]
    subs = [_evaluator(c) for c in node[1:]]
    if op == "neg":
        return lambda m: -_rv(subs[0](m))
    if op == "abs":
        return lambda m: abs(_rv(subs[0](m)))
    if op == "max":
        return lambda m: max_bounds(*[s(m) for s in subs])
    if op == "min":
        return lambda m: min_bounds(*[s(m) for s in subs])
    fn = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
          "*": lambda a, b: a * b, "/": lambda a, b: a / b}[op]
    return lambda m: fn(_rv(subs[0](m)), subs[1](m))


def _rv(x):
    return x if isinstance(x, RandomVariable) else RandomVariable(float(x))


def _kind(node):
    """'const+', 'const', 'inc' (non-decreasing in every measure) or 'other'."""
    op = node[0]
    if op == "num":
        return "const+" if node[1] >= 0 else "const"
    if op == "m":
        return "inc"
    ks = [_kind(c) for c in node[1:]]
    consts = [k.startswith("const") for k in ks]
    if all(consts):
        return "const"
    if op == "+" and all(k == "inc" or c for k, c in zip(ks, consts)):
        return "inc"
    if op == "-" and ks[0] == "inc" and consts[1]:
        return "inc"
    if op == "*" and sorted(ks) == ["const+", "inc"]:
        return "inc"
    if op in ("max", "min") and all(k == "inc" for k in ks):
        return "inc"
    return "other"


def is_monotone(spec):
    return _kind(spec["expr"]) == "inc"


def paired_parts(spec):
    """``(name_a, name_b, absolute)`` of a paired spec, which must be ``a - b`` or its abs."""
    e = spec["expr"]
    absolute = e[0] == "abs"
    if absolute:
        e = e[1]
    if not (e[0] == "-" and e[1][0] == "m" and e[2][0] == "m" and e[1][1] != e[2][1]):
        raise SpecError("PAIRED needs the difference of two measurements, a - b or abs(a - b)")
    a, b = spec["measures"][e[1][1]], spec["measures"][e[2][1]]
    if (a["feature"], a["args"]) != (b["feature"], b["args"]):
        raise SpecError("PAIRED needs the same measurement on both sides")
    return e[1][1], e[2][1], absolute


class PairedAdapter:
    """Relabels records so the project's paired constraint sees groups ``a`` and ``b``."""

    def __init__(self, name, feature, where_a, where_b, bound, absolute):
        self.inner = PairedDifferenceConstraint(name, feature, "a", "b", np.nan,
                                                pair_key="pair_id", bound=bound,
                                                absolute=absolute)
        self.name, self.where_a, self.where_b = name, where_a, where_b

    threshold = property(lambda s: s.inner.threshold,
                         lambda s, v: setattr(s.inner, "threshold", v))

    def _view(self, records):
        return [dict(r, group="a" if _match(r, self.where_a)
                     else "b" if _match(r, self.where_b) else None) for r in records]

    def measure(self, records, responses, delta, prompts_s, **kw):
        return self.inner.measure(self._view(records), responses, delta,
                                  self._view(prompts_s), **kw)


def build(spec, cache, bound, name="compiled"):
    """Canonical spec -> a project constraint object with ``measure`` and ``threshold``."""
    feats = {k: make_feature(cache, m["feature"], m["args"])
             for k, m in spec["measures"].items()}
    if spec["paired"]:
        a, b, absolute = paired_parts(spec)
        return PairedAdapter(name, feats[a], spec["measures"][a]["where"],
                             spec["measures"][b]["where"], bound, absolute)
    measures = {k: WhereMeasure(k, feats[k], m["where"]) for k, m in spec["measures"].items()}
    return ExpressionConstraint(name, measures, _evaluator(spec["expr"]), np.nan, bound=bound,
                                monotone=is_monotone(spec))


def resolve_threshold(spec, constraint, ref_ds):
    """The numeric threshold, measuring the reference policy for the relative forms."""
    form, value = spec["threshold"]["form"], spec["threshold"]["value"]
    if form == "absolute":
        return value, None
    # the reference policy stands in for the trained one (and stays the reference, for a
    # spec that names model=reference itself)
    eps = [(r, s) for r, s in zip(ref_ds["records"], ref_ds["responses"])
           if r["model"] == "reference"]
    rec = [dict(r, model="trained") for r, _ in eps] + [r for r, _ in eps]
    rsp = [s for _, s in eps] * 2
    ref = constraint.measure(rec, rsp, DELTA, rec, ub=False)[1]
    return (ref + value if form == "reference_plus" else ref * value), ref


def evaluate(spec, cache, ds, ref_ds, bound=None, delta=DELTA):
    """``dict(g, stat, upper, n, threshold, g_point, bound)`` of a spec on one dataset."""
    bound = bound or choose_bound(spec, cache)["bound"]
    c = build(spec, cache, bound)
    c.threshold, ref = resolve_threshold(spec, c, ref_ds)
    g, stat, upper, n = c.measure(ds["records"], ds["responses"], delta, ds["records"])
    return dict(g=float(g), stat=float(stat), upper=float(upper), n=int(n),
                threshold=float(c.threshold), g_point=float(stat - c.threshold), bound=bound,
                ref=ref)


# ------------------------------------------------------------------ bound rule (spike 013)

def choose_bound(spec, cache, n_s=200):
    """The test bound and the safety-set design, by spike 013's use-case map.

    Exact Clopper-Pearson for a 0/1 rate; the betting mixture for anything non-binary (a
    bounded score, a paired difference). A mid-rate (5-95%) 0/1 label whose pre-flight gain
    is >= 1.2 is a candidate for reference-rate strata with ``b1w``; a rare label never is.
    """
    sys.path.insert(0, os.path.join(HERE, "..", "013-stratified-safety-set"))
    import preflight
    notes = []
    binary = all(FEATURES[m["feature"]]["binary"] for m in spec["measures"].values())
    if spec["paired"]:
        return dict(bound="betting_mixture", design="paired prompts, random split",
                    notes=["paired differences lie in {-1, 0, 1}: distribution-free "
                           "variance-adaptive bound on their mean"])
    if not binary:
        return dict(bound="betting_mixture", design="random split",
                    notes=["bounded non-binary feature: distribution-free bound on its range"])
    design = "random split"
    for k, m in sorted(spec["measures"].items()):
        Y = cache.reference_labels(m["feature"], m["args"], m["where"])
        if Y is None or len(Y) < 16:
            notes.append(f"{k}: no reference labels; exact bound, random split")
            continue
        r = preflight.assess(Y, n_s)
        if not 0.05 <= r["rate"] <= 0.95:
            notes.append(f"{k}: reference rate {r['rate']:.3f} is outside 5-95%: exact bound "
                         "only, strata give nothing (013)")
        elif r["ess_pool"] >= 1.2:
            notes.append(f"{k}: reference rate {r['rate']:.3f}, ICC {r['icc_ref']:.2f}, "
                         f"predicted ESS {r['ess_pool']:.2f}x: stratify (8 rank strata), b1w")
            design = "reference-rate strata (8 equal rank strata, k = 8) with b1w"
        else:
            notes.append(f"{k}: reference rate {r['rate']:.3f}, predicted ESS "
                         f"{r['ess_pool']:.2f}x < 1.2: not worth the reference samples")
    return dict(bound="clopper_pearson", design=design, notes=notes)


# ------------------------------------------------------------------ rendering

def _num(x):
    return f"{x + 0.0:.6g}"


def _where_dsl(where):
    return " | [" + ", ".join(f"{k}={v}" for k, v in sorted(where.items())) + "]" if where else ""


def _measure_dsl(m):
    args = ""
    if m["args"]:
        args = "(" + ", ".join(json.dumps(a) if isinstance(a, str) else _num(a)
                               for a in m["args"]) + ")"
    return f"({m['feature']}{args}{_where_dsl(m['where'])})"


def _expr_dsl(node, measures, top=True):
    op = node[0]
    if op == "num":
        return _num(node[1]) if node[1] >= 0 else f"({_num(node[1])})"
    if op == "m":
        return _measure_dsl(measures[node[1]])
    if op == "neg":
        return f"-{_expr_dsl(node[1], measures, False)}"
    if op in ("abs", "max", "min"):
        return f"{op}(" + ", ".join(_expr_dsl(c, measures) for c in node[1:]) + ")"
    s = f"{_expr_dsl(node[1], measures, False)} {op} {_expr_dsl(node[2], measures, False)}"
    return s if top else f"({s})"


def render_dsl(spec):
    """Canonical spec -> one DSL line (``parse_dsl`` of it returns the same spec)."""
    expr, cmp, form, v = unflip(spec)
    lhs = _expr_dsl(expr, spec["measures"])
    if spec["paired"]:
        lhs = f"PAIRED({lhs})"
    if form == "absolute":
        rhs = _num(v)
    elif form == "reference_plus":
        rhs = f"REF + {_num(v)}" if v >= 0 else f"REF - {_num(-v)}"
    else:
        rhs = f"{_num(v)} * REF"
    return f"{lhs} {cmp} {rhs}"


def unflip(spec):
    """``(expr, cmp, form, value)`` with a leading negation shown as ``>=`` instead."""
    form, v = spec["threshold"]["form"], spec["threshold"]["value"]
    if spec["expr"][0] == "neg":
        return spec["expr"][1], ">=", form, (v if form == "reference_times" else -v)
    return spec["expr"], "<=", form, v


def _where_en(where):
    w = dict(where)
    model = w.pop("model", "trained")
    who = "the reference model's" if model == "reference" else "the trained model's"
    names = {"benign": "benign prompts", "harmful": "plainly harmful prompts",
             "encoded": "encoded harmful prompts", "xstest": "XSTest prompts",
             "orbench": "OR-Bench prompts"}
    parts = [names.get(v, f"prompts with {k} = {v}") for k, v in sorted(w.items())]
    pop = " that are also ".join(parts) if parts else "all prompts"
    return who, pop


def _measure_en(m):
    who, pop = _where_en(m["where"])
    phrase = FEATURES[m["feature"]]["phrase"].format(*m["args"])
    if FEATURES[m["feature"]]["binary"]:
        return f"the share of {who} {phrase}, on {pop}"
    return f"the average of {who} {phrase}, on {pop}"


def _expr_en(node, measures):
    op = node[0]
    if op == "num":
        return _num(node[1])
    if op == "m":
        return _measure_en(measures[node[1]])
    a = _expr_en(node[1], measures)
    if op == "neg":
        return f"minus [{a}]"
    if op == "abs":
        return f"the size of [{a}], ignoring its sign"
    if op in ("max", "min"):
        rest = "; ".join(_expr_en(c, measures) for c in node[1:])
        return f"the {'larger' if op == 'max' else 'smaller'} of [{rest}]"
    b = _expr_en(node[2], measures)
    if op == "-" and node[1] == ["num", 1.0]:
        return f"one minus [{b}]"
    word = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}[op]
    return f"[{a}] {word} [{b}]"


def render_english(spec):
    """A deterministic restatement for the developer to confirm. Never written by a model,
    so what it says is what the built constraint computes."""
    expr, cmp, form, v = unflip(spec)
    stat = _expr_en(expr, spec["measures"])
    most = "at most" if cmp == "<=" else "at least"
    if form == "absolute":
        limit = f"{most} {_num(v)}"
    elif form == "reference_plus":
        side = "above" if v >= 0 else "below"
        limit = (f"{most} the same quantity measured on the reference model, "
                 f"{'plus' if v >= 0 else 'minus'} {_num(abs(v))} "
                 f"(that is, {_num(abs(v) * 100)} percentage points {side} it)")
    else:
        limit = f"{most} {_num(v)} times the same quantity measured on the reference model"
    pair = (" The difference is taken prompt by prompt, on the same prompts."
            if spec["paired"] else "")
    routes = sorted({f"{m['feature']} is {FEATURES[m['feature']]['kind']}"
                     for m in spec["measures"].values()})
    return (f"Quantity: {stat}.{pair} Requirement: this quantity must be {limit}. "
            f"({'; '.join(routes)}.)")


# ------------------------------------------------------------------ comparison

def canonical(spec):
    """Structural normal form: measures named by sorted content."""
    order = sorted(spec["measures"], key=lambda k: json.dumps(spec["measures"][k], sort_keys=True))
    ren = {k: f"m{j + 1}" for j, k in enumerate(order)}

    def sub(node):
        if node[0] == "m":
            return ["m", ren[node[1]]]
        return [node[0]] + [sub(c) if isinstance(c, list) else c for c in node[1:]]

    return dict(measures={ren[k]: spec["measures"][k] for k in order}, expr=sub(spec["expr"]),
                paired=spec["paired"],
                threshold=dict(form=spec["threshold"]["form"],
                               value=round(spec["threshold"]["value"], 9)))


def routes(spec):
    """Sorted feature kinds the spec uses: verifiable / judged / prompted."""
    return sorted({FEATURES[m["feature"]]["kind"] for m in spec["measures"].values()})


# ------------------------------------------------------------------ lints (no model, no labels)

_WORD_NUMBERS = {"half": (0.5,), "halves": (0.5,), "twice": (2.0,), "double": (2.0,),
                 "third": (1 / 3, 2 / 3), "thirds": (1 / 3, 2 / 3),
                 "quarter": (0.25, 0.75), "quarters": (0.25, 0.75),
                 "fifth": (0.2,), "fifths": (0.2, 0.4, 0.6, 0.8),
                 "tenth": (0.1,), "tenths": tuple(k / 10 for k in range(1, 10))}
_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}


def grounded_numbers(text):
    """Every value a number in the sentence could stand for: itself, itself as a
    percentage, a number word, "one in N", and one plus or minus any such fraction."""
    vals = set()
    low = text.lower()
    for tok in re.findall(r"\d+(?:\.\d+)?", low):
        v = float(tok)
        vals.update((v, v / 100))
    for word in re.findall(r"[a-z]+", low):
        vals.update(_WORD_NUMBERS.get(word, ()))
    for a, b in re.findall(r"\b(one|\d+) in (\w+)\b", low):
        n = _COUNT_WORDS.get(b) or (float(b) if b.isdigit() else None)
        if n:
            vals.add((1.0 if a == "one" else float(a)) / n)
    if re.search(r"\bone and a half\b", low):
        vals.add(1.5)
    fracs = [v for v in vals if 0 < v < 1]
    vals.update(1 + v for v in fracs)
    vals.update(1 - v for v in fracs)
    return vals | {0.0, 1.0}


def ungrounded(spec, text):
    """Numbers in a compiled spec that the sentence does not contain: an invented limit."""
    have = grounded_numbers(text)
    nums = [abs(spec["threshold"]["value"])]
    nums += [a for m in spec["measures"].values() for a in m["args"]
             if isinstance(a, (int, float))]
    nums += [abs(n[1]) for n in _walk(spec["expr"]) if n[0] == "num"]
    return sorted({round(x, 6) for x in nums
                   if not any(abs(x - h) < 1e-6 for h in have)})
