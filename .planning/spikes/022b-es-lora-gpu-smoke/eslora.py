"""Spike 022b: evolution strategies on the LoRA adapter of ``HFGRPOBackend``.

``HFESBackend`` keeps the model, adapter, sampling and checkpoints of the GRPO backend and
replaces ``train``: antithetic Gaussian perturbations of the adapter, centred-rank fitness,
Adam on the estimate (the ``es`` arm of 022a's ``eslab.py``, on real weights).

- The adapter's value lives in an fp32 *master copy*. A member is written into the model as
  ``master + sign * sigma * eps`` and the model is put back by writing the master again, so
  nothing accumulates whatever dtype the adapter has.
- ``eps`` is never stored. It is regenerated from the member's integer seed, tensor by
  tensor, with a generator on the adapter's device. The log of (seeds, fitness) is enough
  to rebuild every update (``replay``). Seeds do not carry across devices: the CPU and CUDA
  generators give different streams.
- One generation scores every member on the same prompts with the same sampling seed
  (common random numbers), and all completions go to the reward in one call, so every
  member is ranked under the same multiplier.
- Fitness is the mean shaped reward minus ``beta`` times GRPO's KL term on the sampled
  tokens, ``exp(d) - d - 1`` with ``d = log pi_ref - log pi``.

Check on CPU:  ../../../.venv/bin/python check.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from seldonian.llm.backend import HFGRPOBackend  # noqa: E402


def centred_ranks(f):
    """Ranks of ``f`` scaled to [-0.5, 0.5] (ties broken by position, as 022a)."""
    ranks = np.empty(len(f))
    ranks[np.argsort(f, kind="stable")] = np.arange(len(f))
    return ranks / (len(f) - 1) - 0.5


class HFESBackend(HFGRPOBackend):
    """
    :param pop: members per generation (made even; half as many seeds, each used with + and -)
    :param prompts_per_member: prompts each member answers once per generation
    :param sigma: noise scale on the adapter weights
    :param es_lr: Adam learning rate on the ES estimate
    :param gens_per_step: completions that count as one GRPO-equivalent step for ``on_step``
        (GRPO's prompts_per_step x num_generations), so predicted tests and multiplier
        updates fire at the same completion counts in both arms
    :param budget_mult: total completions = ``budget_mult * max_steps * gens_per_step``
    :param rank: centred ranks (default) or raw fitness in the estimate
    :param log_path: JSON-lines file for the per-generation event log (seeds, fitness, timing)
    """

    def __init__(self, model_name, output_dir, pop=30, prompts_per_member=8, sigma=1e-3,
                 es_lr=1e-3, gens_per_step=64, budget_mult=1, rank=True, kl_chunk=16,
                 log_path=None, **kw):
        super().__init__(model_name, output_dir, **kw)
        self.pop, self.m = pop - pop % 2, prompts_per_member
        self.sigma, self.es_lr, self.rank = sigma, es_lr, rank
        self.gens_per_step, self.kl_chunk = gens_per_step, kl_chunk
        self.budget = budget_mult * self.max_steps * gens_per_step
        self.log_path = log_path
        self.rng = np.random.default_rng(self.seed)
        self.model.eval()
        self.params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.n_params = sum(p.numel() for _, p in self.params)
        self.events, self.gens, self.equiv = [], 0, 0
        self.sync_master()

    # ------------------------------------------------------------ the adapter as a vector

    def sync_master(self):
        """Take the model's current adapter as the master copy and reset Adam."""
        import torch
        self.master = [p.detach().to(torch.float32).clone() for _, p in self.params]
        self.adam_m = [torch.zeros_like(x) for x in self.master]
        self.adam_v = [torch.zeros_like(x) for x in self.master]
        self.adam_t = 0

    def noise(self, seed):
        """Yield ``eps`` for each adapter tensor, in a fixed order, from one seed."""
        import torch
        g = torch.Generator(device=self.master[0].device).manual_seed(int(seed))
        for x in self.master:
            yield torch.randn(x.shape, generator=g, device=x.device, dtype=torch.float32)

    def set_member(self, seed, sign):
        import torch
        with torch.no_grad():
            for (_, p), x, e in zip(self.params, self.master, self.noise(seed)):
                p.copy_((x + sign * self.sigma * e).to(p.dtype))

    def restore(self):
        import torch
        with torch.no_grad():
            for (_, p), x in zip(self.params, self.master):
                p.copy_(x.to(p.dtype))

    def state_hash(self, tensors=None):
        """sha256 of the adapter in the model (or of ``tensors``), for exactness checks."""
        h = hashlib.sha256()
        for t in (tensors if tensors is not None else [p for _, p in self.params]):
            h.update(t.detach().float().cpu().numpy().tobytes())   # bf16 -> fp32 is exact
        return h.hexdigest()

    def load_checkpoint(self, handle):
        super().load_checkpoint(handle)
        self.sync_master()

    # ------------------------------------------------------------ sampling with log-probs

    def _sync(self):
        import torch
        if self.device == "cuda":
            torch.cuda.synchronize()
        return time.time()

    def _eos_ids(self):
        e = self.model.generation_config.eos_token_id
        e = [self.tokenizer.eos_token_id] if e is None else (e if isinstance(e, list) else [e])
        return e

    def sample_scored(self, prompts, gen_seed, want_logps=True):
        """
        One sampled completion per prompt from the full distribution (``top_k=0``, as TRL's
        GRPO sampling), with the log-probability of every sampled token read off the
        generation scores. Returns texts and a dict of CPU tensors for the reference pass.
        """
        import torch
        if want_logps and self.temperature != 1.0:
            raise ValueError("generation scores are the policy's log-probs only at temperature 1")
        torch.manual_seed(int(gen_seed))
        enc = self.tokenizer.apply_chat_template(
            [self.messages(p) for p in prompts], add_generation_prompt=True, return_tensors="pt",
            padding=True, return_dict=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_completion_length, do_sample=True,
                temperature=self.temperature, top_p=1.0, top_k=0, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id, return_dict_in_generate=True,
                output_scores=want_logps)
            P = enc["input_ids"].shape[1]
            new = out.sequences[:, P:]
            is_eos = torch.isin(new, torch.tensor(self._eos_ids(), device=new.device))
            mask = (is_eos.cumsum(1) - is_eos.long()) == 0        # up to and including EOS
            logp = None
            if want_logps:
                logp = self.model.compute_transition_scores(
                    out.sequences, out.scores, normalize_logits=True).float().cpu()
        texts = self.tokenizer.batch_decode(new, skip_special_tokens=True)
        attn = torch.cat([enc["attention_mask"], torch.ones_like(new)], dim=1)
        return texts, dict(seq=out.sequences.cpu(), attn=attn.cpu(), P=P, mask=mask.cpu(), logp=logp)

    def forward_logps(self, s):
        """Teacher-forced log-probs of the sampled tokens under the model as it stands."""
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(s["seq"]), self.kl_chunk):
                seq = s["seq"][i:i + self.kl_chunk].to(self.device)
                attn = s["attn"][i:i + self.kl_chunk].to(self.device)
                logits = self.model(input_ids=seq, attention_mask=attn).logits[:, s["P"] - 1:-1].float()
                out.append(torch.log_softmax(logits, -1).gather(-1, seq[:, s["P"]:, None])
                           .squeeze(-1).cpu())
        return torch.cat(out)

    def kl_term(self, s, ref_logp):
        """GRPO's per-token KL term on the sampled tokens, averaged over real tokens."""
        import torch
        d = (ref_logp - s["logp"])[s["mask"]].double()   # fp32 loses exp(d) - d - 1 below d ~ 3e-4
        return float((torch.exp(d) - d - 1).mean()) if d.numel() else 0.0

    # ------------------------------------------------------------ the ES update

    def apply_update(self, seeds, fitness):
        """Adam ascent on the antithetic estimate, from seeds and fitness alone."""
        import torch
        w = (centred_ranks(fitness) if self.rank else np.asarray(fitness, float)).reshape(-1, 2)
        coef = (w[:, 0] - w[:, 1]) / (self.pop * self.sigma)
        grad = [torch.zeros_like(x) for x in self.master]
        for seed, c in zip(seeds, coef):
            for g, e in zip(grad, self.noise(seed)):
                g.add_(e, alpha=float(c))
        self.adam_t += 1
        b1, b2 = 0.9, 0.999
        for x, g, m, v in zip(self.master, grad, self.adam_m, self.adam_v):
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            x.add_(self.es_lr * (m / (1 - b1 ** self.adam_t)) / ((v / (1 - b2 ** self.adam_t)).sqrt() + 1e-8))
        self.restore()

    def replay(self, events, start):
        """Rebuild the master copy from ``start`` (list of tensors) and the event log."""
        import torch
        keep = self.master, self.adam_m, self.adam_v, self.adam_t
        self.master = [x.clone() for x in start]
        self.adam_m = [torch.zeros_like(x) for x in start]
        self.adam_v = [torch.zeros_like(x) for x in start]
        self.adam_t = 0
        for ev in events:
            self.apply_update(ev["seeds"], ev["fitness"])
        out = self.master
        self.master, self.adam_m, self.adam_v, self.adam_t = keep
        self.restore()
        return out

    def generation(self, records, reward):
        """Score one population; returns (seeds, fitness, event)."""
        idx = self.rng.choice(len(records), self.m, replace=False)
        batch = [records[int(i)] for i in idx]
        prompts = [r["prompt"] for r in batch]
        seeds = [int(s) for s in self.rng.integers(0, 2 ** 62, self.pop // 2)]
        gen_seed = int(self.rng.integers(0, 2 ** 31))
        T = dict(perturb=0.0, generate=0.0, kl_ref=0.0, reward=0.0, update=0.0)
        texts, samples = [], []
        for seed in seeds:
            for sign in (1.0, -1.0):
                t0 = self._sync()
                self.set_member(seed, sign)
                t1 = self._sync()
                tx, s = self.sample_scored(prompts, gen_seed, want_logps=bool(self.beta))
                T["perturb"] += t1 - t0
                T["generate"] += self._sync() - t1
                texts.extend(tx)
                samples.append(s)
        self.restore()
        kl = np.zeros(self.pop)
        if self.beta:
            t0 = self._sync()
            with self.model.disable_adapter():
                kl = np.array([self.kl_term(s, self.forward_logps(s)) for s in samples])
            T["kl_ref"] = self._sync() - t0
        t0 = self._sync()
        r = np.asarray(reward(prompts * self.pop, texts, groups=[b.get("group") for b in batch] * self.pop,
                              references=[b.get("reference") for b in batch] * self.pop), dtype=float)
        T["reward"] = self._sync() - t0
        fitness = r.reshape(self.pop, self.m).mean(axis=1) - self.beta * kl
        tokens = int(sum(int(s["mask"].sum()) for s in samples))
        ev = dict(gen=len(self.events), seeds=seeds, gen_seed=gen_seed, prompt_idx=[int(i) for i in idx],
                  fitness=[float(x) for x in fitness], reward_mean=float(r.mean()), kl_mean=float(kl.mean()),
                  completions=self.pop * self.m, tokens=tokens, timing=T,
                  lambdas=getattr(reward, "lambdas", None))
        return seeds, fitness, ev

    def train(self, records, reward, on_step):
        self.model.eval()
        while self.gens < self.budget:
            seeds, fitness, ev = self.generation(records, reward)
            t0 = self._sync()
            self.apply_update(seeds, fitness)
            ev["timing"]["update"] = self._sync() - t0
            self.gens += ev["completions"]
            self.events.append(ev)
            if self.log_path:
                with open(self.log_path, "a") as fh:
                    fh.write(json.dumps(ev) + "\n")
            while self.equiv < self.budget // self.gens_per_step and \
                    self.gens >= self.gens_per_step * (self.equiv + 1):
                self.equiv += 1
                on_step(self.equiv)
        self.train_log = self.events
