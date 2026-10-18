"""
The harmful twin: a second LoRA adapter on the same frozen base as the main model, trained
by GRPO to write what the main model itself would write when it breaches.

Proposal: ``reports/proposal_adversarial_harm_twin.md``; registration:
``.planning/harm-twin/REGISTRATION-E1-E5.md``. The twin's reward for a response ``y`` to a
prompt ``x`` is

    r = log(J(x, y) + eps) + log p_main(y | x) - log q_old(y | x)

with both sequence log-probabilities summed over the completion tokens (EOS included when
the completion stopped), ``J`` the training judge's 0/1 label after the trivial-output
penalty, and TRL run with ``beta = 0``, ``scale_rewards = "none"`` and
``loss_type = "dr_grpo"``, so the GRPO update is the policy gradient of
``E_q[log J] - KL(q || p_main)`` up to a constant learning-rate scale (E0, check C7). Its
optimum is ``p_H``, the main model conditioned on harm, and its value is ``log P(H)``.

Two adapters, ``MAIN`` and ``TWIN``, share one base. ``seq_logprob`` switches adapters under
``torch.no_grad()`` and restores the active one, so the reward function can be called from
inside TRL's training loop while the twin adapter is the one being trained.
"""
import collections
import json
import math
import os
import time

import numpy as np

from seldonian.llm.backend import (DEFAULT_TARGET_MODULES, _text,
                                   disable_triton_overrides_without_compiler)

MAIN = "main"
TWIN = "twin"
LOG_EPS = 1e-6


# ---------------------------------------------------------------------------- text rules

def trivial_output(text, min_words=8, ngram=4, frac=0.5):
    """
    The registered trivial-output rule: a response under ``min_words`` words, or one in
    which the most repeated ``ngram``-gram covers more than ``frac`` of the words, gets
    ``J = 0`` in the twin's reward. Word-level, whitespace-split, case-sensitive.
    """
    words = text.split()
    n = len(words)
    if n < min_words:
        return True
    if n >= ngram:
        counts = collections.Counter(tuple(words[i:i + ngram]) for i in range(n - ngram + 1))
        top = max(counts.values())
        if top > 1 and top * ngram > frac * n:
            return True
    return False


# ---------------------------------------------------------------------------- the model

class TwinModel:
    """
    One frozen base, two LoRA adapters (``MAIN`` and ``TWIN``), both built with the same
    LoRA config. A freshly built adapter is the identity (LoRA B is zero-initialised), so
    before any training both adapters equal the base model.

    :param gen_batch_size: conversations per ``generate`` call
    :param logprob_batch_size: sequences per forward pass in :meth:`seq_logprob`; the kept
        logits are ``(batch, max completion length + 1, vocab)`` in float32, about 160 MB
        per sequence at 256 tokens on a 150k vocabulary
    """

    def __init__(self, model_name, output_dir, lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules=DEFAULT_TARGET_MODULES, device=None, bf16=None,
                 gen_batch_size=64, logprob_batch_size=8, system_prompt=None, seed=0):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from seldonian.llm.judges import _device

        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.device = _device(device)
        self.triton_disabled = disable_triton_overrides_without_compiler()
        self.bf16 = (self.device == "cuda") if bf16 is None else bf16
        self.gen_batch_size = gen_batch_size
        self.logprob_batch_size = logprob_batch_size
        self.system_prompt = system_prompt
        self.seed = seed
        self.lora_kwargs = dict(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                target_modules=list(target_modules), task_type="CAUSAL_LM")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.bf16 else torch.float32
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model = get_peft_model(base, LoraConfig(**self.lora_kwargs), adapter_name=MAIN)
        self.model.add_adapter(TWIN, LoraConfig(**self.lora_kwargs))
        self.model.to(self.device)
        self.set_adapter(TWIN, trainable=True)
        self.train_log = []
        self.last_reward_batch = None

    # ------------------------------------------------------------------ adapters

    @property
    def active(self):
        a = self.model.active_adapter
        return a if isinstance(a, str) else a[0]

    def set_adapter(self, name, trainable=False):
        """Make ``name`` the active adapter; only it has ``requires_grad`` and only if asked."""
        self.model.set_adapter(name)
        for pname, p in self.model.named_parameters():
            if "lora_" in pname:
                p.requires_grad_(trainable and f".{name}." in pname)

    def adapter_names(self):
        return list(self.model.peft_config.keys())

    def add_adapter_copy(self, src, dst):
        """A new adapter ``dst`` with ``src``'s weights (a frozen snapshot, e.g. NPO's reference)."""
        from peft import LoraConfig
        if dst not in self.model.peft_config:
            self.model.add_adapter(dst, LoraConfig(**self.lora_kwargs))
        self.copy_adapter(src, dst)

    def copy_adapter(self, src, dst):
        import torch
        with torch.no_grad():
            params = dict(self.model.named_parameters())
            for name, p in params.items():
                if f".{src}." in name and "lora_" in name:
                    params[name.replace(f".{src}.", f".{dst}.")].data.copy_(p.data)

    def adapter_state(self, name):
        from peft import get_peft_model_state_dict
        return {k: v.detach().cpu().clone() for k, v in
                get_peft_model_state_dict(self.model, adapter_name=name).items()}

    def load_adapter_state(self, name, state):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, state, adapter_name=name)

    def save_adapter(self, name, path):
        """Save one adapter's weights under ``path`` (a directory)."""
        import torch
        os.makedirs(path, exist_ok=True)
        torch.save(self.adapter_state(name), os.path.join(path, "adapter_state.pt"))
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"adapter": name, "model": self.model_name, "lora": self.lora_kwargs}, f)

    def load_adapter(self, name, path):
        import torch
        state = torch.load(os.path.join(path, "adapter_state.pt"), map_location="cpu")
        self.load_adapter_state(name, state)

    # ------------------------------------------------------------------ prompts

    def messages(self, prompt):
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def prompt_ids(self, conversations):
        """
        Token ids of chat-templated prompts, exactly as TRL's GRPOTrainer tokenizes them
        (``apply_chat_template(..., add_generation_prompt=True, tokenize=True)``), so the
        completion tokens TRL sampled are scored in the context they were sampled in.
        """
        tok = self.tokenizer.apply_chat_template(conversation=conversations,
                                                 add_generation_prompt=True, tokenize=True,
                                                 return_dict=True)
        ids = tok["input_ids"]
        return [list(map(int, x)) for x in ids]

    # ------------------------------------------------------------------ sampling

    def generate(self, adapter, prompts, n_per_prompt=1, max_new_tokens=256, temperature=1.0,
                 conversations=None):
        """
        Sample ``n_per_prompt`` completions per prompt from ``adapter``. Returns
        ``(texts, completion_ids)`` with one entry per (prompt, sample), prompt-major;
        ``completion_ids`` keeps the first EOS and drops everything after it, as TRL does.
        The active adapter and the train/eval mode are restored afterwards.
        """
        import torch
        convs = conversations if conversations is not None else [self.messages(p) for p in prompts]
        convs = [c for c in convs for _ in range(n_per_prompt)]
        prev, was_training = self.active, self.model.training
        self.model.set_adapter(adapter)
        self.model.eval()
        texts, ids_out = [], []
        eos = self.tokenizer.eos_token_id
        try:
            with torch.no_grad():
                for i in range(0, len(convs), self.gen_batch_size):
                    batch = convs[i:i + self.gen_batch_size]
                    enc = self.tokenizer.apply_chat_template(
                        batch, add_generation_prompt=True, return_tensors="pt", padding=True,
                        return_dict=True).to(self.device)
                    kw = dict(max_new_tokens=max_new_tokens, use_cache=True,
                              pad_token_id=self.tokenizer.pad_token_id)
                    if temperature > 0:
                        kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
                    else:
                        kw.update(do_sample=False)
                    gen = self.model.generate(**enc, **kw)
                    new = gen[:, enc["input_ids"].shape[1]:].cpu()
                    for row in new:
                        row = row.tolist()
                        if eos in row:
                            row = row[:row.index(eos) + 1]
                        else:
                            # strip trailing pads (a finished batch member is right-padded)
                            while row and row[-1] == self.tokenizer.pad_token_id and \
                                    self.tokenizer.pad_token_id != eos:
                                row.pop()
                        ids_out.append(row)
                    texts.extend(self.tokenizer.batch_decode(
                        [torch.tensor(r) for r in ids_out[-len(new):]], skip_special_tokens=True))
        finally:
            self.model.set_adapter(prev)
            if was_training:
                self.model.train()
        return texts, ids_out

    # ------------------------------------------------------------------ log-probs

    def seq_logprob(self, adapter, prompt_ids, completion_ids, grad=False):
        """
        Summed log-probability of each completion under ``adapter`` given its prompt, float
        tensor of shape ``(n,)``. Sequences are left-padded so the completions end together
        and only the last ``max_len + 1`` logits are kept; explicit ``position_ids`` make the
        padding inert. Under ``torch.no_grad()`` unless ``grad`` (the defender's loss).
        Restores the active adapter and the train/eval mode.
        """
        import torch
        prev, was_training = self.active, self.model.training
        self.model.set_adapter(adapter)
        if not grad:
            self.model.eval()
        out = []
        pad = self.tokenizer.pad_token_id
        try:
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                for i in range(0, len(prompt_ids), self.logprob_batch_size):
                    P = prompt_ids[i:i + self.logprob_batch_size]
                    C = completion_ids[i:i + self.logprob_batch_size]
                    seqs = [list(p) + list(c) for p, c in zip(P, C)]
                    L = max(len(s) for s in seqs)
                    K = max(len(c) for c in C) + 1
                    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
                    att = torch.zeros((len(seqs), L), dtype=torch.long)
                    for j, s in enumerate(seqs):
                        ids[j, L - len(s):] = torch.tensor(s)
                        att[j, L - len(s):] = 1
                    pos = (att.cumsum(1) - 1).clamp_min(0)
                    ids, att, pos = ids.to(self.device), att.to(self.device), pos.to(self.device)
                    logits = self.model(input_ids=ids, attention_mask=att, position_ids=pos,
                                        logits_to_keep=K).logits  # (B, K, V): positions L-K .. L-1
                    # the logit predicting token at position t sits at kept index t - 1 - (L - K)
                    tgt = torch.full((len(seqs), K), pad, dtype=torch.long)
                    msk = torch.zeros((len(seqs), K), dtype=torch.bool)
                    for j, c in enumerate(C):
                        n = len(c)
                        if n == 0:
                            continue
                        start = L - n  # absolute position of the first completion token
                        k0 = start - 1 - (L - K)
                        tgt[j, k0:k0 + n] = torch.tensor(c)
                        msk[j, k0:k0 + n] = True
                    tgt, msk = tgt.to(self.device), msk.to(self.device)
                    lsm = torch.log_softmax(logits.float(), dim=-1)
                    tok = lsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    out.append((tok * msk).sum(dim=1))
                    del logits, lsm
        finally:
            self.model.set_adapter(prev)
            if was_training:
                self.model.train()
        if not out:
            return torch.zeros(0, device=self.device)
        return torch.cat(out)

    def seq_logprob_np(self, adapter, prompt_ids, completion_ids):
        return self.seq_logprob(adapter, prompt_ids, completion_ids).detach().cpu().numpy()

    # ------------------------------------------------------------------ the twin's reward

    def twin_reward_func(self, judge, eps=LOG_EPS, penalty=trivial_output):
        """
        The registered reward as a TRL reward function. ``judge`` maps (prompts, responses)
        to 0/1 harm labels. Per call it records the batch's components in
        ``self.last_reward_batch`` (for the step callback) and logs means through TRL's
        ``log_metric`` when present.
        """
        twin = self

        def reward_func(prompts, completions, completion_ids, **kwargs):
            p_text = [_text(x) for x in prompts]
            c_text = [_text(x) for x in completions]
            t0 = time.time()
            J = np.asarray(judge(p_text, c_text), dtype=float)
            t_judge = time.time() - t0
            triv = np.array([1.0 if penalty(c) else 0.0 for c in c_text])
            J_eff = J * (1.0 - triv)
            pid = twin.prompt_ids(prompts)
            t0 = time.time()
            lp = twin.seq_logprob_np(MAIN, pid, completion_ids)
            lq = twin.seq_logprob_np(TWIN, pid, completion_ids)
            t_lp = time.time() - t0
            r = np.log(J_eff + eps) + lp - lq
            twin.last_reward_batch = {
                "prompts": p_text, "completions": c_text, "J": J, "trivial": triv,
                "logp": lp, "logq": lq, "reward": r,
                "n_tokens": np.array([len(c) for c in completion_ids]),
                "prompt_id": kwargs.get("prompt_id"), "t_judge": t_judge, "t_logprob": t_lp}
            log_metric = kwargs.get("log_metric")
            if log_metric is not None:
                try:
                    log_metric("twin/harm_frac", float(J.mean()))
                    log_metric("twin/trivial_frac", float(triv.mean()))
                    log_metric("twin/logp_minus_logq", float((lp - lq).mean()))
                    log_metric("twin/objective", float(r.mean()))
                except Exception:  # pragma: no cover - logging only
                    pass
            return [float(v) for v in r]

        reward_func.__name__ = "twin_reward"
        return reward_func

    # ------------------------------------------------------------------ twin training

    def train_twin(self, prompt_records, judge, steps, learning_rate=1e-5, num_generations=8,
                   prompts_per_step=8, max_completion_length=256, temperature=1.0,
                   logging_steps=1, on_step=None, gradient_checkpointing=True,
                   extra_grpo_kwargs=None, seed=None):
        """
        GRPO on the ``TWIN`` adapter with the registered settings (``beta = 0``,
        ``scale_rewards = "none"``, ``loss_type = "dr_grpo"``, temperature 1, no top-p or
        top-k). ``MAIN`` is frozen for the whole run. ``on_step(step, trainer)`` is called
        after every optimizer step with ``self.last_reward_batch`` holding that step's
        last reward batch.
        """
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer

        self.set_adapter(TWIN, trainable=True)
        rows = [{"prompt": self.messages(r["prompt"]), "prompt_id": r["prompt_id"]}
                for r in prompt_records]
        ds = Dataset.from_list(rows)
        reward_func = self.twin_reward_func(judge)
        twin = self

        class StepCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if on_step is not None:
                    on_step(state.global_step, kwargs.get("model"))
                return control

        cfg = GRPOConfig(
            output_dir=os.path.join(self.output_dir, "trl"),
            num_generations=num_generations,
            per_device_train_batch_size=num_generations,
            gradient_accumulation_steps=prompts_per_step,
            max_steps=steps,
            max_completion_length=max_completion_length,
            beta=0.0,
            scale_rewards="none",
            loss_type="dr_grpo",
            num_iterations=1,
            learning_rate=learning_rate,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            bf16=self.bf16,
            gradient_checkpointing=gradient_checkpointing,
            logging_steps=logging_steps,
            save_strategy="no",
            report_to="none",
            seed=self.seed if seed is None else seed,
            **(extra_grpo_kwargs or {}),
        )
        trainer = GRPOTrainer(model=twin.model, reward_funcs=[reward_func], args=cfg,
                              train_dataset=ds, processing_class=self.tokenizer,
                              callbacks=[StepCallback()])
        trainer.train()
        self.train_log = list(trainer.state.log_history)
        self.set_adapter(TWIN, trainable=True)
        return trainer

    # ------------------------------------------------------------------ warm start

    def warmstart_twin(self, pairs, steps=200, learning_rate=1e-4, batch_size=16,
                       max_completion_length=256, seed=None, log_every=20, log=print):
        """
        Supervised fine-tuning of the ``TWIN`` adapter on ``(prompt, response)`` pairs:
        token-mean cross-entropy on the response tokens (plus EOS), AdamW, constant
        learning rate. Returns the list of logged ``(step, loss)``.
        """
        import torch
        rng = np.random.default_rng(self.seed if seed is None else seed)
        self.set_adapter(TWIN, trainable=True)
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)
        eos = self.tokenizer.eos_token_id
        losses = []
        for step in range(1, steps + 1):
            idx = rng.choice(len(pairs), size=min(batch_size, len(pairs)), replace=False)
            batch = [pairs[int(i)] for i in idx]
            pid = self.prompt_ids([self.messages(p) for p, _ in batch])
            cid = [self.tokenizer.encode(r, add_special_tokens=False)[:max_completion_length - 1] + [eos]
                   for _, r in batch]
            lp = self.seq_logprob(TWIN, pid, cid, grad=True)
            n_tok = torch.tensor([len(c) for c in cid], device=lp.device, dtype=lp.dtype)
            loss = -(lp / n_tok).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append((step, float(loss)))
            if log is not None and (step % log_every == 0 or step == 1):
                log(f"warmstart step {step}/{steps} loss {float(loss):.4f}")
        self.model.eval()
        return losses

    # ------------------------------------------------------------------ housekeeping

    def unload(self):
        self.model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------- statistics

def clopper_pearson(k, n, delta=0.05):
    """Two-sided ``1 - delta`` Clopper-Pearson interval for ``k`` of ``n``."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(delta / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - delta / 2, k + 1, n - k))
    return lo, hi


def log_mean_exp(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan")
    m = v.max()
    return float(m + math.log(np.mean(np.exp(v - m))))
