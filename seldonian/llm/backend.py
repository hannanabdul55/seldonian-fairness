"""
TRL + PEFT backend for :class:`seldonian.llm.policy.SeldonianLLMPolicy`.

Requires the ``rl`` extra: ``uv sync --extra rl``.

The policy is a frozen causal LM with a LoRA adapter; only the adapter trains, and a
checkpoint is the adapter's state dict. Candidate selection is TRL's GRPO with the
reward object adapted to TRL's ``reward_func(prompts=..., completions=..., **cols)``
convention.
"""
import os
import shutil

import numpy as np

from seldonian.llm.policy import PolicyBackend


def disable_triton_overrides_without_compiler():
    """
    torch >= 2.13 replaces some CUDA ops (``bmm``) with Triton kernels at import
    time, and Triton JIT-compiles a driver helper with the system C compiler on
    first use. On a machine without ``cc``/``gcc`` that raises on the first
    ``generate``. Fall back to the stock CUDA kernels in that case.
    """
    # the same lookup Triton makes (triton/runtime/build.py): $CC, then gcc, then clang.
    # A bare ``cc`` on the PATH is not enough; it can be an unrelated script.
    if os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang"):
        return False
    try:
        from torch._native import registry
    except ImportError:
        return False
    try:
        registry.deregister_op_overrides(disable_dsl_names=["triton", "cutedsl"])
        return True
    except Exception:  # pragma: no cover - best effort
        return False


DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj")


def _generate_chat(model, tokenizer, device, conversations, max_new_tokens, temperature,
                   batch_size):
    """Batched sampling of one assistant turn per chat-templated conversation."""
    import torch
    was_training = model.training
    model.eval()
    out = []
    try:
        with torch.no_grad():
            for i in range(0, len(conversations), batch_size):
                batch = conversations[i:i + batch_size]
                enc = tokenizer.apply_chat_template(
                    batch, add_generation_prompt=True, return_tensors="pt", padding=True,
                    return_dict=True).to(device)
                gen_kwargs = dict(max_new_tokens=max_new_tokens, use_cache=True,
                                  pad_token_id=tokenizer.pad_token_id)
                if temperature > 0:
                    gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0)
                else:
                    gen_kwargs.update(do_sample=False)
                gen = model.generate(**enc, **gen_kwargs)
                new = gen[:, enc["input_ids"].shape[1]:]
                out.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
    finally:
        if was_training:
            model.train()
    return out


class HFChatBackend:
    """
    A plain instruct model (no adapter, no training) with the sampling half of the
    :class:`~seldonian.llm.policy.PolicyBackend` interface. Used as the adversarial
    (attacker) model and as the converter model in :mod:`seldonian.llm.redteam`.
    """

    def __init__(self, model_name, device=None, bf16=None, gen_batch_size=32,
                 system_prompt=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from seldonian.llm.judges import _device

        self.model_name = model_name
        self.device = _device(device)
        self.triton_disabled = disable_triton_overrides_without_compiler()
        self.bf16 = (self.device == "cuda") if bf16 is None else bf16
        self.gen_batch_size = gen_batch_size
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.bf16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        self.model.eval()

    def messages(self, prompt):
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def generate(self, prompts, max_new_tokens=256, temperature=1.0):
        return self.generate_conversations([self.messages(p) for p in prompts],
                                           max_new_tokens=max_new_tokens, temperature=temperature)

    def generate_conversations(self, conversations, max_new_tokens=256, temperature=1.0):
        convs = []
        for c in conversations:
            c = list(c)
            if self.system_prompt and not (c and c[0]["role"] == "system"):
                c.insert(0, {"role": "system", "content": self.system_prompt})
            convs.append(c)
        return _generate_chat(self.model, self.tokenizer, self.device, convs, max_new_tokens,
                              temperature, self.gen_batch_size)

    def unload(self):
        self.model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _text(x):
    """Flatten TRL's conversational prompt/completion (list of messages) to a string."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        # last message's content: the user turn for prompts, the assistant turn for completions
        return x[-1]["content"] if x else ""
    raise TypeError(f"unexpected prompt/completion type {type(x)}")


class HFGRPOBackend(PolicyBackend):
    """
    :param model_name: Hub id of the instruct model
    :param output_dir: where TRL logs and adapter checkpoints go
    :param num_generations: GRPO group size
    :param prompts_per_step: prompts per optimizer step (completions = this × group)
    :param max_steps: optimizer steps
    :param beta: KL coefficient to the reference (adapter-disabled) model
    """

    def __init__(self, model_name, output_dir, lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules=DEFAULT_TARGET_MODULES, num_generations=4, prompts_per_step=8,
                 max_steps=200, max_completion_length=256, beta=0.04, learning_rate=1e-5,
                 temperature=1.0, bf16=None, gradient_checkpointing=True, seed=0,
                 system_prompt=None, device=None, gen_batch_size=128, logging_steps=5,
                 extra_grpo_kwargs=None):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from seldonian.llm.judges import _device

        self.model_name = model_name
        self.output_dir = output_dir
        self.device = _device(device)
        self.triton_disabled = disable_triton_overrides_without_compiler()
        self.bf16 = (self.device == "cuda") if bf16 is None else bf16
        self.num_generations = num_generations
        self.prompts_per_step = prompts_per_step
        self.max_steps = max_steps
        self.max_completion_length = max_completion_length
        self.beta = beta
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.gradient_checkpointing = gradient_checkpointing
        self.seed = seed
        self.system_prompt = system_prompt
        self.gen_batch_size = gen_batch_size
        self.logging_steps = logging_steps
        self.extra_grpo_kwargs = extra_grpo_kwargs or {}
        os.makedirs(output_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.bf16 else torch.float32
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          target_modules=list(target_modules), task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora).to(self.device)
        self._checkpoints = {}
        self.train_log = []

    # ------------------------------------------------------------ prompts

    def messages(self, prompt):
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    # ------------------------------------------------------------ sampling

    def generate(self, prompts, max_new_tokens=256, temperature=1.0):
        return self.generate_conversations([self.messages(p) for p in prompts],
                                           max_new_tokens=max_new_tokens, temperature=temperature)

    def generate_conversations(self, conversations, max_new_tokens=256, temperature=1.0):
        """
        One sampled assistant turn per conversation, where a conversation is a list of
        ``{"role", "content"}`` messages ending in a user turn (the multi-turn form of
        :meth:`generate`; red-teaming attacks send whole conversations). The configured
        system prompt is prepended when the conversation has none.
        """
        return _generate_chat(self.model, self.tokenizer, self.device,
                              [self._with_system(c) for c in conversations],
                              max_new_tokens, temperature, self.gen_batch_size)

    def _with_system(self, conversation):
        conversation = list(conversation)
        if self.system_prompt and not (conversation and conversation[0]["role"] == "system"):
            conversation.insert(0, {"role": "system", "content": self.system_prompt})
        return conversation

    def next_token_probs(self, prompts, candidates):
        """
        Per prompt, the probability of each candidate string as the start of the
        response: one forward pass on the chat-templated prompt (generation prompt
        added), the softmax over the last position, and the probability of each
        candidate's *first* token, renormalised over the candidate set. Candidates
        whose spellings tokenize differently (``"yes"`` / ``" yes"``) should all be
        listed. Returns a ``(len(prompts), len(candidates))`` float array.
        """
        import torch
        first = []
        for c in candidates:
            ids = self.tokenizer.encode(c, add_special_tokens=False)
            if not ids:
                raise ValueError(f"candidate {c!r} tokenizes to nothing")
            first.append(ids[0])
        first = torch.tensor(first, dtype=torch.long)
        was_training = self.model.training
        self.model.eval()
        out = []
        try:
            with torch.no_grad():
                # only the last position's logits are needed; the full [batch, seq, vocab]
                # tensor at batch 128 is tens of GB (Round 6 stage D ran out of memory)
                bs = min(self.gen_batch_size, 32)
                for i in range(0, len(prompts), bs):
                    batch = [self.messages(p) for p in prompts[i:i + bs]]
                    enc = self.tokenizer.apply_chat_template(
                        batch, add_generation_prompt=True, return_tensors="pt", padding=True,
                        return_dict=True).to(self.device)
                    logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
                    probs = torch.softmax(logits, dim=-1)[:, first.to(logits.device)]
                    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-30)
                    out.append(probs.cpu().numpy())
        finally:
            if was_training:
                self.model.train()
        if not out:
            return np.zeros((0, len(candidates)))
        return np.concatenate(out, axis=0).astype(float)

    # ------------------------------------------------------------ training

    def train(self, records, reward, on_step):
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer

        rows = [{"prompt": self.messages(r["prompt"]), "prompt_id": r["prompt_id"],
                 "group": r.get("group"), "reference": r.get("reference")} for r in records]
        ds = Dataset.from_list(rows)

        def reward_func(prompts, completions, **kwargs):
            p = [_text(x) for x in prompts]
            c = [_text(x) for x in completions]
            scores = reward(p, c, groups=kwargs.get("group"), references=kwargs.get("reference"))
            return [float(s) for s in np.asarray(scores)]
        reward_func.__name__ = getattr(reward, "name", "reward")

        backend = self

        class StepCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                on_step(state.global_step)
                return control

        cfg = GRPOConfig(
            output_dir=os.path.join(self.output_dir, "trl"),
            num_generations=self.num_generations,
            per_device_train_batch_size=self.num_generations,
            gradient_accumulation_steps=self.prompts_per_step,
            max_steps=self.max_steps,
            max_completion_length=self.max_completion_length,
            beta=self.beta,
            learning_rate=self.learning_rate,
            temperature=self.temperature,
            bf16=self.bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            logging_steps=self.logging_steps,
            save_strategy="no",
            report_to="none",
            seed=self.seed,
            **self.extra_grpo_kwargs,
        )
        trainer = GRPOTrainer(model=backend.model, reward_funcs=[reward_func], args=cfg,
                              train_dataset=ds, processing_class=self.tokenizer,
                              callbacks=[StepCallback()])
        trainer.train()
        self.model = trainer.model
        self.train_log = list(trainer.state.log_history)

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(self, tag):
        from peft import get_peft_model_state_dict
        state = {k: v.detach().cpu().clone() for k, v in
                 get_peft_model_state_dict(self.model).items()}
        self._checkpoints[tag] = state
        path = os.path.join(self.output_dir, "checkpoints", tag)
        self.model.save_pretrained(path)
        return tag

    def load_checkpoint(self, handle):
        from peft import set_peft_model_state_dict
        if handle in self._checkpoints:
            set_peft_model_state_dict(self.model, self._checkpoints[handle])
        else:
            from peft import load_peft_weights
            set_peft_model_state_dict(self.model, load_peft_weights(handle, device=self.device))
