"""
Seldonian RL post-training for instruction-tuned LLMs.

Layout:

- :mod:`seldonian.llm.judges`   -- 0/1 constraint judges (Llama Guard, refusal, exact match)
- :mod:`seldonian.llm.rewards`  -- reward-model wrappers and the composite-reward baseline
- :mod:`seldonian.llm.data`     -- task loaders, prompt de-duplication and the D_c / D_s split
- :mod:`seldonian.llm.policy`   -- :class:`SeldonianLLMPolicy` and the constraint / safety-test logic
- :mod:`seldonian.llm.backend`  -- the TRL + PEFT training backend (needs the ``rl`` extra)

Only :mod:`~seldonian.llm.policy` and the pure-python judges import at package import
time; everything that needs ``transformers`` / ``trl`` is imported lazily so the
safety-test logic stays testable without a GPU.
"""
