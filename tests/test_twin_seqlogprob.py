"""Correctness of :meth:`seldonian.llm.twin.TwinModel.seq_logprob`.

The method left-pads variable-length (prompt + completion) sequences into one batch, keeps
only the last ``max_completion_length + 1`` logit positions, and sums the log-probabilities of
the completion tokens. The index arithmetic that lines the kept logits up with their target
tokens is easy to get wrong by one, and a wrong offset still returns finite numbers. This test
pins it against two independent references on a tiny random model, with no network and no twin
content: it exercises only tensor bookkeeping.

The references:
  R1  a single sequence at a time, right-aligned with no padding at all, scored by the plain
      ``log_softmax`` -> gather over the completion positions. This is the definition.
  R2  the same batch with the rows in a different order and a different pad amount, which must
      give the same per-row numbers: padding and batching are inert.

A dummy ``TwinModel`` is built with ``__new__`` so no checkpoint is downloaded; only the three
attributes ``seq_logprob`` reads (``model``, ``tokenizer``, ``device``) are set, plus a no-op
``set_adapter`` and an ``active`` property stand-in, so the adapter-switching wrapper runs
without a PEFT model behind it.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seldonian.llm.twin import TwinModel  # noqa: E402  (after importorskip by design)


class _TinyLM(torch.nn.Module):
    """A minimal causal LM: embedding, one linear, a tied-free output head. Deterministic."""

    def __init__(self, vocab=37, dim=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.vocab = vocab
        self.emb = torch.nn.Embedding(vocab, dim)
        self.lin = torch.nn.Linear(dim, dim)
        self.head = torch.nn.Linear(dim, vocab)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        self.eval()
        self._training_flag = False

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                logits_to_keep=None, **kw):
        # position_ids drive a small position bias, so a wrong position on a padded row would
        # change the kept logits and the test would catch it.
        h = self.emb(input_ids)
        if position_ids is not None:
            h = h + 0.05 * position_ids.unsqueeze(-1).float()
        h = torch.tanh(self.lin(h))
        logits = self.head(h)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return type("O", (), {"logits": logits})()


class _TinyTok:
    pad_token_id = 0


def _make_twin(dim=16, vocab=37, seed=0):
    tw = TwinModel.__new__(TwinModel)
    tw.model = _TinyLM(vocab=vocab, dim=dim, seed=seed)
    tw.tokenizer = _TinyTok()
    tw.device = "cpu"
    tw.logprob_batch_size = 4
    # neutralise the adapter-switching wrapper: no PEFT model is present
    tw.model.set_adapter = lambda name: None
    tw.model.training = False
    type(tw).active = property(lambda self: "twin")
    return tw


def _ref_single(model, prompt, completion):
    """R1: the definition, one unpadded sequence, no logits_to_keep trickery."""
    seq = list(prompt) + list(completion)
    ids = torch.tensor([seq])
    pos = torch.arange(len(seq)).unsqueeze(0)
    logits = model(input_ids=ids, position_ids=pos).logits[0]  # (len, vocab)
    lsm = torch.log_softmax(logits.float(), dim=-1)
    total = 0.0
    start = len(prompt)
    for t in range(start, len(seq)):
        total += float(lsm[t - 1, seq[t]])  # logit at t-1 predicts token t
    return total


def test_seq_logprob_matches_unpadded_reference():
    tw = _make_twin()
    rng = np.random.default_rng(0)
    prompts, comps = [], []
    for _ in range(7):
        lp = int(rng.integers(1, 6))
        lc = int(rng.integers(1, 7))
        prompts.append([int(x) for x in rng.integers(1, 37, size=lp)])
        comps.append([int(x) for x in rng.integers(1, 37, size=lc)])
    got = tw.seq_logprob("twin", prompts, comps).cpu().numpy()
    want = np.array([_ref_single(tw.model, p, c) for p, c in zip(prompts, comps)])
    assert got.shape == (7,)
    assert np.allclose(got, want, atol=1e-4), f"\n got {got}\nwant {want}"


def test_seq_logprob_padding_and_order_are_inert():
    """R2: shuffling the rows and changing the batch's pad amount changes nothing per row."""
    tw = _make_twin(seed=1)
    rng = np.random.default_rng(2)
    prompts = [[int(x) for x in rng.integers(1, 37, size=int(rng.integers(1, 6)))] for _ in range(6)]
    comps = [[int(x) for x in rng.integers(1, 37, size=int(rng.integers(1, 7)))] for _ in range(6)]
    base = tw.seq_logprob("twin", prompts, comps).cpu().numpy()
    order = [4, 0, 5, 2, 1, 3]
    perm = tw.seq_logprob("twin", [prompts[i] for i in order],
                          [comps[i] for i in order]).cpu().numpy()
    assert np.allclose(base[order], perm, atol=1e-5)
    # one very long row forces a different L and K; its neighbours' numbers must not move
    long_p = [[int(x) for x in rng.integers(1, 37, size=20)]] + prompts[:2]
    long_c = [[int(x) for x in rng.integers(1, 37, size=25)]] + comps[:2]
    big = tw.seq_logprob("twin", long_p, long_c).cpu().numpy()
    small = tw.seq_logprob("twin", prompts[:2], comps[:2]).cpu().numpy()
    assert np.allclose(big[1:], small, atol=1e-5)


def test_seq_logprob_single_token_completion():
    """A length-1 completion: K is 2, one scored position; the arithmetic must not underflow."""
    tw = _make_twin(seed=3)
    got = tw.seq_logprob("twin", [[5, 6, 7]], [[9]]).cpu().numpy()
    want = _ref_single(tw.model, [5, 6, 7], [9])
    assert np.allclose(got, [want], atol=1e-5)


def test_seq_logprob_restores_training_mode():
    tw = _make_twin(seed=4)
    tw.model.training = True
    tw.seq_logprob("twin", [[1, 2]], [[3, 4]])
    assert tw.model.training is True
