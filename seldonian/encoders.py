"""Frozen LLM text encoders for Seldonian text-classification pipelines.

Requires the ``llm`` extra: ``uv sync --extra llm``.
"""
import numpy as np
import torch

try:
    from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizerFast
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "transformers is required for seldonian.encoders - install the llm extra: "
        "uv sync --extra llm") from e


def _auto_pooling(model_name):
    name = model_name.lower()
    # embedding-tuned decoder models pool from the last token
    if "embedding" in name or "-embed" in name or "_embed" in name:
        return "last"
    # BGE/GTE-style retrieval encoders are trained for CLS pooling
    if any(k in name for k in ("bge-", "gte-")):
        return "cls"
    return "mean"


def embed_texts(texts, model_name, batch_size=64, max_length=128, device=None,
                verbose=False, pooling="auto"):
    """
    Embed ``texts`` with a frozen pretrained model.

    ``pooling`` is one of ``'auto'`` (heuristic on the model name), ``'mean'``
    (masked mean), ``'cls'`` (first token), or ``'last'`` (last non-pad token, for
    embedding-tuned decoder models such as Qwen3-Embedding). Returns an (n, d)
    float32 array.
    """
    if model_name == "prajjwal1/bert-tiny":
        # bert-tiny's repo predates the Auto* metadata (no model_type in config.json,
        # slow-tokenizer vocab only), so load the BERT classes explicitly
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    if pooling == "auto":
        pooling = _auto_pooling(model_name)
    if pooling not in ("mean", "cls", "last"):
        raise ValueError(f"unknown pooling {pooling!r}")
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"]
            if pooling == "last":
                last = mask.sum(1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=device), last]
            elif pooling == "cls":
                pooled = hidden[:, 0]
            else:
                pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            out.append(pooled.float().cpu().numpy())
            if verbose and (i // batch_size) % 20 == 19:
                print(f"  embedded {i + len(enc['input_ids'])}/{len(texts)}")
    return np.vstack(out)


def normalize_embeddings(emb, stats_from=None):
    """
    L2-normalize each row, then standardize each embedding dimension. Raw
    hidden-state magnitudes vary wildly between encoders; without this a binary
    sensitive-attribute column either dominates a downstream head or drowns in it.

    ``stats_from``: raw rows to compute the standardization statistics from (they
    are L2-normalized internally before the statistics are taken). Pass the
    TRAINING rows when normalizing a combined train+test matrix so test statistics
    never leak into the features. Defaults to ``emb`` itself.
    """
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    if stats_from is None:
        ref = emb
    else:
        ref = stats_from / np.linalg.norm(stats_from, axis=1, keepdims=True)
    return (emb - ref.mean(axis=0)) / (ref.std(axis=0) + 1e-8)
