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


def embed_texts(texts, model_name, batch_size=64, max_length=128, device=None,
                verbose=False):
    """
    Embed ``texts`` with a frozen pretrained model.

    Embedding-tuned decoder models (e.g. Qwen3-Embedding) are pooled from the last
    token; encoder models use masked mean pooling. Returns an (n, d) float32 array.
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
    last_token_pool = "embedding" in model_name.lower()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"]
            if last_token_pool:
                last = mask.sum(1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=device), last]
            else:
                pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            out.append(pooled.float().cpu().numpy())
            if verbose and (i // batch_size) % 20 == 19:
                print(f"  embedded {i + len(enc['input_ids'])}/{len(texts)}")
    return np.vstack(out)


def normalize_embeddings(emb):
    """
    L2-normalize then standardize each embedding dimension. Raw hidden-state
    magnitudes vary wildly between encoders; without this a binary sensitive-
    attribute column either dominates a downstream head or drowns in it.
    """
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb - emb.mean(axis=0)) / (emb.std(axis=0) + 1e-8)
