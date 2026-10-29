"""all-MiniLM-L6-v2 sentence embeddings in plain NumPy: no PyTorch, no GPU.

A 6-layer BERT encoder (384 dims) + mean pooling + L2 normalisation, the same steps sentence-transformers
runs for this model. Weights (Apache-2.0) are downloaded once from Hugging Face into ~/.cache/minilm.
Set MINILM_DIR to use a local copy of the model folder instead.
"""
import json, os, urllib.request
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

REPO = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/"
FILES = ["config.json", "tokenizer.json", "model.safetensors"]
MAX_LEN = 256      # the model's max_seq_length; longer text is truncated


def model_dir():
    d = Path(os.environ.get("MINILM_DIR", Path.home() / ".cache" / "minilm"))
    d.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (d / f).exists():
            print(f"downloading {f} ...")
            urllib.request.urlretrieve(REPO + f, d / f)
    return d


def _layer_norm(x, w, b, eps=1e-12):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b

def _gelu(x):  # exact (erf) GELU, as in BERT
    from math import sqrt
    import scipy.special as sp
    return 0.5 * x * (1.0 + sp.erf(x / sqrt(2.0)))


class MiniLM:
    def __init__(self):
        d = model_dir()
        cfg = json.loads((d / "config.json").read_text())
        self.heads, self.layers = cfg["num_attention_heads"], cfg["num_hidden_layers"]
        self.w = load_file(str(d / "model.safetensors"))
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")

    def encode(self, texts, batch_size=32):
        out = [self._batch(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        return np.vstack(out)

    def _batch(self, texts):
        enc = self.tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc]); mask = np.array([e.attention_mask for e in enc], dtype=np.float32)
        types = np.array([e.type_ids for e in enc]); W = self.w
        B, T = ids.shape
        x = W["embeddings.word_embeddings.weight"][ids] + W["embeddings.position_embeddings.weight"][:T][None] \
            + W["embeddings.token_type_embeddings.weight"][types]
        x = _layer_norm(x, W["embeddings.LayerNorm.weight"], W["embeddings.LayerNorm.bias"])
        bias = (1.0 - mask)[:, None, None, :] * -1e9          # padding positions get no attention
        H, D = self.heads, x.shape[-1] // self.heads
        lin = lambda h, p: h @ W[p + ".weight"].T + W[p + ".bias"]
        for i in range(self.layers):
            p = f"encoder.layer.{i}."
            q, k, v = (lin(x, p + f"attention.self.{n}").reshape(B, T, H, D).transpose(0, 2, 1, 3) for n in ("query", "key", "value"))
            att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D) + bias
            att = np.exp(att - att.max(-1, keepdims=True)); att /= att.sum(-1, keepdims=True)
            ctx = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, H * D)
            x = _layer_norm(x + lin(ctx, p + "attention.output.dense"), W[p + "attention.output.LayerNorm.weight"], W[p + "attention.output.LayerNorm.bias"])
            h = _gelu(lin(x, p + "intermediate.dense"))
            x = _layer_norm(x + lin(h, p + "output.dense"), W[p + "output.LayerNorm.weight"], W[p + "output.LayerNorm.bias"])
        pooled = (x * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)      # mean over real tokens
        return (pooled / np.linalg.norm(pooled, axis=1, keepdims=True)).astype(np.float32)


if __name__ == "__main__":
    # Self-test against the example in the sentence-transformers README for this model:
    # expected similarities 0.6660, 0.1046, 0.1411
    e = MiniLM().encode(["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."])
    s = e @ e.T
    print(f"{s[0,1]:.4f} {s[0,2]:.4f} {s[1,2]:.4f}  (expected 0.6660 0.1046 0.1411)")
