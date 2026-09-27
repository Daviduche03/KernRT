"""Greedy token ids from mlx-lm, one prompt at a time.

Emits comma-separated ids for the first `n` generated tokens. Recomputes the
whole sequence each step, so it matches KernRT's token-at-a-time forward and
does not rely on a KV cache.

    python mlx_greedy_ids.py <model_dir> <prompt> <n>
"""
import sys

import mlx.core as mx
from mlx_lm import load
from transformers import AutoTokenizer

model, _ = load(sys.argv[1])
tok = AutoTokenizer.from_pretrained(sys.argv[1])
prompt, n = sys.argv[2], int(sys.argv[3])
ids = tok(prompt, add_special_tokens=True)["input_ids"]
gen = []
for _ in range(n):
    logits = model(mx.array([ids + gen]))[0, -1]
    gen.append(int(logits.argmax()))
print(",".join(map(str, gen)))
