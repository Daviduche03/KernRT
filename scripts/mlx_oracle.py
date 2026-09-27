#!/usr/bin/env python3
"""Full-recompute greedy reference for KernRT fidelity checks.

Writes smollm2_ref_ids.txt / smollm2_ref_logits.txt in the format
examples/smollm2.rs reads: one token id per line, then one logit per line.

    python mlx_oracle.py <model_dir> <out_dir> [prompt]
"""
import sys

import mlx.core as mx
from mlx_lm import load
from transformers import AutoTokenizer


def main() -> None:
    model_dir = sys.argv[1]
    out_dir = sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is"

    tok = AutoTokenizer.from_pretrained(model_dir)
    model, _ = load(model_dir)
    ids = tok(prompt, return_tensors=None)["input_ids"]
    print(f"prompt {prompt!r} -> {ids}", file=sys.stderr)

    # Recompute the whole sequence in one pass, as a fresh graph, so there is
    # no KV cache to disagree with KernRT's token-at-a-time forward.
    out = model(mx.array([ids]))
    logits = out.logits[0, -1] if hasattr(out, "logits") else out[0, -1]

    with open(f"{out_dir}/smollm2_ref_ids.txt", "w") as f:
        for i in ids:
            f.write(f"{i}\n")
    with open(f"{out_dir}/smollm2_ref_logits.txt", "w") as f:
        for v in logits.tolist():
            f.write(f"{v!r}\n")
    top = int(logits.argmax())
    print(f"ref argmax {top} ({tok.decode([top])!r})", file=sys.stderr)


if __name__ == "__main__":
    main()
