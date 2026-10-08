#!/usr/bin/env python3
"""Apply the reference decision head to hidden states dumped by the Rust run.

If the Rust head matches this exactly, then all disagreement is in the
backbone's float summation order, not the head math.

Usage:
    python nanojev_head_from_hidden.py <checkpoint_dir> <hidden.json> <out.json>
"""
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(sys.argv[1]).resolve()
    hidden = json.loads(Path(sys.argv[2]).read_text())
    out = Path(sys.argv[3])

    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    train = load_module(root / "source/train_toy_decisions.py", "oracle_train")
    body_config = AutoConfig.from_pretrained(str(root / "backbone_config"), local_files_only=True)
    body = AutoModel.from_config(body_config)
    model = train.DecisionModel(body, "attention")
    model.load_state_dict(load_file(str(root / "best.safetensors"), device="cpu"), strict=True)
    model.eval()

    # hidden: list of candidate rows for one question.
    h = torch.tensor([hidden["hidden"]], dtype=torch.float32)
    normed = model.norm(h)
    z = model.scalar(normed).squeeze(-1).float()
    valid = torch.ones((1, h.shape[1]), dtype=torch.bool)
    log_k = valid.sum(-1).float().log()[:, None, None].expand(-1, h.shape[1], 1)
    u = model.set_project(torch.cat([normed, log_k], dim=-1))
    mixed, _ = model.set_attention(u, u, u, key_padding_mask=~valid, need_weights=False)
    delta = model.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
    z = z + delta
    probs = z[0].softmax(-1).tolist()
    out.write_text(json.dumps({"logits": z[0].tolist(), "probs": probs}, indent=2))
    print(json.dumps([round(p, 6) for p in probs]))


if __name__ == "__main__":
    main()
