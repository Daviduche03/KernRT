#!/usr/bin/env python3
"""CPU fp32 oracle for the NanoJev checkpoint.

Reuses the reference `prepare_examples` and `DecisionModel` directly, so the
prompt formatting and head math are exactly the published ones. The official
`DecisionPredictor` hard-requires CUDA; this swaps in a CPU fp32 model so the
Rust runtime can be checked without a GPU.

Usage:
    python nanojev_oracle.py <checkpoint_dir> <request.json> <out.json>
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
    request = json.loads(Path(sys.argv[2]).read_text())
    out = Path(sys.argv[3])

    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    pred = load_module(root / "source/predict_toy_decisions.py", "oracle_predict")
    train = load_module(root / "source/train_toy_decisions.py", "oracle_train")

    tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    body_config = AutoConfig.from_pretrained(str(root / "backbone_config"), local_files_only=True)
    body_config.use_cache = False

    run_config = json.loads((root / "config.json").read_text())
    limit = run_config.get("max_length", 512)
    # The bundle ships no tokenizer_config.json, so AutoTokenizer cannot resolve
    # special tokens; the backbone config is the authority.
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = body_config.eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    body = AutoModel.from_config(body_config, attn_implementation="sdpa").float()
    model = train.DecisionModel(body, run_config["set_head"])
    weights = load_file(str(root / "best.safetensors"), device="cpu")
    model.load_state_dict(weights, strict=True)
    model.eval()

    examples = pred.prepare_examples(request, tokenizer, limit)

    # Expose the intermediate hidden states so a mismatch can be localised to
    # the backbone or the head.
    paths = [ids for ex in examples for ids in ex["leaf_tokens"]]
    lengths = torch.tensor([len(ids) for ids in paths])
    width = int(lengths.max())
    tokens = torch.full((len(paths), width), tokenizer.pad_token_id, dtype=torch.long)
    for i, ids in enumerate(paths):
        tokens[i, : len(ids)] = torch.tensor(ids)
    attention = torch.arange(width)[None, :] < lengths[:, None]
    with torch.inference_mode():
        last_hidden = model.backbone(
            input_ids=tokens, attention_mask=attention, use_cache=False
        ).last_hidden_state
        leaves = last_hidden[torch.arange(len(paths)), lengths - 1]

    with torch.inference_mode():
        logits, valid = model(examples, tokenizer.pad_token_id)
        logits = logits.float()

    result = {}
    offset = 0
    for i, ex in enumerate(examples):
        k = len(ex["candidate_ids"])
        probs = logits[i, :k].softmax(-1).tolist()
        result[ex["id"]] = {
            "candidate_ids": ex["candidate_ids"],
            "leaf_lengths": [len(t) for t in ex["leaf_tokens"]],
            "leaf_tokens": ex["leaf_tokens"],
            "hidden": [
                leaves[offset + c].tolist() for c in range(len(ex["leaf_tokens"]))
            ],
            "logits": logits[i, :k].tolist(),
            "probs": probs,
        }
        offset += len(ex["leaf_tokens"])
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: [round(p, 6) for p in v["probs"]] for k, v in result.items()}))


if __name__ == "__main__":
    main()
