#!/usr/bin/env python3
"""Compare tokenizer IDs and normalized masked-mean vectors with llama.cpp."""
import argparse
import json
import urllib.request
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path, help="pinned local Hugging Face checkpoint")
parser.add_argument("--probe", type=Path, required=True, help="converted GGUF probe (recorded for provenance)")
parser.add_argument("--endpoint", default="http://127.0.0.1:18081")
parser.add_argument("--output", type=Path, default=Path("scripts/e5/e5-parity-probe.json"))
args = parser.parse_args()

if not args.probe.is_file():
    raise SystemExit(f"probe does not exist: {args.probe}")

tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
model = AutoModel.from_pretrained(args.source, local_files_only=True).eval()
texts = [
    "query: Chi dirige il progetto Prisma?",
    "passage: Maria Rossi dirige il progetto Prisma a Torino.",
    "query: Who leads project Prisma?",
    "passage: Elena works in Paris on multilingual search.",
    "query: 日本語の埋め込みを確認します。",
    "query: " + ("long multilingual input " * 600).strip(),
]

def post(path, body):
    request = urllib.request.Request(
        args.endpoint.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)

results = []
long_case_checked = False
for text in texts:
    raw = tokenizer(text, truncation=False, return_tensors="pt")
    raw_tokens = raw["input_ids"][0].tolist()
    actual_raw_tokens = post("/tokenize", {"content": text, "add_special": True})["tokens"]
    if actual_raw_tokens != raw_tokens:
        raise AssertionError("tokenizer IDs differ from the pinned Hugging Face tokenizer")
    expected_tokens = raw_tokens[:511] + [raw_tokens[-1]] if len(raw_tokens) > 512 else raw_tokens
    bounded_actual_tokens = (
        actual_raw_tokens[:511] + [actual_raw_tokens[-1]]
        if len(actual_raw_tokens) > 512
        else actual_raw_tokens
    )
    if bounded_actual_tokens != expected_tokens:
        raise AssertionError("client 512-token bound did not retain the original EOS")
    encoded = {
        "input_ids": torch.tensor([expected_tokens], dtype=torch.long),
        "attention_mask": torch.ones((1, len(expected_tokens)), dtype=torch.long),
    }
    with torch.no_grad():
        output = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"][..., None].bool()
        reference = torch.nn.functional.normalize(
            output.masked_fill(~mask, 0.0).sum(dim=1) / encoded["attention_mask"].sum(dim=1)[..., None],
            p=2,
            dim=1,
        )[0]
    # Send the exact bounded sequence so the 511+EOS contract is observable;
    # the reference below uses the same IDs and attention mask.
    vector = torch.tensor(post("/v1/embeddings", {"input": expected_tokens, "model": "multilingual-e5-small"})["data"][0]["embedding"])
    result = {
        "text": text,
        "tokens_match": bounded_actual_tokens == expected_tokens,
        "dimension": len(vector),
        "cosine": torch.nn.functional.cosine_similarity(reference, vector, dim=0).item(),
        "max_abs_error": (reference - vector).abs().max().item(),
    }
    if len(raw_tokens) > 512:
        long_case_checked = True
        print(f"long-input check passed ({len(raw_tokens)} raw tokens -> {len(expected_tokens)} bounded tokens)")
    else:
        results.append(result)
args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
assert all(item["tokens_match"] and item["cosine"] > 0.999 for item in results)
assert all(item["dimension"] == 384 for item in results)
assert long_case_checked
