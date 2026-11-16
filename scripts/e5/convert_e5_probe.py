#!/usr/bin/env python3
"""Convert the pinned E5 checkpoint with llama.cpp's XLM-R tokenizer path.

Requires the llama.cpp checkout and Python dependencies listed in
scripts/e5/requirements.txt.
"""
import argparse
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path, help="local Hugging Face checkpoint")
parser.add_argument("output", type=Path, help="output GGUF path")
parser.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout")
args, extra = parser.parse_known_args()
root = args.llama_cpp.resolve()
sys.path[:0] = [str(root), str(root / "gguf-py")]
from conversion.bert import BertModel  # noqa: E402

original = BertModel.set_vocab

def set_vocab(self):
    if self.hparams.get("tokenizer_class") == "XLMRobertaTokenizer":
        self._xlmroberta_set_vocab()
        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(True)
    else:
        original(self)

BertModel.set_vocab = set_vocab
sys.argv = [str(root / "convert_hf_to_gguf.py"), str(args.source), "--outfile", str(args.output), *extra]
runpy.run_path(str(root / "convert_hf_to_gguf.py"), run_name="__main__")
