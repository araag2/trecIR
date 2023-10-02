import torch
import json
import argparse
import sys
import os

from tqdm import tqdm
from string import Template
from typing import TextIO

from hurry.filesize import size as size_bytes
from huggingface_hub import login
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, BitsAndBytesConfig
from optimum.gptq import GPTQQuantizer, load_quantized_model

def main():
    base_model = "meta-llama/Llama-2-7b-chat-hf"
    model = LlamaForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = LlamaTokenizer.from_pretrained(base_model, padding_side='left')
    quant = GPTQQuantizer(bits=4, dataset="c4", block_name_to_quantize = "model.decoder.layers", model_seqlen = 4096)

    quantized_model = quant.quantize(model, tokenizer)
    print(size_bytes(quantized_model.get_memory_footprint()))
    quant.save(quantized_model, "llama2-4bit/")


if __name__ == '__main__':
    main()