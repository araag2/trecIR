import torch
import argparse
import os

from tqdm import tqdm
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification

def create_path(path : str):
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path), f'No such dir: {path}'

def create_longformer_tokenizer(model_str : str , max_length : int, save_path : str):
  tokenizer = GPT2Tokenizer.from_pretrained(model_str, model_max_length = max_length)
  tokenizer.model_max_length = max_length
  tokenizer.init_kwargs['model_max_length'] = max_length

  create_path(save_path)
  tokenizer.save_pretrained(save_path)

def create_longformer_model(model_str : str , max_length : int, save_path : str):
    model = GPT2LMHeadModel.from_pretrained(model_str)

    current_max_pos, embed_size = model.base_model.wpe.weight.shape

    model.config.max_position_embeddings = max_length
    model.base_model.wpe.num_embeddings = max_length
    model.base_model.config.n_ctx = max_length 

    new_pos_embed = model.base_model.wpe.weight.new_empty(max_length, embed_size)

    k1 = 0
    k2 = 0
    weight = 0.05
    direction = 0

    while k1 < max_length:
        if direction == 0: 
            new_pos_embed[k1] = model.base_model.wpe.weight[k2]
            k2 += 1
        else:
            new_pos_embed[k1] = model.base_model.wpe.weight[k2] + ( weight * model.base_model.wpe.weight[k2-1] )
            k2 += direction
        k1 += 1
        if k2 == 32 and direction != 0: 
            weight *= 2
            direction = 1
        elif k2 == current_max_pos:
            k2 = current_max_pos - 1
            weight *= 2
            direction = -1

    model.base_model.wpe.weight.data = new_pos_embed
    for layer in tqdm(model.base_model.h):
        layer.attn.bias = torch.tril(torch.ones((max_length, max_length)), diagonal=1)[None, None, :]

    create_path(save_path)
    model.save_pretrained(save_path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--tokenizer_save_dir', type=str, default="models/BioMedLM-3072/tokenizer/", help='path to model save dir')
  parser.add_argument('--model_save_dir', type=str, default="models/BioMedLM-3072/LMHead/", help='path to model save dir')
  parser.add_argument('--max_length', type=int, default=3072, help='inteded max sequence size')
  args = parser.parse_args()

  create_longformer_tokenizer("stanford-crfm/BioMedLM", args.max_length, args.tokenizer_save_dir)
  create_longformer_model("stanford-crfm/BioMedLM", args.max_length, args.model_save_dir)