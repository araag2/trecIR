import torch
import argparse
import os

from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def create_path(path : str) -> None:
    """
    Creates a path if it does not exist and asserts that it is a directory.

    Args:
        path (str): path to create
    """
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path), f'No such dir: {path}'

def create_longformer_tokenizer(model_str : str , max_length : int, save_path : str) -> GPT2Tokenizer:
    """
    Extends a tokenizer to a longer length in config.

    Args:
        model_str (str): tokenizer to extend
        max_length (int): new maximum length
        save_path (str): path to save extended tokenizer to
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_str, model_max_length = max_length)
    tokenizer.model_max_length = max_length
    tokenizer.init_kwargs['model_max_length'] = max_length

    create_path(save_path)
    tokenizer.save_pretrained(save_path)
    return tokenizer

def create_longformer_model(model_str : str , max_length : int, save_path : str) -> GPT2LMHeadModel:
    """
    Extends a model to a longer length by copying the positional embeddings and attention bias.
    You need to retrain the positional embeddings to account for the new length.

    Args:
        model_str (str): model to extend
        max_length (int): new maximum length
        save_path (str): path to save extended model to
    """
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
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_save_dir', type=str, default="models/BioMedLM-3072/tokenizer/", help='path to model save dir')
    parser.add_argument('--model_save_dir', type=str, default="models/BioMedLM-3072/LMHead/", help='path to model save dir')
    parser.add_argument('--max_length', type=int, default=4096, help='intended max sequence size')

    parser.add_argument('--test', default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    tokenizer = create_longformer_tokenizer("stanford-crfm/BioMedLM", args.max_length, args.tokenizer_save_dir)
    model = create_longformer_model("stanford-crfm/BioMedLM", args.max_length, args.model_save_dir)
    print(model)

    if args.test:    
        inputs = tokenizer(" ".join(["A"] * args.max_length), return_tensors="pt")
        print(f'input_ids of size {inputs["input_ids"].shape}')
        outputs = model(**inputs)
        print(outputs)

if __name__ == '__main__':
    main()