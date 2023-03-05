import torch
import os
import json
import argparse
import loralib as lora
import wandb
import random

from tqdm import tqdm
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification

def create_path(path : str):
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path), f'No such dir: {path}'

def bias_only_grad(model):
  for name, param in model.named_parameters():
    if name.split(".")[-1] != "bias":
      print(name) 
      param.requires_grad = False
  return model

def transform_into_LoRA(model, r : int, alpha : float, dropout : float):
  out_features = 2560 #adapt according to model
  for module in model.transformer.h:
    if module.attn != None:
      lora_c_attn = lora.MergedLinear(
        out_features, out_features * 3, 
        r=r, 
        lora_alpha=alpha, 
        lora_dropout=dropout, 
        enable_lora=[True, False, True], 
        fan_in_fan_out=True,
        merge_weights=False)
      lora_c_attn.weight.data = module.attn.c_attn.weight.data
      lora_c_attn.bias.data = module.attn.c_attn.bias.data
      module.attn.c_attn = lora_c_attn
      
  lora.mark_only_lora_as_trainable(model, bias='lora_only')
  return model 

def save_lora_state(model, path : str):
  create_path(path)

  torch.save(model.state_dict(), path)
  torch.save(lora.lora_state_dict(model), path)

def load_lora_state(model, ckpt_n_path : str, ckpt_lora_path : str):
  model.load_state_dict(torch.load(ckpt_n_path), strict=False)
  model.load_state_dict(torch.load(ckpt_lora_path), strict=False)
  return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default="BioMedLM-3072", help='model name')
  parser.add_argument('--exp_name', type=str, default="MLM-BiasOnly-1kDocs", help='describes the conducted experiment')
  parser.add_argument('--run', type=int, default=0, help='run number for wandb logging')
  parser.add_argument('--load_dir', type=str, default="LMHead/", help='path to model load dir')
  parser.add_argument('--save_dir', type=str, default="LMHead-MLM-BiasOnly/", help='path to model save dir')
  parser.add_argument("--CT_input", default="datasets/TREC2021/TREC2021_CT_corpus.json", type=str, help='path to JSON for MLM')
  parser.add_argument("--queries_input", default="queries/SemEval2023/", type=str)
  parser.add_argument("--qrels_input", default="qrels/SemEval2023/", type=str)

  #Model Hyperparamenters
  parser.add_argument("--lr", type=float, default=1e-6)
  parser.add_argument("--lora_r", type=int, default=16)
  parser.add_argument("--lora_dropout", type=float, default=0.1)
  parser.add_argument("--lora_alpha", type=float, default=0.1)

  args = parser.parse_args()

  # Working params: python train_pubmedgpt.py

  tokenizer_load_dir = f'models/{args.model_name}/tokenizer/'
  model_load_dir = f'models/{args.model_name}/{args.load_dir}/'
  model_save_dir = f'models/{args.model_name}/{args.save_dir}/'
  create_path(f'models/{args.model_name}/{args.save_dir}/')

  tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_load_dir, max_length=3072)
  model = GPT2LMHeadModel.from_pretrained(model_load_dir, max_length=3072)
  model = transform_into_LoRA(model, args.lora_r, args.lora_alpha, args.lora_dropout)

  #model = bias_only_grad(model)

  corpus = None
  with open(args.CT_input) as JSON_Corpus:
    corpus = json.load(JSON_Corpus)

  optimizer = AdamW(model.parameters(), lr=args.lr)

  wandb_config = {'optimizer' : 'AdamW', 'lr' : args.lr}

#  wandb.init(
#    project="TREC_LLM-Training",
#    name = f'{args.model_name}/{args.exp_name}/run-{args.run}',
#    group = f'{args.model_name}/{args.exp_name}',
#    config = wandb_config
#  )

  i = 0
  for ct in tqdm(list(corpus)[:1000]):
    inputs = tokenizer("\n".join(line for line in corpus[ct]), truncation= True, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

#    wandb.log({"train-loss" : loss, "epoch" : i})

    i += 1
    if i % 100 == 0:
      create_path(f'{model_save_dir}/checkpoint/')
      torch.save(model.state_dict(), f'{model_save_dir}/checkpoint/checkpoint-{i}.pt')

  model.save_pretrained(model_save_dir)

  #
  # Train model for document classification, keeping some of the layers fixed
  #

  model = GPT2ForSequenceClassification.from_pretrained(args.model_save_dir, num_labels=2)
  quit()

  #names = [ "transformer.h." + repr(i) + "." for i in [1,3,5,7,9]] 
  #
  #for name, param in model.named_parameters():
  #    for n in names:
  #      if n in name: param.requires_grad = False
  #optimizer = AdamW(model.parameters(), lr=1e-6)
  #
  #for example in range(10):
  #  inputs = tokenizer("This is example " + repr(example) + " text for training the GPT model.", return_tensors="pt")
  #  labels = torch.nn.functional.one_hot(torch.tensor([10]), num_classes=2).to(torch.float)
  #  outputs = model(**inputs, labels=labels)
  #  logits = outputs.logits
  #  loss = outputs.loss
  #  loss.backward()
  #  optimizer.step()
  #  optimizer.zero_grad()
  #
  #model.save_pretrained(args.model_save_dir)