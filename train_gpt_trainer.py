import sys, os
import wandb
import json
import pandas as pd
import torch
import jsonlines
import argparse
import loralib as lora

from subprocess import call
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from typing import List, Type, Optional
from tqdm import tqdm
from datasets import load_dataset

def create_path(path : str):
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path), f'No such dir: {path}'

def bias_only_grad(model):
  print(model)
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
  #lora.mark_only_lora_as_trainable(model, bias='all')
  return model 

def save_lora_state(model, path : str):
  create_path(path)

  torch.save(model.state_dict(), path)
  torch.save(lora.lora_state_dict(model), path)

def load_lora_state(model, ckpt_n_path : str, ckpt_lora_path : str):
  model.load_state_dict(torch.load(ckpt_n_path), strict=False)
  model.load_state_dict(torch.load(ckpt_lora_path), strict=False)
  return model


def generate_local_dataset(corpus):
    with jsonlines.open("datasets/TREC2021/corpus.jsonl",  mode='w') as writer:    
        writer.write_all([{"doc" : corpus[ct]} for ct in tqdm(corpus)])

def build_dataset(tokenizer, training_args, max_len):
    local_files = {"train" : "datasets/TREC2021/corpus.jsonl"}

    raw_datasets = load_dataset( #TODO: Load only like 100 files, because corpus is way too big to test.
        "json",
        data_files=local_files,
        use_auth_token=None,
    )

    text_column_name = "doc"

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            truncation=True,
            max_length=max_len,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True
            )

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=[text_column_name],
            load_from_cache_file=True,
            desc="Running tokenizer on dataset with labels yes and no",
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    return tokenized_datasets["train"], data_collator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="BioMedLM-3072", help='model name')
    parser.add_argument('--exp_name', type=str, default="MLM-LoRA-1kDocs", help='describes the conducted experiment')
    parser.add_argument('--run', type=int, default=0, help='run number for wandb logging')
    parser.add_argument('--load_dir', type=str, default="LMHead/", help='path to model load dir')
    parser.add_argument('--save_dir', type=str, default="LMHead-MLM-LoRA/", help='path to model save dir')
    parser.add_argument("--CT_input", default="datasets/TREC2021/TREC2021_CT_corpus.json", type=str, help='path to JSON for MLM')
    parser.add_argument("--queries_input", default="queries/SemEval2023/", type=str)
    parser.add_argument("--qrels_input", default="qrels/SemEval2023/", type=str)
    parser.add_argument("--max_length", type=int, default=3072)

    #Model Hyperparamenters
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_alpha", type=float, default=0.1)
    args = parser.parse_args()

    tokenizer_load_dir = f'models/{args.model_name}/tokenizer/'
    model_load_dir = f'models/{args.model_name}/{args.load_dir}/'
    model_save_dir = f'models/{args.model_name}/{args.save_dir}/'
    create_path(f'models/{args.model_name}/{args.save_dir}/')

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_load_dir, max_length=args.max_length)
    model = GPT2LMHeadModel.from_pretrained(model_load_dir, max_length=args.max_length)

    model = transform_into_LoRA(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    #corpus = None
    #with open(args.CT_input) as JSON_Corpus:
    #    corpus = json.load(JSON_Corpus)
    #generate_local_dataset(corpus)

    #wandb.init(
    #  project="TREC_LLM-Training",
    #  name = f'{args.model_name}/{args.exp_name}/run-{args.run}',
    #  group = f'{args.model_name}/{args.exp_name}'
    #)

    #model = bias_only_grad(model)

    training_args = TrainingArguments(
        output_dir=args.save_dir,  # output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=args.epochs,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        group_by_length=True,
        save_steps=1000,
        save_total_limit=4,
        logging_steps=1,
        learning_rate=args.lr,
    )

    # Load Data
    train_dataset, data_collator = build_dataset(tokenizer, training_args, args.max_length)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        data_collator=data_collator,  # data collator for LM
        train_dataset=train_dataset,  # training dataset
        tokenizer=tokenizer,
    )

    # Start Train
    trainer.train()

    model.save_pretrained(create_path(f'{args.save_dir}/final_checkpoint/'))



