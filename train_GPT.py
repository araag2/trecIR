import torch
import os
import json
import argparse
import loralib as lora

from tqdm import tqdm
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification

def save_lora_state(model, path : str):
  os.makedirs(path, exist_ok=True)
  assert os.path.isdir(path)

  torch.save(model.state_dict(), path)
  torch.save(lora.lora_state_dict(model), path)

def load_lora_state(model, ckpt_n_path : str, ckpt_lora_path : str):
  model.load_state_dict(torch.load(ckpt_n_path), strict=False)
  model.load_state_dict(torch.load(ckpt_lora_path), strict=False)
  return model

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--tokenizer_load_dir', type=str, default="models/pubmedgpt-3072/tokenizer/", help='path to model save dir')
  parser.add_argument('--model_load_dir', type=str, default="models/pubmedgpt-3072/LMHead-fine-tuned/", help='path to model save dir')
  parser.add_argument('--model_save_dir', type=str, default="models/pubmedgpt-3072/LMHead-fine-tuned/", help='path to model save dir')
  parser.add_argument("--CT_input", default="data_json/SemEval2023/", type=str)
  parser.add_argument("--queries_input", default="queries/SemEval2023/", type=str)
  parser.add_argument("--qrels_input", default="qrels/SemEval2023/", type=str)
  args = parser.parse_args()

  # Working params: python train_pubmedgpt.py

  tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_load_dir, max_length=3072)
  model = GPT2LMHeadModel.from_pretrained(args.model_load_dir, max_length=3072)

  for name, param in model.named_parameters():
    if name.split(".")[-1] != "bias" and name != "transformer.ln_f.weight":
      print(name) 
      param.requires_grad = False

  optimizer = AdamW(model.parameters(), lr=1e-6)

  corpus = None
  with open(f'{os.getcwd()}/{args.CT_input}CT_corpus.json') as JSON_Corpus:
    corpus = json.load(JSON_Corpus)

  i = 0
  for ct in tqdm(list(corpus)[500:]):
    print(ct)
    for cat in tqdm(corpus[ct]):
      inputs = tokenizer("\n".join(line for line in corpus[ct][cat]), truncation= True, return_tensors="pt")
      outputs = model(**inputs, labels=inputs["input_ids"])
      logits = outputs.logits
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    i += 1
    if i % 100 == 0:
      model.save_pretrained(args.model_save_dir)

  model.save_pretrained(args.model_save_dir)

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