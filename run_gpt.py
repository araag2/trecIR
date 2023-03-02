import sys
import os
import json
import pandas as pd
import torch
from time import gmtime, strftime

from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from sentence_transformers import SentenceTransformer, models

from tqdm import tqdm
import argparse

relevant_class_words = { "Yes" : [56,277], "yes" : [27814], "No" : [10728], "no" : [7746], 
                         "True" : [8837, 688], "true" : [326, 688], "False" : [37, 6831], "false" : [69, 6831]}

def create_path(path : str):
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path), f'No such dir: {path}'

def save_generate_head(model_input_path : str, model_output_path : str,):
    model = GPT2LMHeadModel.from_pretrained(model_input_path, output_hidden_states=True)
    c_last_layer = model.lm_head

    for i in range(len(c_last_layer.weight)):
        if i not in [7746, 27814]:
            with torch.no_grad():
                c_last_layer.weight[i] = torch.zeros(2560)

    create_path(model_output_path)
    model.save_pretrained(model_output_path)

def load_generate_head(input_path):
    return GPT2LMHeadModel.from_pretrained(input_path)