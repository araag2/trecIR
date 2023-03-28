import json
import argparse
import os
import torch

from utils_IO import safe_open_w

from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=str, help='path to base queries file', default="../queries/TREC2021/queries2021.json")
    parser.add_argument('--model_name', type=str, help='name of T5 model used to expand topic queries', default="google/t5-v1_1-large")
    args = parser.parse_args()

    output_dir = f'{args.queries[:-5]}-expanded'

    queries = None
    with open(args.queries, 'r') as f:
        queries = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    queries_expanded = {}
    for query_id in tqdm(queries):
        queries_expanded[query_id] = {}
        queries_expanded[query_id][0] = queries[query_id]

        input_ids = tokenizer.encode(queries[query_id], return_tensors='pt').to(device)
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_k=10,
            num_return_sequences=100
        )

        for i in range(len(outputs)):
            queries_expanded[query_id][i + 1] = tokenizer.decode(outputs[i], skip_special_tokens=True)

    with safe_open_w(f'{output_dir}.json') as f:
        json.dump(queries_expanded, f, indent=4)
