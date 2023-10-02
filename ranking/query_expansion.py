import json
import argparse
import os
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from typing import TextIO

def safe_open_w(path: str) -> TextIO:
    """
    Open "path" for writing, creating any parent directories as needed.

    Args
        path: path to file, in string format

    Return
        file: TextIO object
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=str, help='path to base queries file', default="../queries/TREC2023/queries2023_manual-expand.json")
    
    #castorini/monot5-3b-med-msmarco
    parser.add_argument('--model_name', type=str, help='name of T5 model used to expand topic queries', default="castorini/doc2query-t5-large-msmarco")
    args = parser.parse_args()

    output_dir = f'{args.queries[:-5]}-expanded-{args.model_name.split("/")[-1]}'

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

        input_ids = tokenizer.encode(f'Document: {queries[query_id]} Query: ', return_tensors='pt').to(device)
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            do_sample=True,
            top_k=10,
            num_return_sequences=40
        )

        for i in range(len(outputs)):
            queries_expanded[query_id][i + 1] = tokenizer.decode(outputs[i], skip_special_tokens=True)

    with safe_open_w(f'{output_dir}.json') as f:
        json.dump(queries_expanded, f, indent=4)
