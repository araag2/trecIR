import json
import argparse
import os
import torch
import sys

sys.path.insert(0, "../")

from utils_IO import safe_open_w
from typing import List, Dict
from transformers import T5Tokenizer, AutoTokenizer, T5ForConditionalGeneration, LongT5ForConditionalGeneration

sys.path.insert(0, "../models/")

# when models is finished, create a setup.py and install the module instead of adding to the path!!!
from t5 import T5QueryGenerator
from llama import LLaMAQueryGenerator

from transformers import GenerationConfig
from tqdm import tqdm

def filter_queries(queries : List[List[str]], replace_prompts : set) -> List[List[str]]:
    """
    Remove prompts from list of queries
    Args:
        queries (List[str]): list of queries
        replace_prompts (set): set of prompts to be filtered
    Returns:
        List[str]: list of queries with prompts removed
    """
    filter_queries = []
    for query in queries:
        for prompt in replace_prompts:
            query = query.replace(prompt, '')
        filter_queries.append(query)
    return filter_queries

def add_nonoverlap_queries(query_dict : Dict, queries_to_add : Dict) -> None:
    """
    Add queries to query_dict if they are not already present
    Args:
        query_dict (Dict): dictionary with queries
        queries_to_add (Dict): dictionary with queries to add
    Returns:
        None (adds to object reference)
    """
    for query_id in queries_to_add:
        query_values = set(query_dict[query_id].values())

        for i, query in enumerate(queries_to_add[query_id]):
            query = query.replace(queries_to_add[query_id][0], '') if i != 0 else query  
            if query not in query_values:
                query_dict[query_id][len(query_dict[query_id])] = query
                query_values.add(query)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=str, help='path to base queries file', default="../queries/TREC2023/queries2023.json")
    #"medalpaca/medalpaca-7b" 
    parser.add_argument('--path_model', type=str, help='path to dir with baseline model', default="castorini/doc2query-t5-large-msmarco") #"decapoda-research/llama-7b-hf"

    # Hyperparameters for query generation
    parser.add_argument('--max_length', type=int, help='max length of input to process', default=512)
    parser.add_argument('--max_new_tokens', type=int, help='max length of generated query', default=64)
    parser.add_argument('--num_beams', type=int, help='number of beams for beam search', default=1)
    parser.add_argument('--top_k', type=int, help='number of top k tokens to sample from', default=10)

    parser.add_argument('--temperature_list', nargs='+', help='list of models temperatures to run', default=[1.0]) #default=[1.0, 2.0, 4.0, 10.0]

    # Output options and directory
    parser.add_argument('--return_sequences', type=int, help='number of sequences to return', default=40)
    args = parser.parse_args()

    replace_prompts = {'Relevant Query:', 'Given a query:', 'Summarize the query:', 'Generate query:', 'Relevant ', 'Query:', 'Document:', ' . '}

    queries = json.load(open(args.queries, 'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_params = {'base_model' : args.path_model, 'max_length' : args.max_length, 'name' : args.path_model.split('/')[-1]}
    temperature_list = args.temperature_list

    gen_config = GenerationConfig(num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, min_new_tokens=5, 
    top_k=args.top_k, do_sample=True, num_return_sequences=args.return_sequences)
 
    queries_expanded = {}

    for query_id in queries:
        queries_expanded[query_id] = {"0": queries[query_id]}

    model = T5QueryGenerator(base_model=model_params['base_model'], max_tokens=model_params['max_length'], device=device)
    #model = LLaMAQueryGenerator(base_model=model_params['base_model'], max_tokens=model_params['max_length'], device=device)

    for temperature in tqdm(temperature_list):
        gen_config.temperature = temperature

        queries_temp_expanded = {}

        new_queries = filter_queries(list(queries.values()), replace_prompts)
        generated_queries = filter_queries(model.inference(new_queries, batchsize = 1, generation_config = gen_config), replace_prompts) 
        
        for i, query_id in enumerate(queries):
            queries_temp_expanded[query_id] = []
            base_q = queries[query_id]
            queries_temp_expanded[query_id].append(base_q)
            for j in range(args.return_sequences):
                queries_temp_expanded[query_id].append(generated_queries[i*args.return_sequences + j])
                #queries_temp_expanded[query_id].append(f'{base_q} {generated_queries[i*args.return_sequences + j]}')

        add_nonoverlap_queries(queries_expanded, queries_temp_expanded)

    output_dir = f'{args.queries[:-5]}_{model_params["name"]}'
    with safe_open_w(f'{output_dir}.json') as f:
        json.dump(queries_expanded, f, indent=4)

if __name__ == '__main__':
    main()