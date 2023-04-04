import json
import argparse
import os
import pandas as pd
import pickle

from utils_IO import safe_open_w

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from typing import Dict
from ranx import Qrels, Run, evaluate

def get_index_paths(base_dir : str) -> Dict:
    """
    Returns dictionary with paths to pyserini indexes

    Args:
        base_dir (str): path to directory with indexes
    """
    indexes = {}
    for term in os.listdir(base_dir):
        indexes[term] = f'{base_dir}{term}/'
    return indexes

def encode_corpus(model : SentenceTransformer, corpus_path : str, save_path : str) -> Dict:
    corpus_raw = json.load(open(corpus_path))

    for doc in tqdm(corpus_raw):
        print(corpus_raw[doc])
        quit()

    corpus_processed = model.encode("")

    if save_path != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with safe_open_w(f'{save_path}.pkl', 'wb') as f_out:
            pickle.dump(corpus_processed, f_out, protocol = pickle.HIGHEST_PROTOCOL)

    return corpus_processed

def load_corpus_from_memory(corpus_memory_path : str) -> Dict:
    return pickle.load(open(corpus_memory_path, 'rb'))

def main():
    parser = argparse.ArgumentParser()
    # Path to indexes directory
    parser.add_argument('--model_name', type=str, help='name of the bi-encoder model used to encode', default='msmarco-distilbert-base-v4')

    # Path to corpus for encoding purpuoses, and wether to save it to memory
    parser.add_argument('--corpus_path', type=str, help='path to corpus file', default="../datasets/TREC2021/TREC2021_CT_corpus.json")
    parser.add_argument('--corpus_save_path', type=str, help='save corpus to memory', default="../datasets/TREC2021/encoded_corpus/corpus")

    # Wether to load corpus from memory, and path from where to load it
    parser.add_argument('--corpus_load', type=str, help='load corpus from memory', choices=['y', 'n'], default='n')
    parser.add_argument('--corpus_load_path', type=str, help='path to corpus memory file', default="")

    # Path to queries and qrels files
    parser.add_argument('--queries', type=str, help='path to queries file', default="../queries/TREC2021/queries2021.json")
    parser.add_argument('--queries_expanded', type=str, help='path to queries file used for RRF', default="../queries/TREC2021/queries2021-expanded.json")
    parser.add_argument('--qrels_bin', type=str, help='path to qrles file in binary form', default="../qrels/TREC2021/qrels2021/qrels2021_binary.json")
    parser.add_argument('--qrels_similiar', type=str, help='path to qrles file in similarity form', default="../qrels/TREC2021/qrels2021/qrels2021_similiar.json")

    # List of metrics to calculate
    parser.add_argument('--metrics_bin', nargs='+', type=str, help='list of metrics to calculate from binary labels', default=["precision@10", "r-precision", "mrr", \
    "recall@10", "recall@100", "recall@500", "recall@1000", "recall"])
    parser.add_argument('--metrics_similiar', nargs='+', type=str, help='list of metrics to calculate from 0 1 2 labels', default=["ndcg@10"])

    # How many docs to retrieve for each query
    parser.add_argument('--run', type=int, help='run number', default=1)

    # Output options and directory
    parser.add_argument('--save_hits', type=str, help='save hit dictionaries', choices=['y', 'n'], default='n')
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="../outputs/TREC2021/ranking/")
    args = parser.parse_args()

    model = SentenceTransformer(args.model_name)
    corpus = load_corpus_from_memory(args.corpus_load_path) if args.corpus_load == 'y' else encode_corpus(model, args.corpus_path, f'{args.corpus_save_path}-{args.model_name}')

    queries = json.load(open(args.queries))

    qrels_bin = json.load(open(args.qrels_bin))
    qrels_similiar = json.load(open(args.qrels_similiar))

    metrics_bin = args.metrics_bin
    metrics_similiar = args.metrics_similiar

    run_name = f'{args.output_dir}run-{args.run}-bi-encoder-{args.model_name}'

    # TODO: Implement ranking

    # Evaluate
    results = {}
    if metrics_bin:
        results = evaluate(Qrels(qrels_bin), run, metrics_bin)
    if metrics_similiar:
        results.update({"ndcg": evaluate(Qrels(qrels_similiar), run, metrics_similiar)})

    for metric in results:
        results[metric] = round(results[metric], 4)

    with safe_open_w(f'{run_name}/res-metrics.json') as output_f:
        json.dump(results, output_f, indent=4)

if __name__ == '__main__':
    main()