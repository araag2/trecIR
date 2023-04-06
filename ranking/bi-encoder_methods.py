import json
import argparse
import os
import pickle
import torch
import numpy as np

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

def encode_subset_corpus(model : SentenceTransformer, corpus_path : str, number_files : int) -> Dict:
    """
    Just for testing purposes, encodes subset of corpus using SentenceTransformer model
    """
    corpus_raw = json.load(open(corpus_path))
    txt_encoded = model.encode([corpus_raw[doc_id] for doc_id in corpus_raw][:number_files], show_progress_bar=True)
    corpus_processed = {}
    for doc_id, enc_txt in zip(corpus_raw, txt_encoded):
        corpus_processed[doc_id] = enc_txt
    return corpus_processed

def encode_corpus(model : SentenceTransformer, corpus_path : str, save_path : str) -> Dict:
    """
    Encodes corpus using SentenceTransformer model

    Args:
        model (SentenceTransformer): model used to encode corpus
        corpus_path (str): path to corpus file
        save_path (str): path to save encoded corpus to memory
    """
    corpus_raw = json.load(open(corpus_path))
    txt_encoded = model.encode([corpus_raw[doc_id] for doc_id in corpus_raw], show_progress_bar=True)
    
    corpus_processed = {}
    for doc_id, enc_txt in zip(corpus_raw, txt_encoded):
        corpus_processed[doc_id] = enc_txt

    if save_path != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(f'{save_path}.pkl', 'wb') as f_out:
            pickle.dump(corpus_processed, f_out, protocol = pickle.HIGHEST_PROTOCOL)

    return corpus_processed

def load_corpus_from_memory(corpus_memory_path : str) -> Dict:
    """
    Loads corpus from memory using pickle

    Args:
        corpus_memory_path (str): path to corpus memory file
    """
    return pickle.load(open(corpus_memory_path, 'rb'))

def main():
    parser = argparse.ArgumentParser()
    # Path to indexes directory
    parser.add_argument('--model_name', type=str, help='name of the bi-encoder model used to encode', default='msmarco-distilbert-base-v4')

    # Path to corpus for encoding purpuoses, and wether to save it to memory
    parser.add_argument('--corpus_path', type=str, help='path to corpus file', default="../datasets/TREC2021/TREC2021_CT_corpus.json")
    parser.add_argument('--corpus_save_path', type=str, help='save corpus to memory', default="../datasets/TREC2021/encoded_corpus/corpus")

    # Wether to load corpus from memory, and path from where to load it
    parser.add_argument('--corpus_load', type=str, help='load corpus from memory', choices=['y', 'n'], default='y')
    parser.add_argument('--corpus_load_path', type=str, help='path to corpus memory file', default="../datasets/TREC2021/encoded_corpus/corpus-msmarco-distilbert-base-v4-full.pkl")

    # Path to queries and qrels files
    parser.add_argument('--queries', type=str, help='path to queries file', default="../queries/TREC2021/queries2021.json")
    parser.add_argument('--queries_expanded', type=str, help='path to queries file used for RRF', default="../queries/TREC2021/queries2021-expanded.json")
    parser.add_argument('--qrels_bin', type=str, help='path to qrles file in binary form', default="../qrels/TREC2021/qrels2021/qrels2021_binary.json")
    parser.add_argument('--qrels_similiar', type=str, help='path to qrles file in similarity form', default="../qrels/TREC2021/qrels2021/qrels2021_similiar.json")

    # List of metrics to calculate
    parser.add_argument('--metrics_bin', nargs='+', type=str, help='list of metrics to calculate from binary labels', default=["precision@10", "r-precision", "mrr", \
    "recall@10", "recall@100", "recall@500", "recall@1000", "recall"])
    parser.add_argument('--metrics_similiar', nargs='+', type=str, help='list of metrics to calculate from 0 1 2 labels', default=["ndcg@10"])

    # Top k documents to retrieve
    parser.add_argument('--top_k', type=int, help='retrieve top K documents', default=1000)

    # How many docs to retrieve for each query
    parser.add_argument('--run', type=int, help='run number', default=1)

    # Output options and directory
    parser.add_argument('--save_hits', type=str, help='save hit dictionaries', choices=['y', 'n'], default='n')
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="../outputs/TREC2021/ranking/")
    args = parser.parse_args()

    model = SentenceTransformer(args.model_name, device = 'cuda' if torch.cuda.is_available() else 'cpu')
    corpus = load_corpus_from_memory(args.corpus_load_path) if args.corpus_load == 'y' else encode_corpus(model, args.corpus_path, f'{args.corpus_save_path}-{args.model_name}')

    queries = json.load(open(args.queries))

    qrels_bin = json.load(open(args.qrels_bin))
    qrels_similiar = json.load(open(args.qrels_similiar))

    metrics_bin = args.metrics_bin
    metrics_similiar = args.metrics_similiar

    run_name = f'{args.output_dir}run-{args.run}-bi-encoder-{args.model_name}'

    query_ids = list(queries.keys())
    encoded_queries = model.encode([queries[query_id] for query_id in queries], show_progress_bar=True)

    corpus_ids = list(corpus.keys())
    corpus = np.array([corpus[doc_id] for doc_id in corpus])

    hits = util.semantic_search(encoded_queries, corpus, top_k=args.top_k)
    
    run = {}
    for query_id, query_hits in zip(query_ids, hits):
        run[query_id] = {}
        for hit_id in query_hits:
            run[query_id][corpus_ids[hit_id['corpus_id']]] = hit_id['score']
    run = Run.from_dict(run)

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