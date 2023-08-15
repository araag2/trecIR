import json
import argparse
import os
import pandas as pd
import sys

sys.path.insert(0, "../")

from utils_IO import safe_open_w

from tqdm import tqdm
from typing import Dict
from pyserini.search.lucene import LuceneSearcher
from pyserini.trectools import TrecRun
from pyserini.fusion import reciprocal_rank_fusion
from ranx import Qrels, Run, evaluate

sys.path.insert(0, '../')

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

def main():
    parser = argparse.ArgumentParser()
    # Path to indexes directory
    parser.add_argument('--index_dir', type=str, help='path to dir with several indexes', default="../datasets/TREC2023/indexes/")

    # Path to queries and qrels files
    parser.add_argument('--queries', type=str, help='path to queries file', default="../queries/TREC2023/queries2023_manual-expand.json")

    # BM25 parameters
    parser.add_argument('--K', type=int, help='retrieve top K documents', default=10000)

    # Output options and directory
    parser.add_argument('--save_hits', type=str, help='save hit dictionaries', choices=['y', 'n'], default='y')
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="../outputs/TREC2023/ranking/")
    args = parser.parse_args()

    missing_trials = {'NCT04510038', 'NCT04418518', 'NCT04205084', 'NCT02954432', 'NCT04659772', 'NCT01227460', 'NCT02441088', 'NCT04701840', 'NCT02692924', 'NCT04753255', 'NCT04361123', 'NCT03391388', 'NCT04359277', 'NCT03793829', 'NCT04595149', 'NCT01761058', 'NCT04363502', 'NCT03422237', 'NCT04707222', 'NCT04817787'}

    index_paths = get_index_paths(args.index_dir)

    queries = json.load(open(args.queries))

    run_name = f'{args.output_dir}run-BM25'


    for index_name in index_paths:
        index_output_name = f'{run_name}/res-{index_name}'

        searcher = LuceneSearcher(index_paths[index_name])
        searcher.set_bm25()

        results = {}

        for query_id in tqdm(queries):
            results[query_id] = {'docs' : [], 'scores' : []}

            hits = searcher.search(queries[query_id], k=args.K)
            for hit in hits:
                if hit.docid not in missing_trials:
                    results[query_id]['docs'].append(hit.docid)
                    results[query_id]['scores'].append(hit.score)

        with safe_open_w(f'{index_output_name}-top-10000-hits.json') as output_f:
            json.dump(results, output_f, indent=4)

if __name__ == '__main__':
    main()