import json
import argparse
import os
import pandas as pd

from utils_IO import safe_open_w

from tqdm import tqdm
from typing import Dict
from pyserini.search.lucene import LuceneSearcher
from pyserini.trectools import TrecRun
from pyserini.fusion import reciprocal_rank_fusion
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

def main():
    parser = argparse.ArgumentParser()
    # Path to indexes directory
    parser.add_argument('--index_dir', type=str, help='path to dir with several indexes', default="../datasets/TREC2021/pyserini_indexes/")

    # Path to queries and qrels files
    parser.add_argument('--queries', type=str, help='path to queries file', default="../queries/TREC2021/queries2021.json")
    parser.add_argument('--queries_expanded', type=str, help='path to queries file used for RRF', default="../queries/TREC2021/queries2021-expanded.json")
    parser.add_argument('--qrels_bin', type=str, help='path to qrles file in binary form', default="../qrels/TREC2021/qrels2021/qrels2021_binary.json")
    parser.add_argument('--qrels_similiar', type=str, help='path to qrles file in similarity form', default="../qrels/TREC2021/qrels2021/qrels2021_similiar.json")

    # List of metrics to calculate
    parser.add_argument('--metrics_bin', nargs='+', type=str, help='list of metrics to calculate from binary labels', default=["precision@10", "r-precision", "mrr", \
    "recall@10", "recall@100", "recall@500", "recall@1000", "recall"])
    parser.add_argument('--metrics_similiar', nargs='+', type=str, help='list of metrics to calculate from 0 1 2 labels', default=["ndcg@10"])

    # BM25 parameters
    parser.add_argument('--K', type=int, help='retrieve top K documents', default=1000)

    # Binary flags to enable or disable ranking methodss
    parser.add_argument('--rm3', type=str, help='enable or disable rm3', choices=['y', 'n'], default='y')
    parser.add_argument('--rrf', type=str, help='enable or disable rrf', choices=['y', 'n'], default='y')
    
    # Run number
    parser.add_argument('--run', type=int, help='run number', default=1)

    # Output options and directory
    parser.add_argument('--save_hits', type=str, help='save hit dictionaries', choices=['y', 'n'], default='n')
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="../outputs/TREC2021/ranking/")
    args = parser.parse_args()

    index_paths = get_index_paths(args.index_dir)

    queries = json.load(open(args.queries)) if args.rm3 == 'n' else json.load(open(args.queries_expanded))
    qrels_bin = json.load(open(args.qrels_bin))
    qrels_similiar = json.load(open(args.qrels_similiar))
    metrics_bin = args.metrics_bin
    metrics_similiar = args.metrics_similiar

    run_name = f'{args.output_dir}run-{args.run}-BM25'
    run_name += '-RM3' if args.rm3 == 'y' else ''
    run_name += '-RRF' if args.rrf == 'y' else ''

    for index_name in tqdm(index_paths):
        index_output_name = f'{run_name}/res-{index_name}'

        searcher = LuceneSearcher(index_paths[index_name])
        searcher.set_bm25()

        if args.rm3 == 'y':
            searcher.set_rm3()

        run_result = None
        run = None

        # Retrieve
        if args.rrf == 'y':
            run_result = []

            for query_id in tqdm(queries):
                for sintetic_query_id in queries[query_id]:
                    hits = searcher.search(queries[query_id][sintetic_query_id], k=args.K)
                    sintetic_query_results = []
                    for rank, hit in enumerate(hits, start=1):
                        sintetic_query_results.append((query_id, 'Q0', hit.docid, rank, hit.score, f'{query_id}_{sintetic_query_id}'))
                    run_result.append(TrecRun.from_list(sintetic_query_results))

            run = reciprocal_rank_fusion(run_result, depth=args.K, k=args.K)
            run = pd.DataFrame(data=run.to_numpy(), columns=['q_id', '_1', 'doc_id', '_2', 'score', '_3'])
            run = run.astype({'score': 'float'})
            run = Run.from_df(run)

        elif args.rrf == 'n':
            run_result = {}

            for query_id in tqdm(queries):
                if query_id not in run_result:
                    run_result[query_id] = {}

                hits = searcher.search(queries[query_id], k=args.K)
                for hit in hits:
                    run_result[query_id][hit.docid] = hit.score

            run = Run(run_result)

        if args.save_hits == 'y':
            safe_open_w(f'{index_output_name}-hits.json')
            run.save(f'{index_output_name}-hits.json')

        # Evaluate
        results = {}
        if metrics_bin:
            results = evaluate(Qrels(qrels_bin), run, metrics_bin)
        if metrics_similiar:
            results.update({"ndcg": evaluate(Qrels(qrels_similiar), run, metrics_similiar)})

        for metric in results:
            results[metric] = round(results[metric], 4)

        with safe_open_w(f'{index_output_name}-metrics.json') as output_f:
            json.dump(results, output_f, indent=4)

if __name__ == '__main__':
    main()