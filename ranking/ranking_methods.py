import json
import argparse
import os

from utils.utils_IO import safe_open_w

from typing import Dict
from pyserini.search.lucene import LuceneSearcher
from ranx import Qrels, Run, evaluate


def get_index_paths(base_dir : str) -> Dict:
    indexes = {}
    for term in os.listdir(base_dir):
        if term == 'freetxt':
            indexes['all'] = f'{base_dir}{term}/'
        else:
            indexes[term] = f'{base_dir}{term}/'
    return indexes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', type=str, help='path to dir with several indexes', required=True)
    parser.add_argument('--queries', type=str, help='path to queries file', default="../queries/queries2021.json")

    parser.add_argument('--qrels_bin', type=str, help='path to qrles file in binary form', default="../qrels/qrels2021/qrels2022_binary.json")
    parser.add_argument('--qrels_similiar', type=str, help='path to qrles file in similarity form', default="../qrels/qrels2021/qrels2022_similiar.json")

    parser.add_argument('--metrics_bin', nargs='+', type=str, help='list of metrics to calculate from binary labels', default=["precision@10", "r-precision", "mrr", \
    "recall@10", "recall@100", "recall@500", "recall@1000", "recall"])
    parser.add_argument('--metrics_similiar', nargs='+', type=str, help='list of metrics to calculate from 0 1 2 labels', default=["ndcg@10"])
    parser.add_argument('--K', type=int, help='retrieve top K documents', default=1000)

    parser.add_argument('--run', type=int, help='run number', default=1)
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="../outputs/ranking/")
    args = parser.parse_args()

    index_paths = get_index_paths(args.index_dir)

    queries = json.load(open(args.queries))
    qrels_bin = json.load(open(args.qrels_bin))
    qrels_similiar = json.load(open(args.qrels_similiar))
    metrics_bin = args.metrics_bin
    metrics_similiar = args.metrics_similiar

    for index_name in index_paths:

        run_dict = {}
        searcher = LuceneSearcher(index_paths[index_name])
        searcher.set_bm25()

        # Retrieve
        for query_id in queries:
            if query_id not in run_dict:
                run_dict[query_id] = {}

            hits = searcher.search(queries[query_id], k=args.K)
            for hit in hits:
                run_dict[query_id][hit.docid] = hit.score

        run = Run(run_dict, name=f"BM25_{index_name}")
        run.save(f'{args.output_dir}run-{args.run}/res_mbm25.json')

        # Evaluate
        results = {}
        if metrics_bin:
            results = evaluate(Qrels(qrels_bin), run, metrics_bin)
        if metrics_similiar:
            results.update(evaluate(Qrels(qrels_similiar), run, metrics_similiar))

        for metric in results:
            results[metric] = round(results[metric], 4)

        with safe_open_w(f'{args.output_dir}run-{args.run}/res_metrics.json') as output_f:
            json.dump(results, output_f, indent=4)