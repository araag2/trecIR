import json
import argparse
import os

from typing import Dict
from pyserini.search.lucene import LuceneSearcher
from ranx import Qrels, Run, evaluate

def safe_open_w(path: str):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

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
    parser.add_argument('--qrels', type=str, help='path to qrels file', default="../qrels/qrels2021/qrels2022.json")
    parser.add_argument('--qrels_bin', type=str, help='path to qrles file in binary form', default="../qrels/qrels2021/qrels2022_binary.json")
    parser.add_argument('--qrels_similiar', type=str, help='path to qrles file in similarity form', default="../qrels/qrels2021/qrels2022_similiar.json")

    parser.add_argument('--metrics', type=str, help='list of metrics to calculate', default="")
    parser.add_argument('--K', type=int, help='retrieve top K documents', default=1000)

    parser.add_argument('--run', type=int, help='run number', default=1)
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="../outputs/ranking/")
    args = parser.parse_args()

    index_paths = get_index_paths(args.index_dir)
    quit()

    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))
    qrels_bin = json.load(open(args.qrels_bin))
    qrles_similiar = json.load(open(args.qrels_similiar))

    #with open("/home/bd.cardoso/my_datasets/trec_ct_2021/qrels/qrels_3_scale.json", "r") as r:
    #    qrels_3_scale = json.load(r)
    #with open("/home/bd.cardoso/my_datasets/trec_ct_2021/qrels/qrels_2_scale_v1.json", "r") as r:
    #    qrels_2_scale_v1 = json.load(r)

    for index_name in index_paths:

        run_dict = {}
        searcher = LuceneSearcher(index_paths[index_name])
        searcher.set_bm25()

        # Retrieve
        for topic_id in topics:
            if topic_id not in run_dict:
                run_dict[topic_id] = {}

            hits = searcher.search(topics[topic_id], k=args.K)
            for hit in hits:
                run_dict[topic_id][hit.docid] = hit.score

        run = Run(run_dict, name=f"BM25_{index_name}")
        run.save(f'{args.output_dir}run-{args.run}/res_mbm25.json')

        # Evaluate
        ndcg = evaluate(Qrels(qrels_3_scale), run, "ndcg@10")
        results = evaluate(Qrels(qrels_2_scale_v1), run,
                        ["precision@10", "r-precision", "mrr", "recall@10", "recall@100", "recall@500", "recall@1000",
                            "recall"])
        results.update({"ndcg@10": ndcg})


        for metric in results:
            results[metric] = round(results[metric], 4)


        with safe_open_w(f'{args.output_dir}run-{args.run}/res_metrics.json') as output_f:
            json.dump(results, output_f, indent=4)