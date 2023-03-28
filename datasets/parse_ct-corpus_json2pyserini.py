import os
import json
import argparse

from utils.utils_IO import safe_open_w
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', type=str, help='path to input file for free txt form', required=True)
    parser.add_argument('--input_struct', type=str, help='path to input file for struct txt form', required=True)
    parser.add_argument('--output', type=str, help='path to output dir/file for full corpus', required=True)
    args = parser.parse_args()

    # python parse_ct-corpus_json2pyserini.py --input_txt TREC2021/TREC2021_CT_corpus.json \
    # --input_struct TREC2021/TREC2021_CT_corpus-struct.json \
    # --output TREC2021/pyserini_base_index/

    with open(args.input_txt, 'r') as json_freetxt_file:
        with safe_open_w(f'{args.output}/freetxt/index_freetxt.jsonl') as output_file_json:
            struct_data = json.load(json_freetxt_file)
            res = []
            for ct in tqdm(struct_data):
                res.append({'id' : ct, 'contents' : struct_data[ct]})
            json.dump(res, output_file_json, indent=4)
    
    #with open(args.input_struct, 'r') as json_struct_file:
    #    struct_data = json.load(json_struct_file)
    #
    #    res_by_field = {}
    #
    #    for ct in tqdm(struct_data):
    #        for field in struct_data[ct]:
    #            if field == 'id':
    #                continue
    #
    #            if field not in res_by_field:
    #                res_by_field[field] = []
    #
    #            res_by_field[field].append({'id' : ct, 'contents' : struct_data[ct][field]})                
    #
    #    for field in tqdm(res_by_field):
    #        with safe_open_w(f'{args.output}/{field}/index_{field}.json') as output_file_json:
    #            json.dump(res_by_field[field], output_file_json, indent=4)