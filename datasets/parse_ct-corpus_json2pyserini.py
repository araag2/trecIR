import os
import json
import argparse

from utils_IO import safe_open_w
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', type=str, help='path to input file for free txt form', required=True)
    parser.add_argument('--input_struct', type=str, help='path to input file for struct txt form', required=True)
    parser.add_argument('--output', type=str, help='path to output dir/file for full corpus', required=True)
    parser.add_argument('--fields', type=str, help='fields to analyze', nargs='+', \
        default = ['free_txt', 'brief_title', 'official_title', 'brief_summary', 'detailed_description', 'study_pop', 'criteria'])
    
    args = parser.parse_args()

    # python parse_ct-corpus_json2pyserini.py --input_txt TREC2021/TREC2021_CT_corpus.json --input_struct TREC2021/TREC2021_CT_corpus-struct.json --output TREC2021/pyserini_base_index/

    with open(args.input_struct, 'r') as json_struct_file:
        struct_data = json.load(json_struct_file)
    
        res_by_field = {}
        for field in args.fields:
            res_by_field[field] = []
    
        for ct in tqdm(struct_data):
            for field in struct_data[ct]:
                if field == 'id' or field == 'condition':
                    continue
    
                elif field == 'eligibility': 
                    for subfield in struct_data[ct]['eligibility']:
                        if subfield == 'study_pop' or subfield == 'criteria':
                            res_by_field[subfield].append({'id' : ct, 'contents' : struct_data[ct]['eligibility'][subfield]})

                else:
                    res_by_field[field].append({'id' : ct, 'contents' : struct_data[ct][field]})

            # Append free txt concatenated in order of importance
            ct_free_txt = ""
            for field in args.fields:
                if field != 'free_txt' and res_by_field[field] != [] and res_by_field[field][-1]['id'] == ct and res_by_field[field][-1]['contents'] != None:
                    ct_free_txt += " " + res_by_field[field][-1]['contents'] 

            res_by_field["free_txt"].append({'id' : ct, 'contents' : ct_free_txt})                
    
        for field in tqdm(res_by_field):
            with safe_open_w(f'{args.output}/{field}/index_{field}.json') as output_file_json:
                json.dump(res_by_field[field], output_file_json, indent=4)