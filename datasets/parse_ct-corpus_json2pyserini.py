import os
import json
import argparse

from tqdm import tqdm

def safe_open_w(path: str):
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
    parser.add_argument('--input', type=str, help='path to input file for struct txt form', required=True)
    parser.add_argument('--output', type=str, help='path to output dir/file for full corpus', required=True)
    parser.add_argument('--fields', type=str, help='fields to analyze', nargs='+', \
        default = ['free_txt'])
    
    args = parser.parse_args()

    # python parse_ct-corpus_json2pyserini.py --input /user/home/aguimas/data/PhD/trecIR/datasets/TREC2021/TREC2021_CT.json
    # --output TREC2021/base_indexes/

    with open(args.input, 'r') as json_struct_file:
        struct_data = json.load(json_struct_file)
    
        res_by_field = {}
        for field in args.fields:
            res_by_field[field] = []
    
        if args.fields == ["free_txt"]:
            for ct in tqdm(struct_data):
                res_by_field["free_txt"].append({'id' : ct, 'contents' : struct_data[ct]})

            with safe_open_w(f'{args.output}/free_txt/index_free_txt.json') as output_file_json:
                json.dump(res_by_field["free_txt"], output_file_json, indent=4)

        else:
            for ct in tqdm(struct_data):
                for field in struct_data[ct]:
                    if field not in res_by_field:
                        res_by_field[field] = []

                    elif field == 'eligibility': 
                        for subfield in struct_data[ct]['eligibility']:
                            if subfield not in res_by_field:
                                res_by_field[subfield] = []
                            res_by_field[subfield].append({'id' : ct, 'contents' : struct_data[ct]['eligibility'][subfield]})

                    else:
                        res_by_field[field].append({'id' : ct, 'contents' : struct_data[ct][field]})

                if 'free_txt' in args.fields:
                    # Append free txt concatenated in order of importance
                    ct_free_txt = ""
                
                    for field in struct_data[ct]:
                        for subfield in struct_data[ct][field]:
                            ct_free_txt += subfield + "\n"

                    res_by_field["free_txt"].append({'id' : ct, 'contents' : ct_free_txt})                
        
            for field in tqdm(args.fields):
                with safe_open_w(f'{args.output}/{field}/index_{field}.json') as output_file_json:
                    json.dump(res_by_field[field], output_file_json, indent=4)