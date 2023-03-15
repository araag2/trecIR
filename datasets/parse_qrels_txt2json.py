import os
import json
import argparse

from tqdm import tqdm

def safe_open_w(path: str):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to input dir/file', required=True)
    parser.add_argument('--output', type=str, help='path to output dir/file', required=True)
    parser.add_argument("--rel", nargs="+", help='0 is non-relevant, 1 is excluded and 2 is eligible', default=[0, 1, 2])
    parser.add_argument("--delimiter", type=str, help='delimiter between fields', default=' ')
    args = parser.parse_args()

    # python parse_qrels_txt2json.py --input TREC2021/TREC2021_raw/qrels2021.txt --output ../qrels/TREC2021/qrels2021.json

    output_dict = {}

    with open(args.input, 'r') as input_raw:
        for line in tqdm(input_raw.readlines()):
            line = line.rstrip().split(args.delimiter)
            query = line[0]
            document = line[2]
            relevance = args.rel[int(line[3])]

            if query not in output_dict:
                output_dict[query] = {}
            output_dict[query][document] = relevance

    with safe_open_w(args.output) as output_file_json:
        # Output to file
        output_file_json.write(json.dumps(output_dict, ensure_ascii=False, indent=4))