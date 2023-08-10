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
    parser.add_argument('--input', type=str, default="../queries/queries2021.json")
    parser.add_argument('--output', type=str, default="../queries/queries2021_split-lines.json")
    args = parser.parse_args()

    output_dict = {}

    with open(args.input, 'r') as input_file:
        input_file_json = json.load(input_file)

        for topic_id, description in tqdm(input_file_json.items()):
            res = ""
            counter = 0
            for line in description.split(".")[:-1]:
                res += f"{counter}. {line.strip()}\n"
                counter += 1
            output_dict[topic_id] = res

    with safe_open_w(args.output) as output_file_json:
        # Output to file
        output_file_json.write(json.dumps(output_dict, ensure_ascii=False, indent=4))