import os
import json
import argparse
import xml.etree.ElementTree as ET

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
    args = parser.parse_args()

    # python parse_topics_xml2json.py --input TREC2021/TREC2021_raw/topics2021.xml --output ../queries/TREC2021/queries2021.json
    # python parse_topics_xml2json.py --input TREC2021/TREC2021_raw/topics2022.xml --output ../queries/TREC2021/queries2022.json

    output_dict = {}

    xml_root  = ET.parse(args.input).getroot()
    for element in tqdm(xml_root.findall('topic')):
        topic_id = element.attrib['number']
        #description = ct_utils.remove_whitespaces_except_one_space_from_field(element.text) TODO: Check why this was here
        output_dict [topic_id] = element.text

    with safe_open_w(args.output) as output_file_json:
        # Output to file
        output_file_json.write(json.dumps(output_dict, ensure_ascii=False, indent=4))