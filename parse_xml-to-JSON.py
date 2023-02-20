import os
import json
import glob
import string
import argparse
import pandas as pd
import xml.etree.ElementTree as ET

from collections import OrderedDict
from tqdm import tqdm
from typing import List

def safe_open_w(path: str):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def remove_whitespaces_except_one_space_from_field(field: str) -> str:
    """Removes all the whitespaces in the texts that are more than a single neighbouring space
    
    Parameters
    ----------   
    field : str
        field's text

    Returns
    -------        
    str
        new string with a single whitespace between words
    """

    whitespace_except_space = string.whitespace.replace(' ', '')

    field.strip(whitespace_except_space)
    field = ' '.join(field.split())
    return field

def getXMLDataRecursive(element: str, tags: List[str]) -> List[str]:
    ''' Get information recursevely from .xml files, intending to extract valueable info only (bottom-most level)
    '''
    data = list()

    # only end-of-line elements have important text, at least in this example
    if len(element) == 0:
        if element.text is not None:
            data.append(remove_whitespaces_except_one_space_from_field(element.text))

    # otherwise, go deeper and add to the current tag
    else:
        for el in element:
            if tags and el.tag not in tags:
                continue
            within = getXMLDataRecursive(el, tags)
            for data_point in within:
                data.append(data_point)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to input dir/file', required=True)
    parser.add_argument('--output', type=str, help='path to output dir/file for full corpus', required=True)
    parser.add_argument("--list_tags", nargs="+", help='list of tags to parse', default=['brief_summary', 'detailed_description', 'eligibility','criteria', 'textblock'])
    parser.add_argument("--delimiter", type=str, help='delimiter between fields', default='\n')
    args = parser.parse_args()

    # python parse_xml-to-JSON.py --input datasets/TREC2021/TREC2021_CT_corpus.json --output datasets/TREC2021/TREC2021_CT_corpus.json

    with open("datasets/TREC2021/TREC2021_CT_corpus.json") as output_file_json_lines:
        json_input_queries = json.load(output_file_json_lines)
        print(json_input_queries)
        quit()

    with safe_open_w(args.output) as output_file_json:
        with safe_open_w(f'{args.output}l') as output_file_json_lines:
            output_dict = {}

            #Recursively lists all xml files from input dir
            for xml_f_path in tqdm(glob.iglob(f'{args.input}/**/*.xml', recursive=True)):
                xml_f = ET.parse(xml_f_path)
                parsed_data = getXMLDataRecursive(xml_f.getroot(), args.list_tags)
                CT_content = ''.join([str(field) + args.delimiter for field in parsed_data]).strip("\n")

                #add to CT Corpus JSON
                CT_name = str(xml_f_path.split('/')[-1][:-4])
                output_dict[CT_name] = CT_content
                
                #add to CT Corpus JSON Lines, if we want to read a certain number of CT's at a time
                curr_CT = {"id": CT_name, "content": CT_content}
                output_file_json_lines.write(json.dumps(curr_CT, ensure_ascii=False, indent=4) + '\n')


        # Output to file
        output_file_json.write(json.dumps(output_dict, ensure_ascii=False, indent=4))