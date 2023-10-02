import os
import json
import glob
import string
import argparse
import pandas as pd
import xml.etree.ElementTree as ET


from bcardoso_xml2json import parse_file_list
from collections import OrderedDict
from tqdm import tqdm
from typing import List

def safe_open_w(path: str):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def getXMLDataRecursive_str(element: str, tags: List[str]) -> List[str]:
    ''' Get information recursevely from .xml files, intending to extract info (bottom-most level) and pass to a list form
    '''
    data = list()
    # only end-of-line elements have important text, at least in this example
    if len(element) == 0:
        if element.text is not None:
            data.append(element.text)
    # otherwise, go deeper and add to the current tag
    else:
        for el in element:
            if el.tag in tags:
                for data_point in getXMLDataRecursive_str(el, tags):
                    data.append(data_point)
    return data

def getXMLDataRecursive_dict(element: str, tags: List[str], parent: str) -> List[str]:
    ''' Get information recursevely from .xml files, intending to extract info (bottom-most level) and pass to a list form
    '''
    data = {}
    # only end-of-line elements have important text, at least in this example
    if len(element) == 0:
        if element.text is not None:
            data[parent](element.text)

    # otherwise, go deeper and add to the current tag
    else:
        for el in element:
            if el.tag in tags:
                struct_tag = parent if el.tag == "textblock" else el.tag
                recursive_res = getXMLDataRecursive_dict(el, tags, struct_tag)
                for data_point in recursive_res:
                    data[struct_tag] = recursive_res[data_point]
    return data

full_relevant_tags = ["brief_title","official_title","brief_summary", "condition", "eligibility",
                      "study_pop","criteria","gender","minimum_age","maximum_age","healthy_volunteers", "textblock"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default= "TREC2023/raw")
    parser.add_argument('--output', type=str, default= "TREC2023/TREC2023_CT-TrialGPT-exclusion.json")
    parser.add_argument("--list_tags", nargs="+", help='list of tags to parse', default=['brief_summary', 'detailed_description', 'eligibility','criteria', 'textblock'])
    parser.add_argument("--struct", type=str, help='seperate fields in .json', choices=["y", "yes", "n", "no"], default='y')
    parser.add_argument("--delimiter", type=str, help='delimiter between fields', default='\n')
    args = parser.parse_args()

    # python parse_ct-corpus_xml2json.py --input TREC2021/raw --output TREC2021/TREC2021_CT.json

    with safe_open_w(args.output) as output_file_json:
        with safe_open_w(f'{args.output}l') as output_file_json_lines:
            output_dict = {}

            #Recursively lists all xml files from input dir, concatenate all content
            if args.struct in ["n", "no"]:
                pass
                for xml_f_path in tqdm(glob.iglob(f'{args.input}/**/NCT*.xml', recursive=True)):
                    xml_f = ET.parse(xml_f_path)
                    parsed_data = getXMLDataRecursive_str(xml_f.getroot(), args.list_tags)
                    CT_content = ''.join([str(field) + args.delimiter for field in parsed_data]).strip("\n")

                    #add to CT Corpus JSON
                    CT_name = str(xml_f_path.split('/')[-1][:-4])
                    output_dict[CT_name] = CT_content
                    
                    #add to CT Corpus JSON Lines, if we want to read a certain number of CT's at a time
                    output_file_json_lines.write(json.dumps({"id": CT_name, "content": CT_content}, ensure_ascii=False, indent=4) + '\n')
            
            elif args.struct in ["y", "yes"]:
                output_dict = parse_file_list(glob.iglob(f'{args.input}/**/NCT*.xml', recursive=True))
                #output_dict = parse_file_list(glob.iglob(f'{args.input}/**/NCT*.xml', recursive=True))

                for ct in tqdm(output_dict):
                    output_str = f'Title: {output_dict[ct]["brief_title"] if not output_dict[ct]["official_title"] else output_dict[ct]["official_title"]}\n'
                    
                    if output_dict[ct]["condition"]:
                        output_str += f'Target Diseases: {output_dict[ct]["condition"]}\n'

                    if output_dict[ct]["intervention"]:
                        output_str += f'Interventions: {output_dict[ct]["intervention"]}\n'

                    if output_dict[ct]["brief_summary"]:
                        output_str += f'Summary: {output_dict[ct]["brief_summary"]}\n'

                    output_str += f'Exclusion Criteria: \n'
                    if output_dict[ct]["eligibility"]["minimum_age"] or output_dict[ct]["eligibility"]["maximum_age"]:
                        if output_dict[ct]["eligibility"]["minimum_age"] and "N/A" not in output_dict[ct]["eligibility"]["minimum_age"]:
                            output_str += f'age <= {output_dict[ct]["eligibility"]["minimum_age"]}'
                        if output_dict[ct]["eligibility"]["maximum_age"] and "N/A" not in output_dict[ct]["eligibility"]["maximum_age"]:
                            output_str += f'age => {output_dict[ct]["eligibility"]["maximum_age"]}'
                    
                    if output_dict[ct]["eligibility"]["gender"] and output_dict[ct]["eligibility"]["gender"] != "All" and output_dict[ct]["eligibility"]["gender"] != "ALL":
                        output_str += f' gender != {output_dict[ct]["eligibility"]["gender"]}\n'

                    if output_dict[ct]["eligibility"]["exclusion_criteria"]:
                        output_str += f' {output_dict[ct]["eligibility"]["exclusion_criteria"][20:]}\n'

                    output_dict[ct] = output_str
                    output_file_json_lines.write(json.dumps(output_dict[ct], ensure_ascii=False, indent=4) + '\n')
            else:
                raise TypeError("Option not available for CT parsing.")

        # Output to file
        output_file_json.write(json.dumps(output_dict, ensure_ascii=False, indent=4))