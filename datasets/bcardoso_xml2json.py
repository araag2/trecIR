import os
import re
import xml.etree.ElementTree as ET
import json
import string

from tqdm import tqdm

def get_brief_title(root):
    try:
        return root.find('brief_title').text
    except AttributeError:
        return None


def get_official_title(root):
    try:
        return root.find('official_title').text
    except AttributeError:
        return None


def get_brief_summary(root):
    try:
        return root.find('brief_summary').findtext('textblock')
    except AttributeError:
        return None


def get_detailed_description(root):
    try:
        return root.find('detailed_description').findtext('textblock')
    except AttributeError:
        return None


def get_conditions(root):
    try:
        _ = root.find('condition').text  # Just to throw exception
        return list(map(lambda e: e.text, root.findall('condition')))
    except AttributeError:
        return None


def get_eligibility_study_pop(root):
    try:
        return root.find('eligibility').find('study_pop').findtext('textblock')
    except AttributeError:
        return None


def get_eligibility_criteria(root):
    try:
        return root.find('eligibility').find('criteria').findtext('textblock')
    except AttributeError:
        return None


def get_eligibility_gender(root):
    try:
        return root.find('eligibility').find('gender').text
    except AttributeError:
        return None


def get_eligibility_minimum_age(root):
    try:
        return root.find('eligibility').find('minimum_age').text
    except AttributeError:
        return None


def get_eligibility_maximum_age(root):
    try:
        return root.find('eligibility').find('maximum_age').text
    except AttributeError:
        return None


def get_eligibility_healthy_volunteers(root):
    try:
        return root.find('eligibility').find('healthy_volunteers').text
    except AttributeError:
        return None


def is_eligibility_criteria_semi_structured(criteria):
    if len(re.findall(r'Inclusion Criteria:[\s\S]*Exclusion Criteria:', criteria, flags=re.MULTILINE | re.DOTALL)) == 0:
        return False
    else:
        return True


def is_field_applicable(field):
    field = remove_whitespaces_except_one_space_from_field(field)
    if field == 'N/A':
        return False
    else:
        return True


def remove_whitespaces_except_one_space_from_field(field):
    if field is None:
        return None

    whitespace_except_space = string.whitespace.replace(' ', '')

    field.strip(whitespace_except_space)
    field = ' '.join(field.split())
    return field


def concatenate_fields(fields):
    return ' '.join(filter(None, fields))


def space_tokenizer(field):
    return field.split(' ')


def parse_file_list(file_list):
    data = {}
    for xml_f_path in tqdm(file_list):
        root = ET.parse(xml_f_path).getroot()

        brief_title = remove_whitespaces_except_one_space_from_field(
            get_brief_title(root))

        if brief_title == "[Trial of device that is not approved or cleared by the U.S. FDA]":
            continue

        official_title = remove_whitespaces_except_one_space_from_field(
            get_official_title(root))

        brief_summary = remove_whitespaces_except_one_space_from_field(
            get_brief_summary(root))

        detailed_description = remove_whitespaces_except_one_space_from_field(
            get_detailed_description(root))

        conditions = get_conditions(root)

        eligibility_study_pop = remove_whitespaces_except_one_space_from_field(
            get_eligibility_study_pop(root))

        eligibility_criteria = remove_whitespaces_except_one_space_from_field(
            get_eligibility_criteria(root))

        eligibility_gender = get_eligibility_gender(root)
        eligibility_minimum_age = get_eligibility_minimum_age(root)
        eligibility_maximum_age = get_eligibility_maximum_age(root)
        eligibility_healthy_volunteers = get_eligibility_healthy_volunteers(root)

        CT_name = str(xml_f_path.split('/')[-1][:-4])

        ct = {
            "id": CT_name,
            "brief_title": brief_title,
            "official_title": official_title,
            "brief_summary": brief_summary,
            "detailed_description": detailed_description,

            "condition": conditions,

            "eligibility": {
                "study_pop": eligibility_study_pop,
                "criteria": eligibility_criteria,
                "gender": eligibility_gender,
                "minimum_age": eligibility_minimum_age,
                "maximum_age": eligibility_maximum_age,
                "healthy_volunteers": eligibility_healthy_volunteers
            }
        }

        data[CT_name] = ct
    return data