
import os 
import yaml 
import json
from datetime import datetime

def partition_list(input_list, k):

    n = len(input_list)
    if n == 0: return [[] for _ in range(k)]  
    chunks = []
    start = 0
    
    for i in range(k):
        end = start + n // k + (1 if i < n % k else 0)
        chunks.append(input_list[start:end])
        start = end

    return chunks

def save_json(inputs: list, save_path: str):
    '''Save a list of dictionaries as JSON. If the file already exists, add the current list to the existing one only if it doesn't contain duplicates. If not, create one.'''
    if os.path.exists(save_path):
        
        with open(save_path, 'r') as file:
            data = json.load(file)
            existing_json_strings = {json.dumps(existing_dict, sort_keys=True) for existing_dict in data}
            new_data = [input_dict for input_dict in inputs if json.dumps(input_dict, sort_keys=True) not in existing_json_strings]
            data.extend(new_data)
        
        with open(save_path, 'w') as file:
            json.dump(data, file)
            
    else:
        with open(save_path, 'w') as file:
            json.dump(inputs, file)

def compare_dates(year_month, last_updated):
    iso_datetime = datetime.strptime(last_updated, "%Y-%m-%dT%H:%M:%SZ")
    given_datetime = datetime.strptime(year_month, "%Y-%m")
    return "False" if given_datetime > iso_datetime else "True" if given_datetime < iso_datetime else True

def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        elif isinstance(item, set):
            result.extend(item)
        else:
            result.append(item)
    return result
    
def load_yaml(path):

    with open(path, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return data

def load_or_create_yaml_list(path):

    try:
        with open(path, 'r') as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        data = []

    return data

def save_yaml(data, path):

    existing_data = load_or_create_yaml_list(path)
    combined_data = existing_data + data
    with open(path, 'w') as yaml_file:
        yaml.dump(combined_data, yaml_file, default_flow_style=False)

    print(f"Successfully saved yaml to {path}")

