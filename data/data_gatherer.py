import os, base64, time, concurrent.futures, logging, datetime, json, zipfile
from io import BytesIO
import requests, pickle
from bs4 import BeautifulSoup
from typing import List, Union
from urllib.parse import urlparse
from data.utils import *
import queue

'''Class for gathering github open-source projects'''

GIT_ACCESS_TOKEN = "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"
REQ_HEADER = {
		"Authorization": f"token {GIT_ACCESS_TOKEN}",
		"Accept": "application/vnd.github.v3+json"
	}
_token_status = None 
_req_history = set()
_failed_req_history = set()
logging.basicConfig(filename="scrap.log", level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

##############################################################################################

def get_from_repos_name(inputs: Union[str, dict], onfly=False): 
    '''
    Get all files from a repository name
    Args:
        inputs (str): Repository name
        onfly (bool): Whether to get file contents on the fly or not
    '''
    start_time = time.time()

    if isinstance(inputs, dict) and "url" in inputs: # already received request JSON
        response_json = inputs
        response = None 

    elif isinstance(inputs, str):
        parsed = urlparse(inputs)
        if all([parsed.scheme, parsed.netloc]): # url
            reqUrl = inputs
        else:
            assert len(inputs.split("/")) == 2
            splited_name = inputs.split("/")
            reqUrl = f"https://api.github.com/repos/{splited_name[0]}/{splited_name[1]}"

        response = cus_request_get(reqUrl, data="")
        response_json = response.json()

    elif isinstance(inputs, requests.models.Response) and inputs.status_code == 200:
        response = inputs
        reqUrl = response.url
        response_json = response.json()
    else:
        raise ValueError("Unrecognized input type")

    if onfly:
        treeUrl = f"{reqUrl}/contents"
        file_req = []
        def tb_serach(url):
            response_tb = cus_request_get(url)
            reponse_tb_json = response_tb.json()

            for item_tree in reponse_tb_json:
                if item_tree['type'] == "dir":
                    tb_serach(item_tree['url']) 
                else:
                    response_file = cus_request_get(item_tree['url'])
                    file_req.append(response_file)
        tb_serach(treeUrl)
        return file_req
    
    else: # load from api and process locally
        urlLoad = f"{reqUrl}/zipball/{response_json['default_branch']}"
        response = cus_request_get(urlLoad)
        
        byte_content_io = BytesIO(response.content)
        contents = []
        with zipfile.ZipFile(byte_content_io, "r") as zip_ref:
            file_list = zip_ref.namelist()

            for file_name in file_list:
                with zip_ref.open(file_name) as file:
                    try:
                        content_decoded = file.read().decode("utf-8")
                        contents.append({
                            "name" : file_name,
                            "content" : content_decoded,
                            "size" : len(content_decoded),
                        })
                    except UnicodeDecodeError as e:
                        logging.error(f"Failed to decode one, {str(e)}")

    logging.info(f"Done processing get file for {inputs} with elapse time of {time.time() - start_time}")  
    
    return contents 

def get_all_repos(input_name: str, num_load_per_page:int=100):
    '''
    Get all repository from input_name (owner)
    Args: 
        input_name (str): Owner's input_name 
    Returns: 
        list: A list containing corresponding JSON for each repository
    '''
    if isinstance(input_name, str):
        parsed = urlparse(input_name)
        if all([parsed.scheme, parsed.netloc]):
            start_url = f"{input_name}/repos?per_page={num_load_per_page}&page=1"
        else:
            start_url = f"https://api.github.com/users/{input_name}/repos?per_page={num_load_per_page}&page=1"

    elif isinstance(input_name, requests.models.Response) and input_name.url.split("/")[-1] != "repos":
        start_url = f"{input_name.json()['url']}/repos?per_page={num_load_per_page}&page=1"

    elif isinstance(input_name, dict):
        start_url = input_name["repos_url"] 
    else:
        raise ValueError("Unrecognized input_name type")
    curr_page = cus_request_get(start_url)

    if curr_page is None:
        return None 
    
    if "link" in curr_page.headers:
        repo_response = []
        parsed_linkable = parse_link_header(curr_page.headers["link"])
        while "next" in parsed_linkable:
            curr_page = cus_request_get(parsed_linkable["next"])
            if curr_page is None :
                break
            parsed_linkable = parse_link_header(curr_page.headers["link"])
            repo_response.append(curr_page)
    else:
        repo_response = [curr_page]

    return repo_response

#TODO: (perhaps) use deque and save instead of list for handling limit requests
def process_multiple(owner_multiple: list, 
                     num_partition=100,
                     min_fork: int=5,
                     min_star: int=5,
                     updated: str="2020-01",
                     save_dir="all_repository"): ## Deprecated, too many requests are needed to run, use offline processing instead

    start_time = time.time()
    queue_search = queue.Queue()
    parted_owner = partition_list(owner_multiple, num_partition)
    for item in parted_owner: queue_search.put(item)

    def single_seq(repo):
        all_repos = get_all_repos(repo, num_load_per_page=100)
        if all_repos is None : return 
        all_repos_json = flatten([rep.json() for rep in all_repos])

        return all_repos_json
    
    while parted_owner.qsize() != 0:

        owners = parted_owner.get()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_retrievals = list(executor.map(single_seq, owners))
        
        all_retrievals = flatten(all_retrievals)
        all_retrievals = [repo for repo in all_retrievals if repo is not None]
        excluded_retrievals = set()
        print(f"Get a total of {len(all_retrievals)} repositories")

        for rep in all_retrievals:

            if not compare_dates(updated, rep["updated_at"]):
                excluded_retrievals.extend(excluded_retrievals)
                all_retrievals.remove(rep)

            elif rep["forks_count"] < min_fork and rep["allow_forking"]:
                excluded_retrievals.extend(excluded_retrievals)
                all_retrievals.remove(rep)
            
            elif rep["stargazers_count"] < min_star:
                excluded_retrievals.extend(excluded_retrievals)
                all_retrievals.remove(rep)
        
        print(f"Filtered out, left with {len(all_retrievals)} repositories")
        print(f"Finished search repository at {time.time() - start_time}")

        save_json(all_retrievals, save_dir)
        save_json(excluded_retrievals, "excluded_repo")

################################Disc/Rev Strategy ################################

def search_pattern(search_queries: list, language="python", type_search="repositories", max_retry:int=10):
    '''Get multiple corresponding repo at once
    Args:
        search_queries (list) : list of names to search
        type_search (str) : type either "topic", "repository", or "code"

    Returns: 
        tuple: A tuple containing a list of JSON responses for each discoverable and a list of unprocessed search queries as a result of late limit
    '''
    topics = ["topic", "repositories", "code"]
    assert type_search in topics, f"type_search must be in {topics}"
    per_page = 100
    page = 1

    def single_search(search_query):

        url = f"https://api.github.com/search/{type_search}?q={search_query}+language:{language}&sort=stars&order=desc&per_page={per_page}&page={page}"
        curr_page = cus_request_get(url, headers=REQ_HEADER)

        if curr_page is None: return None

        if "link" in curr_page.headers:
            repo_response = []
            parsed_linkable = parse_link_header(curr_page.headers["link"])
            while "next" in parsed_linkable:
                curr_page = cus_request_get(parsed_linkable["next"])
                if curr_page is None :
                    break
                parsed_linkable = parse_link_header(curr_page.headers["link"])
                repo_response.append(curr_page)
        else:
            repo_response = [curr_page]

        return repo_response

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_query = {executor.submit(single_search, query): query for query in search_queries}
        results = []
        unprocessed_queries = []
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"An error occurred while processing {query}: {e}")
                unprocessed_queries.append(query)
            else:
                if result is None:
                    unprocessed_queries.append(query)
                else:
                    results.append(result)
                    
    if len(unprocessed_queries) > 0 and max_retry > 0:
        print(f"Unprocessed queries: {len(unprocessed_queries)}, retrying...")
        m_results, m_unprocessed_queries = search_pattern(unprocessed_queries, language, type_search, max_retry=max_retry-1)
        results.extend(m_results)
        unprocessed_queries = m_unprocessed_queries

    return results, unprocessed_queries

################################Utility################################

def cus_request_get(url: str, 
                    max_retry:int=3, 
                    headers=REQ_HEADER,
                    saved_failed_url:Union[bool, str]=True, 
                    *args, 
                    **kwargs):
    
    global _token_status

    with requests.Session() as session:
        for attempt in range(max_retry):
            try:
                response = session.get(url, headers=headers, **kwargs)

                if "x-ratelimit-remaining" in response.headers:
                    _token_status = int(response.headers["x-ratelimit-remaining"])
                
                if _token_status % 100 == 0: 
                    logging.info(msg=f"Current token limit: {_token_status}")

                if response.status_code == 200:
                    _req_history.update(response)
                    return response

                elif response.status_code == 429:  # rate limit
                    
                    time_restart = datetime.datetime.utcfromtimestamp(int(response.headers["x-ratelimit-reset"]))
                    logging.info(msg=f"429 rate limit error: {response.url}, next reset time: {time_restart}, sleeping...")
                    if "x-ratelimit-remaining" in response.headers:
                        time.sleep(3600)
                        logging.info(msg=f"continue processing from sleeping")
                else:
                    msg_error = f"Failed to retrieve: {response.url} with code {response.status_code} with attempt: {attempt}"
                    logging.error(msg=msg_error)
                    
                    if saved_failed_url and attempt == max_retry - 1:  # save failed attempt
                        _failed_req_history.update({response.url: response.status_code})
                        return None
                
                # Wait before retrying
                time.sleep(2 ** attempt)

            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed due to network error: {e}")
                time.sleep(2 ** attempt)

def parse_link_header(link_header):
    '''Parse with symbol of prev, next, last, first'''
    return {rel.strip('"').replace('rel="', '') : url.strip("<>") for url, rel in (part.split("; ") \
        for part in link_header.split(", "))
    }

def load_from_code_paper():
    load_link = "https://production-media.paperswithcode.com/about/links-between-papers-and-code.json.gz"
    response_decoded = requests.get(load_link)
    return response_decoded
