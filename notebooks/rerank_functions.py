
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel 
from deutschland import klinikatlas
from deutschland.klinikatlas.api import default_api
from deutschland.klinikatlas.model.fileadmin_json_icd_codes_json_get200_response_inner import (
    FileadminJsonIcdCodesJsonGet200ResponseInner as klinikatlas_datatype
)
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from FlagEmbedding import FlagReranker

def preprocess_input_string(input_str:str| list[str])->list[dict]:
    '''
    Takes a string or list of strings and then transforms it such that it can be used in german_medical_embedding. It is meant to be used on the 
    user search and not the data base or any larger set.  

    Args:
    input_str: String or list of strings. If the value is string it is space seperated while if it is a list of string it is seperated like by the list.

    Returns: 
    list_preprocessed_string: List of dictionaries all containing the key 'description' with value being the either the key seperated word or the
    different strings in the list. 
    '''
    input_str = input_str.split() if isinstance(input_str, str) else input_str
        
    list_preprocessed_string = []
    for value in input_str:
        list_preprocessed_string.append({'description':value})
    return list_preprocessed_string

def german_medical_embedding (data: list[klinikatlas_datatype]|list[dict]|str, 
                              bs:int, 
                              tokenizer:XLMRobertaTokenizer, 
                              model:XLMRobertaModel) -> torch.Tensor:
    '''
    Takes a list of strings then apply a pre trained model for medical semantic search to the list of strings returning a list of tensors.


    Args:
    data: String or list of stings being embeded.
    bs: Batch size 
    tokenizer, model: Part of applying the pre trained model to our model. 

    Returns:
    embedding: Embedded list of tensors

    '''
    data = preprocess_input_string(data) if isinstance(data, str) or isinstance(data[0], str) else data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding = []
    for i in tqdm(np.arange(0, len(data), bs)):
        batch = data[i:i+bs]
        descriptions = [item["description"] for item in batch] 
        toks = tokenizer(
            descriptions,
            padding="max_length",
            max_length=40,
            truncation=True,
            return_tensors="pt"
        )

        tokse = {}
        for key,value in toks.items():
            tokse[key] = value.to(device)
        cls_rep = model(**tokse)[0][:,0,:]
        embedding.append(cls_rep.cpu().detach())
    embedding = torch.cat(embedding)
    return (embedding)

def cos_sim(a, b):
    '''
    Takes two lists vectors and then returns the matrix of the angular difference. 

    Args:
    a,b: List of tensor vectors.

    Return: 
    Torch.mm: The matrix given by the angle difference of all values in 'a' to all values in 'b'
    '''
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

# cosine similarity of first entity with all the entities

def max_log(embedded_database: torch.Tensor, embedded_input: torch.Tensor)-> int:
    '''
    From the matrix given by cos_sim take the max of the logs summed of the ones corresponding to the same database embeddin. 

    Args:
    embedded_input: the input already embedded
    embedded_database: the database already embedded

    Return: 
    index: The index of the argmax of log summed angular difference from the input to the database
    '''
    similarity_tensor = cos_sim(embedded_database, embedded_input)
    log = torch.log(similarity_tensor).sum(dim=1)
    log = log.nan_to_num(nan = float('-inf'))
    index= log.argmax().item()
    return(index)

def max_nolog(embedded_database: torch.Tensor, embedded_input: torch.Tensor)-> int:
    '''
    From the matrix given by cos_sim take the sum of the ones corresponding to the same database embedding.

    Args:
    embedded_input: the input already embedded
    embedded_database: the database already embedded

    Return: 
    index: The index of the argmax of summed angular difference from the input to the database
    '''
    similarity_tensor = cos_sim(embedded_database, embedded_input)
    sums = similarity_tensor.sum(dim=1)
    index= sums.argmax().item()
    return(index)

def n_max_log(embedded_database: torch.Tensor, embedded_input: torch.Tensor,amount :int)-> list[int]:
    '''
    From the matrix given by cos_sim take the max of the logs summed of the ones corresponding to the same database embeddin. 

    Args:
    embedded_input: the input already embedded
    embedded_database: the database already embedded
    amount: the amount of indexes for the reranker

    Return: 
    a list of indexes 
    '''
    similarity_tensor = cos_sim(embedded_database, embedded_input)
    log = torch.log(similarity_tensor).sum(dim=1)
    log = log.nan_to_num(nan = float('-inf'))
    indices = torch.topk(log, k=amount).indices.tolist()
    return(indices)

def reranker_after(data: list[klinikatlas_datatype]|list[dict],
             indices:list[int], 
             query:str,
             reranker:FlagReranker,
             bs:int):
    candidates = [data[i] for i in indices]
    pairs = [
        [query, item["description"]]
        for item in candidates
    ]
    scores = reranker.compute_score(
        pairs,
        batch_size=16
    )
    ranked_candidates = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
        )
    return ranked_candidates