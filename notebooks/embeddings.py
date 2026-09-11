import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
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

    return (
        AutoModel,
        AutoTokenizer,
        XLMRobertaModel,
        XLMRobertaTokenizer,
        default_api,
        klinikatlas,
        klinikatlas_datatype,
        np,
        torch,
        tqdm,
    )


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    return (device,)


@app.cell
def _(default_api, klinikatlas):
    BASE_URL = "https://bundes-klinik-atlas.de"

    configuration = klinikatlas.Configuration(
        host=BASE_URL
    )

    with klinikatlas.ApiClient(configuration) as api_client:
        api = default_api.DefaultApi(api_client)

        icd_data = api.fileadmin_json_icd_codes_json_get()
    return (icd_data,)


@app.cell
def _(AutoModel, AutoTokenizer, device):
    tokenizer = AutoTokenizer.from_pretrained("permediq/SapBERT-DE", use_fast=True)
    model = AutoModel.from_pretrained("permediq/SapBERT-DE").to(device)
    return model, tokenizer


@app.function
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


@app.cell
def _(
    XLMRobertaModel,
    XLMRobertaTokenizer,
    klinikatlas_datatype,
    np,
    torch,
    tqdm,
):
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
        embedding =[]
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
        return embedding

    return (german_medical_embedding,)


@app.cell
def _(torch):
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
    return (cos_sim,)


@app.cell
def _(cos_sim, torch):
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

    return (max_log,)


@app.cell
def _(cos_sim, torch):
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

    return (max_nolog,)


@app.cell
def _(german_medical_embedding, icd_data, model, tokenizer):
    bs = 32
    all_embs = german_medical_embedding(icd_data, bs, tokenizer,model)
    return all_embs, bs


@app.cell
def _(all_embs, cos_sim, np):
    testing = cos_sim(all_embs[0].unsqueeze(0), all_embs)
    print(testing)
    print(np.argmax(testing))
    return


@app.cell
def _(
    all_embs,
    bs,
    german_medical_embedding,
    icd_data,
    max_log,
    max_nolog,
    model,
    tokenizer,
):
    x = 'Typhus'
    y = preprocess_input_string(x)
    z = ['Sonstige', 'näher', 'bezeichnete', 'Karzinome', 'der', 'Leber']
    Z = ['Sonstige näher bezeichnete Karzinome der Leber']
    print(x)
    this_embed = german_medical_embedding(x, bs, tokenizer, model)
    _k = max_log(all_embs, this_embed)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    _k = max_nolog(all_embs, this_embed)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    return Z, y, z


@app.cell
def _(
    all_embs,
    bs,
    german_medical_embedding,
    icd_data,
    max_log,
    max_nolog,
    model,
    tokenizer,
    y,
):
    print(y)
    this_embed_1 = german_medical_embedding(y, bs, tokenizer, model)
    _k = max_log(all_embs, this_embed_1)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    _k = max_nolog(all_embs, this_embed_1)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    return


@app.cell
def _(
    all_embs,
    bs,
    german_medical_embedding,
    icd_data,
    max_log,
    max_nolog,
    model,
    tokenizer,
    z,
):
    print(z)
    this_embed_2 = german_medical_embedding(z, bs, tokenizer, model)
    _k = max_log(all_embs, this_embed_2)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    _k = max_nolog(all_embs, this_embed_2)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    return


@app.cell
def _(
    Z,
    all_embs,
    bs,
    german_medical_embedding,
    icd_data,
    max_log,
    max_nolog,
    model,
    tokenizer,
):
    print(Z)
    this_embed_3 = german_medical_embedding(Z, bs, tokenizer, model)
    _k = max_log(all_embs, this_embed_3)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    _k = max_nolog(all_embs, this_embed_3)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    return


if __name__ == "__main__":
    app.run()
