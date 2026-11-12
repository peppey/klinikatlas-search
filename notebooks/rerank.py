import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib
    import rerank_functions as rrf
    import marimo as mo
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

    return (
        AutoModel,
        AutoTokenizer,
        FlagReranker,
        default_api,
        importlib,
        klinikatlas,
        mo,
        np,
        rrf,
        torch,
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


@app.cell
def _(icd_data, model, rrf, tokenizer):
    bs = 32
    all_embs = rrf.german_medical_embedding(icd_data, bs, tokenizer,model)
    return all_embs, bs


@app.cell
def _(all_embs, np, rrf):
    testing = rrf.cos_sim(all_embs[0].unsqueeze(0), all_embs)
    print(testing)
    print(np.argmax(testing))
    return


@app.cell
def _(all_embs, bs, icd_data, model, rrf, tokenizer):
    x = 'Typhus'
    y = rrf.preprocess_input_string(x)
    z = ['Sonstige', 'näher', 'bezeichnete', 'Karzinome', 'der', 'Leber']
    Z = ['Sonstige näher bezeichnete Karzinome der Leber']
    print(x)
    this_embed = rrf.german_medical_embedding(x, bs, tokenizer, model)
    _k = rrf.max_log(all_embs, this_embed)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    _k = rrf.max_nolog(all_embs, this_embed)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    return Z, x, z


@app.cell
def _(all_embs, bs, icd_data, model, rrf, tokenizer, z):
    print(z)
    this_embed_2 = rrf.german_medical_embedding(z, bs, tokenizer, model)
    _k = rrf.max_log(all_embs, this_embed_2)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    _k = rrf.max_nolog(all_embs, this_embed_2)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    return


@app.cell
def _(Z, all_embs, bs, icd_data, model, rrf, tokenizer):
    print(Z)
    this_embed_3 = rrf.german_medical_embedding(Z, bs, tokenizer, model)
    _k = rrf.max_log(all_embs, this_embed_3)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    _k = rrf.max_nolog(all_embs, this_embed_3)
    print(_k)
    print(icd_data[_k]['description'])
    print(icd_data[_k]['icdcode'])
    return (this_embed_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now starts the part relating to the reranker
    """)
    return


@app.cell
def _(all_embs, icd_data, importlib, rrf, this_embed_3):
    importlib.reload(rrf)
    _k = rrf.n_max_log(all_embs, this_embed_3,10)
    print(_k)
    for i in range(0,10):
        print(icd_data[_k[i]])
    
    return


@app.cell
def _(FlagReranker):
    reranker = FlagReranker(
        'BAAI/bge-reranker-v2-m3',
        use_fp16=False
    )
    return (reranker,)


@app.cell
def _(all_embs, bs, icd_data, importlib, reranker, rrf, this_embed_3, x):
    importlib.reload(rrf)
    _k = rrf.n_max_log(all_embs, this_embed_3,200)
    reranked_result = rrf.reranker_after(icd_data, _k, x, reranker, bs)
    print(reranked_result[0][0])

    for j in range(0,10):
        print('Icd:', reranked_result[j][0], '      Score:', reranked_result[j][1])

    return


if __name__ == "__main__":
    app.run()
