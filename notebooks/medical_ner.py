import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, XLMRobertaTokenizer, XLMRobertaModel
    import torch.nn.functional as F


    import marimo as mo
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from deutschland import klinikatlas
    from deutschland.klinikatlas.api import default_api
    from deutschland.klinikatlas.model.fileadmin_json_icd_codes_json_get200_response_inner import (
        FileadminJsonIcdCodesJsonGet200ResponseInner as klinikatlas_datatype
    )

    return (
        AutoModel,
        AutoModelForTokenClassification,
        AutoTokenizer,
        F,
        XLMRobertaModel,
        XLMRobertaTokenizer,
        default_api,
        klinikatlas,
        klinikatlas_datatype,
        mo,
        np,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Named entity recognition of medical keywords

    This notebook is an attempt to extract diagnoses, treatments, and specialties (which we can all use in our search) from arbitrary queries.
    """)
    return


@app.cell
def _():
    queries = [
            "Ich habe seit mehreren Tagen starke Kopfschmerzen "
            "und Schwindel und brauche ein MRT.",
            "Knie-OP Chirurgie in München",
            "Ich habe starke Kopfschmerzen und Schwindel und brauche ein MRT.",
            "Ich habe starkeKopfschmerzen und Schwindel.",
            "Ich habe starke KopfschmerzenundSchwindel.",
            "Ich habe heute morgen die Wäsche aufgehängt"
    ]
    return (queries,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load medical NER ner_modell and implement token processing
    """)
    return


@app.cell
def _():
    NER_MODEL_NAME = "HUMADEX/german_medical_ner"
    SCORE_THRESHOLD = 0.7
    return NER_MODEL_NAME, SCORE_THRESHOLD


@app.cell
def _(torch):
    def get_device():
        """
        Determine the best available computation device.

        Returns
        -------
        torch.device
            CUDA, Apple MPS, or CPU device.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    return (get_device,)


@app.cell
def _(
    AutoModelForTokenClassification,
    AutoTokenizer,
    NER_MODEL_NAME,
    SCORE_THRESHOLD,
    get_device,
    torch,
):
    def load_medical_ner(
        model_name: str = NER_MODEL_NAME,
    ):
        """
        Load the tokenizer and medical NER model.

        Parameters
        ----------
        model_name : str
            Hugging Face model identifier.

        Returns
        -------
        tuple
            Tokenizer, model and computation device.
        """
        device = get_device()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        model = AutoModelForTokenClassification.from_pretrained(
            model_name
        )

        model.to(device)
        model.eval()

        return tokenizer, model, device


    def predict_tokens(
        query: str,
        tokenizer,
        model,
        device,
    ):
        """
        Predict NER labels and character spans for tokens.

        Parameters
        ----------
        query : str
            Input query.
        tokenizer
            Hugging Face tokenizer.
        model
            Token classification model.
        device : torch.device
            Computation device.

        Returns
        -------
        list of dict
            Token predictions with character offsets.
        """
        encoded = tokenizer(
            query,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )

        offsets = encoded.pop("offset_mapping")[0]

        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = model(**model_inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=-1,
        )[0]

        tokens = tokenizer.convert_ids_to_tokens(
            model_inputs["input_ids"][0]
        )

        results = []

        for token, prediction, probability, offset in zip(
            tokens,
            predictions,
            probabilities[0],
            offsets,
        ):
            start, end = offset.tolist()

            if start == end:
                continue

            label = model.config.id2label[
                prediction.item()
            ]

            results.append(
                {
                    "token": token,
                    "label": label,
                    "score": float(
                        probability[prediction]
                    ),
                    "start": start,
                    "end": end,
                }
            )

        return results


    def group_subtokens(
        query: str,
        predictions,
    ):
        """
        Group WordPiece subtokens into complete words.

        Parameters
        ----------
        query : str
            Original query.
        predictions : list of dict
            Token predictions.

        Returns
        -------
        list of dict
            Predictions grouped into words.
        """
        words = []
        current = None

        for prediction in predictions:
            start = prediction["start"]
            end = prediction["end"]

            text = query[start:end]

            if current is None:
                current = {
                    "text": text,
                    "start": start,
                    "end": end,
                    "predictions": [prediction],
                }
                continue

            previous_end = current["end"]
            gap = query[previous_end:start]

            is_subtoken = prediction["token"].startswith("##")

            if is_subtoken or gap == "":
                current["text"] += text
                current["end"] = end
                current["predictions"].append(prediction)
            else:
                words.append(current)

                current = {
                    "text": text,
                    "start": start,
                    "end": end,
                    "predictions": [prediction],
                }

        if current is not None:
            words.append(current)

        return words


    def get_word_label(
        word,
    ):
        """
        Determine the dominant NER label for a grouped word.

        Parameters
        ----------
        word : dict
            Grouped word containing token predictions.

        Returns
        -------
        tuple
            Entity label and confidence score.
        """
        scores = {}

        for prediction in word["predictions"]:
            label = prediction["label"]

            if label == "O":
                continue

            entity_type = label.split("-", 1)[-1]

            scores.setdefault(
                entity_type,
                [],
            )

            scores[entity_type].append(
                prediction["score"]
            )

        if not scores:
            return "O", 0.0

        label = max(
            scores,
            key=lambda key: sum(scores[key]),
        )

        score = sum(scores[label]) / len(
            scores[label]
        )

        return label, score


    def extract_medical_entities(
        query: str,
        tokenizer,
        model,
        device,
        score_threshold: SCORE_THRESHOLD,
    ):
        """
        Extract medical entities above a confidence threshold.

        Parameters
        ----------
        query : str
            German medical query.
        tokenizer
            Hugging Face tokenizer.
        model
            Token classification model.
        device : torch.device
            Computation device.
        score_threshold : float
            Minimum confidence required for an entity.

        Returns
        -------
        list of dict
            Extracted medical entities.
        """
        predictions = predict_tokens(
            query=query,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        words = group_subtokens(
            query=query,
            predictions=predictions,
        )

        entities = []

        for word in words:
            label, score = get_word_label(word)

            if label == "O":
                continue

            if score < score_threshold:
                continue

            entities.append(
                {
                    "text": word["text"],
                    "label": label,
                    "score": score,
                    "start": word["start"],
                    "end": word["end"],
                }
            )

        return entities



    def extract_medical_keywords(
        query: str,
        tokenizer,
        model,
        device,
        score_threshold
    ):
        """
        Extract medical keywords grouped by entity type.

        Parameters
        ----------
        query : str
            German medical query.
        tokenizer
            Hugging Face tokenizer.
        model
            Token classification model.
        device : torch.device
            Computation device.
        score_threshold
            Minimum score for keywords

        Returns
        -------
        dict
            Medical keywords grouped by entity type.
        """
        entities = extract_medical_entities(
            query=query,
            tokenizer=tokenizer,
            model=model,
            device=device,
            score_threshold=score_threshold
        )

        keywords = {
            "PROBLEM": [],
            "TEST": [],
            "TREATMENT": [],
        }

        for entity in entities:
            label = entity["label"]

            if label in keywords:
                keywords[label].append(
                    entity["text"]
                )

        return keywords

    return extract_medical_keywords, load_medical_ner


@app.cell
def _(SCORE_THRESHOLD, extract_medical_keywords, load_medical_ner, queries):
    ner_tokenizer, ner_model, device = load_medical_ner()

    print(f"Using device: {device}")

    keywords_in_queries = []


    for query in queries:
            print()
            print(query)

            keywords = extract_medical_keywords(
                query=query,
                tokenizer=ner_tokenizer,
                model=ner_model,
                device=device,
                score_threshold=SCORE_THRESHOLD
            )

            keywords_in_queries.append(keywords)
    
            print(keywords)
    return device, keywords_in_queries


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mapping extracted keywords to ICD codes, OPS codes and specialties
    """)
    return


@app.cell
def _(default_api, klinikatlas):
    API_URL = "https://bundes-klinik-atlas.de"

    configuration = klinikatlas.Configuration(
        host=API_URL
    )

    with klinikatlas.ApiClient(configuration) as api_client:
        api = default_api.DefaultApi(api_client)

        icd_data = api.fileadmin_json_icd_codes_json_get()
        ops_data = api.fileadmin_json_ops_codes_json_get()
    return icd_data, ops_data


@app.cell
def _(AutoModel, AutoTokenizer, device):
    embedding_tokenizer = AutoTokenizer.from_pretrained("permediq/SapBERT-DE", use_fast=True)
    embedding_model = AutoModel.from_pretrained("permediq/SapBERT-DE").to(device)
    return embedding_model, embedding_tokenizer


@app.cell
def _(XLMRobertaModel, XLMRobertaTokenizer, device, np, torch, tqdm):
    def german_medical_embedding (documents: list[str],
                                  bs:int, 
                                  tokenizer:XLMRobertaTokenizer, 
                                  model:XLMRobertaModel) -> torch.Tensor:
        '''
        Takes a list of strings then apply a pre trained model for medical semantic search to the list of strings returning a list of tensors.


        Args:
        documents: list of stings being embeded.
        bs: Batch size 
        tokenizer, model: Part of applying the pre trained model to our model. 

        Returns:
        embeddings: Embedded list of tensors

        '''

        embeddings =[]

        for i in tqdm(np.arange(0, len(documents), bs)):
            batch = documents[i:i+bs]
            toks = tokenizer(
                batch,
                padding="max_length",
                max_length=40,
                truncation=True,
                return_tensors="pt"
            )

            tokse = {}
            for key,value in toks.items():
                tokse[key] = value.to(device)
            cls_rep = model(**tokse)[0][:,0,:] 
            embeddings.append(cls_rep.cpu().detach())
        embeddings = torch.cat(embeddings)

        return embeddings

    return (german_medical_embedding,)


@app.cell
def _(klinikatlas_datatype):
    def preprocess_codes(data: list[klinikatlas_datatype]|list[dict]|str) -> list[str]:
        '''
        Takes icd or opcs codes retrieved from api

        Args:
        data: icd or opcs codes retrieved from api

        Returns:
        List of strings (icd or ops descriptions)

        '''

        return [item["description"] for item in data]


    return (preprocess_codes,)


@app.cell
def _(icd_data, ops_data, preprocess_codes):
    preprocessed_icd_data = preprocess_codes(icd_data)
    preprocessed_ops_data = preprocess_codes(ops_data)
    return (preprocessed_icd_data,)


@app.cell
def _(
    embedding_model,
    embedding_tokenizer,
    german_medical_embedding,
    preprocessed_icd_data,
):
    bs = 32

    icd_embeddings = german_medical_embedding(preprocessed_icd_data, bs, embedding_tokenizer, embedding_model)
    return (icd_embeddings,)


@app.cell
def _():
    return


@app.cell
def _(F, torch):
    def show_results_for_query(
        problem_embeddings_from_query: torch.Tensor,
        icd_embeddings: torch.Tensor,
        keywords_in_queries: list[dict],
        query_idx: int,
        preprocessed_icd_data: list
    ) -> None:
        """
        Find and print the most similar ICD diagnosis for each problem
        extracted from a query.

        For every problem embedding, cosine similarity is calculated against
        all ICD embeddings. The ICD diagnosis with the highest similarity is
        then printed together with the original problem keyword and similarity
        score.

        Parameters
        ----------
        problem_embeddings_from_query : torch.Tensor
            Embeddings of the individual problems extracted from the query.
            Expected shape: (n_problems, embedding_dim).

        icd_embeddings : torch.Tensor
            Embeddings of all ICD diagnoses.
            Expected shape: (n_icd_codes, embedding_dim).

        keywords_in_queries : list[dict]
            List containing the extracted keywords for each query. Each
            dictionary must contain a "PROBLEM" key with a list of problem
            keywords.

        query_idx : int
            Index of the query whose problem keywords are being evaluated.

        preprocessed_icd_data : list
            List of preprocessed ICD diagnoses corresponding to the rows
            of `icd_embeddings`.

        Returns
        -------
        None
            Prints the keyword, most similar ICD diagnosis, and cosine
            similarity for each problem.
        """

        for idx in range(len(problem_embeddings_from_query)):

            similarities = F.cosine_similarity(
                problem_embeddings_from_query[idx],
                icd_embeddings,
                dim=1
            )

            print(
                "Keyword: "
                + str(keywords_in_queries[query_idx]["PROBLEM"][idx])
            )

            best_idx = torch.argmax(similarities).item()

            most_similar_diagnosis = preprocessed_icd_data[best_idx]

            print(
                "ICD Diagnosis: "
                + str(most_similar_diagnosis)
            )

            best_similarity = similarities[best_idx].item()

            print(
                "Similarity: "
                + str(best_similarity)
            )

    return (show_results_for_query,)


@app.cell
def _(
    embedding_model,
    embedding_tokenizer,
    german_medical_embedding,
    icd_embeddings,
    keywords_in_queries,
    preprocessed_icd_data,
    queries,
    show_results_for_query,
):
    for query_idx in range(len(queries)):
        problem_embeddings_from_query = german_medical_embedding(keywords_in_queries[query_idx]["PROBLEM"], 1, 
                                 embedding_tokenizer, embedding_model)

        print("Results for " + str(queries[query_idx]))
        print()
        if len(problem_embeddings_from_query) > 0:
            show_results_for_query(
        problem_embeddings_from_query=problem_embeddings_from_query,
        icd_embeddings=icd_embeddings,
        keywords_in_queries=keywords_in_queries,
        query_idx=query_idx,
        preprocessed_icd_data=preprocessed_icd_data
    )
    
    return


if __name__ == "__main__":
    app.run()
