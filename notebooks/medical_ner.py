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

    from typing import Any

    return (
        Any,
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

    This notebook is an attempt to extract diagnoses and treatments (which we can all use in our search) from arbitrary queries.
    """)
    return


@app.cell
def _():
    queries = [
        # ============================================================
        # Neurologie / MRT
        # ============================================================
        "Ich habe seit mehreren Tagen starke Kopfschmerzen "
        "und Schwindel und brauche ein MRT.",

        "Ich habe starke Kopfschmerzen und Schwindel und brauche ein MRT.",

        "Ich habe starke Kopfschmerzen und Schwindel.",

        "Ich habe starkeKopfschmerzen und Schwindel.",

        "Ich habe starke KopfschmerzenundSchwindel.",

        "Seit Tagen Kopfschmerzen und Schwindel.",

        "Untersuchung wegen starker Kopfschmerzen.",

        "MRT vom Kopf wegen Kopfschmerzen.",

        "MRT Gehirn",

        "Kernspintomographie Kopf",

        "Neurologie und MRT",

        "Krankenhaus für neurologische Untersuchungen",

        "Ich brauche eine neurologische Untersuchung.",

        "Schwindel und Kopfschmerzen neurologisch abklären lassen.",


        # ============================================================
        # Orthopädie / Knie
        # ============================================================
        "Knie-OP Chirurgie in München",

        "Ich brauche eine Operation am Knie.",

        "Knieoperation",

        "Operation am Kniegelenk",

        "Kniechirurgie in München",

        "Orthopädische Klinik für Knieoperationen",

        "Krankenhaus für Knie-OP",

        "Meniskus Operation",

        "Kreuzband OP",

        "Knieprothese",

        "Kniegelenkersatz",

        "Ich habe starke Knieschmerzen und brauche eine Untersuchung.",


        # ============================================================
        # Geburtshilfe
        # ============================================================
        "Geburtshilfe Nürnberg",

        "Geburtsklinik in München",

        "Krankenhaus für Geburtshilfe",

        "Ich suche ein Krankenhaus für die Geburt.",

        "Geburt im Krankenhaus",

        "Entbindungsklinik",

        "Geburtsstation",

        "Kreißsaal in München",

        "Schwangerschaft und Geburt",

        "Klinik mit Geburtshilfe",


        # ============================================================
        # Kardiologie
        # ============================================================
        "Krankenhaus für Herzkrankheiten",

        "Kardiologie in München",

        "Ich brauche eine Untersuchung meines Herzens.",

        "Herzkatheter Untersuchung",

        "Herzprobleme im Krankenhaus untersuchen lassen",

        "Kardiologische Klinik",

        "Herzklinik in München",

        "Untersuchung wegen Herzrhythmusstörungen",


        # ============================================================
        # Unfallchirurgie / Notaufnahme
        # ============================================================
        "Unfallchirurgie in München",

        "Krankenhaus nach einem Unfall",

        "Ich habe mich beim Fahrradfahren verletzt.",

        "Ich bin gestürzt und brauche eine Untersuchung.",

        "Notaufnahme in München",

        "Krankenhaus für Knochenbrüche",

        "Fraktur behandeln lassen",

        "Behandlung eines gebrochenen Arms",

        "Unfallchirurgie und Orthopädie",


        # ============================================================
        # Innere Medizin
        # ============================================================
        "Krankenhaus für Innere Medizin",

        "Internistische Klinik in München",

        "Ich brauche eine internistische Untersuchung.",

        "Behandlung von Magenproblemen im Krankenhaus",

        "Krankenhaus für Bauchschmerzen",

        "Untersuchung wegen starken Bauchschmerzen",

        "Gastroenterologie in München",

        "Magen-Darm Untersuchung",


        # ============================================================
        # Chirurgie allgemein
        # ============================================================
        "Chirurgische Klinik in München",

        "Ich brauche eine Operation.",

        "Krankenhaus für chirurgische Eingriffe",

        "Allgemeinchirurgie",

        "Chirurgie Krankenhaus München",

        "Operation im Krankenhaus",


        # ============================================================
        # Onkologie
        # ============================================================
        "Krankenhaus für Krebsbehandlung",

        "Onkologie in München",

        "Krebsbehandlung im Krankenhaus",

        "Tumorzentrum München",

        "Klinik für Onkologie",

        "Chemotherapie Krankenhaus",


        # ============================================================
        # Augen
        # ============================================================
        "Augenklinik in München",

        "Krankenhaus für Augenoperationen",

        "Augenoperation",

        "Grauer Star Operation",

        "Netzhautoperation",

        "Augenheilkunde Krankenhaus",


        # ============================================================
        # HNO
        # ============================================================
        "HNO Klinik in München",

        "Krankenhaus für HNO",

        "Hals Nasen Ohren Klinik",

        "Operation an der Nase",

        "Nasennebenhöhlen Operation",

        "HNO Untersuchung",


        # ============================================================
        # Urologie
        # ============================================================
        "Urologie in München",

        "Krankenhaus für urologische Erkrankungen",

        "Nierensteine behandeln lassen",

        "Urologische Klinik",

        "Operation an der Niere",

        "Urologische Untersuchung",


        # ============================================================
        # Psychiatrie / Psychosomatik
        # ============================================================
        "Psychiatrische Klinik in München",

        "Krankenhaus für psychische Erkrankungen",

        "Psychiatrische Behandlung",

        "Psychosomatische Klinik",

        "Stationäre psychiatrische Behandlung",

        "Klinik für Psychiatrie",


        # ============================================================
        # Verschiedene Schreibweisen / Suchmaschinen-artig
        # ============================================================
        "MRT München",

        "MRT Kopf München",

        "Neurologie München",

        "Knie OP München",

        "Kniechirurgie München",

        "Geburtshilfe München",

        "Herzklinik München",

        "Notaufnahme München",

        "Unfallchirurgie München",

        "Gastroenterologie München",


        # ============================================================
        # Bewusst irrelevante Queries
        # ============================================================
        "Ich habe heute morgen die Wäsche aufgehängt",

        "Wie wird das Wetter morgen?",

        "Ich möchte morgen einkaufen gehen.",

        "Was kann ich heute Abend kochen?",

        "Ich suche ein gutes Restaurant in München.",

        "Wie komme ich zum Hauptbahnhof?",

        "Mein Fahrrad hat einen platten Reifen.",

        "Ich möchte einen neuen Laptop kaufen.",
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
    SCORE_THRESHOLD = 0.5
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
    return preprocessed_icd_data, preprocessed_ops_data


@app.cell
def _(
    embedding_model,
    embedding_tokenizer,
    german_medical_embedding,
    preprocessed_icd_data,
):
    bs = 32

    icd_embeddings = german_medical_embedding(preprocessed_icd_data, bs, embedding_tokenizer, embedding_model)
    return bs, icd_embeddings


@app.cell
def _(
    bs,
    embedding_model,
    embedding_tokenizer,
    german_medical_embedding,
    preprocessed_ops_data,
):
    ops_embeddings = german_medical_embedding(preprocessed_ops_data, bs, embedding_tokenizer, embedding_model)
    return (ops_embeddings,)


@app.cell
def _(Any, F, torch):
    def find_similar_codes(
        keywords: list[str],
        query_embeddings: torch.Tensor,
        code_embeddings: torch.Tensor,
        preprocessed_codes: list[str],
        code_type: str,
        similarity_score_threshold: float,
    ) -> list[dict[str, Any]]:
        """
        Find the most similar code for each keyword and keep only
        results above the similarity score threshold.

        Parameters
        ----------
        keywords : list[str]
            Keywords extracted from the query.

        query_embeddings : torch.Tensor
            Embeddings of the keywords.
            Shape: (n_keywords, embedding_dim).

        code_embeddings : torch.Tensor
            Embeddings of all ICD or OPS codes.
            Shape: (n_codes, embedding_dim).

        preprocessed_codes : list[str]
            Preprocessed descriptions corresponding to `code_embeddings`.

        code_type : str
            Type of the keyword. Expected values are "diagnosis",
            "treatment", or "test".

        similarity_score_threshold : float
            Minimum cosine similarity required for a result to be included.

        Returns
        -------
        list[dict[str, Any]]
            List containing the keyword, most similar code, code type,
            and similarity score for all matches above the threshold.
        """

        results = []

        for keyword, query_embedding in zip(keywords, query_embeddings):
            similarities = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                code_embeddings,
                dim=1,
            )

            best_idx = torch.argmax(similarities).item()
            best_similarity = similarities[best_idx].item()

            if best_similarity >= similarity_score_threshold:
                results.append(
                    {
                        "keyword": keyword,
                        "code": preprocessed_codes[best_idx],
                        "type": code_type,
                        "similarity_score": best_similarity,
                    }
                )

        return results

    return (find_similar_codes,)


@app.cell
def _(Any, find_similar_codes, german_medical_embedding, torch):
    def find_codes_for_queries(
        queries: list[str],
        keywords_in_queries: list[dict[str, list[str]]],
        embedding_tokenizer,
        embedding_model,
        icd_embeddings: torch.Tensor,
        ops_embeddings: torch.Tensor,
        preprocessed_icd_data: list[str],
        preprocessed_ops_data: list[str],
        similarity_score_threshold: float,
    ) -> list[dict[str, Any]]:
        """
        Find ICD and OPS codes matching keywords from multiple queries.

        For each query, embeddings are generated for diagnosis, treatment,
        and test keywords. Only the best matching ICD or OPS code is returned
        when its cosine similarity is above the specified threshold.

        Parameters
        ----------
        queries : list[str]
            Original queries.

        keywords_in_queries : list[dict[str, list[str]]]
            Extracted keywords for each query. Each dictionary should contain
            "PROBLEM", "TREATMENT", and "TEST".

        embedding_tokenizer
            Tokenizer used by the medical embedding model.

        embedding_model
            Medical embedding model.

        icd_embeddings : torch.Tensor
            Embeddings of all ICD codes.

        ops_embeddings : torch.Tensor
            Embeddings of all OPS codes.

        preprocessed_icd_data : list[str]
            ICD code descriptions corresponding to `icd_embeddings`.

        preprocessed_ops_data : list[str]
            OPS code descriptions corresponding to `ops_embeddings`.

        similarity_score_threshold : float
            Minimum cosine similarity required for a result to be included.

        Returns
        -------
        list[dict[str, Any]]
            Results containing the query, keyword, matched code, type,
            and similarity score.
        """

        results = []

        keyword_config = {
            "PROBLEM": {
                "result_type": "diagnosis",
                "embeddings": icd_embeddings,
                "preprocessed_data": preprocessed_icd_data,
            },
            "TREATMENT": {
                "result_type": "treatment",
                "embeddings": ops_embeddings,
                "preprocessed_data": preprocessed_ops_data,
            },
            "TEST": {
                "result_type": "test",
                "embeddings": ops_embeddings,
                "preprocessed_data": preprocessed_ops_data,
            },
        }

        for query_idx, query in enumerate(queries):
            query_keywords = keywords_in_queries[query_idx]

            for keyword_type, config in keyword_config.items():
                keywords = query_keywords.get(keyword_type, [])

                if not keywords:
                    continue

                query_embeddings = german_medical_embedding(
                    keywords,
                    bs=1,
                    tokenizer=embedding_tokenizer,
                    model=embedding_model,
                )

                matches = find_similar_codes(
                    keywords=keywords,
                    query_embeddings=query_embeddings,
                    code_embeddings=config["embeddings"],
                    preprocessed_codes=config["preprocessed_data"],
                    code_type=config["result_type"],
                    similarity_score_threshold=similarity_score_threshold,
                )

                for match in matches:
                    results.append(
                        {
                            "query": query,
                            **match,
                        }
                    )

        return results

    return (find_codes_for_queries,)


@app.cell
def _(
    embedding_model,
    embedding_tokenizer,
    find_codes_for_queries,
    icd_embeddings,
    keywords_in_queries,
    ops_embeddings,
    preprocessed_icd_data,
    preprocessed_ops_data,
    queries,
):
    results = find_codes_for_queries(
        queries=queries,
        keywords_in_queries=keywords_in_queries,
        embedding_tokenizer=embedding_tokenizer,
        embedding_model=embedding_model,
        icd_embeddings=icd_embeddings,
        ops_embeddings=ops_embeddings,
        preprocessed_icd_data=preprocessed_icd_data,
        preprocessed_ops_data=preprocessed_ops_data,
        similarity_score_threshold=0.7,
    )
    return (results,)


@app.cell
def _(results):
    results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    All keywords (for the moment regardless of they are of category PROBLEM, TEST or TREATMENT) are compared with the embeddings of the ICD and OPS codes. They are only taken into account if they have a similarity score of at least 0.7 with one of these embeddings.

    This filters out irrelevant keywords (like Wäsche) quite well. On the other hand, "Geburthilfe" is not matched to any ops code. This could be because OPS codes are very specific (e.g "Spontane und vaginale operative Entbindung bei Beckenendlage: Spontane Entbindung ohne Komplikationen" instead of "Entbindung") and it could be that "Geburtshilfe" would match better with a specialty which are going to be added later.
    """)
    return


if __name__ == "__main__":
    app.run()
