from functools import lru_cache
from string import punctuation

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from klinikatlas_search.model_loading import MODEL_LOAD_LOCK

NER_MODEL_NAME = "HUMADEX/german_medical_ner"
SCORE_THRESHOLD = 0.5
KEYWORD_LABELS = ("PROBLEM", "TEST", "TREATMENT")


def get_device() -> torch.device:
    """Return the best available device: CUDA, Apple MPS or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_medical_ner(model_name: str = NER_MODEL_NAME):
    """Load tokenizer, model and device once per process (thread-safe)."""
    with MODEL_LOAD_LOCK:
        return _load_medical_ner(model_name)


@lru_cache(maxsize=1)
def _load_medical_ner(model_name: str):
    """
    Load tokenizer and medical NER model once per process.

    Returns
    -------
    tuple
        Tokenizer, model and computation device.
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_tokens(query: str, tokenizer, model, device) -> list[dict]:
    """
    Predict NER labels and character spans for each token.

    Returns
    -------
    list of dict
        Token predictions with label, score and character offsets.
    """
    encoded = tokenizer(
        query,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )

    offsets = encoded.pop("offset_mapping")[0]

    model_inputs = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**model_inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)
    predictions = torch.argmax(outputs.logits, dim=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(model_inputs["input_ids"][0])

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

        results.append(
            {
                "token": token,
                "label": model.config.id2label[prediction.item()],
                "score": float(probability[prediction]),
                "start": start,
                "end": end,
            }
        )

    return results


def group_subtokens(query: str, predictions: list[dict]) -> list[dict]:
    """Group WordPiece subtokens into complete words."""
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

        gap = query[current["end"]:start]
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


def get_word_label(word: dict) -> tuple[str, float]:
    """Return the dominant entity label of a grouped word and its mean score."""
    scores: dict[str, list[float]] = {}

    for prediction in word["predictions"]:
        label = prediction["label"]

        if label == "O":
            continue

        entity_type = label.split("-", 1)[-1]
        scores.setdefault(entity_type, []).append(prediction["score"])

    if not scores:
        return "O", 0.0

    label = max(scores, key=lambda key: sum(scores[key]))

    return label, sum(scores[label]) / len(scores[label])


def merge_adjacent_entities(query: str, entities: list[dict]) -> list[dict]:
    """Merge neighbouring entities of the same type that are only separated by whitespace."""
    merged: list[dict] = []

    for entity in entities:
        previous = merged[-1] if merged else None

        if (
            previous
            and previous["label"] == entity["label"]
            and previous["text"][-1].isalnum()
            and not query[previous["end"] : entity["start"]].strip()
        ):
            count = previous.get("count", 1)
            previous["score"] = (previous["score"] * count + entity["score"]) / (count + 1)
            previous["count"] = count + 1
            previous["end"] = entity["end"]
            previous["text"] = query[previous["start"] : entity["end"]]
        else:
            merged.append(dict(entity))

    for entity in merged:
        entity.pop("count", None)

    return merged


def extract_medical_entities(
    query: str,
    tokenizer,
    model,
    device,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict]:
    """Extract entities whose score is at least `score_threshold`; neighbouring words of one type form one entity."""
    predictions = predict_tokens(query, tokenizer, model, device)
    words = group_subtokens(query, predictions)

    entities = []

    for word in words:
        label, score = get_word_label(word)

        if label == "O" or score < score_threshold:
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

    return merge_adjacent_entities(query, entities)


def get_medical_keywords(
    text: str,
    minimum_score: float = SCORE_THRESHOLD,
) -> list[dict]:
    """
    Extract medical keywords from a query.

    Parameters
    ----------
    text : str
        German query, e.g. "Knie-OP wegen Kreuzbandriss".
    minimum_score : float
        Minimum NER confidence for a keyword.

    Returns
    -------
    list of dict
        [{"text": keyword, "label": "PROBLEM" | "TEST" | "TREATMENT", "score": float}]
        in order of appearance, without leading or trailing punctuation.
    """
    tokenizer, model, device = load_medical_ner()

    entities = extract_medical_entities(
        text,
        tokenizer,
        model,
        device,
        minimum_score,
    )

    keywords = []

    for entity in entities:
        keyword = entity["text"].strip(punctuation + "„“”‚‘’…–—«»")

        if entity["label"] in KEYWORD_LABELS and keyword:
            keywords.append(
                {"text": keyword, "label": entity["label"], "score": entity["score"]}
            )

    return keywords
