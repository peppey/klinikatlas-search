import json
import re
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from klinikatlas_search.medical_ner import get_device
from klinikatlas_search.model_loading import MODEL_LOAD_LOCK

EMBEDDING_MODEL_NAME = "permediq/SapBERT-DE"
MAX_LENGTH = 40
BATCH_SIZE = 32
MINIMUM_SCORE = 0.7
SEED = 42

DATA_DIR = Path(__file__).parent / "data"
CODE_TYPES = ("icd", "ops")

# ICD category range such as "J09-J11"; OPS codes contain a hyphen themselves and are only matched by prefix.
_ICD_RANGE = re.compile(r"[A-Z]\d\d-[A-Z]\d\d")


def _load_embedding_model():
    with MODEL_LOAD_LOCK:
        return _load_embedding_model_once()


@lru_cache(maxsize=1)
def _load_embedding_model_once():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model, device


def embed(texts: list[str], batch_size: int = BATCH_SIZE, progress: bool = False) -> torch.Tensor:
    """Embed texts with the CLS vector of SapBERT-DE. Returns a float32 CPU tensor (n, dim)."""
    tokenizer, model, device = _load_embedding_model()

    batches = range(0, len(texts), batch_size)
    embeddings = []

    for i in tqdm(batches, disable=not progress):
        tokens = tokenizer(
            texts[i : i + batch_size],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            embeddings.append(model(**tokens)[0][:, 0, :].float().cpu())

    return torch.cat(embeddings)


def _descriptions_path(code_type: str) -> Path:
    return DATA_DIR / f"{code_type}_codes.json"


def _embeddings_path(code_type: str) -> Path:
    return DATA_DIR / f"{code_type}_embeddings.pt"


def _umbrella_terms_path(code_type: str) -> Path:
    return DATA_DIR / f"{code_type}_umbrella_terms.json"


def _umbrella_embeddings_path(code_type: str) -> Path:
    return DATA_DIR / f"{code_type}_umbrella_embeddings.pt"


def _load_entries(code_type: str) -> list[dict]:
    """Return [{"code": str | None, "description": str}]; plain string items have no code."""
    with open(_descriptions_path(code_type), encoding="utf-8") as f:
        items = json.load(f)

    return [
        {"code": None, "description": item}
        if isinstance(item, str)
        else {"code": item[f"{code_type}code"], "description": item["description"]}
        for item in items
    ]


def _load_descriptions(code_type: str) -> list[str]:
    return [entry["description"] for entry in _load_entries(code_type)]


def build_embeddings() -> None:
    """Embed all ICD/OPS descriptions and umbrella terms and store them in data/ (float16 to keep files small)."""
    torch.manual_seed(SEED)

    for code_type in CODE_TYPES:
        embeddings = embed(_load_descriptions(code_type), progress=True)
        torch.save(embeddings.half(), _embeddings_path(code_type))
        build_umbrella_embeddings(code_type)


@lru_cache(maxsize=1)
def _load_index() -> dict[str, tuple[list[dict], torch.Tensor]]:
    """Return {code_type: (entries, L2-normalized embeddings)}; builds missing embedding files."""
    if not all(_embeddings_path(code_type).exists() for code_type in CODE_TYPES):
        build_embeddings()

    index = {}

    for code_type in CODE_TYPES:
        entries = _load_entries(code_type)
        embeddings = torch.load(_embeddings_path(code_type), weights_only=True).float()

        if len(entries) != len(embeddings):
            raise ValueError(
                f"{_embeddings_path(code_type).name} does not match {_descriptions_path(code_type).name}; "
                "delete it to rebuild."
            )

        index[code_type] = (entries, F.normalize(embeddings, dim=1))

    return index


def _load_umbrella_groups(code_type: str) -> list[dict]:
    with open(_umbrella_terms_path(code_type), encoding="utf-8") as f:
        return json.load(f)


def _umbrella_terms(groups: list[dict]) -> list[str]:
    """Unique terms in order of first appearance."""
    return list(dict.fromkeys(term for group in groups for term in group["terms"]))


def build_umbrella_embeddings(code_type: str) -> None:
    """Embed all umbrella terms together with the term list they belong to."""
    torch.manual_seed(SEED)

    terms = _umbrella_terms(_load_umbrella_groups(code_type))
    torch.save(
        {"terms": terms, "embeddings": embed(terms, progress=True).half()},
        _umbrella_embeddings_path(code_type),
    )


def _matcher(patterns: list[str]):
    """
    Return a function telling whether a code matches any pattern.

    A pattern is a code prefix ("J45", "I80.1", "5-812") or, for ICD, an inclusive range of
    3-character categories ("J09-J11").
    """
    ranges = [pattern.split("-") for pattern in patterns if _ICD_RANGE.fullmatch(pattern)]
    prefixes = tuple(pattern for pattern in patterns if not _ICD_RANGE.fullmatch(pattern))

    def matches(code: str) -> bool:
        return code.startswith(prefixes) or any(first <= code[:3] <= last for first, last in ranges)

    return matches


def _code_matches(code: str, pattern: str) -> bool:
    return _matcher([pattern])(code)


@lru_cache(maxsize=None)
def _load_umbrella_index(code_type: str) -> dict:
    """
    Return the umbrella terms with their embeddings and codes.

    {"terms": [str], "embeddings": normalized tensor, "term_groups": [[group index]], "group_codes": [[entry]]}
    Rebuilds the stored embeddings when the term file changed.
    """
    groups = _load_umbrella_groups(code_type)
    terms = _umbrella_terms(groups)
    embeddings_path = _umbrella_embeddings_path(code_type)

    stored = torch.load(embeddings_path, weights_only=True) if embeddings_path.exists() else None

    if stored is None or stored["terms"] != terms:
        build_umbrella_embeddings(code_type)
        stored = torch.load(embeddings_path, weights_only=True)

    entries = [entry for entry in _load_entries(code_type) if entry["code"]]
    term_position = {term: i for i, term in enumerate(terms)}
    term_groups = [[] for _ in terms]
    group_codes = []

    for group_index, group in enumerate(groups):
        for term in group["terms"]:
            term_groups[term_position[term]].append(group_index)

        matches = _matcher(group[code_type])
        group_codes.append([entry for entry in entries if matches(entry["code"])])

    return {
        "terms": terms,
        "embeddings": F.normalize(stored["embeddings"].float(), dim=1),
        "term_groups": term_groups,
        "group_codes": group_codes,
    }


def retrieve_by_umbrella_terms(
    keywords: list[str],
    code_type: str,
    top_k: int = 3,
    minimum_score: float = MINIMUM_SCORE,
) -> dict[str, list[dict]]:
    """
    Match keywords against everyday umbrella terms and return all codes below them.

    Parameters
    ----------
    keywords : list[str]
        Keywords, e.g. from `medical_ner.get_medical_keywords`.
    code_type : str
        "icd" (umbrella terms like "Grippe") or "ops" (umbrella terms like "Kniegelenkersatz").
    top_k : int
        Maximum number of umbrella groups per keyword.
    minimum_score : float
        Minimum cosine similarity between keyword and umbrella term.

    Returns
    -------
    dict
        {keyword: [{"umbrella_term": str, "score": float, "<code_type>_codes": [{"code": str, "description": str}]}]}
        sorted by descending score; a keyword without results maps to [].
    """
    if not keywords:
        return {}

    index = _load_umbrella_index(code_type)
    scores = F.normalize(embed(keywords), dim=1) @ index["embeddings"].T

    results = {}

    for keyword, keyword_scores in zip(keywords, scores):
        matches = []
        seen_groups = set()

        for term_index in keyword_scores.argsort(descending=True).tolist():
            score = keyword_scores[term_index].item()

            if score < minimum_score or len(matches) >= top_k:
                break

            for group_index in index["term_groups"][term_index]:
                if group_index in seen_groups or len(matches) >= top_k:
                    continue

                seen_groups.add(group_index)
                matches.append(
                    {
                        "umbrella_term": index["terms"][term_index],
                        "score": score,
                        f"{code_type}_codes": index["group_codes"][group_index],
                    }
                )

        results[keyword] = matches

    return results


def retrieve_icd_by_umbrella_terms(
    keywords: list[str],
    top_k: int = 3,
    minimum_score: float = MINIMUM_SCORE,
) -> dict[str, list[dict]]:
    """ICD codes below the best matching umbrella terms; see `retrieve_by_umbrella_terms`."""
    return retrieve_by_umbrella_terms(keywords, "icd", top_k, minimum_score)


def retrieve_ops_by_umbrella_terms(
    keywords: list[str],
    top_k: int = 3,
    minimum_score: float = MINIMUM_SCORE,
) -> dict[str, list[dict]]:
    """OPS codes below the best matching umbrella terms; see `retrieve_by_umbrella_terms`."""
    return retrieve_by_umbrella_terms(keywords, "ops", top_k, minimum_score)


def retrieve_similar_results(
    keywords: list[str],
    top_k: int = 10,
    minimum_score: float = MINIMUM_SCORE,
) -> dict[str, list[dict]]:
    """
    Find the most similar ICD and OPS descriptions for each keyword.

    Parameters
    ----------
    keywords : list[str]
        Keywords, e.g. from `medical_ner.get_medical_keywords`.
    top_k : int
        Number of results per keyword (ICD and OPS ranked together).
    minimum_score : float
        Minimum cosine similarity for a result to be returned.

    Returns
    -------
    dict
        {keyword: [{"description": str, "code": str | None, "type": "icd" | "ops", "score": cosine similarity}, ...]}
        sorted by descending score; a keyword without results maps to [].
    """
    if not keywords:
        return {}

    index = _load_index()
    keyword_embeddings = F.normalize(embed(keywords), dim=1)

    scores = torch.cat(
        [keyword_embeddings @ index[code_type][1].T for code_type in CODE_TYPES],
        dim=1,
    )
    entries = [(code_type, entry) for code_type in CODE_TYPES for entry in index[code_type][0]]

    top_scores, top_indices = scores.topk(min(top_k, scores.shape[1]), dim=1)

    return {
        keyword: [
            {
                "description": entries[i][1]["description"],
                "code": entries[i][1]["code"],
                "type": entries[i][0],
                "score": score,
            }
            for score, i in zip(row_scores.tolist(), row_indices.tolist())
            if score >= minimum_score
        ]
        for keyword, row_scores, row_indices in zip(keywords, top_scores, top_indices)
    }


if __name__ == "__main__":
    build_embeddings()
