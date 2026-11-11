import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from klinikatlas_search.medical_ner import get_device

EMBEDDING_MODEL_NAME = "permediq/SapBERT-DE"
MAX_LENGTH = 40
BATCH_SIZE = 32
MINIMUM_SCORE = 0.7
SEED = 42

DATA_DIR = Path(__file__).parent / "data"
CODE_TYPES = ("icd", "ops")
UMBRELLA_TERMS_PATH = DATA_DIR / "icd_umbrella_terms.json"
UMBRELLA_EMBEDDINGS_PATH = DATA_DIR / "icd_umbrella_embeddings.pt"


@lru_cache(maxsize=1)
def _load_embedding_model():
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


def _load_entries(code_type: str) -> list[dict]:
    """Return [{"code": str | None, "description": str}]; OPS data has no code numbers."""
    with open(_descriptions_path(code_type), encoding="utf-8") as f:
        items = json.load(f)

    return [
        {"code": None, "description": item}
        if isinstance(item, str)
        else {"code": item["icdcode"], "description": item["description"]}
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

    build_umbrella_embeddings()


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


def _load_umbrella_groups() -> list[dict]:
    with open(UMBRELLA_TERMS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _umbrella_terms(groups: list[dict]) -> list[str]:
    """Unique terms in order of first appearance."""
    return list(dict.fromkeys(term for group in groups for term in group["terms"]))


def build_umbrella_embeddings() -> None:
    """Embed all umbrella terms together with the term list they belong to."""
    torch.manual_seed(SEED)

    terms = _umbrella_terms(_load_umbrella_groups())
    torch.save(
        {"terms": terms, "embeddings": embed(terms, progress=True).half()},
        UMBRELLA_EMBEDDINGS_PATH,
    )


def _code_matches(code: str, pattern: str) -> bool:
    """A pattern is a code prefix ("J45", "I80.1") or an inclusive range of 3-character categories ("J09-J11")."""
    if "-" in pattern:
        first, last = pattern.split("-")
        return first <= code[:3] <= last

    return code.startswith(pattern)


@lru_cache(maxsize=1)
def _load_umbrella_index() -> dict:
    """
    Return the umbrella terms with their embeddings and ICD codes.

    {"terms": [str], "embeddings": normalized tensor, "term_groups": [[group index]], "group_codes": [[entry]]}
    Rebuilds the stored embeddings when the term file changed.
    """
    groups = _load_umbrella_groups()
    terms = _umbrella_terms(groups)

    stored = torch.load(UMBRELLA_EMBEDDINGS_PATH, weights_only=True) if UMBRELLA_EMBEDDINGS_PATH.exists() else None

    if stored is None or stored["terms"] != terms:
        build_umbrella_embeddings()
        stored = torch.load(UMBRELLA_EMBEDDINGS_PATH, weights_only=True)

    icd_entries = _load_entries("icd")
    term_position = {term: i for i, term in enumerate(terms)}
    term_groups = [[] for _ in terms]
    group_codes = []

    for group_index, group in enumerate(groups):
        for term in group["terms"]:
            term_groups[term_position[term]].append(group_index)

        group_codes.append(
            [
                entry
                for entry in icd_entries
                if any(_code_matches(entry["code"], pattern) for pattern in group["icd"])
            ]
        )

    return {
        "terms": terms,
        "embeddings": F.normalize(stored["embeddings"].float(), dim=1),
        "term_groups": term_groups,
        "group_codes": group_codes,
    }


def retrieve_icd_by_umbrella_terms(
    keywords: list[str],
    top_k: int = 3,
    minimum_score: float = MINIMUM_SCORE,
) -> dict[str, list[dict]]:
    """
    Match keywords against everyday umbrella terms ("Grippe", "Herzinfarkt") and return all ICD codes below them.

    Parameters
    ----------
    keywords : list[str]
        Keywords, e.g. from `medical_ner.get_medical_keywords`.
    top_k : int
        Maximum number of umbrella groups per keyword.
    minimum_score : float
        Minimum cosine similarity between keyword and umbrella term.

    Returns
    -------
    dict
        {keyword: [{"umbrella_term": str, "score": float, "icd_codes": [{"code": str, "description": str}]}]}
        sorted by descending score; a keyword without results maps to [].
    """
    if not keywords:
        return {}

    index = _load_umbrella_index()
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
                        "icd_codes": index["group_codes"][group_index],
                    }
                )

        results[keyword] = matches

    return results


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
        sorted by descending score; a keyword without results maps to []. OPS entries have no code.
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
