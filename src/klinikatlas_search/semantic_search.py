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


def _load_descriptions(code_type: str) -> list[str]:
    with open(_descriptions_path(code_type), encoding="utf-8") as f:
        return json.load(f)


def build_embeddings() -> None:
    """Embed all ICD and OPS descriptions and store them in data/ (float16 to keep files small)."""
    torch.manual_seed(SEED)

    for code_type in CODE_TYPES:
        embeddings = embed(_load_descriptions(code_type), progress=True)
        torch.save(embeddings.half(), _embeddings_path(code_type))


@lru_cache(maxsize=1)
def _load_index() -> dict[str, tuple[list[str], torch.Tensor]]:
    """Return {code_type: (descriptions, L2-normalized embeddings)}; builds missing embedding files."""
    if not all(_embeddings_path(code_type).exists() for code_type in CODE_TYPES):
        build_embeddings()

    index = {}

    for code_type in CODE_TYPES:
        descriptions = _load_descriptions(code_type)
        embeddings = torch.load(_embeddings_path(code_type), weights_only=True).float()

        if len(descriptions) != len(embeddings):
            raise ValueError(
                f"{_embeddings_path(code_type).name} does not match {_descriptions_path(code_type).name}; "
                "delete it to rebuild."
            )

        index[code_type] = (descriptions, F.normalize(embeddings, dim=1))

    return index


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
        {keyword: [{"description": str, "type": "icd" | "ops", "score": cosine similarity}, ...]}
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
    entries = [(code_type, description) for code_type in CODE_TYPES for description in index[code_type][0]]

    top_scores, top_indices = scores.topk(min(top_k, scores.shape[1]), dim=1)

    return {
        keyword: [
            {
                "description": entries[i][1],
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
