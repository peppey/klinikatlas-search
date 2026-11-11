import json

import pytest
import torch

from klinikatlas_search import semantic_search


@pytest.fixture(autouse=True)
def clear_index_cache():
    semantic_search._load_index.cache_clear()
    yield
    semantic_search._load_index.cache_clear()


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic_search, "DATA_DIR", tmp_path)
    return tmp_path


def _write_codes(data_dir, code_type, descriptions, embeddings):
    (data_dir / f"{code_type}_codes.json").write_text(json.dumps(descriptions), encoding="utf-8")
    torch.save(torch.tensor(embeddings).half(), data_dir / f"{code_type}_embeddings.pt")


def _fake_embed(vectors):
    return lambda texts, *args, **kwargs: torch.tensor([vectors[text] for text in texts])


def test_retrieve_similar_results_empty_keywords():
    assert semantic_search.retrieve_similar_results([]) == {}


def test_retrieve_similar_results_ranks_icd_and_ops_together(tmp_data_dir, monkeypatch):
    _write_codes(tmp_data_dir, "icd", ["Kopfschmerz", "Fraktur"], [[1.0, 0.0], [0.0, 1.0]])
    _write_codes(tmp_data_dir, "ops", ["MRT Schädel"], [[0.9, 0.1]])
    monkeypatch.setattr(semantic_search, "embed", _fake_embed({"Kopf": [1.0, 0.0]}))

    results = semantic_search.retrieve_similar_results(["Kopf"], top_k=2)

    assert list(results) == ["Kopf"]
    assert [(r["description"], r["type"]) for r in results["Kopf"]] == [
        ("Kopfschmerz", "icd"),
        ("MRT Schädel", "ops"),
    ]
    assert results["Kopf"][0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert results["Kopf"][0]["score"] > results["Kopf"][1]["score"]


def test_retrieve_similar_results_one_entry_per_keyword(tmp_data_dir, monkeypatch):
    _write_codes(tmp_data_dir, "icd", ["Kopfschmerz", "Fraktur"], [[1.0, 0.0], [0.0, 1.0]])
    _write_codes(tmp_data_dir, "ops", ["MRT Schädel"], [[0.9, 0.1]])
    monkeypatch.setattr(
        semantic_search,
        "embed",
        _fake_embed({"Kopf": [1.0, 0.0], "Knochen": [0.0, 1.0]}),
    )

    results = semantic_search.retrieve_similar_results(["Kopf", "Knochen"], top_k=1)

    assert results["Kopf"][0]["description"] == "Kopfschmerz"
    assert results["Knochen"][0]["description"] == "Fraktur"
    assert all(len(entries) == 1 for entries in results.values())


def test_retrieve_similar_results_top_k_larger_than_index(tmp_data_dir, monkeypatch):
    _write_codes(tmp_data_dir, "icd", ["A"], [[1.0, 0.0]])
    _write_codes(tmp_data_dir, "ops", ["B"], [[0.0, 1.0]])
    monkeypatch.setattr(semantic_search, "embed", _fake_embed({"x": [1.0, 0.0]}))

    assert len(semantic_search.retrieve_similar_results(["x"], top_k=10)["x"]) == 2


def test_load_index_rejects_mismatching_embeddings(tmp_data_dir):
    _write_codes(tmp_data_dir, "icd", ["A", "B"], [[1.0, 0.0]])
    _write_codes(tmp_data_dir, "ops", ["C"], [[0.0, 1.0]])

    with pytest.raises(ValueError, match="icd_embeddings.pt"):
        semantic_search._load_index()


def test_load_index_builds_missing_embeddings(tmp_data_dir, monkeypatch):
    for code_type in semantic_search.CODE_TYPES:
        (tmp_data_dir / f"{code_type}_codes.json").write_text(json.dumps(["A", "B"]), encoding="utf-8")
    monkeypatch.setattr(semantic_search, "embed", lambda texts, *args, **kwargs: torch.eye(len(texts)))

    index = semantic_search._load_index()

    for code_type in semantic_search.CODE_TYPES:
        assert index[code_type][0] == ["A", "B"]
        assert (tmp_data_dir / f"{code_type}_embeddings.pt").exists()


@pytest.mark.parametrize("code_type", semantic_search.CODE_TYPES)
def test_shipped_embeddings_match_descriptions(code_type):
    embeddings_path = semantic_search._embeddings_path(code_type)

    if not embeddings_path.exists():
        pytest.skip("embeddings not generated yet")

    descriptions = semantic_search._load_descriptions(code_type)
    embeddings = torch.load(embeddings_path, weights_only=True)

    assert embeddings.shape[0] == len(descriptions)
    assert torch.isfinite(embeddings.float()).all()
