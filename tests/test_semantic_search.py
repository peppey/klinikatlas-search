import json

import pytest
import torch

from klinikatlas_search import semantic_search


@pytest.fixture(autouse=True)
def clear_index_cache():
    semantic_search._load_index.cache_clear()
    semantic_search._load_umbrella_index.cache_clear()
    yield
    semantic_search._load_index.cache_clear()
    semantic_search._load_umbrella_index.cache_clear()


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic_search, "DATA_DIR", tmp_path)
    monkeypatch.setattr(semantic_search, "UMBRELLA_TERMS_PATH", tmp_path / "icd_umbrella_terms.json")
    monkeypatch.setattr(semantic_search, "UMBRELLA_EMBEDDINGS_PATH", tmp_path / "icd_umbrella_embeddings.pt")
    return tmp_path


def _write_codes(data_dir, code_type, descriptions, embeddings):
    (data_dir / f"{code_type}_codes.json").write_text(json.dumps(descriptions), encoding="utf-8")
    torch.save(torch.tensor(embeddings).half(), data_dir / f"{code_type}_embeddings.pt")


def _write_icd_with_codes(data_dir, items):
    (data_dir / "icd_codes.json").write_text(
        json.dumps([{"icdcode": code, "description": description} for code, description in items]),
        encoding="utf-8",
    )


def _write_umbrella_groups(data_dir, groups):
    (data_dir / "icd_umbrella_terms.json").write_text(json.dumps(groups), encoding="utf-8")


def _fake_embed(vectors):
    return lambda texts, *args, **kwargs: torch.tensor([vectors[text] for text in texts])


@pytest.mark.parametrize(
    "code, pattern, expected",
    [
        ("J10.1", "J10", True),
        ("J10.1", "J10.1", True),
        ("J10.1", "J10.2", False),
        ("J10", "J10.1", False),
        ("J10.1", "J09-J11", True),
        ("J09", "J09-J11", True),
        ("J11.8", "J09-J11", True),
        ("J12", "J09-J11", False),
        ("I80.1", "I80-I82", True),
    ],
)
def test_code_matches(code, pattern, expected):
    assert semantic_search._code_matches(code, pattern) is expected


def test_retrieve_icd_by_umbrella_terms_empty_keywords():
    assert semantic_search.retrieve_icd_by_umbrella_terms([]) == {}


def test_retrieve_icd_by_umbrella_terms_returns_all_codes_below_umbrella(tmp_data_dir, monkeypatch):
    _write_icd_with_codes(
        tmp_data_dir,
        [
            ("J09", "Grippe durch zoonotische Viren"),
            ("J10.1", "Grippe mit Manifestationen"),
            ("J11", "Grippe, Viren nicht nachgewiesen"),
            ("J12", "Viruspneumonie"),
            ("I21", "Akuter Myokardinfarkt"),
        ],
    )
    _write_umbrella_groups(
        tmp_data_dir,
        [
            {"terms": ["Grippe", "Influenza"], "icd": ["J09-J11"]},
            {"terms": ["Herzinfarkt"], "icd": ["I21"]},
        ],
    )
    monkeypatch.setattr(
        semantic_search,
        "embed",
        _fake_embed(
            {
                "Grippe": [0.9, 0.1],
                "Influenza": [1.0, 0.0],
                "Herzinfarkt": [0.0, 1.0],
            }
        ),
    )

    results = semantic_search.retrieve_icd_by_umbrella_terms(["Influenza"], minimum_score=0.7)

    assert len(results["Influenza"]) == 1
    match = results["Influenza"][0]
    assert match["umbrella_term"] == "Influenza"
    assert match["score"] == pytest.approx(1.0, abs=1e-3)
    assert match["icd_codes"] == [
        {"code": "J09", "description": "Grippe durch zoonotische Viren"},
        {"code": "J10.1", "description": "Grippe mit Manifestationen"},
        {"code": "J11", "description": "Grippe, Viren nicht nachgewiesen"},
    ]


def test_retrieve_icd_by_umbrella_terms_filters_by_minimum_score(tmp_data_dir, monkeypatch):
    _write_icd_with_codes(tmp_data_dir, [("I21", "Akuter Myokardinfarkt")])
    _write_umbrella_groups(tmp_data_dir, [{"terms": ["Herzinfarkt"], "icd": ["I21"]}])
    monkeypatch.setattr(
        semantic_search,
        "embed",
        _fake_embed({"Herzinfarkt": [0.0, 1.0], "Wäsche": [1.0, 0.0]}),
    )

    results = semantic_search.retrieve_icd_by_umbrella_terms(["Herzinfarkt", "Wäsche"], minimum_score=0.7)

    assert len(results["Herzinfarkt"]) == 1
    assert results["Wäsche"] == []


def test_retrieve_icd_by_umbrella_terms_term_in_several_groups_and_top_k(tmp_data_dir, monkeypatch):
    _write_icd_with_codes(tmp_data_dir, [("K40", "Leistenbruch"), ("S72", "Fraktur des Femurs")])
    _write_umbrella_groups(
        tmp_data_dir,
        [
            {"terms": ["Bruch"], "icd": ["S72"]},
            {"terms": ["Bruch"], "icd": ["K40"]},
        ],
    )
    monkeypatch.setattr(semantic_search, "embed", _fake_embed({"Bruch": [1.0, 0.0]}))

    both = semantic_search.retrieve_icd_by_umbrella_terms(["Bruch"], top_k=3)
    one = semantic_search.retrieve_icd_by_umbrella_terms(["Bruch"], top_k=1)

    assert [[c["code"] for c in m["icd_codes"]] for m in both["Bruch"]] == [["S72"], ["K40"]]
    assert len(one["Bruch"]) == 1


def test_umbrella_embeddings_are_rebuilt_when_terms_change(tmp_data_dir, monkeypatch):
    _write_icd_with_codes(tmp_data_dir, [("I21", "Akuter Myokardinfarkt")])
    _write_umbrella_groups(tmp_data_dir, [{"terms": ["Herzinfarkt"], "icd": ["I21"]}])
    torch.save(
        {"terms": ["Alt"], "embeddings": torch.tensor([[1.0, 0.0]]).half()},
        tmp_data_dir / "icd_umbrella_embeddings.pt",
    )
    monkeypatch.setattr(semantic_search, "embed", _fake_embed({"Herzinfarkt": [0.0, 1.0]}))

    index = semantic_search._load_umbrella_index()

    assert index["terms"] == ["Herzinfarkt"]
    stored = torch.load(tmp_data_dir / "icd_umbrella_embeddings.pt", weights_only=True)
    assert stored["terms"] == ["Herzinfarkt"]


def test_retrieve_similar_results_includes_icd_code_and_none_for_ops(tmp_data_dir, monkeypatch):
    _write_icd_with_codes(tmp_data_dir, [("I21", "Akuter Myokardinfarkt")])
    torch.save(torch.tensor([[1.0, 0.0]]).half(), tmp_data_dir / "icd_embeddings.pt")
    _write_codes(tmp_data_dir, "ops", ["MRT Schädel"], [[0.9, 0.1]])
    monkeypatch.setattr(semantic_search, "embed", _fake_embed({"x": [1.0, 0.0]}))

    results = semantic_search.retrieve_similar_results(["x"])

    assert [(r["type"], r["code"]) for r in results["x"]] == [("icd", "I21"), ("ops", None)]


def test_shipped_umbrella_groups_all_match_icd_codes():
    entries = semantic_search._load_entries("icd")

    empty = [
        group["terms"][0]
        for group in semantic_search._load_umbrella_groups()
        if not any(
            _code_in_group(entry["code"], group["icd"]) for entry in entries
        )
    ]

    assert empty == []


def _code_in_group(code, patterns):
    return any(semantic_search._code_matches(code, pattern) for pattern in patterns)


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

    results = semantic_search.retrieve_similar_results(["x"], top_k=10, minimum_score=0.0)

    assert len(results["x"]) == 2


def test_retrieve_similar_results_filters_by_minimum_score(tmp_data_dir, monkeypatch):
    _write_codes(tmp_data_dir, "icd", ["A"], [[1.0, 0.0]])
    _write_codes(tmp_data_dir, "ops", ["B"], [[0.0, 1.0]])
    monkeypatch.setattr(semantic_search, "embed", _fake_embed({"x": [1.0, 0.0], "y": [1.0, 1.0]}))

    results = semantic_search.retrieve_similar_results(["x", "y"], minimum_score=0.9)

    assert [r["description"] for r in results["x"]] == ["A"]
    assert results["y"] == []


def test_load_index_rejects_mismatching_embeddings(tmp_data_dir):
    _write_codes(tmp_data_dir, "icd", ["A", "B"], [[1.0, 0.0]])
    _write_codes(tmp_data_dir, "ops", ["C"], [[0.0, 1.0]])

    with pytest.raises(ValueError, match="icd_embeddings.pt"):
        semantic_search._load_index()


def test_load_index_builds_missing_embeddings(tmp_data_dir, monkeypatch):
    for code_type in semantic_search.CODE_TYPES:
        (tmp_data_dir / f"{code_type}_codes.json").write_text(json.dumps(["A", "B"]), encoding="utf-8")
    _write_umbrella_groups(tmp_data_dir, [{"terms": ["A"], "icd": ["Z"]}])
    monkeypatch.setattr(semantic_search, "embed", lambda texts, *args, **kwargs: torch.eye(len(texts)))

    index = semantic_search._load_index()

    for code_type in semantic_search.CODE_TYPES:
        assert index[code_type][0] == [{"code": None, "description": "A"}, {"code": None, "description": "B"}]
        assert (tmp_data_dir / f"{code_type}_embeddings.pt").exists()
    assert (tmp_data_dir / "icd_umbrella_embeddings.pt").exists()


@pytest.mark.parametrize("code_type", semantic_search.CODE_TYPES)
def test_shipped_embeddings_match_descriptions(code_type):
    embeddings_path = semantic_search._embeddings_path(code_type)

    if not embeddings_path.exists():
        pytest.skip("embeddings not generated yet")

    descriptions = semantic_search._load_descriptions(code_type)
    embeddings = torch.load(embeddings_path, weights_only=True)

    assert embeddings.shape[0] == len(descriptions)
    assert torch.isfinite(embeddings.float()).all()
