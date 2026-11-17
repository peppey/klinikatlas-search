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
    return tmp_path


def _write_codes(data_dir, code_type, descriptions, embeddings):
    (data_dir / f"{code_type}_codes.json").write_text(json.dumps(descriptions), encoding="utf-8")
    torch.save(torch.tensor(embeddings).half(), data_dir / f"{code_type}_embeddings.pt")


def _write_entries(data_dir, code_type, items):
    (data_dir / f"{code_type}_codes.json").write_text(
        json.dumps([{f"{code_type}code": code, "description": description} for code, description in items]),
        encoding="utf-8",
    )


def _write_umbrella_groups(data_dir, code_type, groups):
    (data_dir / f"{code_type}_umbrella_terms.json").write_text(json.dumps(groups), encoding="utf-8")


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
        ("5-812.5", "5-812", True),
        ("5-812", "5-81", True),
        ("5-813", "5-812", False),
        ("5-812.5", "5-812.6", False),
    ],
)
def test_code_matches(code, pattern, expected):
    assert semantic_search._code_matches(code, pattern) is expected


UMBRELLA_CASES = {
    "icd": {
        "items": [
            ("J09", "Grippe durch zoonotische Viren"),
            ("J10.1", "Grippe mit Manifestationen"),
            ("J11", "Grippe, Viren nicht nachgewiesen"),
            ("J12", "Viruspneumonie"),
            ("I21", "Akuter Myokardinfarkt"),
        ],
        "groups": [
            {"terms": ["Grippe", "Influenza"], "icd": ["J09-J11"]},
            {"terms": ["Herzinfarkt"], "icd": ["I21"]},
        ],
        "expected": ["J09", "J10.1", "J11"],
    },
    "ops": {
        "items": [
            ("5-812", "Arthroskopische Operation am Gelenkknorpel"),
            ("5-812.5", "Arthroskopische Meniskusresektion"),
            ("5-813", "Arthroskopische Refixation am Kapselbandapparat"),
            ("5-820", "Implantation einer Endoprothese am Hüftgelenk"),
        ],
        "groups": [
            {"terms": ["Meniskus-OP", "Meniskusoperation"], "ops": ["5-812"]},
            {"terms": ["Hüftprothese"], "ops": ["5-820"]},
        ],
        "expected": ["5-812", "5-812.5"],
    },
}


@pytest.mark.parametrize("code_type", ["icd", "ops"])
def test_retrieve_by_umbrella_terms_returns_all_codes_below_umbrella(code_type, tmp_data_dir, monkeypatch):
    case = UMBRELLA_CASES[code_type]
    first_group, second_group = case["groups"]
    _write_entries(tmp_data_dir, code_type, case["items"])
    _write_umbrella_groups(tmp_data_dir, code_type, case["groups"])
    best, close = first_group["terms"][1], first_group["terms"][0]
    monkeypatch.setattr(
        semantic_search,
        "embed",
        _fake_embed({best: [1.0, 0.0], close: [0.9, 0.1], second_group["terms"][0]: [0.0, 1.0]}),
    )

    results = semantic_search.retrieve_by_umbrella_terms([best], code_type, minimum_score=0.7)

    assert len(results[best]) == 1
    match = results[best][0]
    assert match["umbrella_term"] == best
    assert match["score"] == pytest.approx(1.0, abs=1e-3)
    assert [entry["code"] for entry in match[f"{code_type}_codes"]] == case["expected"]
    assert all(set(entry) == {"code", "description"} for entry in match[f"{code_type}_codes"])


def test_icd_and_ops_wrappers_use_their_own_umbrella_files(tmp_data_dir, monkeypatch):
    for code_type in ("icd", "ops"):
        case = UMBRELLA_CASES[code_type]
        _write_entries(tmp_data_dir, code_type, case["items"])
        _write_umbrella_groups(tmp_data_dir, code_type, case["groups"])
    monkeypatch.setattr(
        semantic_search,
        "embed",
        _fake_embed(
            {
                "Influenza": [1.0, 0.0],
                "Grippe": [0.9, 0.1],
                "Herzinfarkt": [0.0, 1.0],
                "Meniskus-OP": [1.0, 0.0],
                "Meniskusoperation": [0.9, 0.1],
                "Hüftprothese": [0.0, 1.0],
            }
        ),
    )

    icd = semantic_search.retrieve_icd_by_umbrella_terms(["Influenza"])
    ops = semantic_search.retrieve_ops_by_umbrella_terms(["Meniskus-OP"])

    assert "icd_codes" in icd["Influenza"][0]
    assert "ops_codes" in ops["Meniskus-OP"][0]


def test_retrieve_by_umbrella_terms_empty_keywords():
    assert semantic_search.retrieve_by_umbrella_terms([], "ops") == {}


def test_retrieve_by_umbrella_terms_filters_by_minimum_score(tmp_data_dir, monkeypatch):
    _write_entries(tmp_data_dir, "icd", [("I21", "Akuter Myokardinfarkt")])
    _write_umbrella_groups(tmp_data_dir, "icd", [{"terms": ["Herzinfarkt"], "icd": ["I21"]}])
    monkeypatch.setattr(
        semantic_search,
        "embed",
        _fake_embed({"Herzinfarkt": [0.0, 1.0], "Wäsche": [1.0, 0.0]}),
    )

    results = semantic_search.retrieve_by_umbrella_terms(["Herzinfarkt", "Wäsche"], "icd", minimum_score=0.7)

    assert len(results["Herzinfarkt"]) == 1
    assert results["Wäsche"] == []


def test_retrieve_by_umbrella_terms_term_in_several_groups_and_top_k(tmp_data_dir, monkeypatch):
    _write_entries(tmp_data_dir, "icd", [("K40", "Leistenbruch"), ("S72", "Fraktur des Femurs")])
    _write_umbrella_groups(
        tmp_data_dir,
        "icd",
        [
            {"terms": ["Bruch"], "icd": ["S72"]},
            {"terms": ["Bruch"], "icd": ["K40"]},
        ],
    )
    monkeypatch.setattr(semantic_search, "embed", _fake_embed({"Bruch": [1.0, 0.0]}))

    both = semantic_search.retrieve_by_umbrella_terms(["Bruch"], "icd", top_k=3)
    one = semantic_search.retrieve_by_umbrella_terms(["Bruch"], "icd", top_k=1)

    assert [[c["code"] for c in m["icd_codes"]] for m in both["Bruch"]] == [["S72"], ["K40"]]
    assert len(one["Bruch"]) == 1


@pytest.mark.parametrize("code_type", ["icd", "ops"])
def test_umbrella_embeddings_are_rebuilt_when_terms_change(code_type, tmp_data_dir, monkeypatch):
    _write_entries(tmp_data_dir, code_type, [("X1", "Beschreibung")])
    _write_umbrella_groups(tmp_data_dir, code_type, [{"terms": ["Neu"], code_type: ["X1"]}])
    stored_path = tmp_data_dir / f"{code_type}_umbrella_embeddings.pt"
    torch.save({"terms": ["Alt"], "embeddings": torch.tensor([[1.0, 0.0]]).half()}, stored_path)
    monkeypatch.setattr(semantic_search, "embed", _fake_embed({"Neu": [0.0, 1.0]}))

    index = semantic_search._load_umbrella_index(code_type)

    assert index["terms"] == ["Neu"]
    assert torch.load(stored_path, weights_only=True)["terms"] == ["Neu"]


def test_retrieve_similar_results_includes_codes_for_icd_and_ops(tmp_data_dir, monkeypatch):
    _write_entries(tmp_data_dir, "icd", [("I21", "Akuter Myokardinfarkt")])
    _write_entries(tmp_data_dir, "ops", [("5-812", "Arthroskopische Operation")])
    torch.save(torch.tensor([[1.0, 0.0]]).half(), tmp_data_dir / "icd_embeddings.pt")
    torch.save(torch.tensor([[0.9, 0.1]]).half(), tmp_data_dir / "ops_embeddings.pt")
    monkeypatch.setattr(semantic_search, "embed", _fake_embed({"x": [1.0, 0.0]}))

    results = semantic_search.retrieve_similar_results(["x"])

    assert [(r["type"], r["code"]) for r in results["x"]] == [("icd", "I21"), ("ops", "5-812")]


@pytest.mark.parametrize("code_type", ["icd", "ops"])
def test_shipped_umbrella_groups_all_match_codes(code_type):
    codes = [entry["code"] for entry in semantic_search._load_entries(code_type)]

    empty = [
        group["terms"][0]
        for group in semantic_search._load_umbrella_groups(code_type)
        if not any(semantic_search._matcher(group[code_type])(code) for code in codes)
    ]

    assert empty == []


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
        _write_umbrella_groups(tmp_data_dir, code_type, [{"terms": ["A"], code_type: ["Z"]}])
    monkeypatch.setattr(semantic_search, "embed", lambda texts, *args, **kwargs: torch.eye(len(texts)))

    index = semantic_search._load_index()

    for code_type in semantic_search.CODE_TYPES:
        assert index[code_type][0] == [{"code": None, "description": "A"}, {"code": None, "description": "B"}]
        assert (tmp_data_dir / f"{code_type}_embeddings.pt").exists()
        assert (tmp_data_dir / f"{code_type}_umbrella_embeddings.pt").exists()


@pytest.mark.parametrize("code_type", semantic_search.CODE_TYPES)
def test_shipped_embeddings_match_descriptions(code_type):
    embeddings_path = semantic_search._embeddings_path(code_type)

    if not embeddings_path.exists():
        pytest.skip("embeddings not generated yet")

    descriptions = semantic_search._load_descriptions(code_type)
    embeddings = torch.load(embeddings_path, weights_only=True)

    assert embeddings.shape[0] == len(descriptions)
    assert torch.isfinite(embeddings.float()).all()


@pytest.mark.parametrize("code_type", semantic_search.CODE_TYPES)
def test_shipped_umbrella_embeddings_match_terms(code_type):
    stored = torch.load(semantic_search._umbrella_embeddings_path(code_type), weights_only=True)
    terms = semantic_search._umbrella_terms(semantic_search._load_umbrella_groups(code_type))

    assert stored["terms"] == terms
    assert stored["embeddings"].shape[0] == len(terms)


@pytest.mark.parametrize("code_type", semantic_search.CODE_TYPES)
def test_shipped_umbrella_groups_cover_every_code(code_type):
    matchers = [semantic_search._matcher(group[code_type]) for group in semantic_search._load_umbrella_groups(code_type)]
    codes = [entry["code"] for entry in semantic_search._load_entries(code_type)]

    uncovered = [code for code in codes if not any(matches(code) for matches in matchers)]

    assert uncovered == []
