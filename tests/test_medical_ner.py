import pytest

from klinikatlas_search import medical_ner


def _prediction(token, label, score, start, end):
    return {"token": token, "label": label, "score": score, "start": start, "end": end}


def test_group_subtokens_merges_wordpieces_and_splits_on_gaps():
    query = "Kniegelenk Schmerz"
    predictions = [
        _prediction("Knie", "B-PROBLEM", 0.9, 0, 4),
        _prediction("##gelenk", "I-PROBLEM", 0.8, 4, 10),
        _prediction("Schmerz", "B-PROBLEM", 0.7, 11, 18),
    ]

    words = medical_ner.group_subtokens(query, predictions)

    assert [word["text"] for word in words] == ["Kniegelenk", "Schmerz"]
    assert [(word["start"], word["end"]) for word in words] == [(0, 10), (11, 18)]
    assert len(words[0]["predictions"]) == 2


def test_get_word_label_outside_only():
    word = {"predictions": [_prediction("Wetter", "O", 0.99, 0, 6)]}

    assert medical_ner.get_word_label(word) == ("O", 0.0)


def test_get_word_label_picks_dominant_type_with_mean_score():
    word = {
        "predictions": [
            _prediction("a", "B-PROBLEM", 0.9, 0, 1),
            _prediction("##b", "I-PROBLEM", 0.7, 1, 2),
            _prediction("##c", "I-TEST", 0.5, 2, 3),
        ]
    }

    label, score = medical_ner.get_word_label(word)

    assert label == "PROBLEM"
    assert score == (0.9 + 0.7) / 2


def test_get_medical_keywords_returns_text_label_score_and_filters_other_labels(monkeypatch):
    entities = [
        {"text": "Kopfschmerzen", "label": "PROBLEM", "score": 0.95, "start": 0, "end": 13},
        {"text": "MRT", "label": "TEST", "score": 0.8, "start": 20, "end": 23},
        {"text": "links", "label": "ANATOMY", "score": 0.9, "start": 30, "end": 35},
    ]
    monkeypatch.setattr(medical_ner, "load_medical_ner", lambda: (None, None, None))
    monkeypatch.setattr(medical_ner, "extract_medical_entities", lambda *args, **kwargs: entities)

    keywords = medical_ner.get_medical_keywords("irrelevant")

    assert keywords == [
        {"text": "Kopfschmerzen", "label": "PROBLEM", "score": 0.95},
        {"text": "MRT", "label": "TEST", "score": 0.8},
    ]


def test_get_medical_keywords_passes_threshold(monkeypatch):
    received = {}

    def fake_extract(query, tokenizer, model, device, score_threshold):
        received["threshold"] = score_threshold
        return []

    monkeypatch.setattr(medical_ner, "load_medical_ner", lambda: (None, None, None))
    monkeypatch.setattr(medical_ner, "extract_medical_entities", fake_extract)

    assert medical_ner.get_medical_keywords("x", minimum_score=0.8) == []
    assert received["threshold"] == 0.8


def test_get_medical_keywords_strips_punctuation(monkeypatch):
    entities = [
        {"text": "MRT,", "label": "TEST", "score": 0.8, "start": 0, "end": 4},
        {"text": "(Knie-OP)", "label": "TREATMENT", "score": 0.9, "start": 5, "end": 14},
        {"text": ".", "label": "PROBLEM", "score": 0.9, "start": 15, "end": 16},
    ]
    monkeypatch.setattr(medical_ner, "load_medical_ner", lambda: (None, None, None))
    monkeypatch.setattr(medical_ner, "extract_medical_entities", lambda *args, **kwargs: entities)

    keywords = medical_ner.get_medical_keywords("irrelevant")

    assert [keyword["text"] for keyword in keywords] == ["MRT", "Knie-OP"]


def _entity(text, label, score, start):
    return {"text": text, "label": label, "score": score, "start": start, "end": start + len(text)}


def test_merge_adjacent_entities_joins_neighbours_of_same_type():
    query = "Fraktur Knie"
    entities = [_entity("Fraktur", "PROBLEM", 0.8, 0), _entity("Knie", "PROBLEM", 0.6, 8)]

    merged = medical_ner.merge_adjacent_entities(query, entities)

    assert len(merged) == 1
    assert merged[0]["text"] == "Fraktur Knie"
    assert (merged[0]["start"], merged[0]["end"]) == (0, 12)
    assert merged[0]["score"] == pytest.approx(0.7)
    assert "count" not in merged[0]


def test_merge_adjacent_entities_averages_score_over_all_words():
    query = "a b c"
    entities = [_entity("a", "TEST", 0.9, 0), _entity("b", "TEST", 0.6, 2), _entity("c", "TEST", 0.3, 4)]

    merged = medical_ner.merge_adjacent_entities(query, entities)

    assert [e["text"] for e in merged] == ["a b c"]
    assert merged[0]["score"] == pytest.approx(0.6)


def test_merge_adjacent_entities_keeps_different_types_apart():
    query = "Knie OP"
    entities = [_entity("Knie", "PROBLEM", 0.9, 0), _entity("OP", "TREATMENT", 0.9, 5)]

    assert [e["text"] for e in medical_ner.merge_adjacent_entities(query, entities)] == ["Knie", "OP"]


def test_merge_adjacent_entities_does_not_cross_other_words_or_punctuation():
    query = "Kopfschmerzen und Schwindel, Husten"
    entities = [
        _entity("Kopfschmerzen", "PROBLEM", 0.9, 0),
        _entity("Schwindel,", "PROBLEM", 0.9, 18),
        _entity("Husten", "PROBLEM", 0.9, 29),
    ]

    merged = medical_ner.merge_adjacent_entities(query, entities)

    assert [e["text"] for e in merged] == ["Kopfschmerzen", "Schwindel,", "Husten"]


def test_merge_adjacent_entities_does_not_modify_input():
    query = "Fraktur Knie"
    entities = [_entity("Fraktur", "PROBLEM", 0.8, 0), _entity("Knie", "PROBLEM", 0.6, 8)]

    medical_ner.merge_adjacent_entities(query, entities)

    assert entities[0]["text"] == "Fraktur"
