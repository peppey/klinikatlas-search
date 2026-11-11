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

    assert medical_ner.get_medical_keywords("x", score_threshold=0.8) == []
    assert received["threshold"] == 0.8
