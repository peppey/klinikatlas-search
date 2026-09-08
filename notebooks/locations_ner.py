
def extract_locations(texts, model):
    locations = []
    docs = [model(text) for text in texts]
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ == 'LOC':
                locations.append(ent.text)
    return locations