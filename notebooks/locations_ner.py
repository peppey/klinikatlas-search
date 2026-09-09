import requests
nominatim_search_endpoint = 'https://nominatim.openstreetmap.org/search?'


def extract_locations(texts, model):
    locations = []
    docs = [model(text) for text in texts]
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ == 'LOC':
                locations.append(ent.text)
    return locations

def load_coordinates(city_name = None, plz=None):
    query_params = {"country": "Deutschland"}
    if city_name:
        query_params["city"] = city_name

    if plz: 
        query_params["postalcode"] = plz

    response = requests.get(f"{nominatim_search_endpoint}q={city_name}")
    return response