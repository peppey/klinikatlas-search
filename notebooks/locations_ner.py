import re
from geopy.geocoders import Nominatim


with open("hospital_locations.txt","r") as f:
    hospitals = f.read().split('\n')
f.close()

def extract_locations(texts, model):
    locations = []
    docs = [model(text) for text in texts]
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ == 'LOC':
                locations.append(ent.text)
    return locations

def load_coordinates(query_params):
    locator = Nominatim(user_agent="klinik-atlas") #might need to look into this for commerical use

    country = "Deutschland"

    place = locator.geocode(f"{query_params["city"]}{", " + query_params["plz"] if query_params["plz"] else ""}, {country}")

    return place.latitude, place.longitude

def retrieve_possible_plz(entity):
    plzs = re.findall("\D[0-9]{5}\D", entity) # not sure about more specific plz rules than "has to be 5 digit"
    return [possible_plz[1:6] for possible_plz in plzs] #slice of only the number

def check_city(entity):
    # todo, not sure how this would look like. secondary NER model perhaps
    return

def determine_city(entity):
    query_params = {}
    possible_plzs = retrieve_possible_plz(entity)
    if possible_plzs:
        query_params["plz"] = possible_plzs[0] # take first viable number
    possible_city_names = check_city(entity)
    if possible_city_names:
        query_params["city"] = possible_city_names[0] # take first viable name
    return query_params

def determine_location_query_parameter(possible_location_entities): #not sure yet how multiple entities should be handled
    query_params = {"city": "", "plz":""}
    for entity in possible_location_entities:
        if entity.lower() in hospitals:
            return entity
        else:
            query_params = determine_city(entity=entity)
            
    if query_params:
        return load_coordinates(query_params)
