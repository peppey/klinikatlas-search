import re
import pandas
from flair.data import Sentence
from flair.models import SequenceTagger
from geopy.geocoders import Nominatim


tagger = SequenceTagger.load("flair/ner-german-large")

df = pandas.read_csv("hospital_locations.txt")

def lookup_coordinates(possible_city_names, possible_plz):
    """search OpenStreetMap data for recognized entities.
    Last resort if none of the entites were found in existing hospital list.

    
    Parameters
    ----------
    possible_locations: List[String]
        list of NER results
    possible_plz: List[String]
        list of regex results
        
    
    Returns
    -------
    List
        JSON style list of matching locations in the format below or []
        [{
            Search: entity extracted from query | NER entity or PLZ
            City: city name or full address
            Plz: zip code if available
            latlon: latitude,longitude of found result 
        }]
            
    """

    reqs = []
    result = []
    if possible_city_names:
        reqs.extend(possible_plz)
    if possible_plz:
        reqs.extend(possible_plz)

    
    locator = Nominatim(user_agent="klinik-atlas") 
    country = "Deutschland"

    for req in reqs:

        place = locator.geocode(f"{req}, {country}", timeout=60)

        if place:
            result.append({
                "Search": req,
                "City": place.adress,
                "Plz": None,
                "latlon": f"{place.latitude}{place.longitude}"
            })
    return result

def load_coordinates_by_name(possible_locations):
    """search existing base of hospital locations for a matching city name
    
    Parameters
    ----------
    possible_locations: List[String]
        list of NER results
    
    Returns
    -------
    List
        JSON style list of matching locations in the format below or []
        [{
            Search: entity extracted from query | NER entity or PLZ
            City: city name or full address
            Plz: zip code if available
            latlon: latitude,longitude of found result 
        }]
            
    """
    matching_coordinates = []
    for entity in possible_locations:
        for index, row in df.iterrows():
            if row["City"].lower().startswith(entity.lower()):
                matching_coordinates.append({
                    "Search": row["City"],
                    "City":row["City"],
                    "Plz": row["Plz"],
                    "latlon": f"{row["Latitude"]},{row["Longitude"]}"
                })
    return matching_coordinates


def load_coordinates_by_plz(possible_plz):
    """search existing base of hospital locations for a matching zip code
    
    Parameters
    ----------
    possible_plz: List[String]
        list of regex results
    
    Returns
    -------
    List
        JSON style list of matching locations in the format below or []
        [{
            Search: entity extracted from query | NER entity or PLZ
            City: city name or full address
            Plz: zip code if available
            latlon: latitude,longitude of found result 
        }]
            
    """
    matching_coordinates = []
    for entity in possible_plz:
        for index, row in df.iterrows():
            if entity == row["Plz"]:
                matching_coordinates.append({
                    "Search": row["Plz"],
                    "City":row["City"],
                    "Plz": row["Plz"],
                    "latlon": f"{row["Latitude"]},{row["Longitude"]}"
                })   
    return matching_coordinates

def filter_five_digit_numbers(text_query):
    """apply regex to filter for 5 digit numbers as possible zip codes
    
    Parameters
    ----------
    text_query: String
    
    Returns
    -------
    List[String]
        List with all 5 digit numbers in the text query
            
    """
    pat = r'\D[0-9]{5}\D*'
    plzs = re.findall(pat, text_query) 

    return [possible_plz[1:6] for possible_plz in plzs]

def filter_one_per_city(locations, plzs):
    """combine NER results with regex results and remove duplicate cities
    
    Parameters
    ----------
    locations: List[Dict]
        coordinates for found locations
    plzs: List[Dict]
        coordinates for found zip codes
    
    Returns
    -------
    List
        JSON style list of locations in the format below or []
        [{
            Search: entity extracted from query | NER entity or PLZ
            City: city name or full address
            Plz: zip code if available
            latlon: latitude,longitude of found result 
        }]
            
    """
    result_set = {}

    for location in locations:
        result_set[location["City"]] = {
            
            "Search": location["City"],
            "City": location["City"],
            "Plz": location["Plz"],
            "latlon": location["latlon"]  
        }
    for plz in plzs:
        result_set[plz["City"]] = {
            
            "Search": plz["Plz"],
            "City": plz["City"],
            "Plz": plz["Plz"],
            "latlon": plz["latlon"]   
        }
    return list(result_set.values())

def get_coordinates_from_entities(possible_location_entities, possible_plzs): 
    """find coordinates for the provided entities.
    
    Parameters
    ----------
    possible_location_entities: List[String]
        NER results
    possible_plzs: List[String]
        regex results
    Returns
    -------
    List
        JSON style list of locations in the format or []
        [{
            Search: entity extracted from query | NER entity or PLZ
            City: city name or full address
            Plz: zip code if available
            latlon: latitude,longitude of found result 
        }]
            
    """
    existing_locations = load_coordinates_by_name(possible_location_entities)
    existing_plzs = load_coordinates_by_plz(possible_plzs)

    #does not look up coordinates for unmatched entities if atleast one entity was matched
    if not existing_plzs and not existing_locations: 
        return lookup_coordinates(possible_location_entities, possible_plzs)
    
    else:
        return filter_one_per_city(existing_locations, existing_plzs)


def process_text_for_location_query(text_query):
    """apply NER for names and regex for possible zip codes.

    Parameters
    ----------
    text_query: str
        the text query to process

    Returns
    -------
    Two Lists with NER and Regex results
        
    """
    possible_plzs = filter_five_digit_numbers(text_query)

    sentence = Sentence(text_query)
    tagger.predict(sentence)
    loc_enities = []
    for entity in sentence.get_spans('ner'):
        if entity.labels[0].value == "LOC":
            loc_enities.append(entity.text)      

    return loc_enities, possible_plzs

def get_coordinates(text_query):
    """Process text for location data and try to retrieve corresponding coordinates.

    Parameters
    ----------
    text_query: str
        the text query to process

    Returns
    -------
    List
        JSON style list of locations in the format or []
        [{
            Search: entity extracted from query | NER entity or PLZ
            City: city name or full address
            Plz: zip code if available
            latlon: latitude,longitude of found result 
        }]
    """

    loc_enities, possible_plzs = process_text_for_location_query(text_query)

    return get_coordinates_from_entities(loc_enities, possible_plzs)


