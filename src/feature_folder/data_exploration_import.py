# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:23:21 2026

@author: carlh
"""

from deutschland import klinikatlas
from deutschland.klinikatlas.api import default_api
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://bundes-klinik-atlas.de"

configuration = klinikatlas.Configuration(
    host=BASE_URL
)

with klinikatlas.ApiClient(configuration) as api_client:
    api = default_api.DefaultApi(api_client)

    icd_codes = api.fileadmin_json_icd_codes_json_get()
    ops_codes = api.fileadmin_json_ops_codes_json_get()
    locations = api.fileadmin_json_locations_json_get()
def search_hospitals(
    icd=None,
    ops=None,
    location=None,
    start=0,
    rows=10,
):
    """Search hospitals using ICD, OPS, and geographic filters.

    Parameters
    ----------
    icd : str, optional
        ICD code used to filter hospitals by diagnosis.
    ops : str, optional
        OPS code used to filter hospitals by procedure or treatment.
    location : str, optional
        Geographic label used to filter hospitals by location.
    start : int, default=0
        Index of the first result to return.
    rows : int, default=10
        Maximum number of results to return.

    Returns
    -------
    dict
        JSON response from the Bundes-Klinik-Atlas search API.
    """
    params = {
        "tx_solr_start": start,
        "tx_solr_rows": rows,
    }

    if icd is not None:
        params["tx_solr_icd"] = icd

    if ops is not None:
        params["tx_solr_ops"] = ops

    if location is not None:
        params["tx_solr_geolabel"] = location

    response = requests.get(
        f"{configuration.host}/searchresults/",
        params=params,
    )

    response.raise_for_status()

    return response.json()
def parse_hospital(html: str, hospital_id: int | None = None) -> dict:
    """Parse a Bundes-Klinik-Atlas hospital HTML page which
    we get as a result for an id-specific search.

    Parameters
    ----------
    html : str
        HTML returned by `krankenhaussuche_krankenhaus_id_get`.

    Returns
    -------
    dict
        Structured hospital information.
    """

    soup = BeautifulSoup(html, "html.parser")

    title = soup.find("title")
    name = title.get_text(strip=True).split(" | ")[0] if title else None

    hospital = {
        "id": hospital_id,
        "name": name,
        "address": None,
        "phone": None,
        "email": None,
        "website": None,
    }

    address = soup.find("address")
    if address:
        hospital["address"] = address.get_text(" ", strip=True)

    phone = soup.select_one('a[href^="tel:"]')
    if phone:
        hospital["phone"] = phone.get_text(" ", strip=True)

    email = soup.select_one('a[href^="mailto:"]')
    if email:
        hospital["email"] = email.get_text(" ", strip=True)

    website = soup.select_one(
        'a[href^="http"]:not([href*="bundes-klinik-atlas.de"])'
    )
    if website:
        hospital["website"] = website.get("href")

    return hospital
def get_hospital(hospital_id):
    """Get the detail page of a specific hospital.

    Parameters
    ----------
    hospital_id : int
        Unique ID of the hospital.

    Returns
    -------
    dict
        content of the hospital's detail page.
    """
    with klinikatlas.ApiClient(configuration) as api_client:
        api = default_api.DefaultApi(api_client)

        html = api.krankenhaussuche_krankenhaus_id_get(
            id=hospital_id
        )

    return parse_hospital(html, hospital_id=hospital_id)
