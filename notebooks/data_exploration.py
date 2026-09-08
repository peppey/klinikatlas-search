import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploration of the Klinikatlas API
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Retrieve ICD Codes, OPS Codes and basic hospital information
    """)
    return


@app.cell
def _():
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
    return (
        BeautifulSoup,
        configuration,
        default_api,
        icd_codes,
        klinikatlas,
        locations,
        ops_codes,
        requests,
    )


@app.cell
def _(icd_codes):
    icd_codes[:5]
    return


@app.cell
def _(ops_codes):
    ops_codes[:5]
    return


@app.cell
def _(locations):
    locations[:5]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Retrieve ids for hospitals via search criteria
    """)
    return


@app.cell
def _(configuration, requests):
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

    return (search_hospitals,)


@app.cell
def _(search_hospitals):
    exemplary_results = search_hospitals(
        icd="A00.0",
        rows=10,
        location="Hamburg"
    )

    exemplary_results["results"][:2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Find hospital via ID
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Searching for an ID returns a HTML page. In this notebook, the HTML page is parsed in a very simple preliminary way to return a JSON with some information for the hospital. We can add more information later.
    """)
    return


@app.cell
def _(BeautifulSoup):
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

    return (parse_hospital,)


@app.cell
def _(configuration, default_api, klinikatlas, parse_hospital):
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

    return (get_hospital,)


@app.cell
def _(get_hospital):
    exemplary_hospital = get_hospital(773675)

    exemplary_hospital
    return


if __name__ == "__main__":
    app.run()

