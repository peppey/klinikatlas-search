import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Importing list of
    """)
    return


@app.cell
def _():
    from deutschland import klinikatlas
    from deutschland.klinikatlas.api import default_api
    import requests
    from bs4 import BeautifulSoup
    from feature_folder import data_exploration_import as de

    return BeautifulSoup, de, requests


@app.cell
def _(de):
    tur = de.search_hospitals(rows=20)
    tur
    return (tur,)


@app.cell
def _(tur):
    a = tur['results'][0]['detailLink']
    b = "https://bundes-klinik-atlas.de"+a
    print(b)
    return (b,)


@app.cell
def _(BeautifulSoup, b, requests):
    response = requests.get(b)
    response.raise_for_status()

    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    ins= soup.find("herz")
    print(soup)
    return (soup,)


@app.cell
def _(soup):
    zertificate = soup.find("ul", class_="c-checklist")
    fachabteilungen = soup.find("ul", class_="rte_ul")
    allz = zertificate.find_all("li")
    allfach = fachabteilungen.find_all("li")
    for item in allz:
        print(item.get_text(strip=True))
    for item in allfach:
        print(item.get_text(strip=True))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
