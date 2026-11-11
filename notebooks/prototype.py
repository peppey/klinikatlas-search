import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from klinikatlas_search import locations_ner as loc
    from klinikatlas_search import medical_ner as med
    from klinikatlas_search import semantic_search

    return loc, med, mo, semantic_search


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hier ein kleiner Protoype zum Ausprobieren
    [(und hier der Code dazu)](https://github.com/peppey/klinikatlas-search)
    """)
    return


@app.cell
def _(mo):
    med_input = mo.ui.text(placeholder="Suchanfrage - Med")
    loc_input = mo.ui.text(placeholder="Suchanfrage - ORT")
    mo.hstack([med_input, loc_input]) 
    return loc_input, med_input


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Derzeitige Verarbeitung der medizinischen Anfrage:
    """)
    return


@app.cell
def _(med, med_input):
    keywords = med.get_medical_keywords(med_input.value)
    keywords
    return (keywords,)


@app.cell
def _(keywords, semantic_search):
    similar_results = semantic_search.retrieve_similar_results(
        [keyword["text"] for keyword in keywords]
    )
    similar_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Derzeitige Verarbeitung der örtlichen Anfrage:
    """)
    return


@app.cell
def _(loc, loc_input):
    location = loc.get_coordinates(loc_input.value)
    location
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Details
    ### Med

    ### Location
    Current workflow logic:

    1. get text query
    2. identify named location entities and zip codes(5 digit numbers not at start of string)
    3. compare named entities and zip codes to csv file of existing hospitals
    4. return coordinates for all matched entities/zip codes found in existing csv
    5. if none are found, search OSM data for coordinates of the unmatched entities/zip codes
    6. return results of the OSM data search
    7. if none are found return empty list
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Open Questions
    1. Result Ordering: Order by Nearest, Order by Most Specializied, Hybrid (i.e. order nearest 10 by specialty or most specialized 10 by distance)?
    2. OSM Search: if no existing hospital at location, do Open Street Map search or not?
    3. Multiple Locations: Return most likely match for location or return multiple?
    """)
    return


if __name__ == "__main__":
    app.run()
