# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
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

    Bitte das Notebook erst einmal ausführen und dann in die Felder anstatt "Suchanfrage - Med" und "Suchanfrage - ORT" Suchbegriffe eingeben.
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
    print(type(med_input))
    keywords = med.get_medical_keywords(med_input.value, minimum_score=0.5)
    return (keywords,)


@app.cell
def _(icd_umbrella_results):
    icd_umbrella_results
    return


@app.cell
def _(keywords, semantic_search):
    icd_umbrella_results = semantic_search.retrieve_icd_by_umbrella_terms(
        [keyword["text"] for keyword in keywords],
        top_k=3,
        minimum_score=0.7,
    )
    icd_umbrella_results
    return (icd_umbrella_results,)


@app.cell
def _(keywords, semantic_search):
    ops_umbrella_results = semantic_search.retrieve_ops_by_umbrella_terms(
        [keyword["text"] for keyword in keywords],
        top_k=3,
        minimum_score=0.7,
    )
    ops_umbrella_results
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

    1. Ein Modell extrahiert medizinische Keywords aus der Suchanfrage
    2. Die Keywords werden semantisch mit "Umbrella Terms" gematcht, die jeweils für eine Reihe ICD- oder OPS-Codes stehen
    3. Momentan werden alle Umbrella-Terms und zugehörige Codes zurückgegeben, die einen Score von mindestens 70% haben (was anpassbar ist)

    ### Location
    Aktueller Ablauf:

    1. Suchanfrage entgegennehmen
    2. Benannte Orte und Postleitzahlen erkennen (fünfstellige Zahlen, die nicht am Textanfang stehen)
    3. Erkannte Orte und Postleitzahlen mit der CSV-Datei der bekannten Krankenhäuser abgleichen
    4. Koordinaten aller Orte/Postleitzahlen zurückgeben, die in der CSV gefunden wurden
    5. Wenn keine gefunden wurden, in den OSM-Daten nach Koordinaten der nicht zugeordneten Orte/Postleitzahlen suchen
    Ergebnisse der OSM-Suche zurückgeben
    6. Wenn auch dort nichts gefunden wird, eine leere Liste zurückgeben



    ## Einschränkungen

    - Momentan kann nur mit Begriffen, die aus einem Wort bestehen, umgegangen werden, nicht beispielsweise mit "Fraktur Sprunggelenk"
    - Es ist noch nicht möglich, OPS- oder ICD-Codes als Zahlencode einzugeben, nur Beschreibungen von den Codes
    - Die Umbrella Terms wurden von einem LLM geschrieben und müssen nachbearbeitet werden. Momentan gibt es für weniger als die Hälfte der Codes einen Umbrella Terms; Es muss auch bestimmt werden, wie mit den mehreren Codes umgegeben werden soll, die für einen Umbrella-Term zurückgegeben werden sollen. Sie könnten auch noch sortiert werden
    - Die Modelle können noch finegetuned und quantitativ evaluiert werden
    - Es gibt noch keine Ähnlichkeitssuche nach Krankenhausspezialisierungen
    - Wir haben die Suche nach OPS und ICD Codes im Moment getrennt, natürlich würde sie später zusammengeführt werden
    - Orte mit Rechtschreibfehlern werden noch nicht erkannt; Außerdem hätten wir noch Fragen dazu, wie die Suche nach Orten aussehen sollte
    """)
    return


if __name__ == "__main__":
    app.run()
