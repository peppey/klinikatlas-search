import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    ###Imports
    from src.klinikatlas_search import locations_ner as loc
    from src.klinikatlas_search import medical_ner as med
    from src.klinikatlas_search import semantic_search as se_se

    ###Input

    print("Write the medical search term then press enter")
    print("If no terms appear you can try for example (lungenkrebs or influenza)")
    med_input = input()
    keywords = med.get_medical_keywords(med_input, minimum_score=0.5)
    icd_umbrella_results = se_se.retrieve_icd_by_umbrella_terms(
        [keyword["text"] for keyword in keywords],
        top_k=3,
        minimum_score=0.7,
    )
    for i in icd_umbrella_results:
        print("Search term", i)
        n = len(icd_umbrella_results[i])
        for j in range(0,n):
            umbrella_term =icd_umbrella_results[i][j]
            print( umbrella_term['umbrella_term'])
            print("Score:", umbrella_term['score'])
            print("Codes:")
            for k in umbrella_term['icd_codes']:
                print(k["description"])
            print("\n \n\n")

    print("Write the location search term then press enter")
    print("If no location appears you can try for example (karlsruhe)")
    loc_input = input()
    location = loc.get_coordinates(loc_input)
    print(location)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
