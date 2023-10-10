import plotly.express as px
import plotly.graph_objects as go
import pandas as pn
from plots.kohonen_heatmap import *
import pycountry


def map_europe():
    countries_codes = {}
    for country in pycountry.countries:
        countries_codes[country.name] = country.alpha_3

    data = heatmap_winner_neurons()


    # Estos son los códigos como ARG para Argentina, DEU para Alemania, etc. Sirven para pintar el mapa
    codes = [countries_codes.get(country, 'Unknown code') for country in data["countries"]]

    seen_values = {}
    curr_color=0
    colors = []
    for i in range(0, len(data["winner_row"])):
        row = data["winner_row"][i]
        col = data["winner_col"][i]
        value = seen_values.get((row,col),None)
        if value == None:
            seen_values[(row,col)] = curr_color
            curr_color+=7   # Ir probando cuál conviene
        colors.append(value)

    print(seen_values)
    dictionary = { "country": data["countries"], "colors": colors, "iso_alpha":codes}
    df2 = pn.DataFrame.from_dict(dictionary)

    fig2 = px.choropleth(df2, locations="iso_alpha",
                        color="colors", # Un color por cada neurona
                        hover_name="country", # column to add to hover information
                        hover_data={"colors": False, "iso_alpha": False}
                        )
    fig2.update_geos(fitbounds="locations")
    fig2.show()

if __name__ == '__main__':
    map_europe()
