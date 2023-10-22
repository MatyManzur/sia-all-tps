import plotly.express as px
import plotly.graph_objects as go
import pandas as pn
from plots.kohonen_heatmap import *
import pycountry


def map_europe():
    countries_codes = {}
    for country in pycountry.countries:
        countries_codes[country.name] = country.alpha_3

    data = heatmap_winner_neurons(4, 1000, math.sqrt(2), lambda prev, epoch: INITIAL_RADIUS - 0.5 * (epoch // 250), lambda epoch: 0.1 * (1.0 - (epoch / MAX_ITERATIONS)), False)


    # Estos son los códigos como ARG para Argentina, DEU para Alemania, etc. Sirven para pintar el mapa
    codes = [countries_codes.get(country, 'Unknown code') for country in data["countries"]]


    increments_of = 255/GRID_SIZE
    
    color_by_country=[]
    for i in range(0, len(data["winner_row"])):
        color_by_country.append('rgb({0}, {1}, 0)'.format(data["winner_row"][i] * increments_of, data["winner_col"][i] * increments_of))



    """ Forma sin tomar en cuenta la grilla
    colors = ["red","blue","green","yellow","orange","purple","pink","brown","cyan","magenta","violet","teal","lime","indigo","maroon","navy","silver","gold","olive","gray","black","white","turquoise","orchid","salmon","peru","slategray","darkgreen","tomato"]

    seen_values = {}
    curr_color=0
    color_values = []
    for i in range(0, len(data["winner_row"])):
        row = data["winner_row"][i]
        col = data["winner_col"][i]
        value = seen_values.get((row,col),None)
        if value == None:
            seen_values[(row,col)] = curr_color
            value = curr_color
            curr_color+=1   # Ir probando cuál conviene
        color_values.append(value)

    color_names = [colors[value] for value in color_values]

    dictionary = { "country": data["countries"], "colors": color_names, "iso_alpha":codes}
    df2 = pn.DataFrame.from_dict(dictionary)

    df2["colors"] = df2["colors"].astype(str)

    """

    dictionary = { "country": data["countries"], "colors": color_by_country, "iso_alpha":codes}
    df2 = pn.DataFrame.from_dict(dictionary)

    fig2 = px.choropleth(df2, locations="iso_alpha",
                        color="colors", # Un color por cada neurona
                        hover_name="country", # column to add to hover information
                        hover_data={"colors": False, "iso_alpha": False}
                        )
    fig2.update_traces(showlegend=False)
    fig2.update_geos(fitbounds="locations")
    fig2.show()

if __name__ == '__main__':
    map_europe()
