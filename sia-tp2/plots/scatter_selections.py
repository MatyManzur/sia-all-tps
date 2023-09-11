import json
from sys import argv

import pandas as pd
import plotly.express as px


def main():
    raw_data = json.load(open(argv[1], mode='r'))

    data = list(raw_data.values())[0]
    df = pd.DataFrame(data)
    print(df)
    fig = px.scatter(df, x="avg_generations", y="avg_fitness", color="selection_method",
                     error_x="err_generations_plus", error_x_minus="err_generations_minus",
                     error_y="err_fitness_plus", error_y_minus="err_fitness_minus")
    fig.show()


if __name__ == "__main__":
    main()
