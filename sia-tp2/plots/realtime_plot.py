import json
import time

import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


def main():
    fig = go.FigureWidget()
    fig.add_scatter()
    scatter = fig.data[0]
    raw_data = json.load(open('result.json', mode='r'))
    best_fitnesses = []
    for gen in raw_data['all_generations'].values():
        best_fitnesses.append(gen['population'][0]['fitness'])
        scatter.y = best_fitnesses
        time.sleep(10/1000)


if __name__ == '__main__':
    main()