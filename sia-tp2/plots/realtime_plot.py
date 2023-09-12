import json
import time
from sys import argv

import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


def main():
    fig = go.FigureWidget()
    fig.add_scatter()
    scatter = fig.data[0]
    result_file = argv[1] if len(argv) > 1 else 'result.json'
    raw_data = json.load(open(result_file, mode='r'))
    best_fitnesses = []
    for gen in raw_data['all_generations'].values():
        best_fitnesses.append(gen['population'][0]['fitness'])
        scatter.y = best_fitnesses
        time.sleep(10/1000)


if __name__ == '__main__':
    main()