import json
from sys import argv
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px

RESULT_FILES = ["results/selection/elite.json", "results/selection/roulette.json", "results/selection/ranking.json",
                "results/selection/universal.json", "results/selection/boltzmann.json",
                "results/selection/deterministic_tournament.json", "results/selection/probabilistic_tournament.json"]
RESULT_NAMES = ["Elite", "Roulette", "Ranking", "Universal", "Boltzmann",
                "Deterministic Tournament", "Probabilistic Tournament"]


def main():
    input_files = RESULT_FILES  # argv[1:] if len(argv) > 1 else ['results.json']
    data = []
    for j, file in enumerate(input_files):
        raw_data = json.load(open(file, mode='r'))
        for i, gen in enumerate(raw_data['all_generations'].values()):
            # average_fitness = np.mean(list(map(lambda c: c['fitness'], gen['population'])))
            max_fitness = max(list(map(lambda c: c['fitness'], gen['population'])))
            data.append([RESULT_NAMES[j], i, max_fitness])
    df = pd.DataFrame(data, columns=['Selection Method', 'Generation', 'Highest Fitness'])
    fig = px.line(df, x="Generation", y="Highest Fitness", color='Selection Method', title='Evolution of Highest Fitness by Selection Method', markers=True)
    fig.show()


if __name__ == '__main__':
    main()
