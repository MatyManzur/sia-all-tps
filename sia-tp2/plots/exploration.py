import json
from sys import argv
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px

MAX_DIFFERENCE = np.sqrt((5 * 150 ** 2) + (0.7 ** 2))


def chromosome_difference(c1: Tuple, c2: Tuple) -> float:
    dif = np.subtract(c1, c2)
    return 100 * (np.linalg.norm(dif)) / MAX_DIFFERENCE


def main():
    input_files = RESULT_FILES  # argv[1:] if len(argv) > 1 else ['results.json']
    data = []
    for j, file in enumerate(input_files):
        raw_data = json.load(open(file, mode='r'))

        gens = raw_data['all_generations']
        amount_bichos = len(gens['gen_0']['population'])
        amount_generations = len(gens.keys())

        for i in range(1, amount_generations):
            print(f'Generation - {i}')
            prev_gen = gens[f"gen_{i - 1}"]['population']
            this_gen = [x for x in gens[f"gen_{i}"]['population'] if x not in prev_gen]

            generation_exploration = 0
            for bichos_prev in prev_gen:
                genes_prev = tuple(bichos_prev['genes'].values())
                for bicho_this in this_gen:
                    dif = chromosome_difference(genes_prev, tuple(bicho_this['genes'].values()))
                    generation_exploration += dif / (amount_bichos ** 2)

            data.append([RESULT_NAMES[j], i, generation_exploration])
    df = pd.DataFrame(data, columns=['Mutation Method', 'Generation', 'Exploration'])
    fig = px.line(df, x="Generation", y="Exploration", color='Mutation Method', title='Evolution of Exploration by Mutation Method', markers=True)
    fig.show()


if __name__ == '__main__':
    main()
