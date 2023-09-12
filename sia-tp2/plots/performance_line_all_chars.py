import json

import plotly.express as px
import pandas as pd
import plotly.io as pio
from sys import argv
pio.renderers.default = "browser"


def main():
    raw_data = json.load(open(argv[1], mode='r'))
    data = []
    for char, gens in raw_data["characters"].items():
        for gen, info in gens['all_generations'].items():
            gen_number = int(gen.split('_')[1])
            data.append([char, gen_number, info['best']['fitness'], info['worst']['fitness']])
    df = pd.DataFrame(data, columns=['Class', 'Generation', 'Best', 'Worst'])
    print(df)
    fig = px.line(df, x="Generation", y="Best", color='Class', title='Best fitness by generation', markers=True)
    fig.show()


if __name__ == '__main__':
    main()
