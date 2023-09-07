import json

import plotly.express as px
import pandas as pd


def main():
    raw_data = json.load(open('result.json', mode='r'))
    data = []
    for gen in raw_data['all_generations'].values():
        data.append(list(map(lambda c: c['fitness'], gen['population'])))
    df = pd.DataFrame(data)
    print(df)
    fig = px.imshow(df)
    fig.update_xaxes(autorange="reversed")
    fig.show()


if __name__ == '__main__':
    main()
