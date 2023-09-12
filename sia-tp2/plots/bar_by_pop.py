import json
import pandas as pd
import plotly.express as px
import numpy as np

file = "results/populations.json"
def main():
    data = []
    raw_data = json.load(open(file, mode='r'))
    for pop, pop_data in raw_data['all_populations'].items():
        fitness = list(map(lambda x: x['fitness'], pop_data))
        data.append([pop, np.mean(fitness), np.std(fitness)])
    df = pd.DataFrame(data, columns=['Population Distribution','Best Average Fitness','STD'])
    print(df)
    fig = px.bar(df, x="Population Distribution", y="Best Average Fitness", error_y="STD",title='Best Average Fitness from different Population Distributions')
    fig.show()



if __name__ == '__main__':
    main()