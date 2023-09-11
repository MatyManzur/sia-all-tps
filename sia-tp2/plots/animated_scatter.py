import random

import pandas as pd
import plotly.express as px
import plotly
import json

FILE = "results/animated_traditional.json"
def main():
    complete_data = json.load(open(FILE, mode='r'))
    gens = complete_data['all_generations']
    data = []
    for i, gen in enumerate(gens.values()):
        random.shuffle(gen['population'])
        for j,bicho in enumerate(gen['population']):
            data.append([i,j, bicho['fitness']])
    df = pd.DataFrame(data, columns=['Generation','Number in generation','Fitness'])
    fig = px.scatter(df,y="Fitness",x='Number in generation', title='Evolution of Fitness by Generation',animation_frame="Generation",
                     range_y=[0, 45])
    fig.show()





if __name__ == '__main__':
    main()


