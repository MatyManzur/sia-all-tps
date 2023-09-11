import json
import pandas as pd
import plotly.express as px
import numpy as np

RESULT_FILES = ["results/complete.json","results/gen.json","results/limited.json","results/uniform.json",
                "results/decreasing.json","results/increasing.json","results/sinusoidal.json"]
RESULT_NAMES = ["Complete","Gen","Limited","Uniform","Decreasing","Increasing","Sinusoidal"]



def main():
    data = []
    for j, file in enumerate(RESULT_FILES):
        raw_data = json.load(open(file, mode='r'))
        chars= {'Warrior': [], 'Rogue': [], 'Warden': [], 'Archer': []}
        for char_data in raw_data['all_iterations']:
            chars[char_data['class']].append(char_data['fitness'])
        for char,fitness in chars.items():
            data.append([RESULT_NAMES[j],char,np.average(fitness), np.std(fitness)])


    df = pd.DataFrame(data, columns=['Mutation Type','Class','Average Fitness', 'STD'])
    print(df)
    fig = px.bar(df, x="Mutation Type", y="Average Fitness", error_y="STD",color='Class',barmode="group",title='Average Fitness by Mutation Type and Class')
    fig.show()





if __name__ == '__main__':
    main()