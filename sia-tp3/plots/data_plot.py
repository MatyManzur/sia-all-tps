import pandas as pd
import plotly.express as px
from sys import argv

def main():
    df = pd.read_csv(argv[1])
    print(df.head())
    fig = px.scatter_3d(df, x='x1', y='x2', z='x3', color='y')
    fig.show()


if __name__ == '__main__':
    main()
