import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler


def boxplot():
    df = pd.read_csv('../data/europe.csv')

    fig = px.box(df, y=df.columns[1:], title='Boxplots for Variables')
    fig.show()

    numeric_df = df.iloc[:, 1:]

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numeric_df)

    standardized_df = pd.DataFrame(standardized_data, columns=numeric_df.columns)

    fig = px.box(standardized_df, y=standardized_df.columns, title='Boxplots for Standardized Variables')
    fig.show()


if __name__ == '__main__':
    boxplot()
