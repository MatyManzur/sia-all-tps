import plotly.graph_objects as go
import json
from src.ej_1_main import step_perceptron, DATA_AND, DATA_OR_EXC


def plot_ej_1(title, correct_x_y, incorrect_x_y, x, y):
    fig = go.Figure(
        data=[go.Scatter(x=x, y=y[0])],
        layout=go.Layout(
            xaxis=dict(range=[-2, 2], autorange=False),
            yaxis=dict(range=[-2, 2], autorange=False),
            title=title,
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Empezar Transformación",
                              method="animate",
                              args=[None])])]
        ),
        frames=[go.Frame(data=[go.Scatter(x=x, y=values)]) for values in y[1:]]
    )

    fig.add_trace(
        go.Scatter(x=correct_x_y[0], y=correct_x_y[1], marker_symbol='diamond-open-dot', marker_size=15, mode='markers',
                   name='Datos'))
    fig.add_trace(
        go.Scatter(x=incorrect_x_y[0], y=incorrect_x_y[1], marker_symbol='hexagon-open-dot', marker_size=15,
                   mode='markers',
                   fillcolor='aliceblue',
                   name='Datos'))

    fig.show()


def extract_data():
    x = [-1.5, 1.5]
    y = []
    data = json.load(open("results_step.json", mode='r'))
    for iteration in data["weights"].values():
        y.append([-iteration['w1'] / iteration['w2'] * i - iteration['w0'] / iteration['w2'] for i in x])
    return x, y


def plot_or_exc():
    step_perceptron(100, DATA_OR_EXC, 0.01)
    (x, y) = extract_data()
    plot_ej_1("Visualization Exclusive OR", [[-1, 1], [1, -1]], [[-1, 1], [-1, 1]], x, y)


def plot_and():
    step_perceptron(1000, DATA_AND, 0.01)
    (x, y) = extract_data()
    plot_ej_1("Visualization AND", [[1], [1]], [[-1, -1, 1], [-1, 1, -1]], x, y)


if __name__ == "__main__":
    plot_and()
    plot_or_exc()

""" PARA UNA FOTO DE CÓMO QUEDA LA RECTA AL FINAL
    df = pd.DataFrame(dict(
        x=x,
        y=y[-1]
    ))
    fig = px.line(df, x="x", y="y", title="Recta")

    fig.add_trace(go.Scatter(x=[-1, -1, 1], y=[-1, 1, -1], marker_symbol='diamond-open-dot', marker_size=15, mode='markers',
                             name='Datos'))
    fig.add_trace(
        go.Scatter(x=[1], y=[1], marker_symbol='hexagon-open-dot', marker_size=15, mode='markers', fillcolor='aliceblue',
                   name='Datos'))

    fig.show()"""
