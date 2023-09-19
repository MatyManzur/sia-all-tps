import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import json
from sys import argv



x = [-1.5,1.5]
y=[]
data = json.load(open("results_step.json", mode='r'))
for iteration in data["weights"].values():
    y.append([-iteration['w1']/iteration['w2']*i- iteration['w0']/iteration['w2'] for i in x ])


df = pd.DataFrame(dict(
    x = x,
    y = y[-1]
))
fig = px.line(df, x="x", y="y", title="Recta") 


fig.add_trace(go.Scatter(x=[-1,-1,1], y=[-1,1,-1], marker_symbol='diamond-open-dot', marker_size=15, mode='markers', name='Datos'))
fig.add_trace(go.Scatter(x=[1], y=[1], marker_symbol='hexagon-open-dot', marker_size=15, mode='markers', fillcolor='aliceblue', name='Datos'))


fig.show()



fig = go.Figure(
    data=[go.Scatter(x=x, y=y[0])],
    layout=go.Layout(
        xaxis=dict(range=[-2, 2], autorange=False),
        yaxis=dict(range=[-2, 2], autorange=False),
        title="Visualización del AND",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Empezar Transformación",
                          method="animate",
                          args=[None])])]
    ),
    frames=[go.Frame(data=[go.Scatter(x=x, y=values)]) for values in y[1:]]
)


#frames=[go.Frame(data=[go.Scatter(x=x, y=y[i])]) for i in range(len(y))]

fig.add_trace(go.Scatter(x=[-1,-1,1], y=[-1,1,-1], marker_symbol='diamond-open-dot', marker_size=15, mode='markers', name='Datos'))
fig.add_trace(go.Scatter(x=[1], y=[1], marker_symbol='hexagon-open-dot', marker_size=15, mode='markers', fillcolor='aliceblue', name='Datos'))

fig.show()
