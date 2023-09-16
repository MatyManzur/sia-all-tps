import plotly.express as px
import pandas as pd
import plotly.graph_objects as go



x = [-1.5,1.5]
y = [-0.8636881433540377*i+0.5498923464051484 for i in x ]

df = pd.DataFrame(dict(
    x = x,
    y = y
))
fig = px.line(df, x="x", y="y", title="Recta") 


fig.add_trace(go.Scatter(x=[-1,-1,1,1], y=[-1,1,-1,1], marker_symbol='diamond-open-dot', marker_size=15, mode='markers', name='Datos'))


fig.show()




x = [-1.5,1.5]
y = [-2.125806496275092 * X + -1.055477873623969 for X in x ]
y1 = [-1.0789594975415593 * X + -0.36708886249016376 for X in x]
y2 = [-0.5573843144851826 * X + -0.024109778698086594 for X in x]
y3 = [-1.0789594975415593 * X + 0.30272035313532114 for X in x]

fig = go.Figure(
    data=[go.Scatter(x=x, y=y)],
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
    frames=[go.Frame(data=[go.Scatter(x=x, y=y1)]),
            go.Frame(data=[go.Scatter(x=x, y=y2)]), 
            go.Frame(data=[go.Scatter(x=x, y=y3)], layout=go.Layout(title_text="Correcto!"))]
)

fig.add_trace(go.Scatter(x=[-1,-1,1], y=[-1,1,-1], marker_symbol='diamond-open-dot', marker_size=15, mode='markers', name='Datos'))
fig.add_trace(go.Scatter(x=[1], y=[1], marker_symbol='hexagon-open-dot', marker_size=15, mode='markers', fillcolor='aliceblue', name='Datos'))

fig.show()
