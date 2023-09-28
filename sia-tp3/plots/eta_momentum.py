import json
import plotly.graph_objects as go

greater_comparison_x = []
greater_comparison_y = []

xs=[]
ys=[]

for i in range(0,10):
    file = f"../results/test-no_momentum_no_changing_learning-{i}.json"
    data = json.load(open(file, mode='r'))
    xs.append([])
    ys.append([])
    for epoch_data in data["iterations"]:
        xs[i].append(epoch_data["epoch"])
        ys[i].append(epoch_data["error"])

fig = go.Figure()

[ fig.add_trace(go.Scatter(x=xs[i], y=ys[i], mode='lines', name=f"line {i}")) for i in range(0,10)]

# Edit the layout
fig.update_layout(title='Error Por Epoch No Momentum',
                   xaxis_title='Epoch',
                   yaxis_title='Error')

fig.show()
greater_comparison_x.append(xs[1])
greater_comparison_y.append(ys[1])

#NOW WITH MOMENTUM

mxs=[]
mys=[]

for i in range(0,5):
    file = f"../results/test-momentum_03-{i}.json"
    data = json.load(open(file, mode='r'))
    mxs.append([])
    mys.append([])
    for epoch_data in data["iterations"]:
        mxs[i].append(epoch_data["epoch"])
        mys[i].append(epoch_data["error"])

fig = go.Figure()

[ fig.add_trace(go.Scatter(x=mxs[i], y=mys[i], mode='lines', name=f"line {i}")) for i in range(0,5)]

# Edit the layout
fig.update_layout(title='Error Por Epoch Momentum 0.3',
                   xaxis_title='Epoch',
                   yaxis_title='Error')

greater_comparison_x.append(mxs[1])
greater_comparison_y.append(mys[1])

fig.show()

mxs=[]
mys=[]

for i in range(0,5):
    file = f"../results/test-momentum_05-{i}.json"
    data = json.load(open(file, mode='r'))
    mxs.append([])
    mys.append([])
    for epoch_data in data["iterations"]:
        mxs[i].append(epoch_data["epoch"])
        mys[i].append(epoch_data["error"])

fig = go.Figure()

[ fig.add_trace(go.Scatter(x=mxs[i], y=mys[i], mode='lines', name=f"line {i}")) for i in range(0,5)]

# Edit the layout
fig.update_layout(title='Error Por Epoch Momentum 0.5',
                   xaxis_title='Epoch',
                   yaxis_title='Error')

greater_comparison_x.append(mxs[1])
greater_comparison_y.append(mys[1])

fig.show()



mxs=[]
mys=[]

for i in range(0,5):
    file = f"../results/test-momentum_07-{i}.json"
    data = json.load(open(file, mode='r'))
    mxs.append([])
    mys.append([])
    for epoch_data in data["iterations"]:
        mxs[i].append(epoch_data["epoch"])
        mys[i].append(epoch_data["error"])

fig = go.Figure()

[ fig.add_trace(go.Scatter(x=mxs[i], y=mys[i], mode='lines', name=f"line {i}")) for i in range(0,5)]

# Edit the layout
fig.update_layout(title='Error Por Epoch Momentum 0.7',
                   xaxis_title='Epoch',
                   yaxis_title='Error')

greater_comparison_x.append(mxs[1])
greater_comparison_y.append(mys[1])

fig.show()

mxs=[]
mys=[]

for i in range(0,5):
    file = f"../results/test-momentum_09-{i}.json"
    data = json.load(open(file, mode='r'))
    mxs.append([])
    mys.append([])
    for epoch_data in data["iterations"]:
        mxs[i].append(epoch_data["epoch"])
        mys[i].append(epoch_data["error"])

fig = go.Figure()

[ fig.add_trace(go.Scatter(x=mxs[i], y=mys[i], mode='lines', name=f"line {i}")) for i in range(0,5)]

# Edit the layout
fig.update_layout(title='Error Por Epoch Momentum 0.9',
                   xaxis_title='Epoch',
                   yaxis_title='Error')

greater_comparison_x.append(mxs[1])
greater_comparison_y.append(mys[1])

fig.show()



fig = go.Figure()

comments = ['No Momentum', 'Momentum 0.3', 'Momentum 0.5', 'Momentum 0.7', 'Momentum 0.9']

[ fig.add_trace(go.Scatter(x=greater_comparison_x[i], y=greater_comparison_y[i], mode='lines', name=comments[i])) for i in range(0,5)]

# Edit the layout
fig.update_layout(title='Error Por Epoch Combinación Distintos Momentum',
                   xaxis_title='Epoch',
                   yaxis_title='Error')

fig.show()





# NOW WITH VARIABLE ETA


exs=[]
eys=[]

for i in range(0,5):
    file = f"../results/test-changing_learning_zero_one-{i}.json"
    data = json.load(open(file, mode='r'))
    exs.append([])
    eys.append([])
    for epoch_data in data["iterations"]:
        exs[i].append(epoch_data["epoch"])
        eys[i].append(epoch_data["error"])

fig = go.Figure()

[ fig.add_trace(go.Scatter(x=exs[i], y=eys[i], mode='lines', name=f"line {i}")) for i in range(0,5)]

# Edit the layout
fig.update_layout(title='Error Por Epoch eta Adaptativo 0.1',
                   xaxis_title='Epoch',
                   yaxis_title='Error')

greater_comparison_x.append(exs[2])
greater_comparison_y.append(eys[2])

fig.show()



exs=[]
eys=[]

for i in range(0,5):
    file = f"../results/test-changing_learning_zero_zero_two-{i}.json"
    data = json.load(open(file, mode='r'))
    exs.append([])
    eys.append([])
    for epoch_data in data["iterations"]:
        exs[i].append(epoch_data["epoch"])
        eys[i].append(epoch_data["error"])

fig = go.Figure()

[ fig.add_trace(go.Scatter(x=exs[i], y=eys[i], mode='lines', name=f"line {i}")) for i in range(0,5)]

# Edit the layout
fig.update_layout(title='Error Por Epoch eta Adaptativo 0.002',
                   xaxis_title='Epoch',
                   yaxis_title='Error')

greater_comparison_x.append(exs[2])
greater_comparison_y.append(eys[2])

fig.show()




exs=[]
eys=[]

for i in range(0,5):
    file = f"../results/test-changing_learning_zero_zero_seis-{i}.json"
    data = json.load(open(file, mode='r'))
    exs.append([])
    eys.append([])
    for epoch_data in data["iterations"]:
        exs[i].append(epoch_data["epoch"])
        eys[i].append(epoch_data["error"])

fig = go.Figure()

[ fig.add_trace(go.Scatter(x=exs[i], y=eys[i], mode='lines', name=f"line {i}")) for i in range(0,5)]

# Edit the layout
fig.update_layout(title='Error Por Epoch eta Adaptativo 0.006',
                   xaxis_title='Epoch',
                   yaxis_title='Error')

greater_comparison_x.append(exs[2])
greater_comparison_y.append(eys[2])

fig.show()





fig = go.Figure()

comments = ['No Momentum', 'Momentum 0.3', 'Momentum 0.9', 'Eta 0.01', 'Eta 0.002', 'Eta 0.006']

greater_comparison_x = greater_comparison_x[:2] + greater_comparison_x[4:]
greater_comparison_y = greater_comparison_y[:2] + greater_comparison_y[4:]


[ fig.add_trace(go.Scatter(x=greater_comparison_x[i], y=greater_comparison_y[i], mode='lines', name=comments[i])) for i in range(0,6)]

# Edit the layout
fig.update_layout(title='Error Por Epoch Combinación Distintos Momentum y Eta',
                   xaxis_title='Epoch',
                   yaxis_title='Error')

fig.show()