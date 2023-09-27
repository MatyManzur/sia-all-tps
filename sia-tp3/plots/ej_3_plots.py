import json
import plotly.graph_objects as go
from src.metrics import accuracy, precision

x = []
y = []
accuracy_data_test = []
precision_data_test = []
accuracy_data_train = []
precision_data_train = []

data = json.load(open("../results/test-0_1_2_3_test-0.json", mode='r'))
for epoch_data in data["iterations"]:
    x.append(epoch_data["epoch"])
    y.append(epoch_data["error"])
    accuracy_data_test.append(accuracy(epoch_data["class_metrics_test"][0])) # Me quedo solo con la del 0
    precision_data_test.append(precision(epoch_data["class_metrics_test"][0]))
    accuracy_data_train.append(accuracy(epoch_data["class_metrics_train"][0])) # Me quedo solo con la del 0
    precision_data_train.append(precision(epoch_data["class_metrics_train"][0]))

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,mode='lines',name='lines'))
#fig.add_trace(go.Scatter(x=x, y=y,mode='lines+markers',name='lines+markers'))
#fig.add_trace(go.Scatter(x=x, y=y,mode='markers', name='markers'))

# Edit the layout
fig.update_layout(title='Error Por Epoch',
                   xaxis_title='Epoch',
                   yaxis_title='Error')


fig.show()

x = x[0:3000:1]
# Puse máximo 3000
accuracy_data_test = accuracy_data_test[0:3000:1]
accuracy_data_train = accuracy_data_train[0:3000:1]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=accuracy_data_test, mode='lines', name='test'))
fig.add_trace(go.Scatter(x=x, y=accuracy_data_train, mode='lines', name='train'))
fig.update_layout(title='Accuracy of Class 0 per Epoch',
                   xaxis_title='Epoch',
                   yaxis_title='Accuracy')
fig.show()

# Puse máximo 3000
precision_data_test = precision_data_test[0:3000:1]
precision_data_train = precision_data_train[0:3000:1]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=precision_data_test, mode='lines', name='test'))
fig.add_trace(go.Scatter(x=x, y=precision_data_train, mode='lines', name='train'))
fig.update_layout(title='Precision of Class 0 per Epoch',
                   xaxis_title='Epoch',
                   yaxis_title='Precision')
fig.show()