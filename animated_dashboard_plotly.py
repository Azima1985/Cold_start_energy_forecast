import pandas as pd
import plotly as py
import numpy as np

df = pd.read_csv('D:\Data\Schneider\consumption_train.csv', index_col=0)
df.series_id.unique()

#data (list of traces)
trace1 = dict(
    type = 'scatter',
    x = np.arange(672),
    y = df[df.series_id==100003]['consumption'],
    mode = 'lines',
    #marker = dict(color = 'r'),
    line = dict(shape = 'spline', smoothing = 0.2,color ='red'),
    name = '1',
    showlegend = True
)

trace2 = dict(
    type = 'scatter',
    x = np.arange(672),
    y = df[df.series_id==100684]['consumption'],
    mode = 'lines',
    #marker = dict(color = 'b'),
    line = dict(shape='spline',smoothing=0.2,color = 'blue'),
    name = '2',
    showlegend = True
)

trace3 = dict(
    type = 'scatter',
    x = np.arange(672),
    y = df[df.series_id==101959]['consumption'],
    mode = 'lines',
    #marker = dict(color = 'b'),
    line = dict(shape='spline',smoothing=0.2,color = 'blue'),
    name = '3',
    showlegend = True
)
trace4 = dict(
    type = 'scatter',
    x = np.arange(672),
    y = df[df.series_id==100304]['consumption'],
    mode = 'lines',
    #marker = dict(color = 'b'),
    line = dict(shape='spline',smoothing=0.2,color = 'blue'),
    name = '4',
    showlegend = True
)

trace5 = dict(
    type = 'scatter',
    x = np.arange(672),
    y = df[df.series_id==102564]['consumption'],
    mode = 'lines',
    #marker = dict(color = 'b'),
    line = dict(shape='spline',smoothing=0.2,color = 'blue'),
    name = '5',
    showlegend = True
)

trace6 = dict(
    type = 'scatter',
    x = np.arange(672),
    y = df[df.series_id==103605]['consumption'],
    mode = 'lines',
    #marker = dict(color = 'b'),
    line = dict(shape='spline',smoothing=0.2,color = 'blue'),
    name = '6',
    showlegend = True
)

trace7 = dict(
    type = 'scatter',
    x = np.arange(672),
    y = df[df.series_id==103466]['consumption'],
    mode = 'lines',
    #marker = dict(color = 'b'),
    line = dict(shape='spline',smoothing=0.2,color = 'blue'),
    name = '7',
    showlegend = True
)
data = [trace1]

#layout (title, x-labels, y-labels,..etc.)
layout = dict(
    title = '',
    xaxis = dict(title='Time steps',range=[0,700]),
    yaxis = dict(title='Energy consumption',range=[0,700000]),
    width = 1000,
    height = 500,
    updatemenus = [dict(type='buttons',
                   buttons = [{
                'args': [None, {'frame': {'duration': 2000},
                         'fromcurrent': True, 'transition': {'duration': 2000, 'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            }]
                      )]
)

#frames
frames =[
    dict(data=[{'x':np.arange(672),'y':df[df.series_id==100003]['consumption'],'mode':'lines','line':{'color':'red'},'name':'1'}]),
    dict(data=[{'x':np.arange(672),'y':df[df.series_id==100684]['consumption'],'mode':'lines','line':{'color':'blue'},'name':'2'}]),
    dict(data=[{'x':np.arange(672),'y':df[df.series_id==101959]['consumption'],'mode':'lines','line':{'color':'green'},'name':'3'}]),
    dict(data=[{'x':np.arange(672),'y':df[df.series_id==100304]['consumption'],'mode':'lines','line':{'color':'yellow'},'name':'4'}]),
    dict(data=[{'x':np.arange(672),'y':df[df.series_id==102564]['consumption'],'mode':'lines','line':{'color':'black'},'name':'5'}]),
    dict(data=[{'x':np.arange(672),'y':df[df.series_id==103605]['consumption'],'mode':'lines','line':{'color':'grey'},'name':'6'}]),
    dict(data=[{'x':np.arange(672),'y':df[df.series_id==103466]['consumption'],'mode':'lines','line':{'color':'purple'},'name':'7'}]),
]

#figure
fig = dict(
    data = data,
    layout = layout,
    frames = frames
)
py.offline.init_notebook_mode(connected=True)
py.offline.iplot(fig)