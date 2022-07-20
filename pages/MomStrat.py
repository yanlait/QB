# import all the modules
from datetime import date

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from dash import dcc, html
from plotly.subplots import make_subplots

from assets import MomVectorBacktester as Mom

# Initiate the App

dash.register_page(__name__, path='/ac')
# Read the files
sc = 10000
mombt = Mom.MomVectorBacktester("TSLA",'2020-01-01', '2022-06-01',sc,0)
mombt.run_strategy(momentum=1)
data = mombt.get_data()
res = mombt.get_results()

# Read available stocks list
url = "https://drive.google.com/uc?id=17PC74DXlz0dlKvf04Mjv6xEh7wK7Mtt2"
path = "assets/Yftickers.csv"
stocks = pd.read_csv(path, sep=";")
stocks["Label"] = stocks["Ticker"] + " | " + stocks["Name"]
stocks.rename(columns={'Label': "label", 'Ticker': "value"}, inplace=True)
std = stocks[['label', 'value']].to_dict("records")


# Make plots functions
def plotreturns(asset, data, momentum, tc):
    title = 'Gross performance compared to the Mom-based strategy<br><sup> %s | TC = %.4f | Momentum = %.f' % (asset, tc, momentum)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["cstrategy"],
                             mode='lines',
                             name='Strategy'))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["creturns"],
                             mode='lines',
                             name='Buy/Hold'))

    fig.update_layout(title=title, legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig

def plothist(data):
    fighist = px.histogram(data["strategy"], x="strategy", marginal="box", title="Returns distribution")
    return fighist

def sparkline(data):
    #fig = px.line(data, height=200, width=200)
    fig = go.Figure(data=go.Scatter(x=data["price"], y=data["Date"], mode='lines'))
    # hide and lock down axes
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)

    # remove facet/subplot labels
    fig.update_layout(annotations=[], overwrite=True)

    # strip down the rest of the plot
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(t=20, l=20, b=20, r=20)
    )
# config=dict(displayModeBar=False)
    return fig

def plotfin(data, sdate, edate, sc):
    SP = yf.Ticker("SPY").history(start=sdate, end=edate, actions=False)

    SPReturn = (((SP["Close"].iloc[-1] - SP["Close"].iloc[0]) / SP["Close"].iloc[0]) + 1) * sc
    Assreturn = data["creturns"].iloc[-1]
    Strreturn = data["cstrategy"].iloc[-1]
    fg = go.Figure()
    fg.add_trace(go.Indicator(
        mode="number",
        value=Assreturn,
        title={"text": "Asset"},
        domain={'x': [0, 1], 'y': [0.55, 1]}
    ))
    fg.add_trace(go.Indicator(
        mode="number+delta",
        value=Strreturn,
        title={"text": "Strategy"},
        delta={'reference': Assreturn, 'relative': True, "valueformat": ".000%"},
        domain={'x': [0, 0.4], 'y': [0, 0.45]}
    ))
    fg.add_trace(go.Indicator(
        mode="number+delta",
        value=SPReturn,
        title={"text": "S&P 500"},
        delta={'reference': Assreturn, 'relative': True, "valueformat": ".000%"},
        domain={'x': [0.6, 1], 'y': [0, 0.45]}
    ))
    fg.update_layout(title_text='Returns')
    return fg

def plotfspark(data, asset, sdate, edate, sc):
    SP = yf.Ticker("SPY").history(start=sdate, end=edate, actions=False).reset_index()
    SP["price"] = SP["Close"]
    fig = make_subplots(rows=3, cols=2)
    """Numbers"""
    SPReturn = (((SP["Close"].iloc[-1] - SP["Close"].iloc[0]) / SP["Close"].iloc[0]) + 1) * sc
    Assreturn = data["creturns"].iloc[-1]
    Strreturn = data["cstrategy"].iloc[-1]

    """Sparkline"""
    SPline = sparkline(SP)
    SPline.update_layout(domain={'row': 1, 'column': 1})
    Assline = sparkline(data[["creturns","Date"]].rename(columns={"creturns": "price"}))
    Assline.update_layout(domain={'row': 2, 'column': 1})
    Strline = sparkline(data[["cstrategy","Date"]].rename(columns={"cstrategy": "price"}))
    Strline.update_layout(domain={'row': 3, 'column': 1})

    """Name"""
    SPname = "SPY"
    Assname = asset
    Strname = "Strategy"

    """Indicator"""
    SPind = go.Indicator(
        mode="number+delta",
        value=SPReturn,
        number={'prefix': "$"},
        title={"text": SPname},
        delta={'position': "top", 'reference': Assreturn}
        )
    Assind = go.Indicator(
        mode="number+delta",
        value=Assreturn,
        number={'prefix': "$"},
        title={"text": Assname},
        delta={'position': "top", 'reference': Assreturn}
        )
    Strind = go.Indicator(
        mode="number+delta",
        value=Strreturn,
        number={'prefix': "$"},
        title={"text": Strname},
        delta={'position': "top", 'reference': Assreturn}
        )

    fig.add_trace(SPind, row=1, col=2)
    fig.add_trace(Assind, row=2, col=2)
    fig.add_trace(Strind, row=3, col=2)
    fig.add_trace(SPline)
    fig.add_trace(Assline)
    fig.add_trace(Strline)
    #fig.update_layout(grid={'rows': 3, 'columns': 2})
    return fig

def plotfs(res, asset):

    #title = '%s | TC = %.4f' % (asset, 0)
    title = '%s' % (asset)
    to_plot = []
    to_ret = ["return"]
    fs = go.Figure()
    for m in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        res['position_%d' % m] = np.sign(res['return'].rolling(m).mean())
        res['momentum=%d' % m] = (res['position_%d' % m].shift(1) * res['return'])
        to_plot.append('momentum=%d' % m)

    all = to_ret + to_plot
    resfig = res[all].dropna().cumsum().apply(np.exp)
    resfig["Date"] = res["Date"]
    maxreturn = resfig[to_plot].tail(1).squeeze(axis=0).max()
    maxname = resfig[to_plot].tail(1).squeeze(axis=0).idxmax()
    maxmomentum = int(list(filter(str.isdigit, maxname))[0])
    to_grey = list(filter(lambda x: x != maxname, to_plot))

    for i in range(len(to_grey)):
        fs.add_trace(go.Scatter(x=resfig["Date"], y=resfig[to_grey[i]],
                                mode='lines', line=dict(color='#636EFA', width=1), opacity=.4, name=to_grey[i]))
    fs.add_trace(go.Scatter(x=resfig["Date"], y=resfig["return"],
                            mode='lines', name="asset", line_color="#FFA15A"))
    fs.add_trace(go.Scatter(x=resfig["Date"], y=resfig[maxname],
                            mode='lines', line_color="#00CC96", name=maxname))
    fs.update_layout(title=title)

    return fs, maxmomentum

# Make default plots


fig = plotreturns("TSLA", res, 1, 0)
fighist = plothist(res)
fs, mm = plotfs(res, "TSLA")
rfig = plotfin(res, '2020-01-01', '2022-06-01', sc)
#testsp= plotfspark(res, "TSLA", '2020-01-01', '2022-06-01', sc)
# Build the Components
"""Header_component = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Example", href="#")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("SMA", href="#"),
                dbc.DropdownMenuItem("Coming soon", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="Strategies",
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Telegram", href="https://t.me/yaanison"),
                dbc.DropdownMenuItem("Linkedin", href="https://www.linkedin.com/in/yan-laitila/"),
            ],
            nav=True,
            in_navbar=True,
            label="Contacts",
        ),
    ],
    brand="SMA based strategy",
    brand_href="#",
    color="primary",
    dark=True,
)"""

# Design the app layout
layout = html.Div(children=
[   dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output-1a", style={
            'position': 'absolute',
            'left': '50%',
            'top': '50%',
            'transform': 'translate(-50%, -50%)'
        })
    ),
    dbc.Row([dbc.Col(["Category"], width=3), dbc.Col(["Asset"], width=3),
             dbc.Col(["Date"], width=3), dbc.Col(["TC"], width=1),
             dbc.Col(["Starting capital"], width=1),
             dbc.Col([dbc.Row([""])], width=1)]
            ),
    dbc.Row([dbc.Col([dcc.Dropdown(options=stocks["Category"].unique(),
                                   id='category-dropdowna',
                                   value="Stocks",
                                   style={})], width=3),
             dbc.Col([dcc.Dropdown(options=std,
                                   value="TSLA",
                                   id='assets-dropdowna', style={})], width=3),
             dbc.Col([dcc.DatePickerRange(
                 id='date-picker-rangea',
                 display_format='DD/MM/YYYY',
                 start_date_placeholder_text="Start Date",
                 end_date_placeholder_text="End Date",
                 min_date_allowed=date(2020, 1, 1),
                 max_date_allowed=date(2022, 6, 1),
                 start_date=date(2021, 1, 1),
                 end_date=date(2022, 1, 1), style={})], width=3),
             dbc.Col([dcc.Input(id="TCinput", type="number", value=0, min=0, max=5, step=0.001, style={})], width=1),
             dbc.Col([dcc.Input(id="SCinput", type="number", value=10000, min=1, max=9999999999, step=1, style={})], width=1)]),
    dbc.Row([dbc.Col([dcc.Graph(figure=fs, id="strata")], width=8),
             dbc.Col([dcc.Graph(figure=fighist, id="shista")], width=4)],
            style={'padding-left': 10, 'padding-right': 10, 'flex': 1}),
    dbc.Row([dbc.Col([dcc.Graph(figure=fig, id="splota")], width=8),
             dbc.Col([dcc.Graph(figure=rfig, id="testsp")], width=4)]),
dbc.Tooltip(
            "Noun: rare, "
            "the action or habit of estimating something as worthless.",
            target="TCinput",
        )
])



@dash.callback(
    dash.dependencies.Output('assets-dropdowna', 'options'),
    dash.dependencies.Output('assets-dropdowna', 'value'),
    [dash.dependencies.Input('category-dropdowna', 'value')])
def assets_options(category):
    url = "https://drive.google.com/uc?id=17PC74DXlz0dlKvf04Mjv6xEh7wK7Mtt2"
    path = "assets/Yftickers.csv"
    stocks = pd.read_csv(path, sep=";")
    stocks = stocks[stocks["Category"] == category]
    stocks["Label"] = stocks["Ticker"] + " | " + stocks["Name"]
    stocks.rename(columns={'Label': "label", 'Ticker': "value"}, inplace=True)
    std = stocks[['label', 'value']].to_dict("records")
    if category == "Stocks":
      defass = "TSLA"
    else:
      defass = "BTC-USD"
    return std, defass

@dash.callback(
    dash.dependencies.Output('date-picker-rangea', 'min_date_allowed'),
    dash.dependencies.Output('date-picker-rangea', 'max_date_allowed'),
    [dash.dependencies.Input('assets-dropdowna', 'value')])
def assets_date_range(asset):
    raw = yf.Ticker(asset).history(period="max", actions=False)
    raw = raw.reset_index()
    mind = raw["Date"].min()
    maxd = raw["Date"].max()
    return mind, maxd


@dash.callback(
    dash.dependencies.Output('splota', 'figure'),
    dash.dependencies.Output('shista', 'figure'),
    dash.dependencies.Output('strata', 'figure'),
    dash.dependencies.Output('testsp', 'figure'),
    dash.dependencies.Output("loading-output-1a", "children"),
    [dash.dependencies.Input('assets-dropdowna', 'value'),
     dash.dependencies.Input('TCinput', 'value'),
     dash.dependencies.Input('SCinput', 'value'),
     dash.dependencies.Input('date-picker-rangea', 'start_date'),
     dash.dependencies.Input('date-picker-rangea', 'end_date')])
def upd_chart(asset, tc, sc, sdate, edate):
    load = html.Div(id="loading-output-1", style={'position': 'absolute', 'left': '50%', 'top': '50%', 'transform': 'translate(-50%, -50%)'})

    mombt = Mom.MomVectorBacktester(asset, sdate, edate, sc, tc)

    data = mombt.get_data()

    fs, maxmomentum = plotfs(data, asset)
    mombt.run_strategy(momentum=maxmomentum)
    res = mombt.get_results()

    fighist = plothist(res)

    rfig =plotfin(res, sdate, edate ,sc)
    fig = plotreturns(asset, res, maxmomentum, tc)
    return fig, fighist, fs, rfig, load

