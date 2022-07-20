# import all the modules
import math
from datetime import date

import ciso8601
import plotly.figure_factory as ff
import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from dash import Dash, dcc, html, Input, Output, callback

from assets import SMAVectorBacktester as SMA

# Initiate the App



"""app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP],
                update_title='Calculating, quanting, plotting...')"""

dash.register_page(__name__, path='/a')
# Read the files
smabt = SMA.SMAVectorBacktester("TSLA", 1, 2, '2020-01-01', '2022-06-01')
smabt.run_strategy()
data = smabt.get_results()

# Read available stocks list
url = "https://drive.google.com/uc?id=17PC74DXlz0dlKvf04Mjv6xEh7wK7Mtt2"
path = "/Users/yanlaytila/Desktop/Yftickers.csv"
stocks = pd.read_csv(path, sep=";")
stocks["Label"] = stocks["Ticker"] + " | " + stocks["Name"]
stocks.rename(columns={'Label': "label", 'Ticker': "value"}, inplace=True)
std = stocks[['label', 'value']].to_dict("records")


# Make plots functions
def plotreturns(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["cstrategy"],
                             mode='lines',
                             name='Strategy'))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["creturns"],
                             mode='lines',
                             name='Buy/Hold'))
    fig.update_layout(title="Gross performance compared to the SMA-based strategy", legend=dict(
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

def plotr(data, asset, sdate, edate):
    SP = yf.Ticker("SPY").history(start=sdate, end=edate, actions=False)
    SPReturn = ((SP["Close"][-1] - SP["Close"][0]) / SP["Close"][0]) + 1
    rfig = go.Figure()
    rfig.add_trace(go.Indicator(
        mode="number",
        value=round(data["creturns"].iloc[-1], 2),
        title={"text": "Asset"},
        domain={'x': [0, 1], 'y': [0.55, 1]}))

    rfig.add_trace(go.Indicator(
        mode="number+delta",
        value=round(data["cstrategy"].iloc[-1], 2),
        title={"text": "Strategy"},
        delta={'reference': round(data["creturns"].iloc[-1], 2), 'relative': True, "valueformat": ".000%"},
        domain={'x': [0, 0.4], 'y': [0, 0.45]}
    ))

    rfig.add_trace(go.Indicator(
        mode="number+delta",
        value=round(SPReturn, 2),
        title={"text": "S&P 500"},
        delta={'reference': round(data["creturns"].iloc[-1], 2), 'relative': True, "valueformat": ".000%"},
        domain={'x': [0.6, 1], 'y': [0, 0.45]}))
    rfig.update_layout(title_text='Returns')
    return rfig

def plotfs(data, asset, SMA1, SMA2):

    fs = go.Figure()
    title = "Price chart with SMA's and buy/sell positions<br><sup> %s | SMA1=%d, SMA2=%d </sup>" % (asset, SMA1, SMA2)
    fs.add_trace(go.Scatter(x=data["Date"], y=data["price"],
                            mode='lines',
                            name='Chart'))
    fs.add_trace(go.Scatter(x=data["Date"], y=data['SMA1'],
                            mode='lines',
                            name='SMA1'))
    fs.add_trace(go.Scatter(x=data["Date"], y=data['SMA2'],
                            mode='lines',
                            name='SMA2'))
    fs.add_trace(
        go.Scatter(x=data["Date"][data.entry == 2], y=data['price'][data.entry == 2],
                   name='Buy', mode='markers', marker_symbol="triangle-up", marker=dict(size=10, color='Green')))
    fs.add_trace(
        go.Scatter(x=data["Date"][data.entry == -2], y=data['price'][data.entry == -2],
                   name='Sell', mode='markers', marker_symbol="triangle-down", marker=dict(size=10, color='Red')))

    fs.update_layout(title=title)
    fs.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fs

# Make default plots
fig = plotreturns(data)
fighist = plothist(data)
fs = plotfs(data, "TSLA", 10, 20)
rfig = plotr(data,"TSLA",date(2021, 1, 1),date(2022, 1, 1))

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
[dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Warning!")),
        dbc.ModalBody("SMA1 must be less than SMA2"),
    ],
    id="modal",
    is_open=False,
),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output-1", style={
            'position': 'absolute',
            'left': '50%',
            'top': '50%',
            'transform': 'translate(-50%, -50%)'
        })
    ),
    dbc.Row([dbc.Col(["Category"], width=3), dbc.Col(["Asset"], width=3),
             dbc.Col(["Date"], width=3), dbc.Col(["SMA1"], width=1),
             dbc.Col(["SMA2"], width=1),
             dbc.Col([dbc.Row(["Optimise SMA"]), dbc.Row([""], style={'font-size': 'smaller'})], width=1)],
            style={"background-color": "#ffffff", 'padding-left': 70, 'padding-right': 70, 'flex': 1}
            ),
    dbc.Row([dbc.Col([dcc.Dropdown(options=stocks["Category"].unique(),
                                   id='category-dropdown',
                                   value="Stocks",
                                   style={})], width=3),
             dbc.Col([dcc.Dropdown(options=std,
                                   value="TSLA",
                                   id='assets-dropdown', style={})], width=3),
             dbc.Col([dcc.DatePickerRange(
                 id='date-picker-range',
                 display_format='DD/MM/YYYY',
                 start_date_placeholder_text="Start Date",
                 end_date_placeholder_text="End Date",
                 min_date_allowed=date(2020, 1, 1),
                 max_date_allowed=date(2022, 6, 1),
                 start_date=date(2021, 1, 1),
                 end_date=date(2022, 1, 1), style={})], width=3),
             dbc.Col([dcc.Input(id="SMA1_range", type="number", value=10, min=1, max=100, step=1, style={})], width=1),
             dbc.Col([dcc.Input(id="SMA2_range", type="number", value=20, min=1, max=100, step=1, style={})], width=1),
             dbc.Col(daq.BooleanSwitch(id='Opt_switch', label="≈ 10 sec", labelPosition="top", on=False), width=1,
                     style={"background-color": "#ffffff", "float": "left"})],
            style={"background-color": "#ffffff", 'padding-left': 70, 'padding-right': 70, 'flex': 1}),
    dbc.Row([dbc.Col([dcc.Graph(figure=fs, id="strat")], width=8),
             dbc.Col([dcc.Graph(figure=fighist, id="shist")], width=4)],
            style={'padding-left': 10, 'padding-right': 10, 'flex': 1}),
    dbc.Row([dbc.Col([dcc.Graph(figure=fig, id="splot")], width=8),
             dbc.Col([dcc.Graph(figure=rfig, id='results')], width=4)],
            style={'padding-left': 10, 'padding-right': 10, 'flex': 1})
])


@dash.callback(
    dash.dependencies.Output('assets-dropdown', 'options'),
    dash.dependencies.Output('assets-dropdown', 'value'),
    [dash.dependencies.Input('category-dropdown', 'value')])
def assets_options(category):
    url = "https://drive.google.com/uc?id=17PC74DXlz0dlKvf04Mjv6xEh7wK7Mtt2"
    path = "/Users/yanlaytila/Desktop/Yftickers.csv"
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
    dash.dependencies.Output('date-picker-range', 'min_date_allowed'),
    dash.dependencies.Output('date-picker-range', 'max_date_allowed'),
    [dash.dependencies.Input('assets-dropdown', 'value')])
def assets_date_range(asset):
    raw = yf.Ticker(asset).history(period="max", actions=False)
    raw = raw.reset_index()
    mind = raw["Date"].min()
    maxd = raw["Date"].max()
    return mind, maxd


@dash.callback(
    dash.dependencies.Output('splot', 'figure'),
    dash.dependencies.Output('shist', 'figure'),
    dash.dependencies.Output('strat', 'figure'),
    dash.dependencies.Output('SMA1_range', "max"),
    dash.dependencies.Output('SMA2_range', "max"),
    dash.dependencies.Output('modal', 'is_open'),
    dash.dependencies.Output('results', 'figure'),
    dash.dependencies.Output('SMA1_range', 'value'),
    dash.dependencies.Output('SMA2_range', 'value'),
    dash.dependencies.Output('Opt_switch', 'on'),
    dash.dependencies.Output("loading-output-1", "children"),
    [dash.dependencies.Input('assets-dropdown', 'value'),
     dash.dependencies.Input('date-picker-range', 'start_date'),
     dash.dependencies.Input('date-picker-range', 'end_date'),
     dash.dependencies.Input('SMA1_range', 'value'),
     dash.dependencies.Input('SMA2_range', 'value'),
     dash.dependencies.Input('Opt_switch', 'on'),
     ])
def upd_chart(asset, sdate, edate, SMA1, SMA2, switch):
    load = html.Div(id="loading-output-1", style={'position': 'absolute', 'left': '50%', 'top': '50%', 'transform': 'translate(-50%, -50%)'})
    isopen = False
    diff = ciso8601.parse_datetime(edate) - ciso8601.parse_datetime(sdate)
    sma1max = int(diff.days)
    sma2max = int(diff.days)
    if SMA1 is None:
        SMA1 = 1
    if SMA2 is None:
        SMA2 = 2

    if (SMA1 or SMA2) > sma1max:
        isopen = True

    smabt = SMA.SMAVectorBacktester(asset, SMA1, SMA2, sdate, edate)
    if switch == True:
        resopt = smabt.optimize_parameters((1, round(sma2max / 2), math.ceil(sma2max / 73)),
                                           (1, round(sma2max / 2), math.ceil(sma2max / 73)))
        SMA1 = resopt[0][0]
        SMA2 = resopt[0][1]
        switch = False
    else:
        smabt.run_strategy()
    data = smabt.get_results()

    if (data[data.entry == 2].shape[0] + data[data.entry == -2].shape[0]) == 0:
        d = pd.DataFrame(np.zeros((0, 0)))
        fighist = px.histogram(d, marginal="box", title="Returns distribution (No strategy signals)")
        SP = yf.Ticker("SPY").history(start=sdate, end=edate, actions=False)
        SPReturn = ((SP["Close"][-1] - SP["Close"][0]) / SP["Close"][0]) + 1
        rfig = go.Figure()
        rfig.add_trace(go.Indicator(
            mode="number+delta",
            value=round(data["creturns"].iloc[-1], 2),
            title={"text": asset},
            delta={'reference': round(data["creturns"].iloc[-1], 2), 'relative': True, "valueformat": ".000%"},
            domain={'x': [0, 0.233], 'y': [0.3, 0.7]}))

        rfig.add_trace(go.Indicator(
            mode="number+delta",
            value=0,
            title={"text": "Strategy"},
            delta={'reference': 0, 'relative': True, "valueformat": ".000%"},
            domain={'x': [0.333, 0.566], 'y': [0.3, 0.7]}
        ))

        rfig.add_trace(go.Indicator(
            mode="number+delta",
            value=round(SPReturn, 2),
            title={"text": "S&P 500"},
            delta={'reference': round(data["creturns"].iloc[-1], 2), 'relative': True, "valueformat": ".000%"},
            domain={'x': [0.666, 0.9], 'y': [0.3, 0.7]}))
        rfig.update_layout(title_text='Returns')
    else:
        rfig = plotr(data,asset, sdate, edate)
        fighist = plothist(data)

    fs = plotfs(data, asset, SMA1, SMA2)
    fig = plotreturns(data)

    return fig, fighist, fs, sma1max, sma2max, isopen, rfig, SMA1, SMA2, switch, load

