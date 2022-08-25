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
from dash import dcc, html

from assets import SMAVectorBacktester as SMA

# Assign page name

dash.register_page(__name__, path='/SMA')

# Read stocks list

path = "assets/Yftickers.csv"
stocks = pd.read_csv(path, sep=";")
stocks["Label"] = stocks["Ticker"] + " | " + stocks["Name"]
stocks.rename(columns={'Label': "label", 'Ticker': "value"}, inplace=True)
std = stocks[['label', 'value']].to_dict("records")

# Make loading plot

loading_plot = px.bar(x=[0], y=[0])
loading_plot.update_yaxes(showticklabels=False, title="")
loading_plot.update_xaxes(showticklabels=False, title="")

fig = loading_plot
fighist = loading_plot
fs = loading_plot
rfig = loading_plot


# Make plots functions

def plotreturns(data,asset,sma1,sma2, tc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["cstrategy"],
                             mode='lines', line=dict(color="#FF7F0E"),
                             name='Strategy'))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["creturns"],
                             mode='lines', line=dict(color='#656EF1'),
                             name='Buy/Hold'))
    fig.update_layout(title="Gross performance compared to the SMA-based strategy<br><sup> %s | TC =%.1f | SMA 1=%d | SMA 2=%d </sup>" % (asset, tc, sma1, sma2), legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig


def plothist(data):
    fighist = px.histogram(data['returns in %'], x='returns in %', marginal="box", title="Returns distribution")
    return fighist


def plotr(data, asset, sdate, edate,sc):
    SP = yf.Ticker("SPY").history(start=sdate, end=edate, actions=False)
    SPReturn = (((SP["Close"][-1] - SP["Close"][0]) / SP["Close"][0]) + 1)* sc
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


def plotfs(data):
    fs = go.Figure()
    title = "Price chart with SMA's and buy/sell positions"
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


# Make plots functions
sidebar_full = {"position": "fixed",
                "left": "0vw",
                "height": "100%",
                "bottom": "auto",
                "padding": "0.5rem 1rem",
                "background-color": "#f8f9fa",
                "transition": "all 0.2s",
                "width": "25vw"}

sidebar_hide = {"position": "fixed",
                "left": "-25vw",
                "height": "100%",
                "bottom": "auto",
                "padding": "0rem 0rem",
                "background-color": "#f8f9fa",
                "transition": "all 0.2s",
                "width": "25vw"}

content_full = {"width": "100vw", "transition": "all 0.2s"}
content_hide = {"width": "70vw", "transition": "all 0.2s"}

# Make navbar

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="assets/images/QB_BW.png",
                                         height="50px")),
                        dbc.Col(dbc.NavbarBrand("Quantboard", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="/",
                style={"textDecoration": "none"},
            ),

            dbc.Nav([
                dbc.Button(children=[html.I(className="bi bi-sliders me-2"),
                                     "Strategy settings"], outline=False, size="sm", color="dark",
                           id="btn_sidebar_sma",
                           style={'padding-top': '6px', 'padding-bottom': '6px', 'margin-right': '30px'}),
                dbc.Button(children=[html.I(className="bi bi-info-square me-2"),
                                     "About strategy"], outline=False, size="sm", color="secondary",
                           id="btn_about",
                           style={'padding-top': '6px', 'padding-bottom': '6px', 'margin-right': '60px'}),
                dbc.NavItem(dbc.NavLink("Home", href="/")),
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem("SMA", href='/SMA'),
                        dbc.DropdownMenuItem("Momentum", href='/Momentum'),
                        dbc.DropdownMenuItem("Mean Reversion", href='/MeanReversion'),
                        dbc.DropdownMenuItem("Linear Regression", href="/LinearRegression"),
                        dbc.DropdownMenuItem("Machine Learning", href="/LogisticRegression"),
                        dbc.DropdownMenuItem("Deep learning", href="/DeepLearning"),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Strategies"
                ),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem(children=[html.I(className="bi bi-telegram me-2"),
                                                       "Telegram"], href="https://t.me/yaanison"),
                        dbc.DropdownMenuItem(children=[html.I(className="bi bi-linkedin me-2"),
                                                       "Linkedin"], href="https://www.linkedin.com/in/yan-laitila/"),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Contacts"
                )]),

        ]
    ),
    color="dark",
    dark=True
)

input_groups = html.Div(
    [
        dbc.InputGroup(
            [dbc.InputGroupText("Category", id="category_header_sma"), dbc.Select(options=[
                {"label": "Stocks", "value": "Stocks"},
                {"label": "Crypto", "value": "Crypto"}],
                id='category-dropdown_sma',
                value="Stocks")],
            className="mb-3",
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("Asset", id="asset_header_sma"), dbc.Select(options=std,
                                                                            value="TSLA",
                                                                            id='assets-dropdown_sma')],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Date", id="date_header_sma"),
                dcc.DatePickerRange(
                    id='date-picker-range_sma',
                    display_format='DD/MM/YYYY',
                    start_date_placeholder_text="Start Date",
                    end_date_placeholder_text="End Date",
                    min_date_allowed=date(2020, 1, 1),
                    max_date_allowed=date(2022, 6, 1),
                    start_date=date(2021, 1, 1),
                    end_date=date(2022, 1, 1)),

            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("SMA1", id="sma1_header_sma"),
             dbc.Input(id="SMA1_range", type="number", value=10, min=1, max=100, step=1, style={})
             ],
            className="mb-3"
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("SMA2", id="sma2_header_sma"),
             dbc.Input(id="SMA2_range", type="number", value=20, min=1, max=100, step=1, style={})
             ],
            className="mb-3"
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Transaction cost",id="tc_header_sma"),
                dbc.Input(id="TCinput_sma", type="number", value=0, min=0, max=5, step=0.1, style={}),
                dbc.InputGroupText("%"),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("Start capital", id="sc_header_sma"),
             dbc.Input(id="SCinput_sma", type="number", value=10000, min=1, max=9999999999, step=1, style={}),
             dbc.InputGroupText("$")
             ],
            className="mb-3"
        ),
        dbc.InputGroup([dbc.InputGroupText("Optimise", id="opt_header_sma"), dbc.InputGroupText(
            daq.BooleanSwitch(id='Opt_switch', color="black", labelPosition="left", on=False))])
    ]
)
dashboard = html.Div(children=
                     [dbc.Row([dbc.Col([dcc.Graph(figure=fs, id="strat")], width=8),
                               dbc.Col([dcc.Graph(figure=fighist, id="shist")], width=4)],
                              style={'padding-left': 10, 'padding-right': 10, 'flex': 1}),
                      dbc.Row([dbc.Col([dcc.Graph(figure=fig, id="splot")], width=8),
                               dbc.Col([dcc.Graph(figure=rfig, id='results')], width=4)]),
                      dbc.Modal(
                          [
                              dbc.ModalHeader(dbc.ModalTitle("About strategy")),
                              dbc.ModalBody(children=[
                                  html.H5("Strategy idea", className="card-title"),
                                  html.Hr(),
                                  html.P("The strategy is based on golden cross and a death cross patterns. Golden cross is a chart pattern in which a short-term SMA crosses above a long-term SMA. Death cross is a chart pattern in which a long-term SMA crosses above a short-term SMA. "),
                                  html.P( "SMA (simple moving average) is a major tool in the so-called technical analysis of stock prices. It calculates the average of a selected range of prices."),
                                  html.P("For example, if 10-day SMA crosses above the 20-day SMA, the strategy generates a buy signal. When selecting different pairs of SMAs, the profitability of the strategy will change."),

                                  html.Br(),

                                  html.H5("Parameters", className="card-title"),
                                  html.Hr(),
                                  html.Li("Date"),
                                  html.P("The date range in which the strategy is backtested"),
                                  html.Li("SMA1"),
                                  html.P("The length of the range (in days) for which the SMA is calculated. You can entrer long-term or short-term SMA in this field."),
                                  html.Li("SMA2"),
                                  html.P("Similar to SMA1. The value of SMA 2 must be different from SMA1."),
                                  html.Li("Transaction cost"),
                                  html.P("Proportional transaction costs per trade in %"),
                                  html.Li("Start capital"),
                                  html.P("The amount to invest in $"),
                                  html.Li("Optimise"),
                                  html.P("This function optimizes the strategy by brute force iterating through SMA pairs and returns the most profitable strategy settings."),
                                  html.Br(),

                                  html.H5("Charts explanation", className="card-title"),
                                  html.Hr(),
                                  html.Li("Price chart with SMAs and buy/sell positions"),
                                  html.P("Shows asset and SMA lines, buy and sell strategy signals."),
                                  html.Li("Gross performance compared to the SMA-based strategy"),
                                  html.P("Shows cash flow of asset and strategy. If in the end of the chart the strategy line is lower then the asset line, your strategy is unprofitable and vice versa."),
                                  html.Li("Returns distribution"),
                                  html.P("Shows the distribution of returns of the strategy from day to day. If the distribution has a left skew, the strategy more frequently brings losses"),
                                  html.Li("Returns"),
                                  html.P("Asset shows the amount of capital as if you were buying and holding an asset over the entire time period. Strategy shows the amount of capital as if you followed the strategy. S&P500 is a benchmark index."),
                                  html.Br(),
                              ]),
                              dbc.ModalFooter(
                                  dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)
                              ),
                          ],
                          id="modal_sma",
                          scrollable=True,
                          fullscreen=False,
                          is_open=False,
                      ), dbc.Popover(
                         "Choose an asset category",
                         target="category_header_sma",
                         body=True,
                         trigger="hover",
                     ),
                      dbc.Popover(
                          "Choose an asset from the TOP 200 list sorted by capitalization",
                          target="asset_header_sma",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Choose start and end dates for backtesting",
                          target="date_header_sma",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Enter the transaction cost value per trade",
                          target="tc_header_sma",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Enter the amount of capital that the strategy will work with",
                          target="sc_header_sma",
                          body=True,
                          trigger="hover",
                      ),

                      dbc.Popover(
                          "Enter the number of days to calculate the SMA",
                          target="sma1_header_sma",
                          body=True,
                          trigger="hover",
                      )
                         ,
                      dbc.Popover(
                          "Enter the number of days to calculate the SMA",
                          target="sma2_header_sma",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "The switch activates the algorithm for selecting the most profitable SMA values",
                          target="opt_header_sma",
                          body=True,
                          trigger="hover",
                      )
                      ], id="page-content_sma", style=content_full)

sidebar_sma = html.Div(
    [
        html.H2("SMA", className="display-4"),
        html.Hr(),
        html.P(
            "Strategy parameters", className="lead"
        ),
        input_groups,
    ], style=sidebar_hide, id="sidebar_sma"
)

# Design the app layout

layout = html.Div(children=
[dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Warning!")),
        dbc.ModalBody("SMA1 must be less than SMA2"),
    ],
    id="modal",
    is_open=False,
), dcc.Store(id='side_click_sma'), navbar, dbc.Row([dbc.Col(sidebar_sma), dbc.Col(dashboard)])]), html.Div(dcc.Loading(
    id="loading",
    children=[html.Div([html.Div(id="loading-output_sma")])],
    type="cube",
    color="#7F7F7F"),
    style={'position': 'absolute',
           'top': '50%',
           'left': '50%',
           'transform': 'translate(-50%,-50%)'})


@dash.callback(
    dash.dependencies.Output('assets-dropdown_sma', 'options'),
    dash.dependencies.Output('assets-dropdown_sma', 'value'),
    [dash.dependencies.Input('category-dropdown_sma', 'value')])
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
    dash.dependencies.Output("sidebar_sma", "style"),
    dash.dependencies.Output("page-content_sma", "style"),
    dash.dependencies.Output("side_click_sma", "data"),
    dash.dependencies.Output("btn_sidebar_sma", "children"),
    [dash.dependencies.Input("btn_sidebar_sma", "n_clicks")],

    [dash.dependencies.State("side_click_sma", "data")]
)
def toggle_sidebar_sma(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = sidebar_hide
            content_style = content_full
            cur_nclick = "HIDDEN"
            children = [html.I(className="bi bi-sliders me-2"),
                        "Strategy settings"]

        else:
            sidebar_style = sidebar_full
            content_style = content_hide
            cur_nclick = "SHOW"
            children = [html.I(className="bi bi-layout-sidebar-inset me-2"),
                        "Hide sidebar"]
    else:
        sidebar_style = sidebar_full
        content_style = content_hide
        cur_nclick = 'SHOW'
        children = [html.I(className="bi bi-layout-sidebar-inset me-2"),
                    "Hide sidebar"]

    return sidebar_style, content_style, cur_nclick, children


@dash.callback(
    dash.dependencies.Output("modal_sma", "is_open"),
    [dash.dependencies.Input("btn_about", "n_clicks"), dash.dependencies.Input("close", "n_clicks")],
    [dash.dependencies.State("modal_sma", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


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
    dash.dependencies.Output('date-picker-range_sma', 'min_date_allowed'),
    dash.dependencies.Output('date-picker-range_sma', 'max_date_allowed'),
    dash.dependencies.Output("loading-output_sma", "children"),
    [dash.dependencies.Input('assets-dropdown_sma', 'value'),
     dash.dependencies.Input('date-picker-range_sma', 'start_date'),
     dash.dependencies.Input('date-picker-range_sma', 'end_date'),
     dash.dependencies.Input('SMA1_range', 'value'),
     dash.dependencies.Input('SMA2_range', 'value'),
     dash.dependencies.Input('Opt_switch', 'on'),
     dash.dependencies.Input('TCinput_sma', 'value'),
     dash.dependencies.Input('SCinput_sma', 'value'),
     ])
def upd_chart(asset, sdate, edate, SMA1, SMA2, switch, tc, sc):
    load = []

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

    smabt = SMA.SMAVectorBacktester(asset, SMA1, SMA2, sdate, edate, sc, tc / 100)
    raw = smabt.get_raw().reset_index()
    mind = raw["Date"].min()
    maxd = raw["Date"].max()

    if switch:
        resopt = smabt.optimize_parameters((1, round(sma2max / 2), math.ceil(sma2max / 73)),
                                           (1, round(sma2max / 2), math.ceil(sma2max / 73)))
        SMA1 = resopt[0][0]
        SMA2 = resopt[0][1]
        switch = False
    else:
        smabt.run_strategy()
    data = smabt.get_results()

    rfig = plotr(data, asset, sdate, edate,sc)
    fighist = plothist(data)
    fs = plotfs(data)
    fig = plotreturns(data,asset, SMA1, SMA2,tc)

    return fig, fighist, fs, sma1max, sma2max, isopen, rfig, SMA1, SMA2, switch, mind, maxd, load
