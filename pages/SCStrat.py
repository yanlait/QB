# import all the modules
from datetime import date

import ciso8601
import dash

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from dash import dcc, html

from plotly.subplots import make_subplots

from assets import ScikitVectorBacktester as SCI

# Assign page name

dash.register_page(__name__, path='/LogisticRegression')

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
main_color = '#656EF1'
str_color = '#FF7F0E'

# Make plots functions
def plotreturns(asset, data, tc, lag):
    title = 'Gross performance compared to the logistic regression strategy<br><sup> %s | TC = %.1f | Lag = %.f ' % (asset, tc, lag)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["cstrategy"],
                             mode='lines', line=dict(color=str_color),
                             name='Strategy'))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["creturns"],
                             mode='lines',  line=dict(color=main_color),
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
    fighist = px.histogram(data['returns in %'], x='returns in %', marginal="box", title="Returns distribution")
    return fighist


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


def plotfsn(data, res, asset, sdate_lrn, edate_lrn, sdate, edate):
    hits = np.sign(res["return"] * res["prediction"]).value_counts()

    title_c = "Correct <b>%.1f</b>" % (hits[1] / sum(hits) * 100)
    title_w = "Wrong <b>%.1f</b>" % ((1-hits[1] / sum(hits)) * 100)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2], subplot_titles=("", "Accuracy score"))

    fig.add_trace(go.Scatter(x=data["Date"], y=data["price"],
                             mode='lines',
                             name='Chart'), row=1, col=1)

    fig.add_trace(go.Bar(x=[(1 - (hits[1] / sum(hits))), (hits[1] / sum(hits))], y=[1, 1], orientation="h",
                         marker_color=["#EF553B", "#00CC96"], width=100, showlegend=False, text=([title_w, title_c]),
                         textposition='inside'), row=2, col=1)
    fig.add_vline(x=0.5, row=2, col=1, line_width=0.8)
    fig.update_xaxes(showticklabels=False, range=[0, 1], row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.add_vrect(
        x0=sdate_lrn, x1=edate_lrn,
        annotation_text="Learning", annotation_position='top left',
        fillcolor="#72B7B2", opacity=0.5,
        layer="below", line_width=0, row=1, col=1
    )
    fig.add_vrect(
        x0=sdate, x1=edate,
        annotation_text="Backtesting", annotation_position='bottom left',
        fillcolor="#BAB0AC", opacity=0.5,
        layer="below", line_width=0, row=1, col=1)

    fig.update_layout(title="Price chart & training/backtesting ranges", legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1))

    return fig


# Make sidebar style
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
                           id="btn_sidebar_sc",
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
            [dbc.InputGroupText("Category", id="category_header_sc"), dbc.Select(options=[
                {"label": "Stocks", "value": "Stocks"},
                {"label": "Crypto", "value": "Crypto"}],
                id='category-dropdown_sc',
                value="Stocks")],
            className="mb-3",
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("Asset", id="asset_header_sc"), dbc.Select(options=std,
                                                                           value="TSLA",
                                                                           id='assets-dropdown_sc')],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Training", id="datelrn_header_sc"),
                dcc.DatePickerRange(
                    id='date-picker-range_lrn_sc',
                    display_format='DD/MM/YYYY',
                    start_date_placeholder_text="Start Date",
                    end_date_placeholder_text="End Date",
                    min_date_allowed=date(2020, 1, 1),
                    max_date_allowed=date(2022, 6, 1),
                    start_date=date(2020, 1, 1),
                    end_date=date(2021, 1, 1)),

            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Backtesting", id="date_header_sc"),
                dcc.DatePickerRange(
                    id='date-picker-range_sc',
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
            [
                dbc.InputGroupText("Lag", id="lag_header_sc"),
                dbc.Input(id="Laginput_sc", type="number", value=10, min=1, max=15, step=1, style={})
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Transaction cost", id="tc_header_sc"),
                dbc.Input(id="TCinput_sc", type="number", value=0, min=0, max=5, step=0.1, style={}),
                dbc.InputGroupText("%"),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("Start capital", id="sc_header_sc"),
             dbc.Input(id="SCinput_sc", type="number", value=10000, min=1, max=9999999999, step=1, style={}),
             dbc.InputGroupText("$")
             ],
            className="mb-3"
        )

    ]
)
dashboard = html.Div(children=
                     [dbc.Row([dbc.Col(dcc.Graph(figure=fs, id="strat_sc"), width=8),
                               dbc.Col([dcc.Graph(figure=fighist, id="shist_sc")], width=4)],
                              style={'padding-left': 10, 'padding-right': 10, 'flex': 1}),
                      dbc.Row([dbc.Col([dcc.Graph(figure=fig, id="splot_sc")], width=8),
                               dbc.Col([dcc.Graph(figure=rfig, id="res_sc")], width=4)]),
                      dbc.Modal(
                          [
                              dbc.ModalHeader(dbc.ModalTitle("About strategy")),
                              dbc.ModalBody(children=[
                                  html.H5("Strategy idea", className="card-title"),
                                  html.Hr(),
                                  html.P("Strategy uses scikit-learn Logistic Regression algorithm to predict asset movement."),
                                  html.Br(),

                                  html.H5("Parameters", className="card-title"),
                                  html.Hr(),
                                  html.Li("Training"),
                                  html.P(
                                      "Date range in which the regression model is fitted (trained). The range may overlap with the Backtesting range."),
                                  html.Li("Backtesting"),
                                  html.P(
                                      "Date range in which the regression model is evaluated."),
                                  html.Li("Lag"),
                                  html.P("The lagging variable is used in regression analysis to provide robust estimates of the effects of independent variables. The value determines the number of days for which the lag will be created."),
                                  html.Li("Transaction cost"),
                                  html.P(
                                      "Proportional transaction costs per trade in %."),
                                  html.Li("Start capital"),
                                  html.P("The amount to invest in $."),
                                  html.Br(),

                                  html.H5("Charts explanation", className="card-title"),
                                  html.Hr(),
                                  html.Li("Price chart & learning/backtesting ranges"),
                                  html.P(
                                      "The upper part of the graph shows asset price, training and backtesting areas. The lower part of the graph shows the accuracy with which the model predicts the direction of price movement."),
                                  html.Li("Gross performance compared to the logical regression strategy"),
                                  html.P(
                                      "Shows cash flow of asset and strategy. If in the end of the chart strategy line is lower then asset line - your strategy is unprofitable and vice versa."),
                                  html.Li("Returns distribution"),
                                  html.P("Shows the distribution of returns of the strategy from day to day. If the distribution has a left skew, the strategy more frequently brings losses"),
                                  html.Li("Returns"),
                                  html.P(
                                      "Asset shows the amount of capital as if you were buying and holding an asset over the entire time period. Strategy shows the amount of capital as if you followed the strategy. S&P500 is a benchmark index."),
                                  html.Br(),

                              ]),
                              dbc.ModalFooter(
                                  dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)
                              ),
                          ],
                          id="modal_sc",
                          is_open=False,
                      ), dbc.Popover(
                         "Choose an asset category",
                         target="category_header_sc",
                         body=True,
                         trigger="hover",
                     ),
                      dbc.Popover(
                          "Choose an asset from the TOP 200 list sorted by capitalization",
                          target="asset_header_sc",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Choose start and end dates for model training",
                          target="datelrn_header_sc",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Choose start and end dates for backtesting",
                          target="date_header_sc",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Enter the transaction cost value per trade",
                          target="tc_header_sc",
                          body=True,
                          trigger="hover",
                      ),

                      dbc.Popover(
                          "Enter the amount of capital that the strategy will work with",
                          target="sc_header_sc",
                          body=True,
                          trigger="hover",
                      ),

                      dbc.Popover(
                          "A parameter that adds information about previous periods to the regression equation. The value determines the number of days for which the lag will be created.",
                          target="lag_header_sc",
                          body=True,
                          trigger="hover",
                      )
                      ], id="page-content_sc", style=content_full)

sidebar_sc = html.Div(
    [
        html.H2("Machine Learning", className="display-4"),
        html.H5("Logistic Regression", className="display-6"),
        html.Hr(),
        html.P(
            "Strategy parameters", className="lead"
        ),
        input_groups,
    ], style=sidebar_hide, id="sidebar_sc"
)

# Design the app layout
layout = html.Div(
    [dcc.Store(id='side_click_sc'), navbar, dbc.Row([dbc.Col(sidebar_sc), dbc.Col(dashboard)])]), html.Div(dcc.Loading(
    id="loading",
    children=[html.Div([html.Div(id="loading-output_sc")])],
    type="cube",
    color="#7F7F7F"),
    style={'position': 'absolute',
           'top': '50%',
           'left': '50%',
           'transform': 'translate(-50%,-50%)'})


@dash.callback(
    dash.dependencies.Output('assets-dropdown_sc', 'options'),
    dash.dependencies.Output('assets-dropdown_sc', 'value'),
    [dash.dependencies.Input('category-dropdown_sc', 'value')])
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
    dash.dependencies.Output("sidebar_sc", "style"),
    dash.dependencies.Output("page-content_sc", "style"),
    dash.dependencies.Output("side_click_sc", "data"),
    dash.dependencies.Output("btn_sidebar_sc", "children"),

    [dash.dependencies.Input("btn_sidebar_sc", "n_clicks")],
    [
        dash.dependencies.State("side_click_sc", "data"),
    ]
)
def toggle_sidebar(n, nclick):
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
    dash.dependencies.Output("modal_sc", "is_open"),
    [dash.dependencies.Input("btn_about", "n_clicks"), dash.dependencies.Input("close", "n_clicks")],
    [dash.dependencies.State("modal_sc", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@dash.callback(
    dash.dependencies.Output('splot_sc', 'figure'),
    dash.dependencies.Output('shist_sc', 'figure'),
    dash.dependencies.Output('strat_sc', 'figure'),
    dash.dependencies.Output('res_sc', 'figure'),
    dash.dependencies.Output('date-picker-range_sc', 'min_date_allowed'),
    dash.dependencies.Output('date-picker-range_sc', 'max_date_allowed'),
    dash.dependencies.Output('date-picker-range_lrn_sc', 'min_date_allowed'),
    dash.dependencies.Output('date-picker-range_lrn_sc', 'max_date_allowed'),
    dash.dependencies.Output("loading-output_sc", "children"),
    dash.dependencies.Output("Laginput_sc", "max"),
    [dash.dependencies.Input('assets-dropdown_sc', 'value'),
     dash.dependencies.Input('TCinput_sc', 'value'),
     dash.dependencies.Input('SCinput_sc', 'value'),
     dash.dependencies.Input('date-picker-range_sc', 'start_date'),
     dash.dependencies.Input('date-picker-range_sc', 'end_date'),
     dash.dependencies.Input('date-picker-range_lrn_sc', 'start_date'),
     dash.dependencies.Input('date-picker-range_lrn_sc', 'end_date'),
     dash.dependencies.Input('Laginput_sc', 'value')
     ])
def upd_chart(asset, tc, sc, sdate, edate, sdate_lrn, edate_lrn, lag):
    scibt = SCI.ScikitVectorBacktester(asset, min(sdate, edate, edate_lrn, sdate_lrn),
                                       max(sdate, edate, edate_lrn, sdate_lrn), sc, tc / 100, "logistic")
    lagmax = round(min((ciso8601.parse_datetime(edate) - ciso8601.parse_datetime(sdate)).days,(ciso8601.parse_datetime(edate_lrn) - ciso8601.parse_datetime(sdate_lrn)).days)*0.5)
    data = scibt.get_data()

    raw = scibt.get_raw().reset_index()
    mind = raw["Date"].min()
    maxd = raw["Date"].max()

    scibt.run_strategy(sdate_lrn, edate_lrn, sdate, edate, lags=lag)
    res = scibt.get_results()
    load = []
    resfs = res.copy()
    resfs["fact"] = np.sign(resfs["return"])

    fs = plotfsn(data, resfs, asset, sdate_lrn, edate_lrn, sdate, edate)

    fighist = plothist(res)

    rfig = plotfin(res, sdate, edate, sc)
    fig = plotreturns(asset, res, tc, lag)
    return fig, fighist, fs, rfig, mind, maxd, mind, maxd, load, lagmax
