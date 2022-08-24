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

# Assign page name

dash.register_page(__name__, path='/Momentum')

# Read stocks list

url = "https://drive.google.com/uc?id=17PC74DXlz0dlKvf04Mjv6xEh7wK7Mtt2"
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
def plotreturns(asset, data, momentum, tc):
    title = 'Gross performance compared to the momentum strategy<br><sup> %s | TC = %.1f | Momentum = %.f' % (
    asset, tc, momentum)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["cstrategy"],
                             mode='lines', line=dict(color=str_color),
                             name='Strategy'))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["creturns"],
                             mode='lines', line=dict(color=main_color),
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
    fighist = px.histogram(data['returns in %'], x='returns in %', marginal="box", title="Returns distribution",color_discrete_sequence=[main_color])
    return fighist


def sparkline(data):
    # fig = px.line(data, height=200, width=200)
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
    Assline = sparkline(data[["creturns", "Date"]].rename(columns={"creturns": "price"}))
    Assline.update_layout(domain={'row': 2, 'column': 1})
    Strline = sparkline(data[["cstrategy", "Date"]].rename(columns={"cstrategy": "price"}))
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
    # fig.update_layout(grid={'rows': 3, 'columns': 2})
    return fig


def plotfs(res, asset):
    # title = '%s | TC = %.4f' % (asset, 0)
    title = 'Gross performance of asset and nine momentum strategies'
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
                                mode='lines', line=dict(color=main_color, width=1), opacity=.4, name=to_grey[i]))
    fs.add_trace(go.Scatter(x=resfig["Date"], y=resfig["return"],
                            mode='lines', name="asset", line_color=main_color))
    fs.add_trace(go.Scatter(x=resfig["Date"], y=resfig[maxname],
                            mode='lines', line_color=str_color, name=maxname))
    fs.update_layout(title=title)

    return fs, maxmomentum


# Make default plots


"""fig = plotreturns("TSLA", res, 1, 0)
fighist = plothist(res)
fs, mm = plotfs(res, "TSLA")
rfig = plotfin(res, '2020-01-01', '2022-06-01', sc)"""
# testsp= plotfspark(res, "TSLA", '2020-01-01', '2022-06-01', sc)
# Build the Components

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
                           id="btn_sidebar_m",
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
            [dbc.InputGroupText("Category", id="category_header_m"), dbc.Select(options=[
                {"label": "Stocks", "value": "Stocks"},
                {"label": "Crypto", "value": "Crypto"}],
                id='category-dropdown_m',
                value="Stocks")],
            className="mb-3",
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("Asset", id="asset_header_m"), dbc.Select(options=std,
                                                                          value="TSLA",
                                                                          id='assets-dropdown_m')],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Date", id="date_header_m"),
                dcc.DatePickerRange(
                    id='date-picker-range_m',
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
                dbc.InputGroupText("Transaction cost",id="tc_header_m"),
                dbc.Input(id="TCinput_m", type="number", value=0, min=0, max=5, step=0.1, style={}),
                dbc.InputGroupText("%"),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("Start capital", id="sc_header_m"),
             dbc.Input(id="SCinput_m", type="number", value=10000, min=1, max=9999999999, step=1, style={}),
             dbc.InputGroupText("$")
             ],
            className="mb-3"
        )
    ]
)
dashboard = html.Div(children=
                     [dbc.Row([dbc.Col([dcc.Graph(figure=fs, id="strat_m")], width=8),
                               dbc.Col([dcc.Graph(figure=fighist, id="shist_m")], width=4)],
                              style={'padding-left': 10, 'padding-right': 10, 'flex': 1}),
                      dbc.Row([dbc.Col([dcc.Graph(figure=fig, id="splot_m")], width=8),
                               dbc.Col([dcc.Graph(figure=rfig, id="res_m")], width=4)]),
                      dbc.Modal(
                          [
                              dbc.ModalHeader(dbc.ModalTitle("About strategy")),
                              dbc.ModalBody(children=[
                                  html.H5("Strategy idea", className="card-title"),
                                  html.Hr(),
                                  html.P("The strategy is based on the assumption that if the market is in a rising or falling trend for a certain number of days, the trend will continue."),
                                  html.Br(),

                                  html.H5("Parameters", className="card-title"),
                                  html.Hr(),
                                  html.Li("Date"),
                                  html.P("The date range in which the strategy is backtested"),
                                  html.Li("Transaction cost"),
                                  html.P("Proportional transaction costs per trade in %"),
                                  html.Li("Start capital"),
                                  html.P("The amount to invest in $"),
                                  html.Br(),

                                  html.H5("Charts explanation", className="card-title"),
                                  html.Hr(),
                                  html.Li("Price chart and nine momentum strategies"),
                                  html.P("Shows the asset price and the performance of nine strategies. The strategies differ in the number of days we observe the market  (1-9 days). The most profitable strategy is highlighted on the chart. Momentum parameter in this strategy is automatically used in other charts"),
                                  html.P("For example, if momentum = 8, then the strategy signal is created only when the asset moves in one direction (rising or falling) all 8 days in a row."),
                                  html.Li("Gross performance compared to the momentum strategy"),
                                  html.P("Shows cash flow of asset and strategy. If in the end of the chart strategy line is lower then asset line - your strategy is unprofitable and vice versa."),
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
                          id="modal_m",
                          is_open=False,
                      ), dbc.Popover(
                         "Choose an asset category",
                         target="category_header_m",
                         body=True,
                         trigger="hover",
                     ),
                      dbc.Popover(
                          "Choose an asset from the TOP 200 list sorted by capitalization",
                          target="asset_header_m",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Choose start and end dates for backtesting",
                          target="date_header_m",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Enter the transaction cost value per trade",
                          target="tc_header_m",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Enter the amount of capital that the strategy will work with",
                          target="sc_header_m",
                          body=True,
                          trigger="hover",
                      )
                      ], id="page-content_m", style=content_full)

sidebar_m = html.Div(
    [
        html.H2("Momentum", className="display-4"),
        html.Hr(),
        html.P(
            "Strategy parameters", className="lead"
        ),
        input_groups,
    ], style=sidebar_hide, id="sidebar_m"
)

# Design the app layout
layout = html.Div([dcc.Store(id='side_click_m'), navbar, dbc.Row([dbc.Col(sidebar_m), dbc.Col(dashboard)])]), html.Div(
    dcc.Loading(
        id="loading",
        children=[html.Div([html.Div(id="loading-output_m")])],
        type="cube",
        color="#7F7F7F"),
    style={'position': 'absolute',
           'top': '50%',
           'left': '50%',
           'transform': 'translate(-50%,-50%)'})


@dash.callback(
    dash.dependencies.Output('assets-dropdown_m', 'options'),
    dash.dependencies.Output('assets-dropdown_m', 'value'),
    [dash.dependencies.Input('category-dropdown_m', 'value')])
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


"""@dash.callback(
    dash.dependencies.Output('date-picker-range_m', 'min_date_allowed'),
    dash.dependencies.Output('date-picker-range_m', 'max_date_allowed'),
    [dash.dependencies.Input('assets-dropdowna', 'value')])
def assets_date_range(asset):
    raw = yf.Ticker(asset).history(period="max", actions=False)
    raw = raw.reset_index()
    mind = raw["Date"].min()
    maxd = raw["Date"].max()
    return mind, maxd"""


@dash.callback(
    dash.dependencies.Output("sidebar_m", "style"),
    dash.dependencies.Output("page-content_m", "style"),
    dash.dependencies.Output("side_click_m", "data"),
    dash.dependencies.Output("btn_sidebar_m", "children"),
    [dash.dependencies.Input("btn_sidebar_m", "n_clicks")],
    [
        dash.dependencies.State("side_click_m", "data"),
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
    dash.dependencies.Output("modal_m", "is_open"),
    [dash.dependencies.Input("btn_about", "n_clicks"), dash.dependencies.Input("close", "n_clicks")],
    [dash.dependencies.State("modal_m", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@dash.callback(
    dash.dependencies.Output('splot_m', 'figure'),
    dash.dependencies.Output('shist_m', 'figure'),
    dash.dependencies.Output('strat_m', 'figure'),
    dash.dependencies.Output('res_m', 'figure'),
    dash.dependencies.Output('date-picker-range_m', 'min_date_allowed'),
    dash.dependencies.Output('date-picker-range_m', 'max_date_allowed'),
    dash.dependencies.Output("loading-output_m", "children"),
    [dash.dependencies.Input('assets-dropdown_m', 'value'),
     dash.dependencies.Input('TCinput_m', 'value'),
     dash.dependencies.Input('SCinput_m', 'value'),
     dash.dependencies.Input('date-picker-range_m', 'start_date'),
     dash.dependencies.Input('date-picker-range_m', 'end_date')])

def upd_chart(asset, tc, sc, sdate, edate):
    mombt = Mom.MomVectorBacktester(asset, sdate, edate, sc, tc / 100)

    data = mombt.get_data()

    raw = mombt.get_raw().reset_index()
    mind = raw["Date"].min()
    maxd = raw["Date"].max()

    fs, maxmomentum = plotfs(data, asset)
    mombt.run_strategy(momentum=maxmomentum)
    res = mombt.get_results()
    load = []
    fighist = plothist(res)

    rfig = plotfin(res, sdate, edate, sc)
    fig = plotreturns(asset, res, maxmomentum, tc)
    return fig, fighist, fs, rfig, mind, maxd, load
