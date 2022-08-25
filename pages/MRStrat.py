# import all the modules
from datetime import date

import ciso8601
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from dash import dcc, html

from assets import MRVectorBacktester as MR

# Assign page name

dash.register_page(__name__, path='/MeanReversion')

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
favg = loading_plot
rfig = loading_plot
main_color = '#656EF1'
str_color = '#FF7F0E'

# Make plots functions

def plotreturns(asset, data, tc, thrs, sma):
    title = 'Gross performance compared to the mean reversion strategy<br><sup> %s | TC = %.1f | Threshold = %.f | SMA = %.f' % (asset, tc, thrs, sma)
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


def plotavg(data,asset, threshold):
    thrs = max([abs(data["distance"].min()), abs(data["distance"].max())]) * threshold / 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data["Date"], y=data["distance"],
                             mode='lines',
                             name='Distance'))

    fig.add_hline(y=thrs, annotation_text="Overbuy threshold",
                  annotation_position="top left")
    fig.add_hline(y=0)
    fig.add_hline(y=-thrs, annotation_text="Oversell threshold",
                  annotation_position="bottom left")

    fig.update_layout(title=f'Price of {asset} minus SMA and threshold values', legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    return fig


# Make sidebar style
sidebar_full = {"position": "fixed",
                "left": "0vw",
                "height": "100%",
                "bottom": "auto",
                "padding": "0.5rem 1rem",
                "background-color": "#f7fafc",
                "transition": "all 0.2s",
                "width": "25vw"}

sidebar_hide = {"position": "fixed",
                "left": "-25vw",
                "height": "100%",
                "bottom": "auto",
                "padding": "0rem 0rem",
                "background-color": "#f7fafc",
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
                           id="btn_sidebar_mr",
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

# Make input forms
input_groups = html.Div(
    [
        dbc.InputGroup(
            [dbc.InputGroupText("Category", id="category_header_mr"), dbc.Select(options=[
                {"label": "Stocks", "value": "Stocks"},
                {"label": "Crypto", "value": "Crypto"}],
                id='category-dropdown_mr',
                value="Stocks")],
            className="mb-3",
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("Asset", id="asset_header_mr"), dbc.Select(options=std,
                                                                           value="TSLA",
                                                                           id='assets-dropdown_mr')],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Date", id="date_header_mr"),
                dcc.DatePickerRange(
                    id='date-picker-range_mr',
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
                dbc.InputGroupText("Threshold", id="th_header_mr"),
                dbc.Input(id="THRSinput_mr", type="number", value=35, min=0, max=100, step=1, style={}),
                dbc.InputGroupText("%")
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("SMA", id="sma_header_mr"),
             dbc.Input(id="SMAinput_mr", type="number", value=10, min=1, max=100, step=1, style={})
             ],
            className="mb-3"
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Transaction cost", id="tc_header_mr"),
                dbc.Input(id="TCinput_mr", type="number", value=0, min=0, max=5, step=0.1, style={}),
                dbc.InputGroupText("%"),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("Start capital", id="sc_header_mr"),
             dbc.Input(id="SCinput_mr", type="number", value=10000, min=1, max=9999999999, step=1, style={}),
             dbc.InputGroupText("$")
             ],
            className="mb-3"
        )
    ]
)
sidebar = html.Div(
    [
        html.H2("Mean Reversion", className="display-4"),
        html.Hr(),
        html.P(
            "Strategy parameters", className="lead"
        ),
        input_groups,
    ], style=sidebar_hide, id="sidebar_mr"
)

dashboard = html.Div(children=
                     [dbc.Row([dbc.Col([dcc.Graph(figure=favg, id="thrs_mr")], width=8),
                               dbc.Col([dcc.Graph(figure=fighist, id="shist_mr")], width=4)],
                              style={'padding-left': 10, 'padding-right': 10, 'flex': 1}),
                      dbc.Row([dbc.Col([dcc.Graph(figure=fig, id="splot_mr")], width=8),
                               dbc.Col([dcc.Graph(figure=rfig, id="res_mr")], width=4)]),
                      dbc.Modal(
                          [
                              dbc.ModalHeader(dbc.ModalTitle("About strategy")),
                              dbc.ModalBody(children=[
                                  html.H5("Strategy idea", className="card-title"),
                                  html.Hr(),
                                  html.P("The mean-reversion strategies hypothesis is that stock prices or prices of other financial instruments tend to revert to a mean level or to a trend level when they have deviated too much from such levels."),
                                  html.Br(),

                                  html.H5("Parameters", className="card-title"),
                                  html.Hr(),
                                  html.Li("Date"),
                                  html.P("The date range in which the strategy is backtested"),
                                  html.Li("Threshold"),
                                  html.P("The strategy starts working if the value exceeds the threshold. For example, if the value exceeds the upper threshold, the strategy begins to create short positions.  If it exceeds the lower one it creates long positions. The value is set as a percentage."),
                                  html.Li("SMA"),
                                  html.P("The length of the range (in days) for which the SMA is calculated."),
                                  html.Li("Transaction cost"),
                                  html.P("Proportional transaction costs per trade in %."),
                                  html.Li("Start capital"),
                                  html.P("The amount to invest in $"),
                                  html.Br(),

                                  html.H5("Charts explanation", className="card-title"),
                                  html.Hr(),

                                  html.Li("Price of asset minus SMA and threshold values"),
                                  html.P("Shows the entered threshold values  and the difference between price of asset and SMA."),
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
                          id="modal_mr",
                          is_open=False,
                      ), dbc.Popover(
                         "Choose an asset category",
                         target="category_header_mr",
                         body=True,
                         trigger="hover",
                     ),
                      dbc.Popover(
                          "Choose an asset from the TOP 200 list sorted by capitalization",
                          target="asset_header_mr",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Choose start and end dates for backtesting",
                          target="date_header_mr",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Enter the transaction cost value per trade",
                          target="tc_header_mr",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Уnter a percentage value that will set the threshold in both directions from the 0 value",
                          target="th_header_mr",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Enter the number of days to calculate the SMA",
                          target="sma_header_mr",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Enter the amount of capital that the strategy will work with",
                          target="sc_header_mr",
                          body=True,
                          trigger="hover",
                      ),

                      ], id="page-content_mr", style=content_full)

# Design the app layout
layout = html.Div([dcc.Store(id='side_click_mr'), navbar, dbc.Row([dbc.Col(sidebar), dbc.Col(dashboard)])]), html.Div(dcc.Loading(
    id="loading",
    children=[html.Div([html.Div(id="loading-output_mr")])],
    type="cube",
    color="#7F7F7F"),
    style={'position': 'absolute',
           'top': '50%',
           'left': '50%',
           'transform': 'translate(-50%,-50%)'})


@dash.callback(
    dash.dependencies.Output("sidebar_mr", "style"),
    dash.dependencies.Output("page-content_mr", "style"),
    dash.dependencies.Output("side_click_mr", "data"),
    dash.dependencies.Output("btn_sidebar_mr", "children"),
    [dash.dependencies.Input("btn_sidebar_mr", "n_clicks")],

    [dash.dependencies.State("side_click_mr", "data")]
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
    dash.dependencies.Output("modal_mr", "is_open"),
    [dash.dependencies.Input("btn_about", "n_clicks"), dash.dependencies.Input("close", "n_clicks")],
    [dash.dependencies.State("modal_mr", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@dash.callback(
    dash.dependencies.Output('assets-dropdown_mr', 'options'),
    dash.dependencies.Output('assets-dropdown_mr', 'value'),
    [dash.dependencies.Input('category-dropdown_mr', 'value')])
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
    dash.dependencies.Output('splot_mr', 'figure'),
    dash.dependencies.Output('shist_mr', 'figure'),
    dash.dependencies.Output('thrs_mr', 'figure'),
    dash.dependencies.Output('res_mr', 'figure'),
    dash.dependencies.Output('SMAinput_mr', "max"),
    dash.dependencies.Output('date-picker-range_mr', 'min_date_allowed'),
    dash.dependencies.Output('date-picker-range_mr', 'max_date_allowed'),
    dash.dependencies.Output('SMAinput_mr', 'value'),
    dash.dependencies.Output("loading-output_mr", "children"),
    [dash.dependencies.Input('assets-dropdown_mr', 'value'),
     dash.dependencies.Input('TCinput_mr', 'value'),
     dash.dependencies.Input('SCinput_mr', 'value'),
     dash.dependencies.Input('THRSinput_mr', 'value'),
     dash.dependencies.Input('date-picker-range_mr', 'start_date'),
     dash.dependencies.Input('date-picker-range_mr', 'end_date'),
     dash.dependencies.Input('SMAinput_mr', 'value')])
def upd_chart(asset, tc, sc, threshold, sdate, edate, sma):
    diff = ciso8601.parse_datetime(edate) - ciso8601.parse_datetime(sdate)

    smamax = int(diff.days)

    if sma is None:
        sma = 1

    mrbt = MR.MRVectorBacktester(asset, sdate, edate, sc, tc / 100)
    load = []
    data = mrbt.get_data()

    raw = mrbt.get_raw().reset_index()
    mind = raw["Date"].min()
    maxd = raw["Date"].max()

    mrbt.run_strategy(SMA=sma, threshold=threshold / 100)
    res = mrbt.get_results()

    fighist = plothist(res)
    favg = plotavg(res, asset, threshold)
    rfig = plotfin(res, sdate, edate, sc)
    fig = plotreturns(asset, res, tc, threshold, sma)
    return fig, fighist, favg, rfig, smamax, mind, maxd,sma, load
