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
from plotly.subplots import make_subplots

from assets import DLVectorBacktester as DLR

# Assign page name

dash.register_page(__name__, path='/DeepLearning')

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

histr = loading_plot
fig = loading_plot
fighist = loading_plot
fs = loading_plot
rfig = loading_plot
main_color = '#656EF1'
str_color = '#FF7F0E'

tab_st = {"border": "1px solid black", "align": "center", "border - collapse": "collapse","border-style": "hidden", "padding-top": "10px",
  "padding-bottom": "10px",
  "padding-left": "15px",
  "padding-right": "15px"}

tr_color = '#72B7B2'
val_color = '#4C78A8'
# Make plot functions

def plotreturns(asset, data, tc, lag):
    title = 'Gross performance compared to the deep learning strategy<br><sup> %s | TC = %.1f | Lag = %.f' % (asset, tc, lag)
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


def plotfsn(data, asset, sdate_lrn, edate_lrn, sdate, edate):

    diff = ciso8601.parse_datetime(edate_lrn) - ciso8601.parse_datetime(sdate_lrn)
    valid_st = (ciso8601.parse_datetime(edate_lrn) - diff * 0.2).strftime("%Y-%m-%d")

    fig = px.line(x=data["Date"], y=data["price"])

    fig.add_vrect(
        x0=sdate_lrn, x1=edate_lrn,
        annotation_text="Training", annotation_position='top left',
        fillcolor=tr_color, opacity=0.5,
        layer="below", line_width=0
    )
    fig.add_vrect(
        x0=sdate, x1=edate,
        annotation_text="Backtesting", annotation_position='bottom left',
        fillcolor='#BAB0AC', opacity=0.5,
        layer="below", line_width=0)


    fig.add_vrect(
        x0=valid_st, x1=edate_lrn,
        annotation_text="Val", annotation_position='top left', line=dict(color=val_color, width=1),
        layer="above")

    fig.update_layout(title=asset)
    fig.update_yaxes(title="")
    fig.update_xaxes(title="")
    fig.update_layout(margin=dict(t=100, b=75), legend=dict(
        orientation="h",

        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig


def plotdlr(his):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))

    fig.add_trace(
        go.Scatter(x=his.index, y=his["loss"], mode="lines", name="train", line_color=tr_color),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=his.index, y=his["val_loss"], name="validation", line_color=val_color),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=his.index, y=his["accuracy"], name="train", line_color=tr_color, showlegend=False),
        row=1, col=2)

    fig.add_trace(
        go.Scatter(x=his.index, y=his["val_accuracy"], name="validation", line_color=val_color, showlegend=False), row=1,
        col=2)

    fig.update_layout(title='Deep learning results<br><sup>Epochs = %d , accuracy = <b>%.2f</b>, loss = <b>%.2f</b>' % (
        int(len(his.index)), his.accuracy.iloc[-1], his.loss.iloc[-1]))
    fig.update_layout(legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=-0.2,
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
                           id="btn_sidebar_dl",
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
            [dbc.InputGroupText("Category", id="category_header_dl"), dbc.Select(options=[
                {"label": "Stocks", "value": "Stocks"},
                {"label": "Crypto", "value": "Crypto"}],
                id='category-dropdown_dl',
                value="Stocks")],
            className="mb-3",
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("Asset", id="asset_header_dl"), dbc.Select(options=std,
                                                                           value="TSLA",
                                                                           id='assets-dropdown_dl')],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Training", id="traindate_header_dl"),
                dcc.DatePickerRange(
                    id='date-picker-range_lrn_dl',
                    minimum_nights=100,
                    clearable=True,
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
                dbc.InputGroupText("Backtesting", id="testdate_header_dl"),
                dcc.DatePickerRange(
                    id='date-picker-range_dl',
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
                dbc.InputGroupText("Lag", id="lag_header_dl"),
                dbc.Input(id="Laginput_dl", type="number", value=3, min=1, max=15, step=1, style={})
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Transaction cost", id="tc_header_dl"),
                dbc.Input(id="TCinput_dl", type="number", value=0, min=0, max=5, step=0.1, style={}),
                dbc.InputGroupText("%"),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [dbc.InputGroupText("Start capital", id="sc_header_dl"),
             dbc.Input(id="SCinput_dl", type="number", value=10000, min=1, max=9999999999, step=1, style={}),
             dbc.InputGroupText("$")
             ],
            className="mb-3"
        ),

    ]
)
dashboard = html.Div(children=
                     [dbc.Row([dbc.Col(dcc.Graph(figure=fs, id="strat_dl"), width=7),
                               dbc.Col([dcc.Graph(figure=histr, id="shist_dl")], width=5)], className="g-0"),
                      dbc.Row([dbc.Col([dcc.Graph(figure=fig, id="splot_dl")], width=8),
                               dbc.Col([dcc.Graph(figure=rfig, id="res_dl")], width=4)]),
                      dbc.Modal(
                          [
                              dbc.ModalHeader(dbc.ModalTitle("About strategy")),
                              dbc.ModalBody(children=[
                                  html.H5("Strategy idea", className="card-title"),
                                  html.Hr(),
                                  html.P("The algorithm trains deep neural network for a selected period of time and predicts the direction of movement of the asset price."),
                                  html.P("The strategy uses the open source ML framework «TensorFlow» to predict asset movement direction."),
                                  html.P("Basic parameters of a deep neural network:"),
                                  html.P("The dataset passes through the deep neural network 50 times, and 20% of the training data is used for training validation."),
                                  html.Br(),

                                  html.H5("Parameters", className="card-title"),
                                  html.Hr(),
                                  html.Li("Training"),
                                  html.P(
                                      "Date range in which deep neural network is fitted (trained). The range may overlap with the Backtesting range."),
                                  html.Li("Backtesting"),
                                  html.P(
                                      "Date range in which deep neural network is evaluated."),
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
                                      "Shows asset price, training and backtesting areas. In training area 20% uses for model validation."),
                                  html.Li("Gross performance compared to the deep learning strategy"),
                                  html.P(
                                      "Shows cash flow of asset and strategy. If in the end of the chart strategy line is lower then asset line - your strategy is unprofitable and vice versa."),
                                  html.Li("Deep learning results"),
                                  html.P("Shows loss and accuracy changes through epochs on training and validation datasets."),
                                  html.P("Loss is a value that represents the summation of errors in our model. Loss function is used by the model to learn. The goal of the model is to minimize the value of the loss."),
                                  html.P( "Accuracy measures how well our model predicts by comparing the model predictions with the true values in terms of percentage."),
                                  html.P("If we analyze these two measurements together, we can infer more information about how our model is working:"),

                                  html.Table([html.Tr([html.Th(), html.Th("Low Loss",style=tab_st), html.Th("High Loss",style=tab_st)],style=tab_st),
                                              html.Tr([html.Th("Low Accuracy",style=tab_st), html.Td("A lot of small errors",style=tab_st),
                                                       html.Td("A lot of big errors",style=tab_st)],style=tab_st),
                                              html.Tr([html.Th("High Accuracy",style=tab_st), html.Td("A few small errors",style=tab_st),
                                                       html.Td("A few big errors",style=tab_st)],style=tab_st)],
                                             style=tab_st
                                             ),
                                  html.Li("Returns"),
                                      html.P(
                                          "Asset shows the amount of capital as if you were buying and holding an asset over the entire time period. Strategy shows the amount of capital as if you followed the strategy. S&P500 is a benchmark index."),
                                  html.Br(),

                              ]),
                              dbc.ModalFooter(
                                  dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)
                              ),
                          ],
                          id="modal_dl",
                          is_open=False,
                      ),
                      dbc.Popover(
                          "Choose an asset category",
                          target="category_header_dl",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Choose an asset from the TOP 200 list sorted by capitalization",
                          target="asset_header_dl",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Choose start and end dates for model training",
                          target="traindate_header_dl",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Choose start and end dates for backtesting",
                          target="testdate_header_dl",
                          body=True,
                          trigger="hover",
                      ),
                      dbc.Popover(
                          "Enter the transaction cost value per trade",
                          target="tc_header_dl",
                          body=True,
                          trigger="hover",
                      ),

                      dbc.Popover(
                          "Enter the amount of capital that the strategy will work with",
                          target="sc_header_dl",
                          body=True,
                          trigger="hover",
                      ),

                      dbc.Popover(
                          "A parameter that adds information about previous periods to the regression equation. The value determines the number of days for which the lag will be created.",
                          target="lag_header_dl",
                          body=True,
                          trigger="hover",
                      )
                      ], id="page-content_dl", style=content_full)

sidebar_lr = html.Div(
    [
        html.H2("Deep learning", className="display-4"),
        html.Hr(),
        html.P(
            "Strategy parameters", className="lead"
        ),
        input_groups,
    ], style=sidebar_hide, id="sidebar_dl"
)

# Design the page layout
layout = html.Div(
    [dcc.Store(id='side_click_dl'), navbar, dbc.Row([dbc.Col(sidebar_lr), dbc.Col(dashboard)])]), html.Div(dcc.Loading(
    id="loading",
    children=[html.Div([html.Div(id="loading-output_dl")])],
    type="cube",
    color="#7F7F7F"),
    style={'position': 'absolute',
           'top': '50%',
           'left': '50%',
           'transform': 'translate(-50%,-50%)'})


# Create page callbacks
@dash.callback(
    dash.dependencies.Output('assets-dropdown_dl', 'options'),
    dash.dependencies.Output('assets-dropdown_dl', 'value'),
    [dash.dependencies.Input('category-dropdown_dl', 'value')])
def assets_options(category):
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
    dash.dependencies.Output("sidebar_dl", "style"),
    dash.dependencies.Output("page-content_dl", "style"),
    dash.dependencies.Output("side_click_dl", "data"),
    dash.dependencies.Output("btn_sidebar_dl", "children"),

    [dash.dependencies.Input("btn_sidebar_dl", "n_clicks")],
    [
        dash.dependencies.State("side_click_dl", "data"),
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
    dash.dependencies.Output("modal_dl", "is_open"),
    [dash.dependencies.Input("btn_about", "n_clicks"), dash.dependencies.Input("close", "n_clicks")],
    [dash.dependencies.State("modal_dl", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@dash.callback(
    dash.dependencies.Output('splot_dl', 'figure'),
    dash.dependencies.Output('shist_dl', 'figure'),
    dash.dependencies.Output('strat_dl', 'figure'),
    dash.dependencies.Output('res_dl', 'figure'),
    dash.dependencies.Output("loading-output_dl", "children"),
    dash.dependencies.Output('date-picker-range_dl', 'min_date_allowed'),
    dash.dependencies.Output('date-picker-range_dl', 'max_date_allowed'),
    dash.dependencies.Output('date-picker-range_lrn_dl', 'min_date_allowed'),
    dash.dependencies.Output('date-picker-range_lrn_dl', 'max_date_allowed'),
    dash.dependencies.Output("Laginput_dl", "max"),
    [dash.dependencies.Input('assets-dropdown_dl', 'value'),
     dash.dependencies.Input('TCinput_dl', 'value'),
     dash.dependencies.Input('SCinput_dl', 'value'),
     dash.dependencies.Input('date-picker-range_dl', 'start_date'),
     dash.dependencies.Input('date-picker-range_dl', 'end_date'),
     dash.dependencies.Input('date-picker-range_lrn_dl', 'start_date'),
     dash.dependencies.Input('date-picker-range_lrn_dl', 'end_date'),
     dash.dependencies.Input('Laginput_dl', 'value')
     ])
def upd_chart(asset, tc, sc, sdate, edate, sdate_lrn, edate_lrn, lag):
    dlbt = DLR.DLVectorBacktester(asset, min(sdate, edate, edate_lrn, sdate_lrn),
                                  max(sdate, edate, edate_lrn, sdate_lrn), sc, tc / 100)
    lagmax = round(min((ciso8601.parse_datetime(edate) - ciso8601.parse_datetime(sdate)).days,(ciso8601.parse_datetime(edate_lrn) - ciso8601.parse_datetime(sdate_lrn)).days)*0.5)

    dlbt.run_strategy(sdate_lrn, edate_lrn, sdate, edate, lags=lag)
    mind, maxd = dlbt.get_dpm()
    data = dlbt.get_data()
    res = dlbt.get_results()
    load = []

    fs = plotfsn(data, asset, sdate_lrn, edate_lrn, sdate, edate)

    fighist = plotdlr(dlbt.get_history())

    rfig = plotfin(res, sdate, edate, sc)
    fig = plotreturns(asset, res, tc, lag)
    return fig, fighist, fs, rfig, load, mind, maxd, mind, maxd, lagmax
