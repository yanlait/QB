import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from yahoo_fin import stock_info as si
import yfinance as yf
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path='/')


def make_card(name, price, change, graph):
    card = dbc.Card(
        dbc.CardBody(
            [
                html.H5(name, className="card-title"),
                html.H5(price, className="card-title"),
                html.H6(change, className="card-subtitle"),
                dcc.Graph(figure=graph, id="test", config=dict(displayModeBar=False))
            ]
        ),
        style={"width": "10%"}, color="white", outline=True, className="bg-light rounded-3"
    )

    return card


# formiruem gainers
gainers = []
df = si.get_day_gainers().set_index("Symbol")
symbols = df.index.to_list()
tpl = pd.DataFrame()
tpl.columns.name = "SMB"
i = 0
c = 0

while c <= 9:
    tpl[symbols[i]] = yf.Ticker(symbols[i]).history(period="2d", interval="30m")["Close"].tail(
        len(yf.Ticker(symbols[i]).history(period="1d", interval="30m")["Close"]) + 1)
    tpl = tpl.dropna(axis=1, how='any', thresh=len(tpl) - 2)
    c = len(tpl.columns)
    i += 1

for e in range(len(tpl.columns)):
    line = px.line(tpl[tpl.columns[e]].reset_index(drop=True), height=40, color_discrete_sequence=['green'])
    line.update_xaxes(visible=False, fixedrange=True)
    line.update_yaxes(visible=False, fixedrange=True)

    line.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, l=10, b=10, r=10)
    )

    card = make_card(tpl[tpl.columns[e]].name, str(df.loc[tpl[tpl.columns[e]].name, "Price (Intraday)"]),
                     "+" + str(df.loc[tpl[tpl.columns[e]].name, "% Change"]) + "%", line)
    gainers.append(card)

# formiruem losers
losers = []
df = si.get_day_losers().set_index("Symbol")
symbols = df.index.to_list()
tpl = pd.DataFrame()
tpl.columns.name = "SMB"
i = 0
c = 0

while c <= 9:
    tpl[symbols[i]] = yf.Ticker(symbols[i]).history(period="2d", interval="30m")["Close"].tail(
        len(yf.Ticker(symbols[i]).history(period="1d", interval="30m")["Close"]) + 1)
    tpl = tpl.dropna(axis=1, how='any', thresh=len(tpl) - 2)
    c = len(tpl.columns)
    i += 1

for e in range(len(tpl.columns)):
    line = px.line(tpl[tpl.columns[e]].reset_index(drop=True), height=40, color_discrete_sequence=['red'])
    line.update_xaxes(visible=False, fixedrange=True)
    line.update_yaxes(visible=False, fixedrange=True)

    line.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, l=10, b=10, r=10)
    )

    card = make_card(tpl[tpl.columns[e]].name, str(df.loc[tpl[tpl.columns[e]].name, "Price (Intraday)"]),
                     str(df.loc[tpl[tpl.columns[e]].name, "% Change"]) + "%", line)
    losers.append(card)

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
                        dbc.DropdownMenuItem(children=[html.I(className="bi bi-github me-2"),
                                                       "Github"], href="https://github.com/yanlait"),
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

left_jumbotron_str = dbc.Col(
    html.Div(
        [
            html.H2("SMA", className="display-3"),
            html.Hr(className="my-2"),
            html.P(
                "Basic strategy, which is based on the patterns of the golden and dead cross."
            ), dbc.Row([
            dbc.Button("Dashboard", color="dark", outline=True, href="/SMA", className="bottom")
        ]),
        ],
        className="h-100 p-5 text-black bg-light border rounded-3",
    ),
    md=4,
)
center_jumbotron_str = dbc.Col(
    html.Div(
        [
            html.H2("Momentum", className="display-3"),
            html.Hr(className="my-2"),
            html.P(
                "The idea is to buy rising and sell falling assets. In short, trend following."
            )
            , dbc.Row([
            dbc.Button("Dashboard", color="dark", outline=True, href="/Momentum", className="bottom")
        ])
        ],
        className="h-100 p-5 text-black bg-light border rounded-3",
    ),
    md=4,
)

right_jumbotron_str = dbc.Col(
    html.Div(
        [
            html.H2("Mean reversion", className="display-3"),
            html.Hr(className="my-2"),
            html.P(
                "The idea is to buy if the price is below the mean level and buy if it is higher."
            ),
            dbc.Row([
                dbc.Button("Dashboard", color="dark", outline=True, href="/MeanReversion", className="bottom")
            ]),
        ],
        className="h-100 p-5 text-black bg-light border rounded-3",
    ),
    md=4,
)

left_jumbotron_mp = dbc.Col(
    html.Div(
        [
            html.H2("Linear regression", className="display-3"),
            html.Hr(className="my-2"),
            html.P(
                "The strategy predicts the direction of the market using linear regression."
            ),
            dbc.Row([
            dbc.Button("Dashboard", color="dark", outline=True, href="/SMA", className="bottom")
        ]),
        ],
        className="h-100 p-5 text-black bg-light border rounded-3",
    ),
    md=4,
)
center_jumbotron_mp = dbc.Col(
    html.Div(
        [
            html.H2("Machine learning", className="display-3"),
            html.Hr(className="my-2"),
            html.P(
                "The strategy predicts the direction of the market using logistic regression."
            )
            , dbc.Row([
            dbc.Button("Dashboard", color="dark", outline=True, href="/Momentum", className="bottom")
        ]),
        ],
        className="h-100 p-5 text-black bg-light border rounded-3",
    ),
    md=4,
)

right_jumbotron_mp = dbc.Col(
    html.Div(
        [
            html.H2("Deep learning", className="display-3"),
            html.Hr(className="my-2"),
            html.P(
                "The neural network is trained and tested to predict the movement of the asset price"
            ),
            dbc.Row([
                dbc.Button("Dashboard", color="dark", outline=True, href="/MeanReversion")
            ], className="bottom-container"),
        ],
        className="h-100 p-5 text-black bg-light border rounded-3",
    ),
    md=4,
)

jmbt = html.Div(
    dbc.Container(
        [
            html.H1("Quantboard", className="display-3"),
            html.H2(
                "Strategy backtesting platform"),
            html.Hr(className="my-2", style={'width': '20em'}),
            html.H4("Explore 6 different strategies with easy-to-use dashboard tool for testing trading hypotheses!",
                    style={'width': '20em'})
        ],
        fluid=True,
        className="py-3",
    ),
    className="p-3 rounded-3",
    style={'background-image': 'url(assets/images/double.png)', "background-size": "cover", "height": "55vh"}
)

Jumbotron_str = [left_jumbotron_str, center_jumbotron_str, right_jumbotron_str]
Jumbotron_mp = [left_jumbotron_mp, center_jumbotron_mp, right_jumbotron_mp]
Jumbotron_all = [left_jumbotron_str, center_jumbotron_str, right_jumbotron_str, left_jumbotron_mp, center_jumbotron_mp,
                 right_jumbotron_mp]
layout = html.Div(children=[navbar, jmbt,
                            html.H1("Strategies",
                                    style={"textAlign": "center", 'margin-top': '30px', 'margin-bottom': '10px'}),
                            dbc.Row(Jumbotron_str,
                                    style={'padding-left': '10px', 'padding-right': '10px', 'margin-bottom': '20px'}),
                            dbc.Row(Jumbotron_mp,
                                    style={'padding-left': '10px', 'padding-right': '10px', 'margin-bottom': '20px'}),
                            html.H1("Market overview",
                                    style={"textAlign": "center", 'margin-top': '10px', 'margin-bottom': '10px'}),
                            html.H2("Top gained assets",
                                    style={"textAlign": "left", 'padding-left': '10px', 'padding-right': '10px',
                                           'margin-top': '10px', 'margin-bottom': '10px'}),
                            dbc.Row(gainers,
                                    style={'margin-left': '10px', 'margin-right': '10px', 'margin-bottom': '20px'}),
                            html.H2("Top losed assets",
                                    style={"textAlign": "left", 'padding-left': '10px', 'padding-right': '10px',
                                           'margin-top': '10px', 'margin-bottom': '10px'}),
                            dbc.Row(losers,
                                    style={'margin-left': '10px', 'margin-right': '10px', 'margin-bottom': '20px'}),
                            ])
