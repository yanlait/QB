import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

dash.register_page(__name__, path='/')

"""Jumbotron = dbc.Col(
    html.Div(
        [
            html.H2("Algorithmic strategies dashboards", className="display-3"),
            html.Hr(className="my-2"),
            html.P(
                "Swap the background-color utility and add a `.text-*` color "
                "utility to mix up the look."
            ),
        ],
        className="h-100 p-5 text-white bg-dark rounded-3",
    ),
    md=12,
)"""

left_jumbotron = dbc.Col(
    html.Div(
        [
            html.H2("SMA", className="display-3"),
            html.Hr(className="my-2"),
            html.P(
                "Strategy uses death cross and golden cross patterns."
            ),dbc.Row([
                dbc.Button("Dashboard", color="light", outline=True, href="/ac")
                   ]),
        ],
        className="h-100 p-5 text-white bg-dark rounded-3",
    ),
    md=4,
)
center_jumbotron = dbc.Col(
    html.Div(
        [
            html.H2("Momentum", className="display-3"),
            html.Hr(className="my-2"),
            html.P(
                "Strategy is to buy the stock if the last N returns was positive and to sell it if it was negative."
            )
        , dbc.Row([
                dbc.Button("Dashboard", color="light", outline=True, href="/ac")
                   ]),
        ],
        className="h-100 p-5 text-white bg-dark border rounded-3",
    ),
    md=4,
)

right_jumbotron = dbc.Col(
    html.Div(
        [
            html.H2("Mean reversion", className="display-3"),
            html.Hr(className="my-2"),
            html.P(
                "Coming soon"
            )
        ],
        className="h-100 p-5 bg-light border rounded-3",
    ),
    md=4,
)



carousel = dbc.Carousel(
    items=[
        {"key": "1", "src": "/assets/images/slide_1.jpg"},
        {"key": "2", "src": "/assets/images/slide_2.jpg"},
        {"key": "3", "src": "/assets/images/slide_3.jpg"},
        {"key": "4", "src": "/assets/images/slide_4.jpg"},
    ],
    controls=False,
    indicators=False,
    interval=4000,
    ride="carousel",


)

Jumbotron = [left_jumbotron,center_jumbotron, right_jumbotron]


layout = html.Div(children=[
    dbc.Row([carousel],style={'margin-top':'10px','margin-bottom':'10px'}),
html.H1("Strategies",style={"textAlign": "center",'margin-top':'10px','margin-bottom':'10px'}),
dbc.Row(Jumbotron)])