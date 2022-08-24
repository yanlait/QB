from dash import html
import dash_bootstrap_components as dbc
import dash

dash.register_page(__name__, path="/404")

layout = html.Div(dbc.Button(children=[html.I(className="bi bi-house-door me-2"),"Home"],
                             outline=True, size="sm", color="secondary", href="/",
                             style={'padding-top': '6px', 'padding-bottom': '6px', 'margin-right': '30px'}))
