import dash
import dash_bootstrap_components as dbc
from dash import dcc, Input, Output, State, html
from dash_bootstrap_components._components.Container import Container


app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True, update_title='Calculating, quanting, plotting...',
                external_stylesheets=[dbc.themes.COSMO, dbc.icons.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    children=[dash.page_container],
    id="main",
    fluid=True,
    style={'padding': 0}
)



if __name__ == "__main__":
    app.run_server(debug=True)
