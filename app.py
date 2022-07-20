import dash
import dash_bootstrap_components as dbc
from dash import dcc, Input, Output,State, html

app = dash.Dash(__name__, use_pages=True,suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server
"""navbar = dbc.NavbarSimple(
    dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem(page["name"], href=page["path"])
            for page in dash.page_registry.values()
            if page["module"] != "pages.not_found_404"
        ],
        nav=True,
        label="More Pages",
    ),
    brand="Multi Page App Demo",
    color="primary",
    dark=True,
    className="mb-2",
)"""
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem("SMA",  href='/a'),
                dbc.DropdownMenuItem("Mom",  href='/ac')

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
            label="Contacts"
    )
    ],

    brand="Quantboard",
    brand_href="/",
    color="dark",
    dark=True,
)


app.layout = dbc.Container(
    [dcc.Location(id='url', refresh=False), navbar, dash.page_container],
    fluid=True,
)




if __name__ == "__main__":
    app.run_server(debug=True)
