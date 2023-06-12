# -*- coding: utf-8 -*-
import os

import dash
# import dash_html_components as html
from dash import html
import dash_bootstrap_components as dbc
from dash import dcc
from dash.exceptions import PreventUpdate

from utils import *

import scipy.sparse.linalg
from sklearn.cluster import AgglomerativeClustering

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pymongo
from dash.dependencies import Input, Output

pdf_file_path = os.environ.get('PDF_FILES', '')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__)  #, external_stylesheets=external_stylesheets)
app.title = 'SIR visualization'

app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),

    html.Div(
        className="app-header",
        children=[
            html.Span('SIR visualization', className="app-header--title"),
            html.Span('- showing Vadere results', className="app-header--title-small")
        ]
    ),

    html.Div(
        id="app-hidden-information",
        children=[],
        style={"display": "none"}
    ),

    html.Div(children=[
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Button('Reload', id='button-reload', className="app-button"),
                ]),
                dbc.Col([
                     dcc.Input(id='input-folder-path', type='text', size='200',
                               value=r'C:\Users\XXXXX\output',
                               placeholder='Insert the path to the output folders here', debounce=True)
                ])
                ]),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='output-file-dropdown',
                        value=[''],
                        placeholder="Select SIR files to compare",
                        multi=True
                    ),
                ])
            ]),
            dbc.Row(
                [
                    dbc.Col([
                        dcc.Loading(
                            id="loading-results",
                            children=[
                                dcc.Graph(id='SIR-result-graph'),
                            ],
                            type="circle",
                        )
                    ]
                    )
                ])
        ])]
    )]
)


@app.callback(Output('SIR-result-graph', 'figure'),
              [Input('output-file-dropdown', 'value')])
def update_figure(selected_values):
    if not selected_values or len(selected_values) == 0:
        raise PreventUpdate

    figures = []
    for folder in selected_values:
        if len(folder) == 0:
            continue

        scatter, group_counts = create_folder_data_scatter(folder)
        if scatter:
            figures.extend(scatter)
    if len(figures) > 0:
        fig = go.Figure(data=figures)
        fig.update_layout(title='Susceptible / Infected / Removed')
        return fig

    raise PreventUpdate


@app.callback([Output('output-file-dropdown', 'options'),
               Output('app-hidden-information', 'children')],
              [Input('button-reload', 'n_clicks'),
               Input('input-folder-path', 'value')])
def update_files(btn0, folder_path):
    if not btn0:
        raise PreventUpdate

    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} is not a valid path.")

    folders = [
        os.path.join(folder_path, folder) for folder in
        os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, folder))
    ]

    return [
        {"label": os.path.basename(folder), "value": folder}
        for folder in folders
    ],  folders


if __name__ == '__main__':
    app.run_server(debug=True)
