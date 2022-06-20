import json
import os
import pathlib
import re

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
import cufflinks as cf

# Initialize app

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "US Opioid Epidemic"
server = app.server


# Load data

def calculate_decade(df: pd.DataFrame, year_column: str, period: int, start_year: int) -> pd.Series:
    end_year = df[year_column].max()

    year_range = end_year - start_year
    modulo = year_range % period

    final_start = end_year - period if modulo == 0 else end_year - modulo
    final_end = end_year + 1

    starts = np.arange(start_year, final_start, period).tolist()
    tuples = [(start, start + period) for start in starts]
    # We'll add the last period calculated earlier
    tuples.append(tuple([final_start, final_end]))
    bins = pd.IntervalIndex.from_tuples(tuples, closed='left')
    original_labels = list(bins)
    new_labels = [b.left for b in original_labels]
    label_dict = dict(zip(original_labels, new_labels))

    series = pd.cut(df[year_column], bins=bins, include_lowest=True, precision=0)

    return series.replace(label_dict)


APP_PATH = str(pathlib.Path(__file__).parent.resolve())

YEAR_COLUMN = 'bld_age'
PERIOD = 10
START_YEAR = 1900

df_lat_lon = pd.read_csv(
    os.path.join(APP_PATH, os.path.join("data", "lat_lon_counties.csv"))
)
df_lat_lon["FIPS "] = df_lat_lon["FIPS "].apply(lambda x: str(x).zfill(5))

raw_data = pd.read_csv(
    os.path.join(
        APP_PATH, os.path.join("data", "alldata_wgs84.csv")
    )
)
raw_data.dropna(subset=[YEAR_COLUMN], inplace=True)
raw_data[YEAR_COLUMN] = raw_data[YEAR_COLUMN].astype(int)
df_full_data = raw_data[raw_data[YEAR_COLUMN] >= START_YEAR]

df_full_data['decade'] = calculate_decade(df_full_data, YEAR_COLUMN, PERIOD, START_YEAR)
DECADES = np.unique(df_full_data['decade'].values.tolist())

BUILDING_MATERIAL_COLUMN_PREFIX = 'bcn_mate_'
BUILDING_MATERIALS = [f"{BUILDING_MATERIAL_COLUMN_PREFIX}{mat_id}" for mat_id in list(range(1, 7))]

for mat_id in BUILDING_MATERIALS:
    df_full_data[mat_id] = df_full_data[mat_id].apply(lambda x: float(0) if x >= 1 else x)

BINS = [f"{str(decade)}-{str(decade + PERIOD)}" for decade in DECADES]

DEFAULT_COLORSCALE = [
    "#f2fffb",
    "#bbffeb",
    "#98ffe0",
    "#79ffd6",
    "#6df0c8",
    "#69e7c0",
    "#59dab2",
    "#45d0a5",
    "#31c194",
    "#2bb489",
    "#25a27b",
    "#1e906d",
]

DEFAULT_OPACITY = 0.8

mapbox_access_token = "pk.eyJ1IjoiZGxhbXByaWFkaXMiLCJhIjoiY2trbjBtc21vMnZ4czJ1bW4xYXFyZm1heiJ9.xxVx5fDYx8WhvUH7klNOkw"
mapbox_style = "mapbox://styles/dlampriadis/ckmc4twhb9euf17pgd3kog7c0"

# App layout

app.layout = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                html.A(
                    html.Img(id="logo", src=app.get_asset_url("dash-logo.png")),
                    href="https://plotly.com/dash/",
                ),
                html.A(
                    html.Button("Enterprise Demo", className="link-button"),
                    href="https://plotly.com/get-demo/",
                ),
                html.A(
                    html.Button("Source Code", className="link-button"),
                    href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-opioid-epidemic",
                ),
                html.H4(children="WRITE SOMETHING HERE"),
                html.P(
                    id="description",
                    children="WRITE SOMETHING HERE",
                ),
            ],
        ),
        html.Div(
            id="app-container",
            children=[
                html.Div(
                    id="left-column",
                    children=[
                        html.Div(
                            id="slider-container",
                            children=[
                                html.P(
                                    id="slider-text",
                                    children="Drag the slider to change the decade:",
                                ),
                                dcc.Slider(
                                    id="years-slider",
                                    min=min(DECADES),
                                    max=max(DECADES),
                                    value=min(DECADES),
                                    marks={
                                        str(decade): {
                                            "label": str(decade),
                                            "style": {"color": "#7fafdf"},
                                        }
                                        for decade in DECADES
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            id="heatmap-container",
                            children=[
                                html.P(
                                    "Heatmap of buildings built in {0}'s".format(
                                        min(DECADES)
                                    ),
                                    id="heatmap-title",
                                ),
                                dcc.Graph(
                                    id="county-choropleth",
                                    figure=dict(
                                        layout=dict(
                                            mapbox=dict(
                                                layers=[],
                                                accesstoken=mapbox_access_token,
                                                style=mapbox_style,
                                                center=dict(
                                                    lat=41.3853, lon=2.1687
                                                ),
                                                pitch=0,
                                                zoom=15,
                                            ),
                                            autosize=True,
                                        ),
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="graph-container",
                    children=[
                        dcc.Graph(
                            id="selected-data",
                            figure=dict(
                                data=[dict(x=0, y=0)],
                                layout=dict(
                                    paper_bgcolor="#F4F4F8",
                                    plot_bgcolor="#F4F4F8",
                                    autofill=True,
                                    margin=dict(t=75, r=50, b=100, l=50),
                                ),
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("county-choropleth", "figure"),
    [Input("years-slider", "value")],
    [State("county-choropleth", "figure")],
)
def display_map(decade, figure):
    cm = dict(zip(BINS, DEFAULT_COLORSCALE))

    data = [
        dict(
            lat=df_full_data["y"],
            lon=df_full_data["x"],
            text=df_full_data.apply(lambda row: f"Building Id: {int(row['ogc_fid'])}<br>Built at: {int(row[YEAR_COLUMN])}",
                                    axis=1).values.tolist(),
            type="scattermapbox",
            hoverinfo="text",
            marker=dict(size=5, color="white", opacity=0),
        )
    ]

    annotations = [
        dict(
            showarrow=False,
            align="right",
            text="<b>Buildings<br>per year</b>",
            font=dict(color="#2cfec1"),
            bgcolor="#1f2630",
            x=0.95,
            y=0.95,
        )
    ]

    for i, bin in enumerate(reversed(BINS)):
        color = cm[bin]
        annotations.append(
            dict(
                arrowcolor=color,
                text=bin,
                x=0.95,
                y=0.85 - (i / 20),
                ax=-60,
                ay=0,
                arrowwidth=5,
                arrowhead=0,
                bgcolor="#1f2630",
                font=dict(color="#2cfec1"),
            )
        )

    if "layout" in figure:
        lat = figure["layout"]["mapbox"]["center"]["lat"]
        lon = figure["layout"]["mapbox"]["center"]["lon"]
        zoom = figure["layout"]["mapbox"]["zoom"]
    else:
        lat = 41.3853
        lon = 2.1687
        zoom = 15

    layout = dict(
        mapbox=dict(
            layers=[],
            accesstoken=mapbox_access_token,
            style=mapbox_style,
            center=dict(lat=lat, lon=lon),
            zoom=zoom,
        ),
        hovermode="closest",
        margin=dict(r=0, l=0, t=0, b=0),
        annotations=annotations,
        dragmode="lasso",
    )

    base_url = os.path.join(APP_PATH, "data\\geolayer\\")
    for bin in BINS:
        start_year, end_year = bin.split('-')
        with open(base_url + start_year + start_year + "-" + end_year + ".geojson") as json_file:
            geo_json = json.load(json_file)
            geo_layer = dict(
                source={
                    'type': "FeatureCollection",
                    'features': [dict(type='Feature', geometry=feat['geometry']) for feat in geo_json['features']]
                },
                type="fill",
                color=cm[bin],
                opacity=DEFAULT_OPACITY,
                # CHANGE THIS
                fill=dict(outlinecolor=cm[bin]),
            )
            layout["mapbox"]["layers"].append(geo_layer)

    fig = dict(data=data, layout=layout)
    return fig


@app.callback(Output("heatmap-title", "children"), [Input("years-slider", "value")])
def update_map_title(decade):
    return "Heatmap of buildings built in {0}'s".format(
        decade
    )


@app.callback(
    Output("selected-data", "figure"),
    [
        Input("county-choropleth", "selectedData"),
        Input("years-slider", "value"),
    ],
)
def display_selected_data(selectedData, decade):
    if selectedData is None:
        return dict(
            data=[dict(x=0, y=0)],
            layout=dict(
                title="Click-drag on the map to select buildings",
                paper_bgcolor="#1f2630",
                plot_bgcolor="#1f2630",
                font=dict(color="#2cfec1"),
                margin=dict(t=75, r=50, b=100, l=75),
            ),
        )
    pts = selectedData["points"]
    fips = [str(pt["text"].split("<br>")[0].split()[1].strip()) for pt in pts]
    dff = df_full_data.sort_values(YEAR_COLUMN)

    title = f"Construction Materials ratio in {decade}'s<br><b>{int(len(fips))}</b> buildings selected"
    materials_df = dff[dff["decade"] == decade][BUILDING_MATERIALS].fillna(0)
    ratio_df = pd.DataFrame(materials_df.mean(axis=0).values, columns=['Material Ratio'])

    fig = ratio_df.iplot(
        kind="bar", y="Material Ratio", title=title, asFigure=True
    )

    fig_layout = fig["layout"]
    fig_data = fig["data"]

    fig_data[0]["text"] = [f"{round(ratio[0] * 100, 2)}%" for ratio in ratio_df.values.tolist()]
    fig_data[0]["marker"]["color"] = "#2cfec1"
    fig_data[0]["marker"]["opacity"] = 1
    fig_data[0]["marker"]["line"]["width"] = 0
    fig_data[0]["textposition"] = "outside"
    fig_layout["paper_bgcolor"] = "#1f2630"
    fig_layout["plot_bgcolor"] = "#1f2630"
    fig_layout["font"]["color"] = "#2cfec1"
    fig_layout["title"]["font"]["color"] = "#2cfec1"
    fig_layout["xaxis"]["tickfont"]["color"] = "#2cfec1"
    fig_layout["yaxis"]["tickfont"]["color"] = "#2cfec1"
    fig_layout["xaxis"]["gridcolor"] = "#5b5b5b"
    fig_layout["yaxis"]["gridcolor"] = "#5b5b5b"

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
