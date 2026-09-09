"""
London Bikes dashboard - explore demand drivers and predict hires from
live/historical weather. Built with Dash; deployed on Render.
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, dcc, html, Input, Output

from open_meteo import open_meteo, open_meteo_history

DATA_URL = "https://raw.githubusercontent.com/kostis-christodoulou/am01-code-sep2026/main/data/london_bikes.csv"
COEF_PATH = os.path.join(os.path.dirname(__file__), "model_coefficients.csv")
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

WEATHER_VARS = {
    "temp": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "precip": "Precipitation (mm)",
    "windspeed": "Wind speed (km/h)",
    "cloudcover": "Cloud cover (%)",
}

# --- Palette: fixed categorical slots (never cycled), one hue per role ---
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE, PAGE = "#e1e0d9", "#fcfcfb", "#f9f9f7"

WEEKEND_COLORS = {"False": BLUE, "True": ORANGE}
# Skips the blue/orange pair used for weekend so the two encodings never clash,
# and keeps season pairs well separated (not adjacent palette slots).
SEASON_COLORS = {"Winter": BLUE, "Spring": AQUA, "Summer": YELLOW, "Autumn": RED}
SEASON_SYMBOLS = {"Winter": "circle", "Spring": "diamond", "Summer": "square", "Autumn": "triangle-up"}

pio.templates["bikes"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_2, size=13),
        title=dict(font=dict(color=INK, size=16)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK_MUTED)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK_MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=48, r=24, t=48, b=44),
    )
)
pio.templates.default = "bikes"


def load_bikes():
    df = pd.read_csv(DATA_URL)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=DAY_ORDER, ordered=True)
    df["season_name"] = pd.Categorical(df["season_name"], categories=SEASON_ORDER, ordered=True)
    df = df[df["date"] >= pd.to_datetime("2014-01-01", utc=True)].copy()
    df["weekend"] = df["weekend"].astype(str)
    df["year"] = df["date"].dt.year
    return df


def load_coefficients():
    coef = pd.read_csv(COEF_PATH)
    return dict(zip(coef["term"], coef["coefficient"]))


bike = load_bikes()
coefficients = load_coefficients()
NUMERIC_TERMS = [t for t in coefficients if t != "Intercept" and not t.startswith("day_")]
YEAR_MIN, YEAR_MAX = int(bike["year"].min()), int(bike["year"].max())


def predict_hires(weather: pd.DataFrame) -> pd.DataFrame:
    """Apply the exported linear model to a weather DataFrame (date, day_of_week, ...)."""
    out = weather.copy()
    pred = coefficients["Intercept"]
    for term in NUMERIC_TERMS:
        pred = pred + coefficients[term] * out[term]
    day_effect = out["day_of_week"].map(lambda d: coefficients.get(f"day_{d}", 0.0))
    out["predicted_hires"] = (pred + day_effect).round(0)
    return out


def fetch_predictions():
    """Fetch both weather windows and apply the model; return (jan_df, forecast_df, error)."""
    try:
        jan_weather = open_meteo_history("London", "2026-01-01", "2026-01-07")
        forecast_weather = open_meteo("London", 5)
        return predict_hires(jan_weather), predict_hires(forecast_weather), None
    except Exception as exc:  # keep the page alive even if Open-Meteo is unreachable
        return None, None, str(exc)


app = Dash(__name__)
server = app.server
app.title = "London Bikes Dashboard"


def stat_tile(label, value, sublabel=None):
    return html.Div(
        [
            html.Div(label, className="tile-label"),
            html.Div(value, className="tile-value"),
            html.Div(sublabel, className="tile-sublabel") if sublabel else None,
        ],
        className="tile",
    )


def prediction_section(title, df):
    if df is None:
        return html.Div("Weather data unavailable right now — try refreshing.", className="error-note")

    avg = df["predicted_hires"].mean()
    peak_row = df.loc[df["predicted_hires"].idxmax()]
    kpi_row = html.Div(
        [
            stat_tile("Average predicted hires", f"{avg:,.0f}"),
            stat_tile("Busiest day", peak_row["day_of_week"], peak_row["date"].strftime("%d %b")),
        ],
        className="tile-row",
    )

    fig = px.bar(
        df,
        x=df["date"].dt.strftime("%a %d %b"),
        y="predicted_hires",
        text="predicted_hires",
        labels={"x": "Day", "predicted_hires": "Predicted hires"},
        color_discrete_sequence=[BLUE],
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, yaxis_title="Predicted hires", xaxis_title=None)

    table = html.Table(
        [
            html.Tr(
                [html.Th(c) for c in ["Date", "Day", "Temp (°C)", "Humidity", "Precip", "Wind", "Cloud", "Predicted hires"]]
            )
        ]
        + [
            html.Tr(
                [
                    html.Td(row["date"].strftime("%Y-%m-%d")),
                    html.Td(row["day_of_week"]),
                    html.Td(f"{row['temp']:.1f}"),
                    html.Td(f"{row['humidity']:.0f}%"),
                    html.Td(f"{row['precip']:.1f} mm"),
                    html.Td(f"{row['windspeed']:.1f} km/h"),
                    html.Td(f"{row['cloudcover']:.0f}%"),
                    html.Td(f"{row['predicted_hires']:,.0f}"),
                ]
            )
            for _, row in df.iterrows()
        ],
        className="pred-table",
    )
    return html.Div([html.H4(title), kpi_row, dcc.Graph(figure=fig, config={"displayModeBar": False}), table], className="card")


jan_pred, forecast_pred, fetch_error = fetch_predictions()

explore_tab = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Weather variable"),
                        dcc.Dropdown(
                            id="weather-var",
                            options=[{"label": v, "value": k} for k, v in WEATHER_VARS.items()],
                            value="temp",
                            clearable=False,
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("Colour by"),
                        dcc.RadioItems(
                            id="colour-by",
                            options=[
                                {"label": "Weekend", "value": "weekend"},
                                {"label": "Season", "value": "season_name"},
                            ],
                            value="weekend",
                            inline=True,
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label(id="year-range-label"),
                        dcc.RangeSlider(
                            id="year-range",
                            min=YEAR_MIN,
                            max=YEAR_MAX,
                            step=1,
                            value=[YEAR_MIN, YEAR_MAX],
                            marks={y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1, 2)},
                            tooltip={"placement": "bottom"},
                        ),
                    ],
                    className="control control-wide",
                ),
            ],
            className="controls-row card",
        ),
        html.Div(id="kpi-row"),
        html.Div(dcc.Graph(id="scatter-plot", config={"displayModeBar": False}), className="card"),
        html.Div(dcc.Graph(id="weekday-bar", config={"displayModeBar": False}), className="card"),
    ]
)

predict_tab = html.Div(
    [
        html.P(
            "Predictions from the exported regression model, using real weather from Open-Meteo.",
            className="subtitle",
        ),
        html.Div("Could not reach Open-Meteo: " + fetch_error, className="error-note") if fetch_error else None,
        prediction_section("First week of January 2026 (historical weather)", jan_pred),
        prediction_section("Next 5 days (live forecast)", forecast_pred),
    ]
)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2("London Bikes"),
                html.P("Demand explorer & forecast for TfL's bike scheme", className="subtitle"),
            ],
            className="header",
        ),
        dcc.Tabs(
            [
                dcc.Tab(label="Explore the data", children=[explore_tab], className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Predict", children=[predict_tab], className="tab", selected_className="tab--selected"),
            ]
        ),
    ],
    className="app-container",
)


@app.callback(
    Output("scatter-plot", "figure"),
    Output("weekday-bar", "figure"),
    Output("kpi-row", "children"),
    Output("year-range-label", "children"),
    Input("weather-var", "value"),
    Input("colour-by", "value"),
    Input("year-range", "value"),
)
def update_explore(weather_var, colour_by, year_range):
    lo, hi = year_range
    filtered = bike[(bike["year"] >= lo) & (bike["year"] <= hi)]

    color_map = WEEKEND_COLORS if colour_by == "weekend" else SEASON_COLORS
    symbol_arg = {"symbol": "season_name", "symbol_map": SEASON_SYMBOLS} if colour_by == "season_name" else {}

    scatter = px.scatter(
        filtered,
        x=weather_var,
        y="bikes_hired",
        color=colour_by,
        color_discrete_map=color_map,
        trendline="ols",
        trendline_scope="overall",
        opacity=0.55,
        labels={weather_var: WEATHER_VARS[weather_var], "bikes_hired": "Bikes hired", "weekend": "Weekend", "season_name": "Season"},
        title=f"Daily hires vs {WEATHER_VARS[weather_var].lower()}",
        **symbol_arg,
    )
    for trace in scatter.data:
        if trace.mode == "lines":
            trace.line.color = INK
            trace.line.width = 2
            trace.line.dash = "dot"
            trace.showlegend = False

    avg_by_day = filtered.groupby("day_of_week", observed=True)["bikes_hired"].mean().reindex(DAY_ORDER)
    bar = go.Figure(
        go.Bar(
            x=avg_by_day.index,
            y=avg_by_day.values,
            marker_color=BLUE,
            text=avg_by_day.values,
            texttemplate="%{text:,.0f}",
            textposition="outside",
        )
    )
    bar.update_layout(
        title="Average hires by day of week",
        xaxis_title=None,
        yaxis_title="Average bikes hired",
    )

    corr = filtered[weather_var].corr(filtered["bikes_hired"])
    busiest = avg_by_day.idxmax()
    kpis = html.Div(
        [
            stat_tile("Days in range", f"{len(filtered):,}"),
            stat_tile("Average daily hires", f"{filtered['bikes_hired'].mean():,.0f}"),
            stat_tile("Correlation with " + WEATHER_VARS[weather_var].split(" (")[0].lower(), f"{corr:+.2f}"),
            stat_tile("Busiest day of week", busiest),
        ],
        className="tile-row",
    )
    label = f"Year range: {lo}–{hi}"
    return scatter, bar, kpis, label


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
