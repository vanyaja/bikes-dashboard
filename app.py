"""
London Bikes dashboard - explore demand drivers and predict hires from
live/historical weather. Built with Dash; deployed on Render.
"""

import os
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from open_meteo import open_meteo, open_meteo_history

DATA_URL = "https://raw.githubusercontent.com/kostis-christodoulou/am01-code-sep2026/main/data/london_bikes.csv"
COEF_PATH = os.path.join(os.path.dirname(__file__), "model_coefficients.csv")
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

WEATHER_VARS = {
    "temp": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "precip": "Precipitation (mm)",
    "windspeed": "Wind speed (km/h)",
    "cloudcover": "Cloud cover (%)",
}


def load_bikes():
    df = pd.read_csv(DATA_URL)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=DAY_ORDER, ordered=True)
    df = df[df["date"] >= pd.to_datetime("2014-01-01", utc=True)].copy()
    return df


def load_coefficients():
    coef = pd.read_csv(COEF_PATH)
    return dict(zip(coef["term"], coef["coefficient"]))


bike = load_bikes()
coefficients = load_coefficients()
NUMERIC_TERMS = [t for t in coefficients if t != "Intercept" and not t.startswith("day_")]


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
        jan_pred = predict_hires(jan_weather)
        forecast_pred = predict_hires(forecast_weather)
        return jan_pred, forecast_pred, None
    except Exception as exc:  # keep the page alive even if Open-Meteo is unreachable
        return None, None, str(exc)


app = Dash(__name__)
server = app.server
app.title = "London Bikes Dashboard"


def prediction_block(title, df):
    if df is None:
        return html.Div("Weather data unavailable right now - try refreshing.", className="error-note")
    table = html.Table(
        [html.Tr([html.Th(c) for c in ["Date", "Day", "Temp", "Humidity", "Precip", "Wind", "Cloud", "Predicted hires"]])]
        + [
            html.Tr(
                [
                    html.Td(row["date"].strftime("%Y-%m-%d")),
                    html.Td(row["day_of_week"]),
                    html.Td(f"{row['temp']:.1f}"),
                    html.Td(f"{row['humidity']:.0f}"),
                    html.Td(f"{row['precip']:.1f}"),
                    html.Td(f"{row['windspeed']:.1f}"),
                    html.Td(f"{row['cloudcover']:.0f}"),
                    html.Td(f"{row['predicted_hires']:,.0f}"),
                ]
            )
            for _, row in df.iterrows()
        ],
        className="pred-table",
    )
    fig = px.bar(
        df,
        x=df["date"].dt.strftime("%a %d %b"),
        y="predicted_hires",
        labels={"x": "Day", "predicted_hires": "Predicted hires"},
        title=title,
    )
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
    return html.Div([html.H4(title), dcc.Graph(figure=fig), table])


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
            ],
            className="controls-row",
        ),
        dcc.Graph(id="scatter-plot"),
        dcc.Graph(id="weekday-bar"),
    ]
)

predict_tab = html.Div(
    [
        html.P(
            "Predictions from the exported regression model, using real weather from Open-Meteo.",
            className="subtitle",
        ),
        html.Div("Could not reach Open-Meteo: " + fetch_error, className="error-note") if fetch_error else None,
        prediction_block("First week of January 2026 (historical weather)", jan_pred),
        html.Br(),
        prediction_block("Next 5 days (live forecast)", forecast_pred),
    ]
)

app.layout = html.Div(
    [
        html.H2("London Bikes: demand explorer & forecast"),
        dcc.Tabs(
            [
                dcc.Tab(label="Explore the data", children=[explore_tab]),
                dcc.Tab(label="Predict", children=[predict_tab]),
            ]
        ),
    ],
    className="app-container",
)


@app.callback(
    Output("scatter-plot", "figure"),
    Output("weekday-bar", "figure"),
    Input("weather-var", "value"),
    Input("colour-by", "value"),
)
def update_explore(weather_var, colour_by):
    scatter = px.scatter(
        bike,
        x=weather_var,
        y="bikes_hired",
        color=colour_by,
        opacity=0.5,
        labels={weather_var: WEATHER_VARS[weather_var], "bikes_hired": "Bikes hired"},
        title=f"Daily hires vs {WEATHER_VARS[weather_var].lower()}",
    )
    scatter.update_layout(margin=dict(l=40, r=20, t=40, b=40))

    avg_by_day = bike.groupby("day_of_week", observed=True)["bikes_hired"].mean().reindex(DAY_ORDER)
    bar = go.Figure(go.Bar(x=avg_by_day.index, y=avg_by_day.values))
    bar.update_layout(
        title="Average hires by day of week",
        xaxis_title="Day",
        yaxis_title="Average bikes hired",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return scatter, bar


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
