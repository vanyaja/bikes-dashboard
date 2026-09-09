# London Bikes Dashboard

Interactive Dash app exploring the London bikes dataset and predicting daily
hires from real weather (Open-Meteo), using coefficients from a regression
model fitted in `bikes_assignment.ipynb`.

## Run locally

```
uv sync
uv run python app.py
```

Open http://localhost:8050

## Deploy on Render

- Build command: `pip install uv && uv sync`
- Start command: `uv run gunicorn app:server`

No API key or environment variables are needed.
