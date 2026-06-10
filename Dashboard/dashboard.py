#!/usr/bin/env python3
"""
Seeverkehr Analytics Dashboard
================================
Interaktives Analyse-Dashboard für Destatis Seeverkehrsstatistik-Daten.
Plotly Dash + scikit-learn Machine Learning für Visualisierung und Prognosen.

Start: python dashboard.py
URL:   http://127.0.0.1:8050

Datenpfad: ../Dataset bereinigung/Datasets_cleaned/seeverkehr_bereinigt_*.csv
Fallback:  Automatische Generierung realistischer Demo-Daten
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Pfade
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CLEANED_DIR = BASE_DIR.parent / "Dataset bereinigung" / "Datasets_cleaned"

# ─────────────────────────────────────────────────────────────────────────────
# Konstanten
# ─────────────────────────────────────────────────────────────────────────────
METRIC_OPTIONS = [
    {"label": "Tonnage (t)", "value": "Tonnen"},
    {"label": "TEU (Container)", "value": "TEU"},
    {"label": "Ladeeinheiten", "value": "Anzahl_Ladungstraeger"},
]

MODEL_OPTIONS = [
    {"label": "Random Forest", "value": "random_forest"},
    {"label": "Gradient Boosting", "value": "gradient_boosting"},
    {"label": "Lineare Regression", "value": "linear"},
    {"label": "Polynomiale Regression (Grad 2)", "value": "polynomial"},
]

VERKEHRSBEZIEHUNG_MAP = {
    1: "Eingehend", 2: "Ausgehend", 3: "Durchgehend", 4: "Küstenverkehr",
}

ISO_TO_MAKROREGION: dict[str, str] = {
    "AT": "Europa (EU/EEA)", "BE": "Europa (EU/EEA)", "BG": "Europa (EU/EEA)",
    "CY": "Europa (EU/EEA)", "CZ": "Europa (EU/EEA)", "DE": "Europa (EU/EEA)",
    "DK": "Europa (EU/EEA)", "EE": "Europa (EU/EEA)", "ES": "Europa (EU/EEA)",
    "FI": "Europa (EU/EEA)", "FR": "Europa (EU/EEA)", "GR": "Europa (EU/EEA)",
    "HR": "Europa (EU/EEA)", "HU": "Europa (EU/EEA)", "IE": "Europa (EU/EEA)",
    "IT": "Europa (EU/EEA)", "LT": "Europa (EU/EEA)", "LU": "Europa (EU/EEA)",
    "LV": "Europa (EU/EEA)", "MT": "Europa (EU/EEA)", "NL": "Europa (EU/EEA)",
    "PL": "Europa (EU/EEA)", "PT": "Europa (EU/EEA)", "RO": "Europa (EU/EEA)",
    "SE": "Europa (EU/EEA)", "SI": "Europa (EU/EEA)", "SK": "Europa (EU/EEA)",
    "NO": "Europa (EU/EEA)", "IS": "Europa (EU/EEA)", "LI": "Europa (EU/EEA)",
    "GB": "Europa (nicht EU/EEA)", "CH": "Europa (nicht EU/EEA)",
    "TR": "Europa (nicht EU/EEA)", "UA": "Osteuropa/GUS", "RU": "Osteuropa/GUS",
    "MA": "Nordafrika", "DZ": "Nordafrika", "TN": "Nordafrika", "EG": "Nordafrika",
    "NG": "Westafrika", "GH": "Westafrika", "CI": "Westafrika", "AO": "Westafrika",
    "ZA": "Ost-/Suedafrika", "KE": "Ost-/Suedafrika", "TZ": "Ost-/Suedafrika",
    "SA": "Naher Osten", "AE": "Naher Osten", "QA": "Naher Osten", "IR": "Naher Osten",
    "IN": "Suedasien", "PK": "Suedasien", "BD": "Suedasien", "LK": "Suedasien",
    "CN": "Ostasien", "JP": "Ostasien", "KR": "Ostasien", "HK": "Ostasien",
    "SG": "Suedostasien", "TH": "Suedostasien", "VN": "Suedostasien", "ID": "Suedostasien",
    "MY": "Suedostasien", "PH": "Suedostasien",
    "US": "Nordamerika", "CA": "Nordamerika", "MX": "Nordamerika",
    "PA": "Mittelamerika/Karibik", "CR": "Mittelamerika/Karibik",
    "BR": "Suedamerika", "AR": "Suedamerika", "CL": "Suedamerika", "CO": "Suedamerika",
    "AU": "Ozeanien", "NZ": "Ozeanien",
}

DARK_CARD = {"background": "#1e1e30", "border": "1px solid #2d2d45"}
ACCENT_COLOR = "#00d4aa"
CHART_TEMPLATE = "plotly_dark"


# ─────────────────────────────────────────────────────────────────────────────
# Datenladen & Demo-Generierung
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, bool]:
    """Lädt bereinigten Datensatz oder generiert Demo-Daten. Gibt (df, is_demo) zurück."""
    csv_files = sorted(CLEANED_DIR.glob("seeverkehr_bereinigt_*.csv")) if CLEANED_DIR.exists() else []
    if csv_files:
        df = pd.read_csv(csv_files[-1], sep=";", encoding="utf-8-sig", decimal=",", low_memory=False)
        return _prepare(df), False
    return _prepare(_generate_demo()), True


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Tonnen", "TEU", "Anzahl_Ladungstraeger"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Referenzzeitraum_Jahr", "Referenzzeitraum_Monat"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"Referenzzeitraum_Jahr", "Referenzzeitraum_Monat"}.issubset(df.columns):
        df["Datum"] = pd.to_datetime(
            df["Referenzzeitraum_Jahr"].astype("Int64").astype(str) + "-" +
            df["Referenzzeitraum_Monat"].astype("Int64").astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
    if "Verkehrsbeziehung" in df.columns:
        df["Verkehrsbeziehung"] = pd.to_numeric(df["Verkehrsbeziehung"], errors="coerce")
        df["Verkehrsbeziehung_Label"] = df["Verkehrsbeziehung"].map(VERKEHRSBEZIEHUNG_MAP).fillna("Unbekannt")
    if "Einladeregion_Makroregion" not in df.columns and "Einladeregion_ISO" in df.columns:
        df["Einladeregion_Makroregion"] = df["Einladeregion_ISO"].map(ISO_TO_MAKROREGION).fillna("Unbekannt")
    if "Ausladeregion_Makroregion" not in df.columns and "Ausladeregion_ISO" in df.columns:
        df["Ausladeregion_Makroregion"] = df["Ausladeregion_ISO"].map(ISO_TO_MAKROREGION).fillna("Unbekannt")
    return df


def _generate_demo(n: int = 60_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    years = np.arange(2015, 2025)
    ports = {
        "21111": "Hamburg", "21112": "Bremen/Bremerhaven", "21113": "Rostock",
        "21114": "Lübeck", "21115": "Kiel", "21116": "Wismar",
    }
    port_keys = list(ports.keys())
    port_probs = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]

    year_arr = rng.choice(years, n)
    month_arr = rng.choice(np.arange(1, 13), n)
    port_arr = rng.choice(port_keys, n, p=port_probs)

    trend = 1 + (year_arr - 2015) * 0.025
    seasonal = 1 + 0.18 * np.sin(2 * np.pi * month_arr / 12)
    port_factor = np.where(port_arr == "21111", 2.5, np.where(port_arr == "21112", 1.8, 1.0))
    tonnen = rng.lognormal(np.log(4500 * trend * seasonal * port_factor), 1.1).round(1)

    einlade_iso = rng.choice(
        ["CN", "US", "NL", "DE", "GB", "JP", "KR", "SG", "BR", "IN", "AE", "AU"],
        n, p=[0.18, 0.11, 0.10, 0.12, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.06, 0.06],
    )
    auslade_iso = rng.choice(
        ["DE", "NL", "BE", "PL", "FR", "DK", "SE", "FI", "IT", "ES"],
        n, p=[0.30, 0.15, 0.10, 0.10, 0.09, 0.08, 0.07, 0.05, 0.03, 0.03],
    )

    return pd.DataFrame({
        "EVAS": port_arr,
        "EVAS_Label": [ports[p] for p in port_arr],
        "Referenzzeitraum_Jahr": year_arr,
        "Referenzzeitraum_Monat": month_arr,
        "Einladeregion_ISO": einlade_iso,
        "Ausladeregion_ISO": auslade_iso,
        "Einladeregion_Makroregion": [ISO_TO_MAKROREGION.get(c, "Unbekannt") for c in einlade_iso],
        "Ausladeregion_Makroregion": [ISO_TO_MAKROREGION.get(c, "Europa (EU/EEA)") for c in auslade_iso],
        "Verkehrsbeziehung": rng.choice([1, 2, 3, 4], n, p=[0.45, 0.43, 0.07, 0.05]),
        "Schiffsart": rng.choice(
            ["Containerschiff", "Tanker", "Massengutschiff", "Ro-Ro-Schiff", "Stückgutschiff", "Sonstiges"],
            n, p=[0.28, 0.22, 0.20, 0.12, 0.10, 0.08],
        ),
        "Flagge": rng.choice(
            ["DE", "NL", "PA", "LR", "MH", "BS", "CY", "MT", "SG", "HK"],
            n, p=[0.14, 0.10, 0.12, 0.10, 0.09, 0.08, 0.08, 0.08, 0.11, 0.10],
        ),
        "NST2007": rng.choice(
            ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"], n
        ),
        "Tonnen": tonnen,
        "TEU": np.where(rng.random(n) < 0.38, rng.lognormal(4.6, 1.0, n).round(0), np.nan),
        "Anzahl_Ladungstraeger": np.where(rng.random(n) < 0.4, rng.integers(1, 50, n).astype(float), np.nan),
        "Quelldatei": "demo_data.csv",
    })


# ─────────────────────────────────────────────────────────────────────────────
# ML Forecasting Engine
# ─────────────────────────────────────────────────────────────────────────────

def _make_features(n: int, months: np.ndarray) -> pd.DataFrame:
    t = np.arange(n)
    return pd.DataFrame({
        "t": t,
        "t2": t ** 2,
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12),
        "q_sin": np.sin(2 * np.pi * ((months - 1) // 3 + 1) / 4),
        "q_cos": np.cos(2 * np.pi * ((months - 1) // 3 + 1) / 4),
    })


def _add_lag_features(X: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    X = X.copy()
    for lag in [1, 2, 3, 6, 12]:
        lagged = np.full(len(values), np.nan)
        if len(values) > lag:
            lagged[lag:] = values[:-lag]
        X[f"lag_{lag}"] = lagged
    for w in [3, 6, 12]:
        X[f"roll_{w}"] = pd.Series(values).rolling(w, min_periods=1).mean().values
    return X


def build_forecast(
    series: pd.Series,
    model_name: str = "random_forest",
    horizon: int = 12,
) -> dict:
    """Trainiert Modell und prognostiziert horizon Monate. Gibt Ergebnis-Dict zurück."""
    ts = series.dropna().sort_index()
    if len(ts) < 18:
        return {"error": f"Zu wenige Datenpunkte: {len(ts)} (mind. 18 benötigt)"}

    values = ts.values.astype(float)
    n = len(values)

    if hasattr(ts.index, "month"):
        months = ts.index.month.values
        last_date = ts.index[-1]
    else:
        months = np.array([(i % 12) + 1 for i in range(n)])
        last_date = None

    X = _make_features(n, months)
    X = _add_lag_features(X, values)
    col_means = X.mean()
    X = X.fillna(col_means)

    test_size = min(12, max(6, n // 6))
    train_end = n - test_size

    X_tr, X_te = X.iloc[:train_end], X.iloc[train_end:]
    y_tr, y_te = values[:train_end], values[train_end:]

    _models: dict = {
        "linear": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "polynomial": Pipeline([
            ("poly", PolynomialFeatures(2, include_bias=False)),
            ("sc", StandardScaler(with_mean=False)),
            ("m", Ridge(alpha=1.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            n_jobs=-1, random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
    }

    model = _models.get(model_name, _models["random_forest"])
    model.fit(X_tr, y_tr)

    y_pred_te = np.maximum(0, model.predict(X_te))
    mae = mean_absolute_error(y_te, y_pred_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred_te)))
    r2 = r2_score(y_te, y_pred_te)

    # Rollierendes Forecast (autoregressive Vorhersage)
    buffer = list(values)
    forecast_vals = []
    if last_date is not None:
        future_dates = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        future_months = future_dates.month.values
    else:
        future_months = np.array([((months[-1] - 1 + h) % 12) + 1 for h in range(1, horizon + 1)])
        future_dates = None

    for h in range(horizon):
        new_t = n + h
        m = future_months[h]
        row = {
            "t": new_t, "t2": new_t ** 2,
            "month_sin": np.sin(2 * np.pi * m / 12),
            "month_cos": np.cos(2 * np.pi * m / 12),
            "q_sin": np.sin(2 * np.pi * ((m - 1) // 3 + 1) / 4),
            "q_cos": np.cos(2 * np.pi * ((m - 1) // 3 + 1) / 4),
        }
        for lag in [1, 2, 3, 6, 12]:
            idx = len(buffer) - lag
            row[f"lag_{lag}"] = buffer[idx] if idx >= 0 else col_means.get(f"lag_{lag}", np.mean(buffer))
        for w in [3, 6, 12]:
            row[f"roll_{w}"] = float(np.mean(buffer[-w:])) if len(buffer) >= w else float(np.mean(buffer))

        X_row = pd.DataFrame([row])[X_tr.columns].fillna(col_means)
        pred = float(max(0, model.predict(X_row)[0]))
        forecast_vals.append(pred)
        buffer.append(pred)

    forecast_arr = np.array(forecast_vals)
    uncertainty = rmse * np.sqrt(np.arange(1, horizon + 1))
    lower_ci = np.maximum(0, forecast_arr - 1.96 * uncertainty)
    upper_ci = forecast_arr + 1.96 * uncertainty

    fi: dict = {}
    raw_model = model.named_steps["m"] if hasattr(model, "named_steps") else model
    if hasattr(raw_model, "feature_importances_"):
        feat_names = list(X_tr.columns)
        fi = dict(sorted(zip(feat_names, raw_model.feature_importances_), key=lambda x: -x[1]))

    return {
        "forecast": forecast_arr,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "future_dates": future_dates,
        "metrics": {"MAE": round(mae, 1), "RMSE": round(rmse, 1), "R²": round(r2, 3)},
        "feature_importance": fi,
        "test_actual": y_te,
        "test_pred": y_pred_te,
        "train_end_idx": train_end,
        "ts_index": ts.index,
        "ts_values": values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen für Charts
# ─────────────────────────────────────────────────────────────────────────────

def _chart_layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        height=height,
        title=dict(text=title, font=dict(size=14, color="#ccc")) if title else None,
        legend=dict(font=dict(size=11, color="#aaa")),
        font=dict(color="#ccc"),
    )
    fig.update_xaxes(gridcolor="#2a2a3e", zerolinecolor="#2a2a3e")
    fig.update_yaxes(gridcolor="#2a2a3e", zerolinecolor="#2a2a3e")
    return fig


def _fmt_num(v: float) -> str:
    if v >= 1e9:
        return f"{v/1e9:.2f} Mrd."
    if v >= 1e6:
        return f"{v/1e6:.1f} Mio."
    if v >= 1e3:
        return f"{v/1e3:.1f} Tsd."
    return f"{v:.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard-App Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

DF, IS_DEMO = load_data()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="Seeverkehr Analytics",
)
server = app.server  # für Deployment (z.B. Gunicorn)


# ─────────────────────────────────────────────────────────────────────────────
# Layout-Bausteine
# ─────────────────────────────────────────────────────────────────────────────

def _kpi_card(icon: str, title: str, value: str, color: str, subtitle: str = "") -> dbc.Col:
    return dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(html.I(className=f"fas {icon} fa-2x", style={"color": color}),
                         className="col-3 d-flex align-items-center justify-content-center"),
                html.Div([
                    html.P(title, className="text-muted small mb-1"),
                    html.H5(value, className="fw-bold mb-0", style={"color": color}),
                    html.P(subtitle, className="text-muted small mb-0") if subtitle else None,
                ], className="col-9"),
            ], className="row g-0"),
        ], className="py-3"),
    ], style={**DARK_CARD, "borderLeft": f"3px solid {color}"}), md=3, sm=6, className="mb-3")


def _section_card(title: str, content, icon: str = "fa-chart-line") -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader([
            html.I(className=f"fas {icon} me-2", style={"color": ACCENT_COLOR}),
            html.Strong(title, className="text-light"),
        ], style={"background": "#181828", "borderBottom": "1px solid #2d2d45"}),
        dbc.CardBody(content, style={"background": "#1e1e30"}),
    ], style=DARK_CARD, className="mb-4 shadow")


# ── Haupt-Layout ──────────────────────────────────────────────────────────────

years = sorted(DF["Referenzzeitraum_Jahr"].dropna().unique().astype(int).tolist())
ports = sorted(DF["EVAS_Label"].dropna().unique().tolist()) if "EVAS_Label" in DF.columns else []

app.layout = dbc.Container([

    # Header
    dbc.Row(dbc.Col(dbc.Navbar([
        html.Div([
            html.I(className="fas fa-ship me-3 fa-2x", style={"color": ACCENT_COLOR}),
            html.Div([
                html.H4("Seeverkehr Analytics", className="mb-0 fw-bold"),
                html.Small("Destatis Seeverkehrsstatistik · ML-gestützte Prognosen", className="text-muted"),
            ]),
        ], className="d-flex align-items-center"),
        dbc.Badge(
            "DEMO-DATEN — keine CSV im Datasets_cleaned-Ordner gefunden" if IS_DEMO else "Echtdaten geladen",
            color="warning" if IS_DEMO else "success",
            className="ms-auto fs-6 px-3 py-2",
        ),
    ], dark=True, color="dark", className="px-4 rounded shadow")), className="mb-4"),

    # ── Filter ────────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(_section_card("Globale Filter", dbc.Row([
            dbc.Col([
                dbc.Label("Zeitraum (Jahre)", className="small text-muted"),
                dcc.RangeSlider(
                    id="year-slider",
                    min=years[0] if years else 2015,
                    max=years[-1] if years else 2024,
                    value=[years[0] if years else 2015, years[-1] if years else 2024],
                    marks={y: {"label": str(y), "style": {"color": "#888"}} for y in years[::2]} if years else {},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], md=5),
            dbc.Col([
                dbc.Label("Hafen(e)", className="small text-muted"),
                dcc.Dropdown(
                    id="port-filter",
                    options=[{"label": p, "value": p} for p in ports],
                    value=[],
                    multi=True,
                    placeholder="Alle Häfen",
                    style={"background": "#252538"},
                ),
            ], md=4),
            dbc.Col([
                dbc.Label("Hauptmetrik", className="small text-muted"),
                dcc.Dropdown(
                    id="metric-selector",
                    options=METRIC_OPTIONS,
                    value="Tonnen",
                    clearable=False,
                    style={"background": "#252538"},
                ),
            ], md=3),
        ]), icon="fa-sliders-h"), width=12),
    ]),

    # ── KPI-Karten ────────────────────────────────────────────────────────────
    dbc.Row(id="kpi-row", className="mb-2"),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dbc.Tabs([
        dbc.Tab(html.Div(id="tab-overview"), label="Übersicht",
                tab_id="overview", className="pt-3"),
        dbc.Tab(html.Div(id="tab-timeseries"), label="Zeitreihe",
                tab_id="timeseries", className="pt-3"),
        dbc.Tab(html.Div(id="tab-forecast"), label="ML-Prognose",
                tab_id="forecast", className="pt-3"),
        dbc.Tab(html.Div(id="tab-regions"), label="Regionen",
                tab_id="regions", className="pt-3"),
        dbc.Tab(html.Div(id="tab-ports"), label="Häfen & Schiffe",
                tab_id="ports", className="pt-3"),
    ], id="main-tabs", active_tab="overview",
       className="nav-pills",
       style={"borderBottom": f"2px solid {ACCENT_COLOR}20"}),

    # Footer
    html.Hr(style={"borderColor": "#2d2d45", "marginTop": "3rem"}),
    html.P([
        html.I(className="fas fa-code me-2", style={"color": ACCENT_COLOR}),
        "Seeverkehr Analytics · Plotly Dash · scikit-learn · Destatis Open Data",
    ], className="text-muted small text-center pb-3"),

], fluid=True, style={"background": "#12121e", "minHeight": "100vh", "paddingTop": "1.5rem"})


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktion: gefilterte Daten holen
# ─────────────────────────────────────────────────────────────────────────────

def _filter(years_range: list, ports: list) -> pd.DataFrame:
    df = DF.copy()
    if years_range:
        df = df[df["Referenzzeitraum_Jahr"].between(years_range[0], years_range[1])]
    if ports:
        df = df[df["EVAS_Label"].isin(ports)]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

# ── KPI-Karten ────────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-row", "children"),
    Input("year-slider", "value"),
    Input("port-filter", "value"),
    Input("metric-selector", "value"),
)
def update_kpi(years_range, ports_sel, metric):
    df = _filter(years_range, ports_sel)
    if df.empty:
        return []

    total_tonnen = df["Tonnen"].sum() if "Tonnen" in df.columns else 0
    total_teu = df["TEU"].dropna().sum() if "TEU" in df.columns else 0
    n_records = len(df)
    n_ports = df["EVAS_Label"].nunique() if "EVAS_Label" in df.columns else 0

    return [
        _kpi_card("fa-weight-hanging", "Gesamttonnage", _fmt_num(total_tonnen) + " t", ACCENT_COLOR),
        _kpi_card("fa-box", "Gesamt-TEU", _fmt_num(total_teu), "#e67e22"),
        _kpi_card("fa-database", "Datensätze", _fmt_num(n_records), "#9b59b6"),
        _kpi_card("fa-anchor", "Aktive Häfen", str(n_ports), "#3498db"),
    ]


# ── Übersicht ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-overview", "children"),
    Input("year-slider", "value"),
    Input("port-filter", "value"),
    Input("metric-selector", "value"),
    Input("main-tabs", "active_tab"),
)
def update_overview(years_range, ports_sel, metric, active_tab):
    if active_tab != "overview":
        return dash.no_update

    df = _filter(years_range, ports_sel)
    if df.empty:
        return html.P("Keine Daten für die gewählten Filter.", className="text-muted p-4")

    label = next((o["label"] for o in METRIC_OPTIONS if o["value"] == metric), metric)

    # 1. Jährliche Entwicklung
    yearly = df.groupby("Referenzzeitraum_Jahr")[metric].sum().reset_index()
    yearly.columns = ["Jahr", metric]
    fig_year = px.bar(yearly, x="Jahr", y=metric, color_discrete_sequence=[ACCENT_COLOR],
                      labels={metric: label, "Jahr": "Jahr"})
    fig_year.add_scatter(x=yearly["Jahr"], y=yearly[metric], mode="lines+markers",
                         line=dict(color="#e74c3c", width=2), name="Trend")
    _chart_layout(fig_year, f"Jährliche {label}", height=340)

    # 2. Monatliche Saisonalität
    monthly_avg = df.groupby("Referenzzeitraum_Monat")[metric].mean().reset_index()
    monthly_avg.columns = ["Monat", metric]
    month_names = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                   "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    monthly_avg["Monat_Name"] = monthly_avg["Monat"].apply(
        lambda m: month_names[int(m) - 1] if 1 <= int(m) <= 12 else str(m)
    )
    fig_season = px.bar(monthly_avg, x="Monat_Name", y=metric,
                        color_discrete_sequence=["#3498db"],
                        labels={metric: f"Ø {label}", "Monat_Name": "Monat"})
    _chart_layout(fig_season, "Saisonale Verteilung (Ø)", height=340)

    # 3. Verkehrsrichtung
    if "Verkehrsbeziehung_Label" in df.columns:
        vk = df.groupby("Verkehrsbeziehung_Label")[metric].sum().reset_index()
        fig_dir = px.pie(vk, names="Verkehrsbeziehung_Label", values=metric,
                         color_discrete_sequence=px.colors.qualitative.Bold)
        _chart_layout(fig_dir, "Verkehrsbeziehung", height=320)
    else:
        fig_dir = go.Figure()

    # 4. Top-NST2007
    if "NST2007" in df.columns:
        nst = df.groupby("NST2007")[metric].sum().nlargest(10).reset_index()
        nst.columns = ["NST2007", metric]
        fig_nst = px.bar(nst, x=metric, y="NST2007", orientation="h",
                         color_discrete_sequence=["#9b59b6"],
                         labels={metric: label, "NST2007": "Güterklasse"})
        _chart_layout(fig_nst, "Top-10 Güterklassen (NST2007)", height=320)
    else:
        fig_nst = go.Figure()

    return html.Div([
        dbc.Row([
            dbc.Col(_section_card("Jährliche Entwicklung", dcc.Graph(figure=fig_year, config={"displayModeBar": False}), "fa-chart-bar"), md=8),
            dbc.Col(_section_card("Saisonalität", dcc.Graph(figure=fig_season, config={"displayModeBar": False}), "fa-calendar"), md=4),
        ]),
        dbc.Row([
            dbc.Col(_section_card("Verkehrsbeziehung", dcc.Graph(figure=fig_dir, config={"displayModeBar": False}), "fa-exchange-alt"), md=4),
            dbc.Col(_section_card("Güterklassen NST2007", dcc.Graph(figure=fig_nst, config={"displayModeBar": False}), "fa-boxes"), md=8),
        ]),
    ])


# ── Zeitreihe ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-timeseries", "children"),
    Input("year-slider", "value"),
    Input("port-filter", "value"),
    Input("metric-selector", "value"),
    Input("main-tabs", "active_tab"),
)
def update_timeseries(years_range, ports_sel, metric, active_tab):
    if active_tab != "timeseries":
        return dash.no_update

    df = _filter(years_range, ports_sel)
    label = next((o["label"] for o in METRIC_OPTIONS if o["value"] == metric), metric)

    if df.empty or "Datum" not in df.columns:
        return html.P("Keine Daten verfügbar.", className="text-muted p-4")

    # Monatliche Zeitreihe
    ts_month = df.groupby("Datum")[metric].sum().sort_index()

    # Gleitende Durchschnitte
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_month.index, y=ts_month.values,
        mode="lines", name="Monatlich",
        line=dict(color=ACCENT_COLOR, width=1.5), opacity=0.6,
    ))
    for window, color, name in [(3, "#e67e22", "3M-Ø"), (12, "#e74c3c", "12M-Ø")]:
        roll = ts_month.rolling(window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll.values,
            mode="lines", name=name,
            line=dict(color=color, width=2.5),
        ))
    _chart_layout(fig, f"Monatliche Zeitreihe: {label}", height=400)

    # Jahr-über-Jahr Heatmap
    if "Referenzzeitraum_Jahr" in df.columns and "Referenzzeitraum_Monat" in df.columns:
        hm_data = df.groupby(["Referenzzeitraum_Jahr", "Referenzzeitraum_Monat"])[metric].sum().unstack(fill_value=0)
        month_names = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                       "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
        hm_data.columns = [month_names[int(c) - 1] if 1 <= int(c) <= 12 else str(c)
                           for c in hm_data.columns]
        fig_hm = px.imshow(
            hm_data,
            color_continuous_scale="Teal",
            aspect="auto",
            labels=dict(x="Monat", y="Jahr", color=label),
        )
        _chart_layout(fig_hm, "Jahr-über-Jahr Heatmap", height=350)
    else:
        fig_hm = go.Figure()

    # Jährliche Wachstumsrate
    yearly = df.groupby("Referenzzeitraum_Jahr")[metric].sum()
    growth = yearly.pct_change() * 100
    fig_growth = go.Figure()
    fig_growth.add_bar(
        x=growth.index, y=growth.values,
        marker_color=[ACCENT_COLOR if v >= 0 else "#e74c3c" for v in growth.values],
        name="YoY Wachstum %",
    )
    fig_growth.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
    _chart_layout(fig_growth, "Jahr-über-Jahr Wachstum (%)", height=280)

    return html.Div([
        _section_card("Zeitreihe mit gleitenden Durchschnitten",
                      dcc.Graph(figure=fig, config={"displayModeBar": "hover"}),
                      "fa-chart-line"),
        dbc.Row([
            dbc.Col(_section_card("Saisonale Heatmap",
                                  dcc.Graph(figure=fig_hm, config={"displayModeBar": False}),
                                  "fa-th"), md=7),
            dbc.Col(_section_card("Wachstumsrate YoY",
                                  dcc.Graph(figure=fig_growth, config={"displayModeBar": False}),
                                  "fa-percent"), md=5),
        ]),
    ])


# ── ML-Prognose ───────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-forecast", "children"),
    Input("year-slider", "value"),
    Input("port-filter", "value"),
    Input("metric-selector", "value"),
    Input("main-tabs", "active_tab"),
)
def update_forecast_tab(years_range, ports_sel, metric, active_tab):
    if active_tab != "forecast":
        return dash.no_update

    label = next((o["label"] for o in METRIC_OPTIONS if o["value"] == metric), metric)

    return html.Div([
        _section_card("Modell- & Prognosekonfiguration", dbc.Row([
            dbc.Col([
                dbc.Label("ML-Modell", className="small text-muted"),
                dcc.Dropdown(
                    id="forecast-model",
                    options=MODEL_OPTIONS,
                    value="random_forest",
                    clearable=False,
                    style={"background": "#252538"},
                ),
            ], md=4),
            dbc.Col([
                dbc.Label("Prognosehorizont (Monate)", className="small text-muted"),
                dcc.Slider(
                    id="forecast-horizon",
                    min=3, max=36, step=3, value=12,
                    marks={i: str(i) for i in range(3, 37, 6)},
                    tooltip={"placement": "bottom"},
                ),
            ], md=5),
            dbc.Col([
                dbc.Button(
                    [html.I(className="fas fa-play me-2"), "Prognose erstellen"],
                    id="run-forecast-btn",
                    color="success",
                    className="w-100 mt-3",
                    style={"background": ACCENT_COLOR, "border": "none"},
                ),
            ], md=3),
        ]), icon="fa-brain"),

        dbc.Spinner([
            html.Div(id="forecast-output"),
        ], color="success", type="border"),
    ])


@app.callback(
    Output("forecast-output", "children"),
    Input("run-forecast-btn", "n_clicks"),
    State("year-slider", "value"),
    State("port-filter", "value"),
    State("metric-selector", "value"),
    State("forecast-model", "value"),
    State("forecast-horizon", "value"),
    prevent_initial_call=True,
)
def run_forecast(n_clicks, years_range, ports_sel, metric, model_name, horizon):
    df = _filter(years_range, ports_sel)
    label = next((o["label"] for o in METRIC_OPTIONS if o["value"] == metric), metric)

    if df.empty or "Datum" not in df.columns or metric not in df.columns:
        return dbc.Alert("Keine ausreichenden Daten für die Prognose.", color="warning")

    ts = df.groupby("Datum")[metric].sum().sort_index()
    result = build_forecast(ts, model_name=model_name, horizon=horizon)

    if "error" in result:
        return dbc.Alert(f"Fehler: {result['error']}", color="danger")

    # ── Prognosechart ──
    train_end = result["train_end_idx"]
    ts_vals = result["ts_values"]
    ts_idx = result["ts_index"]

    fig = go.Figure()

    # Trainingsdaten
    fig.add_trace(go.Scatter(
        x=ts_idx[:train_end], y=ts_vals[:train_end],
        mode="lines", name="Trainingsdaten",
        line=dict(color="#3498db", width=2),
    ))
    # Testdaten
    fig.add_trace(go.Scatter(
        x=ts_idx[train_end:], y=result["test_actual"],
        mode="lines+markers", name="Testdaten (tatsächlich)",
        line=dict(color=ACCENT_COLOR, width=2.5),
    ))
    # Testprognose
    fig.add_trace(go.Scatter(
        x=ts_idx[train_end:], y=result["test_pred"],
        mode="lines", name="Testprognose",
        line=dict(color="#e67e22", width=2, dash="dot"),
    ))

    # Zukunftsprognose
    if result["future_dates"] is not None:
        x_future = result["future_dates"]
    else:
        last_date = ts_idx[-1]
        x_future = pd.date_range(start=last_date, periods=horizon + 1, freq="MS")[1:]

    fig.add_trace(go.Scatter(
        x=list(x_future) + list(x_future[::-1]),
        y=list(result["upper_ci"]) + list(result["lower_ci"][::-1]),
        fill="toself", fillcolor="rgba(0,212,170,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="95% KI",
    ))
    fig.add_trace(go.Scatter(
        x=x_future, y=result["forecast"],
        mode="lines+markers", name="Prognose",
        line=dict(color="#e74c3c", width=3),
        marker=dict(size=6),
    ))

    # Trennung Ist/Prognose
    fig.add_vline(x=ts_idx[-1], line_dash="dash", line_color="#888",
                  annotation_text="Jetzt", annotation_position="top right")
    _chart_layout(fig, f"ML-Prognose: {label} ({model_name.replace('_', ' ').title()})", height=420)

    # ── Metriken ──
    m = result["metrics"]
    metric_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("MAE", className="text-muted small mb-1"),
            html.H5(f"{m['MAE']:,.1f}", className="fw-bold", style={"color": "#3498db"}),
        ]), style=DARK_CARD), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("RMSE", className="text-muted small mb-1"),
            html.H5(f"{m['RMSE']:,.1f}", className="fw-bold", style={"color": "#e67e22"}),
        ]), style=DARK_CARD), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("R² (Test)", className="text-muted small mb-1"),
            html.H5(
                f"{m['R²']:.3f}",
                className="fw-bold",
                style={"color": ACCENT_COLOR if m["R²"] > 0.5 else "#e74c3c"},
            ),
        ]), style=DARK_CARD), md=4),
    ], className="mb-3")

    # ── Feature Importance ──
    fi_content = html.Div()
    if result.get("feature_importance"):
        fi = dict(list(result["feature_importance"].items())[:12])
        fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
        fi_df = fi_df.sort_values("Importance", ascending=True)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Teal")
        _chart_layout(fig_fi, "Feature Importance", height=350)
        fi_content = _section_card(
            "Feature Importance",
            dcc.Graph(figure=fig_fi, config={"displayModeBar": False}),
            "fa-sort-amount-down",
        )

    # ── Residuen ──
    residuals = result["test_actual"] - result["test_pred"]
    fig_res = make_subplots(rows=1, cols=2, subplot_titles=["Residuen", "Residuen-Histogramm"])
    fig_res.add_trace(
        go.Scatter(y=residuals, mode="markers",
                   marker=dict(color=ACCENT_COLOR, opacity=0.6), name="Residuen"),
        row=1, col=1,
    )
    fig_res.add_hline(y=0, line_dash="dash", line_color="#888", row=1, col=1)
    fig_res.add_trace(
        go.Histogram(x=residuals, nbinsx=30,
                     marker_color="#9b59b6", opacity=0.8, name="Hist."),
        row=1, col=2,
    )
    _chart_layout(fig_res, height=300)

    return html.Div([
        _section_card(
            f"Prognose: {label}",
            dcc.Graph(figure=fig, config={"displayModeBar": "hover"}),
            "fa-chart-line",
        ),
        dbc.Row([
            dbc.Col([
                html.H6("Modellgüte (Test-Split)", className="text-muted mb-3"),
                metric_cards,
                fi_content,
            ], md=6),
            dbc.Col(_section_card(
                "Residuenanalyse",
                dcc.Graph(figure=fig_res, config={"displayModeBar": False}),
                "fa-wave-square",
            ), md=6),
        ]),
    ])


# ── Regionen ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-regions", "children"),
    Input("year-slider", "value"),
    Input("port-filter", "value"),
    Input("metric-selector", "value"),
    Input("main-tabs", "active_tab"),
)
def update_regions(years_range, ports_sel, metric, active_tab):
    if active_tab != "regions":
        return dash.no_update

    df = _filter(years_range, ports_sel)
    label = next((o["label"] for o in METRIC_OPTIONS if o["value"] == metric), metric)

    if df.empty:
        return html.P("Keine Daten.", className="text-muted p-4")

    # Makroregion-Balkendiagramm
    col = "Einladeregion_Makroregion"
    if col in df.columns:
        macro = df.groupby(col)[metric].sum().sort_values(ascending=True).reset_index()
        macro.columns = ["Region", metric]
        fig_macro = px.bar(macro, x=metric, y="Region", orientation="h",
                           color=metric, color_continuous_scale="Teal",
                           labels={metric: label})
        _chart_layout(fig_macro, f"Einladeregionen nach {label}", height=420)
    else:
        fig_macro = go.Figure()

    # Top-Länder (Einlade)
    if "Einladeregion_ISO" in df.columns:
        top_countries = df.groupby("Einladeregion_ISO")[metric].sum().nlargest(20).reset_index()
        top_countries.columns = ["Land", metric]
        fig_countries = px.bar(
            top_countries.sort_values(metric, ascending=True),
            x=metric, y="Land", orientation="h",
            color_discrete_sequence=[ACCENT_COLOR],
            labels={metric: label},
        )
        _chart_layout(fig_countries, f"Top-20 Länder (Einladung) nach {label}", height=500)
    else:
        fig_countries = go.Figure()

    # Sankey: Einlade-Makroregion → Auslade-Makroregion
    sankey_content = html.Div()
    if "Einladeregion_Makroregion" in df.columns and "Ausladeregion_Makroregion" in df.columns:
        flow = (
            df.groupby(["Einladeregion_Makroregion", "Ausladeregion_Makroregion"])[metric]
            .sum()
            .reset_index()
        )
        flow = flow[flow[metric] > 0].nlargest(30, metric)

        all_nodes = pd.unique(flow[["Einladeregion_Makroregion", "Ausladeregion_Makroregion"]].values.ravel())
        node_idx = {n: i for i, n in enumerate(all_nodes)}
        node_colors = px.colors.qualitative.Pastel * 3

        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                pad=15, thickness=20,
                label=list(all_nodes),
                color=[node_colors[i % len(node_colors)] for i in range(len(all_nodes))],
            ),
            link=dict(
                source=[node_idx[r] for r in flow["Einladeregion_Makroregion"]],
                target=[node_idx[r] for r in flow["Ausladeregion_Makroregion"]],
                value=flow[metric].values,
                color="rgba(0,212,170,0.2)",
            ),
        ))
        _chart_layout(fig_sankey, f"Handelsstrom: Einladung → Ausladung ({label})", height=460)
        sankey_content = _section_card(
            "Handelsstrom-Sankey",
            dcc.Graph(figure=fig_sankey, config={"displayModeBar": False}),
            "fa-project-diagram",
        )

    return html.Div([
        dbc.Row([
            dbc.Col(_section_card("Makroregionen",
                                  dcc.Graph(figure=fig_macro, config={"displayModeBar": False}),
                                  "fa-globe-europe"), md=5),
            dbc.Col(_section_card("Top-Länder",
                                  dcc.Graph(figure=fig_countries, config={"displayModeBar": False}),
                                  "fa-flag"), md=7),
        ]),
        sankey_content,
    ])


# ── Häfen & Schiffe ───────────────────────────────────────────────────────────
@app.callback(
    Output("tab-ports", "children"),
    Input("year-slider", "value"),
    Input("port-filter", "value"),
    Input("metric-selector", "value"),
    Input("main-tabs", "active_tab"),
)
def update_ports(years_range, ports_sel, metric, active_tab):
    if active_tab != "ports":
        return dash.no_update

    df = _filter(years_range, ports_sel)
    label = next((o["label"] for o in METRIC_OPTIONS if o["value"] == metric), metric)

    if df.empty:
        return html.P("Keine Daten.", className="text-muted p-4")

    # Top-Häfen
    if "EVAS_Label" in df.columns:
        top_ports = df.groupby("EVAS_Label")[metric].sum().nlargest(15).sort_values(ascending=True).reset_index()
        top_ports.columns = ["Hafen", metric]
        fig_ports = px.bar(top_ports, x=metric, y="Hafen", orientation="h",
                           color=metric, color_continuous_scale="Blues",
                           labels={metric: label})
        _chart_layout(fig_ports, f"Top-Häfen nach {label}", height=420)
    else:
        fig_ports = go.Figure()

    # Schiffsart
    if "Schiffsart" in df.columns:
        ship_data = df.groupby("Schiffsart")[metric].sum().reset_index()
        ship_data.columns = ["Schiffsart", metric]
        fig_ship = px.pie(ship_data, names="Schiffsart", values=metric,
                          color_discrete_sequence=px.colors.qualitative.Bold,
                          hole=0.4)
        fig_ship.update_traces(textinfo="label+percent")
        _chart_layout(fig_ship, f"Verteilung nach Schiffsart", height=380)
    else:
        fig_ship = go.Figure()

    # Flaggenstaat
    if "Flagge" in df.columns:
        flags = df.groupby("Flagge")[metric].sum().nlargest(15).sort_values(ascending=True).reset_index()
        flags.columns = ["Flagge", metric]
        fig_flags = px.bar(flags, x=metric, y="Flagge", orientation="h",
                           color_discrete_sequence=["#9b59b6"],
                           labels={metric: label})
        _chart_layout(fig_flags, "Top-15 Flaggenstaaten", height=420)
    else:
        fig_flags = go.Figure()

    # Entwicklung der Top-5 Häfen über Zeit
    timeline_content = html.Div()
    if "EVAS_Label" in df.columns and "Datum" in df.columns:
        top5 = df.groupby("EVAS_Label")[metric].sum().nlargest(5).index.tolist()
        df_top5 = df[df["EVAS_Label"].isin(top5)]
        ts_ports = df_top5.groupby(["Datum", "EVAS_Label"])[metric].sum().reset_index()
        fig_port_ts = px.line(ts_ports, x="Datum", y=metric, color="EVAS_Label",
                              labels={metric: label, "EVAS_Label": "Hafen", "Datum": "Datum"},
                              color_discrete_sequence=px.colors.qualitative.Bold)
        _chart_layout(fig_port_ts, "Zeitverlauf: Top-5 Häfen", height=380)
        timeline_content = _section_card(
            "Hafenentwicklung im Zeitverlauf",
            dcc.Graph(figure=fig_port_ts, config={"displayModeBar": "hover"}),
            "fa-chart-area",
        )

    return html.Div([
        dbc.Row([
            dbc.Col(_section_card("Häfen",
                                  dcc.Graph(figure=fig_ports, config={"displayModeBar": False}),
                                  "fa-anchor"), md=7),
            dbc.Col(_section_card("Schiffsarten",
                                  dcc.Graph(figure=fig_ship, config={"displayModeBar": False}),
                                  "fa-ship"), md=5),
        ]),
        dbc.Row([
            dbc.Col(_section_card("Flaggenstaaten",
                                  dcc.Graph(figure=fig_flags, config={"displayModeBar": False}),
                                  "fa-flag"), md=5),
            dbc.Col(timeline_content, md=7),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Start
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket

    def _get_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"

    local_ip = _get_ip()
    print("\n" + "=" * 60)
    print("  Seeverkehr Analytics Dashboard")
    print("=" * 60)
    print(f"  Modus    : {'DEMO (keine CSV gefunden)' if IS_DEMO else 'ECHTDATEN'}")
    print(f"  Datensatz: {len(DF):,} Zeilen × {len(DF.columns)} Spalten")
    print(f"  Lokal    : http://127.0.0.1:8050")
    print(f"  Netzwerk : http://{local_ip}:8050")
    print("=" * 60 + "\n")

    app.run(debug=False, host="0.0.0.0", port=8050)
