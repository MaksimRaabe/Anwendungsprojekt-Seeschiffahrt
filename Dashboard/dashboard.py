#!/usr/bin/env python3
"""
Seeverkehr Analytics Dashboard  –  Large-File Edition
=======================================================
Optimiert für ~10 Mio. Zeilen / 3,5 GB CSV-Dateien.

Strategie:
  1. CSV wird in 500k-Chunks gelesen – nie vollständig im RAM.
  2. Während des Einlesens werden alle benötigten Aggregationen
     direkt akkumuliert (keine Rohdaten gespeichert).
  3. Ergebnis wird als Parquet-Cache gespeichert; Folgestarts
     laden nur den kleinen Cache (~MB statt GB).
  4. Alle Dashboard-Callbacks arbeiten ausschließlich auf den
     kleinen Aggregations-Tabellen.

Start:  python dashboard.py   →   http://127.0.0.1:8050
Cache:  Dashboard/.cache/agg_*.parquet  (automatisch erstellt)
"""

from __future__ import annotations

import json
import time
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
from dash import Input, Output, State, dcc, html

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Pfade & Konstanten
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
CLEANED_DIR = BASE_DIR.parent / "Dataset bereinigung" / "Datasets_cleaned"
CACHE_DIR = BASE_DIR / ".cache"
CACHE_META = CACHE_DIR / "meta.json"
CACHE_DIR.mkdir(exist_ok=True)

CSV_READ_CHUNK = 500_000  # Zeilen pro Chunk beim CSV-Lesen

# Spalten die wir aus der CSV benötigen (Rest wird ignoriert → schneller)
NEEDED_COLS = [
    "Referenzzeitraum_Jahr", "Referenzzeitraum_Monat",
    "Einladeregion_NUTS3", "Einladeregion_HafenID_Label",
    "Ausladeregion_NUTS3", "Ausladeregion_HafenID_Label",
    "Einladeregion_ISO", "Ausladeregion_ISO",
    "Einladeregion_Makroregion", "Ausladeregion_Makroregion",
    "Verkehrsbeziehung_Label", "Schiffsart_Label",
    "Flagge", "NST2007_Label",
    "Tonnen", "TEU", "Anzahl_Ladungstraeger",
]

METRICS = ["Tonnen", "TEU", "Anzahl_Ladungstraeger"]

# Aggregations-Dimensionen: Name → Gruppierungsspalten
# "Hafen_DE" wird beim Einlesen aus NUTS3 abgeleitet (deutscher Hafen je Datensatz)
AGG_DIMS: dict[str, list[str]] = {
    "ts":          ["Referenzzeitraum_Jahr", "Referenzzeitraum_Monat", "Hafen_DE"],
    "schiffsart":  ["Referenzzeitraum_Jahr", "Hafen_DE", "Schiffsart_Label"],
    "flagge":      ["Referenzzeitraum_Jahr", "Hafen_DE", "Flagge"],
    "nst":         ["Referenzzeitraum_Jahr", "Hafen_DE", "NST2007_Label"],
    "vk":          ["Referenzzeitraum_Jahr", "Hafen_DE", "Verkehrsbeziehung_Label"],
    "einlade":     ["Referenzzeitraum_Jahr", "Hafen_DE", "Einladeregion_Makroregion"],
    "auslade":     ["Referenzzeitraum_Jahr", "Hafen_DE", "Ausladeregion_Makroregion"],
    "einlade_iso": ["Referenzzeitraum_Jahr", "Hafen_DE", "Einladeregion_ISO"],
    "auslade_iso": ["Referenzzeitraum_Jahr", "Hafen_DE", "Ausladeregion_ISO"],
    "sankey":      ["Referenzzeitraum_Jahr", "Hafen_DE",
                    "Einladeregion_Makroregion", "Ausladeregion_Makroregion"],
}

METRIC_OPTIONS = [
    {"label": "Tonnage (t)",      "value": "Tonnen"},
    {"label": "TEU (Container)",  "value": "TEU"},
    {"label": "Ladeeinheiten",    "value": "Anzahl_Ladungstraeger"},
]
MODEL_OPTIONS = [
    {"label": "Random Forest",               "value": "random_forest"},
    {"label": "Gradient Boosting",           "value": "gradient_boosting"},
    {"label": "Lineare Regression",          "value": "linear"},
    {"label": "Polynomiale Regression (2°)", "value": "polynomial"},
]

ACCENT = "#00d4aa"
DARK_CARD = {"background": "#1e1e30", "border": "1px solid #2d2d45"}
CHART_TPL = "plotly_dark"

# ─────────────────────────────────────────────────────────────────────────────
# Datenpipeline
# ─────────────────────────────────────────────────────────────────────────────

def _derive_hafen_de(chunk: pd.DataFrame) -> pd.Series:
    """Leitet den deutschen Hafen je Datensatz aus den NUTS3-Codes ab."""
    einlade_de = chunk.get("Einladeregion_NUTS3", pd.Series(dtype=str)).str.startswith("DE", na=False)
    auslade_de = chunk.get("Ausladeregion_NUTS3", pd.Series(dtype=str)).str.startswith("DE", na=False)
    default = pd.Series("Unbekannt", index=chunk.index)
    return np.where(
        einlade_de,
        chunk.get("Einladeregion_HafenID_Label", default),
        np.where(
            auslade_de,
            chunk.get("Ausladeregion_HafenID_Label", default),
            "International",
        ),
    )


def _process_chunk(chunk: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Normalisiert einen Chunk und berechnet alle Aggregationen."""
    # Numerik sicherstellen (CSV verwendet deutsches Dezimalkomma: "342,0" → 342.0)
    for col in METRICS:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(
                chunk[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    chunk["Hafen_DE"] = _derive_hafen_de(chunk)

    parts: dict[str, pd.DataFrame] = {}
    for name, dims in AGG_DIMS.items():
        present_dims = [d for d in dims if d in chunk.columns]
        present_metrics = [m for m in METRICS if m in chunk.columns]
        if not present_dims or not present_metrics:
            continue
        agg = (
            chunk.groupby(present_dims, observed=True, dropna=False)[present_metrics]
            .sum(min_count=1)
        )
        parts[name] = agg
    return parts


def _merge_accumulators(accumulators: dict[str, list]) -> dict[str, pd.DataFrame]:
    """Fasst alle akkumulierten Chunk-Aggregationen zusammen."""
    result: dict[str, pd.DataFrame] = {}
    for name, parts in accumulators.items():
        if not parts:
            continue
        combined = pd.concat(parts)
        # Gleiche Gruppen aus verschiedenen Chunks zusammenführen
        n_levels = combined.index.nlevels
        merged = combined.groupby(level=list(range(n_levels)), observed=True).sum()
        result[name] = merged.reset_index()
    return result


def _find_csv() -> Path | None:
    """Gibt die neueste bereinigte CSV zurück (oder None)."""
    if not CLEANED_DIR.exists():
        return None
    files = sorted(CLEANED_DIR.glob("seeverkehr_bereinigt_*.csv"))
    return files[-1] if files else None


def _cache_valid(csv_path: Path) -> bool:
    """Prüft ob alle Parquet-Caches existieren und neuer als die CSV sind."""
    if not CACHE_META.exists():
        return False
    try:
        meta = json.loads(CACHE_META.read_text())
    except Exception:
        return False
    if meta.get("csv_mtime") != csv_path.stat().st_mtime:
        return False
    return all((CACHE_DIR / f"agg_{name}.parquet").exists() for name in AGG_DIMS)


def _save_cache(aggs: dict[str, pd.DataFrame], csv_path: Path) -> None:
    for name, df in aggs.items():
        df.to_parquet(CACHE_DIR / f"agg_{name}.parquet", index=False)
    CACHE_META.write_text(json.dumps({"csv_mtime": csv_path.stat().st_mtime,
                                       "rows_total": sum(len(v) for v in aggs.values())}))


def _load_cache() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(CACHE_DIR / f"agg_{name}.parquet") for name in AGG_DIMS}


def build_aggregations(csv_path: Path) -> dict[str, pd.DataFrame]:
    """Liest die CSV chunkweise und baut alle Aggregationstabellen.

    Peak-RAM: ~300 MB (ein Chunk + laufende Aggs) statt mehrerer GB.
    """
    print(f"  Datei : {csv_path.name}")
    print(f"  Größe : {csv_path.stat().st_size / 1024**2:.0f} MB")
    print("  Lese in 500k-Chunks …\n")

    # Nur vorhandene Spalten anfordern
    probe = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", nrows=0)
    use_cols = [c for c in NEEDED_COLS if c in probe.columns]

    accumulators: dict[str, list] = {name: [] for name in AGG_DIMS}
    total_rows = 0
    t0 = time.time()

    reader = pd.read_csv(
        csv_path,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        usecols=use_cols,
        chunksize=CSV_READ_CHUNK,
        low_memory=False,
        dtype=str,  # erst als String, Numerik im _process_chunk
    )

    for chunk_num, chunk in enumerate(reader, start=1):
        parts = _process_chunk(chunk)
        for name, agg in parts.items():
            accumulators[name].append(agg)
        total_rows += len(chunk)
        elapsed = time.time() - t0
        speed = total_rows / elapsed if elapsed > 0 else 0
        print(
            f"\r  Chunk {chunk_num:>3}  │  {total_rows:>10,} Zeilen  │  "
            f"{speed/1000:.0f}k Zeilen/s  │  {elapsed:.0f}s",
            end="",
            flush=True,
        )

    elapsed = time.time() - t0
    print(f"\n\n  Fertig: {total_rows:,} Zeilen in {elapsed:.1f}s gelesen.")
    print("  Aggregiere Chunks …")

    aggs = _merge_accumulators(accumulators)

    total_agg_rows = sum(len(v) for v in aggs.values())
    print(f"  Aggregations-Tabellen: {len(aggs)} Tabellen, {total_agg_rows:,} Zeilen gesamt.")
    return aggs


def _generate_demo_aggs() -> dict[str, pd.DataFrame]:
    """Erstellt realistische Demo-Aggregationen wenn keine CSV vorhanden."""
    rng = np.random.default_rng(42)
    years = np.arange(2011, 2026)
    months = np.arange(1, 13)
    ports = ["Hamburg", "Bremerhaven", "Rostock", "Lübeck", "Kiel", "Wismar"]
    regions = ["Europa (EU/EEA)", "Ostasien", "Nordamerika", "Suedostasien",
               "Naher Osten", "Suedasien", "Suedamerika", "Nordafrika"]
    vessels = ["Containerschiff", "Tanker", "Massengutschiff", "Ro-Ro-Schiff", "Stückgutschiff"]
    flags = ["DE", "PA", "LR", "MH", "BS", "CY", "NL", "MT", "SG", "HK"]
    nst = [f"NST-{i:02d}" for i in range(1, 11)]
    vk = ["Versand in das Ausland", "Empfang aus dem Ausland",
          "Durchgangsverkehr", "Küstenverkehr"]
    iso_in = ["CN", "US", "NL", "GB", "JP", "KR", "SG", "BR", "IN", "AE"]
    iso_out = ["CN", "US", "NL", "FR", "IT", "PL", "SE", "FI", "DK", "ES"]

    def _mk(dims: list[str]) -> pd.DataFrame:
        rows = []
        idx_map = {
            "Referenzzeitraum_Jahr": years,
            "Referenzzeitraum_Monat": months,
            "Hafen_DE": ports,
            "Schiffsart_Label": vessels,
            "Flagge": flags,
            "NST2007_Label": nst,
            "Verkehrsbeziehung_Label": vk,
            "Einladeregion_Makroregion": regions,
            "Ausladeregion_Makroregion": regions,
            "Einladeregion_ISO": iso_in,
            "Ausladeregion_ISO": iso_out,
        }
        # Einfaches kartesisches Produkt der ersten 2-3 Dimensionen mit Stichproben
        for _ in range(2000):
            row = {d: rng.choice(idx_map.get(d, ["?"]), 1)[0] for d in dims}
            yr = row.get("Referenzzeitraum_Jahr", 2015)
            port = row.get("Hafen_DE", "Hamburg")
            trend = 1 + (int(yr) - 2011) * 0.025
            port_f = 3.0 if port == "Hamburg" else (1.8 if port == "Bremerhaven" else 1.0)
            row["Tonnen"] = float(rng.lognormal(np.log(8e6 * trend * port_f), 0.5))
            row["TEU"] = float(rng.lognormal(np.log(5e4 * trend * port_f), 0.6)) if rng.random() < 0.5 else np.nan
            row["Anzahl_Ladungstraeger"] = float(rng.integers(100, 5000)) if rng.random() < 0.4 else np.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.groupby(dims, observed=True)[METRICS].sum().reset_index()

    return {name: _mk(dims) for name, dims in AGG_DIMS.items()}


def load_all() -> tuple[dict[str, pd.DataFrame], bool, str]:
    """Haupteinstiegspunkt. Gibt (aggs, is_demo, info_text) zurück."""
    csv_path = _find_csv()

    if csv_path is None:
        print("  Keine CSV gefunden → Demo-Aggregationen werden verwendet.")
        return _generate_demo_aggs(), True, "Demo-Modus"

    if _cache_valid(csv_path):
        print(f"  Cache gültig → lade Parquet-Cache für {csv_path.name}")
        t0 = time.time()
        aggs = _load_cache()
        print(f"  Cache geladen in {time.time()-t0:.1f}s.")
        rows = sum(len(v) for v in aggs.values())
        return aggs, False, f"{csv_path.name} (Cache, {rows:,} Agg-Zeilen)"

    print(f"  Kein gültiger Cache → verarbeite {csv_path.name} …")
    aggs = build_aggregations(csv_path)
    print("  Speichere Parquet-Cache …")
    _save_cache(aggs, csv_path)
    print("  Cache gespeichert.\n")
    rows = sum(len(v) for v in aggs.values())
    return aggs, False, f"{csv_path.name} (neu gebaut, {rows:,} Agg-Zeilen)"


# ─────────────────────────────────────────────────────────────────────────────
# Aggregations-Filterhelfer
# ─────────────────────────────────────────────────────────────────────────────

# Globaler Aggregations-Store (befüllt beim Start)
AGGS: dict[str, pd.DataFrame] = {}
IS_DEMO: bool = True
INFO_TEXT: str = ""


def _agg(name: str, years_range: list | None = None, ports_sel: list | None = None) -> pd.DataFrame:
    """Gibt eine Aggregationstabelle gefiltert nach Jahr und Hafen zurück."""
    df = AGGS.get(name, pd.DataFrame())
    if df.empty:
        return df
    if years_range and "Referenzzeitraum_Jahr" in df.columns:
        df = df[pd.to_numeric(df["Referenzzeitraum_Jahr"], errors="coerce").between(years_range[0], years_range[1])]
    if ports_sel and "Hafen_DE" in df.columns:
        df = df[df["Hafen_DE"].isin(ports_sel)]
    return df


def _group_metrics(df: pd.DataFrame, dims: list[str], metric: str) -> pd.DataFrame:
    """Aggregiert eine bereits gefilterte Agg-Tabelle auf die angegebenen Dimensionen."""
    present = [d for d in dims if d in df.columns]
    if not present or metric not in df.columns:
        return pd.DataFrame()
    return df.groupby(present, observed=True)[metric].sum().reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# ML Forecasting Engine
# ─────────────────────────────────────────────────────────────────────────────

def _ts_from_agg(agg_name: str, metric: str, years_range: list, ports_sel: list) -> pd.Series:
    """Baut eine monatliche Zeitreihe aus der ts-Aggregationstabelle."""
    df = _agg(agg_name, years_range, ports_sel)
    if df.empty:
        return pd.Series(dtype=float)
    grouped = df.groupby(["Referenzzeitraum_Jahr", "Referenzzeitraum_Monat"], observed=True)[metric].sum()
    grouped.index = pd.to_datetime(
        [f"{int(y)}-{int(m):02d}-01" for y, m in grouped.index],
        errors="coerce",
    )
    return grouped.sort_index().dropna()


def build_forecast(series: pd.Series, model_name: str = "random_forest", horizon: int = 12) -> dict:
    ts = series.dropna().sort_index()
    if len(ts) < 18:
        return {"error": f"Zu wenige Datenpunkte: {len(ts)} (mind. 18 benötigt)"}

    values = ts.values.astype(float)
    n = len(values)
    months = ts.index.month.values

    t = np.arange(n)
    X = pd.DataFrame({
        "t": t, "t2": t ** 2,
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12),
        "q_sin": np.sin(2 * np.pi * ((months - 1) // 3 + 1) / 4),
        "q_cos": np.cos(2 * np.pi * ((months - 1) // 3 + 1) / 4),
    })
    for lag in [1, 2, 3, 6, 12]:
        lagged = np.full(n, np.nan)
        if n > lag:
            lagged[lag:] = values[:-lag]
        X[f"lag_{lag}"] = lagged
    for w in [3, 6, 12]:
        X[f"roll_{w}"] = pd.Series(values).rolling(w, min_periods=1).mean().values

    col_means = X.mean()
    X = X.fillna(col_means)

    test_size = min(12, max(6, n // 6))
    train_end = n - test_size
    X_tr, X_te = X.iloc[:train_end], X.iloc[train_end:]
    y_tr, y_te = values[:train_end], values[train_end:]

    _models = {
        "linear":    Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "polynomial": Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                                 ("sc", StandardScaler(with_mean=False)),
                                 ("m", Ridge(alpha=1.0))]),
        "random_forest":     RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3, n_jobs=-1, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42),
    }
    model = _models.get(model_name, _models["random_forest"])
    model.fit(X_tr, y_tr)

    y_pred_te = np.maximum(0, model.predict(X_te))
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred_te)))

    future_dates = pd.date_range(
        start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    buffer = list(values)
    forecast_vals = []
    for h, (new_date) in enumerate(future_dates):
        new_t = n + h
        m = new_date.month
        row = {"t": new_t, "t2": new_t ** 2,
               "month_sin": np.sin(2 * np.pi * m / 12), "month_cos": np.cos(2 * np.pi * m / 12),
               "q_sin": np.sin(2 * np.pi * ((m - 1) // 3 + 1) / 4),
               "q_cos": np.cos(2 * np.pi * ((m - 1) // 3 + 1) / 4)}
        for lag in [1, 2, 3, 6, 12]:
            idx = len(buffer) - lag
            row[f"lag_{lag}"] = buffer[idx] if idx >= 0 else col_means[f"lag_{lag}"]
        for w in [3, 6, 12]:
            row[f"roll_{w}"] = float(np.mean(buffer[-w:])) if len(buffer) >= w else float(np.mean(buffer))
        X_row = pd.DataFrame([row])[X_tr.columns].fillna(col_means)
        pred = float(max(0, model.predict(X_row)[0]))
        forecast_vals.append(pred)
        buffer.append(pred)

    forecast_arr = np.array(forecast_vals)
    uncertainty = rmse * np.sqrt(np.arange(1, horizon + 1))

    fi: dict = {}
    raw = model.named_steps["m"] if hasattr(model, "named_steps") else model
    if hasattr(raw, "feature_importances_"):
        fi = dict(sorted(zip(X_tr.columns, raw.feature_importances_), key=lambda x: -x[1]))

    return {
        "forecast": forecast_arr,
        "lower_ci": np.maximum(0, forecast_arr - 1.96 * uncertainty),
        "upper_ci": forecast_arr + 1.96 * uncertainty,
        "future_dates": future_dates,
        "metrics": {
            "MAE":  round(mean_absolute_error(y_te, y_pred_te), 1),
            "RMSE": round(rmse, 1),
            "R²":   round(r2_score(y_te, y_pred_te), 3),
        },
        "feature_importance": fi,
        "test_actual": y_te,
        "test_pred": y_pred_te,
        "train_end_idx": train_end,
        "ts_index": ts.index,
        "ts_values": values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Daten laden (beim Import)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  Seeverkehr Analytics – Datenlader")
print("=" * 65)
AGGS, IS_DEMO, INFO_TEXT = load_all()

# Häfen für Filter-Dropdown (sortiert nach Gesamt-Tonnage)
if "Hafen_DE" in AGGS.get("ts", pd.DataFrame()).columns:
    _port_ton = (
        AGGS["ts"].groupby("Hafen_DE", observed=True)["Tonnen"].sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    PORT_OPTIONS = [
        {"label": row["Hafen_DE"], "value": row["Hafen_DE"]}
        for _, row in _port_ton.iterrows()
        if str(row["Hafen_DE"]) not in ("nan", "International", "Unbekannt")
    ]
else:
    PORT_OPTIONS = []

YEAR_MIN = int(AGGS["ts"]["Referenzzeitraum_Jahr"].min()) if "ts" in AGGS and not AGGS["ts"].empty else 2011
YEAR_MAX = int(AGGS["ts"]["Referenzzeitraum_Jahr"].max()) if "ts" in AGGS and not AGGS["ts"].empty else 2025

print(f"\n  Bereit: {YEAR_MIN}–{YEAR_MAX}, {len(PORT_OPTIONS)} Häfen im Filter.")
print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Dash App
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="Seeverkehr Analytics",
)
server = app.server


# ── Layout-Helfer ────────────────────────────────────────────────────────────

def _chart(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        template=CHART_TPL, height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=44 if title else 20, b=20),
        title=dict(text=title, font=dict(size=13, color="#ccc")) if title else None,
        legend=dict(font=dict(size=11, color="#aaa")),
        font=dict(color="#ccc"),
    )
    fig.update_xaxes(gridcolor="#2a2a3e", zerolinecolor="#2a2a3e")
    fig.update_yaxes(gridcolor="#2a2a3e", zerolinecolor="#2a2a3e")
    return fig


def _card(title: str, body, icon: str = "fa-chart-bar") -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader([html.I(className=f"fas {icon} me-2", style={"color": ACCENT}),
                        html.Strong(title, className="text-light")],
                       style={"background": "#181828", "borderBottom": "1px solid #2d2d45"}),
        dbc.CardBody(body, style={"background": "#1e1e30"}),
    ], style=DARK_CARD, className="mb-4 shadow")


def _kpi(icon: str, label: str, value: str, color: str) -> dbc.Col:
    return dbc.Col(dbc.Card(dbc.CardBody(
        dbc.Row([
            dbc.Col(html.I(className=f"fas {icon} fa-2x", style={"color": color}),
                    width=3, className="d-flex align-items-center justify-content-center"),
            dbc.Col([html.P(label, className="text-muted small mb-1"),
                     html.H5(value, className="fw-bold mb-0", style={"color": color})],
                    width=9),
        ], className="g-0"),
    ), style={**DARK_CARD, "borderLeft": f"3px solid {color}"}), md=3, sm=6, className="mb-3")


def _fmt(v: float) -> str:
    if v >= 1e9: return f"{v/1e9:.2f} Mrd."
    if v >= 1e6: return f"{v/1e6:.1f} Mio."
    if v >= 1e3: return f"{v/1e3:.1f} Tsd."
    return f"{v:.0f}"


# ── Haupt-Layout ─────────────────────────────────────────────────────────────

year_marks = {y: {"label": str(y), "style": {"color": "#888"}}
              for y in range(YEAR_MIN, YEAR_MAX + 1, 2)}

app.layout = dbc.Container([

    # Header
    dbc.Row(dbc.Col(dbc.Navbar([
        html.Div([
            html.I(className="fas fa-ship me-3 fa-2x", style={"color": ACCENT}),
            html.Div([
                html.H4("Seeverkehr Analytics", className="mb-0 fw-bold"),
                html.Small(INFO_TEXT, className="text-muted"),
            ]),
        ], className="d-flex align-items-center"),
        dbc.Badge(
            "DEMO-DATEN" if IS_DEMO else "Echtdaten geladen",
            color="warning" if IS_DEMO else "success",
            className="ms-auto fs-6 px-3 py-2",
        ),
    ], dark=True, color="dark", className="px-4 rounded shadow")), className="mb-4"),

    # Filter
    dbc.Row(dbc.Col(_card("Filter", dbc.Row([
        dbc.Col([
            dbc.Label("Zeitraum", className="small text-muted"),
            dcc.RangeSlider(
                id="yr", min=YEAR_MIN, max=YEAR_MAX, value=[YEAR_MIN, YEAR_MAX],
                marks=year_marks,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], md=5),
        dbc.Col([
            dbc.Label("Hafen (Deutscher Hafen)", className="small text-muted"),
            dcc.Dropdown(id="ports", options=PORT_OPTIONS, value=[], multi=True,
                         placeholder="Alle Häfen", style={"background": "#252538"}),
        ], md=4),
        dbc.Col([
            dbc.Label("Metrik", className="small text-muted"),
            dcc.Dropdown(id="metric", options=METRIC_OPTIONS, value="Tonnen",
                         clearable=False, style={"background": "#252538"}),
        ], md=3),
    ]), icon="fa-sliders-h"))),

    # KPIs
    dbc.Row(id="kpi-row", className="mb-2"),

    # Tabs
    dbc.Tabs([
        dbc.Tab(html.Div(id="t-overview"),   label="Übersicht",          tab_id="overview",    className="pt-3"),
        dbc.Tab(html.Div(id="t-timeseries"), label="Zeitreihe",           tab_id="timeseries",  className="pt-3"),
        dbc.Tab(html.Div(id="t-forecast"),   label="ML-Prognose",         tab_id="forecast",    className="pt-3"),
        dbc.Tab(html.Div(id="t-regions"),    label="Regionen & Länder",   tab_id="regions",     className="pt-3"),
        dbc.Tab(html.Div(id="t-ports"),      label="Häfen & Schiffe",     tab_id="ports",       className="pt-3"),
    ], id="tabs", active_tab="overview"),

    html.Hr(style={"borderColor": "#2d2d45", "marginTop": "3rem"}),
    html.P([html.I(className="fas fa-database me-2", style={"color": ACCENT}),
            f"Plotly Dash · scikit-learn · {INFO_TEXT}"],
           className="text-muted small text-center pb-3"),

], fluid=True, style={"background": "#12121e", "minHeight": "100vh", "paddingTop": "1.5rem"})


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(Output("kpi-row", "children"),
              Input("yr", "value"), Input("ports", "value"), Input("metric", "value"))
def cb_kpi(yr, ports_sel, metric):
    ts = _agg("ts", yr, ports_sel)
    if ts.empty:
        return []
    total = ts[metric].sum() if metric in ts.columns else 0
    teu   = _agg("ts", yr, ports_sel)["TEU"].sum() if "TEU" in ts.columns else 0
    n_ports = ts["Hafen_DE"].nunique() if "Hafen_DE" in ts.columns else 0
    n_years = ts["Referenzzeitraum_Jahr"].nunique() if "Referenzzeitraum_Jahr" in ts.columns else 0
    return [
        _kpi("fa-weight-hanging", "Gesamt-Tonnage",  _fmt(total) + " t",   ACCENT),
        _kpi("fa-box",            "Gesamt-TEU",       _fmt(teu),             "#e67e22"),
        _kpi("fa-anchor",         "Häfen",            str(n_ports),          "#3498db"),
        _kpi("fa-calendar",       "Jahresspanne",     f"{n_years} Jahre",    "#9b59b6"),
    ]


# ── Tab: Übersicht ────────────────────────────────────────────────────────────

@app.callback(Output("t-overview", "children"),
              Input("yr", "value"), Input("ports", "value"),
              Input("metric", "value"), Input("tabs", "active_tab"))
def cb_overview(yr, ports_sel, metric, active):
    if active != "overview":
        return dash.no_update

    label = next((o["label"] for o in METRIC_OPTIONS if o["value"] == metric), metric)

    # Jährlich
    df_yr = _group_metrics(_agg("ts", yr, ports_sel),
                           ["Referenzzeitraum_Jahr"], metric).rename(
        columns={"Referenzzeitraum_Jahr": "Jahr"})
    fig_yr = go.Figure()
    if not df_yr.empty:
        fig_yr.add_bar(x=df_yr["Jahr"], y=df_yr[metric],
                       marker_color=ACCENT, name="Jährlich")
        fig_yr.add_scatter(x=df_yr["Jahr"], y=df_yr[metric],
                           mode="lines+markers", line=dict(color="#e74c3c", width=2),
                           name="Trend")
    _chart(fig_yr, f"Jährliche {label}", 340)

    # Monatliche Saisonalität
    df_mo = _group_metrics(_agg("ts", yr, ports_sel),
                           ["Referenzzeitraum_Monat"], metric).rename(
        columns={"Referenzzeitraum_Monat": "Monat"})
    mnames = ["Jan","Feb","Mär","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]
    if not df_mo.empty:
        df_mo["Monat_Name"] = df_mo["Monat"].apply(
            lambda m: mnames[int(m)-1] if 1 <= int(m) <= 12 else str(m))
    fig_mo = px.bar(df_mo if not df_mo.empty else pd.DataFrame(),
                    x="Monat_Name" if not df_mo.empty else [], y=metric,
                    color_discrete_sequence=["#3498db"],
                    labels={metric: f"Ø {label}", "Monat_Name": "Monat"})
    _chart(fig_mo, "Saisonale Verteilung", 320)

    # Verkehrsbeziehung
    df_vk = _group_metrics(_agg("vk", yr, ports_sel), ["Verkehrsbeziehung_Label"], metric)
    fig_vk = px.pie(df_vk, names="Verkehrsbeziehung_Label", values=metric,
                    color_discrete_sequence=px.colors.qualitative.Bold, hole=0.35) \
        if not df_vk.empty else go.Figure()
    _chart(fig_vk, "Verkehrsbeziehung", 320)

    # NST2007
    df_nst = _group_metrics(_agg("nst", yr, ports_sel), ["NST2007_Label"], metric)
    if not df_nst.empty:
        df_nst = df_nst.nlargest(12, metric).sort_values(metric, ascending=True)
    fig_nst = px.bar(df_nst, x=metric, y="NST2007_Label", orientation="h",
                     color_discrete_sequence=["#9b59b6"],
                     labels={metric: label, "NST2007_Label": "Güterklasse"}) \
        if not df_nst.empty else go.Figure()
    _chart(fig_nst, "Top-12 Güterklassen (NST2007)", 400)

    return html.Div([
        dbc.Row([
            dbc.Col(_card("Jährliche Entwicklung",
                          dcc.Graph(figure=fig_yr, config={"displayModeBar": False}), "fa-chart-bar"), md=8),
            dbc.Col(_card("Saisonalität",
                          dcc.Graph(figure=fig_mo, config={"displayModeBar": False}), "fa-calendar"), md=4),
        ]),
        dbc.Row([
            dbc.Col(_card("Verkehrsrichtung",
                          dcc.Graph(figure=fig_vk, config={"displayModeBar": False}), "fa-exchange-alt"), md=4),
            dbc.Col(_card("Güterklassen NST2007",
                          dcc.Graph(figure=fig_nst, config={"displayModeBar": False}), "fa-boxes"), md=8),
        ]),
    ])


# ── Tab: Zeitreihe ────────────────────────────────────────────────────────────

@app.callback(Output("t-timeseries", "children"),
              Input("yr", "value"), Input("ports", "value"),
              Input("metric", "value"), Input("tabs", "active_tab"))
def cb_timeseries(yr, ports_sel, metric, active):
    if active != "timeseries":
        return dash.no_update

    label = next((o["label"] for o in METRIC_OPTIONS if o["value"] == metric), metric)
    ts = _ts_from_agg("ts", metric, yr, ports_sel)

    if ts.empty:
        return html.P("Keine Daten für diese Auswahl.", className="text-muted p-4")

    # Zeitreihen-Chart mit gleitenden Durchschnitten
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines",
                             name="Monatlich", line=dict(color=ACCENT, width=1.5), opacity=0.55))
    for w, col, name in [(3, "#e67e22", "3M-Ø"), (12, "#e74c3c", "12M-Ø")]:
        roll = ts.rolling(w, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines",
                                 name=name, line=dict(color=col, width=2.5)))
    _chart(fig, f"Monatliche Zeitreihe: {label}", 400)

    # Jahr × Monat Heatmap
    df_ts = _agg("ts", yr, ports_sel)
    hm_pivot = (
        df_ts.groupby(["Referenzzeitraum_Jahr", "Referenzzeitraum_Monat"], observed=True)[metric]
        .sum()
        .unstack(fill_value=0)
        if not df_ts.empty else None
    )
    if hm_pivot is not None and not hm_pivot.empty:
        mnames = ["Jan","Feb","Mär","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]
        hm_pivot.columns = [mnames[int(c)-1] if 1 <= int(c) <= 12 else str(c)
                            for c in hm_pivot.columns]
        fig_hm = px.imshow(hm_pivot, color_continuous_scale="Teal", aspect="auto",
                           labels=dict(x="Monat", y="Jahr", color=label))
        _chart(fig_hm, "Jahr × Monat Heatmap", 360)
    else:
        fig_hm = go.Figure()

    # YoY Wachstum
    df_yr2 = _group_metrics(df_ts, ["Referenzzeitraum_Jahr"], metric).rename(
        columns={"Referenzzeitraum_Jahr": "Jahr"}).sort_values("Jahr")
    growth = df_yr2[metric].pct_change() * 100 if not df_yr2.empty else pd.Series()
    fig_gr = go.Figure()
    if not growth.empty:
        colors = [ACCENT if v >= 0 else "#e74c3c" for v in growth.values]
        fig_gr.add_bar(x=df_yr2["Jahr"].values, y=growth.values,
                       marker_color=colors, name="YoY %")
        fig_gr.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
    _chart(fig_gr, "Jahr-über-Jahr Wachstum (%)", 280)

    return html.Div([
        _card("Zeitreihe mit gleitenden Durchschnitten",
              dcc.Graph(figure=fig, config={"displayModeBar": "hover"}), "fa-chart-line"),
        dbc.Row([
            dbc.Col(_card("Saisonale Heatmap",
                          dcc.Graph(figure=fig_hm, config={"displayModeBar": False}), "fa-th"), md=7),
            dbc.Col(_card("YoY Wachstum",
                          dcc.Graph(figure=fig_gr, config={"displayModeBar": False}), "fa-percent"), md=5),
        ]),
    ])


# ── Tab: ML-Prognose (Steuerpanel) ───────────────────────────────────────────

@app.callback(Output("t-forecast", "children"),
              Input("tabs", "active_tab"))
def cb_forecast_layout(active):
    if active != "forecast":
        return dash.no_update
    return html.Div([
        _card("Konfiguration", dbc.Row([
            dbc.Col([dbc.Label("Modell", className="small text-muted"),
                     dcc.Dropdown(id="fc-model", options=MODEL_OPTIONS,
                                  value="random_forest", clearable=False,
                                  style={"background": "#252538"})], md=4),
            dbc.Col([dbc.Label("Prognosehorizont (Monate)", className="small text-muted"),
                     dcc.Slider(id="fc-horizon", min=3, max=36, step=3, value=12,
                                marks={i: str(i) for i in range(3, 37, 6)},
                                tooltip={"placement": "bottom"})], md=5),
            dbc.Col([dbc.Button([html.I(className="fas fa-play me-2"), "Prognose"],
                                id="fc-run", color="success", className="w-100 mt-3",
                                style={"background": ACCENT, "border": "none"})], md=3),
        ]), icon="fa-brain"),
        dbc.Spinner(html.Div(id="fc-output"), color="success"),
    ])


@app.callback(
    Output("fc-output", "children"),
    Input("fc-run", "n_clicks"),
    State("yr", "value"), State("ports", "value"), State("metric", "value"),
    State("fc-model", "value"), State("fc-horizon", "value"),
    prevent_initial_call=True,
)
def cb_run_forecast(_, yr, ports_sel, metric, model_name, horizon):
    label = next((o["label"] for o in METRIC_OPTIONS if o["value"] == metric), metric)
    ts = _ts_from_agg("ts", metric, yr, ports_sel)
    if ts.empty:
        return dbc.Alert("Keine Daten verfügbar.", color="warning")

    result = build_forecast(ts, model_name=model_name, horizon=horizon)
    if "error" in result:
        return dbc.Alert(f"Fehler: {result['error']}", color="danger")

    idx = result["ts_index"]
    vals = result["ts_values"]
    tr_end = result["train_end_idx"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx[:tr_end], y=vals[:tr_end], mode="lines",
                             name="Training", line=dict(color="#3498db", width=2)))
    fig.add_trace(go.Scatter(x=idx[tr_end:], y=result["test_actual"], mode="lines+markers",
                             name="Test (ist)", line=dict(color=ACCENT, width=2.5)))
    fig.add_trace(go.Scatter(x=idx[tr_end:], y=result["test_pred"], mode="lines",
                             name="Test (pred.)", line=dict(color="#e67e22", width=2, dash="dot")))
    x_f = result["future_dates"]
    fig.add_trace(go.Scatter(
        x=list(x_f) + list(x_f[::-1]),
        y=list(result["upper_ci"]) + list(result["lower_ci"][::-1]),
        fill="toself", fillcolor="rgba(0,212,170,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="95% KI",
    ))
    fig.add_trace(go.Scatter(x=x_f, y=result["forecast"], mode="lines+markers",
                             name="Prognose", line=dict(color="#e74c3c", width=3), marker=dict(size=6)))
    vline_x = str(idx[-1]) if hasattr(idx[-1], "isoformat") else idx[-1]
    fig.add_shape(type="line", x0=vline_x, x1=vline_x, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(dash="dash", color="#666"))
    fig.add_annotation(x=vline_x, y=1, xref="x", yref="paper",
                       text="Heute", showarrow=False, yanchor="bottom")
    _chart(fig, f"Prognose: {label}", 440)

    m = result["metrics"]
    kpi_row = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.P("MAE", className="text-muted small mb-1"),
                                       html.H5(f"{m['MAE']:,.1f}", style={"color": "#3498db"})]),
                         style=DARK_CARD), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.P("RMSE", className="text-muted small mb-1"),
                                       html.H5(f"{m['RMSE']:,.1f}", style={"color": "#e67e22"})]),
                         style=DARK_CARD), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.P("R² (Test)", className="text-muted small mb-1"),
                                       html.H5(f"{m['R²']:.3f}",
                                               style={"color": ACCENT if m["R²"] > 0.5 else "#e74c3c"})]),
                         style=DARK_CARD), md=4),
    ], className="mb-3")

    fi_block = html.Div()
    if result.get("feature_importance"):
        fi = dict(list(result["feature_importance"].items())[:12])
        fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
        fig_fi = px.bar(fi_df.sort_values("Importance", ascending=True),
                        x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Teal")
        _chart(fig_fi, "Feature Importance", 340)
        fi_block = _card("Feature Importance",
                         dcc.Graph(figure=fig_fi, config={"displayModeBar": False}),
                         "fa-sort-amount-down")

    residuals = result["test_actual"] - result["test_pred"]
    fig_res = make_subplots(rows=1, cols=2,
                            subplot_titles=["Residuen (zeitlich)", "Histogramm"])
    fig_res.add_scatter(y=residuals, mode="markers",
                        marker=dict(color=ACCENT, opacity=0.6), name="Resid.", row=1, col=1)
    fig_res.add_hline(y=0, line_dash="dash", line_color="#666", row=1, col=1)
    fig_res.add_histogram(x=residuals, nbinsx=30, marker_color="#9b59b6",
                          opacity=0.8, name="Hist.", row=1, col=2)
    _chart(fig_res, height=300)

    return html.Div([
        _card(f"Prognose: {label}  [{model_name.replace('_',' ').title()}]",
              dcc.Graph(figure=fig, config={"displayModeBar": "hover"}), "fa-chart-line"),
        dbc.Row([
            dbc.Col([html.H6("Modellgüte (Test-Split)", className="text-muted mb-3"),
                     kpi_row, fi_block], md=6),
            dbc.Col(_card("Residuenanalyse",
                          dcc.Graph(figure=fig_res, config={"displayModeBar": False}),
                          "fa-wave-square"), md=6),
        ]),
    ])


# ── Tab: Regionen ─────────────────────────────────────────────────────────────

@app.callback(Output("t-regions", "children"),
              Input("yr", "value"), Input("ports", "value"),
              Input("metric", "value"), Input("tabs", "active_tab"))
def cb_regions(yr, ports_sel, metric, active):
    if active != "regions":
        return dash.no_update

    label = next((o["label"] for o in METRIC_OPTIONS if o["value"] == metric), metric)

    # Einlade-Makroregion
    df_ein = _group_metrics(_agg("einlade", yr, ports_sel),
                            ["Einladeregion_Makroregion"], metric).sort_values(metric, ascending=True)
    fig_ein = px.bar(df_ein, x=metric, y="Einladeregion_Makroregion", orientation="h",
                     color=metric, color_continuous_scale="Teal",
                     labels={metric: label, "Einladeregion_Makroregion": "Region"}) \
        if not df_ein.empty else go.Figure()
    _chart(fig_ein, f"Einladeregionen nach {label}", 420)

    # Top-20 Länder (Einlade-ISO)
    df_iso = _group_metrics(_agg("einlade_iso", yr, ports_sel), ["Einladeregion_ISO"], metric)
    if not df_iso.empty:
        df_iso = df_iso.nlargest(20, metric).sort_values(metric, ascending=True)
    fig_iso = px.bar(df_iso, x=metric, y="Einladeregion_ISO", orientation="h",
                     color_discrete_sequence=[ACCENT],
                     labels={metric: label, "Einladeregion_ISO": "Land (ISO-2)"}) \
        if not df_iso.empty else go.Figure()
    _chart(fig_iso, "Top-20 Länder (Einladung)", 500)

    # Sankey
    df_sk = _group_metrics(
        _agg("sankey", yr, ports_sel),
        ["Einladeregion_Makroregion", "Ausladeregion_Makroregion"], metric
    )
    sankey_block = html.Div()
    if not df_sk.empty and len(df_sk) > 1:
        df_sk = df_sk[df_sk[metric] > 0].nlargest(40, metric)
        all_nodes = pd.unique(
            df_sk[["Einladeregion_Makroregion", "Ausladeregion_Makroregion"]].values.ravel()
        )
        node_idx = {n: i for i, n in enumerate(all_nodes)}
        colors = px.colors.qualitative.Pastel * 4
        fig_sk = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=18, label=list(all_nodes),
                      color=[colors[i % len(colors)] for i in range(len(all_nodes))]),
            link=dict(
                source=[node_idx[r] for r in df_sk["Einladeregion_Makroregion"]],
                target=[node_idx[r] for r in df_sk["Ausladeregion_Makroregion"]],
                value=df_sk[metric].values,
                color="rgba(0,212,170,0.18)",
            ),
        ))
        _chart(fig_sk, f"Handelsstrom: Einladung → Ausladung ({label})", 460)
        sankey_block = _card("Handelsstrom-Sankey",
                             dcc.Graph(figure=fig_sk, config={"displayModeBar": False}),
                             "fa-project-diagram")

    return html.Div([
        dbc.Row([
            dbc.Col(_card("Makroregionen (Einladung)",
                          dcc.Graph(figure=fig_ein, config={"displayModeBar": False}),
                          "fa-globe-europe"), md=5),
            dbc.Col(_card("Top-20 Länder",
                          dcc.Graph(figure=fig_iso, config={"displayModeBar": False}),
                          "fa-flag"), md=7),
        ]),
        sankey_block,
    ])


# ── Tab: Häfen & Schiffe ──────────────────────────────────────────────────────

@app.callback(Output("t-ports", "children"),
              Input("yr", "value"), Input("ports", "value"),
              Input("metric", "value"), Input("tabs", "active_tab"))
def cb_ports(yr, ports_sel, metric, active):
    if active != "ports":
        return dash.no_update

    label = next((o["label"] for o in METRIC_OPTIONS if o["value"] == metric), metric)

    # Top-Häfen
    df_hp = _group_metrics(_agg("ts", yr, ports_sel), ["Hafen_DE"], metric)
    if not df_hp.empty:
        df_hp = df_hp[~df_hp["Hafen_DE"].isin(["International", "Unbekannt"])]
        df_hp = df_hp.nlargest(15, metric).sort_values(metric, ascending=True)
    fig_hp = px.bar(df_hp, x=metric, y="Hafen_DE", orientation="h",
                    color=metric, color_continuous_scale="Blues",
                    labels={metric: label, "Hafen_DE": "Hafen"}) \
        if not df_hp.empty else go.Figure()
    _chart(fig_hp, f"Top-Häfen nach {label}", 420)

    # Schiffsart
    df_sh = _group_metrics(_agg("schiffsart", yr, ports_sel), ["Schiffsart_Label"], metric)
    fig_sh = px.pie(df_sh, names="Schiffsart_Label", values=metric,
                    color_discrete_sequence=px.colors.qualitative.Bold, hole=0.38) \
        if not df_sh.empty else go.Figure()
    if not df_sh.empty:
        fig_sh.update_traces(textinfo="label+percent")
    _chart(fig_sh, "Schiffsarten", 360)

    # Flaggenstaaten
    df_fl = _group_metrics(_agg("flagge", yr, ports_sel), ["Flagge"], metric)
    if not df_fl.empty:
        df_fl = df_fl.nlargest(15, metric).sort_values(metric, ascending=True)
    fig_fl = px.bar(df_fl, x=metric, y="Flagge", orientation="h",
                    color_discrete_sequence=["#9b59b6"],
                    labels={metric: label, "Flagge": "Flaggenstaat"}) \
        if not df_fl.empty else go.Figure()
    _chart(fig_fl, "Top-15 Flaggenstaaten", 400)

    # Hafen-Zeitverlauf (Top-5)
    ts_block = html.Div()
    df_ts_hp = _agg("ts", yr, ports_sel)
    if not df_ts_hp.empty and "Hafen_DE" in df_ts_hp.columns:
        top5 = (
            df_ts_hp[~df_ts_hp["Hafen_DE"].isin(["International", "Unbekannt"])]
            .groupby("Hafen_DE", observed=True)[metric].sum()
            .nlargest(5).index.tolist()
        )
        df_top5 = df_ts_hp[df_ts_hp["Hafen_DE"].isin(top5)].copy()
        df_top5["Datum"] = pd.to_datetime(
            df_top5["Referenzzeitraum_Jahr"].astype(str) + "-" +
            df_top5["Referenzzeitraum_Monat"].astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
        ts_data = (df_top5.groupby(["Datum", "Hafen_DE"], observed=True)[metric]
                   .sum().reset_index())
        fig_ts = px.line(ts_data, x="Datum", y=metric, color="Hafen_DE",
                         labels={metric: label, "Hafen_DE": "Hafen"},
                         color_discrete_sequence=px.colors.qualitative.Bold)
        _chart(fig_ts, "Top-5 Häfen im Zeitverlauf", 380)
        ts_block = _card("Hafenentwicklung",
                         dcc.Graph(figure=fig_ts, config={"displayModeBar": "hover"}),
                         "fa-chart-area")

    return html.Div([
        dbc.Row([
            dbc.Col(_card("Häfen-Ranking",
                          dcc.Graph(figure=fig_hp, config={"displayModeBar": False}),
                          "fa-anchor"), md=7),
            dbc.Col(_card("Schiffsarten",
                          dcc.Graph(figure=fig_sh, config={"displayModeBar": False}),
                          "fa-ship"), md=5),
        ]),
        dbc.Row([
            dbc.Col(_card("Flaggenstaaten",
                          dcc.Graph(figure=fig_fl, config={"displayModeBar": False}),
                          "fa-flag"), md=5),
            dbc.Col(ts_block, md=7),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Start
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket

    def _local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"

    print("\n" + "=" * 65)
    print("  Seeverkehr Analytics Dashboard")
    print("=" * 65)
    print(f"  Modus     : {'DEMO' if IS_DEMO else 'Echtdaten'}")
    print(f"  Datensatz : {INFO_TEXT}")
    print(f"  Lokal     : http://127.0.0.1:8050")
    print(f"  Netzwerk  : http://{_local_ip()}:8050")
    print("=" * 65 + "\n")

    app.run(debug=False, host="0.0.0.0", port=8050)
