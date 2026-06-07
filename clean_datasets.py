"""
Datenbereinigungs-Pipeline fuer die Seeverkehrsstatistik (Destatis Open Data).

Dateiformat gemaess MRTM-Datensatzbeschreibung:
  - Kodierung : UTF-8 mit BOM
  - Trennzeichen : Semikolon (;)
  - Dezimalzeichen : Komma (,)

Vorgehen (CRISP-DM – Data Preparation):
  1. Einlesen aller CSV-Dateien aus dem Ordner 'Datasets/'
  2. Formatvalidierung (Feldlaengen, erlaubte Wertebereiche)
  3. Behandlung fehlender Werte
  4. Typ-Konvertierung numerischer Spalten
  5. Ausreisser-Erkennung (IQR-Methode) fuer Tonnen / TEU / Anzahl_Ladungstraeger
  6. Duplikaterkennung
  7. Speichern der bereinigten Daten + Cleaning-Log
"""

import os
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "Datasets"
OUTPUT_DIR = BASE_DIR / "Datasets_cleaned"
LOG_DIR = BASE_DIR / "logs"

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_file = LOG_DIR / f"cleaning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Felddefinitionen gemaess Datensatzbeschreibung
# ---------------------------------------------------------------------------

# Numerische Spalten mit Dezimalkomma
NUMERIC_COLS = ["Tonnen", "TEU", "Anzahl_Ladungstraeger"]

# Spalten-Aliases zwischen unterschiedlichen Datensatz-Versionen
# (z.B. Destatis nutzt teils 'Guetergewicht' statt 'Tonnen')
COLUMN_ALIASES = {
    "Guetergewicht": "Tonnen",
}

# Spalten, die nur fehlend sein duerfen, wenn kein Containertransport vorliegt
CONTAINER_OPTIONAL = [
    "Container_Ladezustand",
    "Container_Ladezustand_Label",
    "Container_Groesse",
    "Container_Groesse_Label",
    "TEU",
]

# Ausladeregion-Spalten, die untereinander auf Konsistenz geprueft und
# gegenseitig zur Auffuellung fehlender Werte verwendet werden.
AUSLADE_REGION_SYNC_COLS = [
    "Ausladeregion_NUTS3",
    "Ausladeregion_NUTS3_Label",
    "Ausladeregion_UNLOCODE",
    "Ausladeregion_HafenID",
]

# Einladeregion-Spalten (spiegelbildlich zu Ausladeregion).
EINLADE_REGION_SYNC_COLS = [
    "Einladeregion_NUTS3",
    "Einladeregion_NUTS3_Label",
    "Einladeregion_UNLOCODE",
    "Einladeregion_HafenID",
]

# ISO-2-Codes der Laender, fuer die NUTS3-Codes existieren (EU-27 + EEA).
# Fehlende NUTS3-Werte fuer Haefen ausserhalb dieser Menge sind strukturell erwartet.
NUTS3_COUNTRIES = {
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI",
    "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK",  # EU-27
    "NO", "IS", "LI",                           # EEA
}

# Mapping: NUTS3-Spalte → zugehoerige ISO-Spalte fuer die Erwartet/Unerwartet-Analyse.
NUTS3_COL_ISO_MAP = {
    "Ausladeregion_NUTS3":       "Ausladeregion_ISO",
    "Ausladeregion_NUTS3_Label": "Ausladeregion_ISO",
    "Einladeregion_NUTS3":       "Einladeregion_ISO",
    "Einladeregion_NUTS3_Label": "Einladeregion_ISO",
}

# Mapping ISO-2 → uebergeordnete Makroregion fuer Zeilen ohne NUTS3-Code.
# Wird fuer Auslade- und Einladeregion gleichermassen verwendet.
ISO_TO_MAKROREGION: dict[str, str] = {
    # Europa – EU/EEA (haben NUTS3, werden trotzdem einheitlich gemappt)
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
    # Europa – nicht EU/EEA
    "GB": "Europa (nicht EU/EEA)", "CH": "Europa (nicht EU/EEA)",
    "TR": "Europa (nicht EU/EEA)", "BA": "Europa (nicht EU/EEA)",
    "RS": "Europa (nicht EU/EEA)", "ME": "Europa (nicht EU/EEA)",
    "MK": "Europa (nicht EU/EEA)", "AL": "Europa (nicht EU/EEA)",
    "XK": "Europa (nicht EU/EEA)", "MC": "Europa (nicht EU/EEA)",
    "AD": "Europa (nicht EU/EEA)", "SM": "Europa (nicht EU/EEA)",
    # Osteuropa / GUS
    "RU": "Osteuropa/GUS", "UA": "Osteuropa/GUS", "BY": "Osteuropa/GUS",
    "MD": "Osteuropa/GUS", "GE": "Osteuropa/GUS", "AM": "Osteuropa/GUS",
    "AZ": "Osteuropa/GUS", "KZ": "Osteuropa/GUS", "UZ": "Osteuropa/GUS",
    "TM": "Osteuropa/GUS", "KG": "Osteuropa/GUS", "TJ": "Osteuropa/GUS",
    # Nordafrika
    "MA": "Nordafrika", "DZ": "Nordafrika", "TN": "Nordafrika",
    "LY": "Nordafrika", "EG": "Nordafrika",
    # Westafrika
    "SN": "Westafrika", "GM": "Westafrika", "GN": "Westafrika",
    "GW": "Westafrika", "SL": "Westafrika", "LR": "Westafrika",
    "CI": "Westafrika", "GH": "Westafrika", "TG": "Westafrika",
    "BJ": "Westafrika", "NG": "Westafrika", "CM": "Westafrika",
    "GA": "Westafrika", "CG": "Westafrika", "CD": "Westafrika",
    "AO": "Westafrika", "CV": "Westafrika", "ST": "Westafrika",
    "GQ": "Westafrika",
    # Ost- und Suedafrika
    "TZ": "Ost-/Suedafrika", "KE": "Ost-/Suedafrika", "MZ": "Ost-/Suedafrika",
    "MG": "Ost-/Suedafrika", "ZA": "Ost-/Suedafrika", "NA": "Ost-/Suedafrika",
    "MU": "Ost-/Suedafrika", "DJ": "Ost-/Suedafrika", "SO": "Ost-/Suedafrika",
    "ER": "Ost-/Suedafrika", "ET": "Ost-/Suedafrika", "SD": "Ost-/Suedafrika",
    "SS": "Ost-/Suedafrika", "RE": "Ost-/Suedafrika", "ZM": "Ost-/Suedafrika",
    "ZW": "Ost-/Suedafrika", "MW": "Ost-/Suedafrika", "BI": "Ost-/Suedafrika",
    "RW": "Ost-/Suedafrika", "UG": "Ost-/Suedafrika", "KM": "Ost-/Suedafrika",
    "SC": "Ost-/Suedafrika",
    # Naher Osten
    "SA": "Naher Osten", "AE": "Naher Osten", "QA": "Naher Osten",
    "KW": "Naher Osten", "OM": "Naher Osten", "BH": "Naher Osten",
    "YE": "Naher Osten", "IR": "Naher Osten", "IQ": "Naher Osten",
    "IL": "Naher Osten", "JO": "Naher Osten", "SY": "Naher Osten",
    "LB": "Naher Osten",
    # Suedasien
    "IN": "Suedasien", "PK": "Suedasien", "BD": "Suedasien",
    "LK": "Suedasien", "MM": "Suedasien", "MV": "Suedasien",
    # Ostasien
    "CN": "Ostasien", "JP": "Ostasien", "KR": "Ostasien",
    "HK": "Ostasien", "TW": "Ostasien", "MO": "Ostasien",
    # Suedostasien
    "SG": "Suedostasien", "TH": "Suedostasien", "VN": "Suedostasien",
    "ID": "Suedostasien", "MY": "Suedostasien", "PH": "Suedostasien",
    "KH": "Suedostasien", "BN": "Suedostasien", "TL": "Suedostasien",
    # Nordamerika
    "US": "Nordamerika", "CA": "Nordamerika", "MX": "Nordamerika",
    "PR": "Nordamerika", "BM": "Nordamerika", "VI": "Nordamerika",
    # Mittelamerika / Karibik
    "PA": "Mittelamerika/Karibik", "CR": "Mittelamerika/Karibik",
    "GT": "Mittelamerika/Karibik", "HN": "Mittelamerika/Karibik",
    "NI": "Mittelamerika/Karibik", "SV": "Mittelamerika/Karibik",
    "BZ": "Mittelamerika/Karibik", "CU": "Mittelamerika/Karibik",
    "JM": "Mittelamerika/Karibik", "HT": "Mittelamerika/Karibik",
    "DO": "Mittelamerika/Karibik", "TT": "Mittelamerika/Karibik",
    "BB": "Mittelamerika/Karibik", "BS": "Mittelamerika/Karibik",
    "AW": "Mittelamerika/Karibik", "CW": "Mittelamerika/Karibik",
    # Suedamerika
    "BR": "Suedamerika", "AR": "Suedamerika", "CL": "Suedamerika",
    "CO": "Suedamerika", "PE": "Suedamerika", "VE": "Suedamerika",
    "EC": "Suedamerika", "UY": "Suedamerika", "GY": "Suedamerika",
    "SR": "Suedamerika", "GF": "Suedamerika",
    # Ozeanien
    "AU": "Ozeanien", "NZ": "Ozeanien", "FJ": "Ozeanien",
    "PG": "Ozeanien", "PF": "Ozeanien", "NC": "Ozeanien",
    "GU": "Ozeanien",
}

# Erwartete Wertebereich-Pruefungen (Code-Spalten)
FIELD_CONSTRAINTS = {
    "Referenzzeitraum_Monat": (1, 12),
    "Referenzzeitraum_Jahr": (2000, 2030),
    "Verkehrsbeziehung": (1, 4),
}

# Alle Pflichtfelder (duerfen nicht vollstaendig leer sein)
REQUIRED_COLS = [
    "EVAS",
    "EVAS_Label",
    "Referenzzeitraum_Jahr",
    "Referenzzeitraum_Monat",
    "Einladeregion_ISO",
    "Ausladeregion_ISO",
    "Verkehrsbeziehung",
    "Schiffsart",
    "Flagge",
    "NST2007",
    "Tonnen",
]

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> pd.DataFrame:
    """Liest eine Destatis-CSV (UTF-8-BOM, Semikolon, Dezimalkomma)."""
    if not path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Pfad ist keine Datei (sondern Ordner?): {path}")

    # Erst schnell/streng lesen (C-Engine). Falls einzelne Zeilen kaputt sind,
    # versuchen wir einen toleranteren Fallback.
    try:
        df = pd.read_csv(
            path,
            sep=";",
            encoding="utf-8-sig",   # utf-8 mit BOM
            decimal=",",
            dtype=str,               # erstmal alles als String einlesen
            low_memory=False,
        )
    except pd.errors.ParserError as exc:
        logger.warning(
            "  ParserError beim Einlesen (%s). Fallback: python-engine + on_bad_lines='warn'.",
            exc,
        )
        df = pd.read_csv(
            path,
            sep=";",
            encoding="utf-8-sig",
            decimal=",",
            dtype=str,
            low_memory=False,
            engine="python",
            on_bad_lines="warn",
        )
    except PermissionError as exc:
        raise PermissionError(
            f"Zugriff verweigert auf {path}. Datei ggf. in Excel/DataWrangler geoeffnet?"
        ) from exc

    df = normalize_schema(df)
    logger.info("  Eingelesen: %d Zeilen, %d Spalten", len(df), len(df.columns))
    return df


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Vereinheitlicht Spaltennamen ueber unterschiedliche Datei-Versionen."""
    df.columns = df.columns.str.strip()
    present = set(df.columns)
    for old, new in COLUMN_ALIASES.items():
        if old in present and new not in present:
            df = df.rename(columns={old: new})
    return df


def log_summary(label: str, df: pd.DataFrame) -> None:
    logger.info("  [%s] Shape: %s", label, df.shape)


REQUIRED_FOR_ROW = [
    "EVAS",
    "Referenzzeitraum_Jahr",
    "Referenzzeitraum_Monat",
    "Einladeregion_ISO",
    "Ausladeregion_ISO",
    "Verkehrsbeziehung",
    "Schiffsart",
    "Flagge",
    "NST2007",
    "Tonnen",
]


def drop_rows_missing_required(df: pd.DataFrame, required: list[str]) -> tuple[pd.DataFrame, int]:
    """Entfernt nur Zeilen, bei denen Kernspalten fehlen (statt jede NA irgendwo)."""
    before = len(df)
    # Leere Strings ebenfalls als fehlend behandeln
    df = df.replace(r"^\s*$", np.nan, regex=True)

    required_present = [c for c in required if c in df.columns]
    if not required_present:
        logger.warning("  Keine der REQUIRED_FOR_ROW-Spalten im DataFrame gefunden; kein Drop ausgefuehrt.")
        return df, 0

    df = df.dropna(subset=required_present)
    removed = before - len(df)
    logger.warning("  Zeilen mit fehlenden Kernfeldern entfernt: %d", removed)
    return df, removed


def validate_required_cols(df: pd.DataFrame, filename: str) -> None:
    present = df.columns.tolist()
    for col in REQUIRED_COLS:
        if col not in present:
            logger.warning("  [%s] Pflichtfeld fehlt in Schema: '%s'", filename, col)


def convert_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Konvertiert NUMERIC_COLS; zaehlt und protokolliert Konvertierungsfehler."""
    errors = {}
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        # Komma als Dezimaltrennzeichen (bereits via read_csv decimal=',' gehandelt,
        # aber da dtype=str, muss manuell konvertiert werden)
        series_clean = df[col].str.replace(",", ".", regex=False).str.strip()
        numeric = pd.to_numeric(series_clean, errors="coerce")
        n_err = numeric.isna().sum() - df[col].isna().sum()
        n_err = max(n_err, 0)
        if n_err:
            logger.warning("  Konvertierungsfehler in '%s': %d Werte → NaN gesetzt", col, n_err)
            errors[col] = int(n_err)
        df[col] = numeric
    return df, errors


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Protokolliert fehlende Werte.

    Wichtig: Seitdem nur noch Zeilen mit fehlenden *Kernfeldern* entfernt werden,
    sind NaN in optionalen Spalten (z.B. Container-Merkmale) normal und sollen
    nicht als Fehler bewertet werden.
    """
    report: dict = {}

    na_per_col = df.isna().sum()
    total_na = int(na_per_col.sum())
    report["na_total"] = total_na

    if total_na == 0:
        logger.info("  Keine fehlenden Werte verblieben.")
        return df, report

    # Nur Top-N Spalten reporten, damit JSON/Logs nicht explodieren.
    top_n = 10
    na_top = na_per_col[na_per_col > 0].sort_values(ascending=False).head(top_n)
    report["na_top_cols"] = {k: int(v) for k, v in na_top.items()}

    # Extra: optionalen Container-Block gesondert ausweisen
    container_cols_present = [c for c in CONTAINER_OPTIONAL if c in df.columns]
    if container_cols_present:
        container_na = int(df[container_cols_present].isna().sum().sum())
        report["na_container_optional_total"] = container_na

    # Extra: NUTS3-Luecken in erwartet (Nicht-EU/EEA-Hafen) und unerwartet aufteilen
    nuts3_analysis: dict = {}
    for nuts3_col, iso_col in NUTS3_COL_ISO_MAP.items():
        if nuts3_col not in df.columns or iso_col not in df.columns:
            continue
        is_nuts3_country = df[iso_col].isin(NUTS3_COUNTRIES)
        missing_mask = df[nuts3_col].isna()
        expected = int((missing_mask & ~is_nuts3_country).sum())
        unexpected = int((missing_mask & is_nuts3_country).sum())
        nuts3_analysis[nuts3_col] = {
            "erwartet_fehlend_nicht_eu_eea": expected,
            "unerwartet_fehlend_eu_eea": unexpected,
        }
        if unexpected:
            logger.warning(
                "  Unerwartete NUTS3-Luecken in '%s' (EU/EEA-Hafen ohne NUTS3-Code): %d",
                nuts3_col, unexpected,
            )
        else:
            logger.info(
                "  '%s': alle %d fehlenden Werte strukturell erwartet (Nicht-EU/EEA-Haefen).",
                nuts3_col, expected,
            )
    if nuts3_analysis:
        report["nuts3_missing_analyse"] = nuts3_analysis

    logger.info("  Fehlende Werte gesamt: %d (Top-%d Spalten im Report)", total_na, top_n)
    return df, report


def validate_ranges(df: pd.DataFrame) -> dict:
    """Prueft numerische Code-Felder auf erlaubte Wertebereiche."""
    violations = {}
    for col, (lo, hi) in FIELD_CONSTRAINTS.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        mask = numeric.notna() & ((numeric < lo) | (numeric > hi))
        n = mask.sum()
        if n:
            logger.warning(
                "  Bereichsverletzung '%s' [%d–%d]: %d Zeilen", col, lo, hi, n
            )
            violations[col] = int(n)
    return violations


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 3.0) -> dict:
    """
    IQR-basierte Ausreisser-Erkennung fuer numerische Wertspalten.
    Ausreisser werden markiert (nicht entfernt), um Datenverlust zu vermeiden.
    """
    report = {}
    for col in NUMERIC_COLS:
        if col not in df.columns or df[col].isna().all():
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        n = mask.sum()
        df[f"{col}_outlier"] = mask
        if n:
            logger.warning(
                "  Ausreisser in '%s' (IQR x%.1f): %d Zeilen  "
                "[Q1=%.2f, Q3=%.2f, Grenzen: %.2f / %.2f]",
                col, factor, n, q1, q3, lower, upper,
            )
        report[col] = {
            "n_outliers": int(n),
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
        }
    return report


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.warning("  Duplikate entfernt: %d", removed)
    return df, removed


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    # Explizit object+string angeben (Pandas 3/4 Kompatibilitaet)
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


def _build_value_map(df: pd.DataFrame, source_col: str, target_col: str) -> dict:
    """Erzeugt eine robuste 1:1-Abbildung aus vorhandenen Quellenwerten."""
    subset = df[[source_col, target_col]].dropna()
    if subset.empty:
        return {}

    mapping = {}
    grouped = subset.groupby(source_col, dropna=True)[target_col]
    for source_value, values in grouped:
        non_null_values = values.dropna()
        if non_null_values.empty:
            continue
        mode = non_null_values.mode(dropna=True)
        mapping[source_value] = mode.iloc[0] if not mode.empty else non_null_values.iloc[0]
    return mapping


def fill_related_region_values(
    df: pd.DataFrame,
    sync_cols: list[str],
    anchor_col: str,
    label: str,
    global_lookup: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fuellt fehlende Regions-Werte mit anchor_col als zentrale Referenz.

    Funktioniert fuer Auslade- und Einladeregion gleichermassen.
    global_lookup: optionaler dateiuebergreifender Lookup
      {(source_col, target_col): {source_val: target_val}}
    Lokale Mappings haben Vorrang vor dem globalen Lookup.
    """
    present_cols = [c for c in sync_cols if c in df.columns]
    other_cols = [c for c in present_cols if c != anchor_col]

    report = {
        "zentrale_referenz": anchor_col,
        "gefuellte_werte": {col: 0 for col in present_cols},
        "verbleibende_missing": {col: int(df[col].isna().sum()) for col in present_cols},
    }

    if anchor_col not in present_cols:
        logger.warning("  %s nicht vorhanden; Auffuellung wird uebersprungen.", anchor_col)
        return df, report

    if not other_cols:
        return df, report

    df = df.copy()
    df[present_cols] = df[present_cols].replace(r"^\s*$", np.nan, regex=True)

    def _merged_map(src: str, tgt: str) -> dict:
        """Globalen Lookup mit lokalem Mapping zusammenfuehren (lokal hat Vorrang)."""
        base = (global_lookup or {}).get((src, tgt), {})
        local = _build_value_map(df, src, tgt)
        return {**base, **local}

    max_passes = 3
    for _ in range(max_passes):
        filled_in_pass = 0

        # Schritt 1: anchor_col hat Vorrang – andere Felder von anchor_col fuellen
        anchor_non_null = df[anchor_col].notna()
        if anchor_non_null.any():
            for target_col in other_cols:
                value_map = _merged_map(anchor_col, target_col)
                if not value_map:
                    continue

                fill_mask = anchor_non_null & df[target_col].isna()
                if not fill_mask.any():
                    continue

                fill_values = df.loc[fill_mask, anchor_col].map(value_map)
                can_fill = fill_values.notna()
                if not can_fill.any():
                    continue

                df.loc[fill_values[can_fill].index, target_col] = fill_values[can_fill].values
                filled_count = int(can_fill.sum())
                report["gefuellte_werte"][target_col] += filled_count
                filled_in_pass += filled_count

        # Schritt 2: Falls anchor_col fehlt, von anderen Feldern fuellen
        anchor_missing = df[anchor_col].isna()
        if anchor_missing.any():
            for source_col in other_cols:
                fill_mask = df[source_col].notna() & anchor_missing
                if not fill_mask.any():
                    continue

                value_map = _merged_map(source_col, anchor_col)
                if not value_map:
                    continue

                fill_values = df.loc[fill_mask, source_col].map(value_map)
                can_fill = fill_values.notna()
                if not can_fill.any():
                    continue

                df.loc[fill_values[can_fill].index, anchor_col] = fill_values[can_fill].values
                filled_count = int(can_fill.sum())
                report["gefuellte_werte"][anchor_col] += filled_count
                filled_in_pass += filled_count
                break  # nur die erste verlaessliche Quelle nutzen

        if filled_in_pass == 0:
            break

    report["verbleibende_missing"] = {
        col: int(df[col].isna().sum()) for col in present_cols
    }
    report["gesamt_gefuellt"] = int(sum(report["gefuellte_werte"].values()))

    if report["gesamt_gefuellt"]:
        logger.info(
            "  %s-Werte (HafenID-zentriert) ergaenzt: %d",
            label,
            report["gesamt_gefuellt"],
        )

    return df, report


def validate_iso_codes(df: pd.DataFrame) -> dict:
    """Prueft, ob ISO-Codes aus genau 2 Buchstaben bestehen."""
    report = {}
    for col in ["Einladeregion_ISO", "Ausladeregion_ISO", "Flagge"]:
        if col not in df.columns:
            continue
        invalid = df[col].dropna()
        invalid = invalid[~invalid.str.match(r"^[A-Z]{2}$", na=False)]
        n = len(invalid)
        if n:
            logger.warning("  Ungueltige ISO-Codes in '%s': %d", col, n)
            report[col] = int(n)
    return report


def derive_makroregion_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Leitet Ausladeregion_Makroregion und Einladeregion_Makroregion aus den ISO-Codes ab.

    Die Spalten werden fuer alle Zeilen gesetzt – auch fuer EU/EEA-Haefen, damit
    Analysen einheitlich auf Makroregion-Ebene moeglich sind, ohne auf NUTS3 angewiesen
    zu sein. Unbekannte ISO-Codes bleiben NaN und werden im Report ausgewiesen.
    """
    report: dict = {}
    for iso_col, target_col in [
        ("Ausladeregion_ISO", "Ausladeregion_Makroregion"),
        ("Einladeregion_ISO", "Einladeregion_Makroregion"),
    ]:
        if iso_col not in df.columns:
            continue
        df[target_col] = df[iso_col].map(ISO_TO_MAKROREGION)
        n_mapped = int(df[target_col].notna().sum())
        n_unmapped = int(df[target_col].isna().sum())
        report[target_col] = {"gemappt": n_mapped, "ungemappt": n_unmapped}
        if n_unmapped:
            unknown = sorted(df.loc[df[target_col].isna(), iso_col].dropna().unique())
            logger.warning(
                "  %s: %d Zeilen ohne Makroregion-Mapping (ISO-Codes: %s%s)",
                target_col, n_unmapped,
                ", ".join(unknown[:10]),
                " ..." if len(unknown) > 10 else "",
            )
        else:
            logger.info("  %s: alle %d Zeilen gemappt.", target_col, n_mapped)
    return df, report


# ---------------------------------------------------------------------------
# Dateiuebergreifender Lookup-Aufbau (Ansatz 3)
# ---------------------------------------------------------------------------

def build_global_lookups(csv_files: list[Path]) -> dict[str, dict]:
    """Liest alle CSV-Dateien einmal vorab und baut spaltenpaarbezogene Lookup-Tabellen.

    Rueckgabe: {
      "auslade": { (source_col, target_col): {source_val: target_val}, ... },
      "einlade": { (source_col, target_col): {source_val: target_val}, ... },
    }
    Lokale Mappings in fill_related_region_values haben spaeter Vorrang; der globale
    Lookup erschliesst nur Werte, die in der Einzeldatei nicht beobachtbar sind.
    """
    frames = []
    for path in csv_files:
        try:
            df = read_csv(path)
            df = strip_whitespace(df)
            df = df.replace(r"^\s*$", np.nan, regex=True)
            frames.append(df)
        except Exception as exc:
            logger.warning("  GlobalLookup: Fehler beim Lesen von '%s': %s", path.name, exc)

    if not frames:
        logger.warning("  GlobalLookup: Keine Dateien lesbar; Lookup bleibt leer.")
        return {"auslade": {}, "einlade": {}}

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "  GlobalLookup: %d Zeilen aus %d Datei(en) kombiniert.",
        len(combined), len(frames),
    )

    lookups: dict[str, dict] = {}
    for key, sync_cols in [
        ("auslade", AUSLADE_REGION_SYNC_COLS),
        ("einlade", EINLADE_REGION_SYNC_COLS),
    ]:
        present = [c for c in sync_cols if c in combined.columns]
        pair_maps: dict[tuple[str, str], dict] = {}
        for src in present:
            for tgt in present:
                if src == tgt:
                    continue
                m = _build_value_map(combined, src, tgt)
                if m:
                    pair_maps[(src, tgt)] = m
        lookups[key] = pair_maps
        logger.info(
            "  GlobalLookup [%s]: %d Spaltenpaare mit Mapping.", key, len(pair_maps)
        )

    return lookups


# ---------------------------------------------------------------------------
# Haupt-Pipeline
# ---------------------------------------------------------------------------

def clean_file(path: Path, global_lookups: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Bereinigt eine einzelne CSV-Datei und gibt den DataFrame + Report zurueck.

    global_lookups: dateiuebergreifende Lookup-Tabellen aus build_global_lookups().
    """
    logger.info("=" * 70)
    logger.info("Datei: %s", path.name)

    cleaning_report: dict = {"file": path.name, "steps": {}}

    # 1 – Einlesen
    df = read_csv(path)
    cleaning_report["initial_shape"] = list(df.shape)

    # 2 – Nur Zeilen entfernen, in denen Kernfelder fehlen
    df, n_missing_required = drop_rows_missing_required(df, REQUIRED_FOR_ROW)
    cleaning_report["steps"]["zeilen_mit_fehlenden_kernfeldern_entfernt"] = n_missing_required

    # 3 – Pflichtfelder pruefen
    validate_required_cols(df, path.name)

    # 4 – Whitespace bereinigen
    df = strip_whitespace(df)

    # 5a – Fehlende Ausladeregion-Werte auffuellen
    df, auslade_fill_report = fill_related_region_values(
        df,
        AUSLADE_REGION_SYNC_COLS,
        "Ausladeregion_HafenID",
        "Ausladeregion",
        (global_lookups or {}).get("auslade"),
    )
    cleaning_report["steps"]["ausladeregion_auffuellung"] = auslade_fill_report

    # 5b – Fehlende Einladeregion-Werte auffuellen
    df, einlade_fill_report = fill_related_region_values(
        df,
        EINLADE_REGION_SYNC_COLS,
        "Einladeregion_HafenID",
        "Einladeregion",
        (global_lookups or {}).get("einlade"),
    )
    cleaning_report["steps"]["einladeregion_auffuellung"] = einlade_fill_report

    # 5c – Makroregion-Spalten aus ISO-Codes ableiten (Fallback fuer Nicht-EU/EEA-Haefen)
    df, makroregion_report = derive_makroregion_cols(df)
    cleaning_report["steps"]["makroregion_ableitung"] = makroregion_report

    # 6 – Numerische Konvertierung
    df, conv_errors = convert_numeric(df)
    cleaning_report["steps"]["konvertierungsfehler"] = conv_errors

    # 7 – Fehlende Werte kontrollieren
    df, missing_report = handle_missing_values(df)
    cleaning_report["steps"]["fehlende_werte"] = missing_report

    # 8 – Bereichsvalidierung
    range_violations = validate_ranges(df)
    cleaning_report["steps"]["bereichsverletzungen"] = range_violations

    # 9 – ISO-Code-Validierung
    iso_violations = validate_iso_codes(df)
    cleaning_report["steps"]["iso_fehler"] = iso_violations

    # 10 – Duplikate entfernen
    df, n_dupes = remove_duplicates(df)
    cleaning_report["steps"]["duplikate_entfernt"] = n_dupes

    # 11 – Ausreisser markieren
    outlier_report = detect_outliers_iqr(df)
    cleaning_report["steps"]["ausreisser"] = outlier_report

    # Herkunftsspalte hinzufuegen
    df.insert(0, "Quelldatei", path.name)

    cleaning_report["final_shape"] = list(df.shape)
    log_summary("Ergebnis", df)

    return df, cleaning_report


def run() -> None:
    if not DATASETS_DIR.exists():
        logger.error(
            "Ordner '%s' nicht gefunden. Bitte CSV-Dateien dort ablegen.", DATASETS_DIR
        )
        return

    csv_files = sorted(DATASETS_DIR.glob("*.csv"))
    if not csv_files:
        logger.warning("Keine CSV-Dateien in '%s' gefunden.", DATASETS_DIR)
        return

    logger.info("Starte Bereinigung von %d Datei(en).", len(csv_files))

    # Einmalig dateiuebergreifende Lookup-Tabellen aus allen Rohdaten aufbauen.
    logger.info("Baue dateiuebergreifende Lookup-Tabellen (Auslade- und Einladeregion)...")
    global_lookups = build_global_lookups(csv_files)

    all_reports = []
    cleaned_frames = []

    for csv_path in csv_files:
        try:
            df_clean, report = clean_file(csv_path, global_lookups=global_lookups)
            cleaned_frames.append(df_clean)
            all_reports.append(report)
        except Exception as exc:
            logger.error("Fehler bei '%s': %s", csv_path.name, exc, exc_info=True)

    # Kombinierten Datensatz erstellen und speichern
    if cleaned_frames:
        combined = pd.concat(cleaned_frames, ignore_index=True)

        # Globale Duplikate (ueber alle Dateien) nochmals entfernen
        before = len(combined)
        combined = combined.drop_duplicates(
            subset=[c for c in combined.columns if c != "Quelldatei"]
        )
        n_global_dupes = before - len(combined)
        if n_global_dupes:
            logger.warning("  Dateiuebergreifende Duplikate entfernt: %d", n_global_dupes)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"seeverkehr_bereinigt_{timestamp}.csv"
        combined.to_csv(out_path, sep=";", encoding="utf-8-sig", decimal=",", index=False)

        logger.info("=" * 70)
        logger.info("Kombinierter Datensatz: %d Zeilen, %d Spalten", *combined.shape)
        logger.info("Gespeichert: %s", out_path)

        # Gesamtstatistik in Report aufnehmen
        all_reports.append({
            "combined": {
                "dateien": len(cleaned_frames),
                "zeilen_gesamt": int(combined.shape[0]),
                "spalten": int(combined.shape[1]),
                "dateiuebergreifende_duplikate_entfernt": n_global_dupes,
                "ausgabedatei": out_path.name,
            }
        })

    # Report als JSON speichern
    report_path = LOG_DIR / f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    run()
