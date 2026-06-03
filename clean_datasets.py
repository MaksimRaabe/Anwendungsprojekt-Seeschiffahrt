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

# Spalten, die nur fehlend sein duerfen, wenn kein Containertransport vorliegt
CONTAINER_OPTIONAL = [
    "Container_Ladezustand",
    "Container_Ladezustand_Label",
    "Container_Groesse",
    "Container_Groesse_Label",
    "TEU",
]

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
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",   # utf-8 mit BOM
        decimal=",",
        dtype=str,               # erstmal alles als String einlesen
        low_memory=False,
    )
    df.columns = df.columns.str.strip()
    logger.info("  Eingelesen: %d Zeilen, %d Spalten", len(df), len(df.columns))
    return df


def log_summary(label: str, df: pd.DataFrame) -> None:
    logger.info("  [%s] Shape: %s", label, df.shape)


def drop_fully_empty_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.dropna(how="all")
    removed = before - len(df)
    if removed:
        logger.warning("  Vollstaendig leere Zeilen entfernt: %d", removed)
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
    """
    Strategie:
      - Container-optionale Felder: NaN ist akzeptabel (kein Containertransport).
      - Numerische Pflichtfelder (Tonnen): Zeilen mit NaN entfernen.
      - Kategoriale Code-Felder: Mit 'UNBEKANNT' auffuellen und protokollieren.
    """
    report = {}

    # Zeilen ohne Tonnen-Angabe entfernen
    if "Tonnen" in df.columns:
        before = len(df)
        df = df.dropna(subset=["Tonnen"])
        removed = before - len(df)
        if removed:
            logger.warning("  Zeilen ohne 'Tonnen'-Wert entfernt: %d", removed)
            report["Tonnen_removed"] = removed

    # Kategoriale Code-Spalten auffuellen
    cat_cols = [c for c in df.columns if c not in NUMERIC_COLS + CONTAINER_OPTIONAL]
    for col in cat_cols:
        n_miss = df[col].isna().sum()
        if n_miss:
            df[col] = df[col].fillna("UNBEKANNT")
            logger.info("  '%s': %d fehlende Werte → 'UNBEKANNT'", col, n_miss)
            report[col] = int(n_miss)

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
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


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


# ---------------------------------------------------------------------------
# Haupt-Pipeline
# ---------------------------------------------------------------------------

def clean_file(path: Path) -> dict:
    """Bereinigt eine einzelne CSV-Datei und speichert das Ergebnis."""
    logger.info("=" * 70)
    logger.info("Datei: %s", path.name)

    cleaning_report: dict = {"file": path.name, "steps": {}}

    # 1 – Einlesen
    df = read_csv(path)
    cleaning_report["initial_shape"] = list(df.shape)

    # 2 – Vollstaendig leere Zeilen
    df, n_empty = drop_fully_empty_rows(df)
    cleaning_report["steps"]["leere_zeilen_entfernt"] = n_empty

    # 3 – Pflichtfelder pruefen
    validate_required_cols(df, path.name)

    # 4 – Whitespace bereinigen
    df = strip_whitespace(df)

    # 5 – Numerische Konvertierung
    df, conv_errors = convert_numeric(df)
    cleaning_report["steps"]["konvertierungsfehler"] = conv_errors

    # 6 – Fehlende Werte behandeln
    df, missing_report = handle_missing_values(df)
    cleaning_report["steps"]["fehlende_werte"] = missing_report

    # 7 – Bereichsvalidierung
    range_violations = validate_ranges(df)
    cleaning_report["steps"]["bereichsverletzungen"] = range_violations

    # 8 – ISO-Code-Validierung
    iso_violations = validate_iso_codes(df)
    cleaning_report["steps"]["iso_fehler"] = iso_violations

    # 9 – Duplikate entfernen
    df, n_dupes = remove_duplicates(df)
    cleaning_report["steps"]["duplikate_entfernt"] = n_dupes

    # 10 – Ausreisser markieren
    outlier_report = detect_outliers_iqr(df)
    cleaning_report["steps"]["ausreisser"] = outlier_report

    # Abschlussbericht
    cleaning_report["final_shape"] = list(df.shape)
    log_summary("Ergebnis", df)

    # Speichern
    out_path = OUTPUT_DIR / path.name
    df.to_csv(out_path, sep=";", encoding="utf-8-sig", decimal=",", index=False)
    logger.info("  Gespeichert: %s", out_path)

    return cleaning_report


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

    all_reports = []
    for csv_path in csv_files:
        try:
            report = clean_file(csv_path)
            all_reports.append(report)
        except Exception as exc:
            logger.error("Fehler bei '%s': %s", csv_path.name, exc, exc_info=True)

    # Gesamt-Report als JSON speichern
    report_path = LOG_DIR / f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info("Abgeschlossen. Report: %s", report_path)
    logger.info("Bereinigte Dateien: %s", OUTPUT_DIR)


if __name__ == "__main__":
    run()
