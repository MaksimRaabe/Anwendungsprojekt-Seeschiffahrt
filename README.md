# Anwendungsprojekt-Seeschiffahrt

Analyse und Prognose des Güterverkehrs in der deutschen Seeschifffahrt auf Basis amtlicher Destatis-Statistiken (MRTM-Datensätze des Statistischen Bundesamts).

---

## Überblick

Das Projekt besteht aus zwei Komponenten:

| Komponente | Ordner | Zweck |
|---|---|---|
| Datenbereinigungs-Pipeline | `Dataset bereinigung/` | Rohdaten validieren, bereinigen und zusammenführen |
| Analyse-Dashboard | `Dashboard/` | Interaktive Visualisierung, statistische Analyse und ML-Prognose |

**Verwendete Vorgehensmodell:** CRISP-DM (Cross-Industry Standard Process for Data Mining)

---

## Schritt-für-Schritt-Tutorial

### Voraussetzungen

- Python 3.10 oder neuer
- Die Rohdaten-CSV-Dateien von Destatis (MRTM-Datensätze, Semikolon-getrennt, Dezimalkomma, UTF-8 mit BOM)

### Schritt 1 – Repository klonen

```bash
git clone https://github.com/MaksimRaabe/Anwendungsprojekt-Seeschiffahrt.git
cd Anwendungsprojekt-Seeschiffahrt
```

### Schritt 2 – Abhängigkeiten installieren

**Datenbereinigungs-Pipeline:**

```bash
pip install pandas numpy
```

**Dashboard (alle Pakete):**

```bash
pip install -r Dashboard/requirements.txt
```

Die `requirements.txt` enthält: `dash`, `dash-bootstrap-components`, `plotly`, `pandas`, `numpy`, `scikit-learn`, `pyarrow`, `scipy`, `statsmodels`.

### Schritt 3 – Rohdaten ablegen

Lege alle CSV-Rohdateien von Destatis in den Ordner:

```
Dataset bereinigung/Datasets/
```

Mehrere Jahresdateien können gleichzeitig im Ordner liegen — die Pipeline verarbeitet alle parallel.

### Schritt 4 – Datenbereinigungs-Pipeline ausführen

```bash
cd "Dataset bereinigung"
python clean_datasets.py
```

Die Pipeline läuft parallel (bis zu 3 Prozesse) und gibt laufend Fortschritt aus.
Laufzeit: ca. 1–5 Minuten je nach Datenmenge und Hardware.

**Ausgaben nach erfolgreichem Lauf:**

```
Dataset bereinigung/
├── Datasets_cleaned/
│   └── seeverkehr_bereinigt_YYYYMMDD_HHMMSS.csv   ← bereinigte Gesamtdatei
└── logs/
    ├── cleaning_YYYYMMDD_HHMMSS.log                ← detailliertes Protokoll
    └── cleaning_report_YYYYMMDD_HHMMSS.json        ← maschinenlesbarer Report
```

### Schritt 5 – Dashboard starten

```bash
cd Dashboard
python dashboard.py
```

Beim **ersten Start** liest das Dashboard die bereinigte CSV chunkweise ein (500k Zeilen/Chunk) und baut einen Parquet-Cache auf (`Dashboard/.cache/`). Dauer: 1–3 Minuten.
Alle **weiteren Starts** laden den Cache direkt (wenige Sekunden).

```
Dashboard öffnen: http://127.0.0.1:8050
```

### Schritt 6 – Dashboard bedienen

**Globale Filter (wirken auf alle Tabs):**

| Steuerelement | Funktion |
|---|---|
| Zeitraum-Slider | Jahresbereich einschränken (z. B. 2015–2023) |
| Hafen-Dropdown | Einen oder mehrere deutsche Häfen auswählen (Standard: alle) |
| Metrik | Tonnage (t), TEU (Container) oder Ladeeinheiten |

**Tabs im Dashboard:**

| Tab | Inhalt |
|---|---|
| **Übersicht** | Jährliche Gesamtentwicklung, Saisonalität, Verkehrsbeziehungen, Top-Güterklassen (NST2007) |
| **Zeitreihe** | Monatlicher Verlauf mit gleitenden Durchschnitten (3M, 12M), Heatmap Jahr × Monat, YoY-Wachstum |
| **ML-Prognose** | Zeitreihenprognose mit vier Modellen; vollständige Methodik- und Statistikausgabe (s. u.) |
| **Statistik & Methodik** | Statistische Testbatterie, Kreuzvalidierung, CRISP-DM-Dokumentation (s. u.) |
| **Regionen & Länder** | Makroregionen, Top-20-Länder, Sankey-Diagramm der Handelsströme |
| **Häfen & Schiffe** | Hafenranking, Schiffstypen, Flaggenstaaten, Hafenentwicklung Top-5 |

---

## Projektstruktur

```
Anwendungsprojekt-Seeschiffahrt/
├── Dataset bereinigung/
│   ├── Datasets/                        # Rohdaten (CSV, Destatis MRTM) – nicht im Repo
│   ├── Datasets_cleaned/                # Bereinigte Ausgabedaten – nicht im Repo
│   ├── logs/                            # Prozesslogs und JSON-Reports – nicht im Repo
│   └── clean_datasets.py               # Datenbereinigungs-Pipeline
├── Dashboard/
│   ├── dashboard.py                     # Haupt-Dashboard (~1 200 Zeilen)
│   ├── requirements.txt                 # Python-Abhängigkeiten (9 Pakete)
│   └── .cache/                          # Parquet-Cache (automatisch) – nicht im Repo
└── README.md
```

---

## Datenbereinigungs-Pipeline – Details (`clean_datasets.py`)

### Ausgangssituation

Quelle: **Statistisches Bundesamt (Destatis), MRTM-Seeverkehrsstatistik** (Open Data).

| Eigenschaft | Wert |
|---|---|
| Kodierung | UTF-8 mit BOM |
| Trennzeichen | Semikolon (`;`) |
| Dezimalzeichen | Komma (`,`) |

Typische Qualitätsprobleme in den Rohdaten:

- Fehlende Werte in Pflicht- und optionalen Feldern
- Inkonsistente Spaltennamen je nach Datei-Version (z. B. `Guetergewicht` statt `Tonnen`)
- Führende/nachgelagerte Leerzeichen in Textspalten
- Ungültige oder außerhalb des erlaubten Bereichs liegende Codewerte
- Doppelte Zeilen innerhalb und zwischen Dateien
- Fehlende NUTS3-Codes, UNLOCODE und HafenID-Zuordnungen
- Fehlende Makroregion-Zuordnung für Nicht-EU/EEA-Häfen

### Verarbeitungsschritte (CRISP-DM – Data Preparation)

#### Schritt 1 – Einlesen
Robustes Einlesen mit C-Engine, Fallback auf Python-Engine bei fehlerhaften Zeilen. Spaltennamen werden vereinheitlicht (Alias-Mapping, z. B. `Guetergewicht` → `Tonnen`). Alle Spalten werden initial als `str` eingelesen, um Parsing-Fehler durch das Dezimalkomma zu vermeiden.

#### Schritt 2 – Pflichtfelder prüfen
Zeilen mit fehlenden Kernfeldern werden entfernt:
`EVAS`, `Referenzzeitraum_Jahr`, `Referenzzeitraum_Monat`, `Einladeregion_ISO`, `Ausladeregion_ISO`, `Verkehrsbeziehung`, `Schiffsart`, `Flagge`, `NST2007`, `Tonnen`

#### Schritt 3 – Whitespace-Bereinigung
Führende und nachgelagerte Leerzeichen werden aus allen Textspalten entfernt.

#### Schritt 4 – Regions-Auffüllung
Fehlende Werte in NUTS3-Codes, UNLOCODE und HafenID werden über dateiinterne und dateiübergreifende Lookup-Tabellen aufgefüllt. Lokale Mappings haben Vorrang. Globale Lookups werden einmalig vor der Parallelverarbeitung aus allen Dateien gebaut.

#### Schritt 5 – Makroregion-Ableitung
Aus den ISO-2-Codes werden `Einladeregion_Makroregion` und `Ausladeregion_Makroregion` abgeleitet (EU/EEA, Nordafrika, Naher Osten, Asien, Amerika, Ozeanien u. a.).

#### Schritt 6 – Numerische Konvertierung
`Tonnen`, `TEU` und `Anzahl_Ladungstraeger` werden von String mit Dezimalkomma (z. B. `"342,0"`) in Float konvertiert (`str.replace(",", ".")` vor `pd.to_numeric`). Nicht konvertierbare Werte werden als `NaN` markiert und protokolliert.

#### Schritt 7 – Bereichsvalidierung

| Spalte | Erlaubter Bereich |
|---|---|
| `Referenzzeitraum_Monat` | 1 – 12 |
| `Referenzzeitraum_Jahr` | 2000 – 2030 |
| `Verkehrsbeziehung` | 1 – 4 |

#### Schritt 8 – ISO-Code-Validierung
`Einladeregion_ISO`, `Ausladeregion_ISO` und `Flagge` werden auf das Format `[A-Z]{2}` geprüft.

#### Schritt 9 – Duplikaterkennung
Exakt doppelte Zeilen werden innerhalb jeder Datei und nach dem Zusammenführen dateiübergreifend entfernt.

#### Schritt 10 – Ausreißer-Markierung (IQR-Methode)
Für `Tonnen`, `TEU` und `Anzahl_Ladungstraeger` werden Ausreißer mit Faktor **k = 3,0 × IQR** markiert (nicht entfernt). Neue boolesche Spalten: `Tonnen_outlier`, `TEU_outlier`, `Anzahl_Ladungstraeger_outlier`.

Grenzwerte: `Untere Schranke = Q1 − 3,0 · IQR`, `Obere Schranke = Q3 + 3,0 · IQR`

### Ausgabe-Format

Gleiche Kodierung wie Eingabe (UTF-8 mit BOM, Semikolon, Dezimalkomma). Zusätzliche Spalten:

| Neue Spalte | Typ | Inhalt |
|---|---|---|
| `Quelldatei` | str | Name der Ursprungsdatei |
| `Einladeregion_Makroregion` | str | Abgeleitete Weltregion (Einladung) |
| `Ausladeregion_Makroregion` | str | Abgeleitete Weltregion (Ausladung) |
| `Tonnen_outlier` | bool | `True` wenn Ausreißer (IQR-Methode, k=3,0) |
| `TEU_outlier` | bool | `True` wenn Ausreißer |
| `Anzahl_Ladungstraeger_outlier` | bool | `True` wenn Ausreißer |

### Ausgabe-Logs

| Datei | Inhalt |
|---|---|
| `cleaning_YYYYMMDD_HHMMSS.log` | Detailliertes Prozessprotokoll (Info, Warnungen, Fehler) |
| `cleaning_report_YYYYMMDD_HHMMSS.json` | Maschinenlesbarer Report: Shape, entfernte Zeilen, ergänzte Regions-Werte, Ausreißer-Statistiken (Q1, Q3, IQR-Grenzen) je Spalte |

---

## Dashboard – Technische Details (`dashboard.py`)

### Architektur

Das Dashboard ist für große Dateien (bis ~10 Mio. Zeilen / 3,5 GB CSV) ausgelegt:

1. **Chunk-Verarbeitung**: CSV wird in 500k-Zeilen-Chunks eingelesen – nie vollständig im RAM
2. **Aggregation beim Einlesen**: Alle 10 Aggregationstabellen werden direkt je Chunk akkumuliert; Rohdaten werden verworfen
3. **Parquet-Cache**: Aggregationen werden als 10 Parquet-Dateien gecacht (~MB statt GB); Cache-Invalidierung erfolgt automatisch via Datei-Timestamp
4. **Callbacks auf Agg-Daten**: Alle Dashboard-Interaktionen arbeiten ausschließlich auf den kleinen Aggregationstabellen

### Aggregationstabellen

| Name | Dimensionen | Zweck |
|---|---|---|
| `ts` | Jahr, Monat, Hafen | Zeitreihen, KPIs, Übersicht |
| `schiffsart` | Jahr, Hafen, Schiffsart | Schiffstyp-Analyse |
| `flagge` | Jahr, Hafen, Flagge | Flaggenstaaten |
| `nst` | Jahr, Hafen, NST2007-Klasse | Güterklassen |
| `vk` | Jahr, Hafen, Verkehrsbeziehung | Verkehrsrichtungen |
| `einlade` | Jahr, Hafen, Einladeregion-Makroregion | Herkunftsregionen |
| `auslade` | Jahr, Hafen, Ausladeregion-Makroregion | Zielregionen |
| `einlade_iso` | Jahr, Hafen, Einladeregion-ISO | Herkunftsländer |
| `auslade_iso` | Jahr, Hafen, Ausladeregion-ISO | Zielländer |
| `sankey` | Jahr, Hafen, Ein- + Ausladeregion | Handelsströme |

### Feature-Engineering (ML-Prognose)

Aus der monatlichen Zeitreihe werden **d = 16 Features** abgeleitet:

| Feature-Gruppe | Features | Zweck |
|---|---|---|
| Trendterme | `t`, `t²` | Linearer und quadratischer Trend |
| Saisonalität (Monat) | `month_sin`, `month_cos` | Zyklische Monatskodierung |
| Saisonalität (Quartal) | `q_sin`, `q_cos` | Zyklische Quartalskodierung |
| Lag-Features | `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12` | Autoregressive Komponente |
| Gleitende Mittelwerte | `roll_3`, `roll_6`, `roll_12` | Geglättete Vergangenheitswerte |

### Modelle und Komplexität

| Modell | Formel | Training | Prognose | Seed |
|---|---|---|---|---|
| Ridge-Regression | β̂ = (XᵀX + αI)⁻¹Xᵀy, α = 10 | O(n·d²) | O(d) | – |
| Polynomiale Ridge (Grad 2) | Φ(X) → Ridge, α = 1 | O(n·d⁴) | O(d²) | – |
| Random Forest | f̂(x) = (1/B)·ΣTᵦ(x), B = 200 | O(B·n·d·log n) | O(B·log n) | 42 |
| Gradient Boosting | Fₘ = Fₘ₋₁ + γhₘ, M = 150, η = 0,05 | O(M·n·d·log n) | O(M·log n) | 42 |

**Konfidenzband:** ±1,96 · RMSE · √h (propagierter Prognosefehler über Horizont h).

### Tab: Statistik & Methodik

Dieser Tab enthält die vollständige statistische Analyse der gewählten Zeitreihe:

#### Deskriptive Statistiken
n, Mittelwert μ, Standardabweichung σ, Variationskoeffizient σ/μ, Minimum, Q1, Median, Q3, Maximum, IQR, Schiefe γ₁ (Fisher), Exzess-Kurtosis γ₂.

#### Stationaritätstest (ADF)
Augmented Dickey-Fuller Test. H₀: Einheitswurzel vorhanden (nicht stationär). Lag-Auswahl via AIC-Kriterium. Signifikanzniveau α = 5 %. Ausgabe: Teststatistik, p-Wert, kritische Werte (1 %, 5 %, 10 %), Befund.

#### Normalverteilungstests
- **Shapiro-Wilk** (n ≤ 5 000): prüft Abweichung von Normalverteilung, H₀: normalverteilt
- **Kolmogorov-Smirnov** (n > 5 000): vergleicht empirische mit angepasster Normalverteilung
- **Jarque-Bera**: testet gemeinsam Schiefe = 0 und Exzess-Kurtosis = 0; Teststatistik JB = (n/6)·(γ₁² + γ₂²/4)

Alle Tests mit α = 5 %, p-Wert und explizitem H₀/H₁-Befund.

#### Verteilungsplot
Histogramm der monatlichen Werte (Dichteschätzung) mit übergelagerter Normalverteilungskurve N(μ, σ²) als Referenz.

#### ACF und PACF
Autokorrelationsfunktion (ACF) und Partielle ACF (PACF) für bis zu 24 Lags. Konfidenzband: ±1,96/√n (asymptotisch, α = 5 %). Signifikante Lags indizieren autoregressive Struktur.

#### Walk-forward Cross-Validation
Alle 4 Modelle werden mit 5 Folds im Walk-forward-Verfahren verglichen (keine Datenleckage: Testfold liegt stets nach dem Trainingsfenster). Ausgabe: MAE ± σ, RMSE, R² ± σ je Modell und Fold.

#### CRISP-DM-Dokumentation
Alle 6 Phasen (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment) konkret für dieses Projekt ausgefüllt.

### Tab: ML-Prognose

Neben der Prognosekurve mit 95%-Konfidenzband enthält dieser Tab:

- **Modellgleichung und Verlustfunktion** in formaler Notation
- **Hyperparameter-Tabelle** mit Wert und Bedeutung je Parameter
- **Big-O-Komplexität** für Training und Prognose
- **Modellvoraussetzungen** (Verteilungsannahmen, Linearität)
- **Bias-Varianz-Analyse**: MAE, RMSE, R² getrennt für Train- und Test-Split mit Generalisierungslücke in %
- **Durbin-Watson-Statistik** auf Residuen (d ≈ 2: keine Autokorrelation; d < 1,5: positive; d > 2,5: negative)
- **Residuennormalitätstests** (Shapiro-Wilk + Jarque-Bera) mit H₀-Entscheidung bei α = 5 %
- **Feature-Importance-Diagramm** (Random Forest und Gradient Boosting)
- **Residuendiagramm** (zeitlich + Histogramm) mit beschrifteten Achsen

### Reproduzierbarkeit

| Maßnahme | Umsetzung |
|---|---|
| Zufallsseed | `random_state=42` in allen Ensemble-Modellen |
| Abhängigkeiten | `requirements.txt` mit Mindestversionen |
| Cache-Transparenz | Cache-Metadaten (CSV-Timestamp, Zeilenzahl) in `meta.json` |
| One-Command-Start | `python dashboard.py` aus dem `Dashboard/`-Ordner |

### Cache invalidieren

Das Dashboard erkennt automatisch einen geänderten CSV-Timestamp und baut den Cache neu. Manuelles Löschen:

```bash
# Windows
del /q Dashboard\.cache\*

# Linux / macOS
rm Dashboard/.cache/*
```
