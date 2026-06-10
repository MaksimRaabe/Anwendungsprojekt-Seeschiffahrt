# Anwendungsprojekt-Seeschiffahrt

Analyse und Prognose des Güterverkehrs in der deutschen Seeschifffahrt auf Basis amtlicher Destatis-Statistiken.

---

## Überblick

Das Projekt besteht aus zwei Komponenten:

| Komponente | Ordner | Zweck |
|---|---|---|
| Datenbereinigungs-Pipeline | `Dataset bereinigung/` | Rohdaten bereinigen und zusammenführen |
| Analyse-Dashboard | `Dashboard/` | Interaktive Visualisierung und ML-Prognose |

---

## Schritt-für-Schritt-Tutorial

### Voraussetzungen

- Python 3.10 oder neuer
- Die Rohdaten-CSV-Dateien von Destatis (MRTM-Datensätze)

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

**Dashboard:**

```bash
pip install -r Dashboard/requirements.txt
```

### Schritt 3 – Rohdaten ablegen

Lege alle CSV-Rohdateien von Destatis in den Ordner:

```
Dataset bereinigung/Datasets/
```

Die Dateien müssen das Destatis-MRTM-Format aufweisen (UTF-8 mit BOM, Semikolon-getrennt, Dezimalkomma).
Mehrere Jahresdateien können gleichzeitig im Ordner liegen — die Pipeline verarbeitet alle.

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

Beim ersten Start liest das Dashboard die bereinigte CSV chunkweise ein und baut einen
Parquet-Cache auf (`Dashboard/.cache/`). Das dauert je nach Datenmenge 1–3 Minuten.
Alle weiteren Starts laden den Cache direkt (wenige Sekunden).

```
Dashboard öffnen: http://127.0.0.1:8050
```

### Schritt 6 – Dashboard bedienen

| Steuerelement | Funktion |
|---|---|
| Zeitraum-Slider | Jahrbereich einschränken (z. B. 2015–2023) |
| Hafen-Dropdown | Einen oder mehrere deutsche Häfen auswählen (Standard: alle) |
| Metrik | Tonnage (t), TEU (Container) oder Ladeeinheiten |

**Tabs im Dashboard:**

- **Übersicht** – Jährliche Gesamtentwicklung, Saisonalität, Verkehrsbeziehungen, Top-Güterklassen
- **Zeitreihe** – Monatlicher Verlauf je Hafen mit Trendlinie
- **ML-Prognose** – Vorhersage mit Random Forest, Gradient Boosting, linearer oder polynomialer Regression; wählbarer Prognosehorizont in Jahren
- **Regionen & Länder** – Herkunfts- und Zielregionen (Makroregion und ISO-Länder); Sankey-Diagramm der Handelsströme
- **Häfen & Schiffe** – Hafenvergleich, Schiffstypen, Flaggenverteilung

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
│   ├── dashboard.py                     # Haupt-Dashboard-Applikation
│   ├── requirements.txt                 # Python-Abhängigkeiten
│   └── .cache/                          # Automatisch erstellter Parquet-Cache – nicht im Repo
└── README.md
```

---

## Datenbereinigungs-Pipeline – Details

### Ausgangssituation

Die Rohdaten stammen aus dem **Destatis Open Data**-Angebot (Statistisches Bundesamt) zur Seeverkehrsstatistik (MRTM-Datensätze).

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
- Fehlende Regions-Zuordnungen (NUTS3, UNLOCODE)
- Fehlende Makroregion-Zuordnung für Nicht-EU/EEA-Häfen

### Verarbeitungsschritte (CRISP-DM – Data Preparation)

#### 1. Einlesen
Robustes Einlesen mit C-Engine, Fallback auf Python-Engine bei fehlerhaften Zeilen. Spaltennamen werden vereinheitlicht (Alias-Mapping, z. B. `Guetergewicht` → `Tonnen`).

#### 2. Pflichtfelder prüfen
Zeilen mit fehlenden Kernfeldern werden entfernt:
`EVAS`, `Referenzzeitraum_Jahr`, `Referenzzeitraum_Monat`, `Einladeregion_ISO`, `Ausladeregion_ISO`, `Verkehrsbeziehung`, `Schiffsart`, `Flagge`, `NST2007`, `Tonnen`

#### 3. Whitespace-Bereinigung
Führende und nachgelagerte Leerzeichen werden aus allen Textspalten entfernt.

#### 4. Regions-Auffüllung
Fehlende Werte in NUTS3-Codes, UNLOCODE und HafenID werden über dateiinterne und dateiübergreifende Lookup-Tabellen aufgefüllt. Lokale Mappings haben Vorrang.

#### 5. Makroregion-Ableitung
Aus den ISO-2-Codes werden `Einladeregion_Makroregion` und `Ausladeregion_Makroregion` abgeleitet (EU/EEA, Nordafrika, Naher Osten, Asien, Amerika, Ozeanien u. a.).

#### 6. Numerische Konvertierung
`Tonnen`, `TEU` und `Anzahl_Ladungstraeger` werden von String (Dezimalkomma) in Float konvertiert. Nicht konvertierbare Werte werden als `NaN` markiert und protokolliert.

#### 7. Bereichsvalidierung

| Spalte | Erlaubter Bereich |
|---|---|
| `Referenzzeitraum_Monat` | 1 – 12 |
| `Referenzzeitraum_Jahr` | 2000 – 2030 |
| `Verkehrsbeziehung` | 1 – 4 |

#### 8. ISO-Code-Validierung
`Einladeregion_ISO`, `Ausladeregion_ISO` und `Flagge` werden auf das Format `[A-Z]{2}` geprüft.

#### 9. Duplikaterkennung
Exakt doppelte Zeilen werden innerhalb jeder Datei und nach dem Zusammenführen dateiübergreifend entfernt.

#### 10. Ausreißer-Markierung (IQR-Methode)
Für `Tonnen`, `TEU` und `Anzahl_Ladungstraeger` werden Ausreißer mit Faktor 3,0 × IQR markiert (nicht entfernt). Neue boolesche Spalten: `Tonnen_outlier`, `TEU_outlier`, `Anzahl_Ladungstraeger_outlier`.

### Ausgabe-Format

Die bereinigte CSV hat dasselbe Format wie die Eingabe (UTF-8 mit BOM, Semikolon, Dezimalkomma) und enthält zusätzliche Spalten:

| Neue Spalte | Inhalt |
|---|---|
| `Quelldatei` | Name der Ursprungsdatei |
| `Einladeregion_Makroregion` | Abgeleitete Weltregion der Einladung |
| `Ausladeregion_Makroregion` | Abgeleitete Weltregion der Ausladung |
| `Tonnen_outlier` | `True` wenn Ausreißer |
| `TEU_outlier` | `True` wenn Ausreißer |
| `Anzahl_Ladungstraeger_outlier` | `True` wenn Ausreißer |

---

## Dashboard – Technische Details

### Architektur

Das Dashboard ist für große Dateien (bis ~10 Mio. Zeilen / 3,5 GB CSV) ausgelegt:

1. **Chunk-Verarbeitung**: CSV wird in 500k-Zeilen-Chunks eingelesen – nie vollständig im RAM
2. **Aggregation beim Einlesen**: Alle Kennzahlen werden direkt je Chunk akkumuliert; Rohdaten werden nicht gespeichert
3. **Parquet-Cache**: Aggregationen werden als kompakte Parquet-Dateien gecacht (~MB statt GB); Folgestarts laden nur den Cache
4. **Callbacks auf Agg-Daten**: Alle Dashboard-Interaktionen arbeiten ausschließlich auf den kleinen Aggregationstabellen

### Bekannte Fixes (Stand Juni 2026)

| Problem | Ursache | Lösung |
|---|---|---|
| Tonnage/TEU zeigt 0 | CSV-Zahlen mit Dezimalkomma (`342,0`) wurden von `pd.to_numeric` nicht erkannt | Komma wird vor Konvertierung durch Punkt ersetzt |
| Jahresfilter TypeError | `Referenzzeitraum_Jahr` liegt als String im Cache; Vergleich mit `int` schlug fehl | `pd.to_numeric` beim Filtern in `_agg()` |
| ML-Prognose-Tab 500-Fehler | `add_vline` mit `annotation_position` + Timestamp-x erzeugt internen Plotly-Fehler | Ersetzt durch `add_shape` + `add_annotation` |

### Cache invalidieren

Falls die bereinigte CSV ausgetauscht wird, erkennt das Dashboard automatisch die Änderung (anhand des Datei-Timestamps) und baut den Cache neu. Manuelles Löschen ist möglich:

```bash
# Windows
del /q Dashboard\.cache\*

# Linux / macOS
rm Dashboard/.cache/*
```
