# Anwendungsprojekt-Seeschiffahrt

Analyse und Prognose des Güterverkehrs in der Schiffahrt der Seefahrt.

---

## Datenbereinigungs-Pipeline (`clean_datasets.py`)

### Ausgangssituation

Die Rohdaten stammen aus dem **Destatis Open Data**-Angebot (Statistisches Bundesamt) zur Seeverkehrsstatistik (MRTM-Datensätze). Die CSV-Dateien liegen im Ordner `Datasets/` und haben folgendes Format:

| Eigenschaft       | Wert                  |
|-------------------|-----------------------|
| Kodierung         | UTF-8 mit BOM         |
| Trennzeichen      | Semikolon (`;`)       |
| Dezimalzeichen    | Komma (`,`)           |

Die Rohdaten können folgende Qualitätsprobleme aufweisen:

- Fehlende Werte in Pflicht- und optionalen Feldern
- Inkonsistente Spaltennamen je nach Datei-Version (z. B. `Guetergewicht` statt `Tonnen`)
- Führende/nachgelagerte Leerzeichen in Textspalten
- Ungültige oder außerhalb des erlaubten Bereichs liegende Codewerte
- Doppelte Zeilen innerhalb und zwischen Dateien
- Fehlende Regions-Zuordnungen (NUTS3, UNLOCODE) für Ein- und Ausladeregionen
- Fehlende Makroregion-Zuordnung für Häfen außerhalb der EU/EEA

---

### Was die Pipeline tut (CRISP-DM – Data Preparation)

Die Pipeline verarbeitet alle CSV-Dateien aus `Datasets/` parallel (bis zu 3 Prozesse gleichzeitig) und durchläuft für jede Datei folgende Schritte:

#### 1. Einlesen
Robustes Einlesen der CSV-Dateien (C-Engine mit Fallback auf Python-Engine bei fehlerhaften Zeilen). Spaltennamen werden vereinheitlicht (Alias-Mapping, z. B. `Guetergewicht` → `Tonnen`).

#### 2. Zeilen mit fehlenden Kernfeldern entfernen
Zeilen, bei denen mindestens eines der folgenden Pflichtfelder fehlt, werden entfernt:
`EVAS`, `Referenzzeitraum_Jahr`, `Referenzzeitraum_Monat`, `Einladeregion_ISO`, `Ausladeregion_ISO`, `Verkehrsbeziehung`, `Schiffsart`, `Flagge`, `NST2007`, `Tonnen`

#### 3. Pflichtfeld-Validierung
Prüfung, ob alle erwarteten Pflichtfelder im Schema vorhanden sind. Fehlende Felder werden im Log als Warnung ausgewiesen.

#### 4. Whitespace-Bereinigung
Führende und nachgelagerte Leerzeichen werden aus allen Textspalten entfernt.

#### 5a. Ausladeregion-Auffüllung
Fehlende Werte in `Ausladeregion_NUTS3`, `Ausladeregion_NUTS3_Label`, `Ausladeregion_UNLOCODE` und `Ausladeregion_HafenID` werden gegenseitig aufgefüllt. Anker-Spalte ist `Ausladeregion_HafenID`. Dabei werden lokale (dateibezogene) und globale (dateiübergreifende) Lookup-Tabellen kombiniert; lokale Mappings haben Vorrang.

#### 5b. Einladeregion-Auffüllung
Analog zur Ausladeregion für `Einladeregion_NUTS3`, `Einladeregion_NUTS3_Label`, `Einladeregion_UNLOCODE` und `Einladeregion_HafenID` (Anker: `Einladeregion_HafenID`).

#### 5c. Makroregion-Ableitung
Aus den ISO-2-Codes der Ein- und Ausladeregion werden die Spalten `Ausladeregion_Makroregion` und `Einladeregion_Makroregion` abgeleitet. Das Mapping deckt alle relevanten Weltregionen ab (EU/EEA, Nordafrika, Naher Osten, Asien, Amerika, Ozeanien u. a.). Unbekannte ISO-Codes bleiben `NaN` und werden im Report ausgewiesen.

#### 6. Numerische Konvertierung
Die Spalten `Tonnen`, `TEU` und `Anzahl_Ladungstraeger` werden von String (Dezimalkomma) in Float konvertiert. Nicht konvertierbare Werte werden auf `NaN` gesetzt und im Report gezählt.

#### 7. Fehlende-Werte-Analyse
Verbleibende fehlende Werte werden protokolliert. Dabei wird unterschieden zwischen:
- **Strukturell erwarteten** Lücken (z. B. fehlende NUTS3-Codes für Nicht-EU/EEA-Häfen)
- **Unerwarteten** Lücken (z. B. EU/EEA-Hafen ohne NUTS3-Code → Warnung)
- **Optionalen** Container-Feldern (nur relevant bei Containertransport)

#### 8. Bereichsvalidierung
Codierte Felder werden auf erlaubte Wertebereiche geprüft:

| Spalte                    | Erlaubter Bereich |
|---------------------------|-------------------|
| `Referenzzeitraum_Monat`  | 1 – 12            |
| `Referenzzeitraum_Jahr`   | 2000 – 2030       |
| `Verkehrsbeziehung`       | 1 – 4             |

#### 9. ISO-Code-Validierung
`Einladeregion_ISO`, `Ausladeregion_ISO` und `Flagge` werden auf das Format `[A-Z]{2}` geprüft.

#### 10. Duplikaterkennung und -entfernung
Exakt doppelte Zeilen werden innerhalb jeder Datei entfernt. Nach dem Zusammenführen aller Dateien werden dateiübergreifende Duplikate (ohne Berücksichtigung der `Quelldatei`-Spalte) nochmals entfernt.

#### 11. Ausreißer-Markierung (IQR-Methode)
Für `Tonnen`, `TEU` und `Anzahl_Ladungstraeger` werden Ausreißer mit dem Faktor 3,0 × IQR markiert (nicht entfernt). Pro Spalte wird eine boolesche Spalte `*_outlier` hinzugefügt.

---

### Dateiübergreifende Lookup-Tabellen

Vor der Parallelverarbeitung werden alle CSV-Dateien einmalig eingelesen, um **globale Lookup-Tabellen** für die Regions-Auffüllung aufzubauen. So können Regions-Zuordnungen, die nur in einer Datei vorkommen, auch auf andere Dateien angewendet werden.

---

### Ergebnisse

Am Ende der Pipeline entstehen folgende Ausgaben:

#### `Datasets_cleaned/`
Eine bereinigte, kombinierte CSV-Datei mit dem Namen:
```
seeverkehr_bereinigt_YYYYMMDD_HHMMSS.csv
```
- Gleiches Format wie die Eingabe (UTF-8 mit BOM, Semikolon, Dezimalkomma)
- Enthält eine zusätzliche Spalte `Quelldatei` (Herkunft jeder Zeile)
- Enthält neue Spalten `Ausladeregion_Makroregion` und `Einladeregion_Makroregion`
- Enthält Ausreißer-Markierungsspalten (`Tonnen_outlier`, `TEU_outlier`, `Anzahl_Ladungstraeger_outlier`)

#### `logs/`
Zwei Dateien pro Lauf:

| Datei                                        | Inhalt                                                                 |
|----------------------------------------------|------------------------------------------------------------------------|
| `cleaning_YYYYMMDD_HHMMSS.log`               | Detailliertes Prozessprotokoll (Info, Warnungen, Fehler)              |
| `cleaning_report_YYYYMMDD_HHMMSS.json`       | Maschinenlesbarer Report mit Kennzahlen je Datei und Gesamtstatistik  |

Der JSON-Report enthält je Datei u. a.:
- Ursprüngliche und finale Shape (Zeilen × Spalten)
- Anzahl entfernter Zeilen (fehlende Kernfelder, Duplikate)
- Anzahl ergänzter Regions-Werte
- Konvertierungsfehler, Bereichsverletzungen, ISO-Fehler
- Ausreißer-Statistiken (Q1, Q3, Grenzen) pro numerischer Spalte
- Makroregion-Mapping-Statistik

---

### Projektstruktur

```
Anwendungsprojekt-Seeschiffahrt/
├── Datasets/                  # Rohdaten (CSV, Destatis MRTM)
├── Datasets_cleaned/          # Bereinigte Ausgabedaten
├── logs/                      # Prozesslogs und JSON-Reports
└── clean_datasets.py          # Datenbereinigungs-Pipeline
```
