# brink-refinement

A weekly, unattended refinement loop for the **Brink Model**: forecasts of human crises (new or escalating armed conflict, mass atrocity, regime collapse, economic breakdown, large disasters) for every country, month by month.

**Status: under construction. Nothing here runs on a schedule yet.**

## What this repository holds

- The model code and the recipes that download public data. No data is stored here; every run downloads it fresh from the original sources.
- The locked model settings, and the exam fingerprint list (SHA-256 checksums only).
- The idea queue. Each idea is a small data-only file, written and committed before it is tested.
- The weekly run instructions (GitHub Actions).
- Results and the exam record live on the `ledger` branch. They are append-only and never edited, so the track record can be checked by anyone.

## What is not here

- No data tables.
- No exam outcomes for 2015-2025. They are rebuilt only inside an exam check, and checked against the fingerprints before scoring.
- No keys, passwords or personal details.

## How a week runs

1. Download the public data and rebuild the model's signals.
2. List the model's misses and false alarms on the practice years.
3. Take the next ideas from the queue and test each on practice years 2000-2014 (refit, then a 500-draw bootstrap). The test runs in an isolated step with no network, no repository history and no credentials.
4. Record every idea as kept or dropped.
5. A kept idea opens a pull request for the review council. Nothing changes the live model automatically.

Exam checks (2015-2025) run only when started by hand, at most 3 per calendar quarter. Each check is reserved in the ledger before any exam data is built, and a reserved check is never refunded.

## Data sources

All data is downloaded at run time from the sources below. Please cite them. Licence notes are as recorded on 21 October 2026; each source's own current terms apply.

| Source | Use | Licence or terms (as recorded) |
|---|---|---|
| UCDP: GED, UCDP/PRIO Armed Conflict, One-sided, Non-state, Battle-related deaths, Candidate events | Violence signals and outcomes | CC BY 4.0 |
| V-Dem | Regime, civil liberties, polarisation | CC BY-SA 4.0 |
| World Bank World Development Indicators (current and archived editions) | Economy, health, population | CC BY 4.0 / World Bank terms |
| UNHCR Refugee Data Finder | Refugee and displacement signals | UNHCR terms |
| IMF consumer prices and effective exchange rates | Inflation and currency outcomes | IMF terms |
| Bank of Canada / Bank of England sovereign default database | Default outcomes | Bank of Canada terms |
| Center for Systemic Peace: PITF problem set, coups | Outcomes before 2019/2022 | Research use with citation; not redistributed here |
| Mass Mobilization Project | Protest proxy (1990-March 2020) | CC0 1.0 |
| NOAA NCEI hazards | Earthquakes, tsunamis, volcanoes | US public domain |
| Dartmouth Flood Observatory | Floods to 2021 | Research use with citation |
| Early Warning Project statistical risk assessments | Benchmark only | Not redistributed here |
| WFP food prices (via HDX) | Food price signal | CC BY-IGO |
| CRU TS 4.09 country averages | Rain and heat | Open Government Licence v3 |
| GeoDataSource country borders | Neighbour signal | MIT |

ACLED and EM-DAT are deliberately not used: their terms bar this kind of benchmarking or redistribution.
