#  ML Churn Predictor

A web app that predicts which users of **any app** are about to churn. Drop in your app's database (a Duolingo-style learning app, a Waze-style navigation app, a SaaS product, a game) and get a churn dashboard in about a minute:

- **Headline numbers:** how many active users, how many are expected to churn in the next N days, how many are high-risk, and how trustworthy the prediction is.
- **Charts:** users by risk level and the biggest churn drivers.
- **At-risk users:** a searchable list showing each user's risk, last activity and main reason, exportable to CSV.
- **User panel:** a per-user explanation and activity chart.

Under the hood it:
1. **Understands the database on its own.** It finds the users table, the activity tables, the timestamp columns (in whatever format they're stored), numeric measures and categories.
2. **Defines churn from your app's usage rhythm**, for example "no activity for 7 days" for a daily-habit app, or uses your own churn label column.
3. **Trains and validates an XGBoost model** on past snapshots of your data, then scores every active user with a calibrated probability.

It uses only open-source tools: **no API keys, no LLM calls and no paid services.**

---

## 📋 Table of Contents

- [Deploy to Vercel (free)](#deploy-to-vercel-free)
- [Run it locally](#run-it-locally)
- [Self-host with live database connections](#self-host-with-live-database-connections)
- [What data it needs](#what-data-it-needs)
- [How it understands a database](#how-it-understands-a-database)
- [How churn is defined](#how-churn-is-defined)
- [The model](#the-model)
- [Accuracy](#accuracy)
- [API](#api)
- [Project structure](#project-structure)
- [Testing](#testing)

---

## Deploy to Vercel (free)

The site in `site/` is fully static. The Python pipeline runs **in each visitor's browser** using [Pyodide](https://pyodide.org) (Python compiled to WebAssembly), so:

- **Free hosting** that scales to any number of visitors, with no servers to run or pay for.
- **Privacy:** a visitor's file is never uploaded; it is analysed on their own computer.
- **No upload limits** beyond the visitor's browser memory.

**To deploy:**
1. Push this repository to GitHub.
2. In Vercel: **Add New → Project**, import the repository, and click **Deploy**.

`vercel.json` already tells Vercel to serve `site/` with no build step, and `.vercelignore` keeps everything else out of the upload. Or, from a terminal:

```bash
npx vercel deploy --prod
```

Any static host works the same way (Netlify, Cloudflare Pages, GitHub Pages, S3): publish the `site/` folder.

**Good to know:**
- **First load:** the first analysis on a device downloads the engine (about 30 MB, from jsDelivr). Later visits load it from the browser cache.
- **Speed:** a few thousand users take about 10–60 seconds, depending on the data and the device.
- **Large files:** above 50,000 users, a random sample of 50,000 is analysed, and the dashboard says so.
- **Accepted files:** SQLite, CSV, TSV, Excel, Parquet, or a `.zip` of them. A browser can't connect to a live database; for that, self-host (below).
- **Updating:** after changing anything in `backend/`, run `python -m web.build` and commit `site/`. A test fails if the bundled Python code is out of date.

---

## Run it locally

```bash
pip install -r requirements.txt
uvicorn backend.app:app --port 8000      # then open http://localhost:8000
```

When the page is served by this backend, it analyses on the server instead of in the browser. That makes it faster for big files and adds live database connections. To try the pure-browser version locally instead:

```bash
python -m http.server 8000 --directory site
```

---

## Self-host with live database connections

The same UI, backed by the FastAPI server, can connect straight to a database:

```bash
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor           # http://localhost:8000
```

This works on Render, Fly.io, Railway or any machine that runs Docker. On the start screen, choose **Connect with a database URL instead**:

| Database | URL |
|---|---|
| PostgreSQL | `postgresql://user:password@host:5432/dbname` |
| MySQL / MariaDB | `mysql+pymysql://user:password@host:3306/dbname` |
| SQL Server | `mssql+pyodbc://user:password@dsn` (add `pyodbc`) |
| Oracle | `oracle+oracledb://user:password@host:1521/?service_name=svc` (add `oracledb`) |
| SQLite on the server | `sqlite:///path/to/app.db` |

The database is only ever read, never modified.

---

## What data it needs

**Minimum requirement:** at least one table with a **user identifier** and a **timestamp** per row, such as sessions, events, lessons, trips or orders. Everything else is optional and improves accuracy:
- a users table (signup date, plan, country…)
- more activity tables
- numeric measures (duration, xp, distance, amount…)
- categories (event type, feature, lesson type…)
- a churn label

If something is detected wrongly, click **Adjust** on the "Here's what we found" screen to pick the right users table, activity tables, date columns or churn window.

---

## How it understands a database

Schema detection (`backend/schema/`) profiles every table: column types, uniqueness, null rate, and a sample of values. It then scores candidates:

| Question | How it is answered |
|---|---|
| Which table holds the users? | Table names matched against a vocabulary (user, account, customer, member, learner, driver, player, subscriber, …; fuzzy, plural and camelCase aware) plus how many other tables reference it |
| How does an activity table link to users? | 1. a declared foreign key → 2. a matching column name (`learner_id` → `learners`) verified by value overlap → 3. pure value overlap on id-like columns → 4. one join hop (`events → sessions → accounts`) |
| Which column is the event time? | Values are parsed as native dates, ISO strings, `YYYYMMDD` integers, Unix seconds/ms/µs or Excel serials. The column with the best parse rate and name (`occurred_at` > `created_at` > `updated_at`) wins. |
| What are the measures and categories? | Numeric non-key columns become measures. Low-cardinality text columns become categories. |
| Static user attributes? | Signup date (for tenure) and safe attributes like plan or country. Personal data (email, phone, names…) is ignored. Columns that describe the *current* state (`status`, `total_*`, `last_*`, `*_count`) are ignored because they would leak the future into training. |
| Is there a churn label? | Columns like `churned`, `is_churned`, `churnLabel`, `status` (`cancelled` / `active`), `is_active`, or a `cancelled_at` date |

The result is a JSON **schema map** shown in the UI. Change anything that looks wrong, and save it with **Save mapping** (stored in `mappings/`) to reuse it next time.

---

## How churn is defined

Most app databases have no churn column, so churn is defined behaviourally:

> **A user has churned when they have no activity for H days.**

**The churn window H is learned from the data.** It is the 95th percentile of the gap between a user's consecutive active days, rounded up to 3, 7, 14, 21, 30, 45, 60 or 90 days. A daily-habit app ends up around 7 days, a weekly app around 14–21, and a monthly tool around 30–60. You can override it in the UI.

**Training data comes from time-travel snapshots.** At several past cutoff dates (one churn window apart), features are computed using only data up to the cutoff. Each user who was still active at the cutoff is then labelled by whether they went silent for the next H days. This mirrors exactly what the model must do in production. A leakage test guarantees features never see past their cutoff.

**Scoring** happens as of the last complete day in the data (a partial final day is detected and skipped). Users already inactive for H days are reported as *churned*; everyone else gets a probability of churning in the next H days.

**If your database has a churn label column**, it is used instead (choose "Use the label column" or "Inactivity-based" in the UI to force either). A label has no timing, so the app finds the date where labelled churners' activity starts to diverge from everyone else's. It then trains on features from *before* that point. The model therefore learns early warning signs, not the churn that already happened.

---

## The model

**Features** (`backend/features/feature_frame.py`) are generated for whatever tables and columns the mapping contains:
- **Engagement:** recency, active days over windows scaled to the churn window, share of days active, recent vs usual activity, gap mean/std/max, current break vs the user's typical break, current and longest streak, recency-weighted activity, trend, day-of-week variety, weekend share
- **Per activity table:** event counts, events per active day, trend, and time-of-day variety
- **Per measure:** mean, std, total, recent mean, recent vs baseline, and recent vs long-term history
- **Per category:** distinct values, entropy, categories stopped recently, and share of the top categories
- **Static:** tenure since signup and user attributes

**Training** (`backend/model/train_and_score.py`):
- XGBoost with early stopping and a small hyper-parameter search
- Validation on the **most recent snapshot**, with training on earlier ones (a true out-of-time test). With explicit labels or only one snapshot, a held-out 25% of users is used instead.
- Probabilities calibrated with Platt scaling, kept between 0.5% and 99%
- The final model is refit on all snapshots before scoring
- If there is too little history to build labelled snapshots, a cadence-based heuristic score is used instead (an exponential survival model on each user's own gap between visits). The run is flagged as unsupervised.

**Reported metrics:** ROC-AUC, PR-AUC, Brier score, precision/recall/F1 at your threshold, lift in the top 10%, and the AUC of a **recency-only baseline**, so you can see what the model adds.

**Explanations** are exact TreeSHAP values from XGBoost's built-in `pred_contribs`, with readable factor names such as "Current break vs user's typical break" or "Avg xp earned (lessons), last 7 days".

---

## Accuracy

`simulator/` generates three apps with different schemas, id types, time formats and usage rhythms. It writes the activity *after* the observed period to a separate `<app>_future.db` that the pipeline never sees. `python -m simulator.evaluate sample_data/<app>.db` scores predictions against what really happened next.

| Database (3,000 users) | Schema quirks | Churn window | ROC-AUC on real future | Recency baseline (validation) | Predicted vs actual churn rate |
|---|---|---|---|---|---|
| `learning_app` (Duolingo-like) | ISO timestamps with time zone, FKs, 2 activity tables | 7 days (auto) | **0.90** | 0.86 | 4.5% vs 4.0% |
| `navigation_app` (Waze-like) | UUID ids, Unix ms/s timestamps, **no foreign keys** | 7 days (auto) | **0.85** | 0.85 | 8.5% vs 8.9% |
| `saas_app` (B2B SaaS) | events reach accounts only **via sessions** | 7 days (auto) | **0.89** | 0.81 | 7.2% vs 5.6% |
| `saas_app` loaded into **PostgreSQL** | native `TIMESTAMPTZ`, real FK constraints | 7 days (auto) | **0.88** | 0.81 | — |
| `app_usage.db` (original demo) | `YYYYMMDD` ints, `churnLabel` column | label column | **0.96** (held-out users) | 0.64 | — |

The previous version scored users with the same model it had trained on, so its numbers were in-sample. All numbers above are out-of-sample.

The in-browser engine runs the identical Python code (XGBoost 2.1 in Pyodide) with a smaller hyper-parameter search, so it can run in a browser. On the original demo it reaches the same AUC (0.96).

---

## API

| Method & path | Purpose |
|---|---|
| `GET /health`, `GET /engine.json` | Health check; `engine.json` tells the web UI to use this server |
| `POST /db/validate` `{source}` | Profile a database and return tables, capabilities and the detected `schemaMap` (`dbPath` is accepted as an alias of `source`) |
| `POST /db/upload` (multipart) | Upload a `.db` / `.csv` / `.parquet` / `.xlsx` / `.zip`; returns a path to analyse |
| `GET /mappings`, `GET /mappings/{name}`, `POST /mappings` | List, load and save schema mappings |
| `POST /analysis/run` | Start a job. Options: `source`, `threshold`, `highRiskThreshold`, `schemaMap`, `churnWindowDays`, `labelSource` (`auto`/`explicit`/`derived`), `maxUsers`, `observationStart`, `observationEnd` |
| `GET /analysis/jobs/{id}` | Status, progress, churn definition and metrics |
| `GET /analysis/{id}/result` | Everything the dashboard shows: overview numbers, all users (risk, level, last active, main reason), model info |
| `GET /analysis/{id}/export.csv` | All predictions as CSV |
| `GET /analysis/{id}/users/{userId}/bundle` | Everything the user panel shows |
| `GET /analysis/{id}/users?top=&threshold=&sort=&includeChurned=` | Risk leaderboard |
| `GET /analysis/{id}/users/{userId}` | One user's probability, tier, status and metrics |
| `GET /analysis/{id}/users/{userId}/explain` | Top risk factors and a summary |
| `GET /analysis/{id}/users/{userId}/timeseries` | Daily activity and measures |
| `GET /analysis/{id}/aggregates` | Group averages and global feature importance |
| `GET /analysis/{id}/model` | Churn definition, validation metrics and the mapping used |

User ids are strings, so integer, UUID and text ids all work.

---

## Project structure

```
site/                        The web app, deployed as-is to Vercel / any static host
  index.html                 Dashboard UI (React, no build step)
  worker.js                  In-browser engine: runs the Python pipeline with Pyodide
  py/churn_backend.zip       The backend/ package for the browser (generated)
  demo/*.json                Precomputed example results (generated)
web/build.py                 Regenerates site/py and site/demo
vercel.json, .vercelignore   Vercel deployment settings
Dockerfile                   Self-hosted server image
backend/
  app.py                     FastAPI routes; also serves site/
  service.py                 JSON results shared by the server and the browser engine
  connectors/source.py       Open any SQLAlchemy URL / SQLite / CSV / Parquet / Excel / zip read-only
  schema/
    introspect.py            Tables, keys, foreign keys, column profiles
    timeparse.py             Detect & normalise any timestamp format
    naming.py                Name normalisation and vocabularies
    detect.py                Heuristic schema detection -> SchemaMap
  data/
    extract.py               Pull mapped columns into canonical frames
    timeseries.py            Per-user daily series for charts
  labels/churn_definition.py Churn window inference, snapshots, label timing
  features/feature_frame.py  Generic, cutoff-aware feature engineering
  model/train_and_score.py   XGBoost training, validation, calibration
  explain/contrib.py         Readable TreeSHAP explanations
  pipeline/run.py            End-to-end analysis orchestration
  jobs/                      Background job runner and store
  schemas/api.py             Pydantic request/response models
simulator/
  generate.py                Three schema-diverse demo apps with hidden future data
  evaluate.py                Score predictions against the real future
tests/                       Detection, time parsing, leakage, accuracy and API tests
mappings/                    Saved schema mappings
legacy/                      The original command-line prototype (see legacy/README.md)
app_usage.db                 Original demo database
```

---

## Testing

```bash
python -m pytest -q
```

The suite (about 2 minutes) generates the demo apps and checks:
- schema detection on every schema, including through a SQLAlchemy URL and a CSV folder
- timestamp parsing for every supported format
- that features never use data after their cutoff
- out-of-sample accuracy against hidden future data, beating the baseline and with calibrated probabilities
- the full API flow, including mapping overrides, uploads and UUID user ids
- that the bundled browser code in `site/` matches `backend/`, and that the demo data is valid

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Built by [Ryan Vandersar](https://github.com/ryanvsar)*
