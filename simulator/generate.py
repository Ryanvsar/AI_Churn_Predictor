"""
Generate realistic, schema-diverse app databases with hidden ground truth.

Each app shares one latent behaviour model:
- users sign up over time, have their own activity rate and time-of-day habits,
- a per-day churn hazard that depends on plan, engagement and account age,
- a "fade" period before churning (some users quit abruptly), random vacations for
  loyal users and rare comebacks after churning.

Activity after the observed period is written to a separate `<name>_future.db` with the same
schema, so predictions can be scored against what really happened next.

Apps:
- learning_app   (Duolingo-like): learners / lessons / achievements, ISO timestamps, declared FKs
- navigation_app (Waze-like):     drivers / trips / reports, UUID ids, unix timestamps, no FKs
- saas_app       (B2B SaaS):      accounts / sessions / events, events joined to accounts via sessions

Usage:  python -m simulator.generate --all [--out sample_data] [--users 3000]
"""
from __future__ import annotations

import argparse
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

START = datetime(2025, 1, 1, tzinfo=timezone.utc)


@dataclass
class AppSpec:
    observed_days: int
    future_days: int
    base_p: Callable  # rng -> daily activity probability
    hazard: float  # base daily churn hazard
    weekday_mult: np.ndarray  # 7 multipliers Mon..Sun
    plans: Dict[str, tuple]  # plan -> (share, hazard multiplier, activity multiplier)


def _simulate_latent(n: int, spec: AppSpec, rng: np.random.Generator) -> List[dict]:
    total = spec.observed_days + spec.future_days
    plan_names = list(spec.plans)
    plan_p = np.array([spec.plans[p][0] for p in plan_names])
    users = []
    for i in range(n):
        plan = plan_names[rng.choice(len(plan_names), p=plan_p / plan_p.sum())]
        _, hz_mult, act_mult = spec.plans[plan]
        # 45% existed before the window (history starts at day 0), the rest sign up during it.
        signup = int(rng.integers(-120, 0)) if rng.random() < 0.45 else int(rng.integers(0, spec.observed_days - 10))
        base_p = float(np.clip(spec.base_p(rng) * act_mult, 0.03, 0.97))
        hz = spec.hazard * hz_mult * (0.45 / base_p) ** 0.8

        days = np.arange(max(signup, 0), total)
        age = days - signup
        daily_h = np.minimum(hz * np.where(age < 14, 3.0, 1.0), 0.5)
        churn_draw = rng.random(len(days)) < daily_h
        churn_day = int(days[np.argmax(churn_draw)]) if churn_draw.any() else 10**6

        fade = 0 if rng.random() < 0.3 else int(rng.integers(5, 30))
        drift = rng.normal(0, 0.25)  # slow engagement drift for everyone
        p = base_p * spec.weekday_mult[(days + START.weekday()) % 7]
        p = p * np.clip(1 + drift * (days - days[0]) / 180.0, 0.4, 1.6)
        if fade:
            in_fade = (days >= churn_day - fade) & (days < churn_day)
            frac = (days - (churn_day - fade)) / fade
            p = np.where(in_fade, p * (1 - 0.8 * frac), p)
        p = np.where(days >= churn_day, 0.004, p)  # rare comebacks
        if churn_day > total and rng.random() < 0.2:  # vacations for loyal users
            v0 = int(rng.integers(days[0], total))
            p = np.where((days >= v0) & (days < v0 + int(rng.integers(4, 13))), p * 0.05, p)
        intensity = np.where((days >= churn_day - fade) & (days < churn_day) & (fade > 0),
                             1 - 0.6 * (days - (churn_day - fade)) / max(fade, 1), 1.0)
        active = rng.random(len(days)) < np.clip(p, 0, 1)
        users.append({
            "idx": i, "plan": plan, "signup_day": signup, "base_p": base_p,
            "churn_day": churn_day, "active_days": days[active], "intensity": intensity[active],
            "hour": float(np.clip(rng.normal(14, 4), 5, 23)),
        })
    return users


def _ts(day: int, hour: float, rng) -> datetime:
    return START + timedelta(days=int(day), hours=float(np.clip(hour + rng.normal(0, 1.5), 0, 23.9)))


def _write(path: Path, ddl: str, rows: Dict[str, list]) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(ddl)
    for table, data in rows.items():
        if data:
            ph = ",".join("?" * len(data[0]))
            conn.executemany(f"INSERT INTO {table} VALUES ({ph})", data)
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------------------------
# learning_app (Duolingo-like)
# ------------------------------------------------------------------------------------------------

LEARNING_DDL = """
CREATE TABLE learners (
    learner_id INTEGER PRIMARY KEY,
    email TEXT,
    signup_date TEXT,
    plan TEXT,
    native_language TEXT,
    target_language TEXT
);
CREATE TABLE lessons (
    lesson_id INTEGER PRIMARY KEY,
    learner_id INTEGER NOT NULL REFERENCES learners(learner_id),
    completed_at TEXT NOT NULL,
    lesson_type TEXT,
    xp_earned INTEGER,
    duration_sec INTEGER,
    mistakes INTEGER
);
CREATE TABLE achievements (
    achievement_id INTEGER PRIMARY KEY,
    learner_id INTEGER NOT NULL REFERENCES learners(learner_id),
    earned_at TEXT NOT NULL,
    badge TEXT
);
"""


def generate_learning_app(out: Path, n_users: int = 3000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    spec = AppSpec(
        observed_days=180, future_days=60,
        base_p=lambda r: r.beta(4, 2), hazard=0.0035,
        weekday_mult=np.array([1, 1, 1, 1, 0.95, 0.85, 0.9]),
        plans={"free": (0.75, 1.3, 0.95), "super": (0.25, 0.45, 1.1)},
    )
    users = _simulate_latent(n_users, spec, rng)
    langs = ["en", "es", "pt", "de", "fr", "ja", "hi"]
    types = ["vocab", "grammar", "listening", "speaking", "story"]
    badges = ["streak_7", "streak_30", "perfect_lesson", "league_win", "xp_1000"]
    learners, obs, fut = [], {"lessons": [], "achievements": []}, {"lessons": [], "achievements": []}
    lid = aid = 1
    for u in users:
        uid = u["idx"] + 1
        signup = START + timedelta(days=u["signup_day"])
        learners.append((uid, f"learner{uid}@example.com", signup.strftime("%Y-%m-%d"), u["plan"],
                         langs[int(rng.integers(len(langs)))], langs[int(rng.integers(len(langs)))]))
        fav = rng.dirichlet(np.ones(len(types)) * 2)
        for d, inten in zip(u["active_days"], u["intensity"]):
            target = obs if d < spec.observed_days else fut
            for _ in range(1 + rng.poisson(1.6 * inten)):
                lt = types[int(rng.choice(len(types), p=fav))]
                xp = int(max(1, rng.normal(15 * (0.6 + 0.4 * inten), 4)))
                target["lessons"].append((lid, uid, _ts(d, u["hour"], rng).isoformat(timespec="seconds"), lt, xp,
                                          int(max(30, rng.normal(260 * inten, 60))), int(rng.poisson(2 / max(inten, 0.3)))))
                lid += 1
            if rng.random() < 0.06:
                target["achievements"].append((aid, uid, _ts(d, u["hour"], rng).isoformat(timespec="seconds"),
                                               badges[int(rng.integers(len(badges)))]))
                aid += 1
    _write(out / "learning_app.db", LEARNING_DDL, {"learners": learners, **obs})
    _write(out / "learning_app_future.db", LEARNING_DDL, {"learners": learners, **fut})
    return _meta(users, spec, lambda u: u["idx"] + 1)


# ------------------------------------------------------------------------------------------------
# navigation_app (Waze-like): UUID ids, unix timestamps, no declared foreign keys
# ------------------------------------------------------------------------------------------------

NAVIGATION_DDL = """
CREATE TABLE drivers (
    id TEXT PRIMARY KEY,
    joined_at INTEGER,
    vehicle_type TEXT,
    country TEXT,
    phone TEXT
);
CREATE TABLE trips (
    trip_id TEXT PRIMARY KEY,
    driver_id TEXT NOT NULL,
    started_at_ms INTEGER NOT NULL,
    distance_km REAL,
    duration_min REAL,
    avg_speed_kmh REAL
);
CREATE TABLE reports (
    report_id INTEGER PRIMARY KEY,
    driver_id TEXT NOT NULL,
    created_ts INTEGER NOT NULL,
    report_type TEXT
);
"""


def generate_navigation_app(out: Path, n_users: int = 3000, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    spec = AppSpec(
        observed_days=180, future_days=60,
        base_p=lambda r: r.beta(2.2, 3.5), hazard=0.003,
        weekday_mult=np.array([1.35, 1.35, 1.35, 1.35, 1.3, 0.55, 0.45]),
        plans={"car": (0.8, 1.0, 1.0), "motorcycle": (0.12, 1.3, 0.9), "truck": (0.08, 0.6, 1.2)},
    )
    users = _simulate_latent(n_users, spec, rng)
    countries = ["US", "BR", "IL", "FR", "MX", "IN"]
    rtypes = ["traffic_jam", "police", "hazard", "accident", "closure"]
    ids = [str(uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | i)) for i in range(n_users)]
    drivers, obs, fut = [], {"trips": [], "reports": []}, {"trips": [], "reports": []}
    rid = 1
    for u in users:
        did = ids[u["idx"]]
        joined = START + timedelta(days=u["signup_day"], hours=float(rng.uniform(0, 24)))
        drivers.append((did, int(joined.timestamp()), u["plan"], countries[int(rng.integers(len(countries)))],
                        f"+1555{int(rng.integers(10**6, 10**7))}"))
        commute = float(rng.lognormal(2.6, 0.5))
        for d, inten in zip(u["active_days"], u["intensity"]):
            target = obs if d < spec.observed_days else fut
            n_trips = 1 + int(rng.random() < 0.6 * inten) + int(rng.random() < 0.15)
            for k in range(n_trips):
                hour = 8.0 if k == 0 else 17.5 if k == 1 else float(rng.uniform(10, 22))
                ts = START + timedelta(days=int(d), hours=float(np.clip(hour + rng.normal(0, 0.8), 0, 23.9)))
                dist = float(max(0.5, rng.normal(commute, commute * 0.2) * (0.7 + 0.3 * inten)))
                dur = float(max(2, dist * rng.uniform(1.2, 2.4)))
                target["trips"].append((uuid.UUID(int=int(rng.integers(0, 2**63))).hex, did,
                                        int(ts.timestamp() * 1000), round(dist, 2), round(dur, 1), round(dist / dur * 60, 1)))
                if rng.random() < 0.1 * inten:
                    target["reports"].append((rid, did, int(ts.timestamp()) + 300, rtypes[int(rng.integers(len(rtypes)))]))
                    rid += 1
    _write(out / "navigation_app.db", NAVIGATION_DDL, {"drivers": drivers, **obs})
    _write(out / "navigation_app_future.db", NAVIGATION_DDL, {"drivers": drivers, **fut})
    return _meta(users, spec, lambda u: ids[u["idx"]])


# ------------------------------------------------------------------------------------------------
# saas_app: events link to accounts only through sessions (one join hop)
# ------------------------------------------------------------------------------------------------

SAAS_DDL = """
CREATE TABLE accounts (
    account_id TEXT PRIMARY KEY,
    created_at TEXT,
    plan_tier TEXT,
    industry TEXT,
    company_size INTEGER
);
CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(account_id),
    started_at TEXT NOT NULL,
    duration_min REAL
);
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(session_id),
    event_time TEXT NOT NULL,
    event_name TEXT
);
"""


def generate_saas_app(out: Path, n_users: int = 3000, seed: int = 23) -> dict:
    rng = np.random.default_rng(seed)
    spec = AppSpec(
        observed_days=180, future_days=60,
        base_p=lambda r: r.beta(3, 3), hazard=0.0025,
        weekday_mult=np.array([1.3, 1.3, 1.3, 1.3, 1.2, 0.25, 0.15]),
        plans={"starter": (0.6, 1.3, 0.9), "team": (0.3, 0.8, 1.05), "enterprise": (0.1, 0.35, 1.2)},
    )
    users = _simulate_latent(n_users, spec, rng)
    industries = ["fintech", "retail", "health", "education", "media"]
    names = ["create_doc", "edit_doc", "share", "comment", "export", "invite_member"]
    accounts, obs, fut = [], {"sessions": [], "events": []}, {"sessions": [], "events": []}
    sid = eid = 1
    for u in users:
        acc = f"acc_{u['idx'] + 1:05d}"
        created = START + timedelta(days=u["signup_day"], hours=float(rng.uniform(8, 18)))
        size = int(rng.lognormal({"starter": 2, "team": 3.2, "enterprise": 5}[u["plan"]], 0.5))
        accounts.append((acc, created.strftime("%Y-%m-%d %H:%M:%S"), u["plan"],
                         industries[int(rng.integers(len(industries)))], size))
        for d, inten in zip(u["active_days"], u["intensity"]):
            target = obs if d < spec.observed_days else fut
            for _ in range(1 + rng.poisson(0.8 * inten)):
                st = _ts(d, u["hour"], rng)
                target["sessions"].append((sid, acc, st.strftime("%Y-%m-%d %H:%M:%S"),
                                           round(float(max(1, rng.normal(25 * inten, 8))), 1)))
                for k in range(rng.poisson(4 * inten)):
                    et = st + timedelta(minutes=2 * k + float(rng.uniform(0, 2)))
                    nm_p = np.array([3, 5, 2, 2, 1, 0.5 * inten])
                    target["events"].append((eid, sid, et.strftime("%Y-%m-%d %H:%M:%S"),
                                             names[int(rng.choice(len(names), p=nm_p / nm_p.sum()))]))
                    eid += 1
                sid += 1
    _write(out / "saas_app.db", SAAS_DDL, {"accounts": accounts, **obs})
    _write(out / "saas_app_future.db", SAAS_DDL, {"accounts": accounts, **fut})
    return _meta(users, spec, lambda u: f"acc_{u['idx'] + 1:05d}")


def _meta(users, spec: AppSpec, uid) -> dict:
    return {
        "observed_end": (START + timedelta(days=spec.observed_days - 1)).strftime("%Y-%m-%d"),
        "churned_during_future": sum(1 for u in users if spec.observed_days <= u["churn_day"] < spec.observed_days + spec.future_days),
        "users": len(users),
    }


APPS = {
    "learning_app": generate_learning_app,
    "navigation_app": generate_navigation_app,
    "saas_app": generate_saas_app,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("apps", nargs="*", help=f"apps to generate: {', '.join(APPS)} (default: all)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="sample_data")
    ap.add_argument("--users", type=int, default=3000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    unknown = set(args.apps) - set(APPS)
    if unknown:
        ap.error(f"unknown app(s): {', '.join(sorted(unknown))}")
    for name in (args.apps or list(APPS)):
        meta = APPS[name](out, args.users)
        print(f"{name}: {out / (name + '.db')}  (+ {name}_future.db)  {meta}")


if __name__ == "__main__":
    main()
