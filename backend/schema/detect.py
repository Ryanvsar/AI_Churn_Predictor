"""
Heuristic, offline schema understanding.

Given a profiled database, work out:
- which table holds the users (the entity whose churn we predict) and its key column,
- which tables record user activity over time, how they link to users (FK, column name,
  value overlap or one join hop) and which column is the event time,
- which numeric columns are measures and which string columns are categories,
- static user attributes (signup date, plan, country, ...) that are safe to use,
- an optional explicit churn label.

Every decision carries a confidence and a human-readable reason so the UI can show it and the
user can override it. The result is a JSON-serialisable `SchemaMap`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.connectors.source import DataSource
from backend.schema import naming as nm
from backend.schema.introspect import ColumnProfile, DatabaseProfile, TableProfile, count_matching_keys
from backend.schema.timeparse import is_time_like_name

MAX_MEASURES = 8
MAX_CATEGORICALS = 4
MAX_STATIC_CATEGORICALS = 6
MAX_STATIC_NUMERICS = 6
CATEGORY_HINT = ("type", "kind", "category", "feature", "action", "event", "name", "screen", "page", "platform",
                 "source", "mode", "channel", "level", "skill", "topic", "activity", "badge", "report")


@dataclass
class JoinHop:
    table: str  # intermediate table that holds the user link
    local_column: str  # column in the activity table referencing the intermediate table
    remote_column: str  # key column in the intermediate table
    user_column: str  # user link column in the intermediate table


@dataclass
class ActivityMap:
    table: str
    user_column: Optional[str]
    time_column: str
    time_format: str
    measures: List[str] = field(default_factory=list)
    categoricals: List[str] = field(default_factory=list)
    via: Optional[JoinHop] = None
    enabled: bool = True
    confidence: float = 0.0
    reason: str = ""


@dataclass
class LabelMap:
    table: str
    column: str
    user_column: str
    kind: str = "binary"  # binary | not_null
    positive_values: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


@dataclass
class SchemaMap:
    user_table: Optional[str] = None
    user_id_column: Optional[str] = None
    user_confidence: float = 0.0
    user_reason: str = ""
    signup_column: Optional[str] = None
    signup_format: Optional[str] = None
    static_categoricals: List[str] = field(default_factory=list)
    static_numerics: List[str] = field(default_factory=list)
    label: Optional[LabelMap] = None
    activities: List[ActivityMap] = field(default_factory=list)
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def enabled_activities(self) -> List[ActivityMap]:
        return [a for a in self.activities if a.enabled]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SchemaMap":
        d = dict(d or {})
        acts = []
        for a in d.get("activities") or []:
            a = dict(a)
            via = a.get("via")
            a["via"] = JoinHop(**via) if via else None
            acts.append(ActivityMap(**{k: v for k, v in a.items() if k in ActivityMap.__dataclass_fields__}))
        label = d.get("label")
        lab = LabelMap(**{k: v for k, v in label.items() if k in LabelMap.__dataclass_fields__}) if label else None
        base = {k: v for k, v in d.items() if k in SchemaMap.__dataclass_fields__ and k not in ("activities", "label")}
        return SchemaMap(**base, label=lab, activities=acts)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def _is_id_like(col: ColumnProfile) -> bool:
    return bool(nm.ID_NAME_RE.search(col.name)) or col.is_pk


def _entity_key(t: TableProfile) -> Optional[str]:
    if len(t.pk) == 1:
        return t.pk[0]
    target = nm.norm(nm.singular(t.name)) + "id"
    sample_n = min(t.row_count, len(t.sample) if t.sample is not None else 0)
    for c in t.columns.values():
        n = nm.norm(c.name)
        if (n in ("id", target, "uuid", "guid") or n in nm.USER_ID_NAMES) and c.null_frac == 0:
            if sample_n == 0 or c.unique_frac >= 0.999:
                return c.name
    return None


def _strong_link_names(user_table: str, user_key: str) -> set:
    names = {nm.norm(nm.singular(user_table)) + "id"}
    k = nm.norm(user_key)
    if k not in ("id", "uuid", "guid", "key"):
        names.add(k)
    return names


def _distinct_sample(t: TableProfile, col: str, n: int = 200) -> list:
    if t.sample is None or col not in t.sample.columns:
        return []
    return t.sample[col].dropna().drop_duplicates().head(n).tolist()


def _time_candidates(t: TableProfile, exclude: set) -> List[Tuple[str, float]]:
    out = []
    for c in t.columns.values():
        if c.name in exclude or c.time_format is None or c.time_parse_rate < 0.9:
            continue
        pref = 0.0
        for rx, w in nm.TIME_PREFERENCE:
            if rx.search(c.name):
                pref = w
                break
        score = c.time_parse_rate + pref - c.null_frac
        out.append((c.name, score))
    out.sort(key=lambda x: -x[1])
    return out


def _measures(t: TableProfile, exclude: set) -> List[str]:
    picks = []
    for c in t.columns.values():
        if c.name in exclude or c.kind not in ("numeric", "bool"):
            continue
        if _is_id_like(c) or nm.PII_NAME_RE.search(c.name) or c.n_unique < 2:
            continue
        if c.time_format is not None and is_time_like_name(c.name):
            continue
        picks.append((c.name, c.null_frac))
    picks.sort(key=lambda x: x[1])
    return [p[0] for p in picks[:MAX_MEASURES]]


def _categoricals(t: TableProfile, exclude: set, max_n: int = MAX_CATEGORICALS, max_unique: int = 60) -> List[str]:
    picks = []
    sample_n = len(t.sample) if t.sample is not None else 0
    for c in t.columns.values():
        if c.name in exclude or c.kind != "string" or c.time_format is not None:
            continue
        if _is_id_like(c) or nm.PII_NAME_RE.search(c.name):
            continue
        if not (2 <= c.n_unique <= max_unique) or c.avg_len > 40:
            continue
        if sample_n >= 50 and c.unique_frac > 0.5:
            continue
        hint = any(h in c.name.lower() for h in CATEGORY_HINT)
        picks.append((c.name, (0 if hint else 1, c.n_unique)))
    picks.sort(key=lambda x: x[1])
    return [p[0] for p in picks[:max_n]]


def _label_positive_values(col: ColumnProfile, t: TableProfile) -> Optional[List[str]]:
    vals = [str(v).strip().lower() for v in t.sample[col.name].dropna().unique()] if t.sample is not None else []
    vals = [v[:-2] if v.endswith(".0") else v for v in vals]
    vals = sorted(set(vals))
    if len(vals) < 2 or len(vals) > 6:
        return None
    negative_name = bool(nm.NEGATIVE_LABEL_NAME_RE.search(col.name)) and "inactive" not in col.name.lower()
    if negative_name:
        pos = [v for v in vals if v in ("0", "false", "f", "no", "n")]
    else:
        pos = [v for v in vals if nm.CHURN_VALUE_RE.match(v)]
    if not pos or len(pos) == len(vals):
        return None
    return pos


# ----------------------------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------------------------

def detect_schema(source: DataSource, profile: DatabaseProfile) -> SchemaMap:
    tables = profile.tables
    smap = SchemaMap()
    if not tables:
        smap.notes.append("The database has no tables.")
        return smap

    # ---------------- 1. user entity table ----------------
    best: Tuple[float, Optional[str], Optional[str], str] = (0.0, None, None, "")
    for t in tables.values():
        key = _entity_key(t)
        if key is None:
            continue
        name_score = nm.match_any(t.name, nm.USER_SYNONYMS)
        strong = _strong_link_names(t.name, key)
        referrers = {
            o.name for o in tables.values()
            if o.name != t.name and any(nm.norm(c) in strong for c in o.columns if c not in o.pk)
        }
        referrers |= {o.name for o in tables.values() if any(fk.ref_table == t.name for fk in o.fks) and o.name != t.name}
        refs = len(referrers)
        has_fk_out = any(fk.ref_table != t.name for fk in t.fks)
        score = 1.5 * name_score + 0.5 * min(refs, 4) - (0.8 if has_fk_out else 0.0)
        if refs == 0 and name_score < 0.85:
            continue
        if score > best[0]:
            why = []
            if name_score >= 0.85:
                why.append(f"name '{t.name}' looks like a user entity")
            if refs:
                why.append(f"referenced by {refs} other table(s)")
            best = (score, t.name, key, "; ".join(why))

    if best[1] is not None:
        smap.user_table, smap.user_id_column = best[1], best[2]
        smap.user_confidence = round(min(1.0, best[0] / 3.0), 2)
        smap.user_reason = best[3]
    else:
        smap.notes.append("No user table found; users are identified from a user-id column in activity tables.")

    user_rows = tables[smap.user_table].row_count if smap.user_table else 0

    # ---------------- 2. link every other table to the user ----------------
    links: Dict[str, Tuple[str, float, str]] = {}  # table -> (column, confidence, reason)
    for t in tables.values():
        if t.name == smap.user_table:
            continue
        found = None
        if smap.user_table:
            for fk in t.fks:
                if fk.ref_table == smap.user_table and len(fk.columns) == 1:
                    found = (fk.columns[0], 1.0, f"foreign key {fk.columns[0]} → {smap.user_table}")
                    break
            if found is None:
                strong = _strong_link_names(smap.user_table, smap.user_id_column)
                cands = [c for c in t.columns.values() if c.name not in t.pk or len(t.pk) > 1]
                named = [c for c in cands if nm.norm(c.name) in strong] or [
                    c for c in cands if nm.norm(c.name) in nm.USER_ID_NAMES
                ]
                for c in named:
                    vals = _distinct_sample(t, c.name)
                    hit = count_matching_keys(source, smap.user_table, smap.user_id_column, vals) / max(len(vals), 1)
                    if hit >= 0.5:
                        found = (c.name, 0.9 if hit >= 0.8 else 0.7, f"column name {c.name} matches users ({hit:.0%} of values found)")
                        break
            if found is None:
                # Value overlap on id-like columns (no declared FK, unusual name).
                for c in t.columns.values():
                    if c.name in t.pk or not _is_id_like(c) or c.unique_frac > 0.95:
                        continue
                    vals = _distinct_sample(t, c.name)
                    if len(vals) < 5:
                        continue
                    hit = count_matching_keys(source, smap.user_table, smap.user_id_column, vals) / len(vals)
                    if hit >= 0.8:
                        found = (c.name, 0.75, f"{hit:.0%} of {c.name} values are {smap.user_table}.{smap.user_id_column}")
                        break
        else:
            for c in t.columns.values():
                if nm.norm(c.name) in nm.USER_ID_NAMES or (
                    nm.match_any(c.name, nm.USER_SYNONYMS) >= 0.85 and nm.ID_NAME_RE.search(c.name)
                ):
                    found = (c.name, 0.7, f"column {c.name} looks like a user id")
                    break
        if found:
            links[t.name] = found

    # One join hop: activity tables that reference a linked table (events -> sessions -> users).
    hops: Dict[str, JoinHop] = {}
    for t in tables.values():
        if t.name == smap.user_table or t.name in links:
            continue
        for fk in t.fks:
            if fk.ref_table in links and len(fk.columns) == 1:
                hops[t.name] = JoinHop(fk.ref_table, fk.columns[0], fk.ref_columns[0], links[fk.ref_table][0])
                break
        if t.name in hops:
            continue
        for p_name, (p_user_col, _, _) in links.items():
            p = tables[p_name]
            if len(p.pk) != 1:
                continue
            p_key_norms = {nm.norm(nm.singular(p_name)) + "id"}
            if nm.norm(p.pk[0]) not in ("id", "uuid"):
                p_key_norms.add(nm.norm(p.pk[0]))
            match = [c for c in t.columns if c not in t.pk and nm.norm(c) in p_key_norms]
            if match:
                hops[t.name] = JoinHop(p_name, match[0], p.pk[0], p_user_col)
                break

    # ---------------- 3. activity tables ----------------
    n_users_est = user_rows
    if not n_users_est and links:
        t0 = tables[next(iter(links))]
        n_users_est = max(1, t0.columns[links[t0.name][0]].n_unique)

    for t in tables.values():
        if t.name == smap.user_table or (t.name not in links and t.name not in hops):
            continue
        via = hops.get(t.name)
        user_col = None if via else links[t.name][0]
        link_conf = 0.8 if via else links[t.name][1]
        link_reason = f"joined to users via {via.table}.{via.user_column}" if via else links[t.name][2]
        if t.row_count == 0:
            smap.notes.append(f"Table '{t.name}' is empty; skipped.")
            continue
        exclude = set(t.pk) | {c for c in [user_col, via.local_column if via else None] if c}
        for fk in t.fks:
            exclude |= set(fk.columns)
        times = _time_candidates(t, exclude)
        if not times:
            smap.notes.append(f"Table '{t.name}' is linked to users but has no usable timestamp; skipped.")
            continue
        time_col = times[0][0]
        tcol = t.columns[time_col]
        time_cols = {name for name, _ in times}
        act = ActivityMap(
            table=t.name,
            user_column=user_col,
            time_column=time_col,
            time_format=tcol.time_format,
            measures=_measures(t, exclude | time_cols),
            categoricals=_categoricals(t, exclude | time_cols),
            via=via,
            confidence=round(min(1.0, 0.5 * link_conf + 0.5 * min(1.0, tcol.time_parse_rate)), 2),
            reason=f"{link_reason}; time column {time_col} ({tcol.time_format})",
        )
        rows_per_user = t.row_count / max(n_users_est, 1)
        if rows_per_user < 1.2:
            act.enabled = False
            act.reason += f"; only {rows_per_user:.2f} rows per user — looks like a profile table, disabled"
        smap.activities.append(act)

    smap.activities.sort(key=lambda a: (not a.enabled, -tables[a.table].row_count))
    if not smap.enabled_activities():
        smap.notes.append("No activity table with timestamps was found. Map one manually to run the analysis.")

    # ---------------- 4. static attributes + label ----------------
    if smap.user_table:
        ut = tables[smap.user_table]
        exclude = {smap.user_id_column}
        label = _detect_label(ut, smap.user_id_column)
        if label is None:
            for t in tables.values():
                if t.name in links and t.row_count <= max(user_rows, 1) * 1.05:
                    label = _detect_label(t, links[t.name][0])
                    if label:
                        break
        smap.label = label
        if label and label.table == ut.name:
            exclude.add(label.column)

        signup = [
            c for c in ut.columns.values()
            if c.name not in exclude and c.time_format and nm.SIGNUP_NAME_RE.search(c.name)
            and not nm.is_leaky_static(c.name)
        ]
        if signup:
            smap.signup_column, smap.signup_format = signup[0].name, signup[0].time_format
            exclude.add(signup[0].name)
        skipped = []
        for c in ut.columns.values():
            if c.name in exclude or c.time_format is not None:
                continue
            if nm.PII_NAME_RE.search(c.name):
                skipped.append(f"{c.name} (personal data)")
                continue
            if nm.is_leaky_static(c.name):
                skipped.append(f"{c.name} (reflects current state; would leak the future into training)")
                continue
            if _is_id_like(c):
                continue
            if c.kind in ("numeric", "bool") and c.n_unique >= 2 and len(smap.static_numerics) < MAX_STATIC_NUMERICS:
                smap.static_numerics.append(c.name)
        smap.static_categoricals = [
            c for c in _categoricals(ut, exclude | set(smap.static_numerics), MAX_STATIC_CATEGORICALS, 30)
            if not nm.is_leaky_static(c)
        ]
        if skipped:
            smap.notes.append("Ignored user columns: " + ", ".join(skipped) + ".")

    confs = [smap.user_confidence] if smap.user_table else []
    confs += [a.confidence for a in smap.enabled_activities()]
    smap.confidence = round(sum(confs) / len(confs), 2) if confs else 0.0
    return smap


def _detect_label(t: TableProfile, user_col: str) -> Optional[LabelMap]:
    for c in t.columns.values():
        if c.name == user_col or c.is_pk:
            continue
        if nm.LABEL_NAME_RE.match(c.name) and c.time_format is None:
            pos = _label_positive_values(c, t)
            if pos:
                return LabelMap(
                    table=t.name, column=c.name, user_column=user_col, kind="binary", positive_values=pos,
                    confidence=0.9, reason=f"column {c.name}: values {pos} mean churned",
                )
    for c in t.columns.values():
        if c.name == user_col:
            continue
        if c.time_format and nm.LABEL_DATE_NAME_RE.search(c.name) and 0.0 < c.null_frac < 1.0:
            return LabelMap(
                table=t.name, column=c.name, user_column=user_col, kind="not_null", positive_values=[],
                confidence=0.7, reason=f"column {c.name}: a date is set when the user churned",
            )
    return None


def validate_schema_map(smap: SchemaMap, profile: DatabaseProfile) -> List[str]:
    """Return a list of problems with a (possibly user-edited) schema map."""
    errs = []
    tables = profile.tables

    def has(table, col):
        return table in tables and col in tables[table].columns

    if smap.user_table and not has(smap.user_table, smap.user_id_column or ""):
        errs.append(f"User table/key not found: {smap.user_table}.{smap.user_id_column}")
    if not smap.enabled_activities():
        errs.append("At least one enabled activity table is required.")
    for a in smap.activities:
        if a.table not in tables:
            errs.append(f"Activity table not found: {a.table}")
            continue
        if not has(a.table, a.time_column):
            errs.append(f"Time column not found: {a.table}.{a.time_column}")
        elif not a.time_format:
            a.time_format = tables[a.table].columns[a.time_column].time_format or "iso"
        if a.via is None and not has(a.table, a.user_column or ""):
            errs.append(f"User column not found: {a.table}.{a.user_column}")
        if a.via is not None and not (has(a.table, a.via.local_column) and has(a.via.table, a.via.remote_column)
                                      and has(a.via.table, a.via.user_column)):
            errs.append(f"Join path for {a.table} via {a.via.table} is invalid")
        for col in a.measures + a.categoricals:
            if not has(a.table, col):
                errs.append(f"Column not found: {a.table}.{col}")
    if smap.label and not has(smap.label.table, smap.label.column):
        errs.append(f"Label column not found: {smap.label.table}.{smap.label.column}")
    if smap.user_table:
        for col in smap.static_categoricals + smap.static_numerics + ([smap.signup_column] if smap.signup_column else []):
            if not has(smap.user_table, col):
                errs.append(f"User column not found: {smap.user_table}.{col}")
        if smap.signup_column and not smap.signup_format:
            smap.signup_format = tables[smap.user_table].columns[smap.signup_column].time_format or "iso"
    return errs
