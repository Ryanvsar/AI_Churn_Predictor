"""Name normalisation and vocabulary used by schema detection (no external services)."""
from __future__ import annotations

import difflib
import re
from typing import Iterable, List

# Words apps use for "the person whose churn we predict".
USER_SYNONYMS = [
    "user", "account", "customer", "member", "learner", "student", "driver", "rider", "player",
    "subscriber", "profile", "client", "person", "patient", "owner", "visitor", "listener",
    "viewer", "reader", "buyer", "shopper", "athlete", "gamer", "tenant", "employee", "contact",
    "guest", "passenger", "seller", "creator", "host", "device", "install", "installation",
]

# Generic id column names that usually point at the user entity.
USER_ID_NAMES = {"userid", "uid", "user", "accountid", "customerid", "memberid", "clientid", "personid",
                 "profileid", "ownerid", "subscriberid", "playerid", "learnerid", "studentid", "driverid",
                 "riderid", "visitorid", "deviceid", "installid", "installationid", "anonymousid"}

# Preferred names for the event time of an activity table (higher = better).
TIME_PREFERENCE = [
    (re.compile(r"(occurred|event_?time|event_?ts|event_?date|timestamp|^ts$)", re.I), 1.0),
    (re.compile(r"(start|began|begin|opened|login|played|completed|created|logged|recorded|sent|submitted)", re.I), 0.8),
    (re.compile(r"(date|time|_at$|day)", re.I), 0.6),
    (re.compile(r"(end|finish|stop|closed)", re.I), 0.3),
    (re.compile(r"(updated|modified|deleted|expire|birth|dob|due|deadline|scheduled|valid)", re.I), -1.0),
]

SIGNUP_NAME_RE = re.compile(r"(sign_?up|regist|joined|join_?date|created|install|first_?seen|onboard|start)", re.I)

PII_NAME_RE = re.compile(
    r"(e_?mail|phone|mobile|password|passwd|first_?name|last_?name|full_?name|^name$|username|user_?name|"
    r"address|street|zip|postal|ssn|social|credit|card|iban|ip_?addr|^ip$|token|secret|lat$|lon$|latitude|longitude)",
    re.I,
)

# Static user columns that typically reflect the *current* state and would leak the future into
# historical training snapshots (e.g. "last_login", "total_sessions", "status").
LEAKY_STATIC_WORDS = {
    "last", "latest", "recent", "total", "count", "cnt", "num", "number", "lifetime", "days", "streak",
    "score", "level", "xp", "point", "active", "status", "state", "churn", "churned", "cancel", "cancelled",
    "canceled", "end", "ended", "updated", "modified", "deleted", "closed", "balance", "spent", "revenue",
    "ltv", "session", "login", "visit", "reason", "retained", "inactive", "lapsed", "seen", "usage", "spend",
}


def is_leaky_static(name: str) -> bool:
    return any(singular(w) in LEAKY_STATIC_WORDS or w in LEAKY_STATIC_WORDS for w in split_words(name))


LABEL_NAME_RE = re.compile(
    r"^(is_?|has_?)?(churn|churned|churn_?label|churn_?flag|cancelled|canceled|cancel(l)?ed_?flag|"
    r"is_?active|active|retained|status|account_?status|subscription_?status|user_?status|state|lapsed|left)$",
    re.I,
)
LABEL_DATE_NAME_RE = re.compile(r"(churn(ed)?_?(at|date|on)|cancel(l)?ed_?(at|date|on)|deleted_?at|ended_?at|closed_?at)", re.I)

NEGATIVE_LABEL_NAME_RE = re.compile(r"(active|retained)", re.I)  # positive value means NOT churned
CHURN_VALUE_RE = re.compile(r"^(1|true|t|yes|y|churn|churned|cancel|cancelled|canceled|inactive|lapsed|left|closed|deleted|lost|terminated|expired)$", re.I)
ACTIVE_VALUE_RE = re.compile(r"^(0|false|f|no|n|active|retained|current|open|live|subscribed|trial|paying)$", re.I)

ID_NAME_RE = re.compile(r"(^id$|_id$|id$|^uuid$|_uuid$|uuid|_key$|^key$|guid|hash)", re.I)


def split_words(name: str) -> List[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return [w for w in re.split(r"[^0-9a-zA-Z]+", s.lower()) if w]


def singular(word: str) -> str:
    w = word.lower()
    if w == "people":
        return "person"
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def norm(name: str) -> str:
    """'User_IDs' / 'userIds' / 'user-id' -> 'userid'."""
    return "".join(singular(w) for w in split_words(name))


def match_any(name: str, vocab: Iterable[str], cutoff: float = 0.86) -> float:
    """Best fuzzy similarity of any word in `name` against `vocab` (0..1)."""
    vocab = list(vocab)
    best = 0.0
    for w in split_words(name):
        w = singular(w)
        if w in vocab:
            return 1.0
        m = difflib.get_close_matches(w, vocab, n=1, cutoff=cutoff)
        if m:
            best = max(best, difflib.SequenceMatcher(None, w, m[0]).ratio())
    return best


def humanize(name: str) -> str:
    words = split_words(name)
    return " ".join(words) if words else name
