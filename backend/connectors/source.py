"""
Open any relational data source as a read-only SQLAlchemy engine.

Accepted specs:
- A SQLAlchemy URL: postgresql://..., mysql+pymysql://..., mssql+pyodbc://..., sqlite:///path.db, ...
- A SQLite file path (.db / .sqlite / .sqlite3, or any file with the SQLite header)
- A CSV / TSV / Parquet file, or a directory of them (each file becomes a table)
- An Excel workbook (.xlsx / .xls) (each sheet becomes a table)
- A .zip archive containing any of the above

File-based tabular sources are loaded into an in-memory SQLite database so the rest of the
pipeline only ever talks SQL through SQLAlchemy.
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ArgumentError, NoSuchModuleError
from sqlalchemy.pool import StaticPool

SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".db3"}
TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".pq"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}

# Pip packages that provide the DBAPI driver for common dialects.
DRIVER_HINTS = {
    "postgresql": "\"psycopg[binary]\"  (or psycopg2-binary with postgresql+psycopg2://)",
    "mysql": "pymysql  (use mysql+pymysql://...)",
    "mariadb": "pymysql  (use mariadb+pymysql://...)",
    "mssql": "pyodbc  (use mssql+pyodbc://...)",
    "oracle": "oracledb  (use oracle+oracledb://...)",
}


class SourceError(ValueError):
    """Raised when a data source cannot be opened."""


@dataclass
class DataSource:
    spec: str
    kind: str  # sqlite | url | csv | excel | parquet | zip
    display: str  # human readable, password redacted
    engine: Engine

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name


_CACHE: "OrderedDict[str, DataSource]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_SIZE = 8


def _looks_like_url(spec: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+_.-]*://", spec))


def _is_sqlite_file(path: Path) -> bool:
    if path.suffix.lower() in SQLITE_SUFFIXES:
        return True
    try:
        with open(path, "rb") as f:
            return f.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def _table_name_from_file(path: Path, used: set) -> str:
    base = re.sub(r"[^0-9a-zA-Z_]+", "_", path.stem).strip("_") or "table"
    if base[0].isdigit():
        base = "t_" + base
    name, i = base, 2
    while name.lower() in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name.lower())
    return name


def _read_tabular(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    sep = "\t" if suf == ".tsv" else None
    return pd.read_csv(path, sep=sep, engine="python" if sep is None else "c")


def _memory_engine() -> Engine:
    return create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


def _load_frames(frames: dict) -> Engine:
    if not frames:
        raise SourceError("No tables found in the provided file(s).")
    engine = _memory_engine()
    for name, df in frames.items():
        df.to_sql(name, engine, index=False, if_exists="replace")
    return engine


def _open_sqlite_file(path: Path) -> Engine:
    # Read-only URI so the source database can never be modified by the app.
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return create_engine(
        "sqlite+pysqlite:///" + uri + "&uri=true",
        connect_args={"check_same_thread": False},
    )


def _open_directory(path: Path) -> tuple[Engine, str]:
    files = sorted(p for p in path.iterdir() if p.is_file())
    sqlite_files = [p for p in files if _is_sqlite_file(p)]
    if sqlite_files and len(sqlite_files) == 1:
        return _open_sqlite_file(sqlite_files[0]), "sqlite"
    used: set = set()
    frames = {}
    for p in files:
        suf = p.suffix.lower()
        if suf in TABULAR_SUFFIXES:
            frames[_table_name_from_file(p, used)] = _read_tabular(p)
        elif suf in EXCEL_SUFFIXES:
            for sheet, df in pd.read_excel(p, sheet_name=None).items():
                frames[_table_name_from_file(Path(str(sheet)), used)] = df
    if not frames:
        raise SourceError(f"Directory contains no SQLite, CSV, Parquet or Excel files: {path}")
    return _load_frames(frames), "csv"


def _safe_extract(zip_path: Path) -> Path:
    out = Path(tempfile.mkdtemp(prefix="churn_zip_"))
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name  # flatten; prevents path traversal
            if not name or name.startswith("."):
                continue
            with zf.open(member) as src, open(out / name, "wb") as dst:
                dst.write(src.read())
    return out


def _open_uncached(spec: str) -> DataSource:
    spec = spec.strip()
    if not spec:
        raise SourceError("Data source is empty. Provide a file path or a database URL.")

    if _looks_like_url(spec):
        try:
            url = make_url(spec)
        except ArgumentError as e:
            raise SourceError(f"Invalid database URL: {e}") from e
        display = url.render_as_string(hide_password=True)
        backend = url.get_backend_name()
        kwargs = {"pool_pre_ping": True}
        if backend == "sqlite":
            kwargs["connect_args"] = {"check_same_thread": False}
        try:
            engine = create_engine(url, **kwargs)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except (NoSuchModuleError, ModuleNotFoundError, ImportError) as e:
            hint = DRIVER_HINTS.get(backend, "the DBAPI driver for this database")
            raise SourceError(f"Database driver for '{backend}' is not installed. Install: pip install {hint}") from e
        except Exception as e:  # noqa: BLE001 - surface connection problems clearly
            raise SourceError(f"Could not connect to {display}: {e}") from e
        return DataSource(spec=spec, kind="sqlite" if backend == "sqlite" else "url", display=display, engine=engine)

    path = Path(os.path.expanduser(spec))
    if not path.exists():
        raise SourceError(f"File or directory not found: {spec}")

    if path.is_dir():
        engine, kind = _open_directory(path)
        return DataSource(spec=spec, kind=kind, display=str(path), engine=engine)

    suf = path.suffix.lower()
    if suf == ".zip":
        engine, _ = _open_directory(_safe_extract(path))
        return DataSource(spec=spec, kind="zip", display=str(path), engine=engine)
    if suf in TABULAR_SUFFIXES:
        frames = {_table_name_from_file(path, set()): _read_tabular(path)}
        return DataSource(spec=spec, kind="csv", display=str(path), engine=_load_frames(frames))
    if suf in EXCEL_SUFFIXES:
        used: set = set()
        frames = {
            _table_name_from_file(Path(str(sheet)), used): df
            for sheet, df in pd.read_excel(path, sheet_name=None).items()
        }
        return DataSource(spec=spec, kind="excel", display=str(path), engine=_load_frames(frames))
    if _is_sqlite_file(path):
        return DataSource(spec=spec, kind="sqlite", display=str(path), engine=_open_sqlite_file(path))

    raise SourceError(
        f"Unsupported file type '{suf or path.name}'. Use a SQLite file, CSV/Parquet/Excel, a .zip, or a database URL."
    )


def open_source(spec: str, use_cache: bool = True) -> DataSource:
    """Open (or reuse) a read-only data source for the given spec."""
    key = spec.strip()
    if not _looks_like_url(key) and os.path.exists(os.path.expanduser(key)):
        # Include the modification time so a regenerated file is re-opened.
        key = f"{key}@{os.path.getmtime(os.path.expanduser(key))}"
    if use_cache:
        with _CACHE_LOCK:
            if key in _CACHE:
                _CACHE.move_to_end(key)
                return _CACHE[key]
    src = _open_uncached(spec)
    if use_cache:
        with _CACHE_LOCK:
            _CACHE[key] = src
            while len(_CACHE) > _CACHE_SIZE:
                _, old = _CACHE.popitem(last=False)
                old.engine.dispose()
    return src


def clear_cache() -> None:
    with _CACHE_LOCK:
        for src in _CACHE.values():
            src.engine.dispose()
        _CACHE.clear()
