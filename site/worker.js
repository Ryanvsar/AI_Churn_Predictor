/*
 * In-browser analysis engine.
 *
 * Runs the project's Python pipeline (backend/ — bundled as py/churn_backend.zip) inside
 * Pyodide, so a visitor's file is analysed on their own computer and never uploaded.
 *
 * Protocol: the page posts {id, method, args}; the worker replies {id, ok, result|error}
 * and streams {type: "progress", p, msg} while working.
 */
const PYODIDE_VERSION = "0.27.7";
// Optional self-hosted Pyodide: worker.js?pyodide=/path/to/pyodide/ (same-origin paths only).
const SELF_HOSTED = new URL(self.location.href).searchParams.get("pyodide");
const PYODIDE_URL = SELF_HOSTED && /^\/(?!\/)/.test(SELF_HOSTED)
  ? new URL(SELF_HOSTED.endsWith("/") ? SELF_HOSTED : SELF_HOSTED + "/", self.location.origin).href
  : `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "sqlalchemy", "sqlite3"];
const BROWSER_MAX_USERS = 50000;

let pyodide = null;
let readyPromise = null;
let currentFile = null;

function progress(p, msg) {
  self.postMessage({ type: "progress", p, msg });
}

async function boot() {
  progress(0.02, "Loading the analysis engine");
  importScripts(PYODIDE_URL + "pyodide.js");
  pyodide = await self.loadPyodide({ indexURL: PYODIDE_URL });
  progress(0.06, "Loading data science libraries");
  await pyodide.loadPackage(PACKAGES, { messageCallback: () => {}, errorCallback: () => {} });
  const zip = await fetch(new URL("py/churn_backend.zip", self.location.href)).then((r) => {
    if (!r.ok) throw new Error("Could not load the analysis code (py/churn_backend.zip).");
    return r.arrayBuffer();
  });
  pyodide.unpackArchive(zip, "zip", { extractDir: "/app" });
  pyodide.runPython(`
import sys, json
sys.path.insert(0, "/app")
from backend import service as svc
_AN = None

def _dump(obj):
    return json.dumps(svc.clean(obj), allow_nan=False)
`);
  progress(0.1, "Engine ready");
}

function ready() {
  if (!readyPromise) readyPromise = boot();
  return readyPromise;
}

async function ensureExtras(path) {
  // Excel needs openpyxl (pure Python, fetched once from PyPI); Parquet needs pyarrow.
  pyodide.globals.set("_check", path);
  const kinds = pyodide.runPython(`
import zipfile
_names = [_check]
if _check.lower().endswith(".zip"):
    with zipfile.ZipFile(_check) as _z:
        _names = _z.namelist()
",".join(sorted({n.lower().rsplit(".", 1)[-1] for n in _names if "." in n}))
`).split(",");
  if (kinds.some((k) => ["xlsx", "xlsm", "xls"].includes(k))) {
    try {
      await pyodide.loadPackage("micropip");
      await pyodide.runPythonAsync(`
import micropip
try:
    import openpyxl  # noqa: F401
except ImportError:
    await micropip.install("openpyxl")
`);
    } catch (e) {
      throw new Error("Excel support could not be downloaded. Save the sheets as CSV and try again.");
    }
  }
  if (kinds.some((k) => ["parquet", "pq"].includes(k))) {
    await pyodide.loadPackage("pyarrow");
  }
}

const methods = {
  async init() {
    await ready();
    return true;
  },

  async detect({ name, buffer }) {
    await ready();
    progress(0.12, "Reading your file");
    const safe = name.replace(/[^0-9A-Za-z._-]+/g, "_") || "upload.db";
    try { pyodide.FS.mkdirTree("/data"); } catch (e) { /* exists */ }
    for (const f of pyodide.FS.readdir("/data")) {
      if (f !== "." && f !== "..") pyodide.FS.unlink("/data/" + f);
    }
    currentFile = "/data/" + safe;
    pyodide.FS.writeFile(currentFile, new Uint8Array(buffer));
    await ensureExtras(currentFile);
    progress(0.16, "Understanding the database");
    pyodide.globals.set("_path", currentFile);
    pyodide.globals.set("_name", name);
    return JSON.parse(pyodide.runPython("_dump(svc.validate_source(_path, display_name=_name))"));
  },

  async analyze(options) {
    await ready();
    if (!currentFile) throw new Error("Choose a file first.");
    const opts = Object.assign({ quick: true, max_users: BROWSER_MAX_USERS }, options || {}, { source: currentFile });
    pyodide.globals.set("_opts", JSON.stringify(opts));
    pyodide.globals.set("_progress", (p, msg) => progress(0.2 + 0.8 * p, msg));
    return JSON.parse(pyodide.runPython(`
_AN = svc.analyze(json.loads(_opts), lambda p, m: _progress(p, m))
_dump(svc.full_result(_AN))
`));
  },

  async user({ id }) {
    pyodide.globals.set("_uid", String(id));
    return JSON.parse(pyodide.runPython("_dump(svc.user_bundle(_AN, _uid))"));
  },

  async exportCsv() {
    return pyodide.runPython("svc.export_csv(_AN)");
  },
};

self.onmessage = async (ev) => {
  const { id, method, args } = ev.data || {};
  try {
    if (!methods[method]) throw new Error("Unknown method " + method);
    const result = await methods[method](args || {});
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    let msg = String((err && err.message) || err);
    // Python tracebacks: keep the last line, which carries the actual error.
    const lines = msg.trim().split("\n");
    if (lines.length > 1 && /Traceback|File "/.test(msg)) msg = lines[lines.length - 1];
    self.postMessage({ id, ok: false, error: msg.replace(/^\w*Error: /, "") });
  }
};
