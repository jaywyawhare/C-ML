#!/usr/bin/env python3
"""
exp_server.py — experiment-tracking backend (prototype, zero dependencies).

NOTE: this is now imported by viz/serve.py, which serves the whole UI (viz +
experiment dashboard) on a single port (default 6969). Running this standalone
is only a fallback.

Ingests append-only run logs from .cml/experiments/runs/ into an in-memory
SQLite DB (stdlib) and serves a comparison UI + JSON API. This is the
persistence/query/serve half of the W&B-style prototype:

  L2 multi-run + persistence   ingest -> sqlite, survives run restarts
  L3 run comparison            /api/metrics?runs=a,b&keys=...
  L5 dynamic panels            /api/keys -> UI builds one panel per metric
  L9 sweeps                    /api/sweep?id=... -> parallel coordinates

Run from the directory containing .cml/  (e.g. cd /tmp/cmlviz; python3 .../exp_server.py)
"""
import http.server
import socketserver
import sqlite3
import json
import os
import shutil
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.environ.get("CML_EXP_DIR", ".cml/experiments")
RUNS_DIR = os.path.join(EXP_DIR, "runs")
DB_PATH = os.path.join(EXP_DIR, "experiments.db")  # durable on-disk store
PORT = int(os.environ.get("PORT", "6969"))  # standalone fallback; normally embedded in serve.py

_db_lock = threading.Lock()
_db = None
_last_sig = None


def _scan_signature():
    """Cheap change-detector: (path, mtime) of every events.jsonl + meta.json."""
    sig = []
    if not os.path.isdir(RUNS_DIR):
        return tuple(sig)
    for rid in os.listdir(RUNS_DIR):
        rd = os.path.join(RUNS_DIR, rid)
        for fn in ("meta.json", "events.jsonl"):
            p = os.path.join(rd, fn)
            try:
                sig.append((p, os.path.getmtime(p)))
            except OSError:
                pass
    return tuple(sorted(sig))


SCHEMA = """
    DROP TABLE IF EXISTS runs; DROP TABLE IF EXISTS alerts; DROP TABLE IF EXISTS scalars;
    DROP TABLE IF EXISTS hists; DROP TABLE IF EXISTS systemm; DROP TABLE IF EXISTS media;
    DROP TABLE IF EXISTS artifacts; DROP TABLE IF EXISTS curves; DROP TABLE IF EXISTS _meta;
    DROP TABLE IF EXISTS richtables; DROP TABLE IF EXISTS uses;
    CREATE TABLE runs(id TEXT PRIMARY KEY, project TEXT, name TEXT, sweep TEXT,
                      status TEXT, config TEXT, created REAL, host TEXT, os TEXT,
                      git TEXT, notes TEXT, tags TEXT, start REAL, duration REAL,
                      summary TEXT, cmd TEXT);
    CREATE TABLE alerts(run TEXT, level TEXT, message TEXT, wall REAL);
    CREATE TABLE scalars(run TEXT, name TEXT, step INTEGER, value REAL, wall REAL);
    CREATE TABLE hists(run TEXT, name TEXT, step INTEGER, mn REAL, mx REAL, counts TEXT);
    CREATE TABLE systemm(run TEXT, step INTEGER, cpu REAL, rss REAL, wall REAL);
    CREATE TABLE media(run TEXT, kind TEXT, name TEXT, step INTEGER, w INTEGER,
                       h INTEGER, path TEXT, csv TEXT);
    CREATE TABLE artifacts(run TEXT, name TEXT, atype TEXT, path TEXT, size INTEGER,
                           hash TEXT, aliases TEXT);
    CREATE TABLE curves(run TEXT, name TEXT, xlabel TEXT, ylabel TEXT, xs TEXT, ys TEXT);
    CREATE TABLE richtables(run TEXT, name TEXT, columns TEXT, rows TEXT);
    CREATE TABLE uses(run TEXT, artifact TEXT, version TEXT);
    CREATE TABLE _meta(k TEXT PRIMARY KEY, v TEXT);
    CREATE INDEX ix_scalars ON scalars(run, name, step);
"""


SCHEMA_VERSION = "4"  # bump when SCHEMA changes so stale on-disk DBs are discarded


def _build_db(sig):
    # Durable: persists to DB_PATH so it survives restarts; rebuilt only when
    # the run files change. Self-heals a stale/corrupt on-disk DB.
    def fresh():
        try:
            if os.path.isfile(DB_PATH):
                os.remove(DB_PATH)
            return sqlite3.connect(DB_PATH, check_same_thread=False)
        except Exception:
            return sqlite3.connect(":memory:", check_same_thread=False)
    try:
        db = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.executescript(SCHEMA)
    except Exception:
        db = fresh()
        db.executescript(SCHEMA)
    db.execute("INSERT INTO _meta VALUES('schema', ?)", (SCHEMA_VERSION,))
    if not os.path.isdir(RUNS_DIR):
        db.commit()
        return db
    for rid in sorted(os.listdir(RUNS_DIR)):
        rd = os.path.join(RUNS_DIR, rid)
        meta_p = os.path.join(rd, "meta.json")
        if not os.path.isfile(meta_p):
            continue
        try:
            meta = json.load(open(meta_p))
        except Exception:
            continue
        created = meta.get("start") or os.path.getmtime(meta_p)
        db.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (meta.get("id", rid), meta.get("project"), meta.get("name"),
                    meta.get("sweep"), meta.get("status"),
                    json.dumps(meta.get("config", {})), created,
                    meta.get("host"), meta.get("os"), meta.get("git"), meta.get("notes"),
                    json.dumps(meta.get("tags", [])), meta.get("start", 0),
                    meta.get("duration", 0), json.dumps(meta.get("summary", {})), meta.get("cmd")))
        ev_p = os.path.join(rd, "events.jsonl")
        if not os.path.isfile(ev_p):
            continue
        for line in open(ev_p):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            t = e.get("t")
            if t == "scalar":
                db.execute("INSERT INTO scalars VALUES(?,?,?,?,?)",
                           (rid, e["name"], e["step"], e["value"], e.get("wall", 0)))
            elif t == "histogram":
                db.execute("INSERT INTO hists VALUES(?,?,?,?,?,?)",
                           (rid, e["name"], e["step"], e["min"], e["max"], json.dumps(e["counts"])))
            elif t == "system":
                db.execute("INSERT INTO systemm VALUES(?,?,?,?,?)",
                           (rid, e["step"], e["cpu_pct"], e["rss_mb"], e.get("wall", 0)))
            elif t == "image":
                db.execute("INSERT INTO media VALUES(?,?,?,?,?,?,?,?)",
                           (rid, "image", e["name"], e["step"], e["w"], e["h"], e["path"], None))
            elif t == "table":
                db.execute("INSERT INTO media VALUES(?,?,?,?,?,?,?,?)",
                           (rid, "table", e["name"], e["step"], 0, 0, None, e["csv"]))
            elif t == "artifact":
                db.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)",
                           (rid, e["name"], e.get("atype"), e.get("path"),
                            e.get("size", 0), e.get("hash"), e.get("aliases")))
            elif t == "alert":
                db.execute("INSERT INTO alerts VALUES(?,?,?,?)",
                           (rid, e.get("level"), e.get("message"), e.get("wall", 0)))
            elif t == "curve":
                db.execute("INSERT INTO curves VALUES(?,?,?,?,?,?)",
                           (rid, e["name"], e.get("xlabel"), e.get("ylabel"),
                            json.dumps(e["xs"]), json.dumps(e["ys"])))
            elif t == "richtable":
                db.execute("INSERT INTO richtables VALUES(?,?,?,?)",
                           (rid, e["name"], json.dumps(e["columns"]), json.dumps(e["rows"])))
            elif t == "use_artifact":
                db.execute("INSERT INTO uses VALUES(?,?,?)",
                           (rid, e["name"], e.get("version")))
    db.execute("INSERT INTO _meta VALUES('sig', ?)", (json.dumps(sig),))
    db.commit()
    return db


def get_db():
    global _db, _last_sig
    with _db_lock:
        sig = _scan_signature()
        # Reuse the durable on-disk DB across restarts when runs are unchanged.
        if _db is None and os.path.isfile(DB_PATH):
            try:
                cand = sqlite3.connect(DB_PATH, check_same_thread=False)
                stored = cand.execute("SELECT v FROM _meta WHERE k='sig'").fetchone()
                ver = cand.execute("SELECT v FROM _meta WHERE k='schema'").fetchone()
                # Reuse only if run files unchanged AND schema version matches.
                if stored and stored[0] == json.dumps(sig) and ver and ver[0] == SCHEMA_VERSION:
                    cand.execute("SELECT 1 FROM richtables LIMIT 1")  # sanity probe
                    _db, _last_sig = cand, sig
                    return _db
                cand.close()
            except Exception:
                pass
        if _db is None or sig != _last_sig:
            _db = _build_db(sig)
            _last_sig = sig
        return _db


def q(sql, args=()):
    db = get_db()
    with _db_lock:
        cur = db.execute(sql, args)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


# ── API handlers ────────────────────────────────────────────────────────────
def _run_summary(rid):
    """Last value of every scalar key for a run (used for run table + sidebar)."""
    s = q("""SELECT name, value FROM scalars WHERE run=? AND step=
             (SELECT MAX(step) FROM scalars s2 WHERE s2.run=scalars.run AND s2.name=scalars.name)""",
          (rid,))
    return {x["name"]: x["value"] for x in s}


def api_runs(_):
    rows = q("SELECT * FROM runs ORDER BY created")
    for r in rows:
        r["config"] = json.loads(r["config"] or "{}")
        r["tags"] = json.loads(r["tags"] or "[]")
        r["summary"] = {**json.loads(r["summary"] or "{}"), **_run_summary(r["id"])}
    return rows


def api_overview(qs):
    rid = qs.get("run", [""])[0]
    rows = q("SELECT * FROM runs WHERE id=?", (rid,))
    if not rows:
        return {}
    r = rows[0]
    r["config"] = json.loads(r["config"] or "{}")
    r["tags"] = json.loads(r["tags"] or "[]")
    r["summary"] = {**json.loads(r["summary"] or "{}"), **_run_summary(rid)}
    r["n_scalars"] = q("SELECT COUNT(*) c FROM scalars WHERE run=?", (rid,))[0]["c"]
    r["n_metrics"] = q("SELECT COUNT(DISTINCT name) c FROM scalars WHERE run=?", (rid,))[0]["c"]
    return r


def api_logs(qs):
    rid = qs.get("run", [""])[0]
    p = os.path.join(RUNS_DIR, rid, "console.log")
    return {"text": open(p).read() if os.path.isfile(p) else ""}


def api_table(_):
    """The classic W&B runs table: state, tags, config columns + metric summaries."""
    runs = api_runs(None)
    cfg_keys = sorted({k for r in runs for k in r["config"]})
    met_keys = sorted({k for r in runs for k in r["summary"]})
    return {"runs": runs, "config_keys": cfg_keys, "metric_keys": met_keys}


def api_export(qs):
    rid = qs.get("run", [""])[0]
    keys = [x for x in (qs.get("keys", [""])[0].split(",")) if x] or \
           [x["name"] for x in q("SELECT DISTINCT name FROM scalars WHERE run=?", (rid,))]
    steps = {}
    for k in keys:
        for row in q("SELECT step, value FROM scalars WHERE run=? AND name=? ORDER BY step", (rid, k)):
            steps.setdefault(row["step"], {})[k] = row["value"]
    out = ["step," + ",".join(keys)]
    for st in sorted(steps):
        out.append(str(st) + "," + ",".join(str(steps[st].get(k, "")) for k in keys))
    return "\n".join(out)


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def api_importance(qs):
    """Correlation of each numeric hyperparameter with a target metric across a sweep."""
    sid = qs.get("id", [""])[0]
    target = qs.get("target", ["best_val_accuracy"])[0]
    rows = q("SELECT id, config, summary FROM runs WHERE sweep=?", (sid,))
    runs = []
    for r in rows:
        cfg = json.loads(r["config"] or "{}")
        summ = {**json.loads(r["summary"] or "{}"), **_run_summary(r["id"])}
        runs.append((cfg, summ))
    tvals = [s.get(target) for _, s in runs]
    if any(v is None for v in tvals) or len(runs) < 2:
        return {"target": target, "importance": [], "note": "need >=2 runs with the target metric"}
    keys = sorted({k for c, _ in runs for k in c
                   if isinstance(c.get(k), (int, float)) and k != target})
    out = []
    for k in keys:
        xs = [c[k] for c, _ in runs if isinstance(c.get(k), (int, float))]
        if len(xs) == len(tvals) and len(set(xs)) > 1:
            out.append({"param": k, "correlation": round(_pearson(xs, tvals), 3)})
    out.sort(key=lambda d: -abs(d["correlation"]))
    return {"target": target, "importance": out}


def api_keys(_):
    return [x["name"] for x in q("SELECT DISTINCT name FROM scalars ORDER BY name")]


def api_metrics(qs):
    runs = [x for x in (qs.get("runs", [""])[0].split(",")) if x]
    keys = [x for x in (qs.get("keys", [""])[0].split(",")) if x]
    out = {}
    for rid in runs:
        out[rid] = {}
        for k in keys:
            rows = q("SELECT step, value, wall FROM scalars WHERE run=? AND name=? ORDER BY step", (rid, k))
            out[rid][k] = rows
    return out


def api_histograms(qs):
    rid = qs.get("run", [""])[0]
    name = qs.get("name", [""])[0]
    return q("SELECT step, mn, mx, counts FROM hists WHERE run=? AND name=? ORDER BY step", (rid, name))


def api_histkeys(qs):
    rid = qs.get("run", [""])[0]
    return [x["name"] for x in q("SELECT DISTINCT name FROM hists WHERE run=? ORDER BY name", (rid,))]


def api_system(qs):
    rid = qs.get("run", [""])[0]
    return q("SELECT step, cpu, rss, wall FROM systemm WHERE run=? ORDER BY step", (rid,))


def api_media(qs):
    rid = qs.get("run", [""])[0]
    return q("SELECT kind, name, step, w, h, path, csv FROM media WHERE run=?", (rid,))


def api_artifacts(qs):
    rid = qs.get("run", [""])[0]
    return q("SELECT name, atype, path, size, hash, aliases FROM artifacts WHERE run=?", (rid,))


def api_registry(_):
    """Model registry: artifacts with aliases across all runs, newest first."""
    rows = q("""SELECT a.name, a.atype, a.aliases, a.size, a.hash, a.run, r.name AS run_name, r.start
                FROM artifacts a JOIN runs r ON a.run=r.id
                WHERE a.aliases IS NOT NULL AND a.aliases != '' ORDER BY r.start DESC""")
    return rows


def api_alerts(qs):
    """Alerts for one run, or all runs (with run names) if no run given."""
    rid = qs.get("run", [""])[0]
    if rid:
        return q("SELECT level, message, wall FROM alerts WHERE run=? ORDER BY wall", (rid,))
    return q("""SELECT a.level, a.message, a.wall, a.run, r.name AS run_name
                FROM alerts a JOIN runs r ON a.run=r.id ORDER BY a.wall DESC""")


def api_agg(qs):
    """Per-metric aggregation (last/min/max/mean) for a run's summary."""
    rid = qs.get("run", [""])[0]
    out = {}
    for row in q("""SELECT name, MIN(value) mn, MAX(value) mx, AVG(value) av, COUNT(*) n
                    FROM scalars WHERE run=? GROUP BY name""", (rid,)):
        last = q("""SELECT value FROM scalars WHERE run=? AND name=? ORDER BY step DESC LIMIT 1""",
                 (rid, row["name"]))
        out[row["name"]] = {"last": last[0]["value"] if last else None,
                            "min": row["mn"], "max": row["mx"], "mean": row["av"], "n": row["n"]}
    return out


def api_group(qs):
    """Group runs by a config key; return mean±std of a metric per step, per group."""
    key = qs.get("key", ["train/loss"])[0]
    groupby = qs.get("groupby", ["optimizer"])[0]
    runs = q("SELECT id, config FROM runs")
    groups = {}
    for r in runs:
        cfg = json.loads(r["config"] or "{}")
        g = str(cfg.get(groupby, "?"))
        groups.setdefault(g, []).append(r["id"])
    out = {}
    for g, ids in groups.items():
        by_step = {}
        for rid in ids:
            for row in q("SELECT step, value FROM scalars WHERE run=? AND name=? ORDER BY step", (rid, key)):
                by_step.setdefault(row["step"], []).append(row["value"])
        series = []
        for st in sorted(by_step):
            vals = by_step[st]
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            series.append({"step": st, "mean": m, "std": sd, "n": len(vals)})
        if series:
            out[g] = series
    return {"groupby": groupby, "key": key, "groups": out}


def api_curves(qs):
    rid = qs.get("run", [""])[0]
    rows = q("SELECT name, xlabel, ylabel, xs, ys FROM curves WHERE run=?", (rid,))
    for r in rows:
        r["xs"] = json.loads(r["xs"]); r["ys"] = json.loads(r["ys"])
    return rows


def api_diff(qs):
    """Side-by-side diff of two runs: config + metric summary."""
    a, b = qs.get("a", [""])[0], qs.get("b", [""])[0]
    def load(rid):
        rows = q("SELECT id, name, config, summary FROM runs WHERE id=?", (rid,))
        if not rows:
            return None
        r = rows[0]
        return {"id": r["id"], "name": r["name"],
                "config": json.loads(r["config"] or "{}"),
                "summary": {**json.loads(r["summary"] or "{}"), **_run_summary(rid)}}
    ra, rb = load(a), load(b)
    if not ra or not rb:
        return {"error": "both a and b runs required"}
    def diff(da, db):
        keys = sorted(set(da) | set(db))
        return [{"key": k, "a": da.get(k), "b": db.get(k),
                 "same": json.dumps(da.get(k)) == json.dumps(db.get(k))} for k in keys]
    return {"a": ra, "b": rb, "config_diff": diff(ra["config"], rb["config"]),
            "summary_diff": diff(ra["summary"], rb["summary"])}


def api_richtable(qs):
    rid = qs.get("run", [""])[0]
    name = qs.get("name", [""])[0]
    rows = q("SELECT columns, rows FROM richtables WHERE run=? AND name=?", (rid, name))
    if not rows:
        return {"columns": [], "rows": []}
    return {"columns": json.loads(rows[0]["columns"]), "rows": json.loads(rows[0]["rows"])}


def api_richtable_list(qs):
    rid = qs.get("run", [""])[0]
    return [r["name"] for r in q("SELECT DISTINCT name FROM richtables WHERE run=?", (rid,))]


def api_lineage(_):
    """Artifact lineage DAG: dataset --used_by--> run --produces--> model.

    Only runs that actually participate in lineage (produce or consume an
    artifact) are included — disconnected runs would just clutter the graph.
    Parallel edges between the same run/artifact pair are de-duplicated with a
    count so N runs producing one artifact reads as a single labelled edge.
    """
    CAP = 5  # max individual run nodes shown per artifact; rest collapse to "+N runs"
    run_names = {r["id"]: r["name"] for r in q("SELECT id, name FROM runs")}
    produces, consumes, alias = {}, {}, {}   # artifact -> [run ids]
    for a in q("SELECT run, name, aliases FROM artifacts"):
        if a["run"] in run_names:
            produces.setdefault(a["name"], []).append(a["run"])
            if a["aliases"] and not alias.get(a["name"]):
                alias[a["name"]] = a["aliases"]
    for u in q("SELECT run, artifact FROM uses"):
        if u["run"] in run_names:
            consumes.setdefault(u["artifact"], []).append(u["run"])

    nodes, edges, run_ids = [], [], set()
    def add_run(rid):
        if rid not in run_ids:
            run_ids.add(rid)
            nodes.append({"id": "run:" + rid, "type": "run", "label": run_names[rid]})

    for art in set(produces) | set(consumes):
        nodes.append({"id": "art:" + art, "type": "artifact",
                      "label": art + (f"\n[{alias[art]}]" if alias.get(art) else "")})
        prod = produces.get(art, [])
        for rid in prod[:CAP]:
            add_run(rid); edges.append({"from": "run:" + rid, "to": "art:" + art, "label": "produces"})
        if len(prod) > CAP:
            agg = f"agg:{art}:prod"
            nodes.append({"id": agg, "type": "run", "label": f"+{len(prod) - CAP} runs"})
            edges.append({"from": agg, "to": "art:" + art, "label": "produces"})
        con = consumes.get(art, [])
        for rid in con[:CAP]:
            add_run(rid); edges.append({"from": "art:" + art, "to": "run:" + rid, "label": "uses"})
        if len(con) > CAP:
            agg = f"agg:{art}:con"
            nodes.append({"id": agg, "type": "run", "label": f"+{len(con) - CAP} runs"})
            edges.append({"from": "art:" + art, "to": agg, "label": "uses"})
    return {"nodes": nodes, "edges": edges}


def _files(subdir):
    d = os.path.join(EXP_DIR, subdir)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            try:
                out.append(json.load(open(os.path.join(d, fn))))
            except Exception:
                pass
    return out


def api_jobs(_):
    return sorted(_files("jobs"), key=lambda j: j.get("created", 0))


def api_traces(_):
    out = []
    for t in _files("traces"):
        out.append({"id": t["id"], "name": t["name"], "project": t.get("project"),
                    "total_cost": t.get("total_cost", 0), "tokens_in": t.get("tokens_in", 0),
                    "tokens_out": t.get("tokens_out", 0), "latency_ms": t.get("latency_ms", 0),
                    "spans": len(t.get("spans", [])), "created": t.get("created", 0)})
    return sorted(out, key=lambda x: -x["created"])


def api_trace(qs):
    tid = qs.get("id", [""])[0]
    for t in _files("traces"):
        if t["id"] == tid:
            return t
    return {"error": "not found"}


def api_launch(body):
    """Submit a job to the launch queue (drained by an external agent)."""
    import time as _t
    cmd = body.get("command")
    if not cmd:
        return {"error": "command required"}
    d = os.path.join(EXP_DIR, "jobs")
    os.makedirs(d, exist_ok=True)
    jid = f"{int(_t.time())}-{len(os.listdir(d))}"
    job = {"id": jid, "name": body.get("name", jid), "command": cmd, "config": body.get("config", {}),
           "queue": body.get("queue", "default"), "status": "queued", "created": _t.time(),
           "started": 0, "finished": 0, "exit": None, "output": ""}
    json.dump(job, open(os.path.join(d, jid + ".json"), "w"), indent=2)
    global _last_sig
    _last_sig = None
    return {"ok": True, "job": jid}


def api_update(body):
    """Write-back: rename a run / edit notes / edit tags. body: {run, name?, notes?, tags?}."""
    rid = body.get("run")
    if not rid:
        return {"error": "run required"}
    meta_p = os.path.join(RUNS_DIR, rid, "meta.json")
    if not os.path.isfile(meta_p):
        return {"error": "run not found"}
    meta = json.load(open(meta_p))
    if "name" in body:
        meta["name"] = body["name"]
    if "notes" in body:
        meta["notes"] = body["notes"]
    if "tags" in body:
        meta["tags"] = body["tags"]
    json.dump(meta, open(meta_p, "w"), indent=2)
    global _last_sig
    _last_sig = None  # force re-ingest
    return {"ok": True, "run": rid}


def _safe_run_dir(rid):
    d = os.path.abspath(os.path.join(RUNS_DIR, rid))
    return d if d.startswith(os.path.abspath(RUNS_DIR) + os.sep) and os.path.isdir(d) else None


def api_archive(body):
    rid = body.get("run")
    meta_p = os.path.join(RUNS_DIR, rid or "", "meta.json")
    if not os.path.isfile(meta_p):
        return {"error": "run not found"}
    meta = json.load(open(meta_p))
    meta["status"] = "archived" if meta.get("status") != "archived" else "finished"
    json.dump(meta, open(meta_p, "w"), indent=2)
    global _last_sig
    _last_sig = None
    return {"ok": True, "status": meta["status"]}


def api_delete(body):
    d = _safe_run_dir(body.get("run", ""))
    if not d:
        return {"error": "invalid run"}
    shutil.rmtree(d)
    global _last_sig
    _last_sig = None
    return {"ok": True}


def api_sweep(qs):
    sid = qs.get("id", [""])[0]
    rows = q("SELECT id, name, config FROM runs WHERE sweep=? ORDER BY created", (sid,))
    for r in rows:
        r["config"] = json.loads(r["config"] or "{}")
    return {"runs": rows}


API = {
    "/api/runs": api_runs, "/api/keys": api_keys, "/api/metrics": api_metrics,
    "/api/histograms": api_histograms, "/api/histkeys": api_histkeys,
    "/api/system": api_system, "/api/media": api_media,
    "/api/artifacts": api_artifacts, "/api/sweep": api_sweep,
    "/api/overview": api_overview, "/api/logs": api_logs, "/api/table": api_table,
    "/api/importance": api_importance, "/api/registry": api_registry,
    "/api/export": api_export, "/api/alerts": api_alerts, "/api/agg": api_agg,
    "/api/group": api_group, "/api/curves": api_curves, "/api/diff": api_diff,
    "/api/richtable": api_richtable, "/api/richtable_list": api_richtable_list,
    "/api/lineage": api_lineage, "/api/jobs": api_jobs, "/api/traces": api_traces,
    "/api/trace": api_trace,
}
POST_API = {"/api/update": api_update, "/api/archive": api_archive, "/api/delete": api_delete,
            "/api/launch": api_launch}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        from urllib.parse import urlparse
        path = urlparse(self.path).path
        if path in POST_API:
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                return self._send(POST_API[path](body))
            except Exception as ex:
                return self._send({"error": str(ex)}, code=500)
        self._send({"error": "not found"}, code=404)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        if path in API:
            try:
                return self._send(API[path](qs))
            except Exception as ex:
                return self._send({"error": str(ex)}, code=500)
        if path == "/api/image":
            rid = qs.get("run", [""])[0]
            rel = qs.get("path", [""])[0]
            full = os.path.join(RUNS_DIR, rid, rel)
            if os.path.isfile(full) and os.path.abspath(full).startswith(os.path.abspath(RUNS_DIR)):
                return self._send(open(full, "rb").read(), "application/octet-stream")
            return self._send(b"", "application/octet-stream", 404)
        # static
        if path in ("/", "/index.html"):
            path = "/exp.html"
        fp = os.path.join(HERE, path.lstrip("/"))
        if os.path.isfile(fp):
            ctype = ("text/html" if fp.endswith(".html") else
                     "text/javascript" if fp.endswith(".js") else
                     "text/css" if fp.endswith(".css") else "text/plain")
            return self._send(open(fp, "rb").read(), ctype)
        self._send({"error": "not found"}, code=404)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    get_db()  # warm ingest
    n = len(q("SELECT id FROM runs"))
    print(f"C-ML Experiments — {n} run(s) from {os.path.abspath(RUNS_DIR)}")
    print(f"  http://localhost:{PORT}")
    Server(("", PORT), Handler).serve_forever()
