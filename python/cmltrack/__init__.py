"""
cmltrack — a wandb-style experiment-tracking SDK for C-ML (prototype).

Pure stdlib. Writes the same append-only run format the C logger and
viz/exp_server.py already understand, so Python and C runs show up side by side.

    import cmltrack as ct
    run = ct.init(project="demo", name="run-1", config={"lr": 0.01})
    for step in range(100):
        run.log({"loss": loss, "acc": acc}, step=step)
        run.watch({"layer0.weight": w})        # auto weight/grad histograms
        run.log_system(step)                    # cpu/mem/gpu/disk
    run.log_curve("pr", recall, precision, xlabel="recall", ylabel="precision")
    run.summary["best_acc"] = 0.98
    run.finish()
"""
import json
import math
import os
import sys
import time
import socket
import platform
import itertools

__all__ = ["init", "Run"]

_counter = itertools.count()


def _now():
    return time.time()


def _read_git():
    try:
        head = open(".git/HEAD").read().strip()
        if head.startswith("ref: "):
            return open(os.path.join(".git", head[5:])).read().strip()[:12]
        return head[:12]
    except Exception:
        return None


def _pip_freeze():
    try:
        import importlib.metadata as m
        return sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions())
    except Exception:
        return []


class Run:
    def __init__(self, project, name, config, root, sweep, tags, notes, capture_env):
        rid = f"{int(_now())}-{os.getpid()}-{next(_counter)}"
        self.id = rid
        self.project = project or "default"
        self.name = name or rid
        self.sweep = sweep
        self.config = dict(config or {})
        self.summary = {}
        self.tags = list(tags or [])
        self.notes = notes
        self.status = "running"
        self.start = _now()
        self._step = 0
        self.dir = os.path.join(root, "runs", rid)
        os.makedirs(os.path.join(self.dir, "media"), exist_ok=True)
        self._events = open(os.path.join(self.dir, "events.jsonl"), "w")
        self._console = open(os.path.join(self.dir, "console.log"), "w")
        self.host = socket.gethostname()
        self.os = f"{platform.system()} {platform.release()}"
        self.git = _read_git()
        self.cmd = " ".join(sys.argv)
        self._last_cpu = self._proc_cpu()
        self._last_wall = _now()
        if capture_env:
            reqs = _pip_freeze()
            if reqs:
                # store env as an artifact-like file for reproducibility (L: code/env capture)
                envp = os.path.join(self.dir, "requirements.txt")
                open(envp, "w").write("\n".join(reqs))
                self.config["_env_packages"] = len(reqs)
        self._write_meta()

    # ── event helpers ────────────────────────────────────────────────────
    def _emit(self, obj):
        obj["wall"] = _now()
        self._events.write(json.dumps(obj) + "\n")
        self._events.flush()

    def _write_meta(self):
        done = self.status != "running"
        end = _now() if done else 0.0
        meta = {
            "id": self.id, "project": self.project, "name": self.name, "sweep": self.sweep,
            "status": self.status, "host": self.host, "os": self.os, "git": self.git,
            "cmd": self.cmd, "notes": self.notes, "tags": self.tags, "start": self.start,
            "end": end, "duration": (end - self.start) if done else 0.0,
            "summary": self.summary, "config": self.config,
        }
        json.dump(meta, open(os.path.join(self.dir, "meta.json"), "w"), indent=2)

    # ── logging API (mirrors wandb) ──────────────────────────────────────
    def log(self, metrics, step=None):
        if step is None:
            step = self._step
            self._step += 1
        else:
            self._step = max(self._step, step + 1)
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                self._emit({"t": "scalar", "name": k, "step": step, "value": float(v)})
        return step

    def log_histogram(self, name, values, step, bins=30):
        values = list(values)
        if not values:
            return
        mn, mx = min(values), max(values)
        rng = (mx - mn) or 1e-9
        counts = [0] * bins
        for v in values:
            b = min(bins - 1, max(0, int((v - mn) / rng * bins)))
            counts[b] += 1
        self._emit({"t": "histogram", "name": name, "step": step, "min": mn, "max": mx, "counts": counts})

    def watch(self, named_arrays, step=None, bins=32):
        """Auto-log weight/grad histograms (wandb.watch equivalent)."""
        if step is None:
            step = self._step
        for name, arr in named_arrays.items():
            self.log_histogram(name, arr, step, bins)

    def log_image(self, name, rgb, w, h, step):
        rel = f"media/{name}_{step}.rgb"
        with open(os.path.join(self.dir, rel), "wb") as f:
            f.write(bytes(rgb))
        self._emit({"t": "image", "name": name, "step": step, "w": w, "h": h, "path": rel})

    def log_table(self, name, rows, step):
        csv = "\n".join(",".join(str(c) for c in r) for r in rows)
        self._emit({"t": "table", "name": name, "step": step, "csv": csv})

    def log_rich_table(self, name, columns, rows, step=0):
        """W&B Tables: rows where a cell may be a media dict {"_image": rgb, "w":, "h":}."""
        out = []
        for ri, row in enumerate(rows):
            cells = {}
            for col in columns:
                v = row.get(col)
                if isinstance(v, dict) and "_image" in v:
                    rel = f"media/{name}_{step}_{ri}_{col}.rgb"
                    with open(os.path.join(self.dir, rel), "wb") as f:
                        f.write(bytes(v["_image"]))
                    cells[col] = {"img": rel, "w": v["w"], "h": v["h"]}
                else:
                    cells[col] = v
            out.append(cells)
        self._emit({"t": "richtable", "name": name, "step": step, "columns": columns, "rows": out})

    def use_artifact(self, name, version="latest"):
        """Record an input artifact dependency (produces a lineage edge)."""
        self._emit({"t": "use_artifact", "name": name, "version": version})

    def log_curve(self, name, xs, ys, xlabel=None, ylabel=None):
        self._emit({"t": "curve", "name": name, "xlabel": xlabel, "ylabel": ylabel,
                    "xs": [round(float(x), 5) for x in xs], "ys": [round(float(y), 5) for y in ys]})

    def log_artifact(self, name, path, type="model", aliases=None):
        size, h = 0, 1469598103934665603
        try:
            with open(path, "rb") as f:
                for ch in f.read():
                    h ^= ch; h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF; size += 1
        except Exception:
            pass
        self._emit({"t": "artifact", "name": name, "atype": type, "path": path,
                    "aliases": aliases, "size": size, "hash": f"{h:016x}"})

    def alert(self, level, message):
        self._emit({"t": "alert", "level": level, "message": message})
        self._console.write(f"[alert:{level}] {message}\n"); self._console.flush()

    def log_console(self, line):
        self._console.write(line + "\n"); self._console.flush()

    # ── system + GPU + disk telemetry (best-effort /proc + /sys) ──────────
    def _proc_cpu(self):
        try:
            parts = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
            return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")
        except Exception:
            return 0.0

    def _gpu_busy(self):
        import glob
        for p in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"):
            try:
                return float(open(p).read().strip())
            except Exception:
                pass
        return None

    def log_system(self, step):
        rss = 0.0
        try:
            for line in open("/proc/self/status"):
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024.0
                    break
        except Exception:
            pass
        cpu = self._proc_cpu()
        wall = _now()
        dw = wall - self._last_wall
        cpu_pct = (cpu - self._last_cpu) / dw * 100.0 if dw > 1e-6 else 0.0
        self._last_cpu, self._last_wall = cpu, wall
        ev = {"t": "system", "step": step, "cpu_pct": round(cpu_pct, 2), "rss_mb": round(rss, 2)}
        gpu = self._gpu_busy()
        if gpu is not None:
            self._emit({"t": "scalar", "name": "sys/gpu_busy_pct", "step": step, "value": gpu})
        self._emit(ev)

    # ── config / meta mutation ───────────────────────────────────────────
    def set_tag(self, tag):
        if tag not in self.tags:
            self.tags.append(tag); self._write_meta()

    def set_notes(self, notes):
        self.notes = notes; self._write_meta()

    def finish(self, status="finished"):
        self.status = status
        # flush any summary the user set
        self._write_meta()
        self._events.close(); self._console.close()


def init(project=None, name=None, config=None, dir=None, sweep=None, tags=None,
         notes=None, capture_env=True):
    root = dir or os.environ.get("CML_EXP_DIR", ".cml/experiments")
    return Run(project, name, config, root, sweep, tags, notes, capture_env)
