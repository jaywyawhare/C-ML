"""
cmltrack.launch — job orchestration (prototype, W&B Launch-style).

Submit job specs to a file-based queue; an agent polls the queue and runs each
job as a subprocess (with CML_EXP_DIR set so the job's run is tracked), updating
status queued -> running -> finished/failed. Zero dependencies.

    from cmltrack import launch
    launch.submit("python3 train.py --lr 0.01", name="train", config={"lr": 0.01})
    launch.run_agent()     # drains the queue
"""
import json
import os
import time
import glob
import subprocess
import itertools

_c = itertools.count()


def _dir(root):
    d = os.path.join(root or os.environ.get("CML_EXP_DIR", ".cml/experiments"), "jobs")
    os.makedirs(d, exist_ok=True)
    return d


def _path(root, jid):
    return os.path.join(_dir(root), jid + ".json")


def _save(root, job):
    json.dump(job, open(_path(root, job["id"]), "w"), indent=2)


def submit(command, name=None, config=None, queue="default", root=None):
    jid = f"{int(time.time())}-{next(_c)}"
    job = {"id": jid, "name": name or jid, "command": command, "config": config or {},
           "queue": queue, "status": "queued", "created": time.time(),
           "started": 0, "finished": 0, "exit": None, "output": ""}
    _save(root, job)
    return jid


def list_jobs(root=None):
    return sorted((json.load(open(f)) for f in glob.glob(os.path.join(_dir(root), "*.json"))),
                  key=lambda j: j["created"])


def run_agent(root=None, max_jobs=1000, poll=0.2, once=True):
    """Drain the queue. Returns the number of jobs run. `once=True` stops when empty."""
    ran = 0
    while ran < max_jobs:
        queued = [j for j in list_jobs(root) if j["status"] == "queued"]
        if not queued:
            if once:
                break
            time.sleep(poll)
            continue
        job = queued[0]
        job["status"] = "running"
        job["started"] = time.time()
        _save(root, job)
        env = dict(os.environ)
        if root:
            env["CML_EXP_DIR"] = root
        try:
            p = subprocess.run(job["command"], shell=True, env=env, capture_output=True,
                               text=True, timeout=180)
            job["exit"] = p.returncode
            job["status"] = "finished" if p.returncode == 0 else "failed"
            job["output"] = (p.stdout + p.stderr)[-2000:]
        except Exception as e:
            job["status"] = "failed"
            job["output"] = str(e)
        job["finished"] = time.time()
        _save(root, job)
        ran += 1
    return ran
