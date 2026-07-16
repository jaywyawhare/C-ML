#!/usr/bin/env python3
"""Submit a batch of jobs to the Launch queue and drain it with an agent.
Each job runs _launch_job.py (a tracked training) as a subprocess."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from cmltrack import launch

job = os.path.join(HERE, "_launch_job.py")
for lr in (0.005, 0.02, 0.08):
    jid = launch.submit(f"python3 {job} {lr}", name=f"train-lr{lr}", config={"lr": lr})
    print(f"submitted job {jid}  (lr={lr})")

print("\ndraining queue with agent...")
n = launch.run_agent()
print(f"agent ran {n} jobs")
for j in launch.list_jobs():
    print(f"  {j['name']:14s} {j['status']:9s} exit={j['exit']}")
print("\nView: python3 viz/serve.py  ->  http://localhost:6969  (Launch tab)")
