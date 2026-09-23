#!/usr/bin/env python3
"""Dependency-free terminal dashboard for the Clinevo agent."""
import curses
import json
import os
import time
import urllib.request

BASE = os.environ.get("CLINEVO_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("CLINEVO_API_KEY", "")

def get(path):
    req = urllib.request.Request(BASE + path)
    if API_KEY:
        req.add_header("X-API-Key", API_KEY)
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())

def post(path):
    req = urllib.request.Request(BASE + path, method="POST")
    if API_KEY:
        req.add_header("X-API-Key", API_KEY)
    with urllib.request.urlopen(req, timeout=5):
        return

def draw(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    while True:
        try:
            status = get("/api/agent/status")
            jobs = get("/api/agent/jobs")
        except Exception as exc:
            status = {"runtime":"UNAVAILABLE","error":str(exc)}
            jobs = []

        stdscr.erase()
        stdscr.addstr(0, 0, "CLINEVO AGENT TUI  |  p=pause  r=resume  q=quit  R=reconcile")
        stdscr.addstr(2, 0, f"Runtime: {status.get('runtime')}   Policy: {status.get('policyVersion')}")
        labels = ["observed","analyzing","waitingReview","finalized","failed"]
        stdscr.addstr(3, 0, "Queue: " + "  ".join(f"{k}={status.get(k,0)}" for k in labels))
        stdscr.addstr(5, 0, "Recent jobs:")
        for i, job in enumerate(jobs[:15], start=6):
            text = f"{job.get('ID')}  msg={job.get('MESSAGE_ID')}  {job.get('STATE')}  attempts={job.get('ATTEMPT_COUNT')}"
            stdscr.addstr(i, 0, text[:max(1, curses.COLS-1)])
        stdscr.refresh()

        try:
            key = stdscr.getch()
            if key == ord("q"):
                return
            if key == ord("p"):
                post("/api/agent/pause")
            elif key == ord("r"):
                post("/api/agent/resume")
            elif key == ord("R"):
                post("/api/agent/reconcile")
        except Exception:
            pass
        time.sleep(2)

if __name__ == "__main__":
    curses.wrapper(draw)
