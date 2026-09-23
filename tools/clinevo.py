#!/usr/bin/env python3
"""Small operator CLI for the Clinevo supervised agent control plane.

The CLI intentionally talks only to the documented /api/agent endpoints.
It never exposes patient/document contents and never performs a reviewer
decision.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("CLINEVO_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("CLINEVO_API_KEY", "")

def request(method, path):
    req = urllib.request.Request(BASE + path, method=method)
    if API_KEY:
        req.add_header("X-API-Key", API_KEY)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            raw = response.read()
            if not raw:
                return None
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        print(f"clinevo: HTTP {exc.code}: {body}", file=sys.stderr)
        raise SystemExit(1)
    except urllib.error.URLError as exc:
        print(f"clinevo: cannot reach {BASE}: {exc.reason}", file=sys.stderr)
        raise SystemExit(1)

def main():
    parser = argparse.ArgumentParser(prog="clinevo", description="Clinevo agent operator CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("jobs")
    events = sub.add_parser("events")
    events.add_argument("message_id", type=int)
    sub.add_parser("pause")
    sub.add_parser("resume")
    sub.add_parser("reconcile")
    advance = sub.add_parser("advance")
    advance.add_argument("job_id", type=int)
    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(request("GET", "/api/agent/status"), indent=2, default=str))
    elif args.command == "jobs":
        print(json.dumps(request("GET", "/api/agent/jobs"), indent=2, default=str))
    elif args.command == "events":
        print(json.dumps(request("GET", f"/api/agent/messages/{args.message_id}/events"), indent=2, default=str))
    elif args.command == "pause":
        request("POST", "/api/agent/pause")
        print("Clinevo agent paused.")
    elif args.command == "resume":
        request("POST", "/api/agent/resume")
        print("Clinevo agent resumed.")
    elif args.command == "reconcile":
        request("POST", "/api/agent/reconcile")
        print("Clinevo reconciliation completed.")
    elif args.command == "advance":
        request("POST", f"/api/agent/jobs/{args.job_id}/advance")
        print(f"Agent job {args.job_id} advanced.")

if __name__ == "__main__":
    main()
