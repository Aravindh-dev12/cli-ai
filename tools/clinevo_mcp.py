#!/usr/bin/env python3
"""Minimal dependency-free MCP stdio adapter for the Clinevo control plane."""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("CLINEVO_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("CLINEVO_API_KEY", "")

TOOLS = [
    {"name":"clinevo_status","description":"Read Clinevo agent runtime and queue state.","inputSchema":{"type":"object","properties":{}}},
    {"name":"clinevo_jobs","description":"List recent durable Clinevo agent jobs.","inputSchema":{"type":"object","properties":{}}},
    {"name":"clinevo_events","description":"Read the agent audit trail for one message.","inputSchema":{"type":"object","properties":{"message_id":{"type":"integer"}},"required":["message_id"]}},
    {"name":"clinevo_reconcile","description":"Reconcile durable agent jobs with inbox reality. Control-plane operation; does not finalize cases.","inputSchema":{"type":"object","properties":{}}},
    {"name":"clinevo_pause","description":"Pause autonomous agent advancement. Existing work is retained.","inputSchema":{"type":"object","properties":{}}},
    {"name":"clinevo_resume","description":"Resume autonomous agent advancement.","inputSchema":{"type":"object","properties":{}}},
]

def http(method, path):
    req = urllib.request.Request(BASE + path, method=method)
    if API_KEY:
        req.add_header("X-API-Key", API_KEY)
    with urllib.request.urlopen(req, timeout=15) as response:
        body = response.read()
        return json.loads(body) if body else {"ok": True}

def result_text(value):
    return {"content":[{"type":"text","text":json.dumps(value, indent=2, default=str)}]}

def call_tool(name, args):
    if name == "clinevo_status":
        return result_text(http("GET","/api/agent/status"))
    if name == "clinevo_jobs":
        return result_text(http("GET","/api/agent/jobs"))
    if name == "clinevo_events":
        return result_text(http("GET",f"/api/agent/messages/{int(args['message_id'])}/events"))
    if name == "clinevo_reconcile":
        http("POST","/api/agent/reconcile")
        return result_text({"ok":True,"operation":"reconcile"})
    if name == "clinevo_pause":
        http("POST","/api/agent/pause")
        return result_text({"ok":True,"runtime":"PAUSED"})
    if name == "clinevo_resume":
        http("POST","/api/agent/resume")
        return result_text({"ok":True,"runtime":"RUNNING"})
    raise ValueError(f"unknown tool: {name}")

def reply(request_id, value=None, error=None):
    payload = {"jsonrpc":"2.0","id":request_id}
    payload["error"] = error if error else None
    if error is None:
        payload["result"] = value
        payload.pop("error")
    print(json.dumps(payload), flush=True)

for line in sys.stdin:
    if not line.strip():
        continue
    try:
        request = json.loads(line)
        method = request.get("method")
        request_id = request.get("id")

        if method == "initialize":
            reply(request_id, {
                "protocolVersion": request.get("params",{}).get("protocolVersion","2024-11-05"),
                "capabilities":{"tools":{}},
                "serverInfo":{"name":"clinevo","version":"agent-core-1"}
            })
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            reply(request_id, {"tools":TOOLS})
        elif method == "tools/call":
            params = request.get("params",{})
            reply(request_id, call_tool(params["name"], params.get("arguments",{})))
        elif method == "ping":
            reply(request_id, {})
        else:
            if request_id is not None:
                reply(request_id, error={"code":-32601,"message":f"Method not found: {method}"})
    except urllib.error.HTTPError as exc:
        if request_id is not None:
            reply(request_id, error={"code":exc.code,"message":exc.read().decode("utf-8","replace")})
    except Exception as exc:
        if request_id is not None:
            reply(request_id, error={"code":-32000,"message":str(exc)})
