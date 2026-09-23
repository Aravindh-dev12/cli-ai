# Clinevo Agent Core

## Purpose

The agent layer turns Clinevo from a request/response application into a durable, supervised workflow engine.

It does not give an LLM unrestricted access to Oracle or the reviewer workflow. The agent observes existing inbox processing, records durable intent, evaluates policy, and emits auditable transitions.

## Planes

### Control plane

- agent runtime state
- durable agent jobs
- state transitions
- pause/resume
- reconciliation
- operator CLI/TUI
- MCP tools

### Data plane

- mailbox intake
- PDF/OCR/document processing
- deterministic or structured-LLM processing
- existing durable PROCESSING_JOB

### Evidence plane

- extracted facts
- source page/evidence
- confidence
- classification
- reviewer action

The agent only uses the evidence already persisted by the document/AI pipeline; it does not manufacture provenance.

## State machine

~~~
OBSERVED
   |
   v
ANALYZING
   |
   +---- upstream failure ----> FAILED
   |
   v
WAITING_REVIEW
   |
   | existing reviewer audit event
   v
FINALIZED
~~~

FINALIZED is a terminal orchestration state, not permission to transmit a regulatory submission. The current implementation deliberately leaves external regulatory submission out of the agent.

## Invariants

1. Every observed message gets at most one durable AGENT_JOB.
2. Every agent transition emits an AGENT_EVENT.
3. A processing failure becomes an explicit FAILED state.
4. Human review remains mandatory under policy pv-1.
5. The agent never edits the AI classification or extracted facts directly.
6. Reconciliation is idempotent.
7. Pausing the agent retains all durable work.

## Persistence

AGENT_RUNTIME_STATE is a singleton runtime-control record.

AGENT_JOB stores one durable orchestration record per inbox message.

AGENT_EVENT is append-only operational history.

The existing PROCESSING_JOB, CLASSIFICATION, EXTRACTED_FACT, AUDIT_EVENT, ATTACHMENT, and inbox tables remain the source of truth for document processing and reviewer decisions.

## Operator interfaces

### CLI

~~~
python tools/clinevo.py status
python tools/clinevo.py jobs
python tools/clinevo.py events 123
python tools/clinevo.py pause
python tools/clinevo.py resume
python tools/clinevo.py reconcile
~~~

### TUI

~~~
python tools/clinevo_tui.py
~~~

Keys:

- p pause
- r resume
- R reconcile
- q quit

### MCP

~~~
{
  "mcpServers": {
    "clinevo": {
      "command": "python",
      "args": ["tools/clinevo_mcp.py"]
    }
  }
}
~~~

The MCP adapter exposes read-oriented status/jobs/events plus pause/resume/reconcile control operations. It does not expose a tool for accepting or overriding a pharmacovigilance decision.

## Future extensions

The durable control plane is intentionally designed so the following can be added without changing the reviewer UI contract:

- inbox-provider adapters
- model/runtime registry
- versioned prompt registry
- explicit provenance validator
- configurable policy records
- escalation queues
- dead-letter jobs
- distributed worker leases
- OpenTelemetry
- object-store document providers
- domain-specific case state machines
- reviewer assignment and SLA timers
