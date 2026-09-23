package com.clinevo.inbox.agent;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@ConditionalOnProperty(name = "clinevo.agent.enabled", havingValue = "true", matchIfMissing = true)
@RequestMapping("/api/agent")
public class AgentController {
    private final AgentOrchestratorService agent;

    public AgentController(AgentOrchestratorService agent) {
        this.agent = agent;
    }

    @GetMapping("/status")
    public Map<String, Object> status() {
        return agent.status();
    }

    @GetMapping("/jobs")
    public List<Map<String, Object>> jobs() {
        return agent.jobs();
    }

    @GetMapping("/messages/{messageId}/events")
    public List<Map<String, Object>> events(@PathVariable long messageId) {
        return agent.events(messageId);
    }

    @PostMapping("/pause")
    public ResponseEntity<Void> pause() {
        agent.setRuntimeStatus(AgentRuntimeStatus.PAUSED, "api");
        return ResponseEntity.noContent().build();
    }

    @PostMapping("/resume")
    public ResponseEntity<Void> resume() {
        agent.setRuntimeStatus(AgentRuntimeStatus.RUNNING, "api");
        return ResponseEntity.noContent().build();
    }

    @PostMapping("/reconcile")
    public ResponseEntity<Void> reconcile() {
        agent.reconcile();
        return ResponseEntity.noContent().build();
    }

    @PostMapping("/jobs/{jobId}/advance")
    public ResponseEntity<Void> advance(@PathVariable long jobId) {
        agent.advance(jobId);
        return ResponseEntity.noContent().build();
    }
}
