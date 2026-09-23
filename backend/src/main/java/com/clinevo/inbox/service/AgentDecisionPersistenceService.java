package com.clinevo.inbox.service;

import com.fasterxml.jackson.databind.JsonNode;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.time.Instant;

@Service
public class AgentDecisionPersistenceService {

    private final JdbcTemplate jdbc;

    public AgentDecisionPersistenceService(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public void persist(long messageId, Long attachmentId, JsonNode aiResult) {
        JsonNode decision = aiResult == null ? null : aiResult.get("decision");
        if (decision == null || decision.isNull() || decision.isMissingNode()) {
            return;
        }

        jdbc.update("""
                INSERT INTO AGENT_DECISION
                  (ID, MESSAGE_ID, ATTACHMENT_ID, PROVIDER, MODEL, LATENCY_MS,
                   FALLBACK_USED, FALLBACK_REASON, ROUTING_JSON, ANSWERS_JSON,
                   DECISION_JSON, CREATED_AT)
                VALUES
                  (AGENT_DECISION_SEQ.NEXTVAL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                messageId,
                attachmentId,
                limit(decision.path("provider").asText("unknown"), 64),
                limit(decision.path("model").asText("unknown"), 255),
                decision.path("latency_ms").asLong(0),
                Boolean.toString(decision.path("fallback_used").asBoolean(false)),
                limit(decision.path("fallback_reason").asText(""), 1900),
                decision.has("model_routing") ? decision.get("model_routing").toString() : "{}",
                decision.has("answers") ? decision.get("answers").toString() : "{}",
                decision.toString(),
                Instant.now());
    }

    private String limit(String value, int max) {
        if (value == null) return null;
        return value.length() <= max ? value : value.substring(0, max);
    }
}
