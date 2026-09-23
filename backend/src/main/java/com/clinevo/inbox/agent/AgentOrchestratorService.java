package com.clinevo.inbox.agent;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class AgentOrchestratorService {
    private final JdbcTemplate jdbc;
    private final AgentPolicyEngine policyEngine;

    public AgentOrchestratorService(JdbcTemplate jdbc, AgentPolicyEngine policyEngine) {
        this.jdbc = jdbc;
        this.policyEngine = policyEngine;
    }

    @Scheduled(fixedDelayString = "${clinevo.agent-poll-ms:2000}")
    public void tick() {
        if (runtimeStatus() == AgentRuntimeStatus.PAUSED) return;
        observeMessages();
        advanceJobs();
        reconcileReviewOutcomes();
    }

    @Transactional
    public void observeMessages() {
        List<Long> ids = jdbc.query(
            "SELECT ID FROM INBOX_MESSAGE WHERE STATUS IN ('QUEUED','PROCESSING','READY_FOR_REVIEW','RETRY_QUEUED','PROCESSING_FAILED') ORDER BY ID FETCH FIRST 50 ROWS ONLY",
            (rs, rowNum) -> rs.getLong(1));

        for (Long messageId : ids) {
            Integer count = jdbc.queryForObject(
                "SELECT COUNT(*) FROM AGENT_JOB WHERE MESSAGE_ID=?", Integer.class, messageId);
            if (count != null && count > 0) continue;

            long jobId = nextId("AGENT_JOB_SEQ");
            jdbc.update(
                "INSERT INTO AGENT_JOB (ID,MESSAGE_ID,STATE,ATTEMPT_COUNT,POLICY_VERSION,CREATED_AT,UPDATED_AT) VALUES (?,?,?,?,?,?,?)",
                jobId, messageId, AgentState.OBSERVED.name(), 0,
                AgentPolicy.defaultPolicy().version(), Instant.now(), Instant.now());

            event(jobId, messageId, "JOB_OBSERVED", null,
                AgentState.OBSERVED.name(), "SYSTEM", "agent observed inbox message");
        }
    }

    @Transactional
    public void advanceJobs() {
        List<Job> jobs = jdbc.query(
            "SELECT ID,MESSAGE_ID,STATE,ATTEMPT_COUNT FROM AGENT_JOB " +
            "WHERE STATE IN ('OBSERVED','ANALYZING') ORDER BY CREATED_AT FETCH FIRST 20 ROWS ONLY",
            (rs, rowNum) -> new Job(
                rs.getLong("ID"), rs.getLong("MESSAGE_ID"),
                AgentState.valueOf(rs.getString("STATE")), rs.getInt("ATTEMPT_COUNT")));

        for (Job job : jobs) {
            try {
                advance(job);
            } catch (Exception ex) {
                fail(job, ex);
            }
        }
    }

    @Transactional
    public void advance(long jobId) {
        Job job = jdbc.queryForObject(
            "SELECT ID,MESSAGE_ID,STATE,ATTEMPT_COUNT FROM AGENT_JOB WHERE ID=?",
            (rs, rowNum) -> new Job(
                rs.getLong("ID"), rs.getLong("MESSAGE_ID"),
                AgentState.valueOf(rs.getString("STATE")), rs.getInt("ATTEMPT_COUNT")),
            jobId);
        if (job != null) advance(job);
    }

    private void advance(Job job) {
        String messageStatus = jdbc.queryForObject(
            "SELECT STATUS FROM INBOX_MESSAGE WHERE ID=?", String.class, job.messageId());

        if ("PROCESSING_FAILED".equals(messageStatus)) {
            fail(job, new IllegalStateException("upstream inbox processing failed"));
            return;
        }

        if (job.state() == AgentState.OBSERVED) {
            transition(job, AgentState.ANALYZING, "ANALYSIS_STARTED",
                "agent began policy and evidence analysis");
            job = new Job(job.id(), job.messageId(), AgentState.ANALYZING, job.attemptCount() + 1);
        }

        if (!"READY_FOR_REVIEW".equals(messageStatus)) return;

        double confidence = jdbc.query(
            "SELECT CONFIDENCE FROM CLASSIFICATION WHERE MESSAGE_ID=? ORDER BY CONFIDENCE DESC",
            (rs, rowNum) -> rs.getDouble(1), job.messageId())
            .stream().findFirst().orElse(0.0);

        long totalFacts = scalar(
            "SELECT COUNT(*) FROM EXTRACTED_FACT WHERE MESSAGE_ID=?", job.messageId());
        long sourcedFacts = scalar(
            "SELECT COUNT(*) FROM EXTRACTED_FACT WHERE MESSAGE_ID=? " +
            "AND CONFIDENCE > 0 AND EVIDENCE_TEXT IS NOT NULL", job.messageId());

        AgentPolicyEngine.Decision decision =
            policyEngine.evaluate(confidence, sourcedFacts, totalFacts);

        transition(job, AgentState.WAITING_REVIEW, "REVIEW_REQUIRED", decision.reason());
    }

    @Transactional
    public void reconcileReviewOutcomes() {
        List<Job> jobs = jdbc.query(
            "SELECT ID,MESSAGE_ID,STATE,ATTEMPT_COUNT FROM AGENT_JOB " +
            "WHERE STATE='WAITING_REVIEW' ORDER BY UPDATED_AT FETCH FIRST 50 ROWS ONLY",
            (rs, rowNum) -> new Job(
                rs.getLong("ID"), rs.getLong("MESSAGE_ID"),
                AgentState.valueOf(rs.getString("STATE")), rs.getInt("ATTEMPT_COUNT")));

        for (Job job : jobs) {
            Long reviewCount = jdbc.queryForObject(
                "SELECT COUNT(*) FROM AUDIT_EVENT WHERE MESSAGE_ID=? " +
                "AND EVENT_TYPE IN ('REVIEW_ACCEPTED','REVIEW_OVERRIDDEN')",
                Long.class, job.messageId());
            if (reviewCount != null && reviewCount > 0) {
                transition(job, AgentState.FINALIZED, "REVIEW_COMPLETED",
                    "existing human review workflow reached a terminal decision");
                jdbc.update(
                    "UPDATE AGENT_JOB SET COMPLETED_AT=?,UPDATED_AT=? WHERE ID=?",
                    Instant.now(), Instant.now(), job.id());
            }
        }
    }
    private void fail(Job job, Exception ex) {
        String message = ex.getMessage() == null
            ? ex.getClass().getSimpleName() : ex.getMessage();

        jdbc.update(
            "UPDATE AGENT_JOB SET STATE='FAILED',ATTEMPT_COUNT=ATTEMPT_COUNT+1," +
            "LAST_ERROR=?,UPDATED_AT=?,COMPLETED_AT=? WHERE ID=?",
            truncate(message), Instant.now(), Instant.now(), job.id());

        event(job.id(), job.messageId(), "JOB_FAILED",
            job.state().name(), AgentState.FAILED.name(), "SYSTEM", message);
    }

    private void transition(Job job, AgentState to, String type, String detail) {
        jdbc.update(
            "UPDATE AGENT_JOB SET STATE=?,ATTEMPT_COUNT=ATTEMPT_COUNT+1,UPDATED_AT=?," +
            "STARTED_AT=COALESCE(STARTED_AT,?) WHERE ID=?",
            to.name(), Instant.now(), Instant.now(), job.id());

        event(job.id(), job.messageId(), type, job.state().name(),
            to.name(), "SYSTEM", detail);
    }

    public AgentRuntimeStatus runtimeStatus() {
        String status = jdbc.queryForObject(
            "SELECT STATUS FROM AGENT_RUNTIME_STATE WHERE ID=1", String.class);
        return AgentRuntimeStatus.valueOf(status);
    }

    public void setRuntimeStatus(AgentRuntimeStatus status, String actor) {
        jdbc.update(
            "UPDATE AGENT_RUNTIME_STATE SET STATUS=?,UPDATED_AT=?,UPDATED_BY=? WHERE ID=1",
            status.name(), Instant.now(), actor == null ? "api" : actor);
    }

    public Map<String, Object> status() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("runtime", runtimeStatus().name());
        result.put("policyVersion", AgentPolicy.defaultPolicy().version());
        result.put("observed", scalar("SELECT COUNT(*) FROM AGENT_JOB WHERE STATE='OBSERVED'"));
        result.put("analyzing", scalar("SELECT COUNT(*) FROM AGENT_JOB WHERE STATE='ANALYZING'"));
        result.put("waitingReview", scalar("SELECT COUNT(*) FROM AGENT_JOB WHERE STATE='WAITING_REVIEW'"));
        result.put("finalized", scalar("SELECT COUNT(*) FROM AGENT_JOB WHERE STATE='FINALIZED'"));
        result.put("failed", scalar("SELECT COUNT(*) FROM AGENT_JOB WHERE STATE='FAILED'"));
        return result;
    }

    public List<Map<String, Object>> jobs() {
        return jdbc.queryForList(
            "SELECT ID,MESSAGE_ID,STATE,ATTEMPT_COUNT,LAST_ERROR,POLICY_VERSION," +
            "CREATED_AT,UPDATED_AT FROM AGENT_JOB ORDER BY UPDATED_AT DESC FETCH FIRST 100 ROWS ONLY");
    }

    public List<Map<String, Object>> events(long messageId) {
        return jdbc.queryForList(
            "SELECT ID,JOB_ID,MESSAGE_ID,EVENT_TYPE,FROM_STATE,TO_STATE,ACTOR_TYPE," +
            "ACTOR_ID,DETAILS_JSON,CREATED_AT FROM AGENT_EVENT " +
            "WHERE MESSAGE_ID=? ORDER BY CREATED_AT", messageId);
    }

    @Transactional
    public void reconcile() {
        observeMessages();

        jdbc.update(
            "UPDATE AGENT_JOB j SET STATE='FAILED',LAST_ERROR='message no longer exists'," +
            "UPDATED_AT=?,COMPLETED_AT=? WHERE NOT EXISTS " +
            "(SELECT 1 FROM INBOX_MESSAGE m WHERE m.ID=j.MESSAGE_ID) " +
            "AND j.STATE NOT IN ('FINALIZED','FAILED')",
            Instant.now(), Instant.now());
    }

    private void event(long jobId, long messageId, String type, String from,
                       String to, String actor, String detail) {
        jdbc.update(
            "INSERT INTO AGENT_EVENT " +
            "(ID,JOB_ID,MESSAGE_ID,EVENT_TYPE,FROM_STATE,TO_STATE,ACTOR_TYPE,ACTOR_ID,DETAILS_JSON,CREATED_AT) " +
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            nextId("AGENT_EVENT_SEQ"), jobId, messageId, type, from, to,
            actor, "clinevo-agent",
            "{\"detail\":\"" + escape(detail) + "\"}", Instant.now());
    }

    private long scalar(String sql, Object... args) {
        Number n = jdbc.queryForObject(sql, Number.class, args);
        return n == null ? 0 : n.longValue();
    }

    private long nextId(String sequence) {
        Number n = jdbc.queryForObject(
            "SELECT " + sequence + ".NEXTVAL FROM DUAL", Number.class);
        if (n == null) throw new IllegalStateException("sequence returned null");
        return n.longValue();
    }

    private String truncate(String value) {
        return value.length() <= 1900 ? value : value.substring(0, 1900);
    }

    private String escape(String value) {
        return value.replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r");
    }

    private record Job(long id, long messageId, AgentState state, int attemptCount) {}
}
