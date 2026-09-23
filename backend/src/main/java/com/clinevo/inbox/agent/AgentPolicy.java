package com.clinevo.inbox.agent;

public record AgentPolicy(
        String version,
        double minimumConfidence,
        boolean requireEvidence,
        boolean alwaysHumanReview
) {
    public static AgentPolicy defaultPolicy() {
        return new AgentPolicy("pv-1", 0.85, true, true);
    }
}
