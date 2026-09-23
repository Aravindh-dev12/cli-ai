package com.clinevo.inbox.agent;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AgentPolicyEngineTest {

    private final AgentPolicyEngine engine = new AgentPolicyEngine();

    @Test
    void supervisedPolicyAlwaysRoutesToHumanReview() {
        var decision = engine.evaluate(0.99, 10, 10);
        assertTrue(decision.humanReviewRequired());
    }

    @Test
    void lowConfidenceIsExplicitlyExplained() {
        AgentPolicyEngine.Decision decision =
            new AgentPolicyEngine().evaluate(0.40, 10, 10);

        assertTrue(decision.humanReviewRequired());
        assertTrue(decision.reason().toLowerCase().contains("human review"));
    }
}
