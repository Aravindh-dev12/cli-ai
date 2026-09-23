package com.clinevo.inbox.agent;

import org.springframework.stereotype.Component;

@Component
public class AgentPolicyEngine {

    public Decision evaluate(double maxClassificationConfidence, long sourcedFacts, long totalFacts) {
        AgentPolicy policy = AgentPolicy.defaultPolicy();

        if (policy.alwaysHumanReview()) {
            return new Decision(true, "Human review is mandatory for the current supervised policy.");
        }
        if (maxClassificationConfidence < policy.minimumConfidence()) {
            return new Decision(true, "Classification confidence is below the policy threshold.");
        }
        if (policy.requireEvidence() && (totalFacts == 0 || sourcedFacts < totalFacts)) {
            return new Decision(true, "One or more extracted facts lack validated source evidence.");
        }
        return new Decision(false, "Policy permits automated finalization.");
    }

    public record Decision(boolean humanReviewRequired, String reason) {}
}
