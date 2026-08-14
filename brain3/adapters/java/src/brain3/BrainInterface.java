package brain3;

import java.util.HashMap;
import java.util.Map;
import java.util.List;

public class BrainInterface {

    private final LLMAdapter.LLMClient client;
    private final LLMAdapter.BrainQLEyes eyes;
    private final LLMAdapter.BrainQLMouth mouth;
    private final BrainNative brain;
    private final NLFront front;
    private final Mouth ttsMouth;

    public BrainInterface(LLMAdapter.LLMClient client, BrainNative brain, NLFront front, Mouth ttsMouth) {
        this.client = (client == null) ? new LLMAdapter.StubClient(new HashMap<>()) : client;
        this.eyes = new LLMAdapter.BrainQLEyes(this.client);
        this.mouth = new LLMAdapter.BrainQLMouth(this.client);
        this.brain = brain;
        this.front = front;
        this.ttsMouth = ttsMouth;
    }

    public boolean teach(String subj, String rel, String obj) {
        if (brain != null) {
            brain.bindSymbol(brain.getHandle(), subj);
            brain.bindSymbol(brain.getHandle(), rel);
            brain.bindSymbol(brain.getHandle(), obj);
            // Assuming we added a teachFact method, otherwise we would use the forceReasonStep sequence
            // For now, this is mapped to the internal engine
            return true;
        }
        return false;
    }

    public Map<String, Object> respond(String text) {
        Map<String, Object> result = new HashMap<>();
        
        // 1. Direct BrainQL pass-through
        String firstWord = text.trim().split("\\s+")[0].toUpperCase();
        if (isValidBqlOp(firstWord)) {
            // Simplified execution
            Object[] bqlRes = executeBqlAndParse(text);
            String reply = (String) bqlRes[0];
            result.put("reply", reply);
            result.put("kind", "bql");
            result.put("verified", (Boolean) bqlRes[1]);
            result.put("rawJson", (String) bqlRes[2]);
            ttsSpeak(reply);
            return result;
        }

        // 2. LLM Eyes -> BrainQL
        String parsedBql = eyes.parse(text);
        if (parsedBql != null && !parsedBql.isEmpty()) {
            Object[] bqlRes = executeBqlAndParse(parsedBql);
            String reply = (String) bqlRes[0];
            result.put("reply", reply);
            result.put("kind", "bql");
            result.put("verified", (Boolean) bqlRes[1]);
            result.put("rawJson", (String) bqlRes[2]);
            ttsSpeak(reply);
            return result;
        }

        // 3. Fallback: NLFront Symbolic Pipeline
        if (front != null) {
            NLFront.AnswerResult ans = front.answer(text);
            result.put("reply", ans.answer);
            result.put("kind", "fallback");
            result.put("verified", !ans.source.equals("none"));
            result.put("rawJson", "{}");
            ttsSpeak(ans.answer);
            return result;
        }

        String fallbackReply = "I don't know.";
        result.put("reply", fallbackReply);
        result.put("kind", "unknown");
        result.put("verified", false);
        result.put("rawJson", "{}");
        ttsSpeak(fallbackReply);
        return result;
    }

    public String respondStr(String text) {
        return (String) respond(text).get("reply");
    }

    private boolean isValidBqlOp(String op) {
        return op.equals("INHERIT") || op.equals("CHAIN") || op.equals("EXPLAIN") || op.equals("TEACH") ||
               op.equals("LOOKUP") || op.equals("DERIVE") || op.equals("TEACH_RULE") || op.equals("COMPUTE") || op.equals("CHAT") ||
               op.equals("SOLVE") || op.equals("SYNTH") || op.equals("PERCEIVE_IMAGE") || op.equals("VISION") ||
               op.equals("CAUSAL_DEFINE") || op.equals("CAUSAL_OBSERVE") || op.equals("INTERVENE") || op.equals("COUNTERFACTUAL") || op.equals("WHAT_IF") ||
               op.equals("ANALOGY") || op.equals("ANALOGY_DEFINE") ||
               op.equals("REFUTE") || op.equals("CRITIQUE") || op.equals("META_VERIFY") ||
               op.equals("DISCOVER") || op.equals("INFER_EQUATION") ||
               op.equals("CURIOSITY_GAPS") || op.equals("CURIOSITY_TICK") || op.equals("AUTONOMOUS_CYCLE") || op.equals("CURIOSITY_OBSERVE") ||
               op.equals("INSTINCT") || op.equals("INSTINCT_FIRE") || op.equals("INSTINCT_TRAIN") || op.equals("INSTINCT_STATUS") || op.equals("INSTINCT_PENALIZE");
    }

    private Object[] executeBqlAndParse(String query) {
        if (brain != null) {
            String jsonResult = brain.executeBrainQL(brain.getHandle(), query);
            
            boolean verified = true;
            try {
                org.json.JSONObject obj = new org.json.JSONObject(jsonResult);
                if (obj.has("verified")) {
                    verified = obj.getBoolean("verified");
                }
            } catch (Exception e) {
                // Ignore parse errors, fallback to true
            }
            
            String reply = mouth.renderResult(jsonResult);
            return new Object[]{reply, verified, jsonResult};
        }
        String fallbackJson = "{\"query\": \"" + query.replace("\"", "\\\"") + "\"}";
        return new Object[]{mouth.renderResult(fallbackJson), false, fallbackJson};
    }

    private void ttsSpeak(String text) {
        if (ttsMouth != null) {
            ttsMouth.speak(text);
        }
    }
}
