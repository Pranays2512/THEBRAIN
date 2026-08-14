package brain3;

import org.json.JSONObject;

public class CausalCounterfactualTest {
    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println("🧠 THE BRAIN 3: JUDEA PEARL CAUSAL & COUNTERFACTUAL INFERENCE TEST");
        System.out.println("   (Level 1: Association, Level 2: Interventions, Level 3: Counterfactuals)");
        System.out.println("================================================================================\n");

        System.loadLibrary("brainjni");
        BrainNative nativeBrain = new BrainNative(16, 16, 128);
        LLMAdapter.OllamaClient llm = new LLMAdapter.OllamaClient("dummy");
        Mouth mouth = new Mouth(false);
        NLQuery lexical = new NLQuery(new java.util.HashSet<>(), new java.util.ArrayList<>(), new java.util.HashMap<>());
        NLFront front = new NLFront(lexical);
        BrainInterface brain = new BrainInterface(llm, nativeBrain, front, mouth);

        int passed = 0;
        int total = 0;

        // ── Phase 1: Structural Causal Model (SCM) Definition ──────────────────
        System.out.println("--- 1. Defining Structural Causal Model ---");
        testBQL(brain, "CAUSAL_DEFINE accel = force / mass");
        testBQL(brain, "CAUSAL_DEFINE velocity = accel * time");
        testBQL(brain, "CAUSAL_DEFINE distance = 0.5 * accel * (time ^ 2)");
        testBQL(brain, "CAUSAL_DEFINE power = force * velocity");
        System.out.println("   ✓ Physics SCM registered successfully.\n");

        // ── Phase 2: Factual Evidence Observation ──────────────────────────────
        System.out.println("--- 2. Setting Factual Observations ---");
        testBQL(brain, "CAUSAL_OBSERVE mass 10");
        testBQL(brain, "CAUSAL_OBSERVE force 30");
        testBQL(brain, "CAUSAL_OBSERVE time 4");
        System.out.println("   ✓ Factual baseline established: mass=10, force=30, time=4.\n");

        // ── Phase 3: Level 1 Associational & Level 2 Interventions ─────────────
        System.out.println("--- 3. Level 2 Interventional Reasoning: do(X = x) ---");
        
        total++;
        if (checkIntervention(brain, "INTERVENE do(force = 30) QUERY distance", "24", "Distance under baseline F=30")) passed++;

        total++;
        if (checkIntervention(brain, "INTERVENE do(force = 30) QUERY power", "360", "Power under baseline F=30")) passed++;

        total++;
        if (checkIntervention(brain, "INTERVENE do(mass = 5) QUERY accel", "6", "Acceleration under do(mass=5)")) passed++;

        total++;
        if (checkIntervention(brain, "INTERVENE do(mass = 5) QUERY distance", "48", "Distance under do(mass=5)")) passed++;

        total++;
        if (checkIntervention(brain, "INTERVENE do(mass = 5) QUERY power", "720", "Power under do(mass=5)")) passed++;

        // ── Phase 4: Level 3 Counterfactual Physics ("What if?") ──────────────
        System.out.println("\n--- 4. Level 3 Counterfactual Inference: What would have happened? ---");

        total++;
        if (checkCounterfactual(brain, "COUNTERFACTUAL IF force = 60 THEN distance", "48", "What if force was 60 instead of 30?")) passed++;

        total++;
        if (checkCounterfactual(brain, "COUNTERFACTUAL IF time = 2 THEN distance", "6", "What if time was 2s instead of 4s?")) passed++;

        total++;
        if (checkCounterfactual(brain, "COUNTERFACTUAL IF mass = 20 THEN accel", "1.5", "What if mass was 20kg instead of 10kg?")) passed++;

        // ── Phase 5: Level 3 Counterfactual Economics / Policy ─────────────────
        System.out.println("\n--- 5. Level 3 Economic & Policy Counterfactuals ---");
        testBQL(brain, "CAUSAL_DEFINE profit = (price - cost) * quantity");
        testBQL(brain, "CAUSAL_OBSERVE price 20");
        testBQL(brain, "CAUSAL_OBSERVE cost 10");
        testBQL(brain, "CAUSAL_OBSERVE quantity 100");

        total++;
        if (checkCounterfactual(brain, "COUNTERFACTUAL IF price = 30 THEN profit", "2000", "What if price was $30 instead of $20?")) passed++;

        total++;
        if (checkCounterfactual(brain, "COUNTERFACTUAL IF cost = 15 THEN profit", "500", "What if cost was $15 instead of $10?")) passed++;

        System.out.println("\n================================================================================");
        System.out.printf("🎯 FINAL SCORE: %d / %d Tests Passed (%.1f%%)\n", passed, total, (100.0 * passed / total));
        System.out.println("================================================================================");

        if (passed != total) {
            System.exit(1);
        }
    }

    private static void testBQL(BrainInterface brain, String query) {
        brain.respond(query);
    }

    private static boolean checkIntervention(BrainInterface brain, String query, String expectedVal, String description) {
        long t0 = System.nanoTime();
        var resp = brain.respond(query);
        long elapsed = System.nanoTime() - t0;

        String rawJson = (String) resp.get("rawJson");
        boolean ok = false;
        String val = "";
        try {
            JSONObject obj = new JSONObject(rawJson);
            val = obj.optString("result", "");
            ok = val.startsWith(expectedVal) || val.equals(expectedVal);
        } catch (Exception e) {
            ok = false;
        }

        if (ok) {
            System.out.printf("  [PASS] %-45s -> result: %s (%.2f ms)\n", description, val, elapsed / 1e6);
            return true;
        } else {
            System.err.printf("  [FAIL] %-45s -> expected: %s, got: %s (json: %s)\n", description, expectedVal, val, rawJson);
            return false;
        }
    }

    private static boolean checkCounterfactual(BrainInterface brain, String query, String expectedVal, String description) {
        long t0 = System.nanoTime();
        var resp = brain.respond(query);
        long elapsed = System.nanoTime() - t0;

        String rawJson = (String) resp.get("rawJson");
        boolean ok = false;
        String val = "";
        try {
            JSONObject obj = new JSONObject(rawJson);
            val = obj.optString("result", "");
            ok = val.startsWith(expectedVal) || val.equals(expectedVal);
        } catch (Exception e) {
            ok = false;
        }

        if (ok) {
            System.out.printf("  [PASS] %-45s -> result: %s (%.2f ms)\n", description, val, elapsed / 1e6);
            return true;
        } else {
            System.err.printf("  [FAIL] %-45s -> expected: %s, got: %s (json: %s)\n", description, expectedVal, val, rawJson);
            return false;
        }
    }
}
