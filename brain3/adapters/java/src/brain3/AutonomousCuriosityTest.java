package brain3;

import org.json.JSONObject;
import java.util.Map;

public class AutonomousCuriosityTest {
    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println("🧠 THE BRAIN 3: AUTONOMOUS CURIOSITY & SELF-DIRECTED DISCOVERY TEST");
        System.out.println("   (Prediction Error Mining, Shape Prior Learning & Autonomous Idle Cycles)");
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

        // ── Phase 1: Observation Ingestion & Transition Error Tracking ────────
        System.out.println("--- 1. Ingesting Observation Sequences & Measuring Prediction Error ---");
        total++;
        String obsQuery = "CURIOSITY_OBSERVE rain,wet_ground,puddles ; rain,wet_ground,puddles ; study,pass ; study,pass ; dice,one ; dice,two ; dice,three ; dice,four";
        if (checkResponse(brain, obsQuery, "Ingested", "Transition Sequence Ingestion")) passed++;

        // ── Phase 2: Curiosity Gaps & Irreducible Stochastic Noise Detection ──
        System.out.println("\n--- 2. Ranking Curiosity Gaps & Stochastic Noise Detection ---");
        total++;
        if (checkResponse(brain, "CURIOSITY_GAPS 5", "dice", "Curiosity Gaps (Stochastic Dice Noise)")) passed++;

        // ── Phase 3: Idle-Time Curiosity Tick & Deterministic Rule Mining ──────
        System.out.println("\n--- 3. Idle-Time Curiosity Tick & Rule Induction ---");
        total++;
        if (checkResponse(brain, "CURIOSITY_TICK", "rain -> wet_ground", "Idle Tick 1: Transition Rule Induction")) passed++;

        // ── Phase 4: Compounding Bayesian Prior Physical Law Discovery ────────
        System.out.println("\n--- 4. Compounding Bayesian Shape Prior Discovery ---");
        total++;
        // Tick 2 resolves kinetic energy (learns shape half_ab2)
        boolean t2 = checkResponse(brain, "CURIOSITY_TICK", "kinetic_energy", "Idle Tick 2: Kinetic Energy Discovery (Learns Shape Prior)");
        // Tick 3 resolves rotational energy in 1 conjecture
        boolean t3 = checkResponse(brain, "CURIOSITY_TICK", "rotational_energy", "Idle Tick 3: Rotational Energy (Fast 1-Conjecture Transfer)");
        // Tick 4 resolves spring energy in 1 conjecture
        boolean t4 = checkResponse(brain, "CURIOSITY_TICK", "spring_energy", "Idle Tick 4: Spring Energy (Compounding Prior Reinforcement)");
        if (t2 && t3 && t4) passed++;

        // ── Phase 5: Full End-to-End Autonomous Cycle Execution ───────────────
        System.out.println("\n--- 5. Full Autonomous Multi-Tick Cycle Execution ---");
        total++;
        if (checkResponse(brain, "AUTONOMOUS_CYCLE 3", "Autonomous Cycle Complete", "Full Autonomous Cycle Execution")) passed++;

        System.out.println("\n================================================================================");
        System.out.printf("🎯 FINAL SCORE: %d / %d Tests Passed (%.1f%%)\n", passed, total, (100.0 * passed / total));
        System.out.println("================================================================================");

        if (passed != total) {
            System.exit(1);
        }
    }

    private static boolean checkResponse(BrainInterface brain, String query, String expectedSnippet, String description) {
        long t0 = System.nanoTime();
        Map<String, Object> resp = brain.respond(query);
        long elapsed = System.nanoTime() - t0;

        String rawJson = (String) resp.get("rawJson");
        String reply = (String) resp.get("reply");
        boolean ok = false;
        String val = "";

        try {
            JSONObject obj = new JSONObject(rawJson);
            val = obj.optString("result", "");
            boolean verified = obj.optBoolean("verified", false);
            if (verified && (val.contains(expectedSnippet) || rawJson.contains(expectedSnippet) || reply.contains(expectedSnippet))) {
                ok = true;
            }
        } catch (Exception e) {
            ok = false;
        }

        if (ok) {
            System.out.printf("  [PASS] %-45s -> result: %s (%.2f ms)\n", description, val.isEmpty() ? reply : val, elapsed / 1e6);
            return true;
        } else {
            System.err.printf("  [FAIL] %-45s -> expected snippet: '%s', got: %s (json: %s)\n", description, expectedSnippet, val, rawJson);
            return false;
        }
    }
}
