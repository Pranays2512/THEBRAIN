package brain3;

import org.json.JSONObject;
import java.util.Map;

public class MetacognitiveSelfRefutationTest {
    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println("🧠 THE BRAIN 3: DUAL-PROCESS METACOGNITIVE SELF-REFUTATION TEST");
        System.out.println("   (System 1 Fast Intuition vs. System 2 Adversarial Formal Refuter)");
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

        // ── Setup Knowledge Base ──────────────────────────────────────────────
        System.out.println("--- 0. Seeding Epistemic Knowledge Base & Exceptions ---");
        brain.respond("TEACH bird can fly");
        brain.respond("TEACH penguin isa bird");
        brain.respond("TEACH penguin can swim");
        brain.respond("TEACH penguin can <EXCEPTION>");
        brain.respond("TEACH eagle isa bird");
        brain.respond("TEACH bat isa mammal");
        System.out.println("   ✓ Knowledge graph primed with default rules and non-monotonic exceptions.\n");

        // ── Phase 1: Non-Monotonic Exception Blocker Refutation ───────────────
        System.out.println("--- 1. Non-Monotonic Exception Blocker Refutation ---");
        total++;
        if (checkMetacognition(brain, "REFUTE penguin can fly", "REFUTED", "Non-monotonic exception", "Penguin flight fallacy")) passed++;

        // ── Phase 2: Disjoint Taxonomic Clade Refutation ───────────────────────
        System.out.println("\n--- 2. Disjoint Taxonomic Category Conflict Refutation ---");
        total++;
        if (checkMetacognition(brain, "REFUTE bat isa bird", "REFUTED", "Disjoint category", "Bat-Bird taxonomic clash")) passed++;

        // ── Phase 3: Physical Boundary Invariant Refutation ───────────────────
        System.out.println("\n--- 3. Physical Boundary Invariant Refutation ---");
        total++;
        if (checkMetacognition(brain, "REFUTE mass = -5", "REFUTED", "Physical invariant", "Negative mass refutation")) passed++;

        // ── Phase 4: Mathematical Boundary Invariant Refutation ───────────────
        System.out.println("\n--- 4. Mathematical Boundary Invariant Refutation ---");
        total++;
        if (checkMetacognition(brain, "REFUTE divisor = 0", "REFUTED", "division by zero", "Division by zero refutation")) passed++;

        // ── Phase 5: Positive Sound Belief Verification ────────────────────────
        System.out.println("\n--- 5. Positive Metacognitive Sound Verification ---");
        total++;
        if (checkMetacognition(brain, "META_VERIFY eagle can fly", "VERIFIED_SOUND", "eagle can fly", "Eagle flight soundness")) passed++;

        // ── Phase 6: Dual-Process Critique Operator ───────────────────────────
        System.out.println("\n--- 6. Dual-Process CRITIQUE Trace Operator ---");
        total++;
        if (checkMetacognition(brain, "CRITIQUE penguin can fly", "REFUTED", "System 1 naive heuristic", "Dual-Process System 1/2 Trace")) passed++;

        System.out.println("\n================================================================================");
        System.out.printf("🎯 FINAL SCORE: %d / %d Tests Passed (%.1f%%)\n", passed, total, (100.0 * passed / total));
        System.out.println("================================================================================");

        if (passed != total) {
            System.exit(1);
        }
    }

    private static boolean checkMetacognition(BrainInterface brain, String query, String expectedVerdict, String expectedSnippet, String description) {
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
            String expl = obj.optString("explanation", "");
            if ((val.contains(expectedVerdict) || expl.contains(expectedVerdict) || reply.contains(expectedVerdict) || rawJson.contains(expectedVerdict)) &&
                (val.contains(expectedSnippet) || expl.contains(expectedSnippet) || reply.contains(expectedSnippet) || rawJson.contains(expectedSnippet))) {
                ok = true;
            }
        } catch (Exception e) {
            ok = false;
        }

        if (ok) {
            System.out.printf("  [PASS] %-40s -> %s (%.2f ms)\n", description, val, elapsed / 1e6);
            return true;
        } else {
            System.err.printf("  [FAIL] %-40s -> expected: '%s' & '%s', got: %s (json: %s)\n", description, expectedVerdict, expectedSnippet, val, rawJson);
            return false;
        }
    }
}
