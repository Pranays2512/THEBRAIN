package brain3;

import org.json.JSONObject;
import java.util.Map;

public class ScientificDiscoveryTest {
    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println("🧠 THE BRAIN 3: AUTOMATED SCIENTIFIC LAW DISCOVERY TEST");
        System.out.println("   (Empirical Induction, BACON Invariant Mining & Symbolic Regression)");
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

        // ── Phase 1: Kepler's Third Harmonic Law of Planetary Motion ──────────
        System.out.println("--- 1. Planetary Orbital Mechanics (Kepler's Third Law) ---");
        total++;
        if (checkDiscovery(brain, "DISCOVER LAW kepler_planetary", "T", "R", "Kepler's Harmonic Law (T^2 = R^3)")) passed++;

        // ── Phase 2: Newton's Second Law of Motion ────────────────────────────
        System.out.println("\n--- 2. Classical Mechanics (Newton's Second Law) ---");
        total++;
        if (checkDiscovery(brain, "DISCOVER LAW newton_force", "F = m * a", "F", "Newton's Second Law (F = m * a)")) passed++;

        // ── Phase 3: Ohm's Law of Electrical Resistance ───────────────────────
        System.out.println("\n--- 3. Electromagnetism (Ohm's Law) ---");
        total++;
        if (checkDiscovery(brain, "DISCOVER LAW ohm_circuit", "V = I * R", "V", "Ohm's Law (V = I * R)")) passed++;

        // ── Phase 4: Boyle's Ideal Gas Law ────────────────────────────────────
        System.out.println("\n--- 4. Thermodynamics (Boyle's Ideal Gas Law) ---");
        total++;
        if (checkDiscovery(brain, "DISCOVER LAW boyle_gas", "P = 100 / V", "P", "Boyle's Gas Law (P * V = 100)")) passed++;

        // ── Phase 5: Kinetic Energy Quadratic Law ─────────────────────────────
        System.out.println("\n--- 5. Energy Mechanics (Kinetic Energy Quadratic Law) ---");
        total++;
        if (checkDiscovery(brain, "DISCOVER LAW kinetic_energy", "0.5 * m * v^2", "E", "Kinetic Energy (E = 0.5 * m * v^2)")) passed++;

        // ── Phase 6: Real-Time Dynamic Tabular Observation Induction ─────────
        System.out.println("\n--- 6. Real-Time Dynamic Raw Data Induction ---");
        total++;
        if (checkDiscovery(brain, "DISCOVER y FROM x DATA 2:8 ; 3:27 ; 4:64 ; 5:125", "x1^3", "y", "Cubic Power Law Induction")) passed++;

        System.out.println("\n================================================================================");
        System.out.printf("🎯 FINAL SCORE: %d / %d Tests Passed (%.1f%%)\n", passed, total, (100.0 * passed / total));
        System.out.println("================================================================================");

        if (passed != total) {
            System.exit(1);
        }
    }

    private static boolean checkDiscovery(BrainInterface brain, String query, String expectedSnippet, String expectedTarget, String description) {
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
            System.out.printf("  [PASS] %-45s -> equation: %s (%.2f ms)\n", description, val, elapsed / 1e6);
            return true;
        } else {
            System.err.printf("  [FAIL] %-45s -> expected snippet: '%s', got: %s (json: %s)\n", description, expectedSnippet, val, rawJson);
            return false;
        }
    }
}
