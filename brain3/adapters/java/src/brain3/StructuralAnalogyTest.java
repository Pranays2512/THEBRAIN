package brain3;

import org.json.JSONObject;
import java.util.Map;

public class StructuralAnalogyTest {
    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println("🧠 THE BRAIN 3: GENTNER STRUCTURE-MAPPING ANALOGY ENGINE (SME) TEST");
        System.out.println("   (Systematicity, Cross-Domain Relational Alignment & Candidate Projection)");
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

        // ── Phase 1: Rutherford Solar-System Atom Analogy ──────────────────────
        System.out.println("--- 1. Solar System -> Rutherford-Bohr Atomic Model ---");
        total++;
        if (checkAnalogy(brain, "ANALOGY solar_system TO rutherford_atom", 
                "electron revolves_around nucleus", "Rutherford Atom orbit derivation")) passed++;

        // ── Phase 2: Hydraulic-Electrical Circuit Analogy ─────────────────────
        System.out.println("\n--- 2. Hydraulic Water Flow -> Electrical Circuit Analogy ---");
        total++;
        if (checkAnalogy(brain, "ANALOGY hydraulic_system TO electric_circuit", 
                "voltage_difference causes current_flow", "Ohm's Law / Circuit flow derivation")) passed++;

        // ── Phase 3: Biological Cell -> Factory Analogy ────────────────────────
        System.out.println("\n--- 3. Biological Cell -> Industrial Factory Functional Analogy ---");
        total++;
        if (checkAnalogy(brain, "ANALOGY biological_cell TO factory", 
                "matched: (membrane controls transport) <-> (security_gate controls transport)", "Cell-to-Factory Organelle Alignment")) passed++;

        // ── Phase 4: Dynamic Domain Ingestion & Real-Time Projection ──────────
        System.out.println("\n--- 4. Real-Time Custom Domain Ingestion & Mapping ---");
        brain.respond("ANALOGY_DEFINE operating_system kernel manages memory");
        brain.respond("ANALOGY_DEFINE operating_system process executes instructions");
        brain.respond("ANALOGY_DEFINE operating_system kernel schedules process");
        brain.respond("ANALOGY_DEFINE operating_system memory stores data");

        brain.respond("ANALOGY_DEFINE company_org ceo manages budget");
        brain.respond("ANALOGY_DEFINE company_org ceo schedules employee");
        brain.respond("ANALOGY_DEFINE company_org budget stores capital");

        total++;
        if (checkAnalogy(brain, "ANALOGY operating_system TO company_org", 
                "employee executes instructions", "OS-to-Corporate Management Analogy")) passed++;

        // ── Phase 5: Analogy BrainQL Project Operator ─────────────────────────
        System.out.println("\n--- 5. BrainQL Query Variations & Project Operator ---");
        total++;
        if (checkAnalogy(brain, "ANALOGY PROJECT solar_system TO atom", 
                "electron revolves_around nucleus", "Analogy PROJECT syntax test")) passed++;

        System.out.println("\n================================================================================");
        System.out.printf("🎯 FINAL SCORE: %d / %d Tests Passed (%.1f%%)\n", passed, total, (100.0 * passed / total));
        System.out.println("================================================================================");

        if (passed != total) {
            System.exit(1);
        }
    }

    private static boolean checkAnalogy(BrainInterface brain, String query, String expectedSnippet, String description) {
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
            System.out.printf("  [PASS] %-45s -> result: %s (%.2f ms)\n", description, val, elapsed / 1e6);
            return true;
        } else {
            System.err.printf("  [FAIL] %-45s -> expected snippet: '%s', got: %s (json: %s)\n", description, expectedSnippet, val, rawJson);
            return false;
        }
    }
}
