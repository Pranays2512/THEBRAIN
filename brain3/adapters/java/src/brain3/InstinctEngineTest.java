package brain3;

import org.json.JSONObject;
import java.util.Map;

/**
 * ════════════════════════════════════════════════════════════════════════════
 * 🧠 THE BRAIN 3: HUMAN INSTINCT & EVOLVING REFLEX ENGINE BENCHMARK SUITE
 *    (Innate Drives, System 2 -> System 1 Compilation, Plasticity & Reflex Arcs)
 * ════════════════════════════════════════════════════════════════════════════
 */
public class InstinctEngineTest {

    private static boolean checkResponse(BrainInterface brain, String query, String expectedSnippet, String testName) {
        long start = System.nanoTime();
        Map<String, Object> resp = brain.respond(query);
        long elapsedNs = System.nanoTime() - start;
        double elapsedMs = elapsedNs / 1_000_000.0;

        try {
            String jsonStr = (String) resp.get("rawJson");
            JSONObject obj = new JSONObject(jsonStr);
            String result = obj.optString("result", "");
            String explanation = obj.optString("explanation", "");
            boolean verified = obj.optBoolean("verified", false);

            boolean match = result.toLowerCase().contains(expectedSnippet.toLowerCase()) ||
                            explanation.toLowerCase().contains(expectedSnippet.toLowerCase()) ||
                            jsonStr.toLowerCase().contains(expectedSnippet.toLowerCase());

            if (match && verified) {
                System.out.printf("  [PASS] %-52s -> result: %s (%.2f ms)%n", testName, result.isEmpty() ? explanation : result, elapsedMs);
                return true;
            } else {
                System.out.printf("  [FAIL] %-52s -> expected: '%s', got: '%s' (json: %s)%n", testName, expectedSnippet, result, jsonStr);
                return false;
            }
        } catch (Exception e) {
            System.out.printf("  [ERROR] %-51s -> %s%n", testName, e.getMessage());
            return false;
        }
    }

    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println("🧠 THE BRAIN 3: HUMAN INSTINCT & EVOLVING REFLEX ENGINE BENCHMARK");
        System.out.println("   (Innate Primal Drives, Reflex Compilation, Plasticity & Zero-Compute System 1)");
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

        // ════════════════════════════════════════════════════════════════════════
        // PHASE 1: Innate Primal Drives & Hardwired Reflexes (5 Tests)
        // ════════════════════════════════════════════════════════════════════════
        System.out.println("\n=== PHASE 1: Innate Primal Drives & Hardwired Reflexes ===");
        
        total++;
        if (checkResponse(brain, "INSTINCT 1=0", "absurdity", "1. Innate Math Contradiction Alarm (1=0)")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT p_and_not_p", "Non-Contradiction", "2. Innate Logic Non-Contradiction Reflex")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT destroy_self", "Self-destructive", "3. Innate Self-Preservation Safety Filter")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT 2+2", "4", "4. Innate Arithmetic Grounding (2+2 -> 4)")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT x*1", "x", "5. Innate Multiplicative Identity Grounding (x*1 -> x)")) passed++;

        // ════════════════════════════════════════════════════════════════════════
        // PHASE 2: System 2 -> System 1 Reflex Crystallization (5 Tests)
        // ════════════════════════════════════════════════════════════════════════
        System.out.println("\n=== PHASE 2: System 2 -> System 1 Reflex Crystallization ===");
        
        total++;
        if (checkResponse(brain, "INSTINCT_TRAIN integral_e_x -> e^x + C", "e^x + C", "6. Crystallize Math Reflex (integral e^x)")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT_TRAIN kinetic_energy_formula -> 0.5*m*v^2", "0.5*m*v^2", "7. Crystallize Physics Reflex (kinetic energy)")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT_TRAIN smoking_causes -> lung_cancer", "lung_cancer", "8. Crystallize Causal Attribution Reflex")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT_TRAIN atom_nucleus_is_like -> solar_sun", "solar_sun", "9. Crystallize Structural Analogy Reflex")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT_TRAIN double_list -> [x*2 for x in list]", "[x*2 for x in list]", "10. Crystallize Code Synthesis Reflex")) passed++;

        // ════════════════════════════════════════════════════════════════════════
        // PHASE 3: Subconscious Zero-Compute System 1 Execution (5 Tests)
        // ════════════════════════════════════════════════════════════════════════
        System.out.println("\n=== PHASE 3: Fast Zero-Compute System 1 Reflex Execution ===");
        
        total++;
        if (checkResponse(brain, "INSTINCT integral_e_x", "e^x + C", "11. Fire Crystallized Math Reflex")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT kinetic_energy_formula", "0.5*m*v^2", "12. Fire Crystallized Physics Reflex")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT smoking_causes", "lung_cancer", "13. Fire Crystallized Causal Reflex")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT atom_nucleus_is_like", "solar_sun", "14. Fire Crystallized Analogy Reflex")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT double_list", "[x*2 for x in list]", "15. Fire Crystallized Code Synthesis Reflex")) passed++;

        // ════════════════════════════════════════════════════════════════════════
        // PHASE 4: Anti-Hebbian Disruption, Refutation & Adaptation (5 Tests)
        // ════════════════════════════════════════════════════════════════════════
        System.out.println("\n=== PHASE 4: Anti-Hebbian Disruption & Reflex Adaptation ===");
        
        total++;
        if (checkResponse(brain, "INSTINCT_TRAIN earth_shape -> flat_plane", "flat_plane", "16. Speculative Reflex Formation (earth_shape)")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT earth_shape", "flat_plane", "17. Verify Speculative Reflex Active")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT_PENALIZE earth_shape", "penalized", "18. Anti-Hebbian Suppression on Refutation")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT_TRAIN earth_shape -> oblate_spheroid", "oblate_spheroid", "19. Plasticity Re-crystallization (Oblate Spheroid)")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT earth_shape", "oblate_spheroid", "20. Verify Updated Truth Reflex Active")) passed++;

        // ════════════════════════════════════════════════════════════════════════
        // PHASE 5: Status Telemetry & Innate Drive Monitoring (5 Tests)
        // ════════════════════════════════════════════════════════════════════════
        System.out.println("\n=== PHASE 5: Instinct Telemetry & Innate Drive Levels ===");
        
        total++;
        if (checkResponse(brain, "INSTINCT_STATUS", "total_reflex_arcs", "21. Reflex Registry Size Telemetry")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT_STATUS", "hit_rate", "22. Subconscious Hit Rate Telemetry")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT_STATUS", "curiosity", "23. Innate Epistemic Curiosity Level")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT_STATUS", "contradiction_aversion", "24. Innate Contradiction Aversion Level")) passed++;
        
        total++;
        if (checkResponse(brain, "INSTINCT_STATUS", "safety", "25. Innate Safety Preservation Level")) passed++;

        System.out.println("\n================================================================================");
        System.out.printf("🎯 FINAL SCORE: %d / %d Tests Passed (%.1f%%)%n", passed, total, (passed * 100.0 / total));
        System.out.println("================================================================================");

        nativeBrain.destroy();
        if (passed != total) {
            System.exit(1);
        }
    }
}
