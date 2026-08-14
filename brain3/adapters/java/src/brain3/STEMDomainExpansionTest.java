package brain3;

import org.json.JSONObject;
import java.util.Map;

public class STEMDomainExpansionTest {
    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println("🧠 THE BRAIN 3: LARGE-SCALE STEM & MULTI-DOMAIN EXPANSION BENCHMARK");
        System.out.println("   (College Physics, Chemical Equilibria, Legal/Clinical SCMs & Algorithmic Synth)");
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
        // SECTION 1: Advanced College Physics & Quantum Equations (10 Tests)
        // ════════════════════════════════════════════════════════════════════════
        System.out.println("=== SECTION 1: Advanced College Physics & Quantum Laws ===");
        
        total++;
        if (checkResponse(brain, "SOLVE Tc 300 Th 600 eta", "0.5", "1. Carnot Efficiency (Tc=300, Th=600)")) passed++;
        
        total++;
        if (checkResponse(brain, "SOLVE Tc 200 Th 800 eta", "0.75", "2. Carnot Efficiency (Tc=200, Th=800)")) passed++;
        
        total++;
        if (checkResponse(brain, "SOLVE n 2 R 8.314 T 300 V 10 P", "498.84", "3. Ideal Gas Law (n=2, T=300, V=10)")) passed++;
        
        total++;
        if (checkResponse(brain, "SOLVE m1 10 m2 20 r 2 F", "50", "4. Universal Gravitation (m1=10, m2=20, r=2)")) passed++;
        
        total++;
        if (checkResponse(brain, "SOLVE q1 3 q2 4 r 2 F", "3", "5. Coulomb Electrostatic Law (q1=3, q2=4, r=2)")) passed++;
        
        total++;
        if (checkResponse(brain, "SOLVE h 6.626 p 2 lambda", "3.313", "6. De Broglie Quantum Wavelength (h=6.626, p=2)")) passed++;
        
        total++;
        if (checkResponse(brain, "SOLVE eta 2 L 10 r 2 R", "10", "7. Poiseuille Fluid Resistance (eta=2, L=10, r=2)")) passed++;
        
        total++;
        if (checkResponse(brain, "SOLVE W 500 t 10 P", "50", "8. Mechanical Power (W=500, t=10)")) passed++;
        
        total++;
        if (checkResponse(brain, "SOLVE F 100 d 5 W", "500", "9. Mechanical Work (F=100, d=5)")) passed++;
        
        total++;
        if (checkResponse(brain, "SOLVE m 10 v 4 ke", "80", "10. Kinetic Energy (m=10, v=4)")) passed++;

        // ════════════════════════════════════════════════════════════════════════
        // SECTION 2: Physical & Organic Chemistry Laws (10 Tests)
        // ════════════════════════════════════════════════════════════════════════
        System.out.println("\n=== SECTION 2: Physical & Organic Chemistry Laws ===");
        
        total++;
        if (checkResponse(brain, "SOLVE pKa 4.75 base 2.71828 acid 1 pH", "5.7", "11. Henderson-Hasselbalch Buffer pH (pKa=4.75)")) passed++;
        
        total++;
        if (checkResponse(brain, "DISCOVER LAW carnot_efficiency", "eta", "12. Discovery: Carnot Efficiency (eta = 1 - Tc/Th)")) passed++;
        
        total++;
        if (checkResponse(brain, "DISCOVER LAW ideal_gas_pv", "P", "13. Discovery: Ideal Gas Law (P = 8.314 * n * T / V)")) passed++;
        
        total++;
        if (checkResponse(brain, "DISCOVER LAW coulomb_electrostatic", "F", "14. Discovery: Coulomb's Law (F = 9 * q1 * q2 / r^2)")) passed++;
        
        total++;
        if (checkResponse(brain, "DISCOVER LAW poiseuille_fluid", "Q", "15. Discovery: Poiseuille Fluid Flow (Q = 2 * r^4 * dP)")) passed++;
        
        total++;
        if (checkResponse(brain, "SOLVE m 50 V 10 rho", "5", "16. Chemical Density (m=50, V=10)")) passed++;
        
        total++;
        if (checkResponse(brain, "SOLVE I 5 R 12 V", "60", "17. Electrochemical Ohm's Law (I=5, R=12)")) passed++;
        
        total++;
        if (checkResponse(brain, "DISCOVER LAW boyle_gas", "P", "18. Discovery: Boyle's Gas Law (P = 100 / V)")) passed++;
        
        total++;
        if (checkResponse(brain, "DISCOVER LAW newton_force", "F", "19. Discovery: Newton's Second Law (F = m * a)")) passed++;
        
        total++;
        if (checkResponse(brain, "DISCOVER LAW kepler_planetary", "T", "20. Discovery: Kepler's Harmonic Law (T^2 = R^3)")) passed++;

        // ════════════════════════════════════════════════════════════════════════
        // SECTION 3: Legal, Clinical & Macro-Economic Structural Causal Models (10 Tests)
        // ════════════════════════════════════════════════════════════════════════
        System.out.println("\n=== SECTION 3: Legal, Biomedical & Macroeconomic SCMs ===");
        
        total++;
        String legalObs = "CAUSAL_OBSERVE duty_of_care=1, safety_inspection=0, baseline_hazard=2, defect_present=1, accident_occurred=1, medical_cost=5000, lost_wages=3000";
        if (checkResponse(brain, legalObs, "Observed", "21. Legal Tort SCM Evidence Ingestion")) passed++;
        
        total++;
        if (checkResponse(brain, "INTERVENE defect_present=1 QUERY damages", "10000", "22. Legal Liability Interventional Damages")) passed++;
        
        total++;
        if (checkResponse(brain, "COUNTERFACTUAL defect_present=0 THEN liability", "0", "23. Legal 'But-For' Cause In Fact (No defect -> No liability)")) passed++;
        
        total++;
        if (checkResponse(brain, "COUNTERFACTUAL accident_occurred=0 THEN damages", "0", "24. Counterfactual: No accident -> Zero damages")) passed++;
        
        total++;
        String clinicalObs = "CAUSAL_OBSERVE baseline_biomarker=100, drug_treatment=0, disease_severity=2, patient_age=60";
        if (checkResponse(brain, clinicalObs, "Observed", "25. Clinical Trial Evidence Ingestion")) passed++;
        
        total++;
        if (checkResponse(brain, "COUNTERFACTUAL drug_treatment=1 THEN biomarker", "95", "26. Clinical Counterfactual: Biomarker drop with treatment")) passed++;
        
        total++;
        if (checkResponse(brain, "COUNTERFACTUAL drug_treatment=1 THEN recovery_rate", "40", "27. Clinical Counterfactual: Recovery rate boost")) passed++;
        
        total++;
        String macroObs = "CAUSAL_OBSERVE interest_rate=2.0, money_growth=5.0, gdp_stimulus=4.0";
        if (checkResponse(brain, macroObs, "Observed", "28. Macro-Economic Evidence Ingestion")) passed++;
        
        total++;
        if (checkResponse(brain, "COUNTERFACTUAL interest_rate=5.0 THEN inflation", "6.5", "29. Central Bank Rate Hike -> Inflation drop to 6.5%")) passed++;
        
        total++;
        if (checkResponse(brain, "COUNTERFACTUAL interest_rate=5.0 THEN gdp_growth", "3.5", "30. Rate Hike -> GDP Growth Slowdown to 3.5%")) passed++;

        // ════════════════════════════════════════════════════════════════════════
        // SECTION 4: Deep Cross-Disciplinary Structural Analogies (10 Tests)
        // ════════════════════════════════════════════════════════════════════════
        System.out.println("\n=== SECTION 4: Cross-Disciplinary Structural Analogies ===");
        
        total++;
        if (checkResponse(brain, "ANALOGY thermodynamics TO market_economics", "score", "31. Thermo-Economics Systematicity Alignment")) passed++;
        
        total++;
        if (checkResponse(brain, "ANALOGY thermodynamics TO market_economics PROJECT heat_transfer", "trade_flow", "32. Project: Heat Transfer -> Trade Flow")) passed++;
        
        total++;
        if (checkResponse(brain, "ANALOGY thermodynamics TO market_economics PROJECT temperature_difference", "price_difference", "33. Project: Temp Difference -> Price Difference")) passed++;
        
        total++;
        if (checkResponse(brain, "ANALOGY organic_synthesis TO compiler_pipeline", "score", "34. Organic Chemistry - Compiler Alignment")) passed++;
        
        total++;
        if (checkResponse(brain, "ANALOGY organic_synthesis TO compiler_pipeline PROJECT catalyst", "optimizer_pass", "35. Project: Chemical Catalyst -> Optimizer Pass")) passed++;
        
        total++;
        if (checkResponse(brain, "ANALOGY organic_synthesis TO compiler_pipeline PROJECT pure_target_compound", "machine_binary", "36. Project: Target Compound -> Machine Binary")) passed++;
        
        total++;
        if (checkResponse(brain, "ANALOGY cardiovascular_system TO packet_network", "score", "37. Cardio-Network Systematicity Alignment")) passed++;
        
        total++;
        if (checkResponse(brain, "ANALOGY cardiovascular_system TO packet_network PROJECT blood", "data_packets", "38. Project: Blood -> Data Packets")) passed++;
        
        total++;
        if (checkResponse(brain, "ANALOGY cardiovascular_system TO packet_network PROJECT blood_vessels", "fiber_links", "39. Project: Blood Vessels -> Fiber Links")) passed++;
        
        total++;
        if (checkResponse(brain, "ANALOGY hydraulic_system TO electric_circuit", "voltage_difference", "40. Hydraulic-Electrical Flow Analogy")) passed++;

        // ════════════════════════════════════════════════════════════════════════
        // SECTION 5: Algorithmic Synthesis & Program Refinements (10 Tests)
        // ════════════════════════════════════════════════════════════════════════
        System.out.println("\n=== SECTION 5: Algorithmic Synthesis & Program Transformations ===");
        
        total++;
        if (checkResponse(brain, "SYNTH [1, 2, 3] -> [2, 4, 6]", "x * 2", "41. Program Synthesis: Map (*2)")) passed++;
        
        total++;
        if (checkResponse(brain, "SYNTH [1, 2, 3] -> [2, 3, 4]", "x + 1", "42. Program Synthesis: Map (+1)")) passed++;
        
        total++;
        if (checkResponse(brain, "SYNTH [1, 2, 3] -> [0, 1, 2]", "x - 1", "43. Program Synthesis: Map (-1)")) passed++;
        
        total++;
        if (checkResponse(brain, "SYNTH [1, 2, 3] -> [1, 4, 9]", "x ** 2", "44. Program Synthesis: Map (^2)")) passed++;
        
        total++;
        if (checkResponse(brain, "SYNTH [-3, 4, -5] -> [3, 4, 5]", "abs(x)", "45. Program Synthesis: Map (abs)")) passed++;
        
        total++;
        if (checkResponse(brain, "SYNTH [3, -1, 2, -5] -> [3, 2]", "x > 0", "46. Program Synthesis: Filter (>0)")) passed++;
        
        total++;
        if (checkResponse(brain, "SYNTH [3, -1, 2, -5] -> [-1, -5]", "x < 0", "47. Program Synthesis: Filter (<0)")) passed++;
        
        total++;
        if (checkResponse(brain, "SYNTH [1, 2, 3, 4] -> [2, 4]", "x % 2 == 0", "48. Program Synthesis: Filter (even)")) passed++;
        
        total++;
        if (checkResponse(brain, "SYNTH [1, 2, 3] -> [3, 2, 1]", "reversed", "49. Program Synthesis: List (Reverse)")) passed++;
        
        total++;
        if (checkResponse(brain, "SYNTH [3, 1, 2] -> [1, 2, 3]", "sorted", "50. Program Synthesis: List (Sort)")) passed++;

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
            System.out.printf("  [PASS] %-52s -> result: %s (%.2f ms)\n", description, val.isEmpty() ? reply : val, elapsed / 1e6);
            return true;
        } else {
            System.err.printf("  [FAIL] %-52s -> expected: '%s', got: %s (json: %s)\n", description, expectedSnippet, val, rawJson);
            return false;
        }
    }
}
