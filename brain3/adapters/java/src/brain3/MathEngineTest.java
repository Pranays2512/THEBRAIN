package brain3;

import org.json.JSONObject;

public class MathEngineTest {

    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("  📐 Unified Math & Science Reasoning Engine Test 📐");
        System.out.println("==================================================\n");

        System.loadLibrary("brainjni");
        BrainNative brain = new BrainNative(16, 16, 128);
        long handle = brain.getHandle();

        int passed = 0;
        int total = 6;

        // Test 1: Linear Equation Solving
        System.out.println("[Test 1] Linear Equation: SOLVE 2*x + 3 = 7");
        String res1 = brain.executeBrainQL(handle, "SOLVE 2*x + 3 = 7");
        System.out.println("  Output: " + res1);
        JSONObject j1 = new JSONObject(res1);
        if (j1.getString("op").equals("SOLVE") &&
            j1.getBoolean("verified") == true &&
            j1.getBoolean("known") == true &&
            "math_engine".equals(j1.getString("source")) &&
            j1.getString("result").contains("2")) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // Test 2: Another Equation (5*x = 20)
        System.out.println("[Test 2] Linear Equation: SOLVE 5*x = 20");
        String res2 = brain.executeBrainQL(handle, "SOLVE 5*x = 20");
        System.out.println("  Output: " + res2);
        JSONObject j2 = new JSONObject(res2);
        if (j2.getBoolean("verified") == true &&
            j2.getString("result").contains("4")) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // Test 3: Symbolic Integration
        System.out.println("[Test 3] Symbolic Integration: SOLVE int x^3");
        String res3 = brain.executeBrainQL(handle, "SOLVE int x^3");
        System.out.println("  Output: " + res3);
        JSONObject j3 = new JSONObject(res3);
        if (j3.getBoolean("verified") == true &&
            j3.getString("result").contains("x^4")) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // Test 4: Symbolic Differentiation
        System.out.println("[Test 4] Symbolic Differentiation: SOLVE diff x^3");
        String res4 = brain.executeBrainQL(handle, "SOLVE diff x^3");
        System.out.println("  Output: " + res4);
        JSONObject j4 = new JSONObject(res4);
        if (j4.getBoolean("verified") == true &&
            j4.getString("result").contains("x^2")) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // Test 5: Physics Law Calculation (F = m * a)
        System.out.println("[Test 5] Physics Law: SOLVE mass 10 accel 3 force");
        String res5 = brain.executeBrainQL(handle, "SOLVE mass 10 accel 3 force");
        System.out.println("  Output: " + res5);
        JSONObject j5 = new JSONObject(res5);
        if (j5.getBoolean("verified") == true &&
            j5.getString("result").contains("30")) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // Test 6: Natural Language Physics Problem
        System.out.println("[Test 6] NL Physics Word Problem: SOLVE A car travels at speed 20 for time 5. Find distance.");
        String res6 = brain.executeBrainQL(handle, "SOLVE A car travels at speed 20 for time 5. Find distance.");
        System.out.println("  Output: " + res6);
        JSONObject j6 = new JSONObject(res6);
        if (j6.getBoolean("verified") == true &&
            j6.getString("result").contains("100")) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        System.out.println("==================================================");
        System.out.println("  Result: " + passed + "/" + total + " Math Engine Tests Passed!");
        System.out.println("==================================================");
        
        if (passed != total) {
            System.exit(1);
        }
    }
}
