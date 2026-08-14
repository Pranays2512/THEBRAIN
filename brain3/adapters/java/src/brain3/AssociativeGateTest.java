package brain3;

import org.json.JSONObject;

public class AssociativeGateTest {

    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("  🧠 Associative Triple Pre-Verification Gate Test 🧠");
        System.out.println("==================================================\n");
        
        System.loadLibrary("brainjni");
        BrainNative brain = new BrainNative(16, 16, 128);
        long handle = brain.getHandle();
        
        // Setup: Teach the crisp store a fact (Capitalized)
        String teachPayload = brain.executeBrainQL(handle, "TEACH Paris capital France");
        System.out.println("TEACH (Paris capital France) -> " + teachPayload);
        
        int passed = 0;
        int total = 4;
        
        // Test 1: Happy Path (exact case, though predictor shouldn't output this usually)
        System.out.println("\n[Test 1] Happy Path: CHAT TEST_GATE_1");
        String res1 = brain.executeBrainQL(handle, "CHAT TEST_GATE_1");
        JSONObject j1 = new JSONObject(res1);
        if (j1.getBoolean("known") == true && 
            "atomic".equals(j1.getString("objType")) &&
            "gate_verified_associative".equals(j1.getString("source"))) {
            System.out.println("  -> PASS");
            passed++;
        } else {
            System.out.println("  -> FAIL. Output: " + res1);
        }
        
        // Test 2: Case Normalization (Predictor outputs lowercase)
        System.out.println("\n[Test 2] Case Normalization: CHAT TEST_GATE_2");
        String res2 = brain.executeBrainQL(handle, "CHAT TEST_GATE_2");
        JSONObject j2 = new JSONObject(res2);
        if (j2.getBoolean("known") == true && 
            "atomic".equals(j2.getString("objType")) &&
            "gate_verified_associative".equals(j2.getString("source"))) {
            System.out.println("  -> PASS");
            passed++;
        } else {
            System.out.println("  -> FAIL. Output: " + res2);
        }
        
        // Test 3: Negative Path (False Positive Guard)
        System.out.println("\n[Test 3] False Positive Guard: CHAT TEST_GATE_3");
        String res3 = brain.executeBrainQL(handle, "CHAT TEST_GATE_3");
        JSONObject j3 = new JSONObject(res3);
        if (j3.getBoolean("known") == false && 
            "freetext".equals(j3.getString("objType")) &&
            "associative_musing".equals(j3.getString("source"))) {
            System.out.println("  -> PASS");
            passed++;
        } else {
            System.out.println("  -> FAIL. Output: " + res3);
        }
        
        // Test 4: Negative Path (Cache Miss)
        System.out.println("\n[Test 4] Cache Miss: CHAT TEST_GATE_4");
        String res4 = brain.executeBrainQL(handle, "CHAT TEST_GATE_4");
        JSONObject j4 = new JSONObject(res4);
        if (j4.getBoolean("known") == false && 
            "freetext".equals(j4.getString("objType")) &&
            "associative_musing".equals(j4.getString("source"))) {
            System.out.println("  -> PASS");
            passed++;
        } else {
            System.out.println("  -> FAIL. Output: " + res4);
        }
        
        System.out.println("\nTest Complete. " + passed + "/" + total + " payloads passed.");
        if (passed != total) {
            System.exit(1);
        }
    }
}
