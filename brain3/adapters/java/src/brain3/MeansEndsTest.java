package brain3;

import org.json.JSONObject;

public class MeansEndsTest {

    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("  🎯 Means-Ends Analysis Planning Engine JNI Test 🎯");
        System.out.println("==================================================\n");

        System.loadLibrary("brainjni");
        BrainNative brain = new BrainNative(16, 16, 128);
        long handle = brain.getHandle();

        // 1. Direct sub-goal decomposition: force = mass * accel
        brain.executeBrainQL(handle, "TEACH rocket mass 10");
        brain.executeBrainQL(handle, "TEACH rocket accel 3");

        int passed = 0;
        int total = 4;

        System.out.println("[Test 1] 1-Step Goal: COMPUTE rocket force");
        String res1 = brain.executeBrainQL(handle, "COMPUTE rocket force");
        System.out.println("  Output: " + res1);
        JSONObject j1 = new JSONObject(res1);
        if ("COMPUTE".equals(j1.getString("op")) &&
            j1.getBoolean("verified") == true &&
            "means_ends".equals(j1.getString("source")) &&
            "30".equals(j1.getString("result"))) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // 2. Multi-step recursive goal decomposition: power = force * speed = (mass * accel) * speed
        brain.executeBrainQL(handle, "TEACH rocket speed 20");
        System.out.println("[Test 2] 2-Step Recursive Planning: COMPUTE rocket power");
        String res2 = brain.executeBrainQL(handle, "COMPUTE rocket power");
        System.out.println("  Output: " + res2);
        JSONObject j2 = new JSONObject(res2);
        if ("COMPUTE".equals(j2.getString("op")) &&
            j2.getBoolean("verified") == true &&
            "600".equals(j2.getString("result"))) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // 3. Goal planning: speed = distance / time
        brain.executeBrainQL(handle, "TEACH car distance 100");
        brain.executeBrainQL(handle, "TEACH car time 5");
        System.out.println("[Test 3] Division Planning: COMPUTE car speed");
        String res3 = brain.executeBrainQL(handle, "COMPUTE car speed");
        System.out.println("  Output: " + res3);
        JSONObject j3 = new JSONObject(res3);
        if ("COMPUTE".equals(j3.getString("op")) &&
            j3.getBoolean("verified") == true &&
            "20".equals(j3.getString("result"))) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // 4. Electrical Goal planning: voltage = current * resistance
        brain.executeBrainQL(handle, "TEACH circuit current 3");
        brain.executeBrainQL(handle, "TEACH circuit resistance 4");
        System.out.println("[Test 4] Ohm's Law Planning: COMPUTE circuit voltage");
        String res4 = brain.executeBrainQL(handle, "COMPUTE circuit voltage");
        System.out.println("  Output: " + res4);
        JSONObject j4 = new JSONObject(res4);
        if ("COMPUTE".equals(j4.getString("op")) &&
            j4.getBoolean("verified") == true &&
            "12".equals(j4.getString("result"))) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        System.out.println("==================================================");
        System.out.println("  Result: " + passed + "/" + total + " Means-Ends Planning Tests Passed!");
        System.out.println("==================================================");

        if (passed != total) {
            System.exit(1);
        }
    }
}
