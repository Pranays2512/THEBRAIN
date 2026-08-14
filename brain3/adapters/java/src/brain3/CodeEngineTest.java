package brain3;

import org.json.JSONObject;

public class CodeEngineTest {

    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("  💻 Program Synthesis & CodeEngine JNI Test 💻");
        System.out.println("==================================================\n");

        System.loadLibrary("brainjni");
        BrainNative brain = new BrainNative(16, 16, 128);
        long handle = brain.getHandle();

        int passed = 0;
        int total = 4;

        // Test 1: Map * 2
        System.out.println("[Test 1] Synthesis: SYNTH [1, 2, 3] -> [2, 4, 6]");
        String res1 = brain.executeBrainQL(handle, "SYNTH [1, 2, 3] -> [2, 4, 6]");
        System.out.println("  Output: " + res1);
        JSONObject j1 = new JSONObject(res1);
        if ("SYNTH".equals(j1.getString("op")) &&
            j1.getBoolean("verified") == true &&
            j1.getBoolean("known") == true &&
            "code_engine".equals(j1.getString("source")) &&
            j1.getString("result").contains("* 2")) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // Test 2: Map + 1
        System.out.println("[Test 2] Synthesis: SYNTH [1, 2, 3] -> [2, 3, 4]");
        String res2 = brain.executeBrainQL(handle, "SYNTH [1, 2, 3] -> [2, 3, 4]");
        System.out.println("  Output: " + res2);
        JSONObject j2 = new JSONObject(res2);
        if (j2.getBoolean("verified") == true &&
            j2.getString("result").contains("+ 1")) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // Test 3: Filter > 0
        System.out.println("[Test 3] Synthesis: SYNTH [3, -1, 2, -5] -> [3, 2]");
        String res3 = brain.executeBrainQL(handle, "SYNTH [3, -1, 2, -5] -> [3, 2]");
        System.out.println("  Output: " + res3);
        JSONObject j3 = new JSONObject(res3);
        if (j3.getBoolean("verified") == true &&
            j3.getString("result").contains("x > 0")) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // Test 4: List Reverse
        System.out.println("[Test 4] Synthesis: SYNTH [1, 2, 3] -> [3, 2, 1]");
        String res4 = brain.executeBrainQL(handle, "SYNTH [1, 2, 3] -> [3, 2, 1]");
        System.out.println("  Output: " + res4);
        JSONObject j4 = new JSONObject(res4);
        if (j4.getBoolean("verified") == true &&
            j4.getString("result").contains("reversed")) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        System.out.println("==================================================");
        System.out.println("  Result: " + passed + "/" + total + " CodeEngine Tests Passed!");
        System.out.println("==================================================");

        if (passed != total) {
            System.exit(1);
        }
    }
}
