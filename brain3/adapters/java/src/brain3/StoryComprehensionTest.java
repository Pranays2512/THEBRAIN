package brain3;

import org.json.JSONObject;

public class StoryComprehensionTest {

    public static void main(String[] args) {
        System.out.println("======================================================================");
        System.out.println("  📖 Episodic Narrative & Story Comprehension Benchmark 📖");
        System.out.println("======================================================================\n");

        System.loadLibrary("brainjni");
        BrainNative brain = new BrainNative(16, 16, 128);
        long handle = brain.getHandle();

        // 1. Ingest Episode 1 (Alice at the store)
        brain.executeBrainQL(handle, "TEACH alice_ep1 actor alice");
        brain.executeBrainQL(handle, "TEACH alice_ep1 went_to store");
        brain.executeBrainQL(handle, "TEACH alice_ep1 bought red_apple");
        brain.executeBrainQL(handle, "TEACH alice_ep1 ate red_apple");
        brain.executeBrainQL(handle, "TEACH alice_ep1 felt happy");

        // 2. Ingest Episode 2 (Bob at the park)
        brain.executeBrainQL(handle, "TEACH bob_ep1 actor bob");
        brain.executeBrainQL(handle, "TEACH bob_ep1 went_to park");
        brain.executeBrainQL(handle, "TEACH bob_ep1 found brown_dog");
        brain.executeBrainQL(handle, "TEACH bob_ep1 played_with brown_dog");
        brain.executeBrainQL(handle, "TEACH bob_ep1 felt tired");

        // 3. Ingest Episode 3 (Alice at the park)
        brain.executeBrainQL(handle, "TEACH alice_ep2 actor alice");
        brain.executeBrainQL(handle, "TEACH alice_ep2 went_to park");
        brain.executeBrainQL(handle, "TEACH alice_ep2 found green_frog");
        brain.executeBrainQL(handle, "TEACH alice_ep2 played_with green_frog");
        brain.executeBrainQL(handle, "TEACH alice_ep2 felt happy");

        // 4. Ingest Episode 4 (Bob at the store)
        brain.executeBrainQL(handle, "TEACH bob_ep2 actor bob");
        brain.executeBrainQL(handle, "TEACH bob_ep2 went_to store");
        brain.executeBrainQL(handle, "TEACH bob_ep2 bought blue_hat");
        brain.executeBrainQL(handle, "TEACH bob_ep2 wore blue_hat");
        brain.executeBrainQL(handle, "TEACH bob_ep2 felt cool");

        int passed = 0;
        int total = 12;

        // Query tests
        String[][] cases = {
            {"LOOKUP alice_ep1 went_to", "store", "true"},
            {"LOOKUP alice_ep1 bought", "red_apple", "true"},
            {"LOOKUP alice_ep1 felt", "happy", "true"},
            {"LOOKUP bob_ep1 went_to", "park", "true"},
            {"LOOKUP bob_ep1 found", "brown_dog", "true"},
            {"LOOKUP bob_ep1 felt", "tired", "true"},
            {"LOOKUP alice_ep2 went_to", "park", "true"},
            {"LOOKUP alice_ep2 found", "green_frog", "true"},
            {"LOOKUP alice_ep2 felt", "happy", "true"},
            {"LOOKUP bob_ep2 went_to", "store", "true"},
            {"LOOKUP bob_ep2 bought", "blue_hat", "true"},
            {"LOOKUP bob_ep2 felt", "cool", "true"}
        };

        for (int i = 0; i < cases.length; i++) {
            String q = cases[i][0];
            String expected = cases[i][1];
            boolean expectedKnown = Boolean.parseBoolean(cases[i][2]);

            String res = brain.executeBrainQL(handle, q);
            JSONObject j = new JSONObject(res);

            boolean match = expectedKnown == j.getBoolean("known") && expected.equals(j.getString("result"));
            System.out.println("[Test " + (i+1) + "] " + q + " -> " + j.getString("result") + (match ? " (PASS)" : " (FAIL)"));
            if (match) passed++;
        }

        System.out.println("\n======================================================================");
        System.out.println("  Result: " + passed + "/" + total + " Story Comprehension Tests Passed!");
        System.out.println("======================================================================\n");

        if (passed != total) {
            System.exit(1);
        }
    }
}
