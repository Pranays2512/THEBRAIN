package brain3;

import org.json.JSONObject;

public class ConversationalHumanitiesTest {

    public static void main(String[] args) {
        System.out.println("======================================================================");
        System.out.println("  🗣️ Conversational & Humanities Multi-Domain Evaluation Suite 🗣️");
        System.out.println("======================================================================\n");

        System.loadLibrary("brainjni");
        BrainNative brain = new BrainNative(16, 16, 128);
        long handle = brain.getHandle();

        // 1. Ingest Conversational Dialogue & Social Sciences into active instance
        brain.executeBrainQL(handle, "TEACH hello response hi_there");
        brain.executeBrainQL(handle, "TEACH what_is_your_name response i_am_brain");
        brain.executeBrainQL(handle, "TEACH what_can_you_do response i_can_converse_and_reason");
        brain.executeBrainQL(handle, "TEACH what_is_the_sun response a_star");

        brain.executeBrainQL(handle, "TEACH India world_rank_by_area 7");
        brain.executeBrainQL(handle, "TEACH India isa country");
        brain.executeBrainQL(handle, "TEACH Asia isa continent");
        brain.executeBrainQL(handle, "TEACH Himalayan_Mountain_range position north");
        brain.executeBrainQL(handle, "TEACH Thar_Desert position west");

        brain.executeBrainQL(handle, "TEACH science isa process");
        brain.executeBrainQL(handle, "TEACH physics isa field_of_science");
        brain.executeBrainQL(handle, "TEACH chemistry isa field_of_science");
        brain.executeBrainQL(handle, "TEACH biology isa field_of_science");

        brain.executeBrainQL(handle, "TEACH cpp isa programming_language");
        brain.executeBrainQL(handle, "TEACH linux isa operating_system");
        brain.executeBrainQL(handle, "TEACH tcp_ip isa network_protocol");

        int passed = 0;
        int total = 15;

        String[][] testCases = {
            // Conversational
            {"LOOKUP hello response", "hi_there"},
            {"LOOKUP what_is_your_name response", "i_am_brain"},
            {"LOOKUP what_can_you_do response", "i_can_converse_and_reason"},
            {"LOOKUP what_is_the_sun response", "a_star"},
            // Geography & SSC
            {"LOOKUP India world_rank_by_area", "7"},
            {"LOOKUP India isa", "country"},
            {"LOOKUP Asia isa", "continent"},
            {"LOOKUP Himalayan_Mountain_range position", "north"},
            {"LOOKUP Thar_Desert position", "west"},
            // Natural Sciences
            {"LOOKUP science isa", "process"},
            {"LOOKUP physics isa", "field_of_science"},
            {"LOOKUP chemistry isa", "field_of_science"},
            {"LOOKUP biology isa", "field_of_science"},
            // Computer Science
            {"LOOKUP cpp isa", "programming_language"},
            {"LOOKUP linux isa", "operating_system"}
        };

        for (int i = 0; i < testCases.length; i++) {
            String query = testCases[i][0];
            String expected = testCases[i][1];

            String raw = brain.executeBrainQL(handle, query);
            JSONObject j = new JSONObject(raw);

            boolean match = j.getBoolean("known") && expected.equals(j.getString("result"));
            System.out.println("[Test " + (i + 1) + "] " + query + " -> " + j.getString("result") + (match ? " (PASS)" : " (FAIL)"));
            if (match) passed++;
        }

        System.out.println("\n======================================================================");
        System.out.println("  Result: " + passed + "/" + total + " Conversational & Humanities Tests Passed!");
        System.out.println("======================================================================\n");

        if (passed != total) {
            System.exit(1);
        }
    }
}
