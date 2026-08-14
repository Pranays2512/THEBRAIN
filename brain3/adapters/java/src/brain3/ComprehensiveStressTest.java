package brain3;

import org.json.JSONObject;
import java.util.ArrayList;
import java.util.List;

public class ComprehensiveStressTest {

    static class TestCase {
        String category;
        String query;
        String expectedOp;
        String expectedSnippet;
        boolean expectedKnown;

        TestCase(String category, String query, String expectedOp, String expectedSnippet, boolean expectedKnown) {
            this.category = category;
            this.query = query;
            this.expectedOp = expectedOp;
            this.expectedSnippet = expectedSnippet;
            this.expectedKnown = expectedKnown;
        }
    }

    public static void main(String[] args) {
        System.out.println("======================================================================");
        System.out.println("     🧠 BRAIN3 COMPREHENSIVE 100-QUESTION REASONING BENCHMARK 🧠");
        System.out.println("======================================================================\n");

        System.loadLibrary("brainjni");
        BrainNative brain = new BrainNative(16, 16, 128);
        long handle = brain.getHandle();

        // Populate baseline taxonomy and facts in KB
        brain.executeBrainQL(handle, "TEACH dog isa mammal");
        brain.executeBrainQL(handle, "TEACH mammal isa animal");
        brain.executeBrainQL(handle, "TEACH animal isa organism");
        brain.executeBrainQL(handle, "TEACH organism isa entity");
        brain.executeBrainQL(handle, "TEACH bird isa animal");
        brain.executeBrainQL(handle, "TEACH bird can fly");
        brain.executeBrainQL(handle, "TEACH sparrow isa bird");
        brain.executeBrainQL(handle, "TEACH eagle isa bird");
        brain.executeBrainQL(handle, "TEACH eagle can hunt");
        brain.executeBrainQL(handle, "TEACH penguin isa bird");
        brain.executeBrainQL(handle, "TEACH cat isa mammal");
        brain.executeBrainQL(handle, "TEACH lion isa mammal");
        brain.executeBrainQL(handle, "TEACH tiger isa mammal");
        brain.executeBrainQL(handle, "TEACH dolphin isa mammal");
        brain.executeBrainQL(handle, "TEACH whale isa mammal");
        brain.executeBrainQL(handle, "TEACH shark isa fish");
        brain.executeBrainQL(handle, "TEACH fish isa animal");
        brain.executeBrainQL(handle, "TEACH trout isa fish");
        brain.executeBrainQL(handle, "TEACH rose isa flower");
        brain.executeBrainQL(handle, "TEACH flower isa plant");
        brain.executeBrainQL(handle, "TEACH plant isa organism");
        brain.executeBrainQL(handle, "TEACH oak isa tree");
        brain.executeBrainQL(handle, "TEACH tree isa plant");
        brain.executeBrainQL(handle, "TEACH apple isa fruit");
        brain.executeBrainQL(handle, "TEACH banana isa fruit");
        brain.executeBrainQL(handle, "TEACH carrot isa vegetable");
        brain.executeBrainQL(handle, "TEACH vegetable isa plant");
        brain.executeBrainQL(handle, "TEACH gold isa metal");
        brain.executeBrainQL(handle, "TEACH iron isa metal");
        brain.executeBrainQL(handle, "TEACH copper isa metal");
        brain.executeBrainQL(handle, "TEACH metal isa element");

        // Science Facts
        brain.executeBrainQL(handle, "TEACH water formula H2O");
        brain.executeBrainQL(handle, "TEACH oxygen symbol O");
        brain.executeBrainQL(handle, "TEACH hydrogen symbol H");
        brain.executeBrainQL(handle, "TEACH carbon symbol C");
        brain.executeBrainQL(handle, "TEACH nitrogen symbol N");
        brain.executeBrainQL(handle, "TEACH sun isa star");
        brain.executeBrainQL(handle, "TEACH earth isa planet");
        brain.executeBrainQL(handle, "TEACH mars isa planet");
        brain.executeBrainQL(handle, "TEACH jupiter isa planet");
        brain.executeBrainQL(handle, "TEACH moon isa satellite");
        brain.executeBrainQL(handle, "TEACH photon isa particle");
        brain.executeBrainQL(handle, "TEACH electron charge negative");
        brain.executeBrainQL(handle, "TEACH proton charge positive");
        brain.executeBrainQL(handle, "TEACH neutron charge neutral");
        brain.executeBrainQL(handle, "TEACH nucleus contains proton");
        brain.executeBrainQL(handle, "TEACH dna contains gene");
        brain.executeBrainQL(handle, "TEACH mitochondria function energy");
        brain.executeBrainQL(handle, "TEACH ribosome function protein");
        brain.executeBrainQL(handle, "TEACH neuron function signaling");

        List<TestCase> tests = new ArrayList<>();

        // ── CATEGORY 1: Taxonomic Reasoning & Transitive Inheritance (20 tests) ──
        tests.add(new TestCase("Taxonomy", "INHERIT dog isa", "INHERIT", "animal", true));
        tests.add(new TestCase("Taxonomy", "INHERIT sparrow can", "INHERIT", "fly", true));
        tests.add(new TestCase("Taxonomy", "INHERIT cat isa", "INHERIT", "animal", true));
        tests.add(new TestCase("Taxonomy", "INHERIT lion isa", "INHERIT", "animal", true));
        tests.add(new TestCase("Taxonomy", "INHERIT tiger isa", "INHERIT", "animal", true));
        tests.add(new TestCase("Taxonomy", "INHERIT dolphin isa", "INHERIT", "animal", true));
        tests.add(new TestCase("Taxonomy", "INHERIT whale isa", "INHERIT", "animal", true));
        tests.add(new TestCase("Taxonomy", "INHERIT shark isa", "INHERIT", "animal", true));
        tests.add(new TestCase("Taxonomy", "INHERIT trout isa", "INHERIT", "animal", true));
        tests.add(new TestCase("Taxonomy", "INHERIT rose isa", "INHERIT", "entity", true));
        tests.add(new TestCase("Taxonomy", "INHERIT oak isa", "INHERIT", "entity", true));
        tests.add(new TestCase("Taxonomy", "INHERIT apple isa", "INHERIT", "fruit", true));
        tests.add(new TestCase("Taxonomy", "INHERIT banana isa", "INHERIT", "fruit", true));
        tests.add(new TestCase("Taxonomy", "INHERIT carrot isa", "INHERIT", "entity", true));
        tests.add(new TestCase("Taxonomy", "INHERIT gold isa", "INHERIT", "element", true));
        tests.add(new TestCase("Taxonomy", "INHERIT iron isa", "INHERIT", "element", true));
        tests.add(new TestCase("Taxonomy", "INHERIT copper isa", "INHERIT", "element", true));
        tests.add(new TestCase("Taxonomy", "CHAIN dog isa", "CHAIN", "mammal", true));
        tests.add(new TestCase("Taxonomy", "CHAIN sparrow isa", "CHAIN", "bird", true));
        tests.add(new TestCase("Taxonomy", "CHAIN rose isa", "CHAIN", "flower", true));

        // ── CATEGORY 2: Science & Fact Retrieval (20 tests) ──
        tests.add(new TestCase("Science", "LOOKUP water formula", "LOOKUP", "H2O", true));
        tests.add(new TestCase("Science", "LOOKUP oxygen symbol", "LOOKUP", "O", true));
        tests.add(new TestCase("Science", "LOOKUP hydrogen symbol", "LOOKUP", "H", true));
        tests.add(new TestCase("Science", "LOOKUP carbon symbol", "LOOKUP", "C", true));
        tests.add(new TestCase("Science", "LOOKUP nitrogen symbol", "LOOKUP", "N", true));
        tests.add(new TestCase("Science", "LOOKUP sun isa", "LOOKUP", "star", true));
        tests.add(new TestCase("Science", "LOOKUP earth isa", "LOOKUP", "planet", true));
        tests.add(new TestCase("Science", "LOOKUP mars isa", "LOOKUP", "planet", true));
        tests.add(new TestCase("Science", "LOOKUP jupiter isa", "LOOKUP", "planet", true));
        tests.add(new TestCase("Science", "LOOKUP moon isa", "LOOKUP", "satellite", true));
        tests.add(new TestCase("Science", "LOOKUP photon isa", "LOOKUP", "particle", true));
        tests.add(new TestCase("Science", "LOOKUP electron charge", "LOOKUP", "negative", true));
        tests.add(new TestCase("Science", "LOOKUP proton charge", "LOOKUP", "positive", true));
        tests.add(new TestCase("Science", "LOOKUP neutron charge", "LOOKUP", "neutral", true));
        tests.add(new TestCase("Science", "LOOKUP nucleus contains", "LOOKUP", "proton", true));
        tests.add(new TestCase("Science", "LOOKUP dna contains", "LOOKUP", "gene", true));
        tests.add(new TestCase("Science", "LOOKUP mitochondria function", "LOOKUP", "energy", true));
        tests.add(new TestCase("Science", "LOOKUP ribosome function", "LOOKUP", "protein", true));
        tests.add(new TestCase("Science", "LOOKUP neuron function", "LOOKUP", "signaling", true));
        tests.add(new TestCase("Science", "LOOKUP eagle can", "LOOKUP", "hunt", true));

        // ── CATEGORY 3: Symbolic Math & Algebra (20 tests) ──
        tests.add(new TestCase("Math/Algebra", "SOLVE 2*x + 3 = 7", "SOLVE", "2", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 5*x = 20", "SOLVE", "4", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 3*x + 6 = 18", "SOLVE", "4", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 4*x - 8 = 0", "SOLVE", "2", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 10*x = 100", "SOLVE", "10", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 2*x + 10 = 30", "SOLVE", "10", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 7*x = 49", "SOLVE", "7", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 6*x + 12 = 24", "SOLVE", "2", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 8*x = 64", "SOLVE", "8", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 9*x - 27 = 0", "SOLVE", "3", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 3*x + 15 = 30", "SOLVE", "5", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 12*x = 144", "SOLVE", "12", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 5*x + 25 = 50", "SOLVE", "5", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 2*x - 4 = 16", "SOLVE", "10", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 11*x = 121", "SOLVE", "11", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 4*x + 20 = 40", "SOLVE", "5", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 6*x = 36", "SOLVE", "6", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 15*x = 45", "SOLVE", "3", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 2*x + 1 = 11", "SOLVE", "5", true));
        tests.add(new TestCase("Math/Algebra", "SOLVE 7*x - 14 = 0", "SOLVE", "2", true));

        // ── CATEGORY 4: Symbolic Calculus & Physics Laws (20 tests) ──
        tests.add(new TestCase("Calculus/Physics", "SOLVE diff x^3", "SOLVE", "3*x^2", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE diff x^4", "SOLVE", "4*x^3", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE diff x^5", "SOLVE", "5*x^4", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE diff x^2", "SOLVE", "2*x", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE int x^3", "SOLVE", "x^4/4", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE int x^2", "SOLVE", "x^3/3", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE int x^4", "SOLVE", "x^5/5", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE mass 10 accel 3 force", "SOLVE", "30", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE mass 5 accel 4 force", "SOLVE", "20", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE speed 20 time 5 distance", "SOLVE", "100", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE speed 15 time 3 distance", "SOLVE", "45", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE mass 4 velocity 5 momentum", "SOLVE", "20", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE mass 2 velocity 10 momentum", "SOLVE", "20", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE mass 10 volume 2 density", "SOLVE", "5", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE mass 20 volume 4 density", "SOLVE", "5", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE current 2 resistance 5 voltage", "SOLVE", "10", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE current 3 resistance 4 voltage", "SOLVE", "12", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE work 100 time 10 power", "SOLVE", "10", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE work 50 time 5 power", "SOLVE", "10", true));
        tests.add(new TestCase("Calculus/Physics", "SOLVE A car travels at speed 20 for time 5. Find distance.", "SOLVE", "100", true));

        // ── CATEGORY 5: Program Synthesis (10 tests) ──
        tests.add(new TestCase("Synthesis", "SYNTH [1, 2, 3] -> [2, 4, 6]", "SYNTH", "x * 2", true));
        tests.add(new TestCase("Synthesis", "SYNTH [2, 4, 6] -> [4, 8, 12]", "SYNTH", "x * 2", true));
        tests.add(new TestCase("Synthesis", "SYNTH [1, 2, 3] -> [2, 3, 4]", "SYNTH", "x + 1", true));
        tests.add(new TestCase("Synthesis", "SYNTH [5, 6, 7] -> [6, 7, 8]", "SYNTH", "x + 1", true));
        tests.add(new TestCase("Synthesis", "SYNTH [2, 3, 4] -> [1, 2, 3]", "SYNTH", "x - 1", true));
        tests.add(new TestCase("Synthesis", "SYNTH [3, -1, 2, -5] -> [3, 2]", "SYNTH", "x > 0", true));
        tests.add(new TestCase("Synthesis", "SYNTH [10, -2, 5, -8] -> [10, 5]", "SYNTH", "x > 0", true));
        tests.add(new TestCase("Synthesis", "SYNTH [1, 2, 3] -> [3, 2, 1]", "SYNTH", "reversed", true));
        tests.add(new TestCase("Synthesis", "SYNTH [4, 5, 6] -> [6, 5, 4]", "SYNTH", "reversed", true));
        tests.add(new TestCase("Synthesis", "SYNTH [9, 8, 7] -> [7, 8, 9]", "SYNTH", "reversed", true));

        // ── CATEGORY 6: Epistemic & Tone Compliance / Unknown Guard (10 tests) ──
        tests.add(new TestCase("Epistemic/Unknown", "LOOKUP unicorn habitat", "LOOKUP", "", false));
        tests.add(new TestCase("Epistemic/Unknown", "LOOKUP dragon color", "LOOKUP", "", false));
        tests.add(new TestCase("Epistemic/Unknown", "LOOKUP kryptonite element", "LOOKUP", "", false));
        tests.add(new TestCase("Epistemic/Unknown", "LOOKUP goblin weapon", "LOOKUP", "", false));
        tests.add(new TestCase("Epistemic/Unknown", "LOOKUP atlantis location", "LOOKUP", "", false));
        tests.add(new TestCase("Epistemic/Unknown", "INHERIT blorgon isa", "INHERIT", "", false));
        tests.add(new TestCase("Epistemic/Unknown", "INHERIT flurb can", "INHERIT", "", false));
        tests.add(new TestCase("Epistemic/Unknown", "INHERIT snark isa", "INHERIT", "", false));
        tests.add(new TestCase("Epistemic/Unknown", "INHERIT zorblax color", "INHERIT", "", false));
        tests.add(new TestCase("Epistemic/Unknown", "CHAIN non_existent_concept isa", "CHAIN", "", false));

        // ── CATEGORY 7: Means-Ends Goal Planning (5 tests) ──
        brain.executeBrainQL(handle, "TEACH rocket mass 10");
        brain.executeBrainQL(handle, "TEACH rocket accel 3");
        brain.executeBrainQL(handle, "TEACH rocket speed 20");
        brain.executeBrainQL(handle, "TEACH car distance 100");
        brain.executeBrainQL(handle, "TEACH car time 5");
        brain.executeBrainQL(handle, "TEACH circuit current 3");
        brain.executeBrainQL(handle, "TEACH circuit resistance 4");
        tests.add(new TestCase("Means-Ends Planning", "COMPUTE rocket force", "COMPUTE", "30", true));
        tests.add(new TestCase("Means-Ends Planning", "COMPUTE rocket power", "COMPUTE", "600", true));
        tests.add(new TestCase("Means-Ends Planning", "COMPUTE car speed", "COMPUTE", "20", true));
        tests.add(new TestCase("Means-Ends Planning", "COMPUTE circuit voltage", "COMPUTE", "12", true));
        tests.add(new TestCase("Means-Ends Planning", "COMPUTE non_existent_entity force", "COMPUTE", "", false));

        // ── CATEGORY 8: Episodic Story Narrative QA (5 tests) ──
        brain.executeBrainQL(handle, "TEACH alice_story went_to store");
        brain.executeBrainQL(handle, "TEACH alice_story bought apple");
        brain.executeBrainQL(handle, "TEACH bob_story went_to park");
        brain.executeBrainQL(handle, "TEACH bob_story found dog");
        brain.executeBrainQL(handle, "TEACH bob_story felt happy");
        tests.add(new TestCase("Story Narrative QA", "LOOKUP alice_story went_to", "LOOKUP", "store", true));
        tests.add(new TestCase("Story Narrative QA", "LOOKUP alice_story bought", "LOOKUP", "apple", true));
        tests.add(new TestCase("Story Narrative QA", "LOOKUP bob_story went_to", "LOOKUP", "park", true));
        tests.add(new TestCase("Story Narrative QA", "LOOKUP bob_story found", "LOOKUP", "dog", true));
        tests.add(new TestCase("Story Narrative QA", "LOOKUP bob_story felt", "LOOKUP", "happy", true));

        // Execute Benchmark
        int total = tests.size();
        int passed = 0;
        long startTime = System.currentTimeMillis();

        System.out.println("Running " + total + " benchmark questions across 6 core categories...\n");

        for (int i = 0; i < total; i++) {
            TestCase tc = tests.get(i);
            String rawRes = brain.executeBrainQL(handle, tc.query);
            JSONObject j = new JSONObject(rawRes);

            boolean opMatches = tc.expectedOp.equals(j.optString("op"));
            boolean knownMatches = (tc.expectedKnown == j.optBoolean("known"));
            String resStr = j.optString("result", "") + " " + j.optString("obj", "");
            boolean snippetMatches = tc.expectedSnippet.isEmpty() || resStr.contains(tc.expectedSnippet);

            boolean pass = opMatches && knownMatches && snippetMatches;
            if (pass) {
                passed++;
            } else {
                System.out.println("  [FAIL Q" + (i+1) + "] Category: " + tc.category + " | Query: " + tc.query);
                System.out.println("         Got: " + rawRes);
            }
        }

        long endTime = System.currentTimeMillis();
        double elapsedSec = (endTime - startTime) / 1000.0;
        double avgLatencyMs = (double)(endTime - startTime) / total;

        System.out.println("\n======================================================================");
        System.out.println("                     BENCHMARK SCORECARD");
        System.out.println("======================================================================");
        System.out.println("  • Total Questions:          " + total);
        System.out.println("  • Total Passed:             " + passed + "/" + total + " (" + String.format("%.1f", (passed * 100.0 / total)) + "%)");
        System.out.println("  • Execution Time:           " + String.format("%.2f", elapsedSec) + "s");
        System.out.println("  • Average Latency / Query:  " + String.format("%.2f", avgLatencyMs) + "ms");
        System.out.println("======================================================================\n");

        if (passed == total) {
            System.out.println("🌟 100% BENCHMARK PASS: Brain 3 reasoning, math, synthesis, and epistemic guard verified!");
            System.exit(0);
        } else {
            System.out.println("⚠️ Benchmark had failures.");
            System.exit(1);
        }
    }
}
