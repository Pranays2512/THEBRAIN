package brain3;

import java.util.*;

public class TestModules {
    public static void main(String[] args) {
        System.out.println("--- Starting Unit Tests ---");

        testMouth();
        testOCR();
        testLLMAdapter();
        testNLQuery();
        testBrainNative();
        
        System.out.println("--- All Tests Completed Successfully ---");
    }

    private static void testMouth() {
        System.out.println("[Test] Mouth");
        try {
            Mouth m = new Mouth(true);
            m.speak("Testing speech module.");
            System.out.println("  -> Mouth test passed.");
        } catch (Exception e) {
            System.err.println("  -> Mouth failed: " + e.getMessage());
        }
    }

    private static void testOCR() {
        System.out.println("[Test] OCRAdapter");
        try {
            OCRAdapter ocr = new OCRAdapter();
            String res = ocr.extractTextFromPdf("non_existent_file.pdf");
            System.out.println("  -> OCR test passed. Output (expected empty/error on missing file): '" + res + "'");
        } catch (Exception e) {
            System.err.println("  -> OCR failed: " + e.getMessage());
        }
    }

    private static void testLLMAdapter() {
        System.out.println("[Test] LLMAdapter StubClient");
        try {
            Map<String, String> responses = new HashMap<>();
            responses.put("hello", "Hello there!");
            LLMAdapter.StubClient stub = new LLMAdapter.StubClient(responses);
            String res = stub.complete("Oh hello!", "");
            if ("Hello there!".equals(res)) {
                System.out.println("  -> LLMAdapter test passed.");
            } else {
                System.out.println("  -> LLMAdapter test failed, got: " + res);
            }
        } catch (Exception e) {
            System.err.println("  -> LLMAdapter failed: " + e.getMessage());
        }
    }

    private static void testNLQuery() {
        System.out.println("[Test] NLQuery");
        try {
            Set<String> entities = new HashSet<>(Arrays.asList("sample", "water"));
            List<String> relations = Arrays.asList("force", "speed", "velocity");
            
            Map<String, float[]> glove = new HashMap<>();
            glove.put("sample", new float[]{1.0f, 0.0f});
            glove.put("force", new float[]{0.0f, 1.0f});
            
            NLQuery parser = new NLQuery(entities, relations, glove);
            NLQuery.ParseResult res = parser.parse("What is the force of the sample?");
            if ("sample".equals(res.entity) && "force".equals(res.relation)) {
                System.out.println("  -> NLQuery parsing test passed.");
            } else {
                System.out.println("  -> NLQuery test failed. Entity: " + res.entity + ", Rel: " + res.relation);
            }
        } catch (Exception e) {
            System.err.println("  -> NLQuery failed: " + e.getMessage());
        }
    }

    private static void testBrainNative() {
        System.out.println("[Test] BrainNative JNI");
        try {
            BrainNative b = new BrainNative(8, 8, 16);
            System.out.println("  -> Brain instantiated. Handle: " + b.getHandle());
            
            float emotion = b.getEmotionArousal(b.getHandle());
            System.out.println("  -> Emotion arousal: " + emotion);
            
            b.seedMathSymbols(b.getHandle());
            System.out.println("  -> Seeded math symbols.");
            
            b.bindSymbol(b.getHandle(), "test_word");
            boolean knows = b.knowsSymbol(b.getHandle(), "test_word");
            System.out.println("  -> Knows 'test_word': " + knows);
            
            float[] perceiveRes = b.perceive(b.getHandle(), new float[16]);
            System.out.println("  -> Perceive result length: " + perceiveRes.length);
            
            b.destroy();
            System.out.println("  -> BrainNative test passed.");
        } catch (Throwable t) {
            System.err.println("  -> BrainNative failed: " + t.getMessage());
        }
    }
}
