package brain3;

public class TestChatSession {
    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("  🧠 Brain3 + LLM (qwen3) Cognitive Chat Demo 🧠");
        System.out.println("==================================================\n");

        System.out.println("[1] Waking up the Biological Core & LLM Interface...");
        System.loadLibrary("brainjni");
        BrainNative brain = new BrainNative(16, 16, 128);
        System.out.println("Loading 400,000 word GloVe semantic core...");
        brain.loadComponents(brain.getHandle(), 
            "./out/brain_fluent/predictor.bin", "./out/brain_fluent/language.bin", "./out/brain_fluent/som.bin", 
            "./out/brain_fluent/episodic.bin", "./out/brain_fluent/emotion.bin", "./out/brain_fluent/self.bin",
            "./out/brain_fluent/symbolic.bin", "./out/brain_fluent/binding.bin", "./out/brain_fluent/bg.bin",
            "./out/brain_fluent/procedures.bin", "./out/brain_fluent/hpred.bin");
        java.util.Map<String, String> stubMap = new java.util.HashMap<>();
        stubMap.put("solve 3x + 15", "[BrainQL] solve type=\"equation\" data=\"3x+15=45\"");
        stubMap.put("integral of 4x^3", "[BrainQL] solve type=\"integrate\" data=\"4*x^3\"");
        stubMap.put("sky is blue", "[BrainQL] learn subj=\"sky\" rel=\"has_color\" obj=\"blue\"");
        
        // Mock Mouth output
        stubMap.put("{\"equation\":\"3x+15=45\",\"result\":\"10\"", "The answer is x = 10.");
        stubMap.put("{\"integrate\":\"4*x^3\",\"result\":\"1x^4\"", "The integral evaluates to 1*x^4.");
        stubMap.put("sky has_color blue", "I have remembered that the sky is blue.");
        
        // BrainQL Eyes stub
        stubMap.put("how are you today", "CHAT how are you today");
        stubMap.put("what is your name", "CHAT what is your name");
        stubMap.put("hello", "CHAT hello");
        
        // Mouth LLM stub
        stubMap.put("status optimal emotion happy", "I am functioning optimally and feeling very happy today.");
        stubMap.put("identity system type cognitive", "I am a cognitive biological AI system.");
        stubMap.put("intent greeting style friendly", "Hello! It is wonderful to meet you.");

        LLMAdapter.LLMClient llm = new LLMAdapter.StubClient(stubMap);
        Mouth mouth = new Mouth(true);
        NLQuery lexical = new NLQuery(new java.util.HashSet<>(), new java.util.ArrayList<>(), new java.util.HashMap<>());
        NLFront front = new NLFront(lexical);
        BrainInterface bi = new BrainInterface(llm, brain, front, mouth);

        String[] dialogue = {
            "Can you solve 3x + 15 = 45 for me?",
            "hello",
            "how are you today",
            "what is your name"
        };

        System.out.println("\n[2] Starting dialogue session...\n");

        for (String q : dialogue) {
            System.out.println("👤 USER: " + q);
            try {
                long start = System.currentTimeMillis();
                String reply = bi.respondStr(q);
                long elapsed = System.currentTimeMillis() - start;
                System.out.println("🤖 BRAIN: " + reply + "  [took " + elapsed + "ms]\n");
            } catch (Exception e) {
                System.err.println("🤖 BRAIN ERROR: " + e.getMessage() + "\n");
            }
        }

        brain.destroy();
        System.out.println("Session complete. Brain safely powered down.");
    }
}
