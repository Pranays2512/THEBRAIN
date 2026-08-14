import brain3.*;

public class TestChatIntegration {
    public static void main(String[] args) {
        System.out.println("Starting Full Native Chat LLM Integration Test...");
        System.loadLibrary("brainjni");
        
        BrainNative brain = new BrainNative(16, 16, 128);
        System.out.println("Loading 400,000 word GloVe semantic core...");
        brain.loadComponents(brain.getHandle(), 
            "./out/brain_fluent/predictor.bin", "./out/brain_fluent/language.bin", "./out/brain_fluent/som.bin", 
            "./out/brain_fluent/episodic.bin", "./out/brain_fluent/emotion.bin", "./out/brain_fluent/self.bin",
            "./out/brain_fluent/symbolic.bin", "./out/brain_fluent/binding.bin", "./out/brain_fluent/bg.bin",
            "./out/brain_fluent/procedures.bin", "./out/brain_fluent/hpred.bin");
            
        // Setup NLP and LLM
        LLMAdapter.OllamaClient llm = new LLMAdapter.OllamaClient("qwen3:1.7B");
        Mouth mouth = new Mouth(true);
        NLQuery lexical = new NLQuery(new java.util.HashSet<>(), new java.util.ArrayList<>(), new java.util.HashMap<>());
        NLFront front = new NLFront(lexical);
        
        BrainInterface bi = new BrainInterface(llm, brain, front, mouth);
        
        String input = "Hello my friend! I am having a great day today.";
        System.out.println("\n[USER INPUT]: " + input);
        System.out.println("\n(Waiting for LLM Eyes -> C++ Core -> LLM Mouth...)\n");
        
        try {
            String reply = bi.respondStr(input);
            System.out.println("\n[LLM MOUTH (Translated from C++ Core)]: " + reply);
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        brain.destroy();
    }
}
