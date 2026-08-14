package brain3;

import java.util.Scanner;

public class MainRepl {

    public static void main(String[] args) {
        System.out.println("Initializing Brain v3 Java REPL...");

        // Setup the JNI Brain with 128D (GloVe)
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
        
        // Setup the NL pipeline (skipping Glove loading in this stub)
        NLQuery lexical = new NLQuery(new java.util.HashSet<>(), new java.util.ArrayList<>(), new java.util.HashMap<>());
        NLFront front = new NLFront(lexical);

        BrainInterface bi = new BrainInterface(llm, brain, front, mouth);

        System.out.println("\n==================================================");
        System.out.println(" Brain v3 Cognitive Loop is online (Java JNI)!");
        System.out.println(" It will Perceive your text, Think, and Speak.");
        System.out.println(" Type 'quit' to exit.");
        System.out.println("==================================================\n");

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("You: ");
            if (!scanner.hasNextLine()) break;
            
            String input = scanner.nextLine().trim();
            if (input.equalsIgnoreCase("quit") || input.equalsIgnoreCase("exit")) {
                break;
            }
            if (input.isEmpty()) {
                continue;
            }

            try {
                String reply = bi.respondStr(input);
                System.out.println("Brain: " + reply + "\n");
            } catch (Exception e) {
                System.err.println("[error] " + e.getMessage() + "\n");
            }
        }
        
        brain.destroy();
        System.out.println("Brain state destroyed. Shutting down.");
    }
}
