import brain3.BrainNative;

public class TestNativeChat {
    public static void main(String[] args) {
        System.out.println("Starting Native Chat Test...");
        System.loadLibrary("brainjni");
        BrainNative brain = new BrainNative(16, 16, 128);
        System.out.println("Loading 400,000 word GloVe semantic core...");
        brain.loadComponents(brain.getHandle(), 
            "./out/brain_fluent/predictor.bin", "./out/brain_fluent/language.bin", "./out/brain_fluent/som.bin", 
            "./out/brain_fluent/episodic.bin", "./out/brain_fluent/emotion.bin", "./out/brain_fluent/self.bin",
            "./out/brain_fluent/symbolic.bin", "./out/brain_fluent/binding.bin", "./out/brain_fluent/bg.bin",
            "./out/brain_fluent/procedures.bin", "./out/brain_fluent/hpred.bin");
        
        System.out.println("\n--- Brain is ready! Type your message (or 'exit' to quit) ---");
        java.util.Scanner scanner = new java.util.Scanner(System.in);
        while (true) {
            System.out.print("You: ");
            if (!scanner.hasNextLine()) break;
            String input = scanner.nextLine().trim();
            if (input.equalsIgnoreCase("exit") || input.equalsIgnoreCase("quit")) break;
            if (input.isEmpty()) continue;

            String result = brain.executeBrainQL(brain.getHandle(), "CHAT " + input);
            
            // Try to extract just the 'result' value from the JSON
            String reply = result;
            if (result.contains("\"result\": \"")) {
                int start = result.indexOf("\"result\": \"") + 11;
                int end = result.indexOf("\"", start);
                if (end > start) reply = result.substring(start, end);
            }
            
            System.out.println("Brain: " + reply);
        }
        System.out.println("Goodbye!");
    }
}
