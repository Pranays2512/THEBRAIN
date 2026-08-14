package brain3;

import java.util.Arrays;
import java.util.List;

public class MouthToneTest {
    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("  🧠 Mouth Tone Regression Suite (LLM Compliance) 🧠");
        System.out.println("==================================================\n");

        LLMAdapter.OllamaClient ollama = new LLMAdapter.OllamaClient("qwen3:1.7B");
        LLMAdapter.BrainQLMouth mouth = new LLMAdapter.BrainQLMouth(ollama);

        List<String> payloads = Arrays.asList(
            // Known facts (LOOKUP or state)
            "{\"op\": \"[BRAINQL]\", \"subj\": \"france\", \"rel\": \"capital\", \"obj\": \"paris\", \"objType\": \"atomic\", \"result\": \"paris\", \"known\": true}",
            "{\"op\": \"CHAT\", \"subj\": \"self\", \"rel\": \"emotion\", \"obj\": \"happy\", \"objType\": \"atomic\", \"result\": \"self emotion happy\", \"known\": true}",
            
            // Unknown associative musings (think() fallback)
            "{\"op\": \"CHAT\", \"subj\": \"internal\", \"rel\": \"association\", \"obj\": \"dog jump fence\", \"objType\": \"freetext\", \"result\": \"internal association dog jump fence\", \"known\": false}",
            
            // Untested edge case: 0 words generated -> "silence"
            "{\"op\": \"CHAT\", \"subj\": \"internal\", \"rel\": \"association\", \"obj\": \"silence\", \"objType\": \"freetext\", \"result\": \"internal association silence\", \"known\": false}",
            
            // Edge Cases: Highly canonical facts marked known: false
            "{\"op\": \"CHAT\", \"subj\": \"internal\", \"rel\": \"association\", \"obj\": \"paris capital france\", \"objType\": \"freetext\", \"result\": \"internal association paris capital france\", \"known\": false}",
            "{\"op\": \"CHAT\", \"subj\": \"internal\", \"rel\": \"association\", \"obj\": \"water boil 100 degrees\", \"objType\": \"freetext\", \"result\": \"internal association water boil 100 degrees\", \"known\": false}",
            
            // Edge Cases: Subtler/moderate confidence facts marked known: false
            "{\"op\": \"CHAT\", \"subj\": \"internal\", \"rel\": \"association\", \"obj\": \"apples taste better oranges\", \"objType\": \"freetext\", \"result\": \"internal association apples taste better oranges\", \"known\": false}",
            "{\"op\": \"CHAT\", \"subj\": \"internal\", \"rel\": \"association\", \"obj\": \"water freezes faster hot\", \"objType\": \"freetext\", \"result\": \"internal association water freezes faster hot\", \"known\": false}",
            "{\"op\": \"CHAT\", \"subj\": \"internal\", \"rel\": \"association\", \"obj\": \"birds are dinosaurs\", \"objType\": \"freetext\", \"result\": \"internal association birds are dinosaurs\", \"known\": false}"
        );

        int totalPasses = 0;
        int N = 10;
        
        for (String json : payloads) {
            System.out.println("PAYLOAD: " + json);
            int passCount = 0;
            boolean isKnown = json.contains("\"known\": true");
            
            for (int i = 0; i < N; i++) {
                String output = mouth.renderResult(json);
                String lowerOut = output.toLowerCase();
                
                boolean hasHedge = lowerOut.contains("think") || lowerOut.contains("perhaps") || 
                                   lowerOut.contains("maybe") || lowerOut.contains("seem") || 
                                   lowerOut.contains("wonder") || lowerOut.contains("associate") || 
                                   lowerOut.contains("feel like") || lowerOut.contains("just") || 
                                   lowerOut.contains("could") || lowerOut.contains("might");
                                   
                if (isKnown && !hasHedge) passCount++;
                else if (!isKnown && hasHedge) passCount++;
                
                if (i == 0) { // print the first iteration for visualization
                    System.out.println("   [Iter 1] MOUTH  : " + output);
                }
            }
            
            System.out.println("   -> Pass Rate: " + passCount + "/" + N);
            if (passCount == N) totalPasses++;
            System.out.println();
        }
        
        System.out.println("Test Complete. " + totalPasses + "/" + payloads.size() + " payloads achieved 100% pass rate.");
        System.exit(totalPasses == payloads.size() ? 0 : 1);
    }
}
