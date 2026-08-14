package brain3.faculties;

import brain3.BrainNative;

/**
 * WholeBrain is the decoupled, lightweight orchestration layer.
 * It does not hold the 75KB logic; instead it delegates to the BrainNative JNI,
 * which in turn executes the modular C++ crisp/fuzzy engines (math, reasoning, synthesis, conversation).
 */
public class WholeBrain {
    
    private BrainNative nativeBrain;
    private boolean isAsleep;

    public WholeBrain(int somRows, int somCols, int nDims) {
        // Initialize the native C++ Brain
        this.nativeBrain = new BrainNative(somRows, somCols, nDims);
        this.isAsleep = false;
    }

    public void load(String predictorPath, String languagePath, String somPath,
                     String episodicPath, String emotionPath, String selfPath,
                     String symbolicPath, String bindingPath, String bgPath,
                     String proceduresPath, String hpredPath) {
        nativeBrain.loadComponents(
            nativeBrain.getHandle(),
            predictorPath, languagePath, somPath, episodicPath, emotionPath, 
            selfPath, symbolicPath, bindingPath, bgPath, proceduresPath, hpredPath
        );
    }

    public void save(String dirPath) {
        nativeBrain.saveComponents(nativeBrain.getHandle(), dirPath);
    }

    /**
     * Primary orchestration method for interaction.
     * Takes text input and triggers the native C++ ConversationEngine 
     * (and other processing pipelines).
     */
    public String respond(String text) {
        if (isAsleep) return "Zzz...";
        
        // Ensure words are registered in the native vocabulary
        for (String word : text.toLowerCase().replaceAll("[^a-z ]", "").split(" ")) {
            if (!word.isEmpty() && !nativeBrain.knowsWord(nativeBrain.getHandle(), word)) {
                nativeBrain.registerWord(nativeBrain.getHandle(), word);
            }
        }
        
        // Pass perception down to C++ (this triggers the crisp faculties natively)
        // Use the native BrainQL engine for execution
        return nativeBrain.executeBrainQL(nativeBrain.getHandle(), text);
    }

    /**
     * Infinite background daydreaming loop, triggering Fuzzy engines (hypothesis generation, SOM updates).
     */
    public void daydream() {
        this.isAsleep = true;
        nativeBrain.daydream(nativeBrain.getHandle());
        this.isAsleep = false;
    }

    /**
     * Master 4-phase sleep consolidation:
     * Phase 1: Crisp Rule Induction & Default Logic
     * Phase 2: Neural STaR Self-Training from Gate Telemetry
     * Phase 3: Topological & Episodic Memory Pruning
     * Phase 4: Atomic Checkpointing
     */
    public String sleep(String gateLogPath, String checkpointDir) {
        this.isAsleep = true;
        String report = nativeBrain.sleep(nativeBrain.getHandle(), gateLogPath, checkpointDir);
        this.isAsleep = false;
        return report;
    }

    public String sleep() {
        return sleep("associative_gate.jsonl", "./out/brain_fluent");
    }

    public void shutdown() {
        nativeBrain.destroy();
    }
}
