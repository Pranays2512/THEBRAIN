package brain3;

public class BrainNative {
    static {
        System.loadLibrary("brainjni");
    }

    private long nativeHandle; // Pointer to the C++ Brain instance

    public BrainNative(int somRows, int somCols, int nDims) {
        nativeHandle = init(somRows, somCols, nDims);
    }

    public void destroy() {
        if (nativeHandle != 0) {
            destroy(nativeHandle);
            nativeHandle = 0;
        }
    }

    // Lifecycle
    private native long init(int somRows, int somCols, int nDims);
    private native void destroy(long handle);

    public native void loadComponents(long handle, 
        String predictorPath, String languagePath, String somPath,
        String episodicPath, String emotionPath, String selfPath,
        String symbolicPath, String bindingPath, String bgPath,
        String proceduresPath, String hpredPath);

    public native void saveComponents(long handle, String dirPath);

    // Language & Symbolic
    public native void seedMathSymbols(long handle);
    public native void bindSymbol(long handle, String word);
    public native boolean knowsSymbol(long handle, String word);
    
    public native void registerWord(long handle, String word);
    public native boolean knowsWord(long handle, String word);
    public native float[] encodeWord(long handle, String word);
    public native String bestWord(long handle, float[] vec);

    // Emotion & Self
    public native float getEmotionArousal(long handle);
    public native float getEmotionValence(long handle);
    public native float getSelfMeanRecentError(long handle);

    // Episodic
    public native float[] getLastEpisode(long handle);
    public native void commitEpisode(long handle, float salience, float[] vec);

    // Binding
    public native float[][] bindingQueryAll(long handle, float[] vec, float threshold);
    public native void bindingBind(long handle, float[] subj, float[] rel, float[] obj);

    // Scratchpad
    public native void scratchpadClear(long handle);
    public native void scratchpadWrite(long handle, String slotName, float[] vec, String historyCtx);
    public native float[] scratchpadRead(long handle, String slotName);

    // Reasoning
    public native void startReasoning(long handle);
    public native void forceReasonStep(long handle, int op, String context);
    public native String executeBrainQL(long handle, String query);

    // IO
    public native String[] getSpokenWords(long handle);
    public native void clearSpokenWords(long handle);

    // Procedures
    public native int[] proceduresRetrieve(long handle, float[] vec);

    // Working Memory
    public native void workingMemGate(long handle, float[] vec, float salience);
    public native void workingMemTick(long handle);
    public native float[] workingMemContext(long handle);

    // SOM
    public native float[] somActivationMap(long handle, float[] vec);

    // Main
    public native void daydream(long handle);
    public native String sleep(long handle, String gateLogPath, String checkpointDir);
    public native float[] perceive(long handle, float[] vec);
    public native float getLastConfidence(long handle);
    public native void resetSequence(long handle);

    // Get the native handle to pass to methods implicitly if we want to wrap it better
    public long getHandle() {
        return nativeHandle;
    }
}
