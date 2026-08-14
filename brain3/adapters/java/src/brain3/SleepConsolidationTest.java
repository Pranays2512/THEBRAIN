package brain3;

import org.json.JSONObject;
import java.io.File;
import java.io.FileWriter;

public class SleepConsolidationTest {
    public static void main(String[] args) {
        System.out.println("=============================================================");
        System.out.println("  🧠 JAVA JNI SLEEP CONSOLIDATION INTEGRATION TEST");
        System.out.println("=============================================================\n");

        System.loadLibrary("brainjni");
        BrainNative nativeBrain = new BrainNative(16, 16, 128);
        long handle = nativeBrain.getHandle();

        // 1. Setup simulated gate log
        String logPath = "test_jni_gate.jsonl";
        try (FileWriter fw = new FileWriter(logPath)) {
            fw.write("{\"timestamp\": 1786600001, \"inputs\": [\"france\", \"capital\"], \"guess\": \"paris\", \"store_truth\": \"paris\", \"verdict\": \"verified_atomic\"}\n");
            fw.write("{\"timestamp\": 1786600002, \"inputs\": [\"lion\", \"isa\"], \"guess\": \"carnivore\", \"store_truth\": \"carnivore\", \"verdict\": \"verified_atomic\"}\n");
        } catch (Exception e) {
            e.printStackTrace();
        }

        // 2. Call native sleep method via JNI
        String resultJson = nativeBrain.sleep(handle, logPath, "./out/test_jni_sleep");
        System.out.println("Native Sleep Result JSON:\n" + resultJson + "\n");

        JSONObject obj = new JSONObject(resultJson);
        int phase2Records = obj.getInt("phase2_records");
        int phase2Triples = obj.getInt("phase2_triples_trained");
        int phase3Som = obj.getInt("phase3_som_decayed");
        boolean phase4Ckpt = obj.getBoolean("phase4_checkpoint");

        System.out.println("Parsed JNI Report Metrics:");
        System.out.println("  • Phase 2 Records Ingested: " + phase2Records);
        System.out.println("  • Phase 2 Triples Trained:  " + phase2Triples);
        System.out.println("  • Phase 3 SOM Decayed:      " + phase3Som);
        System.out.println("  • Phase 4 Checkpoint:       " + phase4Ckpt);

        if (phase2Records == 2 && phase2Triples == 2 && phase3Som > 0 && phase4Ckpt) {
            System.out.println("\n✓ ALL JNI SLEEP CONSOLIDATION CHECKS PASSED!");
        } else {
            System.err.println("\n✗ JNI SLEEP CHECKS FAILED!");
            System.exit(1);
        }

        // Clean up
        new File(logPath).delete();
        nativeBrain.destroy();
    }
}
