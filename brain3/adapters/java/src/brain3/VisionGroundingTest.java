package brain3;

import org.json.JSONObject;
import java.io.FileOutputStream;
import java.io.File;

public class VisionGroundingTest {

    private static void generateTestImage(String path) throws Exception {
        int w = 300, h = 300;
        FileOutputStream fos = new FileOutputStream(path);
        String header = "P6\n" + w + " " + h + "\n255\n";
        fos.write(header.getBytes());

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                // Red square on left
                if (x >= 30 && x <= 100 && y >= 100 && y <= 170) {
                    fos.write(new byte[]{(byte)255, 0, 0});
                }
                // Blue circle on right
                else if (Math.pow(x - 220, 2) + Math.pow(y - 150, 2) <= 1600) {
                    fos.write(new byte[]{0, 0, (byte)255});
                }
                // Green dot in center
                else if (Math.pow(x - 150, 2) + Math.pow(y - 50, 2) <= 100) {
                    fos.write(new byte[]{0, (byte)255, 0});
                }
                // White background
                else {
                    fos.write(new byte[]{(byte)255, (byte)255, (byte)255});
                }
            }
        }
        fos.close();
    }

    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("  👁️ Visual Grounding & Sensory Feature Binding 👁️");
        System.out.println("==================================================\n");

        String imgPath = "test_vision_grounding.ppm";
        try {
            generateTestImage(imgPath);
        } catch (Exception e) {
            System.err.println("Failed to create test image: " + e.getMessage());
            System.exit(1);
        }

        System.loadLibrary("brainjni");
        BrainNative brain = new BrainNative(16, 16, 128);
        long handle = brain.getHandle();

        int passed = 0;
        int total = 5;

        // 1. Perceive Image
        System.out.println("[Test 1] Visual Perception: PERCEIVE_IMAGE " + imgPath);
        String res1 = brain.executeBrainQL(handle, "PERCEIVE_IMAGE " + imgPath);
        System.out.println("  Output: " + res1);
        JSONObject j1 = new JSONObject(res1);
        if ("PERCEIVE_IMAGE".equals(j1.getString("op")) &&
            j1.getBoolean("verified") == true &&
            "vision_engine".equals(j1.getString("source")) &&
            Integer.parseInt(j1.getString("obj")) >= 2) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // 2. Query Blob 1 Color & Position (Green dot encountered first at y=50)
        System.out.println("[Test 2] Grounded Query: LOOKUP blob_1 color");
        String res2 = brain.executeBrainQL(handle, "LOOKUP blob_1 color");
        System.out.println("  Output: " + res2);
        JSONObject j2 = new JSONObject(res2);
        if (j2.getBoolean("known") == true && "green".equals(j2.getString("result"))) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // 3. Query Blob 2 Color & Position (Red square encountered at y=100)
        System.out.println("[Test 3] Grounded Query: LOOKUP blob_2 color & position");
        String res3 = brain.executeBrainQL(handle, "LOOKUP blob_2 color");
        String res3b = brain.executeBrainQL(handle, "LOOKUP blob_2 position");
        System.out.println("  Output color: " + res3 + " | pos: " + res3b);
        JSONObject j3 = new JSONObject(res3);
        JSONObject j3b = new JSONObject(res3b);
        if (j3.getBoolean("known") == true && "red".equals(j3.getString("result")) &&
            j3b.getBoolean("known") == true && "left".equals(j3b.getString("result"))) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // 4. Query Blob 3 Color & Position (Blue circle encountered at y=150)
        System.out.println("[Test 4] Grounded Query: LOOKUP blob_3 color & position");
        String res4 = brain.executeBrainQL(handle, "LOOKUP blob_3 color");
        String res4b = brain.executeBrainQL(handle, "LOOKUP blob_3 position");
        System.out.println("  Output color: " + res4 + " | pos: " + res4b);
        JSONObject j4 = new JSONObject(res4);
        JSONObject j4b = new JSONObject(res4b);
        if (j4.getBoolean("known") == true && "blue".equals(j4.getString("result")) &&
            j4b.getBoolean("known") == true && "right".equals(j4b.getString("result"))) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // 5. Query Blob 1 Position
        System.out.println("[Test 5] Spatial Query: LOOKUP blob_1 position");
        String res5 = brain.executeBrainQL(handle, "LOOKUP blob_1 position");
        System.out.println("  Output: " + res5);
        JSONObject j5 = new JSONObject(res5);
        if (j5.getBoolean("known") == true && "center".equals(j5.getString("result"))) {
            System.out.println("  -> PASS\n");
            passed++;
        } else {
            System.out.println("  -> FAIL\n");
        }

        // Cleanup
        new File(imgPath).delete();

        System.out.println("==================================================");
        System.out.println("  Result: " + passed + "/" + total + " Visual Grounding Tests Passed!");
        System.out.println("==================================================");

        if (passed != total) {
            System.exit(1);
        }
    }
}
