package brain3;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;
import org.json.JSONArray;

/**
 * High-throughput pipe server for streaming dataset ingestion and BrainQL execution.
 * Communicates via stdin/stdout using JSON protocol.
 */
public class BrainPipeServer {

    public static void main(String[] args) {
        // Initialize native Brain with 16x16 SOM and 128D embeddings
        BrainNative brain = new BrainNative(16, 16, 128);
        long handle = brain.getHandle();

        System.err.println("[BrainPipeServer] Initialized Brain 3 JNI handle: " + handle);
        System.out.println(new JSONObject().put("status", "ready").put("handle", handle).toString());
        System.out.flush();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                if (line.equalsIgnoreCase("QUIT") || line.equalsIgnoreCase("EXIT")) {
                    break;
                }

                try {
                    JSONObject req;
                    if (line.startsWith("{")) {
                        req = new JSONObject(line);
                    } else {
                        req = new JSONObject();
                        req.put("cmd", "bql");
                        req.put("query", line);
                    }

                    String cmd = req.optString("cmd", "bql");
                    JSONObject res = new JSONObject();

                    if (cmd.equalsIgnoreCase("bql")) {
                        String query = req.getString("query");
                        String rawRes = brain.executeBrainQL(handle, query);
                        res.put("status", "ok");
                        res.put("query", query);
                        res.put("result", rawRes);
                    } else if (cmd.equalsIgnoreCase("batch")) {
                        JSONArray queries = req.getJSONArray("queries");
                        JSONArray results = new JSONArray();
                        int success = 0;
                        for (int i = 0; i < queries.length(); i++) {
                            String q = queries.getString(i);
                            String raw = brain.executeBrainQL(handle, q);
                            results.put(raw);
                            if (raw.contains("\"verified\": true") || raw.contains("\"verified\":true") ||
                                raw.contains("\"known\": true") || raw.contains("\"known\":true") ||
                                raw.contains("\"success\": true") || raw.contains("\"success\":true")) {
                                success++;
                            }
                        }
                        res.put("status", "ok");
                        res.put("total", queries.length());
                        res.put("success", success);
                        res.put("results", results);
                    } else if (cmd.equalsIgnoreCase("sleep")) {
                        String gateLog = req.optString("gateLog", "associative_gate.jsonl");
                        String checkpoint = req.optString("checkpoint", "./out/brain_fluent");
                        String report = brain.sleep(handle, gateLog, checkpoint);
                        res.put("status", "ok");
                        res.put("report", report);
                    } else if (cmd.equalsIgnoreCase("ping")) {
                        res.put("status", "ok");
                        res.put("pong", true);
                    } else {
                        res.put("status", "error");
                        res.put("error", "Unknown command: " + cmd);
                    }

                    System.out.println(res.toString());
                    System.out.flush();

                } catch (Exception e) {
                    JSONObject err = new JSONObject();
                    err.put("status", "error");
                    err.put("error", e.getMessage());
                    System.out.println(err.toString());
                    System.out.flush();
                }
            }
        } catch (Exception e) {
            System.err.println("[BrainPipeServer] Fatal error: " + e.getMessage());
        } finally {
            brain.destroy();
            System.err.println("[BrainPipeServer] Brain handle destroyed. Clean exit.");
        }
    }
}
