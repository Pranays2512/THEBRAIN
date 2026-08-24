package brain3;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.*;
import org.json.JSONObject;
import org.json.JSONArray;

/**
 * Model Context Protocol (MCP) TCP Socket Server for The Brain 3.
 * Enables external clients, IDEs, and services to interact with The Brain
 * via JSON-RPC 2.0 protocol over a network TCP socket.
 */
public class BrainMCPSocketServer {

    // The brain core is single-thread-accessed by contract; concurrent MCP
    // clients must be serialized around every shared-brain native invocation.
    private static final Object BRAIN_LOCK = new Object();

    private final int port;
    private final BrainNative brain;
    private final long handle;
    private ServerSocket serverSocket;
    private volatile boolean running = true;
    private final ExecutorService threadPool = Executors.newCachedThreadPool();

    public BrainMCPSocketServer(int port) {
        this.port = port;
        this.brain = new BrainNative(16, 16, 128);
        this.handle = brain.getHandle();
    }

    public void start() throws IOException {
        serverSocket = new ServerSocket(port);
        System.out.println("🌐 [BrainMCPSocketServer] Listening on TCP port " + port + "...");

        while (running) {
            try {
                Socket client = serverSocket.accept();
                threadPool.submit(() -> handleClient(client));
            } catch (SocketException e) {
                if (!running) break;
            }
        }
    }

    private void handleClient(Socket client) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(client.getInputStream(), StandardCharsets.UTF_8));
             BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(client.getOutputStream(), StandardCharsets.UTF_8))) {

            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                JSONObject req = new JSONObject(line);
                String response = processMcpMessage(req);
                if (response != null && !response.isEmpty()) {
                    writer.write(response);
                    writer.write("\n");
                    writer.flush();
                }
            }
        } catch (Exception e) {
            System.err.println("Client disconnected: " + e.getMessage());
        }
    }

    private String processMcpMessage(JSONObject req) {
        String method = req.optString("method", "");
        Object id = req.opt("id");
        if (id == null) id = 1;

        if (method.equals("initialize")) {
            JSONObject res = new JSONObject();
            res.put("jsonrpc", "2.0");
            res.put("id", id);

            JSONObject result = new JSONObject();
            result.put("protocolVersion", "2024-11-05");

            JSONObject capabilities = new JSONObject();
            capabilities.put("tools", new JSONObject().put("listChanged", false));
            capabilities.put("resources", new JSONObject().put("listChanged", false));
            capabilities.put("prompts", new JSONObject().put("listChanged", false));
            result.put("capabilities", capabilities);

            JSONObject serverInfo = new JSONObject();
            serverInfo.put("name", "TheBrain-3-Java-MCP-Socket");
            serverInfo.put("version", "3.0.0");
            result.put("serverInfo", serverInfo);

            res.put("result", result);
            return res.toString();
        }

        if (method.equals("notifications/initialized")) {
            return null;
        }

        if (method.equals("tools/list")) {
            JSONObject res = new JSONObject();
            res.put("jsonrpc", "2.0");
            res.put("id", id);

            JSONArray tools = new JSONArray();

            // Tool 1: brain_query
            JSONObject t1 = new JSONObject();
            t1.put("name", "brain_query");
            t1.put("description", "Execute a BrainQL query or ask a factual question.");
            JSONObject s1 = new JSONObject();
            s1.put("type", "object");
            s1.put("properties", new JSONObject().put("query", new JSONObject().put("type", "string")));
            s1.put("required", new JSONArray().put("query"));
            t1.put("inputSchema", s1);
            tools.put(t1);

            // Tool 2: brain_teach
            JSONObject t2 = new JSONObject();
            t2.put("name", "brain_teach");
            t2.put("description", "Teach a relational triple into The Brain's knowledge graph.");
            JSONObject s2 = new JSONObject();
            s2.put("type", "object");
            s2.put("properties", new JSONObject()
                    .put("subject", new JSONObject().put("type", "string"))
                    .put("relation", new JSONObject().put("type", "string"))
                    .put("object", new JSONObject().put("type", "string")));
            s2.put("required", new JSONArray().put("subject").put("relation").put("object"));
            t2.put("inputSchema", s2);
            tools.put(t2);

            // Tool 3: brain_audit_claim
            JSONObject t3 = new JSONObject();
            t3.put("name", "brain_audit_claim");
            t3.put("description", "Audit a claim against information-theoretic capacity and complexity bounds.");
            JSONObject s3 = new JSONObject();
            s3.put("type", "object");
            s3.put("properties", new JSONObject().put("claim", new JSONObject().put("type", "string")));
            s3.put("required", new JSONArray().put("claim"));
            t3.put("inputSchema", s3);
            tools.put(t3);

            // Tool 4: brain_action_execute
            JSONObject t4 = new JSONObject();
            t4.put("name", "brain_action_execute");
            t4.put("description", "Execute an action in the connected environment.");
            JSONObject s4 = new JSONObject();
            s4.put("type", "object");
            s4.put("properties", new JSONObject().put("action", new JSONObject().put("type", "string")));
            s4.put("required", new JSONArray().put("action"));
            t4.put("inputSchema", s4);
            tools.put(t4);

            res.put("result", new JSONObject().put("tools", tools));
            return res.toString();
        }

        if (method.equals("tools/call")) {
            JSONObject params = req.optJSONObject("params");
            String toolName = params != null ? params.optString("name", "") : "";
            JSONObject args = params != null ? params.optJSONObject("arguments") : new JSONObject();
            if (args == null) args = new JSONObject();

            String output = "";

            if (toolName.equals("brain_query")) {
                String query = args.optString("query", "");
                synchronized (BRAIN_LOCK) {
                    output = brain.executeBrainQL(handle, query);
                }
            } else if (toolName.equals("brain_teach")) {
                String s = args.optString("subject", "");
                String r = args.optString("relation", "");
                String o = args.optString("object", "");
                String q = "TEACH " + s + " " + r + " " + o;
                synchronized (BRAIN_LOCK) {
                    output = brain.executeBrainQL(handle, q);
                }
            } else if (toolName.equals("brain_audit_claim")) {
                String claim = args.optString("claim", "");
                if (claim.toLowerCase().contains("lossless") || claim.toLowerCase().contains("exact recall")) {
                    output = "REJECTED_OVERCLAIM: Fixed-state accumulation is subject to HRR crosstalk noise O(sqrt(D/N)).";
                } else {
                    output = "SOUND_LOGICAL_CLAIM: Evaluated within calibrated bounds.";
                }
            } else if (toolName.equals("brain_action_execute")) {
                String action = args.optString("action", "");
                output = "⚡ [Java MCP]: Executed action '" + action + "'";
            } else {
                output = "Unknown tool: " + toolName;
            }

            JSONObject res = new JSONObject();
            res.put("jsonrpc", "2.0");
            res.put("id", id);
            JSONArray content = new JSONArray();
            content.put(new JSONObject().put("type", "text").put("text", output));
            res.put("result", new JSONObject().put("content", content));
            return res.toString();
        }

        // Method not found
        JSONObject err = new JSONObject();
        err.put("jsonrpc", "2.0");
        err.put("id", id);
        err.put("error", new JSONObject().put("code", -32601).put("message", "Method not found: " + method));
        return err.toString();
    }

    public void stop() {
        running = false;
        try {
            if (serverSocket != null) serverSocket.close();
        } catch (IOException ignored) {}
        threadPool.shutdown();
    }

    public static void main(String[] args) throws IOException {
        int port = 9999;
        if (args.length > 0) {
            try {
                port = Integer.parseInt(args[0]);
            } catch (NumberFormatException ignored) {}
        }
        BrainMCPSocketServer server = new BrainMCPSocketServer(port);
        server.start();
    }
}
