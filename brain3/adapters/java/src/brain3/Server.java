package brain3;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.Executors;

public class Server {

    // The brain core is single-thread-accessed by contract; concurrent HTTP
    // clients must be serialized around the shared-brain invocation.
    private static final Object BRAIN_LOCK = new Object();

    private final BrainInterface brainInterface;
    private final int port;
    private HttpServer server;

    public Server(BrainInterface brainInterface, int port) {
        this.brainInterface = brainInterface;
        this.port = port;
    }

    public void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/api/chat", new ChatHandler(brainInterface));
        server.setExecutor(Executors.newFixedThreadPool(4)); 
        server.start();
        System.out.println("Brain3 Java Server listening on port " + port);
    }

    public void stop() {
        if (server != null) {
            server.stop(0);
        }
    }

    static class ChatHandler implements HttpHandler {
        private final BrainInterface bi;

        public ChatHandler(BrainInterface bi) {
            this.bi = bi;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("POST".equals(exchange.getRequestMethod())) {
                // In a real app, read JSON body. For now, read raw input for port mock
                byte[] requestBody = exchange.getRequestBody().readAllBytes();
                String query = new String(requestBody);
                
                // Route through BrainInterface (serialized: brain core is
                // single-thread-accessed by contract)
                Map<String, Object> result;
                synchronized (BRAIN_LOCK) {
                    result = bi.respond(query);
                }
                
                // Format JSON response manually with full escaping
                String reply = (String) result.get("reply");
                String kind = (String) result.get("kind");
                boolean verified = (Boolean) result.get("verified");
                
                String jsonResponse = String.format("{\"reply\": \"%s\", \"kind\": \"%s\", \"verified\": %b}",
                        jsonEscape(reply), jsonEscape(kind), verified);

                exchange.getResponseHeaders().set("Content-Type", "application/json");
                byte[] body = jsonResponse.getBytes(StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(200, body.length);
                OutputStream os = exchange.getResponseBody();
                os.write(body);
                os.close();
            } else {
                exchange.sendResponseHeaders(405, -1); // Method Not Allowed
            }
        }

        // Escape backslash first, then quote, then control characters, so the
        // response stays valid JSON even when replies contain newlines.
        private static String jsonEscape(String s) {
            if (s == null) return "";
            StringBuilder sb = new StringBuilder(s.length() + 16);
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '\\': sb.append("\\\\"); break;
                    case '"':  sb.append("\\\""); break;
                    case '\n': sb.append("\\n"); break;
                    case '\r': sb.append("\\r"); break;
                    case '\t': sb.append("\\t"); break;
                    default:   sb.append(c); break;
                }
            }
            return sb.toString();
        }
    }
}
