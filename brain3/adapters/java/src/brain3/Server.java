package brain3;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.Map;
import java.util.concurrent.Executors;

public class Server {

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
                
                // Route through BrainInterface
                Map<String, Object> result = bi.respond(query);
                
                // Format JSON response manually
                String reply = (String) result.get("reply");
                String kind = (String) result.get("kind");
                boolean verified = (Boolean) result.get("verified");
                
                String jsonResponse = String.format("{\"reply\": \"%s\", \"kind\": \"%s\", \"verified\": %b}",
                        reply.replace("\"", "\\\""), kind, verified);

                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, jsonResponse.length());
                OutputStream os = exchange.getResponseBody();
                os.write(jsonResponse.getBytes());
                os.close();
            } else {
                exchange.sendResponseHeaders(405, -1); // Method Not Allowed
            }
        }
    }
}
