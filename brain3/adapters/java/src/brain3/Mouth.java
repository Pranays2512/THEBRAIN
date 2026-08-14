package brain3;

import java.io.IOException;

public class Mouth {

    private final boolean enabled;

    public Mouth(boolean enabled) {
        this.enabled = enabled;
    }

    public void speak(String text) {
        if (!enabled || text == null || text.trim().isEmpty()) {
            return;
        }

        // Strip some markdown or unwanted characters that TTS might struggle with
        String cleanText = text.replaceAll("[*#_`]", "");

        try {
            // Using macOS 'say' command
            ProcessBuilder pb = new ProcessBuilder("say", cleanText);
            pb.start(); // Async execution, does not wait for it to finish
        } catch (IOException e) {
            System.err.println("[Mouth] Failed to execute TTS: " + e.getMessage());
        }
    }
}
