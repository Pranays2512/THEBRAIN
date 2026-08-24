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
            Process proc = pb.start();

            // Drain output streams and reap the process on a daemon thread so
            // rapid speech cannot leak processes or block on full pipe buffers.
            Thread reaper = new Thread(() -> {
                try {
                    byte[] buf = new byte[256];
                    try (java.io.InputStream out = proc.getInputStream()) {
                        while (out.read(buf) != -1) { /* drain stdout */ }
                    }
                    try (java.io.InputStream err = proc.getErrorStream()) {
                        while (err.read(buf) != -1) { /* drain stderr */ }
                    }
                    proc.waitFor();
                } catch (java.io.IOException e) {
                    proc.destroyForcibly();
                } catch (InterruptedException e) {
                    proc.destroyForcibly();
                    Thread.currentThread().interrupt();
                }
            }, "Mouth-TTS-Reaper");
            reaper.setDaemon(true);
            reaper.start();
        } catch (IOException e) {
            System.err.println("[Mouth] Failed to execute TTS: " + e.getMessage());
        }
    }
}
