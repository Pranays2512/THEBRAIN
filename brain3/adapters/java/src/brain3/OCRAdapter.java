package brain3;

import java.io.IOException;

public class OCRAdapter {

    public String extractTextFromPdf(String filePath) {
        // In a real port, we might use Apache PDFBox or a CLI command like pdftotext.
        // For keeping dependencies minimal per user constraints, we rely on a system call.
        try {
            ProcessBuilder pb = new ProcessBuilder("pdftotext", filePath, "-");
            Process p = pb.start();
            byte[] output = p.getInputStream().readAllBytes();
            p.waitFor();
            return new String(output);
        } catch (IOException | InterruptedException e) {
            System.err.println("[OCRAdapter] Error reading PDF: " + e.getMessage());
            return "";
        }
    }
}
