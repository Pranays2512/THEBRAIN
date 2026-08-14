import brain3.BrainNative;
import brain3.BrainInterface;

public class TestNativeBrainQL {
    public static void main(String[] args) {
        System.out.println("Starting Native BrainQL Test...");
        System.loadLibrary("brainjni");
        BrainNative nativeBrain = new BrainNative(10, 10, 300);
        BrainInterface iface = new BrainInterface(null, nativeBrain, null, null);
        
        System.out.println("\n[Test 1] Teaching fact: cat isa animal");
        String reply1 = iface.respondStr("TEACH cat isa animal");
        System.out.println("Brain: " + reply1);
        
        System.out.println("\n[Test 2] Teaching fact: animal is alive");
        String reply2 = iface.respondStr("TEACH animal is alive");
        System.out.println("Brain: " + reply2);
        
        System.out.println("\n[Test 3] Logical Deduction: cat is");
        String reply3 = iface.respondStr("INHERIT cat is");
        System.out.println("Brain: " + reply3);

        System.out.println("\n[Test 4] Casual Chat fallback");
        String reply4 = iface.respondStr("CHAT Hi there! I am having a great day.");
        System.out.println("Brain: " + reply4);
        
        System.out.println("\nTest Complete.");
    }
}
