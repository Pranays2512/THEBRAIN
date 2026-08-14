package brain3.faculties;
public class VerifierMonitor {
    private WholeBrain brain;
    public VerifierMonitor(WholeBrain brain) { this.brain = brain; }
    public void logStatus() {
        System.out.println("Monitor checking brain status.");
    }
}
