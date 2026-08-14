package brain3.faculties;

public class AutonomousLoop implements Runnable {
    private WholeBrain brain;
    private volatile boolean running;
    private Thread workerThread;
    private int tickIntervalMs;

    public AutonomousLoop(WholeBrain brain) {
        this(brain, 500);
    }

    public AutonomousLoop(WholeBrain brain, int tickIntervalMs) {
        this.brain = brain;
        this.running = false;
        this.tickIntervalMs = tickIntervalMs;
    }

    public synchronized void start() {
        if (!running) {
            running = true;
            workerThread = new Thread(this, "AutonomousCuriosityLoop");
            workerThread.setDaemon(true);
            workerThread.start();
        }
    }

    public synchronized void stop() {
        running = false;
        if (workerThread != null) {
            workerThread.interrupt();
            workerThread = null;
        }
    }

    public String runTicks(int ticks) {
        if (brain != null) {
            return brain.respond("AUTONOMOUS_CYCLE " + ticks);
        }
        return "{}";
    }

    @Override
    public void run() {
        while (running) {
            try {
                if (brain != null) {
                    brain.respond("CURIOSITY_TICK");
                }
                Thread.sleep(tickIntervalMs);
            } catch (InterruptedException e) {
                break;
            } catch (Exception e) {
                // Log and continue loop
            }
        }
    }
}
