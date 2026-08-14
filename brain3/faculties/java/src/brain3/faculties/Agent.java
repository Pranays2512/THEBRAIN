package brain3.faculties;
import java.util.UUID;
public class Agent {
    private String id;
    private WholeBrain brain;
    public Agent(WholeBrain brain) {
        this.brain = brain;
        this.id = UUID.randomUUID().toString();
    }
    public String getId() { return id; }
}
