package brain3.middleware;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * A lightweight Pub/Sub bus decoupling the native C++ components from the Java execution.
 * Bridges pass messages here rather than holding hard references to each other.
 */
public class EventBus {
    
    private final Map<String, List<Consumer<Object>>> subscribers;
    
    // Singleton pattern as in brain2 (global instance)
    private static final EventBus INSTANCE = new EventBus();
    
    private EventBus() {
        this.subscribers = new HashMap<>();
    }
    
    public static EventBus getInstance() {
        return INSTANCE;
    }
    
    public void subscribe(String topic, Consumer<Object> callback) {
        subscribers.computeIfAbsent(topic, k -> new ArrayList<>()).add(callback);
    }
    
    public void publish(String topic, Object data) {
        List<Consumer<Object>> topicSubscribers = subscribers.get(topic);
        if (topicSubscribers != null) {
            for (Consumer<Object> callback : topicSubscribers) {
                try {
                    callback.accept(data);
                } catch (Exception e) {
                    System.err.println("[EventBus] Error in subscriber for " + topic + ": " + e.getMessage());
                }
            }
        }
    }
}
