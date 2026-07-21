class EventBus:
    """
    A lightweight Pub/Sub bus decoupling the C++ fuzzy state from Python execution.
    The Three Bridges (Topology, Teaching, Curiosity) all pass messages here rather
    than holding hard references to each other.
    """
    def __init__(self):
        self._subscribers = {}

    def subscribe(self, topic, callback):
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def publish(self, topic, data=None):
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"[EventBus] Error in subscriber for {topic}: {e}")

# Global instance for the whole brain
bus = EventBus()
