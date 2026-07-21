from faculties.event_bus import bus

class CuriosityBridge:
    """
    The Curiosity Bridge: A bidirectional channel where both halves of the brain escalate together.
    Condition: Novelty is high (fuzzy side) AND no crisp answer exists (crisp side).
    Result: Escalate to cross-domain reasoning, induction, or daydream replay.
    """
    def __init__(self, brain=None, synth_proposer=None, knowledge_engine=None):
        self.brain = brain
        self.synth_proposer = synth_proposer
        self.knowledge_engine = knowledge_engine
        
        self.consecutive_failures = 0
        self.recent_novelty = 0.0
        
        # Subscribe to fuzzy side events
        bus.subscribe("novelty_spike", self._on_novelty_spike)
        # Subscribe to crisp side events
        bus.subscribe("solved_problem", self._on_solved_problem)
        bus.subscribe("unsolved_problem", self._on_unsolved_problem)
        
    def _on_novelty_spike(self, novelty_val):
        self.recent_novelty = novelty_val
        
    def _on_solved_problem(self, data):
        self.consecutive_failures = 0
        
    def _on_unsolved_problem(self, data):
        self.consecutive_failures += 1

    def tick(self):
        """
        Called when the brain is idle to consolidate or escalate based on recent events.
        """
        # Curiosity Condition: High novelty, but crisp logic is failing.
        if self.recent_novelty > 0.3 and self.consecutive_failures > 2:
            self._escalate()
        # Rest Condition: Low novelty, logic succeeding -> consolidate knowledge
        elif self.recent_novelty < 0.1 and self.consecutive_failures == 0:
            self._consolidate()
            
        # Decay novelty slightly over time if nothing happens
        self.recent_novelty *= 0.9
        
    def _escalate(self):
        print("[CuriosityBridge] Escalating! High novelty & high logic failures.")
        bus.publish("creativity_triggered")
        # In a full implementation, this triggers cross-domain analogy or active probing
        if self.brain:
            print("[CuriosityBridge] Asking fuzzy brain to hallucinate potential analogies...")
            
    def _consolidate(self):
        # Daydreaming and replay
        if self.brain:
            print("[CuriosityBridge] Consolidating recent verified facts into memory...")
            try:
                self.brain.dream_replay_faithful(n_samples=4)
            except Exception:
                pass
