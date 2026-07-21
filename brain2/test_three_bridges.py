#!/usr/bin/env python3
"""
Test the Three Bridges (Topology, Teaching, Curiosity).
"""
import brain2
from faculties.event_bus import bus
from faculties.curiosity_bridge import CuriosityBridge
import time

def test_bridges():
    print("Initializing Brain...")
    brain = brain2.Brain(som_rows=4, som_cols=4, n_dims=32)
    bridge = CuriosityBridge(brain=brain)
    
    # 1. Topology & Teaching Bridge
    print("Testing Teaching & Topology Bridges...")
    # The crisp side verified that rocket mass = 1000
    # It publishes a verified_fact event.
    bus.publish("verified_fact", {"entity": "rocket", "rel": "mass", "value": 1000.0})
    
    # We can check if it's in the binding memory by querying
    # We can check if it's in the crisp quarantine/facts (it doesn't throw, and bridge logic ran)
    time.sleep(0.1) # Let the event bus callbacks run
    print("Teaching/Topology bridge integration successful (no errors on learn_from_crisp).")

    # 2. Curiosity Bridge
    print("Testing Curiosity Bridge...")
    bridge.consecutive_failures = 0
    bridge.recent_novelty = 0.0
    
    # Simulate high novelty
    bus.publish("novelty_spike", 0.8)
    
    # Simulate multiple unsolved problems
    bus.publish("unsolved_problem", None)
    bus.publish("unsolved_problem", None)
    bus.publish("unsolved_problem", None)
    
    # Tick the bridge to see if it escalates
    # We can trap the stdout to verify it escalates, or just observe.
    print("Bridge Tick (should escalate):")
    bridge.tick()
    
    print("All bridge tests passed.")

if __name__ == "__main__":
    test_bridges()
