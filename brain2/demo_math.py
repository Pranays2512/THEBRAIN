#!/usr/bin/env python3
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from engines.math.physics_engine import PhysicsEngine
from engines.math.algebra_engine import AlgebraEngine

def main():
    print("================================================================")
    print("  🧠 THEBRAIN DEMONSTRATION: MATHEMATICAL FORMULA REDISCOVERY 🧠")
    print("================================================================\n")
    
    pe = PhysicsEngine()
    
    print("--- 1. AST ALGEBRAIC ISOLATION (Discovering Inverse Laws) ---")
    print("Teaching the Brain a single raw formula: Newton's Second Law")
    print("  Law: F = m * a")
    pe.add_law("newton2", "F", ("*", "m", "a"))
    
    print("\n[User asks: What is the formula for Acceleration?]")
    print("[Brain algebraically manipulates the AST tree to discover the inverse...]")
    ans, steps = pe.solve("newton2", "a", F=100, m=20)
    for step in steps:
        print(f"  🧠 Derived Step: {step}")
        
    print("\n[User asks: What is the formula for Mass?]")
    ans, steps = pe.solve("newton2", "m", F=100, a=5)
    for step in steps:
        print(f"  🧠 Derived Step: {step}")

    print("\n\n--- 2. COMPLEX NON-LINEAR REDISCOVERY (Kinetic Energy) ---")
    print("Teaching the Brain: Kinetic Energy")
    print("  Law: KE = 0.5 * m * (v ^ 2)")
    pe.add_law("kinetic", "KE", ("*", 0.5, ("*", "m", ("^", "v", 2))))
    
    print("\n[User asks: I know KE and Mass. What is the formula for Velocity (v)?]")
    print("[Brain isolating variable inside a polynomial exponent...]")
    ans, steps = pe.solve("kinetic", "v", KE=1000, m=20)
    for step in steps:
        print(f"  🧠 Derived Step: {step}")
    print(f"  Calculation Result: v = {ans}")


    print("\n================================================================")
    print("  DEMONSTRATION COMPLETE")
    print("================================================================")

if __name__ == "__main__":
    main()
