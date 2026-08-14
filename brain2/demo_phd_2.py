#!/usr/bin/env python3
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from engines.math.physics_engine import PhysicsEngine, render, isolate

def main():
    print("=======================================================================")
    print("  🧠 THEBRAIN DEMONSTRATION: EXTREME COMPLEXITY THEOREM ISOLATION 🧠")
    print("=======================================================================\n")
    
    print("--- GENERAL RELATIVITY: GRAVITATIONAL TIME DILATION ---")
    print("The Brain is given the exact formula for time dilation near a massive body (like a Black Hole).")
    print("  Law: t_prime = t * sqrt(1 - (2 * G * M) / (r * c^2))")
    
    # AST representation
    # t_prime = t * (1 - (2 * G * M) / (r * c^2)) ^ 0.5
    inner_fraction = ("/", ("*", 2, ("*", "G", "M")), ("*", "r", ("^", "c", 2)))
    time_dilation_ast = ("*", "t", ("^", ("-", 1, inner_fraction), 0.5))
    
    print(f"  AST format: {time_dilation_ast}")
    
    print("\n[User asks: I know how much time dilation I want (t_prime). What exact orbital radius (r) must I park my ship at?]")
    print("[Brain must algebraiclly dig 'r' out of a denominator, inside a fraction, inside a subtraction, inside a square root, inside a multiplication...]")
    
    try:
        radius_ast = isolate(time_dilation_ast, "r", "t_prime")
        print(f"\n🧠 BRAIN FLAWLESSLY DERIVES THE INVERSE ORBITAL RADIUS FORMULA:")
        print(f"  💡 Derived Formula: r = {render(radius_ast)}")
    except Exception as e:
        print(f"\n❌ Brain failed to isolate the variable: {e}")

    print("\n=======================================================================")
    print("  DEMONSTRATION COMPLETE")
    print("=======================================================================")

if __name__ == "__main__":
    main()
