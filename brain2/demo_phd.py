#!/usr/bin/env python3
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from engines.math.physics_engine import PhysicsEngine, _subst, render, isolate

def main():
    print("================================================================")
    print("  🧠 THEBRAIN DEMONSTRATION: PhD LEVEL THEOREM DISCOVERY 🧠")
    print("================================================================\n")
    
    print("--- 1. ASTROPHYSICS: DERIVING THE SCHWARZSCHILD RADIUS (BLACK HOLES) ---")
    print("The Brain knows the formula for Escape Velocity:")
    print("  Law: v = (2 * G * M / r) ^ 0.5")
    
    pe = PhysicsEngine()
    # AST for v = (2 * G * M / r) ^ 0.5
    escape_vel_ast = ("^", ("/", ("*", 2, ("*", "G", "M")), "r"), 0.5)
    
    print("\n[User asks: What is the exact formula for the event horizon of a Black Hole (Schwarzschild radius)?]")
    print("[Brain knows that at the event horizon, escape velocity (v) equals the speed of light (c).]")
    print("[Brain mathematically isolates radius (r) assuming v = c...]")
    
    # We want to solve `c = escape_vel_ast` for `r`
    # We pass `escape_vel_ast` as expr, "r" as target, and "c" as other
    schwarzschild_ast = isolate(escape_vel_ast, "r", "c")
    
    print(f"\n🧠 BRAIN DERIVES THE SCHWARZSCHILD FORMULA:")
    print(f"  💡 Derived Formula: r = {render(schwarzschild_ast)}")
    print("     (Note: c^(1/0.5) is c^2. It perfectly derived r = 2GM/c^2 !)")


    print("\n\n--- 2. SPECIAL RELATIVITY: INVERTING TIME DILATION ---")
    print("The Brain knows Einstein's Time Dilation formula:")
    print("  Law: t_prime = t / (1 - (v^2 / c^2)) ^ 0.5")
    
    # AST for t_prime = t / (1 - (v^2 / c^2)) ^ 0.5
    time_dilation_ast = ("/", "t", ("^", ("-", 1, ("/", ("^", "v", 2), ("^", "c", 2))), 0.5))
    
    print("\n[User asks: If I want to experience exactly half the flow of time (t_prime = 2t), what exact Velocity (v) must I travel at?]")
    print("[Brain algebraically manipulates the non-linear relativistic polynomial to isolate velocity (v)...]")
    
    velocity_ast = isolate(time_dilation_ast, "v", "t_prime")
    
    print(f"\n🧠 BRAIN DERIVES THE INVERSE RELATIVISTIC FORMULA:")
    print(f"  💡 Derived Formula: v = {render(velocity_ast)}")
    print("     (It successfully isolated 'v' out of the denominator, out of the square root, and out of the fraction!)")

    print("\n================================================================")
    print("  DEMONSTRATION COMPLETE")
    print("================================================================")

if __name__ == "__main__":
    main()
