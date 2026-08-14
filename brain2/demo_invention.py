#!/usr/bin/env python3
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from engines.math.calculus_engine import CalculusEngine, simplify

def main():
    print("=======================================================================")
    print("  🧠 THEBRAIN DEMONSTRATION: AUTONOMOUS FORMULA INVENTION 🧠")
    print("=======================================================================\n")
    
    print("--- PHYSICS INVENTION: DAMPED HARMONIC OSCILLATOR ---")
    print("The Brain is given the basic position formula for a swinging pendulum with air friction.")
    print("  Law (Position): x(t) = A * exp(-b * t) * cos(w * t)")
    
    # We will differentiate with respect to 't' instead of 'x'. The CalculusEngine defaults to 'x'.
    # We can just use 'x' as the time variable for the engine's sake, so:
    # position = A * exp(-b * x) * cos(w * x)
    # Actually, the CalculusEngine might only differentiate with respect to 'x'. Let's check.
    # We will use 'x' as the variable and 'A', 'b', 'w' as constants.
    
    # Let's simplify the AST for the engine, using numeric constants so it can simplify better, 
    # or just use symbolic constants if it supports them.
    # p(x) = exp(-2 * x) * cos(3 * x)
    
    print("  (Using 'x' for time, '2' for friction, '3' for frequency)")
    print("  Law: p(x) = exp(-2 * x) * cos(3 * x)")
    
    position_ast = ("*", ("exp", ("*", -2, "x")), ("cos", ("*", 3, "x")))
    
    ce = CalculusEngine()
    
    print("\n[User asks: I only know the formula for position. Can you INVENT the exact formula for Velocity (v) and Acceleration (a)?]")
    print("[Brain realizes Velocity is the first derivative of Position, and Acceleration is the second derivative.]")
    print("[Brain begins autonomously applying Chain Rule, Product Rule, Trig Rules, and Exponent Rules...]\n")
    
    # 1st Derivative (Velocity)
    v_res = ce.diff(position_ast)
    v_ast_simplified = simplify(v_res.expr)
    
    print(f"🧠 BRAIN INVENTS VELOCITY FORMULA:")
    print(f"  💡 Derived Formula: v(x) = {v_res.text}")
    print(f"  📐 Rules Used: {', '.join(set(v_res.rules))}")
    
    # 2nd Derivative (Acceleration)
    a_res = ce.diff(v_ast_simplified)
    
    print(f"\n🧠 BRAIN INVENTS ACCELERATION FORMULA:")
    print(f"  💡 Derived Formula: a(x) = {a_res.text}")
    print(f"  📐 Rules Used: {', '.join(set(a_res.rules))}")

    print("\n=======================================================================")
    print("  DEMONSTRATION COMPLETE")
    print("=======================================================================")

if __name__ == "__main__":
    main()
