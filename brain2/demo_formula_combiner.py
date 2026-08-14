#!/usr/bin/env python3
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from engines.math.physics_engine import PhysicsEngine, _subst, render, isolate

def main():
    print("================================================================")
    print("  🧠 THEBRAIN DEMONSTRATION: THEOREM RECOMBINATION 🧠")
    print("================================================================\n")
    
    print("--- COMBINING TWO SEPARATE PHYSICS LAWS ---")
    print("The Brain knows two independent laws from different domains/chapters:")
    print("  Law 1 (Newtonian Dynamics): F = m * a")
    print("  Law 2 (Kinematics): a = v / t")
    
    pe = PhysicsEngine()
    pe.add_law("newton2", "F", ("*", "m", "a"))
    pe.add_law("accel", "a", ("/", "v", "t"))
    
    # Extract ASTs
    f_ast = pe.laws["newton2"][1]
    a_ast = pe.laws["accel"][1]
    
    print("\n[User asks: I only know Mass (m), Velocity (v), and Time (t). Can you find Force (F)?]")
    print("[Brain realizes it lacks a direct formula. It begins substituting known AST theorem trees...]")
    
    # Brain substitutes `a` in the `newton2` AST with the AST for `accel`
    combined_ast = _subst(f_ast, {"a": a_ast})
    
    print(f"\n🧠 BRAIN DISCOVERS NEW COMPOUND FORMULA:")
    print(f"  💡 Derived Formula: F = {render(combined_ast)}")
    
    print("\n[User asks: Okay, now what if I want to solve for Velocity (v) using this completely new formula?]")
    print("[Brain algebraically isolates 'v' inside the newly discovered compound formula...]")
    
    isolated_ast = isolate(combined_ast, "v", "F")
    print(f"  💡 Derived Inverse Formula: v = {render(isolated_ast)}")
    
    print("\n================================================================")
    print("  DEMONSTRATION COMPLETE")
    print("================================================================")

if __name__ == "__main__":
    main()
