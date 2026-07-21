#!/usr/bin/env python3
"""
test_math.py — Testing the Calculus Engine on math problems.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from engines.math.calculus_engine import CalculusEngine, render

def main():
    print("==========================================================")
    print("  Testing the Calculus Engine (Symbolic Differentiation)")
    print("==========================================================")
    
    ce = CalculusEngine()
    
    # Problem 1: Core of Gaussian PDF: d/dx ( exp(-0.5 * x^2) )
    p1 = ("exp", ("*", -0.5, ("^", "x", 2)))
    r1 = ce.diff(p1)
    print(f"\n[Undergrad Test 1: Core Gaussian Probability Density]")
    print(f"Function : {render(p1)}")
    print(f"Derivative: {r1.text}")
    print(f"Rules Fired: {r1.rules}")
    
    # Problem 2: Deeply Nested Composition: d/dx ( sin( cos( exp( ln(x^2 + 1) ) ) ) )
    p2 = ("sin", ("cos", ("exp", ("ln", ("+", ("^", "x", 2), 1)))))
    r2 = ce.diff(p2)
    print(f"\n[Undergrad Test 2: Inception-Level Nested Composition]")
    print(f"Function : {render(p2)}")
    print(f"Derivative: {r2.text}")
    print(f"Rules Fired: {r2.rules}")
    
    # Problem 3: Monster Fraction in Exponent: d/dx ( exp( (x * sin(x)) / (x^2 + 1) ) )
    p3 = ("exp", ("/", ("*", "x", ("sin", "x")), ("+", ("^", "x", 2), 1)))
    r3 = ce.diff(p3)
    print(f"\n[Undergrad Test 3: Complex Rational Function inside Exponential]")
    print(f"Function : {render(p3)}")
    print(f"Derivative: {r3.text}")
    print(f"Rules Fired: {r3.rules}")

if __name__ == "__main__":
    main()
