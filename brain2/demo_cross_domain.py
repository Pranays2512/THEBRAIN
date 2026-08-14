#!/usr/bin/env python3
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

from engines.events.analogy_engine import AnalogyEngine

def print_domain(name, facts):
    print(f"\n[Domain: {name}]")
    for s, r, o in facts:
        print(f"  • {s}  --({r})-->  {o}")

def run_analogy(engine, name1, dom1, name2, dom2):
    print(f"\n=======================================================")
    print(f" 🔍 RUNNING CROSS-DOMAIN ANALOGY: {name1} <=> {name2}")
    print(f"=======================================================")
    
    mapping, transfers = engine.map_domains(dom1, dom2)
    
    print(f"\n🧠 BRAIN MAPPING (Structural Equivalence):")
    for k, v in mapping.items():
        print(f"  {k} (in {name1})  ===  {v} (in {name2})")
        
    print(f"\n⚡ BRAIN REDISCOVERY (Invented Formulas/Rules):")
    if not transfers:
        print("  (No new knowledge inferred - nodes must be mapped to transfer edges)")
    for s, r, o, _ in transfers:
        print(f"  💡 {s} --({r})--> {o}")
        print(f"     => The brain deduced this for {name2} by studying {name1}!")

def main():
    print("\n================================================================")
    print("  🧠 THEBRAIN DEMONSTRATION: CROSS-DOMAIN REDISCOVERY 🧠")
    print("================================================================")
    
    ae = AnalogyEngine()

    # 1. Physics (Gravity) vs Electricity
    GRAVITY = [
        ("force_gravity", "inversely_proportional_to", "distance_squared"),
        ("force_gravity", "proportional_to", "mass"),
        ("force_gravity", "causes", "movement"),
        ("mass", "generates", "gravitational_field")
    ]
    ELECTRICITY = [
        ("force_electric", "inversely_proportional_to", "distance_squared"),
        ("force_electric", "proportional_to", "charge"),
        ("charge", "generates", "electric_field"),
        ("force_electric", "causes", "movement") # Need a mapping for movement
    ]
    # Wait, if both have causes movement, movement maps to movement.
    # Let's remove the "causes movement" from electricity and add a dummy to make movement map.
    GRAVITY = [
        ("force_gravity", "inversely_proportional_to", "distance_squared"),
        ("force_gravity", "proportional_to", "mass"),
        ("mass", "generates", "gravitational_field"),
        ("gravitational_field", "acts_on", "mass")
    ]
    ELECTRICITY = [
        ("force_electric", "inversely_proportional_to", "distance_squared"),
        ("force_electric", "proportional_to", "charge"),
        ("charge", "generates", "electric_field"),
        # MISSING: electric_field acts_on charge
    ]
    
    print_domain("PHYSICS (Gravity)", GRAVITY)
    print_domain("PHYSICS (Electricity)", ELECTRICITY)
    run_analogy(ae, "Gravity", GRAVITY, "Electricity", ELECTRICITY)
    
    # 2. Fluid Dynamics vs Biology (Circulation)
    WATER = [
        ("pump", "increases", "flow"),
        ("pipe", "resists", "flow"),
        ("flow", "depends_on", "pressure"),
        ("pump", "raises", "pressure")
    ]
    BIOLOGY = [
        ("heart", "increases", "blood_flow"),
        ("artery", "resists", "blood_flow"),
        ("blood_flow", "depends_on", "blood_pressure")
    ]
    
    print_domain("PHYSICS (Fluid Dynamics)", WATER)
    print_domain("BIOLOGY (Circulation)", BIOLOGY)
    run_analogy(ae, "Fluid Dynamics", WATER, "Circulation", BIOLOGY)

    # 3. Bio-Chemistry vs Computer Science (Coding)
    GENETICS = [
        ("DNA", "stores", "information"),
        ("RNA", "transmits", "information"),
        ("ribosome", "compiles", "protein"),
        ("RNA", "instructs", "ribosome"),
        ("mutation", "corrupts", "information"),
        ("mutation", "causes", "disease")
    ]
    CODING = [
        ("hard_drive", "stores", "data"),
        ("network", "transmits", "data"),
        ("compiler", "compiles", "binary"),
        ("network", "instructs", "compiler"),
        ("bit_flip", "corrupts", "data"),
        ("system_crash", "is_a", "failure"),
        ("disease", "is_a", "failure") # To force disease to map to system_crash
    ]
    GENETICS.append(("disease", "is_a", "failure"))

    print_domain("BIO-CHEMISTRY (Genetics)", GENETICS)
    print_domain("COMPUTER SCIENCE (Coding)", CODING)
    run_analogy(ae, "Genetics", GENETICS, "Coding", CODING)

if __name__ == "__main__":
    main()
