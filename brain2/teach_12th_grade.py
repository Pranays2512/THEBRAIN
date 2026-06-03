import brain2
import os

print("Initializing Brain v3...")
b = brain2.Brain(som_rows=8, som_cols=8, n_dims=16)

ckpt_dir = "checkpoints/stage4_parsing"
print(f"Loading full Brain architecture from {ckpt_dir}...")
b.load_components(
    predictor_path=f"{ckpt_dir}/predictor.bin",
    language_path=f"{ckpt_dir}/language.bin",
    som_path=f"{ckpt_dir}/som.bin",
    episodic_path=f"{ckpt_dir}/episodic.bin",
    emotion_path=f"{ckpt_dir}/emotion.bin",
    self_path=f"{ckpt_dir}/self.bin",
    symbolic_path=f"{ckpt_dir}/symbolic.bin",
    binding_path=f"{ckpt_dir}/binding.bin",
    bg_path=f"{ckpt_dir}/bg.bin",
    procedures_path=f"{ckpt_dir}/procedures.bin",
    hpred_path=f"{ckpt_dir}/hpred.bin"
)

# 12th Grade Non-Medical Knowledge Graph
high_school = [
    # Physics (Mechanics & Kinematics)
    ("force", "isa", "mass_times_acceleration"),
    ("velocity", "isa", "rate_of_change_of_position"),
    ("acceleration", "isa", "rate_of_change_of_velocity"),
    ("momentum", "isa", "mass_times_velocity"),
    ("gravity", "isa", "force"),
    ("friction", "isa", "force"),
    ("tension", "isa", "force"),
    ("newton", "isa", "unit_of_force"),
    ("joule", "isa", "unit_of_energy"),
    ("watt", "isa", "unit_of_power"),
    ("energy", "cannot_be", "destroyed"),
    
    # Physics (Electromagnetism & Modern)
    ("electron", "has", "negative_charge"),
    ("proton", "has", "positive_charge"),
    ("neutron", "has", "no_charge"),
    ("photon", "isa", "particle_of_light"),
    ("current", "isa", "flow_of_electrons"),
    ("voltage", "isa", "potential_difference"),
    ("resistance", "opposes", "current"),
    ("ohm", "isa", "unit_of_resistance"),
    ("magnetic_field", "deflects", "moving_charge"),

    # Mathematics (Calculus & Algebra)
    ("derivative", "finds", "slope"),
    ("integral", "finds", "area"),
    ("limit", "isa", "boundary_value"),
    ("function", "has", "domain_and_range"),
    ("matrix", "isa", "array"),
    ("determinant", "isa", "scalar_value"),
    ("vector", "has", "magnitude_and_direction"),
    ("scalar", "has", "only_magnitude"),
    ("dot_product", "yields", "scalar"),
    ("cross_product", "yields", "vector"),
    ("probability", "measures", "chance"),

    # Chemistry (Physical & Inorganic)
    ("atom", "has", "nucleus"),
    ("nucleus", "contains", "protons_and_neutrons"),
    ("isotope", "has_different", "neutrons"),
    ("ph_less_than_7", "isa", "acid"),
    ("ph_greater_than_7", "isa", "base"),
    ("ph_7", "isa", "neutral"),
    ("water", "isa", "h2o"),
    ("carbon", "isa", "basis_of_organic_chemistry"),
    ("exothermic", "releases", "heat"),
    ("endothermic", "absorbs", "heat"),
    ("catalyst", "speeds_up", "reaction"),
    ("noble_gas", "isa", "unreactive"),

    # Computer Science
    ("algorithm", "isa", "process"),
    ("cpu", "isa", "processor"),
    ("ram", "isa", "volatile_memory"),
    ("rom", "isa", "non_volatile_memory"),
    ("binary", "uses", "zeros_and_ones"),
    ("compiler", "translates", "code"),
    ("variable", "stores", "data"),
    ("loop", "repeats", "code")
]

print(f"Teaching {len(high_school)} scientific facts (12th Grade level)...")

for subj_w, rel_w, obj_w in high_school:
    if not b.symbolic_table.knows(subj_w): b.learn_word(subj_w)
    if not b.symbolic_table.knows(rel_w): b.learn_word(rel_w)
    if not b.symbolic_table.knows(obj_w): b.learn_word(obj_w)
    
    subj_vec = b.language.encode(subj_w)
    rel_vec = b.language.encode(rel_w)
    obj_vec = b.language.encode(obj_w)
    
    # Store the factual relationship
    b.bind_triple(subj_vec, rel_vec, obj_vec)
    
    # Expose the new vectors to the Brain's predictive Daydreaming system
    b.perceive(subj_vec)
    b.perceive(rel_vec)
    b.perceive(obj_vec)

print("Saving Brain state...")
b.save_components(ckpt_dir)

print("Done! The Brain is now studying for the JEE/Board Exams.")
