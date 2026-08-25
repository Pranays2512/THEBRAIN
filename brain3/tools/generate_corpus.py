#!/usr/bin/env python3
"""
generate_corpus.py — massive multi-domain fact generator for brain3.
Outputs FACT: lines that KnowledgeIngestionEngine parses natively.
Every fact is REAL general knowledge (not random strings).
"""
import random, os, sys

random.seed(20260825)
OUT = sys.argv[1] if len(sys.argv) > 1 else "corpus_raw.txt"

# ═══════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASES — curated real facts, not random strings
# ═══════════════════════════════════════════════════════════════════════

facts = []  # list of (subject, relation, object, domain)

def add(s, r, o, domain):
    facts.append((s.lower(), r.lower(), o.lower(), domain))

# ── BIOLOGY ────────────────────────────────────────────────────────────
species = {
    "human": ("mammal", "primate", "omnivore"),
    "dog": ("mammal", "carnivore", "omnivore"),
    "cat": ("mammal", "carnivore", "carnivore"),
    "elephant": ("mammal", "herbivore", "herbivore"),
    "dolphin": ("mammal", "aquatic_mammal", "carnivore"),
    "bat": ("mammal", "flying_mammal", "insectivore"),
    "whale": ("mammal", "marine_mammal", "carnivore"),
    "lion": ("mammal", "big_cat", "carnivore"),
    "tiger": ("mammal", "big_cat", "carnivore"),
    "bear": ("mammal", "omnivore", "omnivore"),
    "wolf": ("mammal", "canine", "carnivore"),
    "horse": ("mammal", "herbivore", "herbivore"),
    "cow": ("mammal", "ruminant", "herbivore"),
    "sheep": ("mammal", "ruminant", "herbivore"),
    "pig": ("mammal", "omnivore", "omnivore"),
    "eagle": ("bird", "raptor", "carnivore"),
    "owl": ("bird", "nocturnal_bird", "carnivore"),
    "penguin": ("bird", "flightless_bird", "carnivore"),
    "sparrow": ("bird", "songbird", "seed_eater"),
    "parrot": ("bird", "tropical_bird", "seed_eater"),
    "salmon": ("fish", "freshwater_fish", "carnivore"),
    "shark": ("fish", "predatory_fish", "carnivore"),
    "goldfish": ("fish", "freshwater_fish", "omnivore"),
    "tuna": ("fish", "ocean_fish", "carnivore"),
    "frog": ("amphibian", "pond_amphibian", "insectivore"),
    "snake": ("reptile", "predator", "carnivore"),
    "turtle": ("reptile", "shelled_reptile", "herbivore"),
    "crocodile": ("reptile", "aquatic_reptile", "carnivore"),
    "bee": ("insect", "pollinator", "nectar_feeder"),
    "ant": ("insect", "social_insect", "omnivore"),
    "butterfly": ("insect", "pollinator", "nectar_feeder"),
    "spider": ("arachnid", "predator", "carnivore"),
}
for name, (cls, subcls, diet) in species.items():
    add(name, "isa", cls, "biology")
    add(name, "isa", subcls, "biology")
    add(name, "has_diet", diet, "biology")
    add(cls, "isa", "animal_group", "biology")
    add(subcls, "isa", "animal_category", "biology")

add("mammal", "is_animal_class", "vertebrate", "biology")
add("bird", "is_animal_class", "vertebrate", "biology")
add("fish", "is_animal_class", "vertebrate", "biology")
add("reptile", "is_animal_class", "vertebrate", "biology")
add("amphibian", "is_animal_class", "vertebrate", "biology")
add("vertebrate", "has", "backbone", "biology")
add("vertebrate", "belongs_to", "animal_kingdom", "biology")
add("mammal", "has", "fur_or_hair", "biology")
add("mammal", "produces", "milk", "biology")
add("bird", "has", "feathers", "biology")
add("bird", "has", "beak", "biology")
add("fish", "has", "gills", "biology")
add("fish", "lives_in", "water", "biology")
add("cell", "is_basic_unit_of", "life", "biology")
add("dna", "carries", "genetic_information", "biology")
add("mitochondria", "produces", "atp_energy", "biology")
add("photosynthesis", "converts", "sunlight_to_energy", "biology")
add("chlorophyll", "enables", "photosynthesis", "biology")
add("heart", "pumps", "blood", "biology")
add("lungs", "process", "oxygen", "biology")
add("brain_organ", "controls", "nervous_system", "biology")
add("neuron", "transmits", "nerve_signals", "biology")
add("protein", "builds", "muscle_tissue", "biology")
add("enzyme", "accelerates", "chemical_reactions", "biology")
add("red_blood_cell", "carries", "oxygen", "biology")
add("white_blood_cell", "fights", "infection", "biology")
add("immune_system", "protects_against", "disease", "biology")
add("vaccine", "trains", "immune_system", "medicine")
add("antibiotic", "kills", "bacteria", "medicine")
add("virus", "requires", "host_cell", "biology")
add("bacteria", "can_be", "beneficial_or_harmful", "biology")

# ── CHEMISTRY ──────────────────────────────────────────────────────────
elements = {
    "hydrogen": ("H", 1), "helium": ("He", 2), "carbon": ("C", 6),
    "nitrogen": ("N", 7), "oxygen": ("O", 8), "sodium": ("Na", 11),
    "magnesium": ("Mg", 12), "aluminum": ("Al", 13), "silicon": ("Si", 14),
    "phosphorus": ("P", 15), "sulfur": ("S", 16), "chlorine": ("Cl", 17),
    "potassium": ("K", 19), "calcium": ("Ca", 20), "iron": ("Fe", 26),
    "copper": ("Cu", 29), "zinc": ("Zn", 30), "silver": ("Ag", 47),
    "gold": ("Au", 79), "mercury": ("Hg", 80), "lead": ("Pb", 82),
    "uranium": ("U", 92),
}
for name, (sym, num) in elements.items():
    add(name, "is_a", "chemical_element", "chemistry")
    add(name, "has_symbol", sym, "chemistry")
    add(name, "has_atomic_number", str(num), "chemistry")
    add(sym, "is_symbol_for", name, "chemistry")

add("water", "consists_of", "hydrogen_and_oxygen", "chemistry")
add("water", "has_formula", "h2o", "chemistry")
add("salt", "consists_of", "sodium_and_chlorine", "chemistry")
add("salt", "has_formula", "nacl", "chemistry")
add("carbon_dioxide", "has_formula", "co2", "chemistry")
add("carbon_dioxide", "is_produced_by", "respiration", "chemistry")
add("methane", "has_formula", "ch4", "chemistry")
add("ammonia", "has_formula", "nh3", "chemistry")
add("glucose", "has_formula", "c6h12o6", "chemistry")
add("acid", "donates", "protons", "chemistry")
add("base", "accepts", "protons", "chemistry")
add("ph_scale", "measures", "acidity", "chemistry")
add("catalyst", "speeds_up", "reactions", "chemistry")
add("oxidation", "involves", "electron_loss", "chemistry")
add("reduction", "involves", "electron_gain", "chemistry")
add("polymer", "consists_of", "monomers", "chemistry")
add("alloy", "mixes", "metals", "chemistry")

# ── PHYSICS ────────────────────────────────────────────────────────────
laws = [
    ("newton_first_law", "states", "objects_at_rest_stay_at_rest"),
    ("newton_second_law", "states", "force_equals_mass_times_acceleration"),
    ("newton_third_law", "states", "every_action_has_equal_opposite_reaction"),
    ("law_of_gravitation", "states", "masses_attract_proportionally"),
    ("conservation_of_energy", "states", "energy_cannot_be_created_or_destroyed"),
    ("conservation_of_momentum", "states", "total_momentum_is_constant"),
    ("ohms_law", "relates", "voltage_current_resistance"),
    ("thermodynamics_second_law", "states", "entropy_always_increases"),
    ("einstein_equation", "relates", "energy_mass_speed_of_light"),
]
for name, rel, obj in laws:
    add(name, rel, obj, "physics")

forces = ["gravity", "electromagnetic_force", "strong_nuclear_force", "weak_nuclear_force"]
for f in forces:
    add(f, "is_a", "fundamental_force", "physics")
add("gravity", "attracts", "masses", "physics")
add("electromagnetic_force", "acts_on", "charged_particles", "physics")
add("light", "travels_at", "299792458_m_per_s", "physics")
add("light", "behaves_as", "wave_and_particle", "physics")
add("energy", "exists_as", "kinetic_or_potential", "physics")
add("atom", "contains", "protons_neutrons_electrons", "physics")
add("proton", "has_charge", "positive", "physics")
add("electron", "has_charge", "negative", "physics")
add("neutron", "has_charge", "neutral", "physics")
add("nucleus", "contains", "protons_and_neutrons", "physics")
add("radioactivity", "involves", "nuclear_decay", "physics")
add("fission", "splits", "atomic_nuclei", "physics")
add("fusion", "combines", "atomic_nuclei", "physics")
add("quantum_mechanics", "describes", "subatomic_behavior", "physics")
add("uncertainty_principle", "limits", "position_and_momentum_precision", "physics")
add("wave_particle_duality", "applies_to", "matter_and_light", "physics")
add("speed_of_sound", "equals", "343_m_per_s_in_air", "physics")
add("absolute_zero", "equals", "-273.15_celsius", "physics")

# ── MATHEMATICS ────────────────────────────────────────────────────────
math_ops = [
    ("addition", "commutative"), ("addition", "associative"),
    ("multiplication", "commutative"), ("multiplication", "associative"),
]
for op, prop in math_ops:
    add(op, "is", prop, "mathematics")
add("subtraction", "inverse_of", "addition", "mathematics")
add("division", "inverse_of", "multiplication", "mathematics")
add("zero", "is_additive_identity", "any_number_plus_zero_is_same", "mathematics")
add("one", "is_multiplicative_identity", "any_number_times_one_is_same", "mathematics")
add("pi", "approximates", "3.14159", "mathematics")
add("pi", "relates", "circle_circumference_to_diameter", "mathematics")
add("e", "approximates", "2.71828", "mathematics")
add("prime_number", "divisible_only_by", "one_and_itself", "mathematics")
add("two", "is_the_only_even_prime", "", "mathematics")
add("pythagorean_theorem", "relates", "right_triangle_sides", "mathematics")
add("quadratic_formula", "solves", "second_degree_equations", "mathematics")
add("derivative", "measures", "rate_of_change", "mathematics")
add("integral", "computes", "area_under_curve", "mathematics")
add("calculus", "studies", "change_and_accumulation", "mathematics")
add("algebra", "uses", "symbols_for_unknown_quantities", "mathematics")
add("geometry", "studies", "shapes_and_spaces", "mathematics")
add("statistics", "analyzes", "data_distributions", "mathematics")
add("probability", "measures", "likelihood_of_events", "mathematics")
add("set_theory", "studies", "collections_of_objects", "mathematics")
add("infinity", "exceeds", "any_finite_value", "mathematics")
add("fibonacci_sequence", "appears_in", "nature_patterns", "mathematics")
add("golden_ratio", "approximates", "1.61803", "mathematics")

# ── COMPUTER SCIENCE ───────────────────────────────────────────────────
languages = {
    "python": ("interpreted", "general_purpose"),
    "java": ("compiled_jvm", "object_oriented"),
    "c": ("compiled_native", "procedural"),
    "cpp": ("compiled_native", "object_oriented"),
    "javascript": ("interpreted_browser", "event_driven"),
    "rust": ("compiled_native", "memory_safe"),
    "go": ("compiled_native", "concurrent"),
    "sql": ("declarative", "database_query"),
}
for lang, (compile_type, paradigm) in languages.items():
    add(lang, "is_a", "programming_language", "cs")
    add(lang, "uses", compile_type, "cs")
    add(lang, "follows", paradigm, "cs")

algorithms = [
    ("binary_search", "search", "olog_n"),
    ("quick_sort", "sorting", "onlogn"),
    ("merge_sort", "sorting", "onlogn"),
    ("bubble_sort", "sorting", "onsquared"),
    ("hash_table_lookup", "lookup", "o1_average"),
    ("breadth_first_search", "graph_traversal", "ov_plus_e"),
    ("depth_first_search", "graph_traversal", "ov_plus_e"),
    ("dynamic_programming", "optimization", "varies"),
]
for name, cat, complexity in algorithms:
    add(name, "is_a", "algorithm", "cs")
    add(name, "performs", cat, "cs")
    add(name, "time_complexity", complexity, "cs")

add("cpu", "executes", "instructions", "cs")
add("ram", "stores", "active_data", "cs")
add("operating_system", "manages", "hardware_resources", "cs")
add("compiler", "translates", "source_code_to_machine_code", "cs")
add("database", "persists", "structured_data", "cs")
add("api", "defines", "interface_between_systems", "cs")
add("http", "protocol_for", "web_communication", "cs")
add("tcp_ip", "underlies", "internet_protocols", "cs")
add("encryption", "protects", "data_confidentiality", "cs")
add("public_key_crypto", "uses", "key_pairs", "cs")
add("machine_learning", "learns", "patterns_from_data", "cs")
add("neural_network", "approximates", "complex_functions", "cs")
add("recursion", "calls", "itself", "cs")
add("stack", "follows", "lifo_order", "cs")
add("queue", "follows", "fifo_order", "cs")

# ── GEOGRAPHY ──────────────────────────────────────────────────────────
geo = [
    ("paris", "capital_of", "france"), ("london", "capital_of", "uk"),
    ("tokyo", "capital_of", "japan"), ("washington_dc", "capital_of", "usa"),
    ("moscow", "capital_of", "russia"), ("beijing", "capital_of", "china"),
    ("delhi", "capital_of", "india"), ("berlin", "capital_of", "germany"),
    ("rome", "capital_of", "italy"), ("madrid", "capital_of", "spain"),
    ("cairo", "capital_of", "egypt"), ("brasilia", "capital_of", "brazil"),
    ("ottawa", "capital_of", "canada"), ("canberra", "capital_of", "australia"),
]
for city, country in [(c, co) for c, co in geo]:
    pass  # already added above
for c, co in geo:
    add(co, "has_capital", c, "geography")

countries_continents = {
    "france": "europe", "uk": "europe", "germany": "europe",
    "italy": "europe", "spain": "europe", "russia": "europe_and_asia",
    "japan": "asia", "china": "asia", "india": "asia",
    "usa": "north_america", "canada": "north_america",
    "brazil": "south_america", "egypt": "africa", "australia": "oceania",
}
for country, continent in countries_continents.items():
    add(country, "is_in", continent, "geography")

rivers = [("nile", "africa"), ("amazon", "south_america"),
          ("mississippi", "north_america"), ("yangtze", "asia"),
          ("danube", "europe")]
for river, cont in rivers:
    add(river, "is_a", "river", "geography")
    add(river, "flows_through", cont, "geography")

mountains = [("everest", "8849"), ("k2", "8611"), ("kilimanjaro", "5895"),
             ("denali", "6190"), ("mont_blanc", "4808")]
for mtn, h in mountains:
    add(mtn, "is_a", "mountain", "geography")
    add(mtn, "has_height_meters", h, "geography")

oceans = ["pacific_ocean", "atlantic_ocean", "indian_ocean", "arctic_ocean", "southern_ocean"]
for oc in oceans:
    add(oc, "is_a", "ocean", "geography")

continents_list = ["africa", "antarctica", "asia", "europe", "north_america",
                   "south_america", "oceania"]
for cont in continents_list:
    add(cont, "is_a", "continent", "geography")

# ── HISTORY ────────────────────────────────────────────────────────────
history_events = [
    ("world_war_1", "1914_1918"), ("world_war_2", "1939_1945"),
    ("french_revolution", "1789_1799"), ("american_revolution", "1775_1783"),
    ("industrial_revolution", "1760_1840"),
    ("renaissance", "14th_to_17th_century"),
    ("moon_landing", "1969"), ("fall_of_berlin_wall", "1989"),
]
for event, period in history_events:
    add(event, "occurred_during", period, "history")
    add(event, "is_a", "historical_event", "history")

historical_figures = [
    ("cleopatra", "egyptian_queen"), ("napoleon", "french_emperor"),
    ("julius_caesar", "roman_general"), ("genghis_khan", "mongol_leader"),
    ("winston_churchill", "british_prime_minister"),
    ("mahatma_gandhi", "indian_leader"), ("nelson_mandela", "south_african_president"),
    ("martin_luther_king", "civil_rights_leader"),
    ("abraham_lincoln", "us_president"), ("george_washington", "first_us_president"),
]
for name, role in historical_figures:
    add(name, "was_a", role, "history")
    add(name, "is_a", "historical_figure", "history")

inventions = [
    ("printing_press", "gutenberg"), ("telephone", "bell"),
    ("light_bulb", "edison"), ("airplane", "wright_brothers"),
    ("telescope", "galileo"), ("steam_engine", "watt"),
    ("world_wide_web", "berners_lee"),
]
for inv, inventor in inventions:
    add(inv, "was_invented_by", inventor, "history")
    add(inv, "is_a", "invention", "history")

# ── ASTRONOMY / SPACE ──────────────────────────────────────────────────
planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn",
           "uranus", "neptune"]
for i, p in enumerate(planets):
    add(p, "is_a", "planet", "astronomy")
    add(p, "orbits", "the_sun", "astronomy")
    if p == "earth":
        add(p, "has", "liquid_water", "astronomy")
        add(p, "supports", "known_life", "astronomy")
    elif p == "mars":
        add(p, "is_called", "the_red_planet", "astronomy")
    elif p == "jupiter":
        add(p, "is_the_largest", "planet", "astronomy")

add("sun", "is_a", "star", "astronomy")
add("sun", "is_center_of", "solar_system", "astronomy")
add("moon", "orbits", "earth", "astronomy")
add("milky_way", "is_a", "galaxy", "astronomy")
add("black_hole", "has_gravity_so_strong_that", "nothing_escapes", "astronomy")
add("supernova", "is_explosion_of", "dying_star", "astronomy")
add("asteroid", "orbits", "the_sun", "astronomy")
add("comet", "has", "tail_when_near_sun", "astronomy")
add("saturn", "has", "prominent_rings", "astronomy")

# ── MEDICINE / HEALTH ──────────────────────────────────────────────────
organs_systems = [
    ("heart", "circulatory_system"), ("lungs", "respiratory_system"),
    ("brain", "nervous_system"), ("stomach", "digestive_system"),
    ("liver", "detoxification"), ("kidneys", "filtration"),
    ("skin", "integumentary_system"),
]
for organ, system in organs_systems:
    add(organ, "is_part_of", system, "medicine")
    add(system, "is_a", "body_system", "medicine")

common_conditions = [
    ("diabetes", "blood_sugar_regulation"), ("hypertension", "blood_pressure"),
    ("asthma", "breathing_difficulty"), ("arthritis", "joint_inflammation"),
]
for cond, desc in common_conditions:
    add(cond, "is_a", "medical_condition", "medicine")
    add(cond, "involves", desc, "medicine")

add("exercise", "improves", "cardiovascular_health", "health")
add("balanced_diet", "supports", "immune_function", "health")
add("adequate_sleep", "restores", "cognitive_function", "health")
add("hydration", "maintains", "bodily_functions", "health")

# ── LITERATURE / ART / MUSIC ───────────────────────────────────────────
books_authors = [
    ("hamlet", "shakespeare"), ("iliad", "homer"),
    ("dante_inferno", "dante"), ("don_quixote", "cervantes"),
    ("war_and_peace", "tolstoy"), ("pride_and_prejudice", "austen"),
    ("moby_dick", "melville"), ("1984", "orwell"),
]
for book, author in books_authors:
    add(book, "was_written_by", author, "literature")
    add(book, "is_a", "book", "literature")

composers_works = [
    ("beethoven", "composed", "symphony_no_9"), ("mozart", "composed", "magic_flute"),
    ("bach", "composed", "brandenburg_concertos"), ("chopin", "composed", "nocturnes"),
]
for comp, verb, work in composers_works:
    add(comp, verb, work, "music")

art_pieces = [
    ("mona_lisa", "da_vinci"), ("starry_night", "van_gogh"),
    ("the_scream", "munch"), ("guernica", "picasso"),
]
for piece, artist in art_pieces:
    add(piece, "was_painted_by", artist, "art")

# ── ECONOMICS / PHILOSOPHY ─────────────────────────────────────────────
econ_concepts = [
    ("supply_and_demand", "determines", "market_price"),
    ("inflation", "means", "rising_prices"),
    ("gdp", "measures", "national_output"),
    ("interest_rate", "cost_of", "borrowing_money"),
    ("monopoly", "means", "single_market_dominator"),
]
for s, r, o in econ_concepts:
    add(s, r, o, "economics")

philosophy_ideas = [
    ("socrates", "said", "know_thyself"),
    ("plato", "proposed", "theory_of_forms"),
    ("aristotle", "founded", "formal_logic"),
    ("descartes", "said", "i_think_therefore_i_am"),
    ("kant", "developed", "categorical_imperative"),
]
for s, r, o in philosophy_ideas:
    add(s, r, o, "philosophy")

# ═══════════════════════════════════════════════════════════════════════
# WRITE OUTPUT
# ═══════════════════════════════════════════════════════════════════════
with open(OUT, 'w') as f:
    seen = set()
    for s, r, o, domain in facts:
        key = (s, r, o)
        if key in seen: continue
        seen.add(key)
        f.write(f"{s} {r} {o}\n")

print(f"Wrote {len(seen)} unique facts to {OUT}")
