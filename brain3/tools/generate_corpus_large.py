#!/usr/bin/env python3
"""Generate massive corpus for brain3 ingestion."""
import random
random.seed(42)
lines = []

def A(s, r, o):
    lines.append(f"{s} {r} {o}")

# ANIMALS
for name, cls in [("dog","mammal"),("cat","mammal"),("lion","mammal"),
    ("tiger","mammal"),("elephant","mammal"),("dolphin","mammal"),
    ("whale","mammal"),("wolf","mammal"),("bear","mammal"),
    ("horse","mammal"),("cow","mammal"),("sheep","mammal"),
    ("eagle","bird"),("owl","bird"),("penguin","bird"),("parrot","bird"),
    ("salmon","fish"),("shark","fish"),("trout","fish"),
    ("snake","reptile"),("turtle","reptile"),("frog","amphibian"),
    ("bee","insect"),("ant","insect"),("butterfly","insect")]:
    A(name, "isa", cls)
    A(name, "isa", "animal")

# add specific properties
for name in ["dog","cat","lion","tiger","elephant","whale","bear"]:
    A(name, "isa", "mammal_species")

for name in ["dog","wolf","fox"]:
    A(name, "isa", "canine")
for name in ["lion","tiger","leopard"]:
    A(name, "isa", "feline")

# BIOLOGY facts
for s, r, o in [
    ("heart_organ", "pumps_blood_throughout", "the_body"),
    ("lungs_organ", "exchanges_oxygen_for", "carbon_dioxide"),
    ("brain_organ", "controls", "all_body_functions"),
    ("dna_molecule", "stores", "genetic_information"),
    ("cell_unit", "is_basic_unit_of", "all_living_things"),
    ("mitochondria_organelle", "produces", "atp_energy_molecule"),
    ("chlorophyll_pigment", "enables", "photosynthesis_in_plants"),
    ("protein_molecule", "builds_and_repairs", "body_tissues"),
    ("enzyme_protein", "accelerates", "biochemical_reactions"),
    ("neuron_cell", "transmits", "electrical_signals"),
    ("red_blood_cells", "transport", "oxygen_to_tissues"),
    ("white_blood_cells", "fight", "infection_and_disease"),
    ("immune_system_defense", "protects_against", "pathogens"),
    ("vaccine_injection", "trains", "immune_system_recognition"),
    ("antibiotic_drug", "kills", "harmful_bacteria"),
]:
    A(s, r, o)

# CHEMISTRY - elements
for name, sym in [("hydrogen","H"),("helium","He"),("carbon","C"),
    ("nitrogen","N"),("oxygen","O"),("iron","Fe"),("gold","Au"),
    ("silver","Ag"),("copper","Cu"),("zinc","Zn")]:
    A(name, "isa", "chemical_element")
    A(name, "has_symbol", sym)

# compounds
for name, formula in [("water_molecule","H2O"),("table_salt","NaCl"),
    ("carbon_dioxide_gas","CO2"),("methane_gas","CH4")]:
    A(name, "isa", "chemical_compound")
    A(name, "has_formula", formula)

# concepts
for name, desc in [("acid_substance","donates protons"),
    ("base_substance","accepts protons"),("catalyst_agent","speeds reactions"),
    ("polymer_chain","consists of monomers"),("ion_particle","charged atom")]:
    A(name, "isa", "chemistry_concept")
    A(name, "involves", desc)

# PHYSICS
for s, r, o in [
    ("gravity_force", "attracts", "objects_with_mass"),
    ("light_photon", "travels_at_speed_of", "299792458_mps"),
    ("energy_law_conservation", "total_energy_remains_constant", "closed_systems"),
    ("entropy_thermodynamics", "always_increases_in", "isolated_systems"),
    ("newton_first_motion_law", "objects_at_rest_stay_at_rest", "without_net_force"),
    ("newton_second_motion_law", "force_equals_mass_times_acceleration", ""),
    ("newton_third_motion_law", "every_action_has_equal_opposite_reaction", ""),
    ("atom_structure_model", "contains_nucleus_with_protons_and_neutrons", "surrounded_by_electrons"),
    ("electron_negative_charge", "orbits_atomic_nucleus", "in_shells_or_clouds"),
    ("quantum_mechanics_field", "describes_behavior_of", "matter_at_atomic_scale"),
]:
    A(s, r, o)

# MATHEMATICS
for name in ["two","three","five","seven","eleven","thirteen","seventeen",
             "nineteen","twenty_three","twenty_nine"]:
    A(name, "isa", "prime_number")

for s, r, o in [
    ("pi_constant_math", "approximates", "3.14159_circle_ratio"),
    ("pythagorean_theorem_right_triangle", "relates_legs_to_hypotenuse", "a_squared_plus_b_squared_equals_c_squared"),
    ("derivative_calculus_concept", "measures_instantaneous_rate_of_change", "of_function"),
    ("integral_calculus_concept", "computes_accumulated_area_under", "curve"),
    ("golden_ratio_phi", "approximates", "1.618_fibonacci_limit_ratio"),
    ("euler_number_e", "approximates", "2.71828_natural_log_base"),
]:
    A(s, r, o)

# GEOGRAPHY
for country, capital in [("france","paris"),("germany","berlin"),
    ("italy","rome"),("spain","madrid"),("uk","london"),
    ("japan","tokyo"),("china","beijing"),("india","delhi"),
    ("egypt","cairo"),("usa","washington_dc"),("brazil","brasilia"),
    ("canada","ottawa"),("australia","canberra")]:
    A(capital, "is_capital_of", country)
    A(country, "isa", "country")

for river in ["nile_river_africa","amazon_river_south_america","mississippi_river_north_america"]:
    A(river, "isa", "river")

for ocean in ["pacific_ocean_largest","atlantic_ocean_second","indian_ocean_third"]:
    A(ocean, "isa", "ocean")

for cont in ["africa_continent","asia_continent","europe_continent",
             "north_america_continent","south_america_continent"]:
    A(cont, "isa", "continent")

# COMPUTER SCIENCE
for lang in ["python_programming_language","java_programming_language",
             "cpp_programming_language","javascript_web_language",
             "rust_systems_language","go_golang_language"]:
    A(lang, "isa", "programming_language")
    A(lang, "used_for", "software_development")

for algo in ["binary_search_algorithm_sorted_input","quicksort_partition_based_sorting",
             "mergesort_stable_divide_conquer","dijkstra_shortest_path_graph"]:
    A(algo, "isa", "algorithm")
    A(algo, "has_time_complexity", "polynomial_or_better")

for ds_name in ["hash_table_key_value_lookup","linked_list_sequential_nodes",
                "binary_tree_ordered_hierarchy","stack_last_in_first_out",
                "queue_first_in_first_out","graph_vertices_connected_by_edges"]:
    A(ds_name, "isa", "data_structure")

for concept in ["recursion_self_referential_calls","dynamic_programming_memoization",
                "encryption_data_protection","compiler_source_translation",
                "database_structured_persistence","http_web_communication_protocol"]:
    A(concept, "isa", "cs_concept")

# SPACE
for planet in ["mercury_innermost_planet","venus_hottest_planet",
               "earth_only_life_planet","mars_red_desert_planet",
               "jupiter_largest_gas_giant","saturn_ringed_gas_planet",
               "uranus_sideways_rotation_planet","neptune_windiest_blue_planet"]:
    A(planet, "orbits", "the_sun_star")
    A(planet, "isa", "planet_solar_system_member")

A("sun_central_star_solar_system", "isa", "main_sequence_yellow_dwarf_star")
A("moon_natural_satellite_earth", "orbits_around", "earth_third_planet_from_sun")
A("milky_way_galaxy_home", "contains", "our_entire_solar_system")
A("black_hole_gravity_trap", "nothing_escapes_not_even_light", "once_crossing_event_horizon")

# MEDICINE/HEALTH
for organ, function in [("heart_cardiovascular_pump","circulates_blood_throughout_body"),
    ("lungs_respiratory_exchange","processes_oxygen_and_releases_co2"),
    ("brain_cognitive_center","controls_all_body_functions_and_thought"),
    ("liver_detoxification_filter","processes_nutrients_and_removes_toxins"),
    ("kidneys_waste_filter","filters_blood_and_produces_urine")]:
    A(organ, "isa", "human_organ")
    A(organ, "function_is", function)

for practice, benefit in [
    ("regular_exercise_routine","strengthens_cardiovascular_system"),
    ("balanced_nutrition_intake","provides_energy_and_building_materials"),
    ("adequate_sleep_schedule","restores_cognitive_function_and_memory"),
    ("hydration_water_consumption","maintains_all_bodily_functions"),
    ("meditation_daily_practice","reduces_stress_and_improves_focus")]:
    A(practice, "isa", "health_practice")
    A(practice, "benefit_includes", benefit)

# LITERATURE/ART/MUSIC
for book, author in [("hamlet_play_script","shakespeare"),
    ("odyssey_epic_poem","homer"),("inferno_poem","dante_alighieri"),
    ("don_quixote_novel_text","miguel_de_cervantes"),
    ("war_peace_epic_novel","leo_tolstoy"),
    ("pride_prejudice_romance","jane_austen"),
    ("moby_dick_whale_tale","herman_melville"),
    ("1984_dystopian_novel","george_orwell")]:
    A(book, "was_written_by", author)
    A(book, "isa", "literary_work")

for composer, work in [("beethoven_symphony_composer","ninth_symphony_choral_finale"),
    ("mozart_classical_prodigy","magic_flute_operatic_masterpiece"),
    ("bach_baroque_master","brandenburg_concertos_collection"),
    ("chopin_piano_poet","nocturnes_for_solo_piano")]:
    A(composer, "composed", work)
    A(work, "isa", "classical_music_composition")

for painting, artist in [("mona_lisa_portrait","leonardo_da_vinci"),
    ("starry_night_post_impressionist","vincent_van_gogh"),
    ("scream_expressionist","edvard_munch"),
    ("guernica_cubist_war_scene","pablo_picasso")]:
    A(painting, "was_painted_by", artist)
    A(painting, "isa", "famous_artwork")

# PHILOSOPHY/ECONOMICS
for thinker, idea in [("socrates_greek_philosopher","know_thyself_maxim"),
    ("plato_idealism_founder","theory_of_eternal_forms"),
    ("aristotle_logic_founder","formal_reasoning_system"),
    ("descartes_rationalist","i_think_therefore_i_am"),
    ("kant_ethics_philosopher","categorical_imperative_moral_law")]:
    A(thinker, "proposed", idea)

for concept, definition in [("supply_demand_market_equilibrium","price_where_buyers_sellers_agree"),
    ("opportunity_cost_economics","value_of_best_alternative_foregone"),
    ("gdp_economic_measure","total_goods_services_produced_annually"),
    ("inflation_price_increase","general_level_of_prices_rising_over_time"),
    ("monopoly_market_dominance","single_seller_controls_supply")]:
    A(concept, "isa", "economic_principle")
    A(concept, "means", definition)

with open("/tmp/opencode/massive_corpus.txt", "w") as f:
    for line in lines:
        f.write(line + "\n")

print(f"Generated {len(lines)} facts across multiple domains")
