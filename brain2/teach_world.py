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

# 5-year-old Knowledge Graph (Massive Ontology)
basics = [
    # Family
    ("mother", "isa", "parent"),
    ("father", "isa", "parent"),
    ("brother", "isa", "sibling"),
    ("sister", "isa", "sibling"),
    ("baby", "isa", "child"),
    ("grandmother", "isa", "grandparent"),
    ("grandfather", "isa", "grandparent"),
    ("family", "has", "people"),
    
    # Body Parts
    ("head", "isa", "bodypart"),
    ("eye", "isa", "bodypart"),
    ("nose", "isa", "bodypart"),
    ("mouth", "isa", "bodypart"),
    ("ear", "isa", "bodypart"),
    ("hand", "isa", "bodypart"),
    ("leg", "isa", "bodypart"),
    ("foot", "isa", "bodypart"),
    ("hair", "isa", "bodypart"),
    ("tooth", "isa", "bodypart"),
    ("human", "has", "head"),
    ("human", "has", "eye"),
    ("human", "has", "mouth"),
    ("human", "has", "hand"),
    ("human", "has", "leg"),
    
    # Senses
    ("eye", "can", "see"),
    ("ear", "can", "hear"),
    ("nose", "can", "smell"),
    ("mouth", "can", "eat"),
    ("mouth", "can", "talk"),
    ("hand", "can", "touch"),
    
    # Animals & Traits
    ("dog", "isa", "animal"),
    ("cat", "isa", "animal"),
    ("cow", "isa", "animal"),
    ("bird", "isa", "animal"),
    ("pig", "isa", "animal"),
    ("lion", "isa", "animal"),
    ("frog", "isa", "animal"),
    ("fish", "isa", "animal"),
    ("horse", "isa", "animal"),
    ("duck", "isa", "animal"),
    ("mouse", "isa", "animal"),
    ("elephant", "isa", "animal"),
    ("bear", "isa", "animal"),
    
    ("bird", "can", "fly"),
    ("fish", "can", "swim"),
    ("dog", "can", "run"),
    ("frog", "can", "jump"),
    ("horse", "can", "run"),
    
    ("dog", "says", "woof"),
    ("cat", "says", "meow"),
    ("cow", "says", "moo"),
    ("pig", "says", "oink"),
    ("duck", "says", "quack"),
    ("lion", "says", "roar"),
    
    # Colors
    ("red", "isa", "color"),
    ("blue", "isa", "color"),
    ("green", "isa", "color"),
    ("yellow", "isa", "color"),
    ("orange", "isa", "color"),
    ("purple", "isa", "color"),
    ("pink", "isa", "color"),
    ("black", "isa", "color"),
    ("white", "isa", "color"),
    ("brown", "isa", "color"),
    
    ("apple", "hascolor", "red"),
    ("sky", "hascolor", "blue"),
    ("grass", "hascolor", "green"),
    ("sun", "hascolor", "yellow"),
    ("orange", "hascolor", "orange"),
    ("grape", "hascolor", "purple"),
    ("pig", "hascolor", "pink"),
    ("night", "hascolor", "black"),
    ("cloud", "hascolor", "white"),
    ("bear", "hascolor", "brown"),
    
    # Shapes
    ("circle", "isa", "shape"),
    ("square", "isa", "shape"),
    ("triangle", "isa", "shape"),
    ("star", "isa", "shape"),
    ("heart", "isa", "shape"),
    ("ball", "hasshape", "circle"),
    ("box", "hasshape", "square"),
    
    # Food & Drink
    ("apple", "isa", "fruit"),
    ("banana", "isa", "fruit"),
    ("grape", "isa", "fruit"),
    ("orange", "isa", "fruit"),
    ("carrot", "isa", "vegetable"),
    ("potato", "isa", "vegetable"),
    ("corn", "isa", "vegetable"),
    ("bread", "isa", "food"),
    ("cheese", "isa", "food"),
    ("pizza", "isa", "food"),
    ("candy", "isa", "food"),
    ("cake", "isa", "food"),
    ("water", "isa", "drink"),
    ("milk", "isa", "drink"),
    ("juice", "isa", "drink"),
    
    ("candy", "is", "sweet"),
    ("apple", "is", "sweet"),
    ("lemon", "is", "sour"),
    ("pizza", "is", "yummy"),
    
    # Nature & Weather
    ("sun", "isa", "star"),
    ("moon", "isa", "nature"),
    ("rain", "isa", "weather"),
    ("snow", "isa", "weather"),
    ("wind", "isa", "weather"),
    ("tree", "isa", "plant"),
    ("flower", "isa", "plant"),
    ("grass", "isa", "plant"),
    ("dirt", "isa", "nature"),
    ("rock", "isa", "nature"),
    
    ("sun", "is", "hot"),
    ("snow", "is", "cold"),
    ("rain", "is", "wet"),
    ("fire", "is", "hot"),
    ("ice", "is", "cold"),
    
    # Home & Objects
    ("house", "isa", "building"),
    ("home", "isa", "place"),
    ("room", "isa", "place"),
    ("bed", "isa", "furniture"),
    ("chair", "isa", "furniture"),
    ("table", "isa", "furniture"),
    ("door", "isa", "object"),
    ("window", "isa", "object"),
    ("toy", "isa", "object"),
    ("book", "isa", "object"),
    ("ball", "isa", "toy"),
    ("block", "isa", "toy"),
    ("doll", "isa", "toy"),
    ("car", "isa", "vehicle"),
    ("bus", "isa", "vehicle"),
    ("train", "isa", "vehicle"),
    ("airplane", "isa", "vehicle"),
    ("boat", "isa", "vehicle"),
    
    ("bed", "isfor", "sleep"),
    ("chair", "isfor", "sit"),
    ("cup", "isfor", "drink"),
    ("plate", "isfor", "eat"),
    
    # Clothing
    ("shirt", "isa", "clothes"),
    ("pants", "isa", "clothes"),
    ("shoe", "isa", "clothes"),
    ("sock", "isa", "clothes"),
    ("hat", "isa", "clothes"),
    ("coat", "isa", "clothes"),
    
    # Opposites (Stored as distinct relations)
    ("hot", "opposite", "cold"),
    ("big", "opposite", "small"),
    ("fast", "opposite", "slow"),
    ("happy", "opposite", "sad"),
    ("good", "opposite", "bad"),
    ("up", "opposite", "down"),
    ("day", "opposite", "night"),
    ("in", "opposite", "out"),
    ("on", "opposite", "off"),
    ("yes", "opposite", "no"),
    
    # People & Occupations (Simple)
    ("doctor", "isa", "person"),
    ("teacher", "isa", "person"),
    ("police", "isa", "person"),
    ("firefighter", "isa", "person"),
    ("doctor", "can", "help"),
    ("teacher", "can", "teach"),
    
    # Time
    ("day", "has", "sun"),
    ("night", "has", "moon"),
    ("morning", "isa", "time"),
    ("night", "isa", "time")
]

print(f"Teaching {len(basics)} basic facts about the world (5-year-old level)...")

for subj_w, rel_w, obj_w in basics:
    if not b.symbolic_table.knows(subj_w): b.learn_word(subj_w)
    if not b.symbolic_table.knows(rel_w): b.learn_word(rel_w)
    if not b.symbolic_table.knows(obj_w): b.learn_word(obj_w)
    
    subj_vec = b.language.encode(subj_w)
    rel_vec = b.language.encode(rel_w)
    obj_vec = b.language.encode(obj_w)
    
    b.bind_triple(subj_vec, rel_vec, obj_vec)
    
    b.perceive(subj_vec)
    b.perceive(rel_vec)
    b.perceive(obj_vec)

print("Saving Brain state...")
b.save_components(ckpt_dir)

print("Done! The Brain now understands the basics of the world at a 5-year-old level.")
