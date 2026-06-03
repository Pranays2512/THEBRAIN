import random

ontology = [
    # IS-A relations
    ("dog", "isa", "mammal"),
    ("cat", "isa", "mammal"),
    ("mammal", "isa", "animal"),
    ("bird", "isa", "animal"),
    ("fish", "isa", "animal"),
    ("apple", "isa", "fruit"),
    ("banana", "isa", "fruit"),
    ("fruit", "isa", "food"),
    ("pizza", "isa", "food"),
    ("water", "isa", "liquid"),
    ("milk", "isa", "liquid"),
    ("car", "isa", "vehicle"),
    ("truck", "isa", "vehicle"),
    ("airplane", "isa", "vehicle"),
    ("vehicle", "isa", "machine"),
    ("computer", "isa", "machine"),
    ("earth", "isa", "planet"),
    ("mars", "isa", "planet"),
    ("sun", "isa", "star"),
    
    # HAS-A / PART-OF relations
    ("dog", "has", "fur"),
    ("dog", "has", "paws"),
    ("dog", "has", "tail"),
    ("cat", "has", "fur"),
    ("cat", "has", "claws"),
    ("bird", "has", "wings"),
    ("bird", "has", "feathers"),
    ("bird", "has", "beak"),
    ("fish", "has", "scales"),
    ("fish", "has", "gills"),
    ("car", "has", "wheels"),
    ("car", "has", "engine"),
    ("airplane", "has", "wings"),
    ("computer", "has", "cpu"),
    ("computer", "has", "memory"),
    ("earth", "has", "water"),
    ("earth", "has", "life"),
    
    # CAPABILITY / MAKES relations
    ("dog", "makes", "bark"),
    ("cat", "makes", "meow"),
    ("bird", "can", "fly"),
    ("fish", "can", "swim"),
    ("airplane", "can", "fly"),
    ("car", "can", "drive"),
    ("computer", "can", "compute"),
    ("sun", "emits", "light"),
    ("sun", "emits", "heat"),
    ("water", "can", "flow"),
    
    # PROPERTIES
    ("apple", "color", "red"),
    ("banana", "color", "yellow"),
    ("sky", "color", "blue"),
    ("grass", "color", "green"),
    ("sun", "color", "yellow"),
    ("apple", "taste", "sweet"),
    ("water", "taste", "neutral"),
    ("pizza", "taste", "savory"),
    
    # LOCATIONS
    ("fish", "livesin", "water"),
    ("bird", "livesin", "trees"),
    ("earth", "isin", "space"),
    ("moon", "isin", "space")
]

# Write to text file so we can stream it later or just parse it here
with open("ontology_dataset.txt", "w") as f:
    for subj, rel, obj in ontology:
        f.write(f"{subj} {rel} {obj}\n")

print(f"Generated {len(ontology)} facts!")
