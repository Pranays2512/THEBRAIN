from adapters.llm_adapter import OllamaClient, _first_json, EYES_SYSTEM
import json

c = OllamaClient("gpt-oss:120b-cloud")
q = "If one were to inquire about the geographical whereabouts of the institution known as ncert, what would be the most precise city of its location?"
# Try appending a list of valid relations
solvable = ["isa", "part_of", "located_in", "has_property", "mass", "speed"]
prompt = q + "\n\nPick the relation EXACTLY from this list of valid relations: " + ", ".join(solvable)
ans = c.complete(prompt, EYES_SYSTEM)
print("Parsed:", _first_json(ans))
