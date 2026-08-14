from adapters.llm_adapter import OllamaClient, _first_json, EYES_SYSTEM
import json

c = OllamaClient("gpt-oss:120b-cloud")
q = "If one were to inquire about the geographical whereabouts of the institution known as ncert, what would be the most precise city of its location?"
ans = c.complete(q, EYES_SYSTEM)
print("Raw ans:", ans)
try:
    print("Parsed:", _first_json(ans))
except Exception as e:
    print("JSON Error:", e)
