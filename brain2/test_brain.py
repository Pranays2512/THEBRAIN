from faculties.whole_brain import WholeBrain
from adapters.llm_adapter import OllamaClient, LLMEyes

eyes = LLMEyes(OllamaClient("gpt-oss:120b-cloud"))
brain = WholeBrain(eyes=eyes)
q = "If one were to inquire about the geographical whereabouts of the institution known as ncert, what would be the most precise city of its location?"
ans = brain.ask(q)
print("Brain Ask:", ans)
