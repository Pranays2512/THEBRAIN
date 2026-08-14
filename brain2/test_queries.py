from faculties.whole_brain import WholeBrain
from adapters.llm_adapter import OllamaClient, LLMEyes

eyes = LLMEyes(OllamaClient("gpt-oss:120b-cloud"))
brain = WholeBrain(eyes=eyes)

queries = [
    "If a substance turns red litmus blue, what is its nature?",
    "What is the fastest mode of transport?",
    "What happens when zinc reacts with hydrochloric acid?",
    "How did the non-cooperation movement originate?",
    "What is your personal opinion on carbon atoms?"
]

for q in queries:
    print(f"\nQ: {q}")
    ans = brain.ask(q)
    print("A:", ans)
