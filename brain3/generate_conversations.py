import random
import re

questions = [
    "how are you",
    "how is your day",
    "how are you doing",
    "what is up",
    "how goes it"
]

good_answers = [
    "status optimal emotion happy",
    "status good energy high",
    "state positive mode ready",
    "feeling great condition excellent"
]

who_q = [
    "who are you",
    "what is your name",
    "what are you",
    "tell me about yourself"
]

who_a = [
    "identity system type cognitive",
    "identity brain origin artificial",
    "self network type neural",
    "name brain type ai"
]

doing_q = [
    "what are you doing",
    "what do you do",
    "what is your purpose"
]

doing_a = [
    "action learning goal communication",
    "task processing goal understanding",
    "purpose learning objective thinking",
    "activity reading state processing"
]

greetings_q = [
    "hello",
    "hi",
    "hey there",
    "greetings",
    "good morning",
    "good evening"
]

greetings_a = [
    "intent greeting style friendly",
    "intent salutation status ready",
    "intent greeting emotion positive",
    "intent welcome target user"
]

conversations = []
for _ in range(500):
    conversations.append(f"{random.choice(questions)} {random.choice(good_answers)}")
    conversations.append(f"{random.choice(who_q)} {random.choice(who_a)}")
    conversations.append(f"{random.choice(doing_q)} {random.choice(doing_a)}")
    conversations.append(f"{random.choice(greetings_q)} {random.choice(greetings_a)}")

random.shuffle(conversations)

with open("brain_conversations.txt", "w") as f:
    for c in conversations:
        # ensure it's clean (lowercase, no punctuation)
        clean = re.sub(r'[^a-z ]', '', c.lower())
        f.write(clean + "\n")

print(f"Generated {len(conversations)} conversational exchanges in brain_conversations.txt")
