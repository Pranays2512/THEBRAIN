"""
M75: SEMANTIC MEMORY — Fact Storage and Retrieval
==================================================

Stores hard facts detached from emotion — entity-attribute pairs,
learned definitions, and entity relationships.

Unlike M73 (SemanticBinding) which stores word→emotion associations,
M75 stores word→meaning associations:
    "Pranay" → {type: 'person', role: 'user', relation: 'creator'}
    "food"   → {type: 'object', property: 'edible', reward: 'high'}
    "hot"    → {type: 'property', valence: 'negative', sensation: 'warmth'}

BIOLOGICAL BASIS
----------------
The cortex (specifically inferotemporal cortex and prefrontal cortex)
stores semantic facts as distributed patterns that are stable over time
and independent of the episodic context in which they were learned.

A patient with hippocampal damage loses episodic memory (can't form
new autobiographical memories) but retains semantic memory (still knows
that Paris is in France, that dogs bark, etc.).

This is the "fact database" of the brain — accessed by M68 for inference,
by M74 for more coherent speech, and by the response generator for
grounded answers about named things.

INTERFACE
---------
    mem = SemanticMemory()
    mem.store('pranay', type='person', role='user')
    mem.store('food', type='object', edible=True)
    facts = mem.recall('pranay')   # → {'type': 'person', 'role': 'user'}
    desc  = mem.describe('pranay') # → "pranay is a person"
"""

from collections import defaultdict


class SemanticMemory:
    """
    M75: Simple semantic fact store.

    Stores entity → attribute dict mappings.
    Supports partial recall (fuzzy match on known entities).
    """

    def __init__(self):
        self._facts: dict[str, dict] = {}

        # Pre-load basic facts the brain should know about itself
        self._facts['i']    = {'type': 'self', 'role': 'brain', 'alive': True}
        self._facts['me']   = {'type': 'self', 'role': 'brain', 'alive': True}
        self._facts['you']  = {'type': 'other', 'role': 'user'}
        self._facts['food'] = {'type': 'object', 'edible': True, 'reward': 'high'}
        self._facts['water']= {'type': 'object', 'drinkable': True}
        self._facts['pain'] = {'type': 'sensation', 'valence': 'negative'}
        self._facts['love'] = {'type': 'emotion', 'valence': 'positive', 'social': True}
        self._facts['good'] = {'type': 'evaluation', 'valence': 'positive'}
        self._facts['bad']  = {'type': 'evaluation', 'valence': 'negative'}

    def store(self, entity: str, **attributes):
        """
        Store facts about an entity.
        Merges with existing facts — doesn't overwrite.

        Example:
            mem.store('pranay', type='person', role='creator')
        """
        entity = entity.lower().strip()
        if entity not in self._facts:
            self._facts[entity] = {}
        self._facts[entity].update(attributes)

    def recall(self, entity: str) -> dict:
        """Return all known facts about entity, or {} if unknown."""
        return dict(self._facts.get(entity.lower().strip(), {}))

    def knows(self, entity: str) -> bool:
        return entity.lower().strip() in self._facts

    def describe(self, entity: str) -> str:
        """
        Produce a short natural-language description of entity.
        Used by the response generator to answer "what is X" questions.
        """
        facts = self.recall(entity)
        if not facts:
            return f"i do not know {entity}"
            
        if 'meaning' in facts:
            return f"{entity} means {facts['meaning']}"
        if 'description' in facts:
            return f"{entity} is {facts['description']}"

        typ = facts.get('type', '')
        role = facts.get('role', '')
        valence = facts.get('valence', '')

        if typ == 'self':
            return f"i am a {role}"
        if typ == 'person':
            if role:
                return f"{entity} is {role}"
            return f"{entity} is a person"
        if typ == 'object':
            if facts.get('edible'):
                return f"{entity} is edible"
            if facts.get('drinkable'):
                return f"{entity} is drinkable"
            return f"{entity} is a thing"
        if typ == 'emotion':
            return f"{entity} is a feeling"
        if typ == 'evaluation':
            return f"{entity} is {valence}"
        if typ == 'sensation':
            return f"{entity} is {valence}"
        if role:
            return f"{entity} is {role}"
        return f"i know {entity}"

    def learn_from_sentence(self, words: list):
        """
        Attempt to extract a fact from a sentence pattern.

        Recognizes:
            "X is Y"      → store(X, description=Y)
            "X means Y"   → store(X, meaning=Y)
            "my name is X"→ store(X, type='person', role='user')
            "i am X"      → store(X, type='self_state')
            "you are X"   → store('brain', description=X)
        """
        if len(words) < 3:
            return

        # "my name is X"
        if words[:3] == ['my', 'name', 'is'] and len(words) >= 4:
            name = words[3]
            self.store(name, type='person', role='user', relation='creator')
            self.store('you', name=name, type='person', role='user')
            return

        # "your name is X" — user naming the brain
        if words[:3] == ['your', 'name', 'is'] and len(words) >= 4:
            name = words[3]
            self.store('brain', name=name, type='self', role='brain')
            self.store('i', name=name)
            self.store('me', name=name)
            return

        # "i am X" or "i feel X"
        if words[0] == 'i' and words[1] in ('am', 'feel') and len(words) >= 3:
            state = words[2]
            self.store('i', current_state=state)
            return

        # "you are X"
        if words[0] == 'you' and words[1] == 'are' and len(words) >= 3:
            desc = ' '.join(words[2:])
            self.store('brain', description=desc)
            return

        # "X is Y" or "X means Y"
        if len(words) >= 3 and words[1] in ('is', 'means', 'are'):
            subject = words[0]
            predicate = ' '.join(words[2:])
            if words[1] == 'means':
                self.store(subject, meaning=predicate)
            else:
                self.store(subject, description=predicate)

    def summary(self) -> str:
        lines = [f"  Semantic memory: {len(self._facts)} entities"]
        for entity, facts in list(self._facts.items())[:8]:
            lines.append(f"    {entity:12s} → {facts}")
        return '\n'.join(lines)
