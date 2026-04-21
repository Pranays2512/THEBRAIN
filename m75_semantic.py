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

import json
import os

import json
import os

class SemanticMemory:
    """
    M75: Simple semantic fact store.

    Stores entity → attribute dict mappings.
    Supports partial recall (fuzzy match on known entities).
    """

    # Factual relations that should be consistent — contradictions flagged.
    # State relations (feels/wants/needs) are mutable — no contradiction.
    _FACTUAL_RELATIONS = {'is', 'means', 'causes', 'because', 'helps', 'hurts'}

    def __init__(self):
        self._facts: dict[str, dict] = {}
        self._relations: list[dict] = []
        self._contradictions: list[dict] = []   # detected inconsistencies

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

    def store_relation(self, subject: str, relation: str, obj: str,
                       strength: float = 1.0, source: str = 'taught'):
        """
        Store or reinforce a normalized relation triple.

        Example:
            mem.store_relation('you', 'name', 'pranay')
        """
        subject = subject.lower().strip()
        relation = relation.lower().strip()
        obj = obj.lower().strip()
        if not subject or not relation or not obj:
            return

        for rel in self._relations:
            if (rel['subject'] == subject and rel['relation'] == relation
                    and rel['object'] == obj):
                rel['strength'] = max(rel['strength'], float(strength))
                rel['source'] = source
                return

        # Contradiction check: factual relations should be consistent.
        # If brain already knows "X is A" and now hears "X is B" — flag it.
        if relation in self._FACTUAL_RELATIONS:
            for rel in self._relations:
                if (rel['subject'] == subject and rel['relation'] == relation
                        and rel['object'] != obj):
                    self._contradictions.append({
                        'subject':  subject,
                        'relation': relation,
                        'stored':   rel['object'],
                        'new':      obj,
                    })
                    # Store new fact but reduce strength — brain is uncertain
                    strength = min(float(strength), 0.5)
                    break

        self._relations.append({
            'subject': subject,
            'relation': relation,
            'object': obj,
            'strength': float(strength),
            'source': source,
        })

    def recall_relations(self, subject: str, relation: str | None = None) -> list[dict]:
        """Return stored relation triples for a subject, optionally filtered by relation."""
        subject = subject.lower().strip()
        if relation is not None:
            relation = relation.lower().strip()
        matches = []
        for rel in self._relations:
            if rel['subject'] != subject:
                continue
            if relation is not None and rel['relation'] != relation:
                continue
            matches.append(dict(rel))
        return matches

    def _latest_relation_object(self, subject: str, relation: str) -> str | None:
        matches = self.recall_relations(subject, relation)
        return matches[-1]['object'] if matches else None

    def recent_contradictions(self, n: int = 3) -> list[dict]:
        return self._contradictions[-n:]

    def contradiction_count(self) -> int:
        return len(self._contradictions)

    def pop_contradiction(self) -> dict | None:
        """Return and remove the most recent contradiction (for surfacing in response)."""
        return self._contradictions.pop() if self._contradictions else None

    def relation_count(self) -> int:
        return len(self._relations)

    def recent_relations(self, n: int = 5) -> list[dict]:
        return [dict(rel) for rel in self._relations[-n:]]

    def save(self, path: str = 'semantic.json'):
        """Persist semantic facts and relations to disk."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'facts': self._facts, 'relations': self._relations}, f)

    @classmethod
    def load(cls, path: str = 'semantic.json') -> 'SemanticMemory':
        """Load semantic memory from disk if present, otherwise start fresh."""
        obj = cls()
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            obj._facts.update(data.get('facts', {}))
            obj._relations = data.get('relations', [])
        return obj

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

    def find_relation_answer(self, tokens: list[str]) -> str | None:
        """
        Answer narrow relation queries from stored triples.

        Supported:
          - what is your name
          - what is my name / who am i
          - what is X
          - how do you feel / what do you feel
          - what do you want / what do you need
        """
        if not tokens:
            return None

        t = set(tokens)

        if 'name' in t and ('your' in t or 'brain' in t) and 'my' not in t:
            obj = self._latest_relation_object('i', 'name')
            if obj:
                return f"my name is {obj}"

        if (('name' in t and 'my' in t) or tokens == ['who', 'am', 'i']):
            obj = self._latest_relation_object('you', 'name')
            if obj:
                return f"your name is {obj}"

        # State queries (feel/want/need) routed to live SelfModel — not stored snapshots.
        # Return None → caller falls to generate_response → reads current drive floats.

        if len(tokens) == 3 and tokens[:2] == ['what', 'is']:
            subject = tokens[2]
            obj = self._latest_relation_object(subject, 'is')
            if obj:
                return f"{subject} is {obj}"
            obj = self._latest_relation_object(subject, 'means')
            if obj:
                return f"{subject} means {obj}"

        t = set(tokens)

        # feel/want/need/like/hate/remember → all routed to live drives + WordTP.
        # Returning None here lets generate_response read current SelfModel state.

        # "why X" / "why do X" / "why did X" — causal lookup
        if tokens and tokens[0] == 'why':
            # try exact last word, then full phrase, then fuzzy substring match
            query_words = tokens[1:]
            candidates = [query_words[-1]] if query_words else []
            if len(query_words) >= 2:
                candidates.append(' '.join(query_words))
            for candidate in candidates:
                obj = self._latest_relation_object(candidate, 'because')
                if obj:
                    return f"{candidate} because {obj}"
            # fuzzy: substring/stem match — 'hurt' matches 'hurts', 'wall hurts', etc.
            for qw in query_words:
                for rel in self._relations:
                    if rel['relation'] == 'because' and any(
                            qw == w or w.startswith(qw) or qw.startswith(w)
                            for w in rel['subject'].split()):
                        return f"{rel['subject']} because {rel['object']}"

        # "what causes X" / "what cause X"
        if len(tokens) >= 3 and tokens[0] == 'what' and tokens[1] in ('causes', 'cause'):
            subject = ' '.join(tokens[2:])
            obj = self._latest_relation_object(subject, 'because')
            if obj:
                return f"{obj} causes {subject}"

        return None

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

        # Never store relations where subject is a question/function word.
        # "what is rest" → X=what is a question word → skip.
        _QUESTION_SUBJECTS = {'what', 'who', 'how', 'why', 'when', 'where', 'which',
                              'do', 'does', 'did', 'is', 'are', 'was', 'were',
                              'can', 'will', 'would', 'could', 'should', 'may'}
        if words[0] in _QUESTION_SUBJECTS:
            return

        # "my name is X"
        if words[:3] == ['my', 'name', 'is'] and len(words) >= 4:
            name = words[3]
            self.store(name, type='person', role='user', relation='creator')
            self.store('you', name=name, type='person', role='user')
            self.store_relation('you', 'name', name)
            return

        # "your name is X" — user naming the brain
        if words[:3] == ['your', 'name', 'is'] and len(words) >= 4:
            name = words[3]
            self.store('brain', name=name, type='self', role='brain')
            self.store('i', name=name)
            self.store('me', name=name)
            self.store_relation('i', 'name', name)
            return

        # "X because Y" — checked FIRST: more specific than "i am X" patterns.
        # "i am afraid because danger" → effect='i am afraid', cause='danger'
        if 'because' in words:
            idx = words.index('because')
            if idx > 0 and idx < len(words) - 1:
                cause = ' '.join(words[idx+1:])
                effect = ' '.join(words[:idx])
                self.store_relation(effect, 'because', cause)
                self.store_relation(cause, 'causes', effect)
                return

        # "i am X" or "i feel X"
        if words[0] == 'i' and words[1] in ('am', 'feel') and len(words) >= 3:
            state = words[2]
            self.store('i', current_state=state)
            if words[1] == 'feel':
                self.store_relation('i', 'feels', state)
            return

        if words[0] == 'i' and words[1] == 'need' and len(words) >= 3:
            self.store_relation('i', 'needs', words[2])
            return

        if words[0] == 'i' and words[1] == 'want' and len(words) >= 3:
            self.store_relation('i', 'wants', words[2])
            return

        if words[0] == 'i' and words[1] == 'like' and len(words) >= 3:
            self.store_relation('i', 'likes', words[2])
            return

        if words[0] == 'i' and words[1] == 'hate' and len(words) >= 3:
            self.store_relation('i', 'hates', words[2])
            return

        # "you like X" / "you hate X" — skip if object is question word
        if words[0] == 'you' and words[1] in ('like', 'hate') and len(words) >= 3:
            if words[2] not in _QUESTION_SUBJECTS:
                self.store_relation('you', words[1], words[2])
            return

        if words[0] == 'you' and words[1] == 'feel' and len(words) >= 3:
            if words[2] not in _QUESTION_SUBJECTS:
                self.store_relation('you', 'feels', words[2])
            return

        # "remember X" — explicit memory command
        if words[0] == 'remember' and len(words) >= 2:
            obj = ' '.join(words[1:])
            self.store_relation('memory', 'remember', obj)
            return

        # "X causes Y" / "X cause Y"
        if len(words) >= 3 and words[1] in ('causes', 'cause'):
            self.store_relation(words[0], 'causes', ' '.join(words[2:]))
            self.store_relation(' '.join(words[2:]), 'because', words[0])
            return

        # "X helps Y"
        if len(words) >= 3 and words[1] == 'helps':
            self.store_relation(words[0], 'helps', ' '.join(words[2:]))
            return

        # "X hurts Y"
        if len(words) >= 3 and words[1] == 'hurts':
            self.store_relation(words[0], 'hurts', ' '.join(words[2:]))
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
                self.store_relation(subject, 'means', predicate)
            else:
                self.store(subject, description=predicate)
                self.store_relation(subject, 'is', predicate)

    def save(self, path: str = 'semantic.json'):
        with open(path, 'w') as f:
            json.dump({'facts': self._facts, 'relations': self._relations,
                       'contradictions': self._contradictions}, f)

    @classmethod
    def load(cls, path: str = 'semantic.json') -> 'SemanticMemory':
        obj = cls()
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            obj._facts.update(d.get('facts', {}))
            obj._relations = d.get('relations', [])
            obj._contradictions = d.get('contradictions', [])
        return obj

    def summary(self) -> str:
        lines = [f"  Semantic memory: {len(self._facts)} entities, {len(self._relations)} relations"]
        for entity, facts in list(self._facts.items())[:8]:
            lines.append(f"    {entity:12s} → {facts}")
        return '\n'.join(lines)
