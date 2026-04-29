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
    # Negation relations — stored separately so logic can block false conclusions.
    _NEGATION_RELATIONS = {'is_not', 'cannot', 'does_not'}

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

        self._seed_axioms()
        self.infer_all_axioms()

    def _seed_axioms(self):
        """Pre-load World6-grounded facts and numeric ordering. Called once at init."""
        sr = lambda s, r, o, st=1.5: self.store_relation(s, r, o, strength=st, source='axiom')

        # World6 causal facts — strength 1.5 so experience can strengthen, text can revise
        sr('food',    'is',     'good')
        sr('food',    'causes', 'full')
        sr('eat',     'causes', 'full')
        sr('hungry',  'causes', 'want food')
        sr('wall',    'is',     'bad')
        sr('wall',    'causes', 'hurt')
        sr('hurt',    'causes', 'stop')
        sr('danger',  'is',     'bad')
        sr('danger',  'causes', 'afraid')
        sr('afraid',  'causes', 'run')
        sr('button',  'causes', 'door open')
        sr('door',    'causes', 'food')
        sr('tired',   'causes', 'want sleep')
        sr('sleep',   'causes', 'awake')
        sr('thirsty', 'causes', 'want water')
        sr('water',   'causes', 'better')
        sr('pain',    'is',     'bad')
        sr('good',    'causes', 'want')
        sr('bad',     'causes', 'stop')

        # Numeric ordering — axioms, strength 2.0 (should not be revised)
        pairs = [('one','two'), ('two','three'), ('three','four'), ('four','five')]
        for small, big in pairs:
            sr(small, 'less_than',    big,   2.0)
            sr(big,   'greater_than', small, 2.0)
            sr(small, 'before',       big,   2.0)

    def learn_from_event(self, info: dict, fear: float, reward: float):
        """
        Auto-extract (S, V, O) relations from a World6 event.
        Called during training — builds grounded fact base from experience.
        Uses small strength (0.05) so repeated events gradually strengthen facts
        without overwhelming teacher-provided relations (strength 1.0+).
        """
        W = 0.05   # base experiential strength per event
        hunger  = info.get('hunger',  0.0)
        fatigue = info.get('fatigue', 0.0)
        thirst  = info.get('thirst',  0.0)

        if info.get('is_food') and reward > 0:
            self.store_relation('food',   'is',     'good',       W * (1 + reward), 'experience')
            self.store_relation('eat',    'causes', 'full',       W,                'experience')
            if hunger > 0.5:
                self.store_relation('hungry', 'causes', 'want food', W, 'experience')

        if info.get('wall_hit'):
            self.store_relation('wall',  'is',     'bad',  W, 'experience')
            self.store_relation('wall',  'causes', 'hurt', W, 'experience')

        if info.get('is_danger') and fear > 0.2:
            self.store_relation('danger', 'is',     'bad',    W, 'experience')
            self.store_relation('danger', 'causes', 'afraid', W, 'experience')

        if info.get('is_drinking') and reward > 0:
            self.store_relation('water',   'is',     'good',       W, 'experience')
            self.store_relation('drink',   'causes', 'better',     W, 'experience')
            if thirst > 0.5:
                self.store_relation('thirsty', 'causes', 'want water', W, 'experience')

        if info.get('is_button'):
            self.store_relation('button', 'causes', 'door open', W, 'experience')

        if info.get('is_door') and info.get('door_open'):
            self.store_relation('door open', 'causes', 'food', W, 'experience')

        if hunger > 0.7 and not info.get('is_food'):
            self.store_relation('hungry', 'causes', 'want food', W * 0.5, 'experience')

        if fatigue > 0.7:
            self.store_relation('tired', 'causes', 'want sleep', W * 0.5, 'experience')

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

        Belief Revision: when a new factual relation conflicts with an existing one,
        the beliefs compete on strength. Each time the new belief is heard, the
        conflicting old belief is decayed by 0.3. When a belief's strength drops to
        0 or below it is permanently removed (dead belief). The new belief is stored
        with reduced initial strength (0.5) since the brain is uncertain — but if
        heard again, it strengthens and wins via repeated reinforcement.

        State/mutable relations (feels/wants/needs/name) are exempt — they can
        co-exist because they reflect changing states, not fixed facts.

        Example:
            mem.store_relation('you', 'name', 'pranay')
        """
        subject = subject.lower().strip()
        relation = relation.lower().strip()
        obj = obj.lower().strip()
        if not subject or not relation or not obj:
            return

        # Reinforce existing matching triple — strength can only increase here.
        for rel in self._relations:
            if (rel['subject'] == subject and rel['relation'] == relation
                    and rel['object'] == obj):
                rel['strength'] = min(2.0, rel['strength'] + float(strength) * 0.5)
                rel['source'] = source
                # Also decay all competing beliefs for this factual relation.
                if relation in self._FACTUAL_RELATIONS:
                    for rival in self._relations:
                        if (rival['subject'] == subject
                                and rival['relation'] == relation
                                and rival['object'] != obj):
                            rival['strength'] = max(0.0, rival['strength'] - 0.3)
                    # Prune dead beliefs (strength == 0).
                    self._relations = [r for r in self._relations
                                       if r['strength'] > 0.0]
                return

        # New triple — check for contradiction with existing factual beliefs.
        if relation in self._FACTUAL_RELATIONS:
            conflict_found = False
            for rival in self._relations:
                if (rival['subject'] == subject and rival['relation'] == relation
                        and rival['object'] != obj):
                    # Flag contradiction for surface display.
                    self._contradictions.append({
                        'subject':  subject,
                        'relation': relation,
                        'stored':   rival['object'],
                        'new':      obj,
                    })
                    # Decay the existing rival belief — new evidence challenges it.
                    rival['strength'] = max(0.0, rival['strength'] - 0.3)
                    conflict_found = True

            # Prune any rivals that just hit zero.
            self._relations = [r for r in self._relations if r['strength'] > 0.0]

            # Incoming belief enters uncertain — heard once is weak evidence.
            if conflict_found:
                strength = min(float(strength), 0.5)

        self._relations.append({
            'subject':  subject,
            'relation': relation,
            'object':   obj,
            'strength': float(strength),
            'source':   source,
        })

    def recall_relations(self, subject: str, relation=None) -> list:
        """
        Return stored relation triples for a subject, optionally filtered by relation.
        Dead beliefs (strength == 0) are excluded — they were revised away.
        Results are sorted strongest-first so callers naturally get the dominant belief.
        """
        subject = subject.lower().strip()
        if relation is not None:
            relation = relation.lower().strip()
        matches = []
        for rel in self._relations:
            if rel['strength'] <= 0.0:          # dead belief — skip
                continue
            if rel['subject'] != subject:
                continue
            if relation is not None and rel['relation'] != relation:
                continue
            matches.append(dict(rel))
        # Strongest belief first — callers get the dominant fact, not the oldest one.
        matches.sort(key=lambda r: r['strength'], reverse=True)
        return matches

    def negated_words_for(self, zone: str) -> set:
        """
        Return words the brain knows are false/forbidden in this zone context.
        Fed to GRU as `forbidden` during generation — constrained decoding, not grammar rules.
        Brain stays silent on false things rather than explicitly denying them.
        """
        zone = zone.lower().strip()
        negated: set[str] = set()
        for rel in self._relations:
            if rel['strength'] <= 0.0 or rel['relation'] not in self._NEGATION_RELATIONS:
                continue
            if rel['subject'] == zone:
                for w in rel['object'].split():
                    negated.add(w)
            if zone in rel['object'].split():
                for w in rel['subject'].split():
                    negated.add(w)
        return negated

    def transitive_lookup(self, subject: str, max_hops: int = 3) -> list[str]:
        """
        N-hop transitive relation walk from subject.
        Returns all reachable objects within max_hops, ordered by proximity.

        Also follows universal quantifier patterns:
          If subject 'X' has relation 'is food' and 'every_food is edible' exists,
          then 'edible' is reachable from 'X' via the universal chain.
        """
        visited: set[str] = {subject}
        frontier: list[str] = [subject]
        results: list[str] = []
        for _ in range(max_hops):
            next_frontier: list[str] = []
            for node in frontier:
                # Direct relations from node
                for rel in self.recall_relations(node):
                    if rel['relation'] in self._NEGATION_RELATIONS:
                        continue
                    for w in rel['object'].split():
                        if w not in visited:
                            visited.add(w)
                            results.append(w)
                            next_frontier.append(w)
                # Universal quantifier: if node 'is <class>', follow 'every_<class>'
                for rel in self.recall_relations(node, relation='is'):
                    for cls_word in rel['object'].split():
                        universal_key = f'every_{cls_word}'
                        for urel in self.recall_relations(universal_key):
                            if urel['relation'] in self._NEGATION_RELATIONS:
                                continue
                            for w in urel['object'].split():
                                if w not in visited:
                                    visited.add(w)
                                    results.append(w)
                                    next_frontier.append(w)
            frontier = next_frontier
            if not frontier:
                break
        return results

    def is_negated(self, subject: str, obj: str) -> bool:
        """True if brain explicitly knows 'subject is_not/cannot/does_not obj'."""
        subject = subject.lower().strip()
        obj     = obj.lower().strip()
        for rel in self._relations:
            if (rel['subject'] == subject
                    and rel['relation'] in self._NEGATION_RELATIONS
                    and rel['strength'] > 0.0):
                # Check full match or substring — "food is_not dangerous" blocks "food dangerous"
                if obj in rel['object'] or rel['object'] in obj:
                    return True
        return False

    def _latest_relation_object(self, subject: str, relation: str):
        """Return the object of the strongest surviving belief for (subject, relation)."""
        matches = self.recall_relations(subject, relation)
        # recall_relations is already sorted strongest-first, so [0] is the winner.
        return matches[0]['object'] if matches else None

    def recent_contradictions(self, n: int = 3) -> list[dict]:
        return self._contradictions[-n:]

    def contradiction_count(self) -> int:
        return len(self._contradictions)

    def pop_contradiction(self):
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

    def find_relation_answer(self, tokens: list):
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

        # "X is not Y" / "X are not Y" — negation
        if len(words) >= 4 and words[1] in ('is', 'are') and words[2] == 'not':
            subject = words[0]
            negated = ' '.join(words[3:])
            self.store_relation(subject, 'is_not', negated)
            return

        # "X cannot Y" / "X can not Y"
        if len(words) >= 3 and words[1] == 'cannot':
            self.store_relation(words[0], 'cannot', ' '.join(words[2:]))
            return
        if len(words) >= 4 and words[1] == 'can' and words[2] == 'not':
            self.store_relation(words[0], 'cannot', ' '.join(words[3:]))
            return

        # "X does not Y" / "X do not Y"
        if len(words) >= 4 and words[1] in ('does', 'do') and words[2] == 'not':
            self.store_relation(words[0], 'does_not', ' '.join(words[3:]))
            return

        # "all X are Y" / "every X is Y" → universal quantifier
        # Stored as every_<class> is <predicate> so transitive_lookup can follow it.
        if words[0] in ('all', 'every') and len(words) >= 4 and words[2] in ('is', 'are'):
            subject_class = words[1]
            predicate = ' '.join(words[3:])
            self.store_relation(f'every_{subject_class}', 'is', predicate)
            return

        # "no X is Y" → universal negation
        if words[0] == 'no' and len(words) >= 4 and words[2] in ('is', 'are'):
            subject_class = words[1]
            negated = ' '.join(words[3:])
            self.store_relation(f'every_{subject_class}', 'is_not', negated)
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
        obj = cls()   # __init__ seeds axioms + runs infer_all_axioms
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            obj._facts.update(d.get('facts', {}))
            # Merge saved relations on top of axioms (don't overwrite)
            saved = d.get('relations', [])
            for rel in saved:
                obj.store_relation(rel['subject'], rel['relation'], rel['object'],
                                   strength=rel.get('strength', 1.0),
                                   source=rel.get('source', 'taught'))
            obj._contradictions = d.get('contradictions', [])
            obj.infer_all_axioms()
        return obj

    # Relations that carry through a chain: A causes B, B causes C → A causes C
    _CHAIN_RELATIONS = {'causes', 'is', 'means', 'helps', 'hurts', 'less_than',
                        'greater_than', 'before'}

    def infer_chain(self, subject: str, store: bool = True) -> list[dict]:
        """
        Two-hop inference: find conclusions reachable via two relation steps.

        Example:
          'food is good' + 'good causes want' → infer 'food causes want'
          'danger causes afraid' + 'afraid causes run' → infer 'danger causes run'

        Inferred conclusions stored at strength 0.3 (source='inferred') so they
        can be recalled and strengthened by repeated experience — but weak enough
        that taught/axiom relations always dominate.

        Returns list of {'subject', 'via', 'relation', 'object', 'strength'}.
        """
        subject = subject.lower().strip()
        inferred: list[dict] = []
        seen_triples: set[tuple] = set()

        hop1_rels = [r for r in self.recall_relations(subject)
                     if r['relation'] in self._CHAIN_RELATIONS]

        for r1 in hop1_rels:
            mid = r1['object']
            mid_words = mid.split()

            for mid_word in mid_words:
                hop2_rels = [r for r in self.recall_relations(mid_word)
                             if r['relation'] in self._CHAIN_RELATIONS]

                for r2 in hop2_rels:
                    # Skip trivial self-loops
                    if r2['object'] == subject or r2['object'] == mid_word:
                        continue

                    # Resolve relation: same relation type when both hops match,
                    # otherwise 'causes' as generic causal chain
                    if r1['relation'] == r2['relation']:
                        chain_rel = r1['relation']
                    elif 'causes' in (r1['relation'], r2['relation']):
                        chain_rel = 'causes'
                    else:
                        chain_rel = 'causes'

                    triple = (subject, chain_rel, r2['object'])
                    if triple in seen_triples:
                        continue
                    seen_triples.add(triple)

                    # Don't re-infer what's already directly stored at strength ≥ 1.0
                    existing = [r for r in self.recall_relations(subject, chain_rel)
                                if r['object'] == r2['object']]
                    if existing and existing[0]['strength'] >= 1.0:
                        continue

                    chain_strength = min(r1['strength'], r2['strength']) * 0.4
                    result = {
                        'subject':  subject,
                        'via':      mid_word,
                        'relation': chain_rel,
                        'object':   r2['object'],
                        'strength': chain_strength,
                    }
                    inferred.append(result)

                    if store:
                        self.store_relation(subject, chain_rel, r2['object'],
                                            strength=chain_strength, source='inferred')

        inferred.sort(key=lambda x: x['strength'], reverse=True)
        return inferred

    def infer_all_axioms(self):
        """
        Run infer_chain over all subjects that have axiom/taught relations.
        Called once after loading to pre-populate inferred conclusions.
        """
        subjects: set[str] = set()
        for rel in self._relations:
            if rel['source'] in ('axiom', 'taught') and rel['strength'] >= 1.0:
                subjects.add(rel['subject'])
        for s in subjects:
            self.infer_chain(s, store=True)

    def find_chain_answer(self, tokens: list) -> str | None:
        """
        Answer queries that require 2-hop inference.
        'what does food lead to' / 'what causes run' / 'why want food'
        """
        if not tokens:
            return None

        # "what does X lead to" / "what will X cause"
        if len(tokens) >= 4 and tokens[0] == 'what' and tokens[1] in ('does', 'will'):
            subj = tokens[2]
            chains = self.infer_chain(subj, store=False)
            if chains:
                top = chains[0]
                return f"{subj} {top['relation']} {top['object']} via {top['via']}"

        # "what leads to X" — reverse lookup via inferred
        if len(tokens) >= 3 and tokens[0] == 'what' and tokens[1] in ('leads', 'causes'):
            target = ' '.join(tokens[2:])
            candidates = []
            for rel in self._relations:
                if (rel['source'] == 'inferred'
                        and rel['object'] == target
                        and rel['strength'] > 0.0):
                    candidates.append(rel)
            if candidates:
                best = max(candidates, key=lambda r: r['strength'])
                return f"{best['subject']} causes {target}"

        return None

    def summary(self) -> str:
        lines = [f"  Semantic memory: {len(self._facts)} entities, {len(self._relations)} relations"]
        for entity, facts in list(self._facts.items())[:8]:
            lines.append(f"    {entity:12s} → {facts}")
        return '\n'.join(lines)
