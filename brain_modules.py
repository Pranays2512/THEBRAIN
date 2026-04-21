import numpy as np
from collections import deque

class WorkingMemory:
    def __init__(self, n_neurons: int = 5000):
        self._activation = np.zeros(n_neurons, dtype=np.float32)
        self._n = n_neurons
        self._seeking = False      # Phase 2: WM Lock — brain is actively looking for resolution
        self._seek_turns = 0       # turns since seeking began (auto-release after 3 turns)

    def enter_seeking(self):
        """Lock WM into uncertainty-resolution mode. Brain will bias toward 'what/search/know'."""
        self._seeking = True
        self._seek_turns = 0

    def resolve_seeking(self):
        """Release WM lock when reward arrives — resolution felt, not parsed."""
        self._seeking = False
        self._seek_turns = 0

    def tick_seeking(self):
        """Age the seeking state each turn. Auto-release if no resolution arrives."""
        if self._seeking:
            self._seek_turns += 1
            if self._seek_turns >= 3:
                self._seeking = False
                self._seek_turns = 0

    def update(self, bmu: int, heard_words: list, word_to_bmu: dict):
        # Decay everything
        self._activation *= 0.85
        # Spike the heard BMU with Gaussian spread
        indices = np.arange(self._n)
        dist_sq = (indices - bmu) ** 2
        self._activation += np.exp(-dist_sq / (2 * 3.0**2)).astype(np.float32)
        # Also spike BMUs for heard words
        for w in heard_words:
            if w in word_to_bmu:
                wbmu = word_to_bmu[w]
                self._activation[wbmu] = min(1.0, self._activation[wbmu] + 0.4)
        # While seeking: additionally spike uncertainty-resolution BMUs
        if self._seeking:
            for w in ['what', 'search', 'know']:
                if w in word_to_bmu:
                    wb = word_to_bmu[w]
                    self._activation[wb] = min(1.0, self._activation[wb] + 0.3)

    def top_active_words(self, bmu_to_word: dict, n: int = 3) -> list:
        top_idxs = np.argsort(self._activation)[::-1][:20]
        words = []
        for idx in top_idxs:
            if idx in bmu_to_word and len(words) < n:
                words.append(str(bmu_to_word[idx]))
        return words

    def activation_strength(self) -> float:
        return float(np.mean(np.sort(self._activation)[::-1][:10]))


class EpisodicMemory:
    def __init__(self, maxlen: int = 50):
        self._episodes = deque(maxlen=maxlen)
        self._turn = 0

    def record(self, you_words: list, brain_words: list, zone: str, drives: dict, reward: float):
        self._turn += 1
        self._episodes.append({
            'turn':        self._turn,
            'you':         you_words,
            'brain':       brain_words,
            'zone':        zone,
            'drives':      drives.copy(),
            'reward':      reward,
        })

    def recall_similar(self, zone: str, n: int = 3) -> list:
        return [e for e in reversed(self._episodes) if e['zone'] == zone][:n]

    def last_topic(self) -> str | None:
        return self._episodes[-1]['zone'] if self._episodes else None

    def topic_changed(self) -> bool:
        if len(self._episodes) < 3:
            return False
        zones = [e['zone'] for e in list(self._episodes)[-3:]]
        return zones[-1] != zones[-3]

    def count(self) -> int:
        return len(self._episodes)


class SelfModel:
    def __init__(self):
        self._state = {
            'hunger':      0.3,
            'fatigue':     0.2,
            'fear':        0.0,
            'comfort':     0.5,
            'curiosity':   0.5,
            'uncertainty': 0.0,  # epistemic tension — feeling of missing information
        }

    def spike_uncertainty(self, magnitude: float = 0.4):
        """Prediction error or novel BMU → felt gap in knowledge."""
        s = self._state
        s['uncertainty'] = min(1.0, s['uncertainty'] + magnitude)

    def resolve_uncertainty(self, reward: float):
        """Reward received → tension drops. Resolution is felt, not parsed."""
        s = self._state
        if reward > 0.1:
            s['uncertainty'] = max(0.0, s['uncertainty'] - reward * 0.6)

    def update(self, zone: str, reward: float, wm_strength: float, episode_count: int):
        s = self._state
        # Zone affects drives
        if zone == 'food':   s['hunger']  = max(0.0, s['hunger']  - 0.12)
        if zone == 'danger': s['fear']    = min(1.0, s['fear']    + 0.15)
        if zone == 'rest':   s['fatigue'] = max(0.0, s['fatigue'] - 0.10)
        if zone == 'social': s['comfort'] = min(1.0, s['comfort'] + 0.07)
        if zone == 'water':  s['hunger']  = max(0.0, s['hunger']  - 0.05)
        # Passive drift
        s['hunger']     = min(1.0, s['hunger']     + 0.01)
        s['fatigue']    = min(1.0, s['fatigue']    + 0.005)
        s['fear']       = max(0.0, s['fear']       - 0.03)
        s['curiosity']  = max(0.2, 1.0 - wm_strength * 2)
        # Uncertainty decays per turn — faster than spike so it resolves in ~5 turns
        s['uncertainty'] = max(0.0, s['uncertainty'] - 0.08)
        self.resolve_uncertainty(reward)

    def dominant_state(self) -> str:
        s = self._state
        if s['uncertainty'] > 0.6: return 'uncertain'
        if s['fear'] > 0.5:        return 'afraid'
        if s['hunger'] > 0.6:      return 'hungry'
        if s['fatigue'] > 0.6:     return 'tired'
        if s['comfort'] > 0.7:     return 'content'
        if s['curiosity'] > 0.6:   return 'curious'
        return 'calm'

    def state_words(self) -> list:
        mapping = {
            'uncertain': ['search', 'know', 'what'],
            'afraid':    ['afraid', 'careful'],
            'hungry':    ['hungry', 'need'],
            'tired':     ['tired', 'sleep'],
            'content':   ['good', 'calm'],
            'curious':   ['want', 'know'],
            'calm':      ['calm'],
        }
        return mapping.get(self.dominant_state(), [])

    def as_vector(self) -> np.ndarray:
        s = self._state
        return np.array([s['hunger'], s['fatigue'], s['fear'],
                         s['comfort'], s['curiosity'], s['uncertainty']],
                        dtype=np.float32)


class CuriosityModule:
    def __init__(self, n_neurons: int = 5000):
        self._bmu_counts  = np.zeros(n_neurons, dtype=np.int32)
        self._zone_counts = {}

    def update(self, bmu: int, zone: str):
        self._bmu_counts[bmu] += 1
        self._zone_counts[zone] = self._zone_counts.get(zone, 0) + 1

    def novelty_score(self, bmu: int) -> float:
        return 1.0 / (1.0 + self._bmu_counts[bmu])

    def least_visited_zone(self, zones: list) -> str:
        if not zones: return "unknown"
        return min(zones, key=lambda z: self._zone_counts.get(z, 0))

    def curiosity_words(self, zone_anchors: dict, word_to_bmu: dict) -> list:
        target_zone = self.least_visited_zone(list(zone_anchors.keys()))
        anchors = [w for w in zone_anchors.get(target_zone, []) if w in word_to_bmu]
        return anchors[:2]


class NeuromodSystem:
    def __init__(self):
        self._ach      = 0.3   # acetylcholine — novelty → learning rate
        self._ne       = 0.2   # norepinephrine — surprise → temperature
        self._serotonin = 0.5  # reward rate EMA → patience

    def update(self, novelty: float, surprise: float, reward: float):
        alpha = 0.1
        self._ach       = (1-alpha)*self._ach       + alpha*novelty
        self._ne        = (1-alpha)*self._ne        + alpha*surprise
        self._serotonin = (1-alpha)*self._serotonin + alpha*(reward + 0.5)
        # Clamp
        self._ach       = max(0.0, min(1.0, self._ach))
        self._ne        = max(0.0, min(1.0, self._ne))
        self._serotonin = max(0.0, min(1.0, self._serotonin))

    def temperature(self) -> float:
        # High NE (surprise) → higher temperature → more exploratory
        return 0.65 + self._ne * 0.5

    def learning_boost(self) -> float:
        # High ACh (novelty) → faster learning
        return 0.5 + self._ach * 1.5

    def patience(self) -> float:
        return self._serotonin


class TemporalContext:
    def __init__(self):
        self._turn          = 0
        self._turns_since   = {}  # zone → turns since last visit
        self._reward_ema    = 0.5

    def update(self, zone: str, reward: float):
        self._turn += 1
        # Age all zones
        for z in self._turns_since:
            self._turns_since[z] += 1
        # Reset current zone
        self._turns_since[zone] = 0
        # Update reward EMA
        self._reward_ema = 0.9*self._reward_ema + 0.1*(reward + 0.5)

    def turns_since(self, zone: str) -> int:
        return self._turns_since.get(zone, 999)

    def zone_overdue(self, zone: str, threshold: int = 6) -> bool:
        return self.turns_since(zone) > threshold

    def session_phase(self) -> str:
        if self._turn < 5:  return 'early'
        if self._turn < 20: return 'mid'
        return 'late'

    def reward_trend(self) -> float:
        return self._reward_ema


class PrefrontalCortex:
    def __init__(self):
        self._active_goal = None
        
    def check_goal(self, selfmodel: SelfModel):
        if selfmodel._state['hunger'] > 0.7:
            self._active_goal = 'food'
        elif selfmodel._state['fatigue'] > 0.7:
            self._active_goal = 'rest'
        elif selfmodel._state['fear'] > 0.6:
            self._active_goal = 'safe'
        else:
            self._active_goal = None

    def get_goal_seeds(self) -> list[str]:
        if self._active_goal == 'food':
            return ['want', 'food', 'eat', 'need']
        if self._active_goal == 'rest':
            return ['sleep', 'tired', 'calm', 'need']
        if self._active_goal == 'safe':
            return ['run', 'stop', 'afraid', 'careful']
        return []
        
    def imagine(self, brain, episodic: 'EpisodicMemory',
                curiosity: 'CuriosityModule' = None,
                zone_anchors: dict = None) -> str:
        """Generate an imagination thought driven by curiosity toward the least-visited zone."""
        import random
        if curiosity is not None and zone_anchors is not None:
            # Pick from the 2 least-visited zones randomly to avoid spam
            zones = list(zone_anchors.keys())
            zones_sorted = sorted(zones, key=lambda z: curiosity._zone_counts.get(z, 0))
            target_zone = random.choice(zones_sorted[:2])  # one of the 2 least visited
            words_in_vocab = [w for w in zone_anchors.get(target_zone, [])
                              if hasattr(brain, 'word_to_bmu') and w in brain.word_to_bmu]
            if not words_in_vocab:
                # fallback to any zone
                target_zone = random.choice(zones)
                words_in_vocab = [w for w in zone_anchors.get(target_zone, [])
                                  if hasattr(brain, 'word_to_bmu') and w in brain.word_to_bmu]
            context = words_in_vocab[:2]
        else:
            # Fallback: use last topic from episodic
            topic = episodic.last_topic() if episodic else None
            if topic == 'food':       context = ['eat', 'food']
            elif topic == 'danger':   context = ['run', 'afraid']
            elif topic == 'rest':     context = ['sleep', 'calm']
            else:                     context = ['i', 'feel']

        if not context:
            context = ['i', 'feel']

        words = brain.word_tp_generate(context, max_len=4, temperature=0.9)
        return " ".join(words) if words else ""


class LogicModule:
    """
    Lightweight inference engine over grounded drives + semantic relations.

    Not a rule engine — draws conclusions by chaining what brain has actually
    learned (stored in SemanticMemory) with what it currently feels (SelfModel).

    Three inference types:
      1. Drive → action: hunger > 0.6 + semantic('hungry') → ['eat','food']
      2. Percept → consequence: zone='danger' + semantic('danger') → ['run']
      3. 2-hop chain: heard_word → relation object → that object's relations
    """

    def __init__(self):
        self._last_conclusion: list = []
        self._confidence: float = 0.0

    def infer(self, selfmodel, semantic, zone: str,
              heard_words: list, vocabulary: set) -> list:
        """
        Returns conclusion word list to prepend to generation seeds.
        Empty when no strong inference fires.
        """
        conclusions: list = []
        s = selfmodel._state

        # ── Type 1: Drive-anchored ────────────────────────────────────────
        # High internal drive → look up what that drive word relates to.
        # Uses SemanticMemory so conclusions grow as brain learns more relations.
        _drive_anchors = [
            ('hunger',  'hungry',  0.6),
            ('fear',    'danger',  0.5),
            ('fatigue', 'tired',   0.6),
            ('uncertainty', 'search', 0.6),
        ]
        for drive_key, anchor_word, threshold in _drive_anchors:
            if s.get(drive_key, 0.0) > threshold:
                for rel in semantic.recall_relations(anchor_word)[:2]:
                    for w in rel['object'].split()[:2]:
                        if w in vocabulary and w not in conclusions:
                            conclusions.append(w)

        # ── Type 2: Zone-percept → consequence ───────────────────────────
        # What the brain currently perceives (zone) → semantic lookup → action.
        # Falls back to grounded defaults only if no relation stored yet.
        _zone_defaults = {
            'food':   ('food',   ['eat', 'food']),
            'danger': ('danger', ['run', 'careful']),
            'rest':   ('tired',  ['sleep', 'calm']),
            'water':  ('water',  ['drink', 'calm']),
            'pain':   ('hurt',   ['stop', 'hurt']),
        }
        if zone in _zone_defaults:
            anchor, defaults = _zone_defaults[zone]
            rels = semantic.recall_relations(anchor)
            if rels:
                for w in rels[0]['object'].split()[:2]:
                    if w in vocabulary and w not in conclusions:
                        conclusions.append(w)
            else:
                for w in defaults:
                    if w in vocabulary and w not in conclusions:
                        conclusions.append(w)

        # ── Type 3: 2-hop relation chain from heard words ─────────────────
        # heard_word → relation → object (1-hop) → that object's relations (2-hop)
        # This is where "xylophone → music → good" would fire if both stored.
        for hw in heard_words[:3]:
            hop1 = semantic.recall_relations(hw)
            for rel1 in hop1[:1]:
                obj1_words = rel1['object'].split()
                for w in obj1_words[:1]:
                    if w in vocabulary and w not in conclusions:
                        conclusions.append(w)
                # 2-hop
                if obj1_words:
                    hop2 = semantic.recall_relations(obj1_words[0])
                    for rel2 in hop2[:1]:
                        for w in rel2['object'].split()[:1]:
                            if w in vocabulary and w not in conclusions:
                                conclusions.append(w)

        self._last_conclusion = conclusions[:5]
        self._confidence = min(1.0, len(conclusions) * 0.2)
        return self._last_conclusion

    def explain(self) -> str:
        return (f"logic: {self._last_conclusion}  conf={self._confidence:.2f}"
                if self._last_conclusion else "logic: no conclusion")
