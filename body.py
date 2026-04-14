"""
BODY — Sensory/Motor Abstraction Layer
======================================

This is the brain's interface to the physical world.

RIGHT NOW: a simulated 2D environment with touch, vision, sound, and speech.
FUTURE: swap SimBody for RealBody — the brain code never changes.

The brain only ever calls:
    body.sense()    → dict of sensory readings
    body.act(cmd)   → execute an action, returns reward
    body.speak(txt) → produce speech output
    body.hear()     → get current audio frame as MFCC

What "senses" the brain has (all simulated for now):
    touch     — is something touching me? (0.0 / 1.0)
    pain      — is it harmful? (0.0 to 1.0)
    warmth    — temperature feeling (0.0=cold, 0.5=neutral, 1.0=hot)
    vision    — simplified: list of nearby object types as word labels
    hunger    — internal drive (0.0=full, 1.0=starving)
    tiredness — internal drive (0.0=rested, 1.0=exhausted)
    sound     — what sound is happening (as MFCC vector)
    proprioception — what position/posture am I in

SENSOR → MFCC BRIDGE
---------------------
Every sensory reading gets converted to an MFCC vector so the brain can
"hear" its own body the same way it hears words.
This is how the brain builds grounded meaning:
    touching something warm → specific MFCC pattern
    being hungry → specific MFCC pattern
    seeing a face → specific MFCC pattern
These patterns become vocabulary the brain naturally associates with
states (warmth → calm, hunger → urgency, pain → negative valence).

ACTIONS
-------
The brain can choose from:
    'look'      — pay attention to what's visible
    'touch'     — reach out and touch nearest object
    'move'      — move toward something interesting
    'rest'      — reduce tiredness, lower arousal
    'eat'       — consume food if present
    'speak'     — produce speech (handled separately via brain vocal output)

FUTURE HARDWARE SWAP
--------------------
Replace SimBody with:
    class RealBody(BodyInterface):
        def sense(self):
            return {
                'touch':   read_touch_sensor(),
                'vision':  camera_to_labels(camera.capture()),
                'sound':   microphone.get_mfcc_frame(),
                'hunger':  read_hunger_model(),   # time-since-eating
                ...
            }
        def act(self, cmd): ...
        def speak(self, text): speaker.say(text)

The brain never needs to change.
"""

import numpy as np
from abc import ABC, abstractmethod


# ── Sensor MFCC encoding ────────────────────────────────────────────
# Each sense maps to a position in the MFCC vector.
# This gives the brain a consistent "feeling" for each sensation.
#
# v[0]  = energy (always 3.0 for voiced body sensation)
# v[1]  = valence       (+= pleasure, -= pain)
# v[2]  = arousal       (+= exciting/urgent, -= calm/rest)
# v[3]  = social        (+= presence of others)
# v[4]  = concreteness  (+= physical/tangible)
# v[5]  = certainty     (+= familiar, -= novel/uncertain)
# v[6]  = intensity     (magnitude of sensation)
# v[7]  = temporality   (+= happening now, -= memory/past)
# v[8]  = agency        (+= self-caused, -= externally caused)
# v[9]  = familiarity   (how often seen before, 0-1)
# v[10] = embodiment    (+= felt in body)
# v[11] = questioning   (0 for body senses)
# v[12] = spare

N_MFCC = 13


def _sense_to_mfcc(valence=0.0, arousal=0.0, social=0.0,
                   concreteness=0.8, certainty=0.5, intensity=0.5,
                   temporal=0.8, agency=0.5, familiarity=0.5,
                   embodiment=0.9) -> np.ndarray:
    """Convert a sensory event into a 13-dim MFCC vector the brain can process."""
    v = np.zeros(N_MFCC, dtype=np.float32)
    v[0]  = 3.0          # voiced / present
    v[1]  = float(valence)
    v[2]  = float(arousal)
    v[3]  = float(social)
    v[4]  = float(concreteness)
    v[5]  = float(certainty)
    v[6]  = float(intensity)
    v[7]  = float(temporal)
    v[8]  = float(agency)
    v[9]  = float(familiarity)
    v[10] = float(embodiment)
    return v


# ── Pre-built sensory MFCC patterns ────────────────────────────────
# These are the "vocabulary" of body experience.
# The brain will learn to associate these patterns with internal states.

SENSE_MFCC = {
    # Tactile
    'touch_soft':    _sense_to_mfcc(valence= 0.4, arousal= 0.1, intensity=0.3, embodiment=1.0),
    'touch_hard':    _sense_to_mfcc(valence= 0.0, arousal= 0.3, intensity=0.6, embodiment=1.0),
    'pain':          _sense_to_mfcc(valence=-0.9, arousal= 0.9, intensity=0.9, embodiment=1.0),
    'warmth':        _sense_to_mfcc(valence= 0.5, arousal=-0.2, intensity=0.4, embodiment=1.0),
    'cold':          _sense_to_mfcc(valence=-0.2, arousal= 0.4, intensity=0.5, embodiment=1.0),
    # Visual (simplified)
    'see_face':      _sense_to_mfcc(valence= 0.3, arousal= 0.4, social=0.9,  certainty=0.6),
    'see_object':    _sense_to_mfcc(valence= 0.1, arousal= 0.2, social=0.0,  concreteness=0.9),
    'see_nothing':   _sense_to_mfcc(valence= 0.0, arousal=-0.1, certainty=0.2),
    'see_food':      _sense_to_mfcc(valence= 0.6, arousal= 0.5, intensity=0.7, concreteness=0.9),
    # Internal drives
    'hungry':        _sense_to_mfcc(valence=-0.4, arousal= 0.6, intensity=0.7, embodiment=1.0),
    'full':          _sense_to_mfcc(valence= 0.5, arousal=-0.3, intensity=0.2, embodiment=0.8),
    'tired':         _sense_to_mfcc(valence=-0.2, arousal=-0.7, intensity=0.4, embodiment=0.9),
    'rested':        _sense_to_mfcc(valence= 0.3, arousal= 0.1, intensity=0.1, embodiment=0.7),
    # Movement / action feedback
    'moving':        _sense_to_mfcc(valence= 0.2, arousal= 0.5, agency=0.9,   temporal=0.9),
    'still':         _sense_to_mfcc(valence= 0.0, arousal=-0.2, agency=0.5,   temporal=0.5),
    'collision':     _sense_to_mfcc(valence=-0.5, arousal= 0.8, intensity=0.8, embodiment=1.0),
    # Social / interaction
    'spoken_to':     _sense_to_mfcc(valence= 0.3, arousal= 0.5, social=1.0,   certainty=0.4),
    'silence':       _sense_to_mfcc(valence= 0.0, arousal=-0.3, social=0.0,   certainty=0.7),
}


# ── Interface contract ──────────────────────────────────────────────

class BodyInterface(ABC):
    """
    Every body — simulated or real — must implement this interface.
    The brain only calls these methods.
    """

    @abstractmethod
    def sense(self) -> dict:
        """
        Return current sensory state.

        Returns dict with keys:
            touch      float [0,1]   — contact intensity
            pain       float [0,1]   — harm signal
            warmth     float [0,1]   — temperature (0=cold, 0.5=neutral, 1=hot)
            hunger     float [0,1]   — drive (0=full, 1=starving)
            tiredness  float [0,1]   — drive (0=rested, 1=exhausted)
            vision     list[str]     — visible object labels
            mfcc       np.ndarray    — current audio frame (13-dim), zeros if silent
        """

    @abstractmethod
    def act(self, action: str) -> float:
        """
        Execute an action. Returns reward.

        Valid actions: 'look', 'touch', 'move', 'rest', 'eat', 'speak', 'idle'
        """

    @abstractmethod
    def speak(self, text: str):
        """Produce speech output."""

    def sense_as_mfcc(self) -> np.ndarray:
        """
        Convert current sensory state into an MFCC vector.
        This is how body experience becomes something the brain can process —
        the same pathway as hearing a word, but the 'word' is a bodily sensation.
        """
        senses = self.sense()
        v = np.zeros(N_MFCC, dtype=np.float32)
        v[0] = 3.0  # always voiced when body is active

        # Pain dominates — high negative valence and arousal
        if senses.get('pain', 0) > 0.1:
            pattern = SENSE_MFCC['pain'].copy()
            pattern[6] = float(senses['pain'])
            return pattern

        # Hunger creates urgency
        hunger = senses.get('hunger', 0.0)
        if hunger > 0.5:
            pattern = SENSE_MFCC['hungry'].copy()
            pattern[6] = float(hunger)
            return pattern

        # Tiredness lowers arousal
        tiredness = senses.get('tiredness', 0.0)
        if tiredness > 0.6:
            pattern = SENSE_MFCC['tired'].copy()
            pattern[6] = float(tiredness)
            return pattern

        # Touch
        touch = senses.get('touch', 0.0)
        warmth = senses.get('warmth', 0.5)
        if touch > 0.3:
            if warmth > 0.65:
                return SENSE_MFCC['warmth'].copy()
            elif warmth < 0.3:
                return SENSE_MFCC['cold'].copy()
            else:
                return SENSE_MFCC['touch_soft'].copy()

        # Vision — what's visible
        vision = senses.get('vision', [])
        if 'face' in vision:
            return SENSE_MFCC['see_face'].copy()
        if 'food' in vision:
            return SENSE_MFCC['see_food'].copy()
        if vision:
            return SENSE_MFCC['see_object'].copy()

        # Audio from environment
        mfcc = senses.get('mfcc', None)
        if mfcc is not None and float(mfcc[0]) > 0.5:
            return np.array(mfcc, dtype=np.float32)

        # Default: neutral presence
        v[5] = 0.4   # some certainty
        v[7] = 0.5   # present
        return v


# ═══════════════════════════════════════════════════════════════════════
# SIMULATED BODY
# ═══════════════════════════════════════════════════════════════════════

class SimBody(BodyInterface):
    """
    A simple 2D room simulation.

    The brain lives in a small world with:
    - Objects it can interact with (food, a soft thing, a hard thing)
    - A simulated "other agent" (you, when you type)
    - Internal drives: hunger and tiredness
    - Touch/temperature feedback from actions

    This gives grounded meaning to words:
        "hungry" → the brain actually feels hunger → negative valence + urgency
        "warm"   → touching the warm object → pleasant sensation
        "food"   → seeing food while hungry → high reward
        "tired"  → internal tiredness > 0.7 → low arousal, negative drive
        "touch"  → reaching out → tactile feedback

    The world updates every step — drives increase slowly,
    objects persist, actions have physical consequences.
    """

    OBJECTS = ['food', 'soft_thing', 'warm_thing', 'hard_thing', 'face']

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

        # ── Internal drives ───────────────────────────────────
        self.hunger    = 0.2   # starts not very hungry
        self.tiredness = 0.1   # starts rested
        self.warmth    = 0.5   # neutral temperature

        # ── World state ───────────────────────────────────────
        # Objects visible in current moment
        self._visible: list[str] = []
        self._visible_timer = 0      # how many steps an object stays visible

        # Touch state
        self._touching: str  = None
        self._touch_intensity = 0.0
        self._pain = 0.0

        # Other agent (you) — present when you type
        self._other_present = False
        self._other_timer   = 0

        # Movement feedback
        self._just_moved = False
        self._just_collided = False

        # Last action taken
        self._last_action = 'idle'

        # Step counter
        self.t = 0

        # Audio input from environment (set externally when you speak)
        self._env_mfcc: np.ndarray = np.zeros(N_MFCC, dtype=np.float32)

    def set_audio(self, mfcc: np.ndarray):
        """Called by converse.py when you type — sets the audio input."""
        self._env_mfcc = np.array(mfcc, dtype=np.float32)

    def set_other_present(self, present: bool, duration: int = 30):
        """Mark that another agent (you) is present for `duration` steps."""
        self._other_present = present
        self._other_timer = duration if present else 0

    def _update_world(self):
        """Advance the simulation one step."""
        self.t += 1

        # Drives increase slowly over time
        self.hunger    = min(1.0, self.hunger    + 0.0003)
        self.tiredness = min(1.0, self.tiredness + 0.0002)

        # Object visibility fades
        if self._visible_timer > 0:
            self._visible_timer -= 1
        else:
            self._visible = []

        # Touch fades
        if self._touching:
            self._touch_intensity *= 0.85
            if self._touch_intensity < 0.05:
                self._touching = None
                self._touch_intensity = 0.0

        # Pain fades
        self._pain *= 0.7

        # Other agent fades
        if self._other_timer > 0:
            self._other_timer -= 1
            self._other_present = self._other_timer > 0

        # Movement/collision resets
        self._just_moved = False
        self._just_collided = False

        # Audio fades
        self._env_mfcc *= 0.6  # echo decay

    def sense(self) -> dict:
        self._update_world()

        vision = list(self._visible)
        if self._other_present:
            vision.append('face')

        return {
            'touch':     self._touch_intensity,
            'pain':      self._pain,
            'warmth':    self.warmth,
            'hunger':    self.hunger,
            'tiredness': self.tiredness,
            'vision':    vision,
            'mfcc':      self._env_mfcc.copy(),
        }

    def act(self, action: str) -> float:
        """Execute action, return reward signal."""
        self._last_action = action
        reward = 0.0

        if action == 'look':
            # Randomly bring an object into view
            obj = self.OBJECTS[int(self._rng.integers(len(self.OBJECTS)))]
            self._visible = [obj]
            self._visible_timer = 25
            reward = 0.05  # small reward for attending

        elif action == 'touch':
            if self._visible:
                obj = self._visible[0]
                self._touching = obj
                self._touch_intensity = 0.8
                if obj == 'warm_thing':
                    self.warmth = min(1.0, self.warmth + 0.2)
                    reward = 0.3   # warmth is pleasant
                elif obj == 'soft_thing':
                    reward = 0.2   # soft is pleasant
                elif obj == 'hard_thing':
                    self._pain = 0.3
                    reward = -0.2  # hard hurts a bit
                elif obj == 'food':
                    # touching food reveals it's edible
                    reward = 0.1
                elif obj == 'face':
                    reward = 0.4   # social touch — very rewarding

        elif action == 'eat':
            if 'food' in self._visible and self.hunger > 0.2:
                eaten = min(self.hunger, 0.6)
                reward = eaten * 1.5   # more hungry = more rewarding
                self.hunger = max(0.0, self.hunger - 0.6)
                self._visible = []     # food is consumed

        elif action == 'rest':
            reduction = 0.15
            self.tiredness = max(0.0, self.tiredness - reduction)
            reward = reduction * 0.5   # mild reward for reducing drive

        elif action == 'move':
            self._just_moved = True
            # Moving brings new things into view
            if self._rng.random() < 0.4:
                obj = self.OBJECTS[int(self._rng.integers(len(self.OBJECTS)))]
                self._visible = [obj]
                self._visible_timer = 20
                reward = 0.08
            elif self._rng.random() < 0.15:
                # Sometimes bump into things
                self._just_collided = True
                self._pain = 0.2
                reward = -0.1

        elif action == 'idle':
            reward = 0.0

        return reward

    def speak(self, text: str):
        """For simulation: just print. For real body: call TTS."""
        print(f"  [body speaks]: {text}")

    def reward_for_state(self) -> float:
        """
        Intrinsic reward from body state — given every step automatically.
        High hunger/pain/tiredness = negative. Satisfied = positive.
        This is the grounding: words that reduce these drives become good.
        """
        pain_penalty     = -self._pain * 0.8
        hunger_penalty   = -max(0.0, self.hunger - 0.3) * 0.3
        tired_penalty    = -max(0.0, self.tiredness - 0.5) * 0.2
        social_bonus     = 0.15 if self._other_present else 0.0
        return float(pain_penalty + hunger_penalty + tired_penalty + social_bonus)


# ═══════════════════════════════════════════════════════════════════════
# REAL BODY STUB — swap this in when you have real hardware
# ═══════════════════════════════════════════════════════════════════════

class RealBody(BodyInterface):
    """
    Stub for future real hardware.

    When you have:
        - Microphone  → replaces self._env_mfcc with real audio
        - Speaker     → replaces print() in speak() with TTS
        - Camera      → replaces random visibility with real object detection
        - Touch sensor→ replaces simulated touch with physical readings

    Just implement these 3 methods and the brain works unchanged.
    """

    def sense(self) -> dict:
        raise NotImplementedError(
            "Connect real sensors here:\n"
            "  mfcc     = compute_mfcc(microphone.read())\n"
            "  vision   = object_detector(camera.capture())\n"
            "  touch    = touch_sensor.read()\n"
            "  hunger   = time_since_last_food_signal()\n"
        )

    def act(self, action: str) -> float:
        raise NotImplementedError(
            "Connect real actuators here:\n"
            "  'move'  → motor_controller.move()\n"
            "  'touch' → arm_controller.extend()\n"
            "  'eat'   → (reward from human or food sensor)\n"
        )

    def speak(self, text: str):
        raise NotImplementedError(
            "Connect TTS here:\n"
            "  tts_engine.say(text)\n"
            "  speaker.play()\n"
        )
