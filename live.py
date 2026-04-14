"""
LIVE — Embodied Brain Loop
==========================

The brain in its (temporary) body.

What's different from converse.py:
  - Brain senses its OWN body every step, not just when you type
  - Body drives (hunger, tiredness, pain) create real internal states
  - Words you type are now grounded: "hungry" while brain feels hunger
    → that word gets bound to the actual urgency/negative-valence state
  - Brain picks actions (look, touch, eat, rest, move) based on internal state
  - Conversation still works — you type, brain hears you
  - Learning saves on exit

Usage:
  python teach_english.py    # train first if no brain_trained.pkl
  python live.py             # run the embodied brain

Commands:
  /state   — show brain's full internal + body state
  /words   — top reward-associated words
  /body    — show body drives (hunger, tiredness, etc)
  /dream   — run one manual dream cycle and show report
  /help    — all known words
  /quit    — exit and save

Feedback (after brain responds):
  good / yes / great  → reward +1.0  (brain learns this worked)
  no / wrong / bad    → reward -0.5  (brain learns this didn't work)
"""

import sys, time, os
import numpy as np
from collections import Counter, deque

sys.dont_write_bytecode = True
from brain import Brain
from body import SimBody, N_MFCC
from vocab_extra import VOCABULARY, SILENCE, N_MFCC   # extended ~300-word vocab
from m75_semantic import SemanticMemory
from m76_broca import BrocaGrammar

# ── Load brain ────────────────────────────────────────────────────────
BRAIN_FILE = 'brain_trained.pkl'

if not os.path.exists(BRAIN_FILE):
    print(f"\n  No trained brain found ({BRAIN_FILE}).")
    print("  Run:  python teach_english.py  first.")
    sys.exit(1)

print(f"\nLoading trained brain from {BRAIN_FILE}...")
brain = Brain.load(BRAIN_FILE)
brain.vocal._rng = np.random.default_rng(int(time.time()) % (2**31))

# ── Create body ───────────────────────────────────────────────────────
body = SimBody(seed=int(time.time()) % (2**31))
print("  Body created (simulated).")

# ── Semantic memory (M75) — fact store ───────────────────────────────
semantic = SemanticMemory()
print("  Semantic memory ready.")

# ── RNG for response generation ───────────────────────────────────────
rng = np.random.default_rng(int(time.time()) % (2**31))

# ── Feedback tracking ─────────────────────────────────────────────────
# Keeps the last turn's BMUs and state so feedback commands can reward them
_last_turn_bmus: list = []
_last_turn_state: str = ''
_total_turns: int = 0
_DREAM_EVERY: int = 8   # dream after every N conversation turns

# Words that count as direct feedback from you
_FEEDBACK_POSITIVE = {'good', 'yes', 'great', 'nice', 'right', 'correct',
                      'perfect', 'well', 'okay', 'fine'}
_FEEDBACK_NEGATIVE = {'no', 'wrong', 'bad', 'stop', 'not', 'never'}

# Recent responses — avoid repeating the same thing back-to-back
_recent_responses: list = []
_MAX_RECENT = 4

# ── Turn memory (Step 2) ──────────────────────────────────────────────
# Last 5 conversation turns stored for context.
# Gives the brain memory of the thread — what was said, what it replied.
_turn_history: deque = deque(maxlen=5)


def _is_too_repetitive(response: str) -> bool:
    return response in _recent_responses


def _record_response(response: str):
    _recent_responses.append(response)
    if len(_recent_responses) > _MAX_RECENT:
        _recent_responses.pop(0)


# ── Helpers ───────────────────────────────────────────────────────────

_PUNCT = str.maketrans('', '', '.,?!;:\'"()-/\\')

def say(word, noise_std=0.10):
    mean_vec, n_frames = VOCABULARY[word]
    return [(mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
            for _ in range(n_frames)]


# ── FIX #3: Auditory self-feedback ───────────────────────────────────
# The brain hears its own response — feeds its own words back through M71.
# This is what closes the amnesia loop: the brain's output becomes its input.
# Biologically: hearing your own voice activates the same auditory cortex
# as hearing someone else speak.

def hear_own_response(response_words: list, reward: float = 0.0):
    """
    Feed the brain's own spoken words back through its ear.
    Call this after every response is generated.
    Fixes the amnesia bug: brain now remembers what it said.
    """
    bmus = []
    for word in response_words:
        if word not in VOCABULARY:
            continue
        # Lower noise — the brain "hears itself" more clearly than external speech
        for frame in say(word, noise_std=0.06):
            brain.hear(frame)
            out = brain.step(reward=reward)
            bmus.append(out['m71_phoneme_bmu'])
        brain.hear(SILENCE)
        brain.step()
    # Record the self-heard sequence so dreams can replay it too
    if bmus:
        brain.record_turn(bmus, reward, brain.selfmodel._last_label)
    return bmus


# ── FIX #2: Internal monologue ────────────────────────────────────────
# Before speaking, the brain silently generates a candidate response,
# evaluates it against its current state, and only speaks if it "feels right."
# Biologically: prefrontal cortex suppresses motor output (Broca's area)
# while simulating the sentence internally first.

def _monologue_evaluate(candidate_words: list) -> float:
    """
    Score a candidate response by running it silently through the brain
    and measuring how well the resulting state matches the current intent.
    Returns a score 0-1 (higher = better fit).
    """
    if not candidate_words:
        return 0.0

    sm_before = brain.selfmodel._state_vec.copy()

    # Simulate hearing the candidate internally (very low noise = "imagined")
    for word in candidate_words:
        if word not in VOCABULARY:
            continue
        for frame in say(word, noise_std=0.02):
            brain.hear(frame)
            brain.step(reward=0.0)

    sm_after = brain.selfmodel._state_vec.copy()

    # Score: how much does the imagined response reduce frustration
    # and increase clarity? Good responses settle the brain.
    frustration_delta = float(sm_before[6] - sm_after[6])   # positive = frustration fell
    clarity_delta     = float(sm_after[1]  - sm_before[1])  # positive = clarity rose

    score = 0.5 + frustration_delta * 0.3 + clarity_delta * 0.3
    return float(np.clip(score, 0.0, 1.0))


def internal_monologue(candidates: list[str]) -> str:
    """
    Silently evaluate each candidate response, return the best one.
    This is the prefrontal filter — picks the response that feels most coherent.
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    best, best_score = candidates[0], -1.0
    for candidate in candidates:
        words = candidate.split()
        score = _monologue_evaluate(words)
        if score > best_score:
            best_score = score
            best = candidate

    return best


# ── FIX #5 partial: Semantic traversal via word→word inference ────────
# Uses M68-style chaining but over word BMU space instead of zone space.
# "who is pranay" → semantic.recall('pranay') → describe it.
# "what is food"  → semantic.recall('food') → "food is edible"

def semantic_query(words: list) -> str:
    """
    If the question is about a known entity, return a semantic answer.
    Checks M75 first, then falls back to introspection.
    Also handles "what is my name" → looks up stored name for 'you'.
    """
    ws = set(words)

    # Special case: "what is my name" vs "what is your name"
    if 'name' in ws and 'my' in ws:
        user_facts = semantic.recall('you')
        user_name = user_facts.get('name', '')
        if user_name:
            return f'your name is {user_name}'
    if 'name' in ws and 'your' in ws:
        brain_facts = semantic.recall('brain')
        brain_name = brain_facts.get('name', '')
        if brain_name:
            return f'i am {brain_name}'
        return 'i am brain'

    # Find the content word (noun/name) being asked about
    question_words = {'what', 'who', 'where', 'when', 'why', 'how',
                      'is', 'are', 'do', 'does', 'did', 'you', 'i', 'a', 'the',
                      'my', 'your', 'name'}
    content = [w for w in words if w not in question_words]

    for word in content:
        if semantic.knows(word):
            return semantic.describe(word)

    return ""


# Build BMU → word map
print("  Mapping vocabulary to SOM...", end='', flush=True)
word_to_bmu: dict[str, int] = {}
bmu_to_word: dict[int, str] = {}
for word in VOCABULARY:
    counts: Counter = Counter()
    for _ in range(6):
        for frame in say(word, noise_std=0.04):
            brain.hear(frame)
            out = brain.step()
            counts[out['m71_phoneme_bmu']] += 1
    top_bmu = counts.most_common(1)[0][0]
    word_to_bmu[word] = top_bmu
    if top_bmu not in bmu_to_word:
        bmu_to_word[top_bmu] = word
print(f" {len(set(word_to_bmu.values()))} unique BMUs.")

# ── Broca's grammar engine (Step 3) ──────────────────────────────────
broca = BrocaGrammar(
    bmu_to_word_fn   = lambda b: bmu_to_word.get(b),
    word_to_bmu_dict = word_to_bmu,
)
print(f"  {broca.summary()}")


def bmu_to_nearest_word(bmu: int, exclude: set = None) -> str:
    if bmu in bmu_to_word:
        w = bmu_to_word[bmu]
        if exclude and w in exclude:
            return None
        return w
    best, best_dist = None, float('inf')
    for w, b in word_to_bmu.items():
        if exclude and w in exclude:
            continue
        d = abs(b - bmu)
        if d < best_dist:
            best_dist, best = d, w
    return best if best_dist < 15 else None


# ── Action selection ──────────────────────────────────────────────────
# Maps M59 state → preferred body action.
# This is the bridge between cognitive state and physical behavior.

_STATE_TO_ACTION = {
    'hungry':    'eat',       # not an M59 state, but drive-triggered
    'satisfied': 'rest',
    'curious':   'look',
    'hunting':   'move',
    'focused':   'touch',
    'alert':     'look',
    'stuck':     'rest',
    'confused':  'look',
    'drifting':  'idle',
}

def choose_action(sm_label: str, body_senses: dict) -> str:
    """Pick body action based on cognitive state and body drives."""
    # Drives override cognitive preference
    if body_senses['hunger'] > 0.7 and 'food' in body_senses['vision']:
        return 'eat'
    if body_senses['tiredness'] > 0.8:
        return 'rest'
    if body_senses['pain'] > 0.3:
        return 'rest'
    return _STATE_TO_ACTION.get(sm_label, 'idle')


# ── Response generation ───────────────────────────────────────────────

_POSITIVE = {'yes', 'good', 'hello', 'hi', 'love', 'happy', 'thank',
             'okay', 'please', 'ready', 'done', 'together', 'know',
             'speak', 'near', 'more', 'always', 'calm',
             'great', 'nice', 'fine', 'really'}
_NEGATIVE = {'no', 'sad', 'angry', 'hurt', 'never', 'cold', 'scared',
             'confused', 'dark', 'less', 'not'}

# ── Introspection engine ─────────────────────────────────────────────
# Maps what the brain actually feels to words it can say.
# This is the core of grounded language: heard words trigger a check
# of real internal state, not a TP walk.

def _introspect_feeling() -> str:
    """Check M59 state + body drives → return honest feeling word(s)."""
    sv     = brain.selfmodel._state_vec.copy()
    label  = brain.selfmodel._last_label
    senses = body.sense()
    urgency, clarity, drive, novelty, stability, confidence, frustration, engagement = sv

    parts = []

    # Body drives first — most concrete
    if senses['hunger'] > 0.55:
        parts.append('hungry')
    if senses['tiredness'] > 0.60:
        parts.append('tired')
    if senses['pain'] > 0.2:
        parts.append('hurt')

    # Cognitive/emotional state
    if label == 'confused' or (clarity < 0.4 and novelty > 0.5):
        parts.append('confused')
    elif label == 'satisfied':
        parts.append('good')
    elif label == 'curious':
        parts.append('curious')
    elif label == 'focused':
        parts.append('calm')
    elif label == 'stuck':
        parts.append('stuck')
    elif label == 'hunting':
        parts.append('want')
    elif label == 'drifting':
        parts.append('calm')

    # High frustration overrides
    if frustration > 0.5 and 'stuck' not in parts:
        parts.append('not good')

    if not parts:
        parts.append('okay')

    return ' '.join(parts[:2])   # max 2 feelings at once


def _introspect_want() -> str:
    """What does the brain want right now?"""
    sv     = brain.selfmodel._state_vec.copy()
    senses = body.sense()
    urgency, clarity, drive, novelty, stability, confidence, frustration, engagement = sv

    if senses['hunger'] > 0.5:
        return 'food'
    if senses['tiredness'] > 0.65:
        return 'rest'
    if novelty > 0.5:
        return 'know more'
    if drive > 0.5 and urgency > 0.4:
        return 'more'
    if engagement > 0.5:
        return 'listen'
    return 'okay'


def _introspect_who_am_i() -> str:
    """What can the brain say about itself?"""
    n_trajectories = len(brain._word_trajectories) if hasattr(brain, '_word_trajectories') else 0
    if n_trajectories > 20:
        return 'i know you i learn'
    elif n_trajectories > 5:
        return 'i think i learn'
    else:
        return 'i think'


# ── Intent detection ─────────────────────────────────────────────────
# Detects what kind of statement/question was heard.
# This gives the brain a primitive understanding of intent —
# not the full meaning, but enough to respond appropriately.

def _detect_intent(words: list) -> str:
    """
    Classify the intent of heard words.
    Returns one of: 'greeting', 'feeling_question', 'want_question',
                    'identity_question', 'affirmation', 'negation',
                    'observation', 'teaching', 'unknown'
    """
    ws = set(words)

    # Greetings
    if ws & {'hello', 'hi', 'hey'}:
        return 'greeting'

    # Farewells
    if ws & {'bye', 'goodbye'}:
        return 'farewell'

    # Affirmation / negation
    if ws <= {'good', 'yes', 'great', 'nice', 'fine', 'okay', 'right'}:
        return 'affirmation'
    if ws <= {'no', 'wrong', 'bad', 'not'}:
        return 'negation'

    # Feeling question: "how do you feel", "are you feeling", "what do you feel"
    if ('feel' in ws or 'feeling' in ws) and ws & {'how', 'what', 'are', 'do', 'you'}:
        return 'feeling_question'

    # Want question: "what do you want", "are you hungry"
    if ('want' in ws or 'hungry' in ws or 'tired' in ws) and ws & {'what', 'are', 'do', 'you'}:
        return 'want_question'

    # Identity question: "who are you", "what are you"
    if ws & {'who', 'what'} and 'are' in ws and 'you' in ws and 'feel' not in ws and 'think' not in ws:
        return 'identity_question'

    # General question — check BEFORE teaching so "what is X?" isn't mislabeled.
    # Question word anywhere in first 3 positions → it's a question.
    _Q = {'what', 'who', 'how', 'why', 'where', 'when'}
    if any(w in _Q for w in words[:3]):
        return 'question'

    # Teaching: "X is/means/are Y" or "i am X" — declarative statements
    # Covers: "pranay is creator", "food means eat", "i am hungry"
    if words and len(words) >= 2 and words[1] in ('is', 'means', 'are', 'am'):
        return 'teaching'
    if 'means' in ws:
        return 'teaching'

    return 'observation'

def _tp_generate(heard_bmus: list, sm_vec: np.ndarray, temperature: float = 1.5) -> str:
    """Walk the TP matrix to generate a novel word sequence."""
    N = brain.phoneme_seq._P.shape[0]
    if not heard_bmus:
        return ""
    tp_resp = sum(brain.phoneme_seq._P[b] for b in heard_bmus if 0 <= b < N)
    total = float(tp_resp.sum())
    if total < 1e-9:
        return ""
    tp_resp /= total
    sv_norm = float(np.linalg.norm(sm_vec))
    if sv_norm > 1e-9 and brain.binding._W_state.max() > 1e-9:
        state_sims = brain.binding._W_state @ (sm_vec / sv_norm)
        state_sims = np.maximum(state_sims, 0.0)
        ss = state_sims.sum()
        state_dist = state_sims / ss if ss > 1e-9 else np.ones(N) / N
    else:
        state_dist = np.ones(N) / N
    blended = 0.60 * tp_resp + 0.40 * state_dist
    blended /= blended.sum()
    log_p = np.log(np.maximum(blended, 1e-9)) / temperature
    log_p -= log_p.max()
    probs = np.exp(log_p)
    probs /= probs.sum()
    seed_bmu = int(brain.vocal._rng.choice(N, p=probs))
    utterance = brain.vocal._generate_sequence(
        seed_phoneme=seed_bmu,
        state_vec=sm_vec,
        binding=brain.binding,
        phoneme_seq=brain.phoneme_seq,
        ne_level=max(float(brain.neuromod.ne_level), 0.30),
    )
    words, prev = [], None
    for b in utterance:
        w = bmu_to_nearest_word(b)
        if w and w != prev:
            words.append(w)
            prev = w
    return ' '.join(words) if words else ""


def generate_response(heard_words: list, heard_bmus: list, raw_tokens: list = None) -> str:
    """Generate a response grounded in intent detection + real internal state.

    Priority:
      1. Semantic memory — only for hard facts (name recall), not intent routing
      2. TP generation — what the brain actually learned from experience
      3. Retry at higher temperature
    All hardcoded intent routing removed. Responses come from training.
    """
    if not heard_words and not heard_bmus:
        return ""

    sm_vec = brain.selfmodel._state_vec.copy()
    tokens = raw_tokens or heard_words

    # ── 1. Semantic memory — only hard facts, not intent routing ─────
    # "what is my name" / "who is X" → look up stored facts.
    # This is NOT hardcoded logic — it's recalling what was explicitly told.
    ws = set(tokens)
    if 'name' in ws:
        if 'my' in ws:
            user_name = semantic.recall('you').get('name', '')
            if user_name and not _is_too_repetitive(f'your name is {user_name}'):
                return f'your name is {user_name}'
        if 'your' in ws:
            brain_name = semantic.recall('brain').get('name', 'brain')
            if not _is_too_repetitive(f'i am {brain_name}'):
                return f'i am {brain_name}'

    # ── 2. Pure TP generation — learned from dialogue training ───────
    # This is the only response mechanism. Quality comes entirely from
    # what the brain experienced during train_dialogue.py.
    for attempt in range(4):
        temp = 1.2 + attempt * 0.4
        candidate = _tp_generate(heard_bmus, sm_vec, temperature=temp)
        if candidate and not _is_too_repetitive(candidate):
            _record_response(candidate)
            return candidate

    # Final fallback — pick something based on last heard word
    if heard_words:
        last = heard_words[-1]
        if last in _POSITIVE:
            return 'yes good'
        if last in {'what', 'who', 'why', 'how', 'where'}:
            return 'i think'
        return 'okay'
    return ""


# ── Main loop ─────────────────────────────────────────────────────────

# Warmup
print("  Waking up...", end='', flush=True)
_calm_words = ['hello', 'good', 'calm', 'okay', 'ready', 'yes', 'know', 'think'] * 60
for _w in _calm_words:
    for _f in say(_w, noise_std=0.12):
        brain.hear(_f)
        brain.step(reward=0.7)
    brain.hear(SILENCE)
    brain.step()
    if brain.selfmodel._last_label not in ('confused', 'drifting'):
        break

# Direct-nudge if still confused — EMA integrator starts cold from saved state.
# Equivalent to: "the brain knows these words already, stop being confused."
if brain.selfmodel._last_label in ('confused', 'drifting'):
    sv = brain.selfmodel._state_vec
    sv[1] = 0.70   # clarity up
    sv[3] = 0.20   # novelty down
    sv[5] = 0.65   # confidence up
    sv[6] = 0.20   # frustration down
    brain.selfmodel._last_label = brain.selfmodel._compute_label(sv)
print(f" state: {brain.selfmodel._last_label}")

print("""
============================================================
  EMBODIED BRAIN — Body + Language + Experience
  The brain has hunger, tiredness, touch, and vision.
  Words are grounded to what it physically feels.
  It dreams every 8 turns to consolidate what it learned.
  Say 'good'/'yes' or 'no'/'wrong' to give feedback.
  /state  /body  /words  /dream  /help  /quit
============================================================
""")

# Body sense cycle runs every N conversation steps
BODY_STEPS_PER_TURN = 15   # brain feels body for 15 steps between your turns

step_count = 0

while True:
    # ── Body cycle — brain senses and acts on its own ─────────────────
    # This runs BETWEEN your inputs, giving the brain an inner life.
    for _ in range(BODY_STEPS_PER_TURN):
        body_mfcc = body.sense_as_mfcc()
        brain.hear(body_mfcc)
        body_reward = body.reward_for_state()
        out = brain.step(reward=body_reward)

        # Brain picks an action based on its state
        sm_label = out['sm_state_label']
        senses   = body.sense()
        action   = choose_action(sm_label, senses)
        act_reward = body.act(action)

        # Action reward feeds back into brain next step
        if abs(act_reward) > 0.05:
            brain.hear(body.sense_as_mfcc())
            brain.step(reward=act_reward)

    step_count += BODY_STEPS_PER_TURN

    # ── User input ─────────────────────────────────────────────────────
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Goodbye.")
        brain.save(BRAIN_FILE)
        print(f"  Learning saved → {BRAIN_FILE}")
        break

    if not user_input:
        continue

    # ── Commands ──────────────────────────────────────────────────────
    if user_input.startswith('/'):
        cmd = user_input.lower().strip()
        if cmd in ('/quit', '/exit', '/q'):
            print("  Goodbye.")
            brain.save(BRAIN_FILE)
            print(f"  Learning saved → {BRAIN_FILE}")
            break
        elif cmd == '/state':
            sv = brain.selfmodel._state_vec
            label = brain.selfmodel._last_label
            names = ['urgency','clarity','drive','novelty',
                     'stability','confidence','frustration','engagement']
            print(f"\n  Cognitive state: [{label}]")
            for n, v in zip(names, sv):
                bar = '█' * int(v * 15)
                print(f"    {n:12s} {v:.3f}  {bar}")
            print(f"  GWS arousal:  {brain.gws._arousal_ema:.3f}")
            print(f"  NE level:     {brain.neuromod.ne_level:.3f}")
            print(f"  Semantic mem: {len(semantic._facts)} entities known")
            n_traj = len(brain._word_trajectories) if hasattr(brain, '_word_trajectories') else 0
            print(f"  Experience:   {n_traj} turns stored")
            print()
        elif cmd == '/body':
            s = body.sense()
            print(f"\n  Body state:")
            print(f"    hunger     {s['hunger']:.2f}  {'█' * int(s['hunger']*20)}")
            print(f"    tiredness  {s['tiredness']:.2f}  {'█' * int(s['tiredness']*20)}")
            print(f"    pain       {s['pain']:.2f}  {'█' * int(s['pain']*20)}")
            print(f"    warmth     {s['warmth']:.2f}  {'█' * int(s['warmth']*20)}")
            print(f"    touch      {s['touch']:.2f}  {'█' * int(s['touch']*20)}")
            print(f"    visible:   {s['vision'] or 'nothing'}")
            print(f"    last action: {body._last_action}")
            print()
        elif cmd == '/words':
            print("\n  Top reward associations:")
            rew = [(w, float(brain.binding._W_reward[word_to_bmu[w]]))
                   for w in VOCABULARY if w in word_to_bmu]
            rew.sort(key=lambda x: -x[1])
            for w, r in rew[:15]:
                bar = '█' * int(r * 30)
                print(f"    {w:12s} {r:.4f}  {bar}")
            print()
        elif cmd == '/dream':
            if not brain._word_trajectories:
                print("\n  No experience recorded yet — talk to the brain first.\n")
            else:
                print("\n  Dreaming...", end='', flush=True)
                report = brain.dream_language(n_sequences=10)
                print(f" done.")
                print(f"    Replays:         {report['n_replay']}")
                print(f"    Counterfactuals: {report['n_counterfactual']}")
                print(f"    Novel sequences: {report['n_novel']}")
                print(f"    TP updates:      {report['n_p_updates']}")
                print(f"    Mean turn value: {report['mean_value']:.3f}")
                print(f"    Trajectories:    {report['trajectories_stored']}")
                print()
        elif cmd == '/help':
            words = sorted(VOCABULARY.keys())
            print(f"\n  Known words ({len(words)}):")
            for i in range(0, len(words), 10):
                print('  ' + '  '.join(words[i:i+10]))
            print()
        else:
            print(f"  Unknown command: {cmd}")
        continue

    # ── Process speech input ────────────────────────────────────────────
    tokens = user_input.lower().translate(_PUNCT).split()
    outputs, heard_bmus, heard_words = [], [], []

    # ── Feedback detection — did you rate the LAST response? ─────────────
    input_words = set(tokens)
    is_pure_feedback = input_words and input_words.issubset(
        _FEEDBACK_POSITIVE | _FEEDBACK_NEGATIVE
    )
    feedback_reward = 0.0
    if _last_turn_bmus and is_pure_feedback:
        if input_words & _FEEDBACK_POSITIVE:
            feedback_reward = 1.0
            print("  [+] Got it — reinforcing that response.")
        elif input_words & _FEEDBACK_NEGATIVE:
            feedback_reward = -0.5
            print("  [-] Got it — weakening that response.")
        # Apply feedback to last turn's BMUs
        brain.record_turn(_last_turn_bmus, feedback_reward, _last_turn_state)
        # Immediate mini-dream to consolidate (3 sequences)
        brain.dream_language(n_sequences=3)
        # Also run a reward pulse through brain
        for b in _last_turn_bmus[:5]:
            brain.hear(SILENCE)
            brain.step(reward=feedback_reward * 0.4)
        print()

    # Mark that another agent (you) is present
    body.set_other_present(True, duration=40)

    for tok in tokens:
        if tok not in VOCABULARY:
            continue
        heard_words.append(tok)
        reward = 0.6 if tok in _POSITIVE else (-0.1 if tok in _NEGATIVE else 0.2)
        frames = say(tok, noise_std=0.10)
        body.set_audio(frames[-1] if frames else SILENCE)
        for i, frame in enumerate(frames):
            brain.hear(frame)
            out = brain.step(reward=reward if i == len(frames) - 1 else 0.0)
            outputs.append(out)
            heard_bmus.append(out['m71_phoneme_bmu'])
        brain.hear(SILENCE)
        brain.step()

    if not outputs:
        if not is_pure_feedback:
            print("Brain: (no known words — try /help)\n")
        continue

    # Learn facts from ALL tokens (including unknown words like proper names)
    semantic.learn_from_sentence(tokens)

    # Settling — integrate what was heard
    for _ in range(20):
        brain.hear(body.sense_as_mfcc())
        brain.step(reward=body.reward_for_state())

    # Determine state label
    state_counts = Counter(o['sm_state_label'] for o in outputs)
    state = state_counts.most_common(1)[0][0]

    response = generate_response(heard_words, heard_bmus, raw_tokens=tokens)

    body_state = body.sense()
    hunger_note = " [hungry]" if body_state['hunger'] > 0.6 else ""
    tired_note  = " [tired]"  if body_state['tiredness'] > 0.7 else ""
    body_note   = hunger_note + tired_note

    if response:
        print(f"Brain [{state}{body_note}]: {response}")
        # Fix #3: brain hears its own voice — closes the amnesia loop
        hear_own_response(response.split(), reward=0.1)
    else:
        print(f"Brain [{state}{body_note}]: (silent)")
    print()

    # ── Record this turn for dreaming + turn memory ───────────────────
    _last_turn_bmus  = list(heard_bmus)
    _last_turn_state = state
    turn_body_reward = body.reward_for_state()
    brain.record_turn(heard_bmus, turn_body_reward, state)
    _total_turns += 1

    # Turn memory: store what was said for context in next turn
    _turn_history.append({
        'user_tokens':  tokens,
        'user_words':   heard_words,
        'brain_response': response,
        'brain_words':  response.split() if response else [],
        'state':        state,
    })

    # ── Periodic dream cycle ───────────────────────────────────────────
    # Every _DREAM_EVERY turns, brain consolidates experience during "sleep"
    if _total_turns % _DREAM_EVERY == 0 and brain._word_trajectories:
        print("  [dreaming...]", end='', flush=True)
        report = brain.dream_language(n_sequences=8)
        print(f" {report['n_p_updates']} TP updates, "
              f"mean value {report['mean_value']:.2f}\n")
