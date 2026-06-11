#!/usr/bin/env python3
"""
research_agent.py — Web Researcher & Knowledge Harvester

This script allows the Neuro-Symbolic Hybrid Brain to autonomously read 
Wikipedia articles, parse them into sentences, and deeply embed the 
information into its C++ Episodic Memory graph.
"""

import sys, os, re, json, datetime
import warnings

# Suppress Wikipedia's internal BeautifulSoup warnings
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

try:
    import wikipedia
except ImportError:
    print("Error: The 'wikipedia' python package is not installed.")
    print("Please run: pip install wikipedia")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def split_into_sentences(text):
    """Splits a large block of text into individual sentences."""
    # Simple regex to split by periods, question marks, or exclamation points followed by a space
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10] # Filter out short garbage

def run_research(topic):
    print("=====================================================")
    print(f"  NEURO-SYMBOLIC RESEARCHER - Topic: '{topic}'")
    print("=====================================================")
    
    # 1. Fetching Data
    print(f"[*] Searching Wikipedia for '{topic}'...")
    try:
        # Get the actual page
        page = wikipedia.page(topic, auto_suggest=False)
        title = page.title
        content = page.content
        print(f"[*] Found Article: {title}")
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"[!] Disambiguation Error: '{topic}' may refer to:")
        for option in e.options[:5]:
            print(f"    - {option}")
        sys.exit(1)
    except wikipedia.exceptions.PageError:
        print(f"[!] PageError: '{topic}' does not match any pages. Try another query!")
        sys.exit(1)
    except Exception as e:
        print(f"[!] Fetch Error: {e}")
        sys.exit(1)
        
    sentences = split_into_sentences(content)
    # We will read the first 25 sentences to prevent a massive memory overload for this test
    MAX_SENTENCES = 25
    sentences = sentences[:MAX_SENTENCES]
    
    print(f"[*] Parsed {len(sentences)} sentences to read.")

    # 2. Loading Brain Core
    print("\n[*] Waking up the Brain...")
    b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "executive_brain")
    try:
        b.load_components(
            predictor_path=os.path.join(ckpt_dir, "predictor.bin"),
            language_path=os.path.join(ckpt_dir, "language.bin"),
            som_path=os.path.join(ckpt_dir, "som.bin"),
            episodic_path=os.path.join(ckpt_dir, "episodic.bin"),
            emotion_path=os.path.join(ckpt_dir, "emotion.bin"),
            self_path=os.path.join(ckpt_dir, "self.bin"),
            symbolic_path=os.path.join(ckpt_dir, "symbolic.bin"),
            binding_path=os.path.join(ckpt_dir, "binding.bin"),
            bg_path=os.path.join(ckpt_dir, "bg.bin"),
            procedures_path=os.path.join(ckpt_dir, "procedures.bin"),
            hpred_path=os.path.join(ckpt_dir, "hpred.bin")
        )
    except Exception as e:
        print(f"[!] Failed to load checkpoints: {e}")
        return

    # Load Episodic Text Dictionary
    ep_text_path = os.path.join(ckpt_dir, "episodic_text.json")
    episodic_dict = {}
    if os.path.exists(ep_text_path):
        try:
            with open(ep_text_path, "r") as f:
                episodic_dict = json.load(f)
        except Exception:
            pass

    # 3. Continuous Reading Loop
    print(f"\n[*] Brain is reading article: {title} ...\n")
    
    for idx, sentence in enumerate(sentences):
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence) # Clean up special characters
        
        # Store in C++ Graph
        start_id = b.episodic.episode_count
        b.perceive_text(sentence)
        b.episodic.commit(1.0) # Force commit
        end_id = b.episodic.episode_count
        
        # Store exact text mapping in Python for the entire episode trajectory
        for i in range(start_id, end_id + 2):
            episodic_dict[str(i)] = sentence
            
        sys.stdout.write(f"\r    [Reading] Sentence {idx+1}/{len(sentences)} committed to memory.")
        sys.stdout.flush()

    print("\n\n[*] Article fully digested!")
    
    # 4. Autosave
    print("[*] Saving new Episodic Graph and Topological checkpoints...")
    b.save_components(ckpt_dir)
    
    with open(ep_text_path, "w") as f:
        json.dump(episodic_dict, f)
        
    print("[*] Research complete. You can now chat with the Brain about this topic!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python research_agent.py \"Topic Name\"")
        print("Example: python research_agent.py \"Black Holes\"")
        sys.exit(1)
        
    topic = " ".join(sys.argv[1:])
    run_research(topic)
