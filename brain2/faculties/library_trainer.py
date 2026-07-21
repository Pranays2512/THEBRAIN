#!/usr/bin/env python3
"""
library_trainer.py — A resumable orchestration layer over read_book.py.

Given a queue of textbooks in data/curriculum_queue.txt, this script feeds them
sentence-by-sentence to the BookTrainer (Ollama + TemplateCache). 
It writes the extracted logical facts to data/brain_curriculum.txt so that 
they are permanently stored and can be ingested by auto_train.py.

Features:
- Pausable & Resumable: Press Ctrl+C at any time. It saves state (book index,
  sentence index, and learned templates) to data/library_state.json.
- Persistent Extraction: Appends extracted triples directly to data/brain_curriculum.txt.
"""

import os
import json
import sys
import time
from collections import defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
QUEUE_FILE = os.path.join(DATA_DIR, "curriculum_queue.txt")
STATE_FILE = os.path.join(DATA_DIR, "library_state.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "brain_curriculum.txt")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from faculties.read_book import BookTrainer, sentences
from adapters.llm_adapter import OllamaClient


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"queue_idx": 0, "sent_idx": 0, "templates": []}


def save_state(q_idx, s_idx, cache):
    # Serialize the template cache
    # cache.templates is dict: pattern_tuple -> (relation, subj_idx, obj_idx)
    serialized_templates = []
    for pat, (r, sp, op) in cache.templates.items():
        serialized_templates.append({"pat": pat, "r": r, "sp": sp, "op": op})
    
    with open(STATE_FILE, "w") as f:
        json.dump({
            "queue_idx": q_idx,
            "sent_idx": s_idx,
            "templates": serialized_templates
        }, f, indent=2)


def append_fact(s, r, o):
    """Write extracted facts to the permanent brain_curriculum.txt"""
    with open(OUTPUT_FILE, "a") as f:
        if r.lower() == "isa":
            f.write(f"ISA: {s} | {o}\n")
        else:
            f.write(f"FACT: {s} | {r} | {o}\n")


def main():
    if not os.path.exists(QUEUE_FILE):
        print(f"[!] Please create {QUEUE_FILE} with one book file path per line.")
        return

    with open(QUEUE_FILE, "r") as f:
        queue = [line.strip() for line in f if line.strip()]

    if not queue:
        print("[!] Curriculum queue is empty.")
        return

    state = load_state()
    q_idx = state["queue_idx"]
    s_idx = state["sent_idx"]

    if q_idx >= len(queue):
        print("[-] All books in the queue have been processed!")
        return

    print("=" * 70)
    print("  Brain2 Library Trainer (Resumable Extraction)")
    print("=" * 70)

    # Initialize the BookTrainer with Qwen
    model = "qwen3-coder:480b-cloud" # User requested qwen cloud coder
    print(f"  Loading LLM Teacher: Ollama ({model})")
    try:
        client = OllamaClient(model)
        # test connection
        client.complete("hello")
    except Exception as e:
        print(f"  [!] Failed to connect to Ollama: {e}")
        print("  Please ensure Ollama is running (`ollama serve`) and the model is pulled.")
        return

    trainer = BookTrainer(client=client, with_events=False)

    # Restore the template cache from state
    for t in state.get("templates", []):
        trainer.cache.templates[tuple(t["pat"])] = (t["r"], t["sp"], t["op"])

    print(f"  Restored {len(trainer.cache.templates)} grammatical templates from state.")

    try:
        while q_idx < len(queue):
            book_path = queue[q_idx]
            
            # Resolve relative paths to the data directory if they don't exist
            if not os.path.exists(book_path):
                alt_path = os.path.join(DATA_DIR, book_path)
                if os.path.exists(alt_path):
                    book_path = alt_path
                else:
                    print(f"  [!] Could not find book: {book_path}")
                    q_idx += 1
                    s_idx = 0
                    continue

            print(f"\n[-] Reading Book {q_idx + 1}/{len(queue)}: {os.path.basename(book_path)}")
            
            with open(book_path, "r", errors="ignore") as f:
                text = f.read()
            sents = sentences(text)
            
            print(f"    Total sentences: {len(sents)}. Resuming from sentence {s_idx}.")

            while s_idx < len(sents):
                sent = sents[s_idx]
                
                # Use BookTrainer's internal extraction (LLM + Cache)
                triples, used_llm = trainer._triples(sent)
                
                for t in triples:
                    if len(t) == 3:
                        s, r, o = t
                        append_fact(s, r, o)
                        print(f"      {'[LLM]' if used_llm else '[CACHE]'} {s} | {r} | {o}")
                
                s_idx += 1

                # Save state every 10 sentences to avoid heavy I/O
                if s_idx % 10 == 0:
                    save_state(q_idx, s_idx, trainer.cache)
                    
            print(f"[-] Finished reading {os.path.basename(book_path)}")
            q_idx += 1
            s_idx = 0
            save_state(q_idx, s_idx, trainer.cache)

        print("\n[+] Curriculum queue finished!")

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user. Saving state...")
        save_state(q_idx, s_idx, trainer.cache)
        print(f"    State saved. Next run will resume at Book {q_idx + 1}, Sentence {s_idx}.")
        sys.exit(0)

if __name__ == "__main__":
    main()
