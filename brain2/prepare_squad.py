#!/usr/bin/env python3
import json
import os
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return ' '.join(text.split())

def prepare_squad():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    squad_file = os.path.join(data_dir, "train-v2.0.json")
    out_file = os.path.join(data_dir, "squad_qa.json")
    
    if not os.path.exists(squad_file):
        print(f"Error: {squad_file} not found.")
        return
        
    print(f"Loading {squad_file}...")
    with open(squad_file, "r") as f:
        data = json.load(f)
        
    qa_pairs = []
    
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if not qa.get("is_impossible", False) and len(qa["answers"]) > 0:
                    q = clean_text(qa["question"])
                    a = clean_text(qa["answers"][0]["text"])
                    if q and a:
                        qa_pairs.append({"input": q, "target": a})
                        
    print(f"Extracted {len(qa_pairs)} valid Q&A pairs.")
    
    with open(out_file, "w") as f:
        json.dump(qa_pairs, f, indent=2)
        
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    prepare_squad()
