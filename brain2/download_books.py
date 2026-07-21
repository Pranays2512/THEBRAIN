import urllib.request
import json
import os
import re

# Wikipedia API endpoint
API_URL = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&titles="

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
QUEUE_FILE = os.path.join(DATA_DIR, "curriculum_queue.txt")

# The "books" (Wikipedia mega-articles that serve as textbook chapters)
SUBJECTS = {
    "Computer Architecture": "Computer_architecture",
    "Operating Systems": "Operating_system",
    "C++": "C++",
    "Object-Oriented Programming": "Object-oriented_programming",
    "Database Management Systems": "Database",
    "Computer Networks": "Computer_network",
    "Mathematics (Calculus)": "Calculus",
    "Physics (Mechanics)": "Classical_mechanics",
    "Chemistry (Organic)": "Organic_chemistry",
    "Biology (Cellular)": "Cell_biology"
}

def clean_text(text):
    # Remove standard Wikipedia section headers like == See also ==
    text = re.sub(r"==+ .* ==+", "", text)
    # Remove blank lines and artifacts
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()

def download_subjects():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    queue = []
    
    for title, slug in SUBJECTS.items():
        print(f"Downloading textbook substitute for: {title}...")
        try:
            url = API_URL + slug
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                
                pages = data['query']['pages']
                for page_id in pages:
                    extract = pages[page_id].get('extract', '')
                    if extract:
                        clean = clean_text(extract)
                        filename = f"book_{slug.lower()}.txt"
                        filepath = os.path.join(DATA_DIR, filename)
                        
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(clean)
                        
                        queue.append(filename)
                        print(f"  -> Saved {len(clean)} chars to {filename}")
                    else:
                        print(f"  [!] No content found for {slug}")
        except Exception as e:
            print(f"  [!] Failed to download {title}: {e}")

    # Append to the queue
    with open(QUEUE_FILE, "a") as f:
        for filename in queue:
            f.write(filename + "\n")
            
    print(f"\n[+] Successfully added {len(queue)} subjects to data/curriculum_queue.txt!")

if __name__ == "__main__":
    download_subjects()
