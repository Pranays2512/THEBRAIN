#!/usr/bin/env python3
"""
brain3/core/epistemic_web_grounder.py

Autonomous Epistemic Web Grounding & Slang Ingestion Engine for The Brain 3.
Enables real-time internet search when encountering unknown concepts or trending slangs,
and immediately consolidates the discoveries into long-term neurosymbolic memory.
"""

import sys
import os
import json
import re
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, List, Tuple

class EpistemicWebGrounder:
    """Zero-overhead autonomous internet retrieval and memory consolidation."""

    USER_AGENT = "TheBrain3-EpistemicGrounder/1.0"
    TIMEOUT = 4

    @classmethod
    def search_wikipedia(cls, query: str) -> Optional[str]:
        """Fetch concise official summary from Wikipedia REST API."""
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers={"User-Agent": cls.USER_AGENT})
            with urllib.request.urlopen(req, timeout=cls.TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                extract = data.get("extract", "")
                if extract and not extract.startswith("May refer to:"):
                    sentences = re.split(r'(?<=[.!?]) +', extract)
                    return " ".join(sentences[:2])
        except Exception:
            pass
        return None

    @classmethod
    def search_duckduckgo(cls, query: str) -> Optional[str]:
        """Fetch instant definition from DuckDuckGo Instant Answer API."""
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_redirect=1&no_html=1"
            req = urllib.request.Request(url, headers={"User-Agent": cls.USER_AGENT})
            with urllib.request.urlopen(req, timeout=cls.TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                ans = data.get("AbstractText") or data.get("Answer") or data.get("Definition")
                if ans:
                    return ans
        except Exception:
            pass
        return None

    @classmethod
    def search_urban_slang(cls, term: str) -> Optional[str]:
        """Fetch modern trending slang definition from UrbanDictionary API."""
        try:
            url = f"https://api.urbandictionary.com/v0/define?term={urllib.parse.quote(term)}"
            req = urllib.request.Request(url, headers={"User-Agent": cls.USER_AGENT})
            with urllib.request.urlopen(req, timeout=cls.TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                entries = data.get("list", [])
                if entries:
                    defn = entries[0].get("definition", "")
                    clean = re.sub(r'[\[\]]', '', defn).replace('\r\n', ' ').strip()
                    sentences = re.split(r'(?<=[.!?]) +', clean)
                    return " ".join(sentences[:2])
        except Exception:
            pass
        return None

    @classmethod
    def ground_concept(cls, query_term: str) -> Dict[str, Any]:
        """
        Executes multi-tiered web grounding for unknown entities or slang.
        Returns:
            Dict containing:
                - "found": bool
                - "source": "wikipedia" | "duckduckgo" | "urban_slang" | "none"
                - "summary": str
                - "clean_triple": (subj, rel, obj) tuple if extractable
        """
        clean_term = query_term.strip().lower()
        clean_term = re.sub(r'^(what is a|what is an|what is|who is|what does|meaning of|define)\s+', '', clean_term)
        clean_term = re.sub(r'[?!.]', '', clean_term).strip()

        # 1. Try Wikipedia first for factual entities
        wiki_res = cls.search_wikipedia(clean_term)
        if wiki_res:
            return {
                "found": True,
                "source": "wikipedia",
                "term": clean_term,
                "summary": wiki_res,
                "bql_triple": f"TEACH {clean_term.replace(' ', '_')} is_a {wiki_res[:100].replace(' ', '_')}"
            }

        # 2. Try DuckDuckGo Instant Answer
        ddg_res = cls.search_duckduckgo(clean_term)
        if ddg_res:
            return {
                "found": True,
                "source": "duckduckgo",
                "term": clean_term,
                "summary": ddg_res,
                "bql_triple": f"TEACH {clean_term.replace(' ', '_')} is_a {ddg_res[:100].replace(' ', '_')}"
            }

        # 3. Try Urban Dictionary for Slang
        slang_res = cls.search_urban_slang(clean_term)
        if slang_res:
            return {
                "found": True,
                "source": "urban_slang",
                "term": clean_term,
                "summary": slang_res,
                "bql_triple": f"TEACH {clean_term.replace(' ', '_')} is_slang_for {slang_res[:100].replace(' ', '_')}"
            }

        return {
            "found": False,
            "source": "none",
            "term": clean_term,
            "summary": f"I searched across encyclopedic and slang databases for '{clean_term}' but found no verified entries. I'd love to learn—could you tell me what it means?",
            "bql_triple": ""
        }

if __name__ == "__main__":
    test_queries = ["quokka", "capybara", "delulu", "quantum supremacy", "xyz999fakeconcept"]
    for q in test_queries:
        res = EpistemicWebGrounder.ground_concept(q)
        print(f"\n🔍 Query: '{q}'")
        print(f"   Found: {res['found']} (Source: {res['source']})")
        print(f"   Summary: {res['summary'][:120]}...")
