#!/usr/bin/env python3
"""
brain3/training/test_universal_streamer.py

Unit tests for Pillar 1: Universal Knowledge Scaling & Omniscience Streaming Engine.
"""

import unittest
import os
import sys
import json

# Add parent directories to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from brain3.training.universal_knowledge_streamer import UniversalKnowledgeStreamer

class TestUniversalKnowledgeStreamer(unittest.TestCase):

    def test_01_curated_graph_structure(self):
        """Test that the curated universal graph has valid triples and domains."""
        graph = UniversalKnowledgeStreamer.CURATED_UNIVERSAL_GRAPH
        self.assertGreater(len(graph), 20)
        for entry in graph:
            self.assertIn("subj", entry)
            self.assertIn("rel", entry)
            self.assertIn("obj", entry)
            self.assertIn("domain", entry)

    def test_02_conceptnet_triple_streaming(self):
        """Test streaming triples for a target concept."""
        triples = UniversalKnowledgeStreamer.stream_conceptnet_triples("neuron", limit=5)
        self.assertIsInstance(triples, list)
        self.assertGreater(len(triples), 0)
        s, r, o = triples[0]
        self.assertTrue(len(s) > 0)
        self.assertTrue(len(r) > 0)
        self.assertTrue(len(o) > 0)

    def test_03_universal_ingestion_and_query_e2e(self):
        """Test ingesting universal knowledge and querying it via BrainBridge."""
        streamer = UniversalKnowledgeStreamer()
        try:
            # 1. Ingest Master Graph
            taught = streamer.ingest_curated_omniscience_graph()
            self.assertGreater(taught, 20)

            # 2. Query Ingested Knowledge via LOOKUP
            res = streamer.brain.execute_bql("LOOKUP quicksort average_time_complexity")
            res_obj = json.loads(res.get("result", "{}"))
            self.assertEqual(res_obj.get("result"), "o_n_log_n")
            self.assertTrue(res_obj.get("verified"))

            res2 = streamer.brain.execute_bql("LOOKUP myocardium is_a")
            res2_obj = json.loads(res2.get("result", "{}"))
            self.assertEqual(res2_obj.get("result"), "cardiac_muscle_tissue")
            self.assertTrue(res2_obj.get("verified"))

            # 3. Test Sleep Consolidation
            sleep_res = streamer.brain.sleep_consolidate()
            self.assertEqual(sleep_res.get("status"), "ok")
        finally:
            streamer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
