#!/usr/bin/env python3
"""
Unit and Integration Test Suite for THE BRAIN 3 Hugging Face Streaming & Scaled Curriculum Trainer
"""

import unittest
import os
import sys
import shutil
import json

# Add brain3 root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from brain3.training.hf_curriculum_trainer import (
    DiskGuard,
    ZeroDiskHFStreamer,
    FactExtractor,
    SciQCurriculum,
    GSM8KCurriculum,
    OpenBookQACurriculum,
    ARCCurriculum,
    SVAMPCurriculum,
    CommonsenseQACurriculum,
    BrainBridge,
    HFCurriculumTrainer
)

class TestHFCurriculumTrainer(unittest.TestCase):

    def test_01_disk_guard(self):
        """Test DiskGuard metrics and delta tracking."""
        guard = DiskGuard(name="TestGuard")
        free_gb = guard.get_free_gb()
        self.assertGreater(free_gb, 0.0)
        delta = guard.check()
        self.assertAlmostEqual(delta, 0.0, delta=50.0)
        guard.close()

    def test_02_fact_extractor_patterns(self):
        """Test sentence to triple extraction across multiple relational patterns."""
        # Pattern: is_a
        triples = FactExtractor.extract_from_sentence("Eagle is a bird")
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0], ("eagle", "is_a", "bird"))

        # Pattern: %MID% (is the X of Y)
        triples2 = FactExtractor.extract_from_sentence("Chloroplast is the organelle of plant")
        self.assertEqual(len(triples2), 1)
        self.assertEqual(triples2[0], ("chloroplast", "organelle", "plant"))

        # Pattern: produces
        triples3 = FactExtractor.extract_from_sentence("Photosynthesis produces glucose")
        self.assertEqual(len(triples3), 1)
        self.assertEqual(triples3[0], ("photosynthesis", "produces", "glucose"))

        # Pattern: causes
        triples4 = FactExtractor.extract_from_sentence("Gravity causes acceleration")
        self.assertEqual(len(triples4), 1)
        self.assertEqual(triples4[0], ("gravity", "causes", "acceleration"))

        # Pattern: made_of
        triples5 = FactExtractor.extract_from_sentence("Water is made of hydrogen")
        self.assertEqual(len(triples5), 1)
        self.assertEqual(triples5[0], ("water", "made_of", "hydrogen"))

    def test_03_sciq_curriculum_processing(self):
        """Test conversion of SciQ QA rows into BrainQL TEACH commands."""
        sample_row = {
            "question": "What organ pumps blood in humans?",
            "support": "Heart is a muscle. Heart pumps oxygenated blood throughout the human body.",
            "correct_answer": "heart"
        }
        cmds = SciQCurriculum.process_row(sample_row)
        self.assertTrue(any("TEACH heart is_a muscle" in c for c in cmds))
        self.assertTrue(any("TEACH heart is_a organ" in c for c in cmds))

    def test_04_gsm8k_curriculum_processing(self):
        """Test extraction of arithmetic calculations into fast System 1 reflex arcs."""
        sample_row = {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.",
            "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72"
        }
        cmds = GSM8KCurriculum.process_row(sample_row)
        self.assertIn("INSTINCT_TRAIN 48/2 -> 24", cmds)
        self.assertIn("INSTINCT_TRAIN 48+24 -> 72", cmds)

    def test_05_openbookqa_curriculum_processing(self):
        """Test extraction of OpenBookQA facts into BrainQL commands."""
        sample_row = {
            "question_stem": "Which of these is a source of energy for plants?",
            "fact1": "Sunlight is a energy source",
            "answer_key": "A"
        }
        cmds = OpenBookQACurriculum.process_row(sample_row)
        self.assertIn("TEACH sunlight is_a energy_source", cmds)

    def test_06_arc_curriculum_processing(self):
        """Test AI2 ARC causal assertion extraction."""
        sample_row = {
            "question": "Which factor will most likely cause a fever?",
            "choices": {"label": ["A", "B"], "text": ["muscle relaxing", "bacterial infection"]},
            "answerKey": "B"
        }
        cmds = ARCCurriculum.process_row(sample_row)
        self.assertIn("TEACH bacterial_infection causes fever", cmds)

    def test_07_svamp_curriculum_processing(self):
        """Test SVAMP mathematical equation compilation into reflex arcs."""
        sample_row = {
            "Body": "There are 290 bananas in Philip's collection organized into 2 groups.",
            "Question": "How big is each group?",
            "Equation": "( 290.0 / 2.0 )",
            "Answer": "145"
        }
        cmds = SVAMPCurriculum.process_row(sample_row)
        self.assertIn("INSTINCT_TRAIN 290/2 -> 145", cmds)

    def test_08_commonsense_qa_curriculum_processing(self):
        """Test CommonsenseQA relational association extraction."""
        sample_row = {
            "question": "Where do birds lay eggs?",
            "question_concept": "birds",
            "choices": {"label": ["A", "B"], "text": ["tree nest", "underwater ocean"]},
            "answerKey": "A"
        }
        cmds = CommonsenseQACurriculum.process_row(sample_row)
        self.assertIn("TEACH birds related_to tree_nest", cmds)

    def test_09_brain_pipe_bridge_e2e(self):
        """Test end-to-end communication with BrainPipeServer, fact ingestion, reflex firing, and sleep consolidation."""
        bridge = BrainBridge(base_dir=".")
        try:
            # 1. Test basic BrainQL execution
            res = bridge.execute_bql("INSTINCT 2+2")
            self.assertEqual(res.get("status"), "ok")
            result_obj = json.loads(res["result"])
            self.assertTrue(result_obj.get("verified"))
            self.assertEqual(result_obj.get("result"), "4")

            # 2. Test Batch Fact Teaching
            batch_cmds = [
                "TEACH falcon is_a bird",
                "TEACH bird can fly",
                "INSTINCT_TRAIN 12*12 -> 144"
            ]
            batch_res = bridge.execute_batch(batch_cmds)
            self.assertEqual(batch_res.get("status"), "ok")
            self.assertEqual(batch_res.get("total"), 3)
            self.assertEqual(batch_res.get("success"), 3)

            # 3. Verify taught fact via LOOKUP
            lookup_res = bridge.execute_bql("LOOKUP falcon is_a")
            lookup_obj = json.loads(lookup_res["result"])
            self.assertEqual(lookup_obj.get("result"), "bird")
            self.assertTrue(lookup_obj.get("verified"))

            # 4. Verify trained reflex via INSTINCT_FIRE
            instinct_res = bridge.execute_bql("INSTINCT 12*12")
            instinct_obj = json.loads(instinct_res["result"])
            self.assertEqual(instinct_obj.get("result"), "144")
            self.assertTrue(instinct_obj.get("verified"))

            # 5. Test Sleep Consolidation
            sleep_res = bridge.sleep_consolidate()
            self.assertEqual(sleep_res.get("status"), "ok")
            self.assertIn("phase", sleep_res.get("report", "").lower())

        finally:
            bridge.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
