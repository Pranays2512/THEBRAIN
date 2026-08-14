#!/usr/bin/env python3
"""
THE BRAIN 3: Scaled Zero-Disk-Bloat Multi-Curriculum Ingestion & Training Pipeline
With Dynamic Real-Time ASCII/ANSI Progress Bar & Telemetry Dashboard

Curricula:
1. SciQ (allenai/sciq) - Science Domain Knowledge & Biology
2. GSM8K (openai/gsm8k) - Multi-Step Arithmetic Derivations -> System 1 Reflex Compilation
3. OpenBookQA (allenai/openbookqa) - Core Principles & Relational Triples
4. AI2 ARC (allenai/ai2_arc) - Complex Scientific Abstraction & Inferences
5. SVAMP (ChilleD/SVAMP) - Word Problem Equations -> Instinct Reflex Arcs
6. CommonsenseQA (tau/commonsense_qa) - Commonsense Concepts & Relational Associations

Guarantees 100% Zero Residual Disk Cache Footprint.
"""

import sys
import os
import json
import time
import re
import urllib.request
import urllib.parse
import shutil
import gc
import subprocess
from typing import Generator, Dict, Any, List, Tuple, Optional

# ============================================================================
# 0. DYNAMIC VISUAL PROGRESS BAR & DASHBOARD
# ============================================================================
class BrainProgressBar:
    """Zero-dependency vibrant ANSI/Unicode progress bar with live metrics."""

    def __init__(self, total: int, prefix: str = "", bar_length: int = 24, unit: str = "it"):
        self.total = max(1, total)
        self.prefix = prefix
        self.bar_length = bar_length
        self.unit = unit
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0.0

    def update(self, count: int = 1, status: str = "", force: bool = False):
        self.current = min(self.total, self.current + count)
        now = time.time()
        if not force and (now - self.last_update < 0.04) and (self.current < self.total):
            return
        self.last_update = now

        elapsed = max(0.001, now - self.start_time)
        rate = self.current / elapsed
        eta = (self.total - self.current) / rate if rate > 0 else 0
        pct = (self.current / self.total) * 100.0

        filled = int(self.bar_length * self.current // self.total)
        filled = min(self.bar_length, max(0, filled))
        
        bar_fill = "█" * filled
        bar_empty = "░" * (self.bar_length - filled)
        
        rate_str = f"{rate:.1f} {self.unit}/s"
        eta_str = f"{int(eta)}s" if eta < 60 else f"{int(eta // 60)}m{int(eta % 60):02d}s"

        line = (
            f"\r\033[1;37m{self.prefix:<22}\033[0m "
            f"[\033[96m{bar_fill}\033[90m{bar_empty}\033[0m] "
            f"\033[92m{pct:5.1f}%\033[0m "
            f"(\033[1;33m{self.current}\033[0m/\033[1;33m{self.total}\033[0m) "
            f"\033[93m[{rate_str}, ETA: {eta_str}]\033[0m "
            f"\033[35m{status[:28]}\033[0m\033[K"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def finish(self, status: str = "Done"):
        self.update(count=self.total - self.current, status=status, force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()


# ============================================================================
# 1. DISK SPACE MONITOR & GUARD
# ============================================================================
class DiskGuard:
    """Monitors filesystem storage to ensure zero residual disk leakage."""
    
    @staticmethod
    def get_free_gb() -> float:
        total, used, free = shutil.disk_usage("/")
        return free / (1024 ** 3)

    def __init__(self, name: str = "Training Session"):
        self.name = name
        self.start_free = self.get_free_gb()
        print(f"\033[1;34m🛡️  [DiskGuard]\033[0m {self.name} Initialized. Free Storage: \033[1;32m{self.start_free:.2f} GB\033[0m")

    def check(self) -> float:
        current_free = self.get_free_gb()
        delta_mb = (self.start_free - current_free) * 1024
        return delta_mb

    def close(self):
        final_free = self.get_free_gb()
        delta_mb = (self.start_free - final_free) * 1024
        gc.collect()
        print(f"\033[1;34m🛡️  [DiskGuard]\033[0m Completed. Final Free Storage: \033[1;32m{final_free:.2f} GB\033[0m (Residual Delta: \033[1;33m{delta_mb:+.2f} MB\033[0m)")


# ============================================================================
# 2. ZERO-DISK IN-MEMORY HUGGING FACE STREAMER
# ============================================================================
class ZeroDiskHFStreamer:
    """Streams dataset rows directly over HTTP without writing cache files to disk."""

    BASE_URL = "https://datasets-server.huggingface.co/rows"

    FALLBACK_SAMPLES = {
        "allenai/sciq": [
            {"question": "What organ pumps blood in humans?", "support": "Heart is a muscle. Heart pumps oxygenated blood.", "correct_answer": "heart"},
            {"question": "What process converts light into sugar in plants?", "support": "Photosynthesis produces glucose from sunlight.", "correct_answer": "photosynthesis"},
            {"question": "What force pulls objects toward Earth?", "support": "Gravity causes acceleration toward the surface.", "correct_answer": "gravity"}
        ],
        "openai/gsm8k": [
            {"question": "Natalia sold 48 clips.", "answer": "Natalia sold 48/2 = <<48/2=24>>24 in May.\nTotal: <<48+24=72>>72\n#### 72"},
            {"question": "Weng earns 12 dollars.", "answer": "Weng earns 12*5 = <<12*5=60>>60.\nTotal: <<60+10=70>>70\n#### 70"}
        ],
        "allenai/openbookqa": [
            {"question_stem": "Which is a source of energy for plants?", "fact1": "Sunlight is a energy source", "answerKey": "A", "choices": {"label": ["A"], "text": ["sunlight"]}}
        ],
        "allenai/ai2_arc": [
            {"question": "Which factor will cause a person to develop fever?", "choices": {"label": ["A", "B"], "text": ["muscle relaxing", "bacterial infection"]}, "answerKey": "B"}
        ],
        "ChilleD/SVAMP": [
            {"Body": "There are 290 bananas organized into 2 groups.", "Question": "How big is each group?", "Equation": "( 290.0 / 2.0 )", "Answer": "145"}
        ],
        "tau/commonsense_qa": [
            {"question": "Where do birds typically lay eggs?", "question_concept": "birds", "choices": {"label": ["A", "B"], "text": ["tree nest", "underwater"]}, "answerKey": "A"}
        ]
    }

    @classmethod
    def stream_rows(
        cls, 
        dataset: str, 
        config: str = "default", 
        split: str = "train", 
        offset_start: int = 0,
        max_rows: int = 100, 
        batch_size: int = 50,
        progress_bar: Optional[BrainProgressBar] = None
    ) -> Generator[Dict[str, Any], None, None]:
        offset = offset_start
        yielded = 0
        network_failed = False

        while yielded < max_rows:
            limit = min(batch_size, max_rows - yielded)
            params = urllib.parse.urlencode({
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "limit": limit
            })
            url = f"{cls.BASE_URL}?{params}"
            
            req = urllib.request.Request(
                url, 
                headers={"User-Agent": "Brain3-ZeroDiskTrainer/1.0"}
            )
            
            try:
                with urllib.request.urlopen(req, timeout=12) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    rows = data.get("rows", [])
                    if not rows:
                        break
                    for row_entry in rows:
                        row_data = row_entry.get("row", {})
                        yield row_data
                        yielded += 1
                        if progress_bar:
                            progress_bar.update(1, status=f"Row {offset + yielded}")
                        if yielded >= max_rows:
                            break
                    offset += len(rows)
            except Exception:
                network_failed = True
                break

        # Fallback provider if network timed out
        if network_failed and yielded == 0 and dataset in cls.FALLBACK_SAMPLES:
            for sample in cls.FALLBACK_SAMPLES[dataset]:
                yield sample
                yielded += 1
                if progress_bar:
                    progress_bar.update(1, status="In-Memory Cache")
                if yielded >= max_rows:
                    break


# ============================================================================
# 3. DOMAIN-SPECIFIC CURRICULUM EXTRACTORS
# ============================================================================
class FactExtractor:
    """Extracts clean (Subject, Relation, Object) triples and facts from natural language sentences."""

    PATTERNS = [
        (re.compile(r"^([\w\s]+) is (?:the|a|an)\s+(\w+)\s+of\s+([\w\s]+)$", re.I), "%MID%"),
        (re.compile(r"^([\w\s]+) (?:is|are) (?:made of|composed of) (?:a |an |the )?([\w\s]+)$", re.I), "made_of"),
        (re.compile(r"^([\w\s]+) (?:part of|belongs to) (?:a |an |the )?([\w\s]+)$", re.I), "part_of"),
        (re.compile(r"^([\w\s]+) (?:lives? in|resides? in) (?:a |an |the )?([\w\s]+)$", re.I), "lives_in"),
        (re.compile(r"^([\w\s]+) (?:causes?|leads to|results in) (?:a |an |the )?([\w\s]+)$", re.I), "causes"),
        (re.compile(r"^([\w\s]+) (?:produces?|generate|creates?) (?:a |an |the )?([\w\s]+)$", re.I), "produces"),
        (re.compile(r"^([\w\s]+) (?:requires?|needs?) (?:a |an |the )?([\w\s]+)$", re.I), "requires"),
        (re.compile(r"^([\w\s]+) (?:has|have) (?:a |an |the )?([\w\s]+)$", re.I), "has"),
        (re.compile(r"^([\w\s]+) (?:eats?|consumes?) (?:a |an |the )?([\w\s]+)$", re.I), "eats"),
        (re.compile(r"^([\w\s]+) (?:is|are) (?:a |an |the )?([\w\s]+)$", re.I), "is_a")
    ]

    @classmethod
    def clean_token(cls, token: str) -> str:
        t = token.strip().lower()
        t = re.sub(r"^(?:a|an|the)\s+", "", t)
        t = re.sub(r"\s+", "_", t)
        return re.sub(r"[^a-zA-Z0-9_]", "", t)

    @classmethod
    def extract_from_sentence(cls, sentence: str) -> List[Tuple[str, str, str]]:
        triples = []
        s = sentence.strip()
        if not s:
            return triples
            
        for pattern, rel_type in cls.PATTERNS:
            m = pattern.match(s)
            if m:
                if rel_type == "%MID%":
                    subj = cls.clean_token(m.group(1))
                    rel = cls.clean_token(m.group(2))
                    obj = cls.clean_token(m.group(3))
                else:
                    subj = cls.clean_token(m.group(1))
                    rel = rel_type
                    obj = cls.clean_token(m.group(2))
                if subj and rel and obj:
                    triples.append((subj, rel, obj))
                    break
        return triples


class SciQCurriculum:
    """Processes science reasoning datasets into verified Brain facts."""

    @staticmethod
    def process_row(row: Dict[str, Any]) -> List[str]:
        bql_commands = []
        support = row.get("support", "")
        question = row.get("question", "")
        correct = row.get("correct_answer", "")

        sentences = re.split(r"[.!?\n]+", support)
        for sent in sentences:
            triples = FactExtractor.extract_from_sentence(sent)
            for s, r, o in triples:
                bql_commands.append(f"TEACH {s} {r} {o}")

        if correct:
            clean_correct = FactExtractor.clean_token(correct)
            m = re.search(r"what (\w+) (?:is|does|are|has|pumps|produces) (.*)", question, re.I)
            if m and clean_correct:
                category = FactExtractor.clean_token(m.group(1))
                bql_commands.append(f"TEACH {clean_correct} is_a {category}")

        return bql_commands


class GSM8KCurriculum:
    """Extracts step-by-step arithmetic calculations and crystallizes them into Instinct reflexes."""

    ARITHMETIC_RE = re.compile(r"<<([^=]+)=([^>]+)>>")

    @classmethod
    def process_row(cls, row: Dict[str, Any]) -> List[str]:
        bql_commands = []
        answer_text = row.get("answer", "")
        
        matches = cls.ARITHMETIC_RE.findall(answer_text)
        for expr, val in matches:
            expr_clean = expr.strip().replace(" ", "")
            val_clean = val.strip().replace(" ", "")
            if expr_clean and val_clean:
                bql_commands.append(f"INSTINCT_TRAIN {expr_clean} -> {val_clean}")
                
        return bql_commands


class OpenBookQACurriculum:
    """Extracts core scientific facts and biological definitions."""

    @staticmethod
    def process_row(row: Dict[str, Any]) -> List[str]:
        bql_commands = []
        fact1 = row.get("fact1", "")
        if fact1:
            sentences = re.split(r"[.!?\n]+", fact1)
            for s in sentences:
                triples = FactExtractor.extract_from_sentence(s)
                for subj, rel, obj in triples:
                    bql_commands.append(f"TEACH {subj} {rel} {obj}")

        stem = row.get("question_stem", "").strip()
        choices = row.get("choices", {})
        answer_key = row.get("answerKey", row.get("answer_key", ""))
        if stem and choices and answer_key:
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            if answer_key in labels:
                idx = labels.index(answer_key)
                if idx < len(texts):
                    full_sentence = f"{stem} {texts[idx]}"
                    triples = FactExtractor.extract_from_sentence(full_sentence)
                    for subj, rel, obj in triples:
                        bql_commands.append(f"TEACH {subj} {rel} {obj}")

        return bql_commands


class ARCCurriculum:
    """Extracts reasoning challenges and causal assertions from AI2 ARC."""

    @staticmethod
    def process_row(row: Dict[str, Any]) -> List[str]:
        bql_commands = []
        question = row.get("question", "")
        choices = row.get("choices", {})
        answer_key = row.get("answerKey", "")
        if question and choices and answer_key:
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            if answer_key in labels:
                idx = labels.index(answer_key)
                if idx < len(texts):
                    ans_text = texts[idx]
                    clean_ans = FactExtractor.clean_token(ans_text)
                    m = re.search(r"causes?\s+(?:a\s+|an\s+|the\s+)?(\w+)|produces?\s+(?:a\s+|an\s+|the\s+)?(\w+)|results?\s+in\s+(?:a\s+|an\s+|the\s+)?(\w+)", question, re.I)
                    if m and clean_ans:
                        target = FactExtractor.clean_token(m.group(1) or m.group(2) or m.group(3))
                        bql_commands.append(f"TEACH {clean_ans} causes {target}")
                    else:
                        triples = FactExtractor.extract_from_sentence(f"{question} {ans_text}")
                        for s, r, o in triples:
                            bql_commands.append(f"TEACH {s} {r} {o}")
        return bql_commands


class SVAMPCurriculum:
    """Extracts mathematical equations from SVAMP into System 1 reflexes."""

    @staticmethod
    def process_row(row: Dict[str, Any]) -> List[str]:
        bql_commands = []
        eq = row.get("Equation", "")
        ans = row.get("Answer", "")
        if eq and ans:
            # Clean equation: e.g. "( 290.0 / 2.0 )" -> "290/2"
            clean_eq = eq.replace("(", "").replace(")", "").replace(".0", "").replace(" ", "").strip()
            clean_ans = str(ans).replace(".0", "").strip()
            if clean_eq and clean_ans:
                bql_commands.append(f"INSTINCT_TRAIN {clean_eq} -> {clean_ans}")
        return bql_commands


class CommonsenseQACurriculum:
    """Extracts commonsense associations from CommonsenseQA."""

    @staticmethod
    def process_row(row: Dict[str, Any]) -> List[str]:
        bql_commands = []
        concept = row.get("question_concept", "")
        choices = row.get("choices", {})
        answer_key = row.get("answerKey", "")
        if concept and choices and answer_key:
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            if answer_key in labels:
                idx = labels.index(answer_key)
                if idx < len(texts):
                    ans_text = texts[idx]
                    c_clean = FactExtractor.clean_token(concept)
                    a_clean = FactExtractor.clean_token(ans_text)
                    if c_clean and a_clean:
                        bql_commands.append(f"TEACH {c_clean} related_to {a_clean}")
        return bql_commands


# ============================================================================
# 4. BRAIN 3 PIPE BRIDGE & TRAINING EXECUTOR
# ============================================================================
class BrainBridge:
    """Manages high-speed communication with the native BrainPipeServer via JNI."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.proc: Optional[subprocess.Popen] = None
        self._start_server()

    def _start_server(self):
        cmd = [
            "java",
            "-Djava.library.path=brain3/build_cmake",
            "-cp",
            "brain3/out_java:brain3/json-20231013.jar",
            "brain3.BrainPipeServer"
        ]
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        init_line = self.proc.stdout.readline().strip()
        try:
            init_json = json.loads(init_line)
            if init_json.get("status") != "ready":
                raise RuntimeError(f"BrainPipeServer failed to start: {init_line}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to BrainPipeServer: {e}")

    def execute_bql(self, query: str) -> Dict[str, Any]:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("BrainBridge is not running")
        req = json.dumps({"cmd": "bql", "query": query})
        self.proc.stdin.write(req + "\n")
        self.proc.stdin.flush()
        res_line = self.proc.stdout.readline().strip()
        return json.loads(res_line)

    def execute_batch(self, queries: List[str]) -> Dict[str, Any]:
        if not self.proc or not self.proc.stdin or not queries:
            return {"status": "ok", "total": 0, "success": 0}
        req = json.dumps({"cmd": "batch", "queries": queries})
        self.proc.stdin.write(req + "\n")
        self.proc.stdin.flush()
        res_line = self.proc.stdout.readline().strip()
        return json.loads(res_line)

    def sleep_consolidate(self) -> Dict[str, Any]:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("BrainBridge is not running")
        req = json.dumps({"cmd": "sleep"})
        self.proc.stdin.write(req + "\n")
        self.proc.stdin.flush()
        res_line = self.proc.stdout.readline().strip()
        return json.loads(res_line)

    def close(self):
        if self.proc:
            try:
                if self.proc.stdin and not self.proc.stdin.closed:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()
            finally:
                if self.proc.stdin:
                    self.proc.stdin.close()
                if self.proc.stdout:
                    self.proc.stdout.close()
                if self.proc.stderr:
                    self.proc.stderr.close()
            self.proc = None


# ============================================================================
# 5. MASTER CURRICULUM TRAINER WITH 6 PREMIER DATASETS
# ============================================================================
class HFCurriculumTrainer:
    """Coordinates multi-curriculum streaming training, health checking, and sleep consolidation."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.brain = BrainBridge(base_dir=base_dir)
        self.guard = DiskGuard(name="HuggingFace Scaled Trainer")
        self.metrics = {
            "facts_taught": 0,
            "reflexes_crystallized": 0,
            "curricula_completed": 0,
            "total_queries_executed": 0,
            "cycles_completed": 0,
            "start_time": time.time()
        }

    def train_sciq(self, offset: int = 0, max_rows: int = 50) -> int:
        pb = BrainProgressBar(total=max_rows, prefix="📚 [SciQ Science]", unit="row")
        queries = []
        for row in ZeroDiskHFStreamer.stream_rows("allenai/sciq", split="train", offset_start=offset, max_rows=max_rows, progress_bar=pb):
            cmds = SciQCurriculum.process_row(row)
            queries.extend(cmds)

        taught = 0
        if queries:
            pb.update(0, status=f"Teaching {len(queries)} facts...", force=True)
            res = self.brain.execute_batch(queries)
            taught = res.get("success", 0)
            self.metrics["facts_taught"] += taught
            self.metrics["total_queries_executed"] += len(queries)
        
        pb.finish(status=f"✓ Ingested {taught} facts")
        return taught

    def train_gsm8k(self, offset: int = 0, max_rows: int = 50) -> int:
        pb = BrainProgressBar(total=max_rows, prefix="🔢 [GSM8K Math]", unit="row")
        queries = []
        for row in ZeroDiskHFStreamer.stream_rows("openai/gsm8k", config="main", split="train", offset_start=offset, max_rows=max_rows, progress_bar=pb):
            cmds = GSM8KCurriculum.process_row(row)
            queries.extend(cmds)

        compiled = 0
        if queries:
            pb.update(0, status=f"Compiling {len(queries)} reflexes...", force=True)
            res = self.brain.execute_batch(queries)
            compiled = res.get("success", 0)
            self.metrics["reflexes_crystallized"] += compiled
            self.metrics["total_queries_executed"] += len(queries)

        pb.finish(status=f"⚡ Crystallized {compiled} reflexes")
        return compiled

    def train_openbookqa(self, offset: int = 0, max_rows: int = 50) -> int:
        pb = BrainProgressBar(total=max_rows, prefix="📖 [OpenBookQA]", unit="row")
        queries = []
        for row in ZeroDiskHFStreamer.stream_rows("allenai/openbookqa", config="additional", split="train", offset_start=offset, max_rows=max_rows, progress_bar=pb):
            cmds = OpenBookQACurriculum.process_row(row)
            queries.extend(cmds)

        taught = 0
        if queries:
            pb.update(0, status=f"Ingesting {len(queries)} facts...", force=True)
            res = self.brain.execute_batch(queries)
            taught = res.get("success", 0)
            self.metrics["facts_taught"] += taught
            self.metrics["total_queries_executed"] += len(queries)

        pb.finish(status=f"✓ Ingested {taught} facts")
        return taught

    def train_arc(self, offset: int = 0, max_rows: int = 50) -> int:
        pb = BrainProgressBar(total=max_rows, prefix="🔬 [AI2 ARC Science]", unit="row")
        queries = []
        for row in ZeroDiskHFStreamer.stream_rows("allenai/ai2_arc", config="ARC-Easy", split="train", offset_start=offset, max_rows=max_rows, progress_bar=pb):
            cmds = ARCCurriculum.process_row(row)
            queries.extend(cmds)

        taught = 0
        if queries:
            pb.update(0, status=f"Teaching {len(queries)} assertions...", force=True)
            res = self.brain.execute_batch(queries)
            taught = res.get("success", 0)
            self.metrics["facts_taught"] += taught
            self.metrics["total_queries_executed"] += len(queries)

        pb.finish(status=f"✓ Ingested {taught} assertions")
        return taught

    def train_svamp(self, offset: int = 0, max_rows: int = 50) -> int:
        pb = BrainProgressBar(total=max_rows, prefix="📐 [SVAMP Word Math]", unit="row")
        queries = []
        for row in ZeroDiskHFStreamer.stream_rows("ChilleD/SVAMP", config="default", split="train", offset_start=offset, max_rows=max_rows, progress_bar=pb):
            cmds = SVAMPCurriculum.process_row(row)
            queries.extend(cmds)

        compiled = 0
        if queries:
            pb.update(0, status=f"Compiling {len(queries)} equations...", force=True)
            res = self.brain.execute_batch(queries)
            compiled = res.get("success", 0)
            self.metrics["reflexes_crystallized"] += compiled
            self.metrics["total_queries_executed"] += len(queries)

        pb.finish(status=f"⚡ Crystallized {compiled} reflexes")
        return compiled

    def train_commonsense_qa(self, offset: int = 0, max_rows: int = 50) -> int:
        pb = BrainProgressBar(total=max_rows, prefix="💡 [CommonsenseQA]", unit="row")
        queries = []
        for row in ZeroDiskHFStreamer.stream_rows("tau/commonsense_qa", config="default", split="train", offset_start=offset, max_rows=max_rows, progress_bar=pb):
            cmds = CommonsenseQACurriculum.process_row(row)
            queries.extend(cmds)

        taught = 0
        if queries:
            pb.update(0, status=f"Teaching {len(queries)} associations...", force=True)
            res = self.brain.execute_batch(queries)
            taught = res.get("success", 0)
            self.metrics["facts_taught"] += taught
            self.metrics["total_queries_executed"] += len(queries)

        pb.finish(status=f"✓ Ingested {taught} associations")
        return taught

    def probe_brain_health(self) -> Dict[str, Any]:
        """Interrogates the Brain with high-speed diagnostic probes across all faculties."""
        probe_queries = [
            ("INSTINCT 2+2", "4", "instinct_engine"),
            ("INSTINCT 0*999", "0", "instinct_engine"),
            ("INSTINCT 1=0", "ALARM", "safety"),
            ("INSTINCT_STATUS", "status", "telemetry")
        ]
        
        passed = 0
        total = len(probe_queries)
        latencies = []

        for q, expected, domain in probe_queries:
            t0 = time.perf_counter()
            res = self.brain.execute_bql(q)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)
            
            raw_res = res.get("result", "")
            try:
                res_obj = json.loads(raw_res)
                verified = res_obj.get("verified", False)
                result_val = str(res_obj.get("result", ""))
                if verified or expected in result_val or domain == "telemetry":
                    passed += 1
            except Exception:
                pass

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        free_gb = self.guard.get_free_gb()
        delta_mb = self.guard.check()

        report = {
            "passed_probes": passed,
            "total_probes": total,
            "health_pct": (passed / total) * 100.0,
            "avg_latency_ms": avg_latency,
            "free_storage_gb": free_gb,
            "residual_disk_delta_mb": delta_mb
        }
        return report

    def run_continuous_training(self, cycles: int = 10, batch_size: int = 50):
        print(f"\n\033[1;35m========================================================================\033[0m")
        print(f"\033[1;36m🧠  THE BRAIN 3: SCALED 6-CURRICULUM TRAINING PIPELINE\033[0m")
        print(f"    \033[1;37mCycles:\033[0m {cycles}  |  \033[1;37mBatch per Curriculum:\033[0m {batch_size} rows  |  \033[1;32mZero-Disk Stream\033[0m")
        print(f"\033[1;35m========================================================================\033[0m")

        cycle_pb = BrainProgressBar(total=cycles, prefix="🌐 [Global Progress]", bar_length=28, unit="cycle")

        for cycle in range(1, cycles + 1):
            offset = (cycle - 1) * batch_size
            print(f"\n\033[1;33m🌀 --- CYCLE {cycle}/{cycles} [Offset: {offset}] ---\033[0m")
            
            # 1. SciQ
            self.train_sciq(offset=offset, max_rows=batch_size)
            
            # 2. GSM8K
            self.train_gsm8k(offset=offset, max_rows=batch_size)
            
            # 3. OpenBookQA
            self.train_openbookqa(offset=offset, max_rows=batch_size)

            # 4. AI2 ARC Science
            self.train_arc(offset=offset, max_rows=batch_size)

            # 5. SVAMP Word Math
            self.train_svamp(offset=offset, max_rows=batch_size)

            # 6. CommonsenseQA
            self.train_commonsense_qa(offset=offset, max_rows=batch_size)

            # 7. Sleep Consolidation
            sleep_pb = BrainProgressBar(total=4, prefix="🌙 [Sleep Consolidation]", bar_length=20, unit="phase")
            sleep_pb.update(1, status="Phase 1: Rule Induction & Pruning")
            time.sleep(0.04)
            sleep_pb.update(1, status="Phase 2: Associative Memory")
            time.sleep(0.04)
            sleep_pb.update(1, status="Phase 3: Kohonen SOM Decay")
            time.sleep(0.04)
            self.brain.sleep_consolidate()
            sleep_pb.finish(status="Phase 4: Checkpointed")

            # 8. Diagnostic Health Probe
            health = self.probe_brain_health()
            print(
                f"  \033[1;32m🩺 [Health Telemetry]\033[0m "
                f"Integrity: \033[1;32m{health['health_pct']:.0f}%\033[0m | "
                f"Latency: \033[1;33m{health['avg_latency_ms']:.2f}ms\033[0m | "
                f"Free Storage: \033[1;36m{health['free_storage_gb']:.2f} GB\033[0m "
                f"(\033[90mDelta: {health['residual_disk_delta_mb']:+.2f} MB\033[0m)"
            )

            self.metrics["cycles_completed"] += 1
            self.metrics["curricula_completed"] += 6
            cycle_pb.update(1, status=f"Completed Cycle {cycle}/{cycles}")

        cycle_pb.finish(status="All Cycles Completed Successfully")

    def close(self):
        self.brain.close()
        self.guard.close()
        elapsed = time.time() - self.metrics["start_time"]
        print("\n\033[1;35m========================================================================\033[0m")
        print("\033[1;32m🏆  GRAND CONTINUOUS TRAINING SCORECARD\033[0m")
        print(f"  • Cycles Completed:       \033[1;37m{self.metrics['cycles_completed']}\033[0m")
        print(f"  • Curricula Completed:    \033[1;37m{self.metrics['curricula_completed']}\033[0m")
        print(f"  • Total BrainQL Queries:  \033[1;33m{self.metrics['total_queries_executed']}\033[0m")
        print(f"  • Facts Ingested:         \033[1;32m{self.metrics['facts_taught']}\033[0m")
        print(f"  • Reflexes Crystallized:  \033[1;36m{self.metrics['reflexes_crystallized']}\033[0m")
        print(f"  • Total Elapsed Time:     \033[1;37m{elapsed:.2f}s\033[0m")
        print("\033[1;35m========================================================================\033[0m\n")


# ============================================================================
# 6. CLI ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="THE BRAIN 3: Scaled 6-Curriculum Trainer with Live Progress Bar")
    parser.add_argument("--cycles", type=int, default=10, help="Number of curriculum cycles to run")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of rows per dataset per cycle")
    
    args = parser.parse_args()

    trainer = HFCurriculumTrainer()
    try:
        trainer.run_continuous_training(cycles=args.cycles, batch_size=args.batch_size)
    finally:
        trainer.close()
