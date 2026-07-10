#!/usr/bin/env python3
"""read_pdf_train.py — hand the brain a PDF; it READS it (its own eyes) and TRAINS
whichever faculties you ask for on what it read, then PERSISTS so it accumulates.

    eyes  : ocr_pdf         rasterize + OCR each page -> text        (always, custom-font PDFs)
    --read   : template-cache + cloud teacher -> triples -> numeric/reasoning/membrane + consolidate
    --som    : fuzzy C++ SOM grounding (brain2.Brain.perceive_text) over the OCR text
    --lm     : owned neural LM (NeuralLMTorch) over sentences + extracted structure  (the MOUTH)
    --persist: bridge triples into WholeBrain, run the standing loop, save_state()

No flag = train everything (read + som + lm + persist). Each faculty reads the SAME OCR
text and trains its own slice, so you can run them one at a time or together. The teacher
is Ollama CLOUD qwen3-coder (480B) via the local proxy — a real model does extraction; the
crisp faculties own the truth.

    ../venv2/bin/python3 read_pdf_train.py <in.pdf> [model] [--read] [--som] [--lm]
                                           [--persist] [--max-sents N]
    ../venv2/bin/python3 read_pdf_train.py ~/Downloads/NCERT-Class-10-Science.pdf --lm
"""
import os
import re
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TESSDATA_PREFIX", "/opt/homebrew/opt/tesseract/share/tessdata")

from adapters.ocr_pdf import ocr
from read_book import sentences, BookTrainer, _print_report
from adapters.llm_adapter import OllamaClient, SafeClient

CLOUD_MODEL = "qwen3-coder:480b-cloud"


def get_text(pdf):
    """The EYES — OCR the PDF to text once, reuse if already good this session."""
    base = os.path.splitext(os.path.basename(pdf))[0].lower().replace(" ", "_").replace("-", "_")
    txt_path = os.path.join("data", f"{base}.txt")
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 10_000:
        body = open(txt_path, errors="ignore").read()
        if len(re.findall(r"[A-Za-z]{3,}", body)) > 500:
            print(f"[eyes] using existing OCR text: {txt_path}", flush=True)
            return txt_path, body
    print(f"[eyes] OCR -> {txt_path}", flush=True)
    ocr(pdf, txt_path)
    return txt_path, open(txt_path, errors="ignore").read()


def do_read(sents, model):
    """READ + LEARN: template cache + cloud teacher; consolidate. Returns the trainer."""
    print("\n[read] template-cache + cloud teacher; membrane holds ...", flush=True)
    bt = BookTrainer(client=SafeClient(OllamaClient(model)))
    t0 = time.time()
    bt.read(sents, batch=10)
    bt.consolidate()
    rep = bt.report()
    rep["n_sentences"] = len(sents)
    _print_report(rep)
    print(f"  read+consolidate: {time.time() - t0:.0f}s", flush=True)
    return bt


def do_som(texts):
    """FUZZY: perceive the raw text on the C++ SOM (fuzzy grounding), guarded on brain2."""
    print(f"\n[som] fuzzy grounding {len(texts)} sentences on the C++ SOM ...", flush=True)
    try:
        import brain2
        b = brain2.Brain(som_rows=24, som_cols=24, n_dims=32)
        t0 = time.time()
        for t in texts:
            b.perceive_text(t, brain2.ErrorMode.FULL)
        print(f"  perceived={len(texts)} oov={b.oov_count} conflicts={b.crisp_conflicts} "
              f"({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  skipped ({type(e).__name__}: {e})", flush=True)


def do_lm(texts, bt=None):
    """PROBABILISTIC / MOUTH: train the owned neural LM on the book's sentences plus any
    structure the reader extracted (sentence-level language + the fact shapes)."""
    corpus = list(texts)
    if bt is not None:                                # enrich with extracted structure
        for (s, r), v in bt.numeric.items():
            corpus.append(f"{s} {r} {v}")
        for tri in getattr(getattr(bt, "kre", None), "kb", None).facts if getattr(
                getattr(bt, "kre", None), "kb", None) else []:
            if isinstance(tri, (list, tuple)) and len(tri) == 3:
                corpus.append(f"{tri[0]} {tri[1]} {tri[2]}")
    print(f"\n[lm] training owned LM on {len(corpus)} lines ...", flush=True)
    try:
        from core.neural.neural_lm_torch import NeuralLMTorch
        dim = int(os.environ.get("LM_DIM", 256))
        layers = int(os.environ.get("LM_LAYERS", 4))
        ctx = int(os.environ.get("LM_CTX", 16))
        epochs = int(os.environ.get("LM_EPOCHS", 40))
        t0 = time.time()
        lm = NeuralLMTorch(dim=dim, layers=layers, ctx=ctx, epochs=epochs).train(corpus)
        os.makedirs("trained", exist_ok=True)
        lm.save("trained/owned_lm_data.pt")
        print(f"  device={lm.device} params={lm.param_count()} vocab={len(lm.w2i)} "
              f"-> trained/owned_lm_data.pt ({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  skipped ({type(e).__name__}: {e})", flush=True)


def do_persist(bt):
    """KEEP: bridge every learned triple into the faculties, run the loop, save_state()."""
    print("\n[keep] bridging triples -> faculties, running loop, persisting ...", flush=True)
    try:
        from whole_brain import WholeBrain
        wb = WholeBrain()
        kept = 0
        for (s, r), v in bt.numeric.items():
            if wb.remember(s, r, str(v)):
                kept += 1
        for tri in getattr(getattr(bt, "kre", None), "kb", None).facts if getattr(
                getattr(bt, "kre", None), "kb", None) else []:
            if isinstance(tri, (list, tuple)) and len(tri) == 3:
                if wb.remember(str(tri[0]), str(tri[1]), str(tri[2])):
                    kept += 1
        loop = wb.run_loop(ticks=3)
        saved = wb.save_state()
        print(f"  bridged {kept} triples | loop {loop.get('ticks_run')} ticks "
              f"| persisted {saved}", flush=True)
    except Exception as e:
        print(f"  skipped ({type(e).__name__}: {e})", flush=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    pdf = sys.argv[1]
    model, max_sents = CLOUD_MODEL, None
    args = sys.argv[2:]
    flags = {a for a in args if a.startswith("--") and a != "--max-sents"}
    for i, a in enumerate(args):
        if a == "--max-sents" and i + 1 < len(args):
            max_sents = int(args[i + 1])
        elif not a.startswith("--") and not (i > 0 and args[i - 1] == "--max-sents"):
            model = a
    want_read = "--read" in flags
    want_som = "--som" in flags
    want_lm = "--lm" in flags
    want_persist = "--persist" in flags
    if not flags:                                     # no flag = train everything
        want_read = want_som = want_lm = want_persist = True

    print("=" * 70)
    print(f"  read_pdf_train — {os.path.basename(pdf)}  (teacher: Ollama cloud {model})")
    what = [n for n, on in (("read", want_read), ("som", want_som),
                            ("lm", want_lm), ("persist", want_persist)) if on]
    print(f"  training: {', '.join(what)}")
    print("=" * 70)

    _, text = get_text(pdf)
    sents = sentences(text, max_sents)
    print(f"[read] {len(sents)} sentences available", flush=True)

    # persist depends on the reader's triples; auto-read if persist asked without read
    bt = None
    if want_read or want_persist:
        bt = do_read(sents, model)
    if want_som:
        do_som(sents)
    if want_lm:
        do_lm(sents, bt)
    if want_persist:
        do_persist(bt)

    print("\n  done. the brain read the book itself and trained the requested faculties.")


if __name__ == "__main__":
    main()
