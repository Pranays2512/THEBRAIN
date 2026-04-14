"""
EAR — Real Microphone Input
============================

Captures audio from microphone, computes 13-dim MFCC vectors,
and feeds them directly into the brain the same way text words do.

This replaces typed text with real speech — the brain hears your
actual voice, not simulated MFCC vectors.

Usage in live.py:
    from ear import Ear
    ear = Ear()
    ear.start()
    ...
    frames, words_detected = ear.read_turn()   # blocks until silence
    for mfcc in frames:
        brain.hear(mfcc)
        brain.step()

Or in mic-only mode:
    python ear.py   # test the mic, prints detected energy levels

MFCC parameters match what M71 SpeechCortex was trained on:
    N_MFCC = 13
    hop_length = 512  (~25ms at 22050 Hz)
    n_fft = 1024

The brain doesn't know if MFCCs came from text or a real voice —
it just processes the 13-dim vectors. This is the power of the
abstraction: brain.hear() works with any source.
"""

import numpy as np
import sounddevice as sd
import librosa
import threading
import queue
import time

# ── Audio parameters ─────────────────────────────────────────────────
SAMPLE_RATE   = 22050    # Hz — same as librosa default
FRAME_SIZE    = 512      # samples per MFCC frame (~23ms)
N_MFCC        = 13       # must match brain's M71
N_FFT         = 1024     # FFT window
HOP_LENGTH    = FRAME_SIZE

# ── Voice activity detection ─────────────────────────────────────────
ENERGY_THRESH   = 0.01   # RMS energy above this = voiced
SILENCE_FRAMES  = 20     # frames of silence before "turn" ends (~460ms)
MAX_TURN_FRAMES = 300    # max frames per turn (~7 seconds)


class Ear:
    """
    Continuous microphone listener.

    Buffers incoming audio, detects speech (voice activity),
    and returns complete "turns" (utterances ended by silence).
    """

    def __init__(self, device=None, sample_rate=SAMPLE_RATE):
        self.sample_rate  = sample_rate
        self.device       = device
        self._audio_queue = queue.Queue()
        self._stream      = None
        self._running     = False

    def start(self):
        """Start the microphone stream."""
        self._running = True
        self._stream = sd.InputStream(
            samplerate = self.sample_rate,
            channels   = 1,
            dtype      = 'float32',
            blocksize  = FRAME_SIZE,
            device     = self.device,
            callback   = self._callback,
        )
        self._stream.start()

    def stop(self):
        """Stop the microphone stream."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def _callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio frame."""
        if self._running:
            self._audio_queue.put(indata[:, 0].copy())

    def _frame_to_mfcc(self, audio_block: np.ndarray) -> np.ndarray:
        """Convert a raw audio block to 13-dim MFCC vector."""
        mfcc = librosa.feature.mfcc(
            y          = audio_block.astype(np.float32),
            sr         = self.sample_rate,
            n_mfcc     = N_MFCC,
            n_fft      = N_FFT,
            hop_length = HOP_LENGTH,
        )
        # mfcc shape: (13, T) — take mean across time frames
        vec = mfcc.mean(axis=1).astype(np.float32)

        # Normalize to similar range as trained vocab vectors
        # Librosa MFCCs have large first coefficient (energy).
        # Scale to match vocab.py's range (v[0] ≈ 3.0 when voiced).
        energy = float(np.sqrt(np.mean(audio_block ** 2)))  # RMS
        vec[0] = float(np.clip(energy * 30.0, 0.0, 5.0))    # energy → v[0]
        # Scale remaining coefficients to [-2, 2] range
        vec[1:] = np.clip(vec[1:] / 20.0, -2.0, 2.0)
        return vec

    def _is_voiced(self, audio_block: np.ndarray) -> bool:
        """Simple energy-based voice activity detection."""
        rms = float(np.sqrt(np.mean(audio_block ** 2)))
        return rms > ENERGY_THRESH

    def read_turn(self, timeout: float = 10.0) -> list:
        """
        Collect one complete spoken turn.

        Blocks until:
          - Speech detected, then silence follows (normal turn end)
          - MAX_TURN_FRAMES reached (very long input)
          - timeout reached (no speech)

        Returns list of 13-dim MFCC np.ndarray — one per voiced frame.
        Empty list = no speech detected (silence / timeout).
        """
        mfcc_frames  = []
        silence_count = 0
        speech_started = False
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                block = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            voiced = self._is_voiced(block)
            mfcc   = self._frame_to_mfcc(block)

            if voiced:
                speech_started = True
                silence_count  = 0
                mfcc_frames.append(mfcc)
            elif speech_started:
                silence_count += 1
                mfcc_frames.append(mfcc)   # include trailing silence
                if silence_count >= SILENCE_FRAMES:
                    break                  # natural turn end

            if len(mfcc_frames) >= MAX_TURN_FRAMES:
                break

        # Trim trailing silence frames
        while mfcc_frames and float(mfcc_frames[-1][0]) < 0.5:
            mfcc_frames.pop()

        return mfcc_frames

    def energy_level(self) -> float:
        """Return current microphone energy (for /state display)."""
        try:
            block = self._audio_queue.get_nowait()
            return float(np.sqrt(np.mean(block ** 2)))
        except queue.Empty:
            return 0.0


# ── Standalone test ──────────────────────────────────────────────────

if __name__ == '__main__':
    print("Ear test — speak into microphone. Ctrl+C to stop.")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  MFCC dims:   {N_MFCC}")
    print(f"  Energy threshold: {ENERGY_THRESH}")
    print()

    ear = Ear()
    ear.start()

    turn = 0
    try:
        while True:
            print(f"  Listening for turn {turn + 1}... ", end='', flush=True)
            frames = ear.read_turn(timeout=30.0)
            if frames:
                turn += 1
                energies = [float(f[0]) for f in frames]
                print(f"got {len(frames)} frames, "
                      f"energy range [{min(energies):.2f}, {max(energies):.2f}]")
                print(f"  First MFCC: {frames[0][:6].round(3)}")
            else:
                print("(silence)")
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        ear.stop()
