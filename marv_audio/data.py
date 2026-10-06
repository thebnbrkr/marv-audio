"""Small speech samples for tutorials, tests and experiments."""
from __future__ import annotations

import io
import os
import subprocess
import tempfile

import numpy as np

LIBRISPEECH_REPO = "hf-internal-testing/librispeech_asr_dummy"
LIBRISPEECH_FILE = "clean/validation-00000-of-00001.parquet"


def _decode_flac(data: bytes) -> np.ndarray:
    """FLAC bytes -> float32 mono at 16 kHz. Uses soundfile (Linux, Colab,
    anywhere it installs); falls back to macOS's built-in afconvert."""
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    if sf is not None:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"expected 16 kHz audio, got {sr} Hz")
        return audio
    import wave

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "a.flac"), os.path.join(tmp, "a.wav")
        open(src, "wb").write(data)
        try:
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", src, dst], check=True)
        except FileNotFoundError:
            raise ImportError("decoding FLAC needs soundfile: pip install soundfile") from None
        with wave.open(dst) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def librispeech_clips(n: int = 20) -> list[tuple[np.ndarray, str]]:
    """The first `n` clips of a 73-clip LibriSpeech sample (clean validation,
    9 MB download): [(audio float32 at 16 kHz, reference transcript), ...]."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(LIBRISPEECH_REPO, LIBRISPEECH_FILE, repo_type="dataset")
    rows = pq.read_table(path).to_pylist()[:n]
    return [(_decode_flac(r["audio"]["bytes"]), r["text"]) for r in rows]
