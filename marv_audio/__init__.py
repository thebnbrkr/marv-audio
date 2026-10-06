"""marv-audio: MARV for speech models. Importing it registers the Whisper
decoder adapter, after which MARV's own tools (capture_writes,
decompose_logit, patch_sweep, suppress, ...) work on Whisper."""
from .whisper import PARTS, WhisperDecoderAdapter, language_logits, prompt_ids, whisper_inputs
from .listen import TokenSplit, listening_split

__all__ = [
    "PARTS",
    "WhisperDecoderAdapter",
    "language_logits",
    "prompt_ids",
    "whisper_inputs",
    "TokenSplit",
    "listening_split",
]
