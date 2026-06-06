"""WhisperX-backed diarized transcription.

Runs the full WhisperX pipeline (transcribe -> wav2vec2 align -> diarize ->
word-speaker assignment) and maps the result onto the existing
``DiarizedTranscriptionResponse`` so the ASS / ``diarized_json`` serializers are
reused unchanged.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

import numpy as np

from speaches.executors.shared.base_model_manager import BaseModelManager

if TYPE_CHECKING:
    from speaches.audio import Audio
    from speaches.config import WhisperXConfig
    from speaches.diarized_transcription import DiarizedTranscriptionResponse

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _resolve_device(device: str) -> str:
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _hf_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(var)
        if token:
            return token
    return None


class WhisperXModelManager(BaseModelManager):  # the cached value is a whisperx FasterWhisperPipeline
    """Caches the WhisperX ASR pipeline (via the TTL base manager) plus the
    per-language alignment models and per-id diarization pipelines."""

    def __init__(self, ttl: int, whisperx_config: WhisperXConfig) -> None:
        super().__init__(ttl)
        self.config = whisperx_config
        self._align_lock = threading.Lock()
        self._align_models: dict[str, tuple[object, object]] = {}
        self._diarize_lock = threading.Lock()
        self._diarize_pipelines: dict[str, object] = {}

    def _load_fn(self, model_id: str):  # noqa: ANN202 - returns a whisperx FasterWhisperPipeline
        import whisperx

        device = _resolve_device(self.config.inference_device)
        logger.info(f"Loading WhisperX ASR model {model_id} on {device}")
        return whisperx.load_model(
            model_id,
            device=device,
            compute_type=self.config.compute_type,
            threads=self.config.cpu_threads or 4,
            use_auth_token=_hf_token(),
        )

    def _align_model(self, language: str, device: str) -> tuple[object, object]:
        with self._align_lock:
            cached = self._align_models.get(language)
            if cached is None:
                import whisperx

                logger.info(f"Loading WhisperX alignment model for language {language!r}")
                cached = whisperx.load_align_model(
                    language_code=language,
                    device=device,
                    model_name=self.config.align_model,
                )
                self._align_models[language] = cached
            return cached

    def _diarize_pipeline(self, diarization_model: str, device: str) -> object:
        with self._diarize_lock:
            pipeline = self._diarize_pipelines.get(diarization_model)
            if pipeline is None:
                from whisperx.diarize import DiarizationPipeline

                logger.info(f"Loading WhisperX diarization pipeline {diarization_model}")
                pipeline = DiarizationPipeline(
                    model_name=diarization_model,
                    token=_hf_token(),
                    device=device,
                )
                self._diarize_pipelines[diarization_model] = pipeline
            return pipeline

    def transcribe_diarized(  # noqa: PLR0913
        self,
        audio: Audio,
        model_id: str,
        diarization_model: str,
        language: str | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> DiarizedTranscriptionResponse:
        import whisperx

        from speaches.diarization import DiarizationSegment
        from speaches.diarized_transcription import (
            DiarizedSegment,
            DiarizedTranscriptionResponse,
            DiarizedWord,
            dominant_speaker,
        )

        device = _resolve_device(self.config.inference_device)
        samples = np.ascontiguousarray(audio.data, dtype=np.float32)

        with self.load_model(model_id) as model:
            transcription = model.transcribe(samples, batch_size=self.config.batch_size, language=language)

        detected_language = transcription.get("language") or language or "en"
        align_model, metadata = self._align_model(detected_language, device)
        aligned = whisperx.align(
            transcription["segments"],
            align_model,
            metadata,
            samples,
            device,
            return_char_alignments=False,
        )

        diarize_kwargs: dict[str, int] = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers
        diarize_df = self._diarize_pipeline(diarization_model, device)(samples, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_df, aligned)

        words: list[DiarizedWord] = []
        segments: list[DiarizedSegment] = []
        for segment in result.get("segments", []):
            segment_words: list[DiarizedWord] = []
            for word in segment.get("words", []):
                start = word.get("start")
                end = word.get("end")
                if start is None or end is None:
                    continue
                diarized_word = DiarizedWord(
                    start=float(start),
                    end=float(end),
                    word=word.get("word", ""),
                    speaker=word.get("speaker", "UNKNOWN"),
                )
                segment_words.append(diarized_word)
                words.append(diarized_word)
            segments.append(
                DiarizedSegment(
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    text=segment.get("text", "").strip(),
                    speaker=segment.get("speaker") or dominant_speaker(segment_words),
                    words=segment_words,
                )
            )

        diarization_segments = [
            DiarizationSegment(
                start=float(row["start"]),
                end=float(row["end"]),
                speaker=str(row["speaker"]),
            )
            for _, row in diarize_df.iterrows()
        ]

        return DiarizedTranscriptionResponse(
            language=detected_language,
            duration=float(len(samples)) / SAMPLE_RATE,
            text="".join(segment.get("text", "") for segment in result.get("segments", [])).strip(),
            segments=segments,
            words=words,
            diarization_segments=diarization_segments,
        )
