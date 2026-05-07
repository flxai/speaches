from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import openai.types.audio

from speaches.api_types import DEFAULT_TIMESTAMP_GRANULARITIES
from speaches.audio import Audio
from speaches.executors.shared.handler_protocol import (
    NonStreamingTranscriptionResponse,
    TranscriptionRequest,
)
from speaches.executors.silero_vad_v5 import VadOptions
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise

if TYPE_CHECKING:
    import numpy as np

    from speaches.executors.shared.registry import ExecutorRegistry

REALTIME_TRANSCRIPTION_VAD_OPTIONS = VadOptions(min_silence_duration_ms=160, max_speech_duration_s=30)


class AudioSnapshotTranscriber:
    def __init__(self, executor_registry: ExecutorRegistry) -> None:
        self.executor_registry = executor_registry

    async def transcribe(
        self,
        audio_data: np.typing.NDArray[np.float32],
        *,
        model: str,
        language: str | None,
    ) -> str:
        if len(audio_data) == 0:
            return ""
        snapshot = audio_data.copy()
        return await asyncio.to_thread(self._transcribe_blocking, snapshot, model, language)

    def _transcribe_blocking(self, audio_data: np.typing.NDArray[np.float32], model: str, language: str | None) -> str:
        audio = Audio(audio_data, sample_rate=16000)
        model_card_data = get_model_card_data_or_raise(model)
        transcription_executor = find_executor_for_model_or_raise(
            model, model_card_data, self.executor_registry.transcription
        )

        request = TranscriptionRequest(
            audio=audio,
            model=model,
            language=language,
            response_format="text",
            temperature=0.0,
            timestamp_granularities=DEFAULT_TIMESTAMP_GRANULARITIES,
            stream=False,
            hotwords=None,
            speech_segments=[],
            vad_options=REALTIME_TRANSCRIPTION_VAD_OPTIONS,
            without_timestamps=True,
        )
        response = transcription_executor.model_manager.handle_non_streaming_transcription_request(request)
        return transcription_response_to_text(response)


def transcription_response_to_text(response: NonStreamingTranscriptionResponse) -> str:
    if isinstance(response, tuple):
        return response[0]
    if isinstance(response, (openai.types.audio.Transcription, openai.types.audio.TranscriptionVerbose)):
        return response.text
    raise TypeError(f"Unexpected transcription response type: {type(response)}")
