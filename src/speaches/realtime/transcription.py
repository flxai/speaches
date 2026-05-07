from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import openai.types.audio

from speaches.api_types import DEFAULT_TIMESTAMP_GRANULARITIES, TimestampGranularities
from speaches.audio import Audio
from speaches.executors.shared.handler_protocol import (
    NonStreamingTranscriptionResponse,
    TranscriptionRequest,
)
from speaches.executors.silero_vad_v5 import VadOptions
from speaches.realtime.transcription_protocol import TimedTranscript, TimedWord
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

    async def transcribe_timed(
        self,
        audio_data: np.typing.NDArray[np.float32],
        *,
        model: str,
        language: str | None,
        prompt: str | None = None,
    ) -> TimedTranscript:
        if len(audio_data) == 0:
            return TimedTranscript(text="", words=())
        snapshot = audio_data.copy()
        return await asyncio.to_thread(self._transcribe_timed_blocking, snapshot, model, language, prompt)

    def _transcribe_blocking(self, audio_data: np.typing.NDArray[np.float32], model: str, language: str | None) -> str:
        request = self._transcription_request(
            audio_data=audio_data,
            model=model,
            language=language,
            response_format="text",
            timestamp_granularities=DEFAULT_TIMESTAMP_GRANULARITIES,
            without_timestamps=True,
        )
        response = self._handle_non_streaming_transcription_request(model, request)
        return transcription_response_to_text(response)

    def _transcribe_timed_blocking(
        self,
        audio_data: np.typing.NDArray[np.float32],
        model: str,
        language: str | None,
        prompt: str | None,
    ) -> TimedTranscript:
        request = self._transcription_request(
            audio_data=audio_data,
            model=model,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            without_timestamps=False,
            prompt=prompt,
        )
        response = self._handle_non_streaming_transcription_request(model, request)
        return transcription_response_to_timed_transcript(response)

    def _transcription_request(
        self,
        *,
        audio_data: np.typing.NDArray[np.float32],
        model: str,
        language: str | None,
        response_format: openai.types.AudioResponseFormat,
        timestamp_granularities: TimestampGranularities,
        without_timestamps: bool,
        prompt: str | None = None,
    ) -> TranscriptionRequest:
        audio = Audio(audio_data, sample_rate=16000)
        return TranscriptionRequest(
            audio=audio,
            model=model,
            language=language,
            response_format=response_format,
            temperature=0.0,
            timestamp_granularities=timestamp_granularities,
            stream=False,
            prompt=prompt,
            hotwords=None,
            speech_segments=[],
            vad_options=REALTIME_TRANSCRIPTION_VAD_OPTIONS,
            without_timestamps=without_timestamps,
        )

    def _handle_non_streaming_transcription_request(
        self, model: str, request: TranscriptionRequest
    ) -> NonStreamingTranscriptionResponse:
        model_card_data = get_model_card_data_or_raise(model)
        transcription_executor = find_executor_for_model_or_raise(
            model, model_card_data, self.executor_registry.transcription
        )
        response = transcription_executor.model_manager.handle_non_streaming_transcription_request(request)
        return response


def transcription_response_to_text(response: NonStreamingTranscriptionResponse) -> str:
    if isinstance(response, tuple):
        return response[0]
    if isinstance(response, (openai.types.audio.Transcription, openai.types.audio.TranscriptionVerbose)):
        return response.text
    raise TypeError(f"Unexpected transcription response type: {type(response)}")


def transcription_response_to_timed_transcript(response: NonStreamingTranscriptionResponse) -> TimedTranscript:
    if not isinstance(response, openai.types.audio.TranscriptionVerbose):
        raise TypeError(f"Expected verbose transcription response, got {type(response)}")
    return TimedTranscript(
        text=response.text,
        words=tuple(
            TimedWord(word=word.word, start=word.start, end=word.end) for word in (response.words or [])
        ),
    )
