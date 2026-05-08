from typing import Protocol

import numpy as np
from pydantic import BaseModel


class TimedWord(BaseModel):
    word: str
    start: float
    end: float


class TimedTranscript(BaseModel):
    text: str
    words: tuple[TimedWord, ...]
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


class TranscribesAudioSnapshots(Protocol):
    async def transcribe(
        self,
        audio_data: np.typing.NDArray[np.float32],
        *,
        model: str,
        language: str | None,
    ) -> str: ...

    async def transcribe_timed(
        self,
        audio_data: np.typing.NDArray[np.float32],
        *,
        model: str,
        language: str | None,
        prompt: str | None = None,
    ) -> TimedTranscript: ...
