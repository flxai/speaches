from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
from typing import TYPE_CHECKING

from speaches.realtime.input_audio_buffer import MS_SAMPLE_RATE, SAMPLE_RATE
from speaches.realtime.stabilizer import (
    TIMESTAMP_EPSILON_SECONDS,
    RealtimeTranscriptStabilizer,
    append_text,
    format_words,
)
from speaches.realtime.utils import task_done_callback
from speaches.types.realtime import (
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ConversationItemInputAudioTranscriptionHypothesisEvent,
)

if TYPE_CHECKING:
    import numpy as np

    from speaches.realtime.input_audio_buffer import InputAudioBuffer
    from speaches.realtime.pubsub import EventPubSub
    from speaches.realtime.transcription_protocol import TimedTranscript, TimedWord, TranscribesAudioSnapshots
    from speaches.types.realtime import Session

logger = logging.getLogger(__name__)

REALTIME_PARTIAL_MIN_DURATION_MS = 1500
REALTIME_PARTIAL_INTERVAL_SECONDS = 0.5
REALTIME_PARTIAL_HALLUCINATION_MAX_DURATION_MS = 3000
SILENCE_HALLUCINATION_PHRASES = {
    "hello",
    "hi",
    "thank you",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "you",
}


class RealtimePartialTranscriptionWorker:
    def __init__(
        self,
        *,
        pubsub: EventPubSub,
        transcriber: TranscribesAudioSnapshots,
        input_audio_buffer: InputAudioBuffer,
        session: Session,
        min_duration_ms: int = REALTIME_PARTIAL_MIN_DURATION_MS,
        interval_seconds: float = REALTIME_PARTIAL_INTERVAL_SECONDS,
    ) -> None:
        self.pubsub = pubsub
        self.transcriber = transcriber
        self.input_audio_buffer = input_audio_buffer
        self.session = session
        self.min_duration_ms = min_duration_ms
        self.interval_seconds = interval_seconds
        self.stabilizer = RealtimeTranscriptStabilizer()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._last_submitted_size = 0
        self._confirmed_samples = 0
        self._last_published_hypothesis: str | None = None

    @property
    def item_id(self) -> str:
        return self.input_audio_buffer.id

    def start(self) -> None:
        assert self._task is None
        self._task = asyncio.create_task(self._run(), name=f"realtime-partials-{self.item_id}")
        self._task.add_done_callback(task_done_callback)

    async def stop(self) -> None:
        self._stop.set()
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                continue
            except TimeoutError:
                pass

            unconfirmed_samples = self.input_audio_buffer.size - self._confirmed_samples
            if unconfirmed_samples <= 0:
                continue
            if unconfirmed_samples // MS_SAMPLE_RATE < self.min_duration_ms:
                continue
            if self.input_audio_buffer.size == self._last_submitted_size:
                continue

            snapshot = self.input_audio_buffer.data[self._confirmed_samples :].copy()
            self._last_submitted_size = self.input_audio_buffer.size
            await self._transcribe_snapshot(snapshot)

    async def _transcribe_snapshot(self, snapshot: np.typing.NDArray[np.float32]) -> None:
        try:
            hypothesis = await self.transcriber.transcribe_timed(
                snapshot,
                model=self.session.input_audio_transcription.model,
                language=self.session.input_audio_transcription.language,
                prompt=self.stabilizer.prompt_context,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Realtime partial transcription failed")
            return

        if should_drop_partial_hypothesis(hypothesis, snapshot_duration_ms(snapshot)):
            self._publish_hypothesis(provisional="")
            return

        confirmed_until_seconds = 0.0
        delta = self.stabilizer.observe(hypothesis)
        if delta is not None:
            confirmed_until_seconds = delta.confirmed_until_seconds
            confirmed_samples = int(delta.confirmed_until_seconds * SAMPLE_RATE)
            self._confirmed_samples = min(self.input_audio_buffer.size, self._confirmed_samples + confirmed_samples)
            self.pubsub.publish_nowait(
                ConversationItemInputAudioTranscriptionDeltaEvent(item_id=self.item_id, delta=delta.text)
            )
        provisional = format_words(words_after(hypothesis.words, confirmed_until_seconds))
        self._publish_hypothesis(provisional=provisional)

    def _publish_hypothesis(self, *, provisional: str) -> None:
        confirmed_prefix = self.stabilizer.committed
        transcript = append_text(confirmed_prefix, provisional)
        if not transcript and self._last_published_hypothesis is None:
            return
        if transcript == self._last_published_hypothesis:
            return
        self._last_published_hypothesis = transcript
        self.pubsub.publish_nowait(
            ConversationItemInputAudioTranscriptionHypothesisEvent(
                item_id=self.item_id,
                transcript=transcript,
                confirmed_prefix=confirmed_prefix,
                provisional=provisional,
                audio_start_ms=0,
                audio_end_ms=self.input_audio_buffer.duration_ms,
            )
        )


class RealtimePartialTranscriptionManager:
    def __init__(self, transcriber: TranscribesAudioSnapshots) -> None:
        self.transcriber = transcriber
        self._workers: dict[str, RealtimePartialTranscriptionWorker] = {}

    def ensure_started(
        self,
        *,
        pubsub: EventPubSub,
        input_audio_buffer: InputAudioBuffer,
        session: Session,
    ) -> None:
        if input_audio_buffer.id in self._workers:
            return
        worker = RealtimePartialTranscriptionWorker(
            pubsub=pubsub,
            transcriber=self.transcriber,
            input_audio_buffer=input_audio_buffer,
            session=session,
        )
        self._workers[input_audio_buffer.id] = worker
        worker.start()

    async def stop(self, item_id: str) -> None:
        worker = self._workers.pop(item_id, None)
        if worker is not None:
            await worker.stop()

    async def stop_all(self) -> None:
        item_ids = list(self._workers)
        for item_id in item_ids:
            await self.stop(item_id)


def words_after(words: tuple[TimedWord, ...], offset_seconds: float) -> tuple[TimedWord, ...]:
    return tuple(word for word in words if word.end > offset_seconds + TIMESTAMP_EPSILON_SECONDS)


def snapshot_duration_ms(snapshot: np.typing.NDArray[np.float32]) -> int:
    return len(snapshot) // MS_SAMPLE_RATE


def should_drop_partial_hypothesis(hypothesis: TimedTranscript, duration_ms: int) -> bool:
    if not hypothesis.words:
        return True
    if duration_ms > REALTIME_PARTIAL_HALLUCINATION_MAX_DURATION_MS:
        return False
    return normalized_phrase(hypothesis.text) in SILENCE_HALLUCINATION_PHRASES


def normalized_phrase(text: str) -> str:
    return " ".join(word.strip(".,!?;:").casefold() for word in text.split()).strip()
