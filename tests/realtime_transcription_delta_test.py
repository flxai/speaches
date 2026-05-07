from __future__ import annotations

import asyncio
from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

from speaches.realtime.context import SessionContext
from speaches.realtime.input_audio_buffer_event_router import commit_and_transcribe
from speaches.realtime.partial_transcription import RealtimePartialTranscriptionWorker
from speaches.realtime.pubsub import EventPubSub
from speaches.realtime.session import create_session_object_configuration
from speaches.realtime.stabilizer import RealtimeTranscriptStabilizer
from speaches.realtime.transcription_protocol import TimedTranscript, TimedWord
from speaches.types.realtime import (
    SERVER_EVENT_TYPES,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    server_event_type_adapter,
)


class FakeAudioSnapshotTranscriber:
    def __init__(self, *hypotheses: str) -> None:
        self.hypotheses = deque(hypotheses)
        self.calls: list[tuple[int, str, str | None]] = []
        self.prompts: list[str | None] = []
        self.calls_changed = asyncio.Event()

    async def transcribe(
        self,
        audio_data: np.typing.NDArray[np.float32],
        *,
        model: str,
        language: str | None,
    ) -> str:
        self.calls.append((len(audio_data), model, language))
        self.calls_changed.set()
        return self.hypotheses.popleft()

    async def transcribe_timed(
        self,
        audio_data: np.typing.NDArray[np.float32],
        *,
        model: str,
        language: str | None,
        prompt: str | None = None,
    ) -> TimedTranscript:
        self.calls.append((len(audio_data), model, language))
        self.prompts.append(prompt)
        self.calls_changed.set()
        return timed_transcript(self.hypotheses.popleft())


async def wait_for_calls(transcriber: FakeAudioSnapshotTranscriber, count: int) -> None:
    while len(transcriber.calls) < count:
        await asyncio.wait_for(transcriber.calls_changed.wait(), timeout=1)
        transcriber.calls_changed.clear()


def timed_transcript(text: str, word_duration: float = 0.01) -> TimedTranscript:
    return TimedTranscript(
        text=text,
        words=tuple(
            TimedWord(word=word, start=index * word_duration, end=(index + 1) * word_duration)
            for index, word in enumerate(text.split())
        ),
    )


def test_transcription_delta_event_is_a_server_event() -> None:
    event = ConversationItemInputAudioTranscriptionDeltaEvent(item_id="item_123", delta="hello")

    assert "conversation.item.input_audio_transcription.delta" in SERVER_EVENT_TYPES
    parsed = server_event_type_adapter.validate_python(event.model_dump())
    assert isinstance(parsed, ConversationItemInputAudioTranscriptionDeltaEvent)
    assert parsed.delta == "hello"


def test_stabilizer_does_not_commit_misheard_short_prefix() -> None:
    stabilizer = RealtimeTranscriptStabilizer()

    assert stabilizer.observe(timed_transcript("off")) is None
    assert stabilizer.observe(timed_transcript("the front fell off")) is None
    assert stabilizer.committed == ""


def test_stabilizer_keeps_current_right_edge_unconfirmed() -> None:
    stabilizer = RealtimeTranscriptStabilizer()

    assert stabilizer.observe(timed_transcript("thank you")) is None
    assert stabilizer.observe(timed_transcript("thank you")) is None

    delta = stabilizer.observe(timed_transcript("thank you today"))

    assert delta is not None
    assert delta.text == "thank you"


def test_stabilizer_emits_only_new_stable_word_prefixes() -> None:
    stabilizer = RealtimeTranscriptStabilizer()

    assert stabilizer.observe(timed_transcript("the front fell")) is None

    second_delta = stabilizer.observe(timed_transcript("the front fell off"))
    assert second_delta is not None
    assert second_delta.text == "the front fell"
    assert second_delta.confirmed_until_seconds == pytest.approx(0.03)

    third_delta = stabilizer.observe(timed_transcript("off again"))
    assert third_delta is not None
    assert third_delta.text == " off"
    assert third_delta.confirmed_until_seconds == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_partial_worker_publishes_stable_delta_after_audio_grows() -> None:
    pubsub = EventPubSub()
    subscriber = pubsub.subscribe()
    session = create_session_object_configuration("test-model", intent="transcription")
    transcriber = FakeAudioSnapshotTranscriber("the front fell", "the front fell off")
    input_audio_buffer = SessionContext(
        transcription_client=MagicMock(),
        completion_client=MagicMock(),
        executor_registry=MagicMock(),
        vad_model_manager=MagicMock(),
        session=session,
    ).audio_buffers.current
    input_audio_buffer.append(np.ones(800, dtype=np.float32))

    worker = RealtimePartialTranscriptionWorker(
        pubsub=pubsub,
        transcriber=transcriber,
        input_audio_buffer=input_audio_buffer,
        session=session,
        min_duration_ms=1,
        interval_seconds=0.01,
    )
    worker.start()
    await wait_for_calls(transcriber, 1)
    assert subscriber.empty()

    input_audio_buffer.append(np.ones(800, dtype=np.float32))
    event = await asyncio.wait_for(subscriber.get(), timeout=1)
    await worker.stop()

    assert isinstance(event, ConversationItemInputAudioTranscriptionDeltaEvent)
    assert event.item_id == input_audio_buffer.id
    assert event.delta == "the front fell"


@pytest.mark.asyncio
async def test_partial_worker_advances_audio_window_and_uses_prompt_context() -> None:
    pubsub = EventPubSub()
    session = create_session_object_configuration("test-model", intent="transcription")
    transcriber = FakeAudioSnapshotTranscriber("the front fell", "the front fell off", "off again")
    input_audio_buffer = SessionContext(
        transcription_client=MagicMock(),
        completion_client=MagicMock(),
        executor_registry=MagicMock(),
        vad_model_manager=MagicMock(),
        session=session,
    ).audio_buffers.current
    input_audio_buffer.append(np.ones(800, dtype=np.float32))

    worker = RealtimePartialTranscriptionWorker(
        pubsub=pubsub,
        transcriber=transcriber,
        input_audio_buffer=input_audio_buffer,
        session=session,
        min_duration_ms=1,
        interval_seconds=0.01,
    )
    worker.start()
    await wait_for_calls(transcriber, 1)
    input_audio_buffer.append(np.ones(800, dtype=np.float32))
    await wait_for_calls(transcriber, 2)
    input_audio_buffer.append(np.ones(800, dtype=np.float32))
    await wait_for_calls(transcriber, 3)
    await worker.stop()

    assert transcriber.calls == [
        (800, "test-model", None),
        (1600, "test-model", None),
        (1920, "test-model", None),
    ]
    assert transcriber.prompts == [None, None, "the front fell"]


@pytest.mark.asyncio
async def test_manual_commit_transcribes_full_buffer_without_vad_cut() -> None:
    session = create_session_object_configuration("test-model", intent="transcription")
    ctx = SessionContext(
        transcription_client=MagicMock(),
        completion_client=MagicMock(),
        executor_registry=MagicMock(),
        vad_model_manager=MagicMock(),
        session=session,
    )
    transcriber = FakeAudioSnapshotTranscriber("the front fell off")
    ctx.audio_transcriber = transcriber
    input_audio_buffer = ctx.audio_buffers.current
    input_audio_buffer.append(np.ones(16000, dtype=np.float32))
    input_audio_buffer.vad_state.audio_start_ms = 500
    input_audio_buffer.vad_state.audio_end_ms = 700

    await commit_and_transcribe(ctx, input_audio_buffer.id, apply_vad=False)

    completed_events = [
        event for event in ctx.pubsub.events if isinstance(event, ConversationItemInputAudioTranscriptionCompletedEvent)
    ]
    assert transcriber.calls == [(16000, "test-model", None)]
    assert completed_events[-1].transcript == "the front fell off"
