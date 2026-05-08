from __future__ import annotations

import asyncio
from collections import deque
from typing import TypeAlias
from unittest.mock import MagicMock

import numpy as np
import pytest

from speaches.realtime.context import SessionContext
from speaches.realtime.input_audio_buffer_event_router import commit_and_transcribe
from speaches.realtime.partial_transcription import RealtimePartialTranscriptionWorker, trim_timed_transcript_before
from speaches.realtime.pubsub import EventPubSub
from speaches.realtime.session import create_session_object_configuration
from speaches.realtime.stabilizer import RealtimeTranscriptStabilizer
from speaches.realtime.transcription_protocol import TimedTranscript, TimedWord
from speaches.types.realtime import (
    SERVER_EVENT_TYPES,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ConversationItemInputAudioTranscriptionHypothesisEvent,
    server_event_type_adapter,
)


FakeHypothesis: TypeAlias = str | TimedTranscript


class FakeAudioSnapshotTranscriber:
    def __init__(self, *hypotheses: FakeHypothesis) -> None:
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
        hypothesis = self.hypotheses.popleft()
        if isinstance(hypothesis, TimedTranscript):
            return hypothesis
        return timed_transcript(hypothesis)


async def wait_for_calls(transcriber: FakeAudioSnapshotTranscriber, count: int) -> None:
    while len(transcriber.calls) < count:
        await asyncio.wait_for(transcriber.calls_changed.wait(), timeout=1)
        transcriber.calls_changed.clear()


def timed_transcript(
    text: str,
    word_duration: float = 0.1,
    *,
    no_speech_prob: float | None = None,
    avg_logprob: float | None = None,
) -> TimedTranscript:
    return TimedTranscript(
        text=text,
        words=tuple(
            TimedWord(word=word, start=index * word_duration, end=(index + 1) * word_duration)
            for index, word in enumerate(text.split())
        ),
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
    )


def test_transcription_delta_event_is_a_server_event() -> None:
    event = ConversationItemInputAudioTranscriptionDeltaEvent(item_id="item_123", delta="hello")

    assert "conversation.item.input_audio_transcription.delta" in SERVER_EVENT_TYPES
    parsed = server_event_type_adapter.validate_python(event.model_dump())
    assert isinstance(parsed, ConversationItemInputAudioTranscriptionDeltaEvent)
    assert parsed.delta == "hello"


def test_transcription_hypothesis_event_is_a_server_event() -> None:
    event = ConversationItemInputAudioTranscriptionHypothesisEvent(
        item_id="item_123",
        transcript="the front fell",
        confirmed_prefix="the front",
        provisional="fell",
        audio_start_ms=0,
        audio_end_ms=1500,
    )

    assert "conversation.item.input_audio_transcription.hypothesis" in SERVER_EVENT_TYPES
    parsed = server_event_type_adapter.validate_python(event.model_dump())
    assert isinstance(parsed, ConversationItemInputAudioTranscriptionHypothesisEvent)
    assert parsed.transcript == "the front fell"
    assert parsed.confirmed_prefix == "the front"
    assert parsed.provisional == "fell"


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


def test_stabilizer_prompt_context_uses_last_confirmed_words() -> None:
    stabilizer = RealtimeTranscriptStabilizer()
    words = [f"word{i}" for i in range(205)]

    stabilizer.observe(timed_transcript(" ".join(words)))
    delta = stabilizer.observe(timed_transcript(" ".join([*words, "tail"])))

    assert delta is not None
    assert stabilizer.prompt_context == " ".join(f"word{i}" for i in range(5, 205))


def test_stabilizer_emits_only_new_stable_word_prefixes() -> None:
    stabilizer = RealtimeTranscriptStabilizer()

    assert stabilizer.observe(timed_transcript("the front fell")) is None

    second_delta = stabilizer.observe(timed_transcript("the front fell off"))
    assert second_delta is not None
    assert second_delta.text == "the front fell"
    assert second_delta.confirmed_until_seconds == pytest.approx(0.3)

    third_delta = stabilizer.observe(timed_transcript("off again"))
    assert third_delta is not None
    assert third_delta.text == " off"
    assert third_delta.confirmed_until_seconds == pytest.approx(0.1)


def test_transcription_only_sessions_do_not_enable_server_vad() -> None:
    session = create_session_object_configuration("test-model", intent="transcription")

    assert session.turn_detection is None


def test_partial_snapshot_trim_tolerates_timestamp_drift_near_confirmed_boundary() -> None:
    hypothesis = TimedTranscript(
        text="the front fell off",
        words=(
            TimedWord(word="the", start=0.0, end=0.2),
            TimedWord(word="front", start=0.2, end=0.48),
            TimedWord(word="fell", start=0.49, end=0.75),
            TimedWord(word="off", start=0.75, end=1.0),
        ),
    )

    trimmed = trim_timed_transcript_before(hypothesis, 0.5)

    assert [word.word for word in trimmed.words] == ["fell", "off"]
    assert trimmed.words[0].start == pytest.approx(0.0)
    assert trimmed.text == "fell off"


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
    first_event = await asyncio.wait_for(subscriber.get(), timeout=1)
    assert isinstance(first_event, ConversationItemInputAudioTranscriptionHypothesisEvent)
    assert first_event.transcript == "the front fell"

    input_audio_buffer.append(np.ones(800, dtype=np.float32))
    delta_event = await asyncio.wait_for(subscriber.get(), timeout=1)
    hypothesis_event = await asyncio.wait_for(subscriber.get(), timeout=1)
    await worker.stop()

    assert isinstance(delta_event, ConversationItemInputAudioTranscriptionDeltaEvent)
    assert delta_event.item_id == input_audio_buffer.id
    assert delta_event.delta == "the front fell"
    assert isinstance(hypothesis_event, ConversationItemInputAudioTranscriptionHypothesisEvent)
    assert hypothesis_event.item_id == input_audio_buffer.id
    assert hypothesis_event.confirmed_prefix == "the front fell"
    assert hypothesis_event.provisional == "off"
    assert hypothesis_event.transcript == "the front fell off"


@pytest.mark.asyncio
async def test_partial_worker_hypothesis_can_retract_previous_text() -> None:
    pubsub = EventPubSub()
    subscriber = pubsub.subscribe()
    session = create_session_object_configuration("test-model", intent="transcription")
    transcriber = FakeAudioSnapshotTranscriber("hello world", "the front fell")
    input_audio_buffer = SessionContext(
        transcription_client=MagicMock(),
        completion_client=MagicMock(),
        executor_registry=MagicMock(),
        vad_model_manager=MagicMock(),
        session=session,
    ).audio_buffers.current
    input_audio_buffer.append(np.ones(1600, dtype=np.float32))

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
    first_event = await asyncio.wait_for(subscriber.get(), timeout=1)
    input_audio_buffer.append(np.ones(800, dtype=np.float32))
    await wait_for_calls(transcriber, 2)
    second_event = await asyncio.wait_for(subscriber.get(), timeout=1)
    await worker.stop()

    assert isinstance(first_event, ConversationItemInputAudioTranscriptionHypothesisEvent)
    assert first_event.transcript == "hello world"
    assert isinstance(second_event, ConversationItemInputAudioTranscriptionHypothesisEvent)
    assert second_event.transcript == "the front fell"


@pytest.mark.asyncio
async def test_partial_worker_drops_short_silence_hallucination() -> None:
    pubsub = EventPubSub()
    subscriber = pubsub.subscribe()
    session = create_session_object_configuration("test-model", intent="transcription")
    transcriber = FakeAudioSnapshotTranscriber("hello")
    input_audio_buffer = SessionContext(
        transcription_client=MagicMock(),
        completion_client=MagicMock(),
        executor_registry=MagicMock(),
        vad_model_manager=MagicMock(),
        session=session,
    ).audio_buffers.current
    input_audio_buffer.append(np.ones(1600, dtype=np.float32))

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
    await worker.stop()

    assert subscriber.empty()


@pytest.mark.asyncio
async def test_partial_worker_drops_no_speech_probability_hallucination() -> None:
    pubsub = EventPubSub()
    subscriber = pubsub.subscribe()
    session = create_session_object_configuration("test-model", intent="transcription")
    transcriber = FakeAudioSnapshotTranscriber(
        timed_transcript("Thank you", no_speech_prob=0.92, avg_logprob=-1.4)
    )
    input_audio_buffer = SessionContext(
        transcription_client=MagicMock(),
        completion_client=MagicMock(),
        executor_registry=MagicMock(),
        vad_model_manager=MagicMock(),
        session=session,
    ).audio_buffers.current
    input_audio_buffer.append(np.ones(1600, dtype=np.float32))

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
    await worker.stop()

    assert subscriber.empty()


@pytest.mark.asyncio
async def test_partial_worker_keeps_high_confidence_speech_despite_no_speech_probability() -> None:
    pubsub = EventPubSub()
    subscriber = pubsub.subscribe()
    session = create_session_object_configuration("test-model", intent="transcription")
    transcriber = FakeAudioSnapshotTranscriber(
        timed_transcript("the front fell", no_speech_prob=0.7, avg_logprob=-0.2)
    )
    input_audio_buffer = SessionContext(
        transcription_client=MagicMock(),
        completion_client=MagicMock(),
        executor_registry=MagicMock(),
        vad_model_manager=MagicMock(),
        session=session,
    ).audio_buffers.current
    input_audio_buffer.append(np.ones(1600, dtype=np.float32))

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
    event = await asyncio.wait_for(subscriber.get(), timeout=1)
    await worker.stop()

    assert isinstance(event, ConversationItemInputAudioTranscriptionHypothesisEvent)
    assert event.transcript == "the front fell"


@pytest.mark.asyncio
async def test_partial_worker_does_not_erase_existing_text_with_hallucination() -> None:
    pubsub = EventPubSub()
    subscriber = pubsub.subscribe()
    session = create_session_object_configuration("test-model", intent="transcription")
    transcriber = FakeAudioSnapshotTranscriber("the front fell", "thank you")
    input_audio_buffer = SessionContext(
        transcription_client=MagicMock(),
        completion_client=MagicMock(),
        executor_registry=MagicMock(),
        vad_model_manager=MagicMock(),
        session=session,
    ).audio_buffers.current
    input_audio_buffer.append(np.ones(1600, dtype=np.float32))

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
    first_event = await asyncio.wait_for(subscriber.get(), timeout=1)
    input_audio_buffer.append(np.ones(800, dtype=np.float32))
    await wait_for_calls(transcriber, 2)
    await asyncio.sleep(0.03)
    await worker.stop()

    assert isinstance(first_event, ConversationItemInputAudioTranscriptionHypothesisEvent)
    assert first_event.transcript == "the front fell"
    assert subscriber.empty()


@pytest.mark.asyncio
async def test_partial_worker_uses_overlap_and_prompt_context_after_confirmation() -> None:
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
        (2400, "test-model", None),
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
