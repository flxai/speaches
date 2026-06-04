from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from speaches.diarization import DiarizationSegment  # noqa: TC001

if TYPE_CHECKING:
    import openai.types.audio


class DiarizedWord(BaseModel):
    start: float
    end: float
    word: str
    speaker: str = "UNKNOWN"


class DiarizedSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"
    words: list[DiarizedWord] = Field(default_factory=list)


class DiarizedTranscriptionResponse(BaseModel):
    language: str | None = None
    duration: float
    text: str
    segments: list[DiarizedSegment]
    words: list[DiarizedWord]
    diarization_segments: list[DiarizationSegment]


def word_overlap(word_start: float, word_end: float, segment_start: float, segment_end: float) -> float:
    return max(0.0, min(word_end, segment_end) - max(word_start, segment_start))


def speaker_for_word(word: DiarizedWord, diarization_segments: list[DiarizationSegment]) -> str:
    midpoint = (word.start + word.end) / 2
    covering = [
        segment for segment in diarization_segments if segment.start <= midpoint <= segment.end
    ]
    if len(covering) == 1:
        return covering[0].speaker
    if covering:
        return max(
            covering,
            key=lambda segment: word_overlap(word.start, word.end, segment.start, segment.end),
        ).speaker

    best_segment = None
    best_overlap = 0.0
    for segment in diarization_segments:
        overlap = word_overlap(word.start, word.end, segment.start, segment.end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_segment = segment
    return best_segment.speaker if best_segment is not None else "UNKNOWN"


def transcription_words(
    transcription: openai.types.audio.TranscriptionVerbose,
) -> list[DiarizedWord]:
    return [
        DiarizedWord(start=word.start, end=word.end, word=word.word)
        for word in (transcription.words or [])
    ]


def assign_word_speakers(
    words: list[DiarizedWord],
    diarization_segments: list[DiarizationSegment],
) -> list[DiarizedWord]:
    return [
        word.model_copy(update={"speaker": speaker_for_word(word, diarization_segments)})
        for word in words
    ]


def segment_words(segment: openai.types.audio.TranscriptionSegment, words: list[DiarizedWord]) -> list[DiarizedWord]:
    return [
        word
        for word in words
        if word.start >= segment.start - 0.01 and word.end <= segment.end + 0.01
    ]


def dominant_speaker(words: list[DiarizedWord]) -> str:
    if not words:
        return "UNKNOWN"
    durations: dict[str, float] = {}
    for word in words:
        durations[word.speaker] = durations.get(word.speaker, 0.0) + max(0.0, word.end - word.start)
    return max(durations, key=durations.get)


def build_diarized_transcription_response(
    transcription: openai.types.audio.TranscriptionVerbose,
    diarization_segments: list[DiarizationSegment],
) -> DiarizedTranscriptionResponse:
    words = assign_word_speakers(transcription_words(transcription), diarization_segments)
    segments = [
        DiarizedSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text.strip(),
            speaker=dominant_speaker(segment_words(segment, words)),
            words=segment_words(segment, words),
        )
        for segment in transcription.segments
    ]
    return DiarizedTranscriptionResponse(
        language=transcription.language,
        duration=transcription.duration,
        text=transcription.text,
        segments=segments,
        words=words,
        diarization_segments=diarization_segments,
    )
