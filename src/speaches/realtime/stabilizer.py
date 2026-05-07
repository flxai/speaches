from __future__ import annotations

from dataclasses import dataclass
from string import punctuation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from speaches.realtime.transcription_protocol import TimedTranscript, TimedWord

WORD_BOUNDARY_CHARS = " \t\r\n.,!?;:"
PROMPT_CONTEXT_CHARS = 200
TIMESTAMP_EPSILON_SECONDS = 1e-6


@dataclass(frozen=True)
class RealtimeTranscriptDelta:
    text: str
    confirmed_until_seconds: float


class RealtimeTranscriptStabilizer:
    def __init__(self) -> None:
        self._previous_words: tuple[TimedWord, ...] | None = None
        self._committed = ""

    @property
    def committed(self) -> str:
        return self._committed

    @property
    def prompt_context(self) -> str | None:
        if not self._committed:
            return None
        return self._committed[-PROMPT_CONTEXT_CHARS:]

    def observe(self, hypothesis: TimedTranscript) -> RealtimeTranscriptDelta | None:
        current_words = tuple(word for word in hypothesis.words if normalize_word(word.word))
        if not current_words:
            return None

        if self._previous_words is None:
            self._previous_words = current_words
            return None

        stable_words = common_word_prefix(self._previous_words, current_words)
        if not stable_words:
            self._previous_words = current_words
            return None

        if len(stable_words) == len(current_words):
            self._previous_words = current_words
            return None

        stable_text = append_text(self._committed, format_words(stable_words))
        if len(stable_text) <= len(self._committed):
            self._previous_words = current_words
            return None

        delta = stable_text[len(self._committed) :]
        self._committed = stable_text
        confirmed_until_seconds = stable_words[-1].end
        self._previous_words = rebase_words_after(current_words, confirmed_until_seconds)
        return RealtimeTranscriptDelta(text=delta, confirmed_until_seconds=confirmed_until_seconds)


def common_word_prefix(left: Sequence[TimedWord], right: Sequence[TimedWord]) -> tuple[TimedWord, ...]:
    stable_words = []
    for left_word, right_word in zip(left, right, strict=False):
        if normalize_word(left_word.word) != normalize_word(right_word.word):
            break
        stable_words.append(right_word)
    return tuple(stable_words)


def normalize_word(word: str) -> str:
    stripped = word.strip()
    normalized = stripped.strip(punctuation).casefold()
    return normalized or stripped.casefold()


def format_words(words: Sequence[TimedWord]) -> str:
    text = ""
    for word in words:
        text = append_text(text, word.word.strip())
    return text


def append_text(prefix: str, suffix: str) -> str:
    if not suffix:
        return prefix
    if not prefix:
        return suffix
    if prefix[-1].isspace() or suffix[0].isspace() or suffix[0] in WORD_BOUNDARY_CHARS.strip():
        return prefix + suffix
    return f"{prefix} {suffix}"


def rebase_words_after(words: Sequence[TimedWord], offset_seconds: float) -> tuple[TimedWord, ...]:
    from speaches.realtime.transcription_protocol import TimedWord

    return tuple(
        TimedWord(
            word=word.word,
            start=max(0.0, word.start - offset_seconds),
            end=max(0.0, word.end - offset_seconds),
        )
        for word in words
        if word.end > offset_seconds + TIMESTAMP_EPSILON_SECONDS
    )
