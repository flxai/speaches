WORD_BOUNDARY_CHARS = " \t\r\n.,!?;:"


class RealtimeTranscriptStabilizer:
    def __init__(self) -> None:
        self._previous_hypothesis: str | None = None
        self._committed = ""

    @property
    def committed(self) -> str:
        return self._committed

    def observe(self, hypothesis: str) -> str | None:
        normalized = normalize_hypothesis(hypothesis)
        if not normalized:
            return None

        if self._previous_hypothesis is None:
            self._previous_hypothesis = normalized
            return None

        common_prefix = longest_common_prefix(self._previous_hypothesis, normalized)
        self._previous_hypothesis = normalized

        stable_prefix = trim_to_word_boundary(common_prefix, normalized)
        if not stable_prefix.startswith(self._committed):
            return None
        if len(stable_prefix) <= len(self._committed):
            return None

        delta = stable_prefix[len(self._committed) :]
        self._committed = stable_prefix
        return delta


def normalize_hypothesis(text: str) -> str:
    return " ".join(text.split())


def longest_common_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


def trim_to_word_boundary(prefix: str, hypothesis: str) -> str:
    boundary = len(prefix)
    while boundary > 0 and not is_word_boundary(hypothesis, boundary):
        boundary -= 1
    return hypothesis[:boundary]


def is_word_boundary(text: str, index: int) -> bool:
    if index <= 0 or index > len(text):
        return False
    if index == len(text):
        return text[index - 1] in WORD_BOUNDARY_CHARS
    return text[index] in WORD_BOUNDARY_CHARS or text[index - 1] in WORD_BOUNDARY_CHARS
