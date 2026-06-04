from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from speaches.diarized_transcription import DiarizedTranscriptionResponse, DiarizedWord

DEFAULT_MAX_WORDS = 7
DEFAULT_MAX_GAP_SECONDS = 0.8


def ass_time(seconds: float) -> str:
    total_centis = max(0, round(float(seconds) * 100))
    hours, remainder = divmod(total_centis, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def ass_text(text: str) -> str:
    normalized = " ".join(str(text).replace("\r", "\n").split())
    escaped = normalized.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    return escaped.replace("\n", r"\N")


def ass_color(rgb: str) -> str:
    value = rgb.removeprefix("#")
    if len(value) != 6:
        raise ValueError(f"ASS colors must be #RRGGBB, got {rgb!r}")
    red = value[0:2]
    green = value[2:4]
    blue = value[4:6]
    return f"&H00{blue}{green}{red}"


def speaker_order(words: list[DiarizedWord]) -> list[str]:
    speakers = []
    for word in words:
        if word.speaker not in speakers:
            speakers.append(word.speaker)
    return speakers or ["UNKNOWN"]


def style_for_speaker(index: int) -> tuple[int, str]:
    if index == 0:
        return 2, "#ff5f5f"
    if index == 1:
        return 8, "#5fafff"
    return 2, "#ffffff"


def group_words(
    words: list[DiarizedWord],
    max_words: int = DEFAULT_MAX_WORDS,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
) -> list[list[DiarizedWord]]:
    groups: list[list[DiarizedWord]] = []
    current: list[DiarizedWord] = []
    for word in words:
        if (
            current
            and (
                word.speaker != current[-1].speaker
                or (max_words > 0 and len(current) >= max_words)
                or word.start - current[-1].end > max_gap_seconds
            )
        ):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def karaoke_text(words: list[DiarizedWord], event_end: float) -> str:
    chunks = []
    for index, word in enumerate(words):
        next_start = words[index + 1].start if index + 1 < len(words) else event_end
        duration_end = max(word.end, next_start)
        duration_cs = max(1, round((duration_end - word.start) * 100))
        prefix = "" if index == 0 else " "
        chunks.append(f"{prefix}{{\\K{duration_cs}}}{ass_text(word.word)}")
    return "".join(chunks)


def render_ass(
    response: DiarizedTranscriptionResponse,
    *,
    playres_x: int = 1920,
    playres_y: int = 1080,
    max_words: int = DEFAULT_MAX_WORDS,
) -> str:
    speakers = speaker_order(response.words)
    styles = []
    for index, speaker in enumerate(speakers):
        alignment, color = style_for_speaker(index)
        styles.append(
            f"Style: {ass_text(speaker)},Noto Sans,30,{ass_color(color)},&H000000FF,&H00101010,&H64000000,"
            f"0,0,0,0,100,100,0,0,1,2.2,0,{alignment},120,120,60,1"
        )

    header = f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: {playres_x}
PlayResY: {playres_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{chr(10).join(styles)}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    for group in group_words(response.words, max_words=max_words):
        event_start = group[0].start
        event_end = max(group[-1].end, event_start + 0.01)
        style = ass_text(group[0].speaker)
        events.append(
            f"Dialogue: 0,{ass_time(event_start)},{ass_time(event_end)},{style},,0,0,0,,{karaoke_text(group, event_end)}"
        )
    return header + "\n".join(events) + "\n"
