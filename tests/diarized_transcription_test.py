from httpx import AsyncClient
import openai.types.audio
import pytest

from speaches.diarization import DiarizationSegment
from speaches.diarized_transcription import (
    DiarizedWord,
    assign_word_speakers,
    build_diarized_transcription_response,
)
from speaches.subtitles.ass import ass_color, ass_text, render_ass


def test_assign_word_speakers_uses_midpoint() -> None:
    words = [
        DiarizedWord(start=1.0, end=1.4, word="Hello"),
        DiarizedWord(start=2.0, end=2.4, word="there"),
    ]
    diarization_segments = [
        DiarizationSegment(start=0.5, end=1.5, speaker="SPEAKER_00"),
        DiarizationSegment(start=1.8, end=2.8, speaker="SPEAKER_01"),
    ]

    assigned = assign_word_speakers(words, diarization_segments)

    assert [word.speaker for word in assigned] == ["SPEAKER_00", "SPEAKER_01"]


def test_assign_word_speakers_chooses_largest_overlap_when_speakers_overlap() -> None:
    words = [DiarizedWord(start=1.0, end=2.0, word="interrupting")]
    diarization_segments = [
        DiarizationSegment(start=0.0, end=1.55, speaker="SPEAKER_00"),
        DiarizationSegment(start=1.45, end=2.4, speaker="SPEAKER_01"),
    ]

    assigned = assign_word_speakers(words, diarization_segments)

    assert assigned[0].speaker == "SPEAKER_00"


def test_assign_word_speakers_marks_gaps_unknown() -> None:
    words = [DiarizedWord(start=5.0, end=5.2, word="gap")]

    assigned = assign_word_speakers(words, [])

    assert assigned[0].speaker == "UNKNOWN"


def test_render_ass_uses_vertical_speaker_styles_and_escapes_text() -> None:
    response = build_diarized_transcription_response(
        openai.types.audio.TranscriptionVerbose(
            language="en",
            duration=3.0,
            text="Hello {world} No",
            segments=[
                openai.types.audio.TranscriptionSegment(
                    id=0,
                    seek=0,
                    start=0.0,
                    end=3.0,
                    text="Hello {world} No",
                    tokens=[],
                    temperature=0.0,
                    avg_logprob=0.0,
                    compression_ratio=0.0,
                    no_speech_prob=0.0,
                )
            ],
            words=[
                openai.types.audio.TranscriptionWord(start=0.0, end=0.5, word="Hello"),
                openai.types.audio.TranscriptionWord(start=0.5, end=1.0, word="{world}"),
                openai.types.audio.TranscriptionWord(start=2.0, end=2.4, word="No"),
            ],
        ),
        [
            DiarizationSegment(start=0.0, end=1.2, speaker="SPEAKER_00"),
            DiarizationSegment(start=1.8, end=2.6, speaker="SPEAKER_01"),
        ],
    )

    ass = render_ass(response, max_words=7)

    assert "Style: SPEAKER_00" in ass
    assert "Style: SPEAKER_01" in ass
    assert ",2,120,120,60,1" in ass
    assert ",8,120,120,60,1" in ass
    assert r"\{world\}" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,SPEAKER_00" in ass
    assert "Dialogue: 0,0:00:02.00,0:00:02.40,SPEAKER_01" in ass


def test_ass_helpers() -> None:
    assert ass_color("#ff5f5f") == "&H005f5fff"
    assert ass_text(r"a {b} \ c") == r"a \{b\} \\ c"


@pytest.mark.asyncio
async def test_diarized_transcription_endpoint_is_registered(aclient: AsyncClient) -> None:
    response = await aclient.get("/openapi.json")

    assert response.status_code == 200
    assert "/v1/audio/diarized-transcriptions" in response.json()["paths"]
