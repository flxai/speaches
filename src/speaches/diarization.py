from pydantic import BaseModel, ConfigDict

from speaches.audio import Audio


class KnownSpeaker(BaseModel):
    name: str
    audio: Audio

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DiarizationSegment(BaseModel):
    start: float
    """Start timestamp of the segment in seconds."""
    end: float
    """End timestamp of the segment in seconds."""
    speaker: str
    """Speaker label for this segment. When known speakers are provided, the label matches the known speaker name. Otherwise speakers are labeled as SPEAKER_00, SPEAKER_01, etc."""


class DiarizationResponse(BaseModel):
    duration: float
    """Duration of the input audio in seconds."""
    segments: list[DiarizationSegment]
    """Diarization segments annotated with timestamps and speaker labels."""
