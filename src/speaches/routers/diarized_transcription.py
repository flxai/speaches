import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Form, HTTPException, Response
from fastapi.responses import JSONResponse

from speaches.dependencies import AudioFileDependency, ExecutorRegistryDependency
from speaches.diarized_transcription import DiarizedTranscriptionResponse
from speaches.model_aliases import ModelId
from speaches.subtitles.ass import render_ass

logger = logging.getLogger(__name__)
router = APIRouter(tags=["diarized-transcription"])

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


@router.post(
    "/v1/audio/diarized-transcriptions",
    response_model=DiarizedTranscriptionResponse,
    responses={
        200: {
            "content": {
                "application/json": {},
                "text/x-ass": {},
            },
        },
    },
)
def create_diarized_transcription(
    executor_registry: ExecutorRegistryDependency,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    diarization_model: Annotated[ModelId, Form()] = DEFAULT_DIARIZATION_MODEL,
    language: Annotated[str | None, Form()] = None,
    response_format: Annotated[Literal["diarized_json", "ass"], Form()] = "diarized_json",
    min_speakers: Annotated[int | None, Form()] = None,
    max_speakers: Annotated[int | None, Form()] = None,
    speaker_layout: Annotated[Literal["vertical"], Form()] = "vertical",
    max_words: Annotated[int, Form()] = 7,
) -> Response:
    if speaker_layout != "vertical":
        raise HTTPException(status_code=400, detail="Only speaker_layout=vertical is supported.")
    if max_words < 0:
        raise HTTPException(status_code=400, detail="max_words must be non-negative.")
    if min_speakers is not None and min_speakers < 1:
        raise HTTPException(status_code=400, detail="min_speakers must be positive.")
    if max_speakers is not None and max_speakers < 1:
        raise HTTPException(status_code=400, detail="max_speakers must be positive.")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise HTTPException(status_code=400, detail="min_speakers must not exceed max_speakers.")

    # Diarized transcription runs through WhisperX: transcribe -> wav2vec2 align ->
    # diarize -> word-speaker assignment, producing a DiarizedTranscriptionResponse.
    diarized = executor_registry.whisperx.transcribe_diarized(
        audio,
        model,
        diarization_model,
        language,
        min_speakers,
        max_speakers,
    )

    if response_format == "ass":
        return Response(content=render_ass(diarized, max_words=max_words), media_type="text/x-ass")
    return JSONResponse(content=diarized.model_dump())
