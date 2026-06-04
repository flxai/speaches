import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Form, HTTPException, Response
from fastapi.responses import JSONResponse
import openai.types.audio
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
import torch

from speaches.audio import Audio
from speaches.dependencies import AudioFileDependency, ExecutorRegistryDependency
from speaches.diarization import DiarizationSegment, KnownSpeaker
from speaches.diarized_transcription import (
    DiarizedTranscriptionResponse,
    build_diarized_transcription_response,
)
from speaches.executors.shared.handler_protocol import TranscriptionRequest, VadRequest
from speaches.executors.silero_vad_v5 import VadOptions
from speaches.model_aliases import ModelId
from speaches.routers.diarization import _map_to_known_speakers
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise
from speaches.subtitles.ass import render_ass
from speaches.utils import parse_data_url_to_audio

logger = logging.getLogger(__name__)
router = APIRouter(tags=["diarized-transcription"])

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_VAD_OPTIONS = VadOptions(min_silence_duration_ms=160, max_speech_duration_s=30)


def parse_known_speakers(
    known_speaker_names: list[str] | None,
    known_speaker_references: list[str] | None,
) -> list[KnownSpeaker] | None:
    if not known_speaker_names and not known_speaker_references:
        return None
    if not known_speaker_names or not known_speaker_references:
        raise HTTPException(
            status_code=400,
            detail="known_speaker_names[] and known_speaker_references[] must be provided together.",
        )
    if len(known_speaker_names) != len(known_speaker_references):
        raise HTTPException(
            status_code=400,
            detail="known_speaker_names[] and known_speaker_references[] must have the same length.",
        )
    return [
        KnownSpeaker(name=name, audio=Audio(parse_data_url_to_audio(ref), sample_rate=16000))
        for name, ref in zip(known_speaker_names, known_speaker_references, strict=True)
    ]


def diarize_audio(
    executor_registry: ExecutorRegistryDependency,
    audio: Audio,
    model: str,
    known_speakers: list[KnownSpeaker] | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> list[DiarizationSegment]:
    model_card_data = get_model_card_data_or_raise(model)
    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.diarization)

    with executor.model_manager.load_model(model) as pipeline:
        waveform = torch.from_numpy(audio.data).unsqueeze(0).float()
        diarization_kwargs = {}
        if min_speakers is not None:
            diarization_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_kwargs["max_speakers"] = max_speakers
        diarization = pipeline(
            {"waveform": waveform, "sample_rate": audio.sample_rate},
            **diarization_kwargs,
        )
        assert isinstance(diarization, DiarizeOutput), f"Expected DiarizeOutput, got {type(diarization)}"

        speaker_mapping: dict[object, str] | None = None
        if known_speakers:
            try:
                speaker_mapping = _map_to_known_speakers(
                    pipeline,
                    waveform,
                    audio.sample_rate,
                    diarization,
                    known_speakers,
                )
            except Exception:
                logger.exception("Failed to map diarized speakers to known speakers, using default labels")

        return [
            DiarizationSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker_mapping[speaker] if speaker_mapping else str(speaker),
            )
            for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True)
        ]


def transcribe_verbose_json(
    executor_registry: ExecutorRegistryDependency,
    audio: Audio,
    model: str,
    language: str | None,
    prompt: str | None,
    temperature: float,
    hotwords: str | None,
) -> openai.types.audio.TranscriptionVerbose:
    model_card_data = get_model_card_data_or_raise(model)
    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.transcription)
    vad_request = VadRequest(audio=audio, vad_options=DEFAULT_VAD_OPTIONS)
    speech_segments = executor_registry.vad.model_manager.handle_vad_request(vad_request)
    transcription_request = TranscriptionRequest(
        audio=audio,
        model=model,
        language=language,
        prompt=prompt,
        response_format="verbose_json",
        temperature=temperature,
        timestamp_granularities=["word", "segment"],
        stream=False,
        hotwords=hotwords,
        speech_segments=speech_segments,
        vad_options=DEFAULT_VAD_OPTIONS,
        without_timestamps=False,
    )
    transcription = executor.model_manager.handle_transcription_request(transcription_request)
    if not isinstance(transcription, openai.types.audio.TranscriptionVerbose):
        raise HTTPException(status_code=500, detail="Transcription backend did not return verbose_json.")
    if not transcription.words:
        raise HTTPException(status_code=500, detail="Transcription backend returned no word timestamps.")
    return transcription


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
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[Literal["diarized_json", "ass"], Form()] = "diarized_json",
    temperature: Annotated[float, Form()] = 0.0,
    hotwords: Annotated[str | None, Form()] = None,
    known_speaker_names: Annotated[list[str] | None, Form(alias="known_speaker_names[]")] = None,
    known_speaker_references: Annotated[list[str] | None, Form(alias="known_speaker_references[]")] = None,
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

    known_speakers = parse_known_speakers(known_speaker_names, known_speaker_references)
    transcription = transcribe_verbose_json(
        executor_registry,
        audio,
        model,
        language,
        prompt,
        temperature,
        hotwords,
    )
    diarization_segments = diarize_audio(
        executor_registry,
        audio,
        diarization_model,
        known_speakers,
        min_speakers,
        max_speakers,
    )
    diarized = build_diarized_transcription_response(transcription, diarization_segments)

    if response_format == "ass":
        return Response(content=render_ass(diarized, max_words=max_words), media_type="text/x-ass")
    return JSONResponse(content=diarized.model_dump())
