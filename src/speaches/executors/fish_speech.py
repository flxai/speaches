from collections.abc import Generator
import io
import logging
from typing import Literal

import httpx
import numpy as np
from pydantic import BaseModel, computed_field
import soundfile as sf

from speaches.api_types import OPENAI_SUPPORTED_SPEECH_VOICE_NAMES, Model
from speaches.audio import Audio
from speaches.config import FishSpeechConfig
from speaches.executors.shared.handler_protocol import SpeechRequest, SpeechResponse
from speaches.hf_utils import HfModelFilter
from speaches.model_registry import ModelRegistry
from speaches.tracing import traced_generator

SAMPLE_RATE = 44100
TASK_NAME_TAG = "text-to-speech"
TAGS = {"speaches", "fish-speech"}
FISH_SPEECH_LIBRARY_NAME = "fish-speech"
DEFAULT_VOICE_NAMES = ("default", *OPENAI_SUPPORTED_SPEECH_VOICE_NAMES)

logger = logging.getLogger(__name__)


class FishSpeechModelVoice(BaseModel):
    name: str
    language: str = "multilingual"
    gender: Literal["male", "female"] | None = None

    @computed_field
    @property
    def id(self) -> str:
        return self.name


VOICES = [FishSpeechModelVoice(name=name) for name in DEFAULT_VOICE_NAMES]


class FishSpeechModel(Model):
    sample_rate: int
    voices: list[FishSpeechModelVoice]


class FishSpeechModelRegistry(ModelRegistry[FishSpeechModel, None]):
    def __init__(self, config: FishSpeechConfig) -> None:
        super().__init__(
            hf_model_filter=HfModelFilter(
                library_name=FISH_SPEECH_LIBRARY_NAME,
                task=TASK_NAME_TAG,
                tags=TAGS,
            )
        )
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _model(self) -> FishSpeechModel:
        return FishSpeechModel(
            id=self.config.model_id,
            created=0,
            owned_by=self.config.model_id.split("/", maxsplit=1)[0],
            language=None,
            task=TASK_NAME_TAG,
            sample_rate=SAMPLE_RATE,
            voices=VOICES,
        )

    def list_remote_models(self) -> Generator[FishSpeechModel]:
        if self.enabled:
            yield self._model()

    def list_local_models(self) -> Generator[FishSpeechModel]:
        if self.enabled:
            yield self._model()

    def get_model(self, model_id: str) -> FishSpeechModel:
        if not self.enabled or model_id != self.config.model_id:
            raise ValueError(f"Model '{model_id}' not found")
        return self._model()

    def get_model_files(self, model_id: str) -> None:
        if not self.enabled or model_id != self.config.model_id:
            raise ValueError(f"Model '{model_id}' not found")
        return None

    def download_model_files(self, model_id: str) -> None:
        if not self.enabled or model_id != self.config.model_id:
            raise ValueError(f"Model '{model_id}' not found")


class FishSpeechModelManager:
    def __init__(self, config: FishSpeechConfig) -> None:
        self.config = config

    def _reference_id_for_voice(self, voice: str) -> str | None:
        if voice in DEFAULT_VOICE_NAMES:
            return None
        return voice

    @traced_generator()
    def handle_speech_request(
        self,
        request: SpeechRequest,
        **_kwargs,
    ) -> SpeechResponse:
        if request.model != self.config.model_id:
            raise ValueError(f"Model '{request.model}' is not handled by the Fish Speech proxy")

        if request.speed != 1.0:
            logger.warning("Fish Speech does not support OpenAI speech speed; ignoring speed=%s", request.speed)

        payload = {
            "text": request.text,
            "format": "wav",
            "references": [],
            "normalize": True,
            "use_memory_cache": "on",
        }
        reference_id = self._reference_id_for_voice(request.voice)
        if reference_id is not None:
            payload["reference_id"] = reference_id

        base_url = self.config.base_url.rstrip("/")
        url = f"{base_url}/v1/tts"
        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                response = client.post(
                    url,
                    json=payload,
                    headers={
                        "Accept": "audio/wav",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise RuntimeError(f"Fish Speech request failed: {error}") from error

        data, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)

        yield Audio(np.asarray(data, dtype=np.float32), sample_rate=sample_rate)
