from typing import Protocol

import numpy as np


class TranscribesAudioSnapshots(Protocol):
    async def transcribe(
        self,
        audio_data: np.typing.NDArray[np.float32],
        *,
        model: str,
        language: str | None,
    ) -> str: ...
