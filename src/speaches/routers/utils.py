from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import huggingface_hub

    from speaches.executors.shared.executor import Executor

from fastapi import HTTPException
from huggingface_hub import ModelCardData
from huggingface_hub.utils._cache_manager import _scan_cached_repo

from speaches.hf_utils import (
    MODEL_CARD_DOESNT_EXISTS_ERROR_MESSAGE,
    get_model_card_data_from_cached_repo_info,
    get_model_repo_path,
)


def get_synthetic_model_card_data(model_id: str) -> ModelCardData | None:
    from speaches.dependencies import get_config

    config = get_config()
    if config.fish_speech.enabled and model_id == config.fish_speech.model_id:
        return ModelCardData(
            library_name="fish-speech",
            pipeline_tag="text-to-speech",
            tags=[
                "speaches",
                "fish-speech",
                "text-to-speech",
            ],
        )
    return None


def get_model_card_data_or_raise(model_id: str) -> huggingface_hub.ModelCardData:
    model_repo_path = get_model_repo_path(model_id)
    if model_repo_path is None:
        synthetic_model_card_data = get_synthetic_model_card_data(model_id)
        if synthetic_model_card_data is not None:
            return synthetic_model_card_data
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' is not installed locally. You can download the model using `POST /v1/models`",
        )
    cached_repo_info = _scan_cached_repo(model_repo_path)
    model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
    if model_card_data is None:
        raise HTTPException(
            status_code=500,
            detail=MODEL_CARD_DOESNT_EXISTS_ERROR_MESSAGE.format(model_id=model_id),
        )
    return model_card_data


def find_executor_for_model_or_raise[T: Executor](
    model_id: str, model_card_data: huggingface_hub.ModelCardData, executors: Iterable[T]
) -> T:
    for executor in executors:
        if executor.can_handle_model(model_id, model_card_data):
            return executor
    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_id}' is not supported. If you think this is a mistake, please open an issue.",
    )
