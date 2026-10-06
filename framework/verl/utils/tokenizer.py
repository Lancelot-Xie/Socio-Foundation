# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for tokenization."""

import json
import types
import warnings
from pathlib import Path

__all__ = ["hf_tokenizer", "hf_processor"]


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer, __version__ as transformers_version

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107

    # Transformers 5.x can save ``extra_special_tokens`` as a list, while
    # Transformers 4.x expects a mapping and calls ``.keys()`` on it.  Keep
    # the model-specific tokens special by converting the list at load time
    # instead of mutating the checkpoint's tokenizer_config.json.
    try:
        transformers_major = int(transformers_version.split(".", maxsplit=1)[0])
    except (TypeError, ValueError):
        transformers_major = 0
    if transformers_major and transformers_major < 5 and "extra_special_tokens" not in kwargs:
        tokenizer_config_path = Path(name_or_path) / "tokenizer_config.json"
        if tokenizer_config_path.is_file():
            try:
                with tokenizer_config_path.open(encoding="utf-8") as config_file:
                    tokenizer_config = json.load(config_file)
                extra_special_tokens = tokenizer_config.get("extra_special_tokens")
                if isinstance(extra_special_tokens, list):
                    kwargs["extra_special_tokens"] = {
                        f"extra_special_token_{index}": token
                        for index, token in enumerate(extra_special_tokens)
                        if isinstance(token, str)
                    }
                    warnings.warn(
                        "Converted Transformers 5.x list-style extra_special_tokens "
                        "for compatibility with Transformers 4.x",
                        stacklevel=1,
                    )
            except (OSError, TypeError, ValueError) as error:
                warnings.warn(
                    f"Failed to inspect tokenizer_config.json for compatibility: {error}", stacklevel=1
                )

    # Some released Qwen/simulation tokenizer files carry the legacy pre-tokenizer
    # regex that recent Transformers versions identify as incorrect.  Apply
    # the upstream compatibility switch for Qwen tokenizers only; this keeps
    # training, agent-loop rollout, and evaluation tokenization identical.
    if "fix_mistral_regex" not in kwargs:
        tokenizer_config_path = Path(name_or_path) / "tokenizer_config.json"
        if tokenizer_config_path.is_file():
            try:
                with tokenizer_config_path.open(encoding="utf-8") as config_file:
                    tokenizer_config = json.load(config_file)
                tokenizer_class = str(tokenizer_config.get("tokenizer_class") or "")
                if tokenizer_class.startswith("Qwen2Tokenizer"):
                    kwargs["fix_mistral_regex"] = True
            except (OSError, TypeError, ValueError) as error:
                warnings.warn(
                    f"Failed to inspect tokenizer regex compatibility: {error}", stacklevel=1
                )
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoConfig, AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        config = AutoConfig.from_pretrained(name_or_path, **kwargs)

        # Bind vlm model's get_rope_index method to processor
        processor.config = config
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from transformers.models.qwen2_vl import Qwen2VLModel

                processor.get_rope_index = types.MethodType(Qwen2VLModel.get_rope_index, processor)
            case "Qwen2_5_VLProcessor":
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

                processor.get_rope_index = types.MethodType(Qwen2_5_VLModel.get_rope_index, processor)
            case "Qwen3VLProcessor":
                from transformers.models.qwen3_vl import Qwen3VLModel

                processor.get_rope_index = types.MethodType(Qwen3VLModel.get_rope_index, processor)
            case "Glm4vImageProcessor":
                from transformers.models.glm4v import Glm4vModel

                processor.get_rope_index = types.MethodType(Glm4vModel.get_rope_index, processor)
            case _:
                raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
