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

from typing import Callable

_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Dynamically import attention functions based on available hardware."""

    from verl.utils.device import is_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    if is_npu_available:
        from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input
    else:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:
            # These helpers are device-agnostic PyTorch implementations copied
            # from flash-attn.  Reuse them when the CUDA extension is absent;
            # returning the input unchanged here corrupts remove-padding layouts.
            from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args, **kwargs):
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
