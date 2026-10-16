"""Model + tokenizer loading for zhsteer (plan §4: "loader.py — model load").

The residual-stream hook registry the plan mentions lives where it is used:
activation *capture* needs no hooks (``extract`` reads ``output_hidden_states``
from one forward) and *ablation* owns its hook (``zhsteer.ablate``). What stays
here is everything about turning a model reference into loaded weights:

- ``MODEL_ALIASES`` maps short names to HF repo ids. ``qwen3-4b`` is the plan's
  validated target; the schema is model-tagged, so other HF ids resolve fine —
  v0.1 simply makes no claim about them.
- :func:`load_model_and_tokenizer` enforces the plan's hardware boundary
  (CUDA only, macOS/CPU out of scope) and loads in bf16 when the GPU supports
  it — fp32 weights of a 4B model do not fit a 16GB card alongside KV cache.
"""

from __future__ import annotations

import inspect
from typing import NamedTuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ALIASES: dict[str, str] = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
}


def resolve_model_id(ref: str) -> str:
    """Resolve a CLI model reference to an HF repo id.

    Short aliases (``qwen3-4b``) and full repo ids (``Qwen/Qwen3-4B-Instruct-2507``)
    both work; anything else is rejected with the accepted forms listed.
    """
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("model reference is empty")
    if ref in MODEL_ALIASES:
        return MODEL_ALIASES[ref]
    if "/" in ref:  # org/name form — treated as an HF repo id as-is
        return ref
    known = ", ".join(sorted(MODEL_ALIASES))
    raise ValueError(f"unknown model reference {ref!r}; use an alias ({known}) or an HF repo id like org/name")


class Loaded(NamedTuple):
    model: torch.nn.Module
    tokenizer: object
    model_id: str


def _dtype_kwarg() -> str:
    """``transformers>=5`` renamed ``torch_dtype`` to ``dtype``; support both."""
    return "dtype" if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters else "torch_dtype"


def load_model_and_tokenizer(model_id: str) -> Loaded:
    """Load a causal LM + tokenizer onto the CUDA GPU in half precision.

    Raises with a pointer to ``zhsteer doctor`` when no CUDA device is visible —
    the plan scopes v0.1 to CUDA >= 16GB VRAM, and silently falling back to CPU
    would turn a 30-minute GPU run into an overnight one.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available — zhsteer v0.1 requires a CUDA GPU with >= 16GB VRAM "
            "(run `zhsteer doctor` to inspect this machine)"
        )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        **{_dtype_kwarg(): dtype},
    )
    model.to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return Loaded(model=model, tokenizer=tokenizer, model_id=model_id)
