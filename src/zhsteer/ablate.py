"""Generation-time ablation of the Chinese AI-register direction (milestone m2).

The primitive: wrap any transformers ``generate()`` call in :func:`steer` and
the pack's register direction is projected out of the residual stream at the
layer it was extracted from, so the register never reaches the unembed — the
fix lives at the generation source instead of a rewrite pass.

Math: at the chosen decoder layer, every position's residual vector ``x``
becomes ``x - strength * (x @ d) d`` with unit ``d``. ``strength=1.0`` removes
exactly the register component; smaller values damp it, larger values
over-remove into the opposite component. ``d`` points from plain toward AI
register (see ``zhsteer.extract``), so removing its projection strips the
register while leaving the orthogonal content subspace untouched.

Hook placement follows the extraction convention: layer *i* is the residual
stream after decoder layer *i*, so the hook fires on decoder module *i*'s
output — the exact tensor ``extract`` measured as ``hidden_states[i+1]``. The
hook runs on every forward, covering prompt prefill and every decoded token
under the KV cache; keys and values cached in layers above the hook point are
computed from already-ablated residuals, which is the intended cascade.
(Inspecting ``output_hidden_states`` while steering can show pre-ablation
values on transformers 5.x, whose capture hooks may be registered before
zhsteer's; the residuals the next layers actually consume are always ablated.)

:func:`generate_pair` runs one prompt twice under identical inputs and sampling
parameters — once plain, once steered — and returns both texts side by side;
that pair is what the CLI prints and the demo report scores. :func:`is_degenerate`
is the deterministic repetition check behind m2's "zero degenerate repetition
at strength <= 1.0" bar: verbatim loops of a short unit are the signature
collapse mode of an ablation pushed past what the direction explains.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from zhsteer.pack import DirectionPack

# Same tolerance as pack.verify: float32 norm over d_model elements is accurate
# to ~1e-6, so 1e-4 leaves room without letting real drift through.
_NORM_ATOL = 1e-4


def project_out(x: torch.Tensor, direction: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Return ``x - strength * (x @ d) d`` for a unit-norm ``direction``.

    ``x`` is ``[..., d_model]``. The math runs in float32 and casts back to
    ``x``'s dtype, so bf16/fp16 residuals lose nothing to intermediate rounding.
    """
    x32 = x.to(torch.float32)
    d32 = direction.to(device=x32.device, dtype=torch.float32)
    coeff = (x32 @ d32).unsqueeze(-1)
    return (x32 - strength * coeff * d32).to(x.dtype)


def _validated_strength(strength: float) -> float:
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise TypeError(f"strength must be a number, got {type(strength).__name__}")
    strength = float(strength)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError(f"strength must be a finite non-negative number, got {strength}")
    return strength


def _validated_direction(direction: torch.Tensor) -> torch.Tensor:
    if not isinstance(direction, torch.Tensor):
        raise TypeError("direction must be a torch.Tensor")
    if direction.dim() != 1 or direction.numel() == 0:
        raise ValueError(f"direction must be 1-D [d_model], got shape {tuple(direction.shape)}")
    if not torch.isfinite(direction).all():
        raise ValueError("direction contains non-finite values")
    if abs(direction.detach().to(torch.float32).norm().item() - 1.0) > _NORM_ATOL:
        raise ValueError("direction must be unit-norm (load it from a DirectionPack)")
    return direction.detach().to(torch.float32)


def _decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    core = getattr(model, "model", model)
    layers = getattr(core, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList) or len(layers) == 0:
        raise TypeError(
            f"cannot locate decoder layers on {type(model).__name__}: "
            "expected an HF causal LM exposing .model.layers (e.g. Qwen3ForCausalLM)"
        )
    return layers


class _ProjectionOutHook:
    """Forward hook projecting the direction out of one layer's residual output.

    Works whether the decoder layer returns the residual as a bare tensor
    (transformers 5.x) or as a tuple led by it (older releases); anything else
    fails loudly rather than silently leaving the register in place.
    """

    def __init__(self, layer: int, direction: torch.Tensor, strength: float):
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError(f"layer must be an int, got {type(layer).__name__}")
        if layer < 0:
            raise ValueError(f"layer must be non-negative, got {layer}")
        self.layer = layer
        self.strength = strength
        self._direction = _validated_direction(direction)
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def attach(self, layers: torch.nn.ModuleList) -> None:
        module = layers[self.layer]
        self._direction = self._direction.to(next(module.parameters()).device)
        self._handle = module.register_forward_hook(self._ablate)

    def _ablate(self, module, args, output):
        if self.strength == 0.0:
            return None  # pass-through: hooks returning None leave the output untouched
        if isinstance(output, (tuple, list)):
            if not output or not isinstance(output[0], torch.Tensor):
                lead = type(output[0]).__name__ if output else "nothing"
                raise TypeError(
                    f"decoder layer output leads with {lead}, expected a residual tensor — cannot ablate"
                )
            ablated = project_out(output[0], self._direction, self.strength)
            return type(output)((ablated, *output[1:]))
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"decoder layer output is {type(output).__name__}, "
                "expected a residual tensor — cannot ablate"
            )
        return project_out(output, self._direction, self.strength)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@contextlib.contextmanager
def _steering(hook: _ProjectionOutHook, layers: torch.nn.ModuleList) -> Iterator[None]:
    hook.attach(layers)
    try:
        yield
    finally:
        hook.remove()


def steer(
    model: torch.nn.Module, pack: DirectionPack, *, strength: float = 1.0
) -> contextlib.AbstractContextManager[None]:
    """Project the pack's register direction out of the residual stream inside the block.

    The primitive users wrap around an existing transformers call::

        with steer(model, pack, strength=0.9):
            output = model.generate(**inputs)

    Arguments are validated at call time; the hook registers on block entry and
    is removed on exit — even when the block raises — so a steered call can
    never leak into surrounding generations.
    """
    hook = _ProjectionOutHook(pack.layer, pack.direction, _validated_strength(strength))
    layers = _decoder_layers(model)
    if hook.layer >= len(layers):
        raise ValueError(
            f"layer {hook.layer} is out of range for a model with {len(layers)} decoder layers"
        )
    return _steering(hook, layers)


@dataclass(frozen=True)
class GenerationPair:
    """One prompt generated twice: plain, and with the register direction ablated."""

    prompt: str
    before: str
    after: str
    layer: int
    strength: float


def generate_pair(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    pack: DirectionPack,
    *,
    strength: float = 1.0,
    max_new_tokens: int = 512,
    use_chat_template: bool = True,
    generation_kwargs: dict | None = None,
) -> GenerationPair:
    """Generate once plain and once ablated under identical settings.

    Both sides share inputs and sampling parameters, so the only difference is
    the ablation. Generation is greedy unless ``generation_kwargs`` says
    otherwise — a deterministic default keeps before/after attributable to the
    ablation rather than to sampling noise. Instruct models get the tokenizer's
    chat template applied when it ships one; ``use_chat_template=False`` feeds
    the raw prompt (base-model style completion).
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

    device = next(model.parameters()).device
    text = prompt
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    encoded = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in encoded.items()}
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, **(generation_kwargs or {})}
    if kwargs.get("pad_token_id") is None and tokenizer.pad_token_id is None:
        kwargs["pad_token_id"] = tokenizer.eos_token_id

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            before = model.generate(**inputs, **kwargs)
            with steer(model, pack, strength=strength):
                after = model.generate(**inputs, **kwargs)
    finally:
        if was_training:
            model.train()

    prompt_len = inputs["input_ids"].shape[1]
    return GenerationPair(
        prompt=prompt,
        before=tokenizer.decode(before[0, prompt_len:], skip_special_tokens=True),
        after=tokenizer.decode(after[0, prompt_len:], skip_special_tokens=True),
        layer=pack.layer,
        strength=strength,
    )


# Degeneracy bar: a unit of 1-4 characters repeated 6+ times in a row, or a
# unit of 5-24 characters repeated 3+ times. The higher short-unit bar keeps
# legitimate reduplication ("哈哈哈哈") out; genuine 排比 paraphrases its
# clauses, so exact triple repetition of a longer unit is always collapse.
_DEGENERATE_MAX_UNIT = 24
_DEGENERATE_SHORT_UNIT = 4
_DEGENERATE_SHORT_REPEATS = 6
_DEGENERATE_LONG_REPEATS = 3


def is_degenerate(text: str) -> bool:
    """True when a generation collapsed into verbatim repetition (or nothing).

    The quality gate behind m2's "zero degenerate repetition" bar and the demo
    report's degenerate-output check.
    """
    if not text or not text.strip():
        return True
    for unit_len in range(1, _DEGENERATE_MAX_UNIT + 1):
        repeats = (
            _DEGENERATE_SHORT_REPEATS
            if unit_len <= _DEGENERATE_SHORT_UNIT
            else _DEGENERATE_LONG_REPEATS
        )
        if len(text) < unit_len * repeats:
            continue
        if _longest_period_run(text, unit_len) >= (repeats - 1) * unit_len:
            return True
    return False


def _longest_period_run(text: str, period: int) -> int:
    """Longest run of consecutive positions where ``text[i] == text[i - period]``.

    A run of ``(repeats - 1) * period`` matches means ``repeats`` consecutive
    copies of some unit appear back to back.
    """
    best = run = 0
    for index in range(period, len(text)):
        if text[index] == text[index - period]:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best
