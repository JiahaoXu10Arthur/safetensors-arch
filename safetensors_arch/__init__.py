"""Detect what a ``.safetensors`` file actually is, by reading its tensor table.

Not by filename. Not by the ``baseModel`` field a model host put on the page.
By the tensor keys, which cannot lie about what the weights are.

Why this exists
---------------
Loading a LoRA trained for one base-model family onto a different family is a
**silent failure**. Nothing raises. An image comes out. The LoRA simply did
nothing. You conclude "this LoRA is bad" and the conclusion is garbage.

Filename and vendor metadata both lie in practice:

- a file named ``kleinAnimePantyhose...`` was a Flux.2 workflow, not an anime
  SD model
- a LoRA distributed by its host with ``baseModel: <family A>`` had the tensor
  layout of family B

The tensor table does not lie: an SDXL UNet LoRA has ``input_blocks``, a
Cosmos/Anima-style DiT LoRA has ``adaln_modulation``, and so on.

Ordering matters, and getting it wrong is easy
----------------------------------------------
Some DiT LoRA keys look like::

    diffusion_model.blocks.0.adaln_modulation_cross_attn.1.lora_A.weight

That key carries **both** the base-model structure marker
(``adaln_modulation``) and the increment marker (``lora_A``). Check the
base-model markers first and you classify LoRAs as full checkpoints.

So: decide *whether it is an increment* first, and only then decide *whose
increment it is*.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional, Tuple

__all__ = ["read_header", "detect", "Result", "HEADER_LIMIT"]
__version__ = "0.2.1"

#: Refuse to read a "header" larger than this; a real one is kilobytes.
HEADER_LIMIT = 100 * 1024 * 1024

#: Substrings that mark a file as a *delta* (LoRA/LoCon/LyCORIS) rather than a
#: full model. ``.alpha`` is included because some trainers emit alpha scalars
#: alongside otherwise unlabelled up/down pairs.
#: ``.down.weight`` / ``.up.weight`` are here because XLabs writes
#: ``double_blocks.0.processor.proj_lora1.down.weight`` -- no ``lora_down``, no
#: ``.alpha``, no ``lora_A``. Without them that file matched nothing and fell
#: through to the full-model branch, coming back plain ``unknown``: not even
#: recognised as an increment. ``.diff_b`` is the Wan distill spelling.
LORA_MARKERS = ("lora_down", ".lora_A", ".lora_B", "lora_up",
                "lora_unet_", "lora_te", ".alpha",
                ".down.weight", ".up.weight", ".diff_b")

#: Structural markers that name a delta's target family, tested in this order.
#: Only *structure* belongs here. ``lora_te`` used to sit in the SDXL row and
#: does not any more: it is the sd-scripts text-encoder prefix, and
#: ``networks.lora_anima`` is an sd-scripts derivative that emits it too, so
#: standing alone it names no family. Sixteen Anima deltas were called
#: ``lora:sdxl`` on that basis alone.
#:
#: SDXL UNets are written two ways and both are real evidence: compvis
#: (``input_blocks``) and diffusers (``down_blocks``). Matching only the first
#: left a diffusers-style LoRA with unambiguous SDXL structure in its keys
#: falling through to ``lora:unknown``.
#: Each row is ``(kind, any_of, all_of)``: at least one substring from
#: ``any_of`` must appear, and every substring in ``all_of`` must appear.
#: Order is load-bearing and was settled by measurement, not by taste --
#: see DESIGN.md for the matrix over ten real vendor headers.
#:
#: Flux comes first because the diffusers layout writes
#: ``transformer.single_transformer_blocks.N.attn...`` and carries
#: ``add_k_proj``, which is both halves of the Qwen test: two of four real
#: Flux LoRAs were returning ``lora:qwen-image``. Qwen comes last because
#: ``transformer_blocks`` alone is not evidence of anything -- every one of
#: the 101 SDXL LoRAs in the local corpus carries it too.
#:
#: The Wan row asks for ``blocks.`` and ``cross_attn``, not for the
#: ``diffusion_model.`` prefix it used to require. A real Wan 2.2 delta that
#: writes its keys as ``blocks.N.cross_attn...`` was falling through to
#: ``lora:unknown`` purely over that prefix. The looser pair is not a guess:
#: over 335 real headers -- 266 local files, 59 out-of-sample vendor repos,
#: and the ten fixtures -- it matches exactly one file it did not match
#: before, and that file is the Wan delta. The row that had to be checked is
#: dit-adaln, whose ``cross_attn_k_proj`` marker contains ``cross_attn`` and
#: which sits *below* this one: it lost nothing.
_FAMILY_MARKERS = (
    ("lora:flux", ("double_blocks", "single_blocks",
                   "single_transformer_blocks"), ()),
    ("lora:wan", ("ffn",), ("blocks.", "cross_attn")),
    ("lora:dit-adaln", ("adaln_modulation", "cross_attn_k_proj"), ()),
    ("lora:sdxl", ("input_blocks", "output_blocks", "middle_block",
                   "down_blocks", "up_blocks", "mid_block"), ()),
    ("lora:qwen-image", ("add_k_proj",), ("transformer_blocks",)),
)

#: Substrings a trainer's ``modelspec.architecture`` may carry, and the family
#: each one names. Consulted only where the tensor keys named no family at
#: all: a declaration is a claim someone's code typed, so it never overrules
#: the table, but where the table is silent a named family beats ``unknown``.
_DECLARED_FAMILIES = (
    ("anima", "lora:dit-adaln"),
    ("flux", "lora:flux"),
    ("wan", "lora:wan"),
    ("qwen", "lora:qwen-image"),
    ("sdxl", "lora:sdxl"),
    ("stable-diffusion-xl", "lora:sdxl"),
)

Result = Tuple[str, str]
"""``(kind, why)``. ``why`` is a human-readable sentence, meant to be shown."""


def read_header(path) -> Optional[dict]:
    """Return the safetensors JSON header without loading any weights.

    The format is: 8 little-endian bytes of header length, then that many
    bytes of JSON. Returns ``None`` if the declared length is implausible,
    which is the cheap way to reject files that are not safetensors at all.
    """
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            return None
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > HEADER_LIMIT:
            return None
        try:
            return json.loads(f.read(n))
        except (ValueError, UnicodeDecodeError):
            return None


def _spec(header: dict) -> str:
    meta = header.get("__metadata__") or {}
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("modelspec.architecture") or "").lower()


def detect(path) -> Result:
    """Classify a ``.safetensors`` file.

    Returns one of:

    ``lora:dit-adaln``
        Delta weights over a DiT that uses adaptive-layernorm cross-attention
        (the Cosmos-Predict2 / "Anima" lineage).
    ``lora:sdxl``
        Delta weights over an SDXL-family UNet (SDXL, Pony, Illustrious,
        NoobAI, ...). Marked by the UNet block names, in either spelling:
        ``input_blocks`` / ``output_blocks`` / ``middle_block`` (compvis) or
        ``down_blocks`` / ``up_blocks`` / ``mid_block`` (diffusers).
    ``lora:flux``
        Delta weights over a Flux DiT. Written three ways in the wild --
        ``double_blocks`` / ``single_blocks`` (kohya, XLabs) and
        ``single_transformer_blocks`` (diffusers) -- and decided before
        Qwen, which it would otherwise be mistaken for.
    ``lora:wan``
        Delta weights over a Wan video DiT: ``diffusion_model.blocks.`` with
        ``ffn``. The prefix alone is shared with the adaln lineage; ``ffn``
        is what separates them.
    ``lora:qwen-image``
        Delta weights over the Qwen-Image DiT: ``transformer_blocks`` with an
        ``add_k_proj`` text branch. Decided last, because ``transformer_blocks``
        on its own is carried by every SDXL LoRA too.
    ``lora:unknown``
        Definitely a delta, but the target family is not recognised.
    ``dit-adaln``
        A full adaln-cross-attention DiT checkpoint.
    ``sdxl-checkpoint``
        A full SDXL-family checkpoint: ``model.`` + ``conditioner.`` +
        ``first_stage_model.``.
    ``unknown``
        Not recognised, or not readable as safetensors.

    The second element of the tuple always explains the decision, including
    the tensor count, so a wrong answer can be argued with.
    """
    path = Path(path)
    try:
        header = read_header(path)
    except OSError as exc:
        return "unknown", "could not read: %s" % exc
    if not header:
        return "unknown", "no usable safetensors header"

    keys = [k for k in header if k != "__metadata__"]
    if not keys:
        return "unknown", "header contains no tensors"

    top = {k.split(".")[0] for k in keys}
    # Both questions are asked over *every* key. The family question used to
    # read a 400-key prefix, on the argument that a missed family marker only
    # costs a ``lora:unknown``. That argument was wrong: the family branches
    # are ordered, so a missed marker does not fall through to ``unknown``, it
    # falls through to the *next branch*, which answers confidently and wrongly.
    # An Anima delta writes its 588 text-encoder tensors first, putting
    # ``cross_attn_k_proj`` at index 588; the prefix ended at 400, and the SDXL
    # branch fired on ``lora_te`` alone. 16 of 266 real files were misfiled that
    # way, and ``is_compatible`` then called them loadable onto an SDXL
    # checkpoint -- the load-does-nothing failure this module exists to catch.
    # On a 3000-tensor checkpoint the full scan costs about 0.4ms, well inside
    # the header read it rides along on.
    all_keys = "\n".join(keys)
    spec = _spec(header)
    note = " (trainer declared %s)" % spec if spec else ""

    # --- is it a delta? decide this FIRST; see module docstring ---
    if any(m in all_keys for m in LORA_MARKERS):
        # The reason names the markers that actually matched, never the row
        # they came from. Naming the whole row is how a file carrying only
        # cross_attn_k_proj came back "keys carry adaln_modulation /
        # cross_attn_k_proj", and two real SDXL files came back "keys carry
        # input_blocks / lora_te" with no input_blocks in them at all: the
        # right answer citing evidence it did not have.
        for kind, any_of, all_of in _FAMILY_MARKERS:
            if not all(m in all_keys for m in all_of):
                continue
            hit = [m for m in any_of if m in all_keys]
            if hit:
                cited = list(all_of) + hit
                return (kind, "%d tensors of delta weights; keys carry %s%s"
                        % (len(keys), " / ".join(cited), note))
        # The keys named no family. Only now does the declaration get a say,
        # and the reason records that the answer came from it rather than
        # from a marker -- a classifier you are meant to argue with must not
        # cite evidence it did not use.
        for marker, kind in _DECLARED_FAMILIES:
            if marker in spec:
                return (kind,
                        "%d tensors of delta weights; the keys name no "
                        "family, but the trainer declared %s"
                        % (len(keys), spec))
        return ("lora:unknown",
                "%d tensors of delta weights, but the target family is not "
                "recognised%s" % (len(keys), note))

    # --- then: which full model is it? ---
    if any("adaln_modulation_cross_attn" in k for k in keys):
        return ("dit-adaln",
                "%d tensors with adaln_modulation_cross_attn and no delta "
                "markers (full DiT)" % len(keys))
    if {"model", "conditioner", "first_stage_model"} <= top:
        return ("sdxl-checkpoint",
                "%d tensors; model. + conditioner. + first_stage_model. "
                "(full SDXL-family checkpoint)" % len(keys))
    return "unknown", "%d tensors; top-level prefixes %s" % (
        len(keys), sorted(top)[:4])


def is_compatible(lora_kind: str, base_kind: str) -> Optional[bool]:
    """Would this delta actually do anything on this base?

    ``None`` means "not enough information" — callers should let it through
    rather than block on a guess. The point of this function is to catch the
    *definitely wrong* pairing, not to be an authority on every file.
    """
    pairs = {
        ("lora:sdxl", "sdxl-checkpoint"): True,
        ("lora:dit-adaln", "dit-adaln"): True,
    }
    if lora_kind.startswith("lora:") and base_kind and not base_kind.startswith("lora:"):
        if (lora_kind, base_kind) in pairs:
            return True
        if lora_kind in ("lora:unknown",) or base_kind == "unknown":
            return None
        return False
    return None
