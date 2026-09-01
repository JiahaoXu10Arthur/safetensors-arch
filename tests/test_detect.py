"""Tests built on synthetic headers, so no model weights are needed.

A safetensors file is 8 bytes of header length, then that much JSON, then the
raw tensor bytes. Everything ``detect()`` looks at lives in the JSON, so a
valid test fixture is a few hundred bytes.

The case that matters most is ``test_dit_lora_is_not_a_checkpoint``: those
keys carry *both* a base-model marker and a delta marker, and checking them in
the wrong order silently misfiles every LoRA in that family.
"""

import json
import struct

import pytest

from safetensors_arch import detect, read_header, is_compatible


def write(tmp_path, keys, metadata=None, name="m.safetensors"):
    header = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
              for k in keys}
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    p = tmp_path / name
    p.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00\x00")
    return p


def test_reads_header(tmp_path):
    p = write(tmp_path, ["a.weight"])
    assert "a.weight" in read_header(p)


def test_not_safetensors(tmp_path):
    p = tmp_path / "x.safetensors"
    p.write_bytes(b"this is not a safetensors file at all")
    assert detect(p)[0] == "unknown"


def test_truncated(tmp_path):
    p = tmp_path / "x.safetensors"
    p.write_bytes(b"\x01\x02")
    assert detect(p)[0] == "unknown"


def test_absurd_header_length_rejected(tmp_path):
    p = tmp_path / "x.safetensors"
    p.write_bytes(struct.pack("<Q", 2 ** 60) + b"{}")
    assert read_header(p) is None


def test_sdxl_lora(tmp_path):
    p = write(tmp_path, [
        "lora_unet_input_blocks_1_1_proj_in.lora_down.weight",
        "lora_unet_input_blocks_1_1_proj_in.lora_up.weight",
        "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
    ])
    kind, why = detect(p)
    assert kind == "lora:sdxl"
    assert "3 tensors" in why


def test_dit_lora_is_not_a_checkpoint(tmp_path):
    """The ordering trap.

    These keys contain ``adaln_modulation_cross_attn`` — a *base model*
    marker — and also ``lora_A`` — a *delta* marker. Check base-model markers
    first and this comes back as a full checkpoint, which then gets loaded by
    the wrong node and fails in a way that points nowhere near the cause.
    """
    p = write(tmp_path, [
        "diffusion_model.blocks.0.adaln_modulation_cross_attn.1.lora_A.weight",
        "diffusion_model.blocks.0.adaln_modulation_cross_attn.1.lora_B.weight",
    ])
    assert detect(p)[0] == "lora:dit-adaln"


def test_qwen_image_lora(tmp_path):
    p = write(tmp_path, [
        "transformer_blocks.0.attn.add_k_proj.lora_A.weight",
        "transformer_blocks.0.attn.add_k_proj.lora_B.weight",
    ])
    assert detect(p)[0] == "lora:qwen-image"


def test_unrecognised_delta_says_so(tmp_path):
    """An unknown delta must not be guessed into a family."""
    p = write(tmp_path, [
        "some.exotic.module.lora_down.weight",
        "some.exotic.module.lora_up.weight",
    ])
    kind, why = detect(p)
    assert kind == "lora:unknown"
    assert "not recognised" in why


def test_full_dit_checkpoint(tmp_path):
    p = write(tmp_path, [
        "net.blocks.0.adaln_modulation_cross_attn.1.weight",
        "net.blocks.0.self_attn.q_proj.weight",
    ])
    assert detect(p)[0] == "dit-adaln"


def test_full_sdxl_checkpoint(tmp_path):
    p = write(tmp_path, [
        "model.diffusion_model.input_blocks.0.0.weight",
        "conditioner.embedders.0.transformer.text_model.embeddings.weight",
        "first_stage_model.encoder.conv_in.weight",
    ])
    assert detect(p)[0] == "sdxl-checkpoint"


def test_trainer_declaration_is_used_but_not_trusted_alone(tmp_path):
    """A trainer's own declaration is the most reliable signal when present,
    but the file still has to look like a delta first."""
    p = write(tmp_path, ["blocks.0.qkv.lora_down.weight"],
              metadata={"modelspec.architecture": "anima/lora"})
    kind, why = detect(p)
    assert kind == "lora:dit-adaln"
    assert "trainer declared" in why


def test_empty_header(tmp_path):
    p = write(tmp_path, [])
    assert detect(p)[0] == "unknown"


@pytest.mark.parametrize("lora,base,expected", [
    ("lora:sdxl", "sdxl-checkpoint", True),
    ("lora:dit-adaln", "dit-adaln", True),
    ("lora:sdxl", "dit-adaln", False),
    ("lora:dit-adaln", "sdxl-checkpoint", False),
    ("lora:unknown", "sdxl-checkpoint", None),
    ("lora:sdxl", "unknown", None),
])
def test_compatibility(lora, base, expected):
    """Unknown must return None, not False.

    This guard exists to stop the *definitely wrong* pairing. Blocking on
    "I don't recognise this" would make it useless the first time someone
    brings a family the table has never seen.
    """
    assert is_compatible(lora, base) is expected


def test_a_delta_marker_past_the_sampling_window_is_still_a_delta(tmp_path):
    # The delta check and the full-model check must see the same keys. When
    # the delta check sampled a prefix and the full-model check scanned
    # everything, a LoRA whose delta markers sorted late was classified as a
    # checkpoint -- the exact misfiling the ordering in this module exists to
    # prevent, and the one that fails silently downstream.
    keys = ["diffusion_model.blocks.%d.adaln_modulation_cross_attn.weight" % i
            for i in range(450)]
    keys.append("diffusion_model.blocks.0.attn.lora_A.weight")
    p = write(tmp_path, keys)
    assert detect(p)[0] == "lora:dit-adaln"


def test_the_declaration_speaks_only_where_the_keys_do_not(tmp_path):
    # Nothing in the tensor names says which family this delta targets. The
    # trainer wrote it down, and a named family is worth more than unknown --
    # but only here, where the data itself said nothing.
    p = write(tmp_path, ["net.layer.lora_down.weight", "net.layer.lora_up.weight"],
              metadata={"modelspec.architecture": "qwen-image/lora"})
    kind, why = detect(p)
    assert kind == "lora:qwen-image"
    assert "trainer declared" in why


def test_the_keys_outrank_the_declaration_when_they_disagree(tmp_path):
    # A declaration is a claim someone's code typed; the tensor table is what
    # the weights are. When both speak, the table wins.
    p = write(tmp_path, ["lora_unet_input_blocks_1.lora_down.weight"],
              metadata={"modelspec.architecture": "qwen-image/lora"})
    assert detect(p)[0] == "lora:sdxl"


def test_a_declared_family_does_not_claim_markers_the_keys_lack(tmp_path):
    # The reason has to describe the evidence actually used. Saying the keys
    # carry adaln_modulation about a file whose keys carry nothing of the sort
    # is the one thing a classifier you are meant to argue with must not do.
    p = write(tmp_path, ["blocks.0.qkv.lora_down.weight"],
              metadata={"modelspec.architecture": "anima/lora"})
    kind, why = detect(p)
    assert kind == "lora:dit-adaln"
    assert "adaln_modulation" not in why


def test_no_declaration_and_no_family_evidence_is_still_unknown(tmp_path):
    p = write(tmp_path, ["net.layer.lora_down.weight"])
    assert detect(p)[0] == "lora:unknown"


def test_a_declaration_does_not_overrule_a_family_the_keys_named(tmp_path):
    # The sharpest form of the rule. The trainer declared the adaln lineage;
    # the keys carry SDXL structure. The table is what the weights are, so it
    # wins, and the declaration is left to the cases where nothing else spoke.
    p = write(tmp_path, ["lora_unet_input_blocks_1.lora_down.weight"],
              metadata={"modelspec.architecture": "anima/lora"})
    assert detect(p)[0] == "lora:sdxl"


def test_a_family_marker_past_the_sampling_window_still_names_the_family(tmp_path):
    # The sibling of the test above, and the reason the family question stopped
    # sampling too. An Anima delta writes its 588 text-encoder tensors first and
    # its adaln blocks after, so cross_attn_k_proj lands past any prefix. Missing
    # it does not degrade to lora:unknown -- the family branches are ordered, so
    # the next one fires on lora_te alone and returns a confident lora:sdxl.
    # That answer then reads as compatible with an SDXL checkpoint, which is the
    # load-does-nothing failure this package exists to catch.
    keys = ["lora_te_layers_%d_mlp_down_proj.lora_down.weight" % i
            for i in range(588)]
    keys += ["lora_unet_blocks_%d_cross_attn_k_proj.lora_down.weight" % i
             for i in range(84)]
    p = write(tmp_path, keys, metadata={"modelspec.architecture": "anima-preview/lora"})
    assert detect(p)[0] == "lora:dit-adaln"
