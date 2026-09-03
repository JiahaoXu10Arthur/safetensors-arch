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


def test_diffusers_style_sdxl_keys_name_the_family(tmp_path):
    """SDXL UNets are written two ways. The compvis names (input_blocks,
    output_blocks) were matched; the diffusers names (down_blocks, up_blocks,
    mid_block) were not, so a diffusers-style LoRA with no text-encoder keys
    and no declaration fell through to lora:unknown with unambiguous SDXL
    structure sitting in its keys."""
    # The first key is a real one, kept whole because it carries a trap: a
    # diffusers SDXL key contains ``transformer_blocks``, which is also half
    # of the Qwen-Image test. SDXL is tried first and Qwen additionally
    # requires ``add_k_proj``; reorder those two and this file changes family.
    p = write(tmp_path, [
        "lora_unet_down_blocks_2_attentions_1_transformer_blocks_8"
        "_attn1_to_out_0.lora_down.weight",
        "lora_unet_up_blocks_0_attentions_0_proj_in.lora_down.weight",
        "lora_unet_mid_block_attentions_0_proj_out.lora_down.weight",
    ])
    assert detect(p)[0] == "lora:sdxl"


def test_the_reason_names_the_marker_that_actually_matched(tmp_path):
    """Two real files declared stable-diffusion-xl, carried diffusers UNet
    keys plus lora_te, and came back "keys carry input_blocks / lora_te" --
    with no input_blocks anywhere in the file. Right answer, invented
    evidence, which is the one thing a classifier you are meant to argue with
    must not do."""
    p = write(tmp_path, [
        "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
        "lora_unet_down_blocks_2_attentions_1_attn1_to_out_0.lora_down.weight",
    ], metadata={"modelspec.architecture": "stable-diffusion-xl/lora"})
    kind, why = detect(p)
    assert kind == "lora:sdxl"
    assert "down_blocks" in why
    assert "input_blocks" not in why


def test_the_adaln_reason_does_not_claim_both_markers(tmp_path):
    """Same defect in the adaln branch: the 16 Anima files carry
    cross_attn_k_proj and no adaln_modulation, and the reason named both."""
    p = write(tmp_path, [
        "lora_unet_blocks_0_cross_attn_k_proj.lora_down.weight",
    ])
    kind, why = detect(p)
    assert kind == "lora:dit-adaln"
    assert "cross_attn_k_proj" in why
    assert "adaln_modulation" not in why


def test_lora_te_alone_no_longer_claims_the_sdxl_family(tmp_path):
    """lora_te is the sd-scripts text-encoder prefix, not an SDXL structure
    marker; networks.lora_anima emits it too. Standing alone it must not name
    a family -- degrading to the declaration, or to lora:unknown, is an answer
    this package is content to give. A confident wrong family is not."""
    p = write(tmp_path, [
        "lora_te_layers_0_mlp_down_proj.lora_down.weight",
        "lora_te_layers_0_mlp_down_proj.lora_up.weight",
    ])
    assert detect(p)[0] == "lora:unknown"


# ------------------------------------------- real headers, real families

def _real_families():
    import gzip
    import pathlib
    f = pathlib.Path(__file__).parent / "fixtures" / "real_families.json.gz"
    return json.loads(gzip.decompress(f.read_bytes()))["samples"]


def _from_keys(tmp_path, keys, spec=None, name="probe.safetensors"):
    return write(tmp_path, keys,
                 metadata={"modelspec.architecture": spec} if spec else None,
                 name=name)


@pytest.mark.parametrize("s", _real_families(), ids=lambda s: s["source"])
def test_real_family_headers_classify_as_recorded(s, tmp_path):
    """Real headers pulled over HTTP Range from public repos.

    Synthetic cases only prove the matcher fires on shapes already known.
    These are the shapes vendors actually ship, in file order, complete --
    a truncated list would hide the very bug this module was fixed for twice.
    """
    p = _from_keys(tmp_path, s["keys"], s.get("modelspec_architecture"))
    assert detect(p)[0] == s["expect"]


def test_a_flux_lora_is_not_mistaken_for_qwen_image(tmp_path):
    """Two of the four real Flux LoRAs came back lora:qwen-image. The diffusers
    layout writes transformer.single_transformer_blocks.N.attn... and carries
    add_k_proj, which is both halves of the Qwen test. Flux has to be decided
    first, and on a marker Qwen does not share."""
    flux = [s for s in _real_families()
            if s["expect"] == "lora:flux" and "add_k_proj" in "\n".join(s["keys"])]
    assert flux, "fixture should contain a Flux sample carrying add_k_proj"
    for s in flux:
        p = _from_keys(tmp_path, s["keys"], name=s["source"].replace("/", "_") + ".st")
        assert detect(p)[0] == "lora:flux"


def test_a_wan_lora_is_not_mistaken_for_the_adaln_lineage(tmp_path):
    """Both write diffusion_model.blocks.N.cross_attn.* -- 22 files in the
    local corpus share that prefix with Wan. ffn separates them; cross_attn
    alone does not."""
    wan = [s for s in _real_families() if s["expect"] == "lora:wan"]
    for s in wan:
        p = _from_keys(tmp_path, s["keys"], name=s["source"].replace("/", "_") + ".st")
        assert detect(p)[0] == "lora:wan"


def test_a_delta_whose_increment_is_spelled_down_weight_is_still_a_delta(tmp_path):
    """XLabs writes double_blocks.0.processor.proj_lora1.down.weight -- no
    lora_down, no .alpha, no lora_A. LORA_MARKERS missed it entirely, so the
    file fell through to the full-model branch and came back plain `unknown`:
    not even recognised as an increment."""
    p = write(tmp_path, [
        "double_blocks.0.processor.proj_lora1.down.weight",
        "double_blocks.0.processor.proj_lora1.up.weight",
    ])
    kind, why = detect(p)
    assert kind == "lora:flux"
    assert "delta weights" in why


def _oos_families():
    import gzip
    import pathlib
    f = pathlib.Path(__file__).parent / "fixtures" / "oos_families.json.gz"
    return json.loads(gzip.decompress(f.read_bytes()))["samples"]


@pytest.mark.parametrize("s", _oos_families(), ids=lambda s: s["source"])
def test_out_of_sample_headers_classify_as_recorded(s, tmp_path):
    """Headers found *after* the Flux and Wan rows were written.

    real_families.json.gz cannot show that those rows generalise: they were
    picked while the rows were being written. These were checked against a
    ground truth the classifier had no part in choosing -- the uploader's own
    base_model tag -- and four of them are recorded as unknown because unknown
    is the correct answer, not because the table is incomplete.
    """
    p = _from_keys(tmp_path, s["keys"], s.get("modelspec_architecture"),
                   name=s["source"].replace("/", "_") + ".st")
    assert detect(p)[0] == s["expect"]


def test_the_adaln_near_miss_does_not_read_as_wan(tmp_path):
    """The Wan row asks for blocks. + cross_attn + ffn. This real Anima delta
    carries diffusion_model.blocks. AND cross_attn -- two of the three -- and
    stays dit-adaln only because it has no ffn. Wan sits above dit-adaln, so
    this is the file that would break first if the markers or the order drift.
    """
    near = [s for s in _oos_families() if s["expect"] == "lora:dit-adaln"]
    assert near, "fixture should keep a dit-adaln near-miss"
    for s in near:
        joined = "\n".join(s["keys"])
        assert "cross_attn" in joined and "blocks." in joined
        assert "ffn" not in joined
        p = _from_keys(tmp_path, s["keys"], s.get("modelspec_architecture"),
                       name="nearmiss.st")
        assert detect(p)[0] == "lora:dit-adaln"


def test_transformer_blocks_alone_names_no_family_in_either_direction(tmp_path):
    """A FLUX.2 Klein delta and a Qwen-Image delta ship the identical shape:
    transformer.transformer_blocks.N.attn..., no single_transformer_blocks and
    no add_k_proj. They are different families. Any rule that named one would
    name the other wrong, so both must stay lora:unknown -- this is the pair
    that keeps transformer_blocks out of the table."""
    pair = [s for s in _oos_families()
            if s["expect"] == "lora:unknown"
            and "transformer_blocks" in "\n".join(s["keys"])
            and "add_k_proj" not in "\n".join(s["keys"])
            and "single_transformer_blocks" not in "\n".join(s["keys"])]
    assert len(pair) >= 2, "the ambiguous pair is the point of this test"
    assert len({s["base_model_tag"].split("/")[0] for s in pair}) > 1, \
        "the pair must come from two different base models"
    for s in pair:
        p = _from_keys(tmp_path, s["keys"], s.get("modelspec_architecture"),
                       name=s["source"].replace("/", "_") + ".st")
        assert detect(p)[0] == "lora:unknown"


def _full_models():
    import gzip
    import pathlib
    f = pathlib.Path(__file__).parent / "fixtures" / "full_models.json.gz"
    return json.loads(gzip.decompress(f.read_bytes()))["samples"]


@pytest.mark.parametrize("s", _full_models(),
                         ids=lambda s: s["source"] + ":" + s["file"].split("/")[-1])
def test_full_model_headers_classify_as_recorded(s, tmp_path):
    """Real checkpoints, both packaging conventions per family.

    ComfyUI ships one diffusion-model file and diffusers ships a sharded
    transformer/ directory; the two do not carry the same top-level prefixes.
    A row fitted to one of them passes its own tests and fails half its users.
    """
    p = _from_keys(tmp_path, s["keys"], s.get("modelspec_architecture"),
                   name=s["file"].split("/")[-1])
    assert detect(p)[0] == s["expect"]


def test_no_real_header_matches_two_checkpoint_rows(tmp_path):
    """The checkpoint rows are documented as order-independent. That is a claim
    about the evidence, not a wish: it holds only while no real header satisfies
    two rows at once. If one ever does, the order silently starts deciding and
    this is where it shows up."""
    from safetensors_arch import _CHECKPOINT_MARKERS
    for s in _full_models() + _oos_families() + _real_families():
        top = {k.split(".")[0] for k in s["keys"]}
        matched = [kind for kind, need in _CHECKPOINT_MARKERS if need <= top]
        assert len(matched) <= 1, "%s matches %s" % (s["source"], matched)


def test_a_flux_delta_is_not_read_as_a_flux_checkpoint(tmp_path):
    """The ordering trap in its newest shape.

    No sampled vendor delta carries both double_blocks and single_blocks as
    top-level prefixes -- XLabs writes only double_blocks, kohya folds the whole
    path into one underscored prefix -- so the real files do not currently reach
    the flux-checkpoint row at all. That is an observation about today's
    trainers, not a guarantee, so the guarantee is tested directly: a delta
    carrying both prefixes must still be a delta, because the delta question is
    asked first.
    """
    real = [s for s in _real_families()
            if s["expect"] == "lora:flux"
            and "double_blocks" in {k.split(".")[0] for k in s["keys"]}]
    assert real, "fixture should hold a Flux delta with a bare block prefix"
    for s in real:
        p = _from_keys(tmp_path, s["keys"], name=s["source"].replace("/", "_") + ".st")
        assert detect(p)[0] == "lora:flux"

    both = write(tmp_path, [
        "double_blocks.0.img_attn.proj.lora_down.weight",
        "double_blocks.0.img_attn.proj.lora_up.weight",
        "single_blocks.0.linear1.lora_down.weight",
        "single_blocks.0.linear1.lora_up.weight",
    ], name="both_prefixes.safetensors")
    kind, why = detect(both)
    assert kind == "lora:flux", why
    assert "delta" in why


def test_a_delta_is_loadable_on_its_own_family_and_not_on_a_neighbour():
    """Each new checkpoint kind is only useful if is_compatible knows it."""
    for lora, base in (("lora:flux", "flux-checkpoint"),
                       ("lora:wan", "wan-checkpoint"),
                       ("lora:qwen-image", "qwen-image-checkpoint")):
        assert is_compatible(lora, base) is True
        assert is_compatible(lora, "sdxl-checkpoint") is False
    assert is_compatible("lora:sdxl", "flux-checkpoint") is False
