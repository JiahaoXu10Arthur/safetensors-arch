# safetensors-arch

[![test](https://github.com/JiahaoXu10Arthur/safetensors-arch/actions/workflows/test.yml/badge.svg)](https://github.com/JiahaoXu10Arthur/safetensors-arch/actions/workflows/test.yml)

Tell what a `.safetensors` file actually is — by reading its tensor table,
not its filename and not the metadata the model host attached to it.

```console
$ python -m safetensors_arch models/loras/
base_v10.safetensors  sdxl-checkpoint     2515 tensors; model. + conditioner. + first_stage_model. (full SDXL-family checkpoint)
style-a.safetensors   lora:sdxl           2958 tensors of delta weights; keys carry down_blocks / up_blocks / mid_block (trainer declared stable-diffusion-xl/lora)
style-b.safetensors   lora:dit-adaln      1428 tensors of delta weights; keys carry cross_attn_k_proj (trainer declared anima-preview/lora)
```

```python
from safetensors_arch import detect, is_compatible

kind, why = detect("some.safetensors")     # ("lora:sdxl", "2958 tensors of ...")
is_compatible("lora:sdxl", "dit-adaln")    # False  -> would silently do nothing
is_compatible("lora:unknown", "dit-adaln") # None   -> not enough information
```

No dependencies. Reads only the file header, never the weights — classifying a
directory of 20 GB checkpoints takes milliseconds.

Run over one real collection of 269 LoRA files, it returns a kind for every one
and no `unknown`, and the split it reports — 169 `lora:dit-adaln`, 99
`lora:sdxl`, 1 `lora:qwen-image` — matches the three folders their owner had
filed them into, file for file. That is the useful shape of evidence for a
classifier: not that it agrees with itself, but that it agrees with a judgement
made independently of it.

## Why this exists

Loading a LoRA built for one base-model family onto a different family is a
**silent failure**. Nothing raises. An image comes out, and it looks fine. The
LoRA simply contributed nothing.

That is the worst shape a bug can have: silent, reproducible, and it looks
like a *result*. You conclude "this LoRA isn't very good" and move on, and the
conclusion is garbage — you never loaded it.

The two things people normally check both lie in practice:

- **Filenames lie.** A file whose name promised an anime SD model turned out
  to be a Flux.2 workflow. Names are written by people, for people.
- **Vendor metadata lies.** A LoRA distributed with `baseModel` set to one
  family had the tensor layout of another. The field is self-reported and
  frequently just wrong.

The tensor table does not lie. An SDXL UNet delta has the UNet block names, in
whichever spelling its trainer used — `input_blocks` / `output_blocks` /
`middle_block`, or `down_blocks` / `up_blocks` / `mid_block`. A Cosmos-style
DiT delta has `adaln_modulation`. Flux has `double_blocks` / `single_blocks`,
or `single_transformer_blocks` from diffusers. Wan has `blocks.`
and `cross_attn` with `ffn`. Qwen-Image has `transformer_blocks` with
an `add_k_proj` text branch. Those names come from the code that produced the
weights.

Those markers are not guesses. They were read off ten real vendor headers
pulled from public repositories, and the order they are tested in was settled
by which substrings actually separate the families on that evidence — Flux and
Qwen share `add_k_proj`, Wan and the adaln lineage share
`diffusion_model.blocks.`, and every SDXL LoRA carries `transformer_blocks`.
The headers are committed as a fixture.

Those ten were chosen while the rules were being written, so they cannot show
the rules generalise. A second batch was pulled afterwards and checked against
a judgement the classifier had no part in making — the uploader's own
`base_model` tag. Across 59 repositories (23 distinct key layouts), 54 agree
and **not one file was given a wrong family**: every disagreement came back
`unknown`. Four of those are pinned as `unknown` on purpose, including a
FLUX.2 and a Qwen-Image delta that ship byte-identical key shapes — naming
either one would name the other wrong. That batch is a second fixture.

## The ordering trap

This is the part worth stealing even if you never use the package.

Some DiT LoRA keys look like this:

```
diffusion_model.blocks.0.adaln_modulation_cross_attn.1.lora_A.weight
```

That single key carries **both** a base-model structure marker
(`adaln_modulation_cross_attn`) and a delta marker (`lora_A`). If your
classifier asks "is this a full model?" before "is this a delta?", every LoRA
in that family comes back as a full checkpoint.

Downstream, those files get routed to a loader that cannot open them, and the
error message points at the loader — nowhere near the actual cause.

So the order is fixed and load-bearing:

1. **Is this an increment?** (`lora_down`, `.lora_A`, `lora_up`, `lora_te`, …)
2. Only then: **an increment of what?**

`tests/test_detect.py::test_dit_lora_is_not_a_checkpoint` pins this down.

## Design notes

**Every answer comes with its reason.** `detect()` returns
`(kind, why)`, and the CLI prints both. A classifier you cannot argue with is
a classifier you cannot trust — when it is wrong you need to see *which
marker* it matched to know whether to fix the file or fix the rule.

**Unknown is a real answer.** `lora:unknown` means "definitely a delta, target
family not recognised". It is not folded into a nearby guess. An earlier
version of this logic had a fallback that returned a concrete family whenever
it saw a checkpoint-style loader, on zero positive evidence — it confidently
misfiled a whole family of models, and the confidence was the damaging part.

**`is_compatible()` returns `None`, not `False`, when it does not know.** The
guard exists to stop the *definitely wrong* pairing. If it blocked everything
it did not recognise, it would be useless the first time someone brings a
family the table has never seen.

**The trainer's own declaration only speaks where the keys do not.** Some
trainers write `modelspec.architecture` into the header. It is worth reading —
the code that produced the weights wrote it — but it is a claim, and the point
of this package is that the tensor table is not. So it never overrules a family
the keys named; it is consulted only once the file is established as a delta
and no key-based test has recognised the family, where the alternative is
`lora:unknown`. The answer says when it came from there.

## Recognised kinds

| kind | what it is |
|---|---|
| `lora:sdxl` | delta over an SDXL-family UNet (SDXL, Pony, Illustrious, NoobAI…) |
| `lora:dit-adaln` | delta over an adaptive-layernorm cross-attention DiT |
| `lora:qwen-image` | delta over the Qwen-Image DiT |
| `lora:unknown` | definitely a delta, family not recognised |
| `sdxl-checkpoint` | full SDXL-family checkpoint |
| `dit-adaln` | full adaln cross-attention DiT |
| `unknown` | not recognised, or not readable as safetensors |

Adding a family is a couple of lines in `detect()` plus a test with a
synthetic header. Pull requests welcome — especially ones that bring a real
file's key list, since that is the only way the table grows correctly.

## Prior art

Reading a `.safetensors` tensor table to work out what a file is has been done
before, so that is not the claim here.

- **[SafetensorsModelInspector][smi]** classifies by tensor keys across a wide
  family table — Flux, SDXL, Wan, Hunyuan, Qwen and more — and recognises
  LoRA/LyCORIS/LoHa/LoKr/DoRA. It is a PyQt6 desktop application, so it wants a
  GUI stack and a human at the keyboard.
- **[sai_model_spec_tools][sai]** is mostly Ruby scripts for *writing* SAI
  model-spec metadata into a file, plus one Python script that reads it back
  and flags a `CONTRADICTION DETECTED` when the declared base model disagrees
  with what the keys suggest. Its author calls the scripts basic sketches.

What neither of them answers is the question this package exists for:
**given this delta and this base model, would attaching them do anything at
all?** `is_compatible()` is one call, it returns `None` rather than guessing
when the pair is unfamiliar, and it is the whole reason to prefer a
zero-dependency library over a desktop inspector.

Two smaller differences worth stating plainly, since they are the practical
ones: this reads only the header, never the weights, so a directory of 20 GB
checkpoints classifies in milliseconds; and it imports nothing outside the
standard library, so it drops into a build step or a CI job without pulling a
GUI toolkit behind it.

The [ordering trap](#the-ordering-trap) above is the part I would most like
someone to take even if they use one of the others instead.

## Install / test

```console
pip install safetensors-arch
```

Or from a clone, to run the suite:

```console
pip install -e ".[test]"
pytest -q
```

Tests build synthetic safetensors headers in a temp directory, so no model
weights are needed to run them.

See [DESIGN.md](DESIGN.md) for why it is shaped this way, what was
rejected, and what to check before changing it.

## License

MIT

[smi]: https://github.com/MNeMoNiCuZ/SafetensorsModelInspector
[sai]: https://github.com/FNGarvin/sai_model_spec_tools
