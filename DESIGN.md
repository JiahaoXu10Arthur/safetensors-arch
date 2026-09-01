# Design notes

Why the ordering matters, what "unknown" is for, and what to know before
changing it. The README says what the package does; this says why.

## The rule the whole thing follows

> **Read a classification out of the data. Never invent one.**

The tensor table is data the training code emitted. A filename is a claim a
person typed. Vendor metadata is a claim a publisher typed. This package reads
the first and treats the other two as untrusted — and when the data does not
say, it says so rather than picking the nearest plausible family.

## Decision: decide *increment or not* before *increment of what*

Some DiT LoRA keys look like this:

```
diffusion_model.blocks.0.adaln_modulation_cross_attn.1.lora_A.weight
```

That single key carries **both** a base-model structure marker
(`adaln_modulation_cross_attn`) and a delta marker (`lora_A`). Ask "is this a
full model?" first and every LoRA in that family comes back a checkpoint.

Downstream those files get routed to a loader that cannot open them, and the
error points at the loader — nowhere near the cause. The order is fixed and
load-bearing, and `tests/test_detect.py::test_dit_lora_is_not_a_checkpoint`
pins it.

## Decision: `unknown` is a real answer

`lora:unknown` means "definitely a delta, target family not recognised". It is
not folded into a nearby guess.

An earlier version had a fallback that returned a concrete family whenever it
saw a checkpoint-style loader, on zero positive evidence. It misfiled a whole
family of models, and **the confidence was the damaging part** — a wrong answer
that announces itself as uncertain costs a second look; one that announces
itself as certain costs a wrong conclusion.

## Decision: `is_compatible()` returns `None`, not `False`, when it does not know

The guard exists to stop the *definitely wrong* pairing. If it blocked
everything it did not recognise it would be useless the first time someone
brought a family the table has never seen, and it would be turned off.

## Decision: every answer carries its reason

`detect()` returns `(kind, why)` and the CLI prints both. A classifier you
cannot argue with is one you cannot trust: when it is wrong you need to see
*which marker* it matched to know whether to fix the file or fix the rule.

## Decision: the trainer's own declaration speaks only where the keys do not

Some trainers write `modelspec.architecture` into the header. It is useful —
written by the code that produced the weights, not by whoever published them —
but it is still a claim someone typed, and this package's whole position is
that the tensor table is not. So it never overrules a family the keys named.
It is consulted at one point only: after the file is established as a delta
and after every key-based family test has come back empty, where the choice is
between a named family and `lora:unknown`.

It also does not get to borrow the keys' reasons. A file declaring the adaln
lineage whose keys carry no adaln marker used to come back
`lora:dit-adaln, keys carry adaln_modulation / cross_attn_k_proj` — the right
answer citing evidence it never had. A declared family now says so.

## What the evidence actually is

Over one collection of 269 LoRA files this returns a kind for every one with no
`unknown`, and the split it reports — 169 `lora:dit-adaln`, 99 `lora:sdxl`, 1
`lora:qwen-image` — matches the three folders their owner had filed them into,
file for file.

That is the useful shape of evidence for a classifier: agreement with a
judgement made **independently of it**, not agreement with itself.

The README once carried a demo row named `mislabelled.safetensors` classified
as `lora:sdxl`, meant to show classification following the tensor table rather
than the name. It was removed. Anonymising the original filenames made the row
circular — a file called "mislabelled" being called mislabelled proves nothing,
and the name was doing the work the data was supposed to do. Checking whether a
real mismatch existed to put back: in that collection of 269, zero
disagreements between folder and detected family.

## Before you change anything

**Both questions are asked over every key, and the argument for sampling was
wrong twice.** `detect()` originally read a 400-key prefix for both. The delta
question was fixed first: a LoRA whose delta markers sorted past key 400 came
back a full checkpoint. The family question was left sampling, on an asymmetry
argument written into the fix — that a missed delta marker answers confidently
and wrongly, while a missed family marker only costs a `lora:unknown`, an answer
this package is content to give.

That argument was false, and the code it was written next to disproves it. The
family branches are **ordered**. A missed marker does not fall through to
`lora:unknown` at the bottom; it falls through to the *next branch*, which
answers confidently. An Anima delta writes its 588 text-encoder tensors before
its adaln blocks, so `cross_attn_k_proj` sits at index 588, the prefix ended at
400, and the SDXL branch fired on `lora_te` alone. 16 of 266 files in a real
collection came back `lora:sdxl` while every metadata field on them said
`anima` — and `is_compatible` then reported them loadable onto an SDXL
checkpoint, which is precisely the load-does-nothing failure this package was
built to catch. Fixing the window changed those 16 and nothing else.

The lesson is not "scan everything." It is that **an escape hatch is only as
safe as the branch it actually lands in**, and the sampling argument had never
been checked against the control flow one screen below it.

**`lora_te` alone is not an SDXL marker, and the SDXL branch treats it as one.**
In all 16 files above, `input_blocks` and `output_blocks` were absent; the only
thing making them look SDXL was `lora_te`, which is the sd-scripts text-encoder
prefix that Anima's trainer (`networks.lora_anima`, an sd-scripts derivative)
emits too. Scanning every key **masks** this rather than removing it, because
the adaln branch is tested first and now sees its marker. It is left standing
deliberately: tightening the branch to require `input_blocks` / `output_blocks`
would need a real `lora_te`-only SDXL LoRA to prove it breaks nothing, and no
such file was on hand. If you have one, that is the pull request.

**Tests build synthetic headers.** `struct.pack("<Q", len(blob)) + blob` plus
JSON — no model files in the repo, and none needed, because only the header is
ever read.

That is sufficient for parsing, and it is *not* sufficient for the family
table. A synthetic header proves the matcher fires on keys you already knew
about. Adding a family is a couple of lines in `detect()` plus a test, and the
useful pull request brings a **real file's key list**, since that is the only
way the table grows correctly.

**Header only, never weights.** Classifying a directory of 20 GB checkpoints
takes milliseconds because nothing after the header is read. Keep it that way;
`HEADER_LIMIT` exists to refuse a "header" that could not be one.

**Zero third-party dependencies is a hard constraint.**

```console
python -I -c "
import sys; sys.path.insert(0, '.')
before = set(sys.modules)
import safetensors_arch
new = [m for m in set(sys.modules) - before
       if 'site-packages' in str(getattr(sys.modules[m], '__file__', '') or '')]
print('third-party:', new or 'none')"
```

**Prior art is named in the README on purpose.** Classifying a `.safetensors`
by its tensor table has been done before, and a reader who finds that out
afterwards should not be the one to discover it. What is left is
`is_compatible()` — whether a given delta would do anything at all on a given
base — plus header-only reads and no dependencies. Keep the claim that narrow.
