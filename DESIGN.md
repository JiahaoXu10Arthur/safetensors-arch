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

**The delta question is asked over every key; the family question is not.**
`detect()` scans all keys for the delta markers and only a 400-key prefix for
the family markers, and the asymmetry is deliberate. A delta marker missed
returns a confident `dit-adaln` for something that is a LoRA — the misfiling
the ordering above exists to prevent. A family marker missed returns
`lora:unknown`, which is an answer this package is content to give. Both sides
sampled the prefix once, and a LoRA whose delta markers sorted past key 400
came back a full checkpoint.

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
