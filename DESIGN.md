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

## Decision: family order is settled by a matrix, not by taste

Flux and Wan were added by pulling **ten real vendor headers** over HTTP Range
from public HuggingFace repositories — only the header, never the weights,
which is the same trick `detect()` itself uses. Then measuring which substrings
actually separate the families, across those ten plus the 266 local files:

| substring | FLUX | QWEN | WAN | adaln | SDXL |
|---|---|---|---|---|---|
| `single_transformer_blocks` | 2/4 | 0/3 | 0/3 | 0/165 | 0/101 |
| `double_blocks` | 2/4 | 0/3 | 0/3 | 0/165 | 0/101 |
| `add_k_proj` | **2/4** | **2/3** | 0/3 | 0/165 | 0/101 |
| `ffn` | 0/4 | 0/3 | **2/3** | **0/165** | 0/101 |
| `diffusion_model.blocks.` | 0/4 | 0/3 | **2/3** | **22/165** | 0/101 |
| `transformer_blocks` | 2/4 | 3/3 | 0/3 | 0/165 | **101/101** |

Three things fall out of that table, and none of them were guessable:

- **Flux must be decided before Qwen.** The diffusers layout writes
  `transformer.single_transformer_blocks.N.attn...` and carries `add_k_proj`,
  which is both halves of the Qwen test. Two of the four real Flux LoRAs were
  coming back `lora:qwen-image` — a confident wrong family, the failure this
  package exists to prevent.
- **`ffn` separates Wan from the adaln lineage; `diffusion_model.blocks.` does
  not.** 22 local Anima files share that prefix. Writing the Wan rule around
  the prefix would have manufactured a fresh false answer for all 22.
- **Qwen goes last.** `transformer_blocks` alone is evidence of nothing: every
  one of the 101 SDXL LoRAs carries it inside
  `lora_unet_down_blocks_2_attentions_1_transformer_blocks_8_...`.

Two of the ten samples are pinned as `lora:unknown` on purpose. One carries
only the generic `transformer_blocks`; the other names no family in its keys at
all and its `ss_base_model_version` says `minimax_h3` while its repository is
named for Wan. Neither can be resolved from the file, and inventing a marker to
claim them would be exactly the failure the rest of this document is about.

Adding a family means extending the matrix, not adding a row to the table and
hoping. The fixture is `tests/fixtures/real_families.json.gz` — complete key
lists in file order, gzipped because they are 489 KB of highly repetitive text
against a 16 KB package. **Do not truncate them**: a shortened list hides
exactly the bug this module has been fixed for twice, where a marker sits past
a scan window.

## Decision: the rows were checked again on headers they did not choose

The ten headers above cannot show that the Flux and Wan rows *generalise* —
they were picked while those rows were being written. So a second, independent
batch was pulled the same way, afterwards, and checked against a ground truth
the classifier had no part in choosing: **the uploader's own `base_model:` tag**
on HuggingFace, i.e. what the person who trained the delta says they trained it
on. Fifty-nine repositories, which collapse to **23 distinct key lists** — a
`ntc-ai` slider and its five siblings are one structure, not six, and counting
them as six would have overstated the evidence.

**54 of 59 agree with the tag, and not one file was given a wrong family.**
Every disagreement landed on `unknown` or `lora:unknown`. For this package that
is the property that matters: a wrong family is a silent failure at load time,
`unknown` is a question.

The five disagreements, each recorded in `tests/fixtures/oos_families.json.gz`:

- **The Wan row was too strict, and is now looser.** A real Wan 2.2 delta
  writing `blocks.N.cross_attn...` fell through to `lora:unknown` purely
  because it lacked the `diffusion_model.` prefix. The row now asks for
  `blocks.` and `cross_attn` with `ffn`, and `ffn` is still what separates it
  from the adaln lineage. This was measured before it was written: across
  **335 real headers** — 266 local files, the 59 out-of-sample repos, and the
  ten fixtures — the looser pair matches **exactly one** file it did not match
  before, and that file is the Wan delta. dit-adaln, which sits *below* Wan and
  whose `cross_attn_k_proj` marker contains `cross_attn`, lost nothing. The
  fixture keeps `AX1Y2JP/anima_extracted_lora` as the near-miss that pins it:
  two of the three conditions met, held out by the absence of `ffn` alone.
- **A FLUX.2 Klein delta and a Qwen-Image delta ship the identical shape.**
  Both are `transformer.transformer_blocks.N.attn.to_k.lora...` with no
  `single_transformer_blocks` and no `add_k_proj`. They are different families
  and the header does not distinguish them. Both stay `lora:unknown`; a rule
  naming either would name the other wrong. This is the same shape the first
  fixture already pinned in `flymy-ai/qwen-image-realism-lora` — what was one
  oddity is now a class, confirmed from two families.
- **Wan 2.1 in the PEFT layout stays unknown on purpose.**
  `base_model.model.blocks.N.attn1.to_k.lora_A`, no `ffn`, no `cross_attn`.
  Telling it from a diffusers-layout SDXL delta would need negative evidence —
  `blocks` but no `down_blocks` — weaker than anything else in the table.
- **Full models outside SDXL have no kind.** A merged FLUX.1 model carries
  `double_blocks` and `single_blocks`, so its family is not in doubt, but it is
  not a delta and there is no `flux-checkpoint` beside `sdxl-checkpoint`. It
  returns plain `unknown`. Recorded so the asymmetry stays visible.

The eight SDXL structures from that batch are deliberately **not** in the
fixture: SDXL is the best-covered row, unchanged by any of this, and those
eight were over half the bytes.

## Decision: full models get a family too, matched on prefixes not substrings

Until 0.3.0 only SDXL and the adaln lineage had a full-model kind. A merged
FLUX.1 model carrying `double_blocks` and `single_blocks` — its family not in
any doubt — came back plain `unknown`, because there was no `flux-checkpoint`
to return. `AX1Y2JP/FLUX.1-schnell-krea-lora-merged` in the out-of-sample batch
is the file that found that asymmetry, and it is kept in the fixture.

Full models are matched on **top-level key prefixes** — the segment before the
first dot — not on substrings. A checkpoint carries whole subtrees, so the
prefixes are the honest evidence; a substring test would collide with the delta
vocabulary immediately.

| top-level prefixes | FLUX | WAN | QWEN | SDXL ckpt | 266 local deltas |
|---|---|---|---|---|---|
| `double_blocks` + `single_blocks` | **1/1** | 0/2 | 0/2 | 0/2 | 0/266 |
| `blocks` + `patch_embedding` + `time_projection` | 0/1 | **2/2** | 0/2 | 0/2 | 0/266 |
| `transformer_blocks` + `time_text_embed` + `txt_norm` | 0/1 | 0/2 | **2/2** | 0/2 | 0/266 |
| `model` + `conditioner` + `first_stage_model` | 0/1 | 0/2 | 0/2 | **2/2** | 0/266 |

Two things that were not guessable:

- **Every family has to be sampled in both packaging conventions.** ComfyUI
  ships one diffusion-model file; diffusers ships a sharded `transformer/`
  directory. They do not carry the same prefixes — the ComfyUI repackaging of
  Wan 2.1 adds `img_emb` and `control_adapter` that the diffusers one lacks. A
  row fitted to one convention passes its own tests and fails half its users,
  so both are in `tests/fixtures/full_models.json.gz` for every family that has
  both.
- **A middle shard of a sharded checkpoint is correctly `unknown`.** Shards 3
  and 5 of Qwen-Image's nine carry nothing but `transformer_blocks` — the
  substring this document already established names no family, since every SDXL
  LoRA carries it. Shard 1 answers because it holds the embedding subtrees;
  shards 2..n-1 genuinely do not say what they belong to. Three of them are
  pinned as `unknown` in the fixture so nobody "fixes" it later by guessing
  from `transformer_blocks` — a guess that would be wrong on SDXL and on the
  FLUX.2 delta two sections above.

The rows are disjoint on every real header held here, so their order does not
decide anything; a test asserts that rather than trusting it, because the
moment two rows overlap the order starts deciding silently.

The delta question is still asked first, and it has to be. A Flux delta's
top-level prefixes are `double_blocks` — one half of the `flux-checkpoint`
row. No sampled vendor delta carries both halves, but that is an observation
about today's trainers and not a guarantee, so the guarantee is tested
directly with a delta that carries both.

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

**`lora_te` is not an SDXL marker, and it no longer sits in the SDXL row.**
In all 16 files above, `input_blocks` and `output_blocks` were absent; the only
thing making them look SDXL was `lora_te`, which is the sd-scripts text-encoder
prefix that Anima's trainer (`networks.lora_anima`, an sd-scripts derivative)
emits too. Scanning every key only **masked** that, because the adaln branch is
tested first.

Removing it was blocked on a claim that turned out to be a guess: *"no real
`lora_te`-only SDXL LoRA on hand."* Measured instead, 2 of the 99 `lora:sdxl`
files were exactly that — and the reason they were is that they are written in
**diffusers** UNet naming (`down_blocks` / `up_blocks` / `mid_block`) while the
branch only knew **compvis** naming (`input_blocks` / …). Both spellings are
now matched, which leaves **0 of 266 files** relying on `lora_te`, so it could
come out.

Recheck this before adding a family: the guard is that every file the branch
claims carries a real structural marker. A text-encoder-only SDXL LoRA — no
UNet keys at all — now falls to the declaration or to `lora:unknown`. That is
the intended direction: `lora:unknown` is an answer this package is content to
give, a confident wrong family is not.

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
