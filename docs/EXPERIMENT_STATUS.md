# Experiment Status

Last evidence refresh: **2026-08-03 19:42 China Standard Time**.

## 1. Failure that motivated RetentionKD

The first task-aware compressor SFT failed on a fixed private Dev256:

| Metric | Before task SFT | After task SFT |
| --- | ---: | ---: |
| Recall | 70.00% | 10.00% |
| F1 | 66.67% | 14.29% |
| Valid structured outputs | 238 / 256 | 125 / 256 |
| False negatives | 3 | 9 |

This is evidence of behavior collapse under naive task adaptation, not evidence that SFT in general is unsuitable.

## 2. Matched RetentionKD training

Full and Control started from the same pre-retention compressor and used the same private Train1,000 order, seed, global batch 8, AdamW optimizer, learning rate `5e-6`, and 125 updates. Both produced durable checkpoints at u8/u32/u64/u125. Only Full enabled response JSD, VSD, and VLAD in addition to hard CE and feature protection.

The checkpoint sweep selected u64: early Full checkpoints did not consistently beat their matched controls, Full-u64 improved, and u125 regressed. On this first frozen Dev256, Full-u64 versus Control-u64 improved F1 by `8.34pp`, recall by `4.35pp`, and accuracy by `2.73pp`; FP changed `11 -> 7` and FN `8 -> 7`. This is an empirical early-stopping result.

## 3. Independent Dev256 Full versus Control

This confirmation used a private binary-stratified Dev256 that did not overlap the earlier selection Dev256. Both arms used exactly the same 256 members and retained invalid rows in the denominator.

| Model | Precision | Recall | F1 | Accuracy | FP | FN | Valid outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control-u64 | 45.71% | 66.67% | 54.24% | 86.33% | 19 | 8 | 245 / 256 |
| Full-u64 | **58.06%** | **75.00%** | **65.45%** | **89.84%** | **13** | **6** | **246 / 256** |

Full-Control deltas were `+12.35/+8.33/+11.22/+3.52` percentage points for precision/recall/F1/accuracy. Both retained `26.4224%` of visual tokens, so this difference is not caused by unequal compression budgets.

Uncertainty remains material: correctness improved/regressed on 22/13 paired rows, exact two-sided McNemar `p=0.1755`, and the paired-bootstrap F1 95% interval `[-0.0382, 0.2588]` crosses zero.

## 4. Identity comparisons: what is and is not established

### Valid same-member Dev256 #1

The first checkpoint sweep retrospectively reused identity predictions for the exact same 256 members, order, prompt, request seed, and invalid-row denominator. The identity rows came from the complete historical run rather than a rerun inside the Full/Control job. The archived data, manifest, processor, and image limits match, although the older identity schema did not record a separately named image-resolution-contract field. This supports a descriptive matched comparison, not a preregistered superiority test.

| Arm | Precision | Recall | F1 | Accuracy | FN | Valid outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Identity, retrospective matched reference | 60.71% | **73.91%** | 66.67% | 91.02% | **6** | **250 / 256** |
| Full-u64 | **69.57%** | 69.57% | **69.57%** | 91.02% | 7 | 246 / 256 |

Full-u64 had `+8.85pp` precision and `+2.90pp` F1 with equal fixed-denominator accuracy. It also had `-4.35pp` recall, one additional false negative, and `-1.56pp` structured-output validity. The supported wording is therefore:

> Full-u64 exceeded identity on F1 and precision on the first matched Dev256 while preserving accuracy, but it did not dominate identity on recall, false negatives, or output validity.

This is stronger and more defensible than comparing numbers from unrelated splits. It still does not establish statistical significance or natural-distribution full-volume superiority.

### Historical complete Dev2,141 baselines

| Model | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Identity100, no compression | 52.08% | 74.26% | 61.22% | 93.79% |
| Pre-retention compressor | 48.46% | 62.38% | 54.55% | 91.17% |
| Full-u64 | running | running | running | running |

### Why `65.45% > 61.22%` is not yet a valid superiority claim

The Full-u64 F1 of `65.45%` above comes from the independent Dev256, whereas the identity F1 of `61.22%` comes from the complete Dev2,141. They do not share the same members, label prevalence, or denominator. Comparing those two numbers can motivate the ongoing confirmation but cannot establish that Full-u64 beats identity.

### Ongoing same-order Full-u64 Dev2,141

The current C-only run reuses the exact complete identity and pre-retention
baselines and evaluates Full-u64 on the same ordered Dev2,141. At the latest
read-only refresh it had `1,849 / 2,141` durable successful rows (`86.36%`), 0
processing errors, and no finalizer metrics. Attempt records had advanced to
`2,588`; this is a runtime/recovery counter, not a fixed-denominator result.
The conclusion “Full-u64 beats identity100” remains **pending this run's
complete paired result**.

## 5. Token result and remaining gates

The selected compressor reduced visual tokens by about `73.6%` in the relevant matched evaluation and reduced total prompt tokens by about `27.8%` relative to identity. These are token-count results, not measured serving gains.

Still missing:

- completed Full-u64 versus identity/pre-retention Dev2,141;
- component and interaction ablations for JSD, VSD, and VLAD;
- multiple training seeds and closed confidence intervals;
- a second domain or public benchmark for generality;
- held-out private Test evaluation;
- fixed-contract TTFT, prefill latency, throughput, memory, and capacity;
- production or online evidence.

The RetentionKD adaptation used 1,000 task-labeled examples, while the earlier feature-only warm-up used a larger private set. The defensible wording is therefore **limited-label task adaptation**, not “the entire system used only 1,000 examples.”
