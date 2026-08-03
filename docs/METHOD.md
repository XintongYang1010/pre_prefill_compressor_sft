# Method

## 1. Exact training boundary

The system adds a trainable visual-token compressor after the model's native visual merger and before LLM prefill:

```text
image
  -> frozen vision tower
  -> frozen native main merger and DeepStack mergers
  -> trainable compressor
  -> frozen language model
```

The language model remains in the autograd graph for the student path, so task and distillation losses can propagate through it to the compressor. Its parameters do not update. Calling this a full visual-head or Vision Encoder SFT would be inaccurate.

### Main token and DeepStack feature

A **main visual token** is a token in the primary visual sequence consumed by the LLM. With `2 x 2 -> 1` compression, one compressed main token represents a spatial group of four native-merger main tokens. DeepStack features are separate multi-scale side inputs injected at selected LLM depths; they are compressed with the same spatial grouping but are not themselves “main tokens.”

## 2. Compressor

For each image/frame region, let the native-merger feature grid be
`X in R^(T x H x W x D)`. With spatial stride `s=2`, the compressor:

1. replicates the final valid row/column only inside the current image region
   when `H` or `W` is odd;
2. forms non-overlapping spatial groups without crossing image boundaries;
3. computes the historical grouped teacher target as a four-slot mean,
   including replicated odd-edge slots;
4. predicts the compressed representation through a small bottleneck MLP,
   without a direct teacher-mean skip connection;
5. emits one token per group and a local reconstruction used only for protection loss.

The same `ImageTokenPlan` is applied to the main stream, all DeepStack branches, image placeholders, and the LLM-facing position grid. This prevents a quality result from being confounded by inconsistent token ordering or position IDs.

The replicated-slot teacher/reconstruction rule gives odd-edge features repeated
weight. It is disclosed as the exact current recipe, not described as an
unbiased masked estimator. A valid-slot-masked alternative remains an ablation.

## 3. Three-stage training history

### Stage 0: feature-only warm-up

Teacher source: frozen, uncompressed native-merger features for the main stream and multi-scale DeepStack branches.

Student: the new compressor. Only its weights update. The objective protects grouped local reconstruction and normalized main/DeepStack representations. This stage teaches structural compression without task labels.

### Stage 1: naive task-aware SFT

The warm-started compressor was adapted with task supervision while the backbone remained frozen. On the fixed Dev256, recall changed from `70%` to `10%`, F1 from `66.67%` to `14.29%`, valid structured outputs from `238/256` to `125/256`, and false negatives from `3` to `9`. This failure shows that “only a small module is trainable” does not by itself prevent catastrophic behavior drift.

### Stage 2: RetentionKD

RetentionKD uses two reference sources:

- **Behavior teacher:** a frozen copy of the pre-retention compressor and frozen LLM, evaluated with the production prompt contract.
- **Feature teacher:** uncompressed native-merger region features, used for local/main/DeepStack preservation.

The student starts from the same pre-retention compressor checkpoint. It has the same compressed-token budget as the behavior teacher; the goal is task repair without discarding the earlier visual representation.

## 4. Objectives

The Full objective is

```text
L_full = w_ce L_ce
       + w_feat L_feature
       + w_jsd L_response_jsd
       + w_vsd L_vsd
       + w_vlad L_vlad
```

The matched Control keeps only `L_ce + L_feature`.

### Hard-label CE

Task supervision uses a configurable auxiliary non-reasoning prompt and a compact target schema. The production behavior prompt remains separate, preventing reasoning-control tokens from leaking into the supervised target. The clean-room package accepts tensor-level targets and intentionally does not encode a private prompt or label schema.

Class weights use the effective-number formulation and an explicit upper bound, which limits minority-class amplification under a highly imbalanced private dataset.

### Response JSD

At aligned rollout positions, teacher and student next-token distributions are compared with generalized Jensen-Shannon divergence. Training mixes teacher-generated and student-generated prefixes, reducing the gap between gold-prefix distillation and inference-time student states.

### Visual-semantic distillation (VSD)

A bounded set of final-layer visual positions is projected through the frozen LM head. The loss uses reverse KL,
`KL(p_student || p_teacher)`, to protect semantic distributions associated with compressed visual locations.

### Vision-language affinity distillation (VLAD)

Normalized visual and language hidden states form an affinity matrix. Smooth L1 distance between student and teacher affinity matrices protects cross-modal relationships even when individual features differ.

### Feature protection

Local reconstruction plus normalized main and DeepStack feature errors anchor the compressor to the uncompressed visual representation.

## 5. Gradient budgeting

Loss weights alone do not control the optimizer contribution of heterogeneous objectives. The training recipe therefore:

1. computes compressor gradients separately for each active objective;
2. averages each objective gradient across data-parallel workers;
3. tracks an EMA of objective gradient norms;
4. applies bounded inverse-norm weights in `[0.25, 4.0]`;
5. caps the weighted VSD contribution at `1.0 x` the hard-CE contribution;
6. combines gradients and applies global norm clipping at `1.0`.

The clean-room implementation follows this as an explicit vector update: it
DP-averages each objective's parameter-gradient tensors, computes norms and
pairwise cosines from those averaged vectors, applies the CE anchor, installs
the clipped joint vector in `parameter.grad`, and only then steps the optimizer.
It is not implemented as a weighted scalar loss followed by a single backward.

An activation threshold of `1e-12` prevents numerical noise from being treated as an active objective. The selected training run used AdamW, learning rate `5e-6`, global batch 8, and checkpoints after updates 8, 32, 64, and 125.

## 6. Why checkpoint u64 was selected

The matched sweep showed that Full at updates 8 and 32 did not consistently beat Control, Full-u64 improved the matched task metrics, and Full-u125 regressed. The evidence therefore supports checkpoint selection and early stopping around u64, not a claim that more training data or a full epoch must be better.

Durable checkpoints include compressor, optimizer, scheduler, gradient-budget state, RNG state, objective trajectory, and a configuration digest. A resume is valid only under the same arm and semantic/data contract.
