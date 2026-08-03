# Open Evidence and Experiment Backlog

Last live status refresh: **2026-08-03, China Standard Time**.

This ledger distinguishes an absent public artifact, a restricted artifact,
and an experiment that has not produced a result. A reviewer should not treat
those three states as interchangeable.

## Requested-material disposition

| Requested item | Public status | Evidence now supplied | Remaining work |
| --- | --- | --- | --- |
| Actual Qwen integration | Partially available | Sanitized forward pseudocode, tensor invariants, anonymous runtime checks | Exact internal diff is restricted; public end-to-end wrapper pending |
| DeepStack shapes/injection | Available as contract | Main + 3 branches, `[N,5120] -> [N',5120]`; vision taps 8/16/24; decoder consumption after layers 0/1/2 | Public end-to-end trace pending |
| Interleaved M-RoPE | Available as contract | Compacted grid, three-axis position-ID and RoPE-delta recomputation | Public vendor-level unit/E2E test pending |
| Full Dev2,141 and Test | `EXPERIMENT_RUNNING` / `EXPERIMENT_PENDING` | Identity and pre-retention Dev2,141 complete | Full-u64 terminal Dev2,141 and all Test metrics unavailable |
| Per-objective gradient telemetry | Norms available; cosines pending for historical runs | DP-reduced norms, weights, CE cap, joint clip; public code now computes cosines | Log cosine trajectories across updates/seeds; no-cap comparison |
| Stage 0 data/training | Available as aggregates | Train/Dev/Test counts, 3 epochs, 57,930 updates, losses/tokens | Raw data/checkpoints restricted |
| Tried configs/checkpoints/prompts | Partially available | Two arms x four checkpoints; two functional prompt contracts | Complete historical selection ledger pending |
| Prefill/TTFT/memory/QPS | Exploratory diagnostic only | Older non-RetentionKD paired TTFT/token evidence | Fixed-contract selected-Full Stage-P pending |
| FP/FN/invalid costs | Counts available; costs pending | Fixed-denominator error counts and invalid policy | Business-owner utility weights pending |
| Split/dedup/annotation | Partially available | Anonymous sizes/prevalence and exact-key zero-overlap checks | Near-duplicate and annotation-agreement audit pending |

## Live Full-u64 Dev2,141 status

At the latest read-only refresh, Full-u64 had `1,849 / 2,141` durable successful
rows (`86.36%`), `0` processing errors, and no terminal finalizer. The run was
still active. Attempt counters are not row-completion denominators.

```text
Status: EXPERIMENT_RUNNING
Result: not available
No Full-u64 Dev2,141 P/R/F1/Accuracy conclusion is drawn.
```

The frozen Test remains unread for this line:

```text
Status: EXPERIMENT_PENDING (Test0)
Result: not available
No Test conclusion is drawn.
```

## Gradient-mechanism evidence

The public update path now implements the same algorithmic contract as the
audited trainer:

```text
for each active objective:
    differentiate objective with respect to compressor parameters
    DP-all-reduce every gradient tensor and divide by world size
    compute the norm of the DP-averaged gradient vector
update bounded inverse-EMA weights
cap weighted VSD norm at weighted hard-CE norm
explicitly sum weighted gradient tensors
globally clip the joint vector to 1.0
write the result to parameter.grad and step AdamW
```

One audited update reported:

| Objective | Base loss | DP-reduced gradient norm | Final weighted norm |
| --- | ---: | ---: | ---: |
| Hard CE | 2.867829 | 2.675384 | 3.253826 |
| Response JSD | 0.00032255 | 0.004124 | 0.005095 |
| VSD | 0.00378283 | 49.892641 | 3.253826 |
| VLAD | 0.0000086736 | 0.072388 | 0.089437 |
| Feature protection | 0.0315309 | 0.026681 | 0.032965 |

The uncapped weighted VSD norm was `12.4732`; the CE anchor reduced its
optimizer weight from `0.25` to `0.0652166`. The joint norm was `4.610674` and
the global-clip scale was `0.216888`.

These numbers establish that the safeguard activated in one run. They do not
establish that the cap caused the quality improvement. Historical pairwise
gradient-cosine trajectories were not persisted. The public code computes them
for future experiments.

## Serving diagnostic and non-claim

An older paired Dev256 study of a different b128 compressor, not the selected
RetentionKD Full-u64 model, reported:

| Metric | Identity-like reference | Old compressed arm | Delta |
| --- | ---: | ---: | ---: |
| Mean prompt tokens | 3744.15 | 2706.48 | -27.714% |
| Mean TTFT | 1732.62 ms | 1139.70 ms | -34.220% |
| Mean completion tokens | 2255.21 | 3314.59 | +46.975% |
| Mean end-to-end latency | 166.25 s | 239.13 s | +43.84% |
| F1 | 75.00% | 64.00% | -11.00 pp |

This is a useful negative diagnostic: fewer prompt tokens and lower TTFT can be
offset by longer generations and worse quality. It is not evidence of a Full-u64
deployment gain. Full-u64 prefill latency, TTFT, peak memory, KV bytes,
throughput, QPS, and sustainable capacity remain `EXPERIMENT_PENDING` under a
fixed serving contract.

## Must-have before submission

1. Complete the same-order Full-u64 Dev2,141 evaluation. Report fixed-denominator
   confusion counts, validity, paired flips, McNemar, and bootstrap intervals.
2. Freeze the recipe, checkpoint, threshold, prompt, parser, and primary endpoint;
   then run Identity/Control/Full exactly once on the untouched Test.
3. Audit every model recipe, seed, checkpoint, and prompt variant that saw any
   Dev result. Separate semantic trials from infrastructure-only retries.
4. Run a causal matrix under the same initialization/data/update contract:
   CE only; CE+feature; JSD-only addition; VSD-only addition; VLAD-only addition;
   Full fixed weights; Full+EMA; Full+EMA+CE cap.
5. Use at least three training seeds for the key arms and report mean,
   dispersion, worst seed, paired uncertainty, and selection procedure.
6. Predeclare the primary quality/utility endpoint and the relative costs of FP,
   FN, and invalid outputs, or limit claims to the full metric trade-off.
7. Run a fixed-contract serving A/B that separately reports vision encode,
   compressor overhead, LM prefill, TTFT, decode, peak memory, KV bytes,
   throughput, sustainable capacity, and P50/P95 variability.
8. Complete the annotation and near-duplicate audit described in the data card.

## Strongly recommended

- Log per-objective norms and pairwise gradient cosines across updates and seeds;
  compare no-cap, norm-cap, and a direction-aware method such as PCGrad.
- Add matched-token-budget parameter-free pooling, input downsampling, and a
  simple trainable projector/compressor baseline.
- Compare main-only versus main+DeepStack, shared versus branch-specific
  normalization/adapters, warm-up versus no warm-up, multiple token budgets,
  dense versus group-center/stride M-RoPE coordinates, and replicated versus
  valid-masked odd edges.
- Add a second domain or public benchmark before making broad generalization
  claims. This is valuable evidence, not a universal prerequisite for every
  private-data industrial track; venue policy must be checked individually.

## Safe current verdict boundary

The current package supports a technically real compressor-only RetentionKD
case study with promising matched private-Dev evidence and a transparent list
of risks. It does not yet support statistical superiority over identity,
causally identify each retention component, demonstrate cross-domain
generality, or establish deployment speed/capacity gains.
