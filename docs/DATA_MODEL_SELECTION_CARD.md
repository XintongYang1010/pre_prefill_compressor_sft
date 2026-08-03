# Private Data and Model-Selection Card

## Disclosure boundary

The task data are private and cannot be distributed. This card reports approved
anonymous counts, prevalence, split isolation, model-selection rules, and known
audit gaps. It does not disclose rows, images, source or user identifiers,
business-label names, prompts, outputs, manifests, checkpoints, paths, or
infrastructure metadata.

## Dataset accounting

| Stage | Rows | Positive rows | Positive prevalence |
| --- | ---: | ---: | ---: |
| Processed parent pool | 33,197 | not disclosed | not disclosed |
| Allowlisted parent pool | 30,743 | not disclosed | not disclosed |
| Train after leakage exclusions | 19,310 | 948 | 4.909% |
| Dev | 2,141 | 101 | 4.717% |
| Frozen Test | 6,149 | 842 | 13.693% |

Train plus Dev contains `21,451` rows after leakage exclusions. The frozen Test
prevalence is materially different from Train/Dev, so metrics such as accuracy
must not be compared across these splits as if their base rates were identical.

Exact-key checks reported:

- Train/Dev/Test sample overlap: `0`;
- Train/Dev/Test user overlap: `0`;
- parent-pool/Test exact image-URI overlap: `0`;
- parent-pool/Test exact image-path overlap: `0`.

These checks do not establish perceptual-image, near-text, or semantic
deduplication. A near-duplicate audit is pending.

## Stage 0 feature-only warm-up

Stage 0 is feature-only visual-compression warm-up, not task SFT. It used
Train19,310 and Dev2,141, read zero Test rows, ran 3 epochs / 57,930 updates, and
updated 6,579,456 b128 compressor parameters while the backbone remained
frozen.

Approved aggregate diagnostics:

- epoch training loss: `2.287996 -> 2.145734 -> 2.133282`;
- Dev feature loss: `2.129265`;
- visual tokens: `82,421,121 -> 21,168,498`;
- visual-token keep ratio: `25.6833%`;
- one anonymous first-request branch length: `2017 -> 522` for main and each
  of three DeepStack branches.

Stage 0 did not use task labels or an LLM response loss. The later RetentionKD
adaptation used 1,000 task-labeled examples. Consequently, “limited-label task
adaptation” is supported; “the entire system used only 1,000 examples” is not.

## RetentionKD model selection

The current recipe uses two functional prompt contracts: one auxiliary
hard-label contract and one production-behavior distillation/evaluation
contract. Their raw text is restricted.

The matched sweep trained two arms, Control and Full, and retained four
milestones per arm: u8, u32, u64, and u125. Thus eight arm-checkpoint
combinations were inspected in that sweep. u64 was selected on the first fixed
Dev256; an independent, zero-overlap Dev256 was then used for confirmation.

This is not the total historical search count. Earlier feature recipes,
capacity variants, task-SFT attempts, infrastructure retries, and possible
prompt variants have not yet been consolidated into a complete model-selection
ledger. The paper must report the audited total number of development
configurations before submission and must distinguish semantic trials from
retries that did not alter the model recipe.

## Evaluation and invalid-row policy

All expected rows remain in fixed denominators. A malformed or missing
structured output does not silently disappear. Binary metrics, FP/FN, unknown
or invalid counts, and valid-output rates are reported together. Paired tests
require the same members and order.

The full Dev2,141 identity reference contains `75/1933/69/26/38`
TP/TN/FP/FN/unknown, giving Precision/Recall/F1/Accuracy of
`52.0833/74.2574/61.2245/93.7879%`. The pre-retention b512 reference contains
`63/1889/67/38/84`, giving `48.4615/62.3762/54.5454/91.1723%`.

The same-order Full-u64 Dev2,141 result is still running and has no terminal
metrics. Results from a different Dev256 must not be inserted as its third row.

## Annotation and business-utility gaps

The public evidence does not yet establish:

- annotator count, guideline version, independently double-labeled fraction,
  inter-annotator agreement, adjudication, or estimated label noise;
- a predeclared business cost ratio for false positives, false negatives,
  positive-invalid, and negative-invalid outputs;
- a complete historical model/prompt-selection ledger;
- perceptual or semantic near-duplicate isolation.

These fields remain `AUDIT_PENDING`; no value is inferred. Without an approved
cost model, an F1 increase cannot by itself be translated into higher overall
business utility.
