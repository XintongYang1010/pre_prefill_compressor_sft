# Experiment and Disclosure Protocol

## Private data is allowed; private data release is not required

The intended industrial setting uses an internally labeled domain dataset for training, validation, and model selection. A public dataset is not required for reporting the industrial result. This repository publishes only:

- the method and clean-room implementation;
- anonymous split sizes and immutable comparison rules;
- approved aggregate metrics and uncertainty estimates;
- the limitations needed to interpret those metrics.

It does **not** publish rows, images, source identifiers, label taxonomy, prompts, model responses, reasoning traces, manifests, checkpoints, or internal infrastructure metadata.

The synthetic tests in this repository verify software and mathematical contracts. They are not presented as the paper's business-quality evaluation. A user-owned adapter can supply authorized private tensors to an internal integration of these public interfaces.

## Comparison contract

A valid causal comparison must keep the following fixed:

- exact sample IDs and order;
- input preprocessing, image resolution, tokenizer, prompt, and generation settings;
- base model, compressor architecture, token budget, optimizer, seed, and update count;
- invalid-row policy and every reported denominator;
- checkpoint provenance and frozen/trainable parameter sets.

Full versus Control changes only the activation of response JSD, VSD, and VLAD. Full versus identity additionally changes the visual-token budget and therefore requires the same evaluation members; headline numbers from different splits cannot be subtracted.

## Metrics

The primary private-domain metrics are precision, recall, F1, fixed-denominator binary accuracy, false positives, false negatives, and structured-output validity. Exact McNemar and paired bootstrap are reported from row-aligned predictions. Small subtype slices are secondary diagnostics and cannot override the prespecified binary endpoint.

## Evidence levels

1. **Mechanism:** compressor active, token map valid, only intended parameters update.
2. **Training integrity:** all objectives activate, gradients remain finite, checkpoint/resume closes.
3. **Private-domain quality:** fixed-member paired metrics and uncertainty.
4. **Token reduction:** actual before/after visual and prompt-token counts.
5. **Deployment performance:** fixed-contract TTFT, prefill latency, throughput, memory, and capacity.
6. **Promotion:** held-out test and production gates.

Passing one level does not imply the next. The current repository contains evidence through level 4; level 5 and production promotion are pending.

## Publication rule for a running evaluation

A running job may be reported only as a timestamped completion count. Final quality metrics are added only after all expected rows, fixed-denominator scoring, paired artifacts, and validity checks close. Platform `RUNNING`, `DONE`, or exit code alone is not a model result.
