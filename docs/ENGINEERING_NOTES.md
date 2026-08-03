# Training Framework and Engineering Notes

## Why a standalone framework was built

The experiment needed controls that a generic “load model and run SFT” loop did not provide by default:

- only the compressor may update, while gradients still traverse the frozen student LLM;
- behavior and feature teachers come from different reference paths;
- teacher- and student-prefix rollouts must be mixed under a pinned prompt contract;
- five objectives require separate, distributed gradient accounting;
- matched Full/Control arms must differ only in objective activation;
- private-data membership, invalid rows, and denominators must remain fixed across evaluation;
- a long private evaluation must resume by durable row without changing the sample contract.

The resulting code is best described as an experiment/training framework around the proposed recipe. Building a framework is an engineering contribution; it becomes a methodological paper contribution only when experiments show that its specific controls are necessary.

## Current training loop

For each effective batch:

1. Load authorized private samples through a dataset adapter.
2. Produce frozen native main and DeepStack features.
3. Run the trainable student compressor and the frozen behavior teacher.
4. Build an auxiliary hard-label path and a production-behavior distillation path.
5. Generate the scheduled teacher/student rollout prefixes.
6. Compute CE, feature, response-JSD, VSD, and VLAD objectives.
7. Differentiate every active objective with respect to compressor parameters.
8. Average objective gradients across data-parallel workers.
9. Apply bounded EMA balancing and the VSD-to-CE contribution cap.
10. Combine, globally clip, step AdamW, and audit frozen parameters and relative step size.

Control follows the same code path but disables response JSD, VSD, and VLAD before rollout and gradient construction. This avoids paying for or accidentally activating a nominally disabled objective.

## Latest optimization decisions

- **Learning rate:** reduced from `1e-5` to `5e-6` after the earlier task-only adaptation collapsed.
- **CE anchor:** weighted VSD gradient norm cannot exceed the weighted hard-CE gradient norm.
- **Numerical activation gate:** objective loss/gradient values at or below `1e-12` are treated as inactive rather than silently balanced.
- **Early checkpointing:** u8/u32/u64/u125 are all retained; u64 was selected empirically and u125 is not treated as automatically better.
- **Full-state resume:** optimizer, scheduler, gradient controller, per-worker RNG, objective trajectory, and configuration digest accompany compressor weights.
- **Strict objective validation:** terminal validation compares the exact objective key set, independent of serialized dictionary order.
- **Arm-specific runtime fields:** machine-local paths are validated inside each arm but are not required to be byte-identical across arms; semantic model/data contracts still must match.
- **Durable evaluation resume:** a completed row is skipped only after its receipt is valid. Partial prefixes, transport-only success, or malformed rows do not become quality evidence.

## What the public code intentionally leaves out

The clean-room package implements the mathematical and state-management core. It does not include private dataset loaders, production prompts, internal distributed launchers, model-serving wrappers, storage paths, signed receipts, cluster configuration, or platform finalizers. Those integrations are environment-specific and are not needed to review the method.

## Reproducibility levels

- `pytest` verifies tensor maps, losses, gradients, freeze behavior, statistics, and checkpoint round trips with generated data.
- `examples/train_synthetic.py` verifies that an optimizer update can flow through a frozen downstream network into the compressor.
- Reproducing the reported business metrics requires an authorized internal adapter and dataset. The data are intentionally not public.
- Reproducing deployment claims requires a future fixed-contract serving evaluation; no deployment gain is claimed here.
