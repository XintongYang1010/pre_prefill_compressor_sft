# Web Pro Paper Review Packet

Use this file to commission an independent, current literature and feasibility
review. The raw training and evaluation data are private; only the method,
clean-room implementation, aggregate evidence, and disclosure protocol are in
scope for public review.

## 中文使用说明

将下面 `Copy-paste prompt` 整段交给网页版 Pro，并同时提供本仓库链接。要求它必须联网
搜索到审稿当天，优先阅读论文 PDF、出版社页面和官方代码，而不是只看摘要或二手解读。
尤其要求它正面处理三项高重叠先例：EM-KD 的 VSD/VLAD、GKD 的 on-policy response
distillation、FCoT-VL 的冻结主干视觉压缩蒸馏与 post-train。

第一套严格同成员 Dev256 已证明 Full-u64 相对 identity 的 F1 高 `2.90pp`、Precision
高 `8.85pp` 且 Accuracy 持平；但 Recall 低 `4.35pp`、FN 多 1、有效结构化输出少 4 条。
因此只能写“F1/Precision 优于 identity，同时存在 Recall/FN/输出有效性 trade-off”，不能
写成全面优于。另一套 Full-u64 Dev256 的 `65.45%` F1 与历史 identity Dev2,141 的
`61.22%` F1 来自不同 split，严禁相减或作为 superiority 证据。

## Files the reviewer should read

Read these repository files before reaching a verdict:

1. `README.md` — scope and public evidence boundary.
2. `docs/METHOD.md` — architecture, teachers, objectives, and update rule.
3. `docs/EXPERIMENT_STATUS.md` — current aggregate results and pending gates.
4. `docs/NOVELTY_AND_NONCLAIMS.md` — known closest work and forbidden claims.
5. `src/` and `tests/` — clean-room reference implementation and synthetic
   contract tests.

If a listed file is missing or incomplete, treat that as a review finding rather
than filling in the gap by assumption.

## Copy-paste prompt

```text
Act as a skeptical senior reviewer for a multimodal-model efficiency paper. Your
job is not to help market the method. Determine whether the proposed paper has a
defensible contribution, whether the method is technically sound, and what
evidence is still required.

DATE AND SEARCH REQUIREMENT

Perform a live literature and official-code search through today's date. The
repository's literature snapshot is dated 2026-08-03; search for papers and code
released after that date as well. Use primary sources only for factual claims:
publisher/conference pages, arXiv/OpenReview papers, official technical reports,
and repositories linked by the authors. Secondary summaries may help discovery
but may not support the verdict. Distinguish peer-reviewed papers from preprints
and inspect the method/experiments in the full paper or official code rather than
relying on abstracts alone.

METHOD UNDER REVIEW

The system retrofits a lightweight visual-token compressor into a pretrained
Qwen3-VL-style model. The compressor is placed after the native visual merger and
before frozen LLM prefill. The vision tower, native main merger, DeepStack
mergers, and LLM parameters remain frozen; only the new compressor is optimized.
The same image-local spatial mapping is applied to the main visual sequence and
DeepStack side inputs while preserving image boundaries and the model's
positional/token contracts.

Training combines:

1. hard-label task CE for a private-domain structured output;
2. response-distribution KD on a deterministic mixture of student-generated and
   teacher-generated prefixes;
3. Vision Semantic Distillation (VSD): reverse KL between teacher and student
   vocabulary distributions projected from selected final-layer visual states;
4. Vision-Language Affinity Distillation (VLAD): Smooth L1 between normalized
   visual-language affinity matrices;
5. local, main-stream, and multi-level visual feature-protection losses.

The optimizer obtains gradients for the objectives separately, averages them
under data parallelism, uses bounded moving-gradient-norm reweighting, caps the
weighted VSD gradient against hard-label CE, sums the gradients, and applies a
global clip. The stated motivation is that task-only compressor SFT collapsed
recall and structured-output validity, while unconstrained VSD gradients could
dominate the only trainable module.

KNOWN PRIOR-ART OVERLAP THAT MUST NOT BE IGNORED

- EM-KD (AAAI 2026) already introduces the same-named VSD and VLAD objectives and
  combines supervised, response, vision-semantic, and affinity distillation:
  https://ojs.aaai.org/index.php/AAAI/article/view/39254
  https://arxiv.org/abs/2511.21106
- Generalized Knowledge Distillation / On-Policy Distillation already trains on
  student-generated prefixes and supports mixed policies and alternative
  divergences:
  https://arxiv.org/abs/2306.13649
- FCoT-VL already freezes the ViT and LLM, learns a lightweight visual
  compression module/projector using output KL plus hard-label CE, and performs
  post-training with limited data:
  https://arxiv.org/abs/2502.18512
- ETC performs task-aware visual information distillation, includes Qwen3-VL
  experiments, and targets aggressive token/KV-cache compression:
  https://arxiv.org/abs/2606.00543
- MetaCompress studies a learnable, data-efficient compression mapping:
  https://arxiv.org/abs/2603.21701
  https://github.com/MArSha1147/MetaCompress
- TokenPacker, DeCo, VisionZip, FastV, MQT, LLaVA-KD, Vision Remember, GradNorm,
  MetaBalance, EnCodec's loss balancer, and BLIP-2 cover additional parts of the
  architecture, distillation, and gradient-control design. Follow the primary
  links in docs/NOVELTY_AND_NONCLAIMS.md and search beyond them.

PRIVATE-DATA CONSTRAINT

The vertical-domain examples and labels cannot be released. The authors can
publish the method, clean-room reference code, synthetic contract tests,
anonymous aggregate dataset statistics, frozen split protocol, aggregate
metrics, uncertainty, and conclusions. Evaluate the work both as:

A. a general methods paper; and
B. an industrial/application paper based on private data.

For every proposed venue, consult the venue's official current call, track
description, reproducibility policy, and artifact/data rules. Do not assume that
private data is either automatically acceptable or automatically disqualifying.

CURRENT RESULT-CLAIM AUDIT

A valid same-member Dev256 comparison contains:

identity: Precision 60.71%, Recall 73.91%, F1 66.67%, Accuracy 91.02%,
          FN 6, valid structured outputs 250/256
Full-u64: Precision 69.57%, Recall 69.57%, F1 69.57%, Accuracy 91.02%,
          FN 7, valid structured outputs 246/256

Thus Full-u64 has +8.85pp precision and +2.90pp F1 with equal accuracy, but
-4.35pp recall, one more false negative, and -1.56pp structured-output validity.
The safe claim is an F1/precision advantage with explicit quality trade-offs,
not comprehensive domination.

A separate Full-u64 Dev256 has F1 65.45%, while the historical identity
Dev2,141 has F1 61.22%. Those rows use different members, class prevalence, and
denominators; they must not be compared. A same-order Full-u64 Dev2,141 run may
still be in progress. Check the latest EXPERIMENT_STATUS.md and use a newer
terminal paired result only if its complete contract is available.

REQUIRED OUTPUT

Return a report with the following sections.

1. Executive verdict
   - Give separate verdicts for general-method novelty, systems novelty, and
     industrial/application value: strong / borderline / weak.
   - State whether a paper is feasible now, feasible after additional evidence,
     or not defensible in its current framing.

2. Closest-work table
   - List at least the 15 closest papers through the review date.
   - Include publication date, peer-review status, official paper/code links,
     exact overlapping claims/components, and the remaining difference.
   - Rank them by threat to novelty, not by citation count.

3. Claim-by-claim provenance
   - For post-merger compression, frozen-backbone training, spatial 2x2-to-1
     aggregation, DeepStack preservation, hard CE + KD, mixed on/off-policy
     response KD, JSD, VSD, VLAD, feature protection, gradient-norm balancing,
     and the CE-anchored VSD cap, classify each as: established prior art,
     adaptation, potentially new combination, or insufficiently specified.
   - Quote no more than a short phrase from any paper; paraphrase the rest.

4. Technical-correctness review
   - Audit teacher/student definitions and which tensors are actually aligned.
   - Check whether VSD vocabulary projection is semantically justified.
   - Check whether same-layout teacher/student makes token matching unnecessary.
   - Check the mixed-rollout response objective for exposure bias and leakage.
   - Check whether separate per-loss gradients, DP averaging, moving-norm
     weighting, the CE anchor, and global clipping implement the claimed update.
   - Identify memory/compute costs that could erase deployment gains.

5. Experimental sufficiency and causal identification
   - Identify which current comparisons are valid and which mix contracts.
   - Require a minimal ablation matrix that isolates response GKD, VSD, VLAD,
     feature protection, dynamic balancing, and the CE-anchored cap.
   - Require matched token-budget baselines and a terminal identity comparison.
   - Assess sample size, prevalence, invalid outputs, confidence intervals,
     paired tests, multiple seeds, untouched test data, and distribution shift.
   - Decide what can be claimed with private data only and which generalization
     claims require a public benchmark or a second independent domain.

6. Efficiency and deployment audit
   - Separate visual-token reduction from measured prefill latency, TTFT, decode
     latency, peak memory, KV-cache bytes, throughput, and sustainable capacity.
   - State the minimum serving experiment needed to support an efficiency claim.

7. Venue fit
   - Recommend 3-5 realistic current venues/tracks, including industrial or
     application tracks where appropriate.
   - For each, link the official call/policy and state exactly what is missing.
   - Do not recommend a venue solely from reputation or topic keywords.

8. Safe paper framing
   - Propose one title, a one-sentence claim, and 3-4 contributions that survive
     the prior-art review.
   - Rewrite or delete every overclaim, especially “first,” “new framework,”
     “better than identity,” and “token reduction equals speedup.”
   - Clearly credit EM-KD and GKD for inherited objectives.

9. Falsification plan
   - Give the five fastest experiments that could disprove the proposed story.
   - State in advance what result would force a no-go decision for the paper.

10. Final prioritized checklist
    - Divide work into must-have before submission, valuable but optional, and
      claims that should be abandoned.

EVIDENCE RULES

- Put a direct primary-source link next to every literature-dependent claim.
- Label inferences as inferences.
- Do not infer terminal success from a running job, a completed launcher, token
  counts, or a checkpoint existing.
- Do not compare point estimates across different datasets, prompts, decoders,
  denominators, or checkpoints.
- Do not call a combination novel merely because the authors implemented a
  custom trainer.
- If the current evidence cannot support a conclusion, say “not established”
  and list the exact missing artifact or experiment.
```

## Reviewer acceptance checklist

The Pro review is incomplete unless it:

- treats EM-KD as direct prior art for VSD/VLAD;
- treats GKD as direct prior art for mixed-rollout response KD;
- includes ETC, MetaCompress, and FCoT-VL among the highest-risk comparisons;
- searches beyond the seed bibliography through the actual review date;
- separates methods, systems, and industrial/application novelty;
- audits the candidate identity comparison rather than repeating the desired
  conclusion;
- evaluates private-data-only publication against official venue policies;
- identifies a falsifiable minimum experiment set; and
- distinguishes visual-token reduction from measured deployment speed/capacity.
