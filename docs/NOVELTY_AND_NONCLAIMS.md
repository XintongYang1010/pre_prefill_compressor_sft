# Novelty Review and Claim Boundaries

Literature snapshot: **2026-08-03**. This is a research-positioning memo, not a
claim that the method is new and not a legal prior-art opinion. Only primary
papers, publisher pages, and official project repositories are linked below.

## Chinese summary / 中文结论

目前不能把整套方法写成“全新视觉 token 压缩方法”。最直接的原因是：

- **VSD 和 VLAD 与 EM-KD 的定义高度重合**：视觉位置经 LM head 得到词表分布后做
  reverse KL，以及视觉—语言 affinity matrix 的 Smooth L1，均已由
  [EM-KD](https://ojs.aaai.org/index.php/AAAI/article/view/39254) 明确提出。
- Student rollout 上的 response distillation、on/off-policy 混合和可选 divergence
  已属于 [Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649)
  的核心范式。
- 冻结大部分 VLM、只训练轻量视觉压缩/投影模块，再用 hard-label CE 与输出蒸馏
  联合训练，和 [FCoT-VL](https://arxiv.org/abs/2502.18512)、
  [BLIP-2](https://arxiv.org/abs/2301.12597) 等已有做法存在明显重叠。
- 可学习的视觉 token 压缩、局部空间聚合和少数据训练，也分别已有
  [TokenPacker](https://arxiv.org/abs/2407.02392)、
  [MetaCompress](https://arxiv.org/abs/2603.21701)、
  [DeCo](https://arxiv.org/abs/2405.20985) 等先例。

仍可能形成论文的部分，不是上述单个 loss，而是一个需要实验支撑的**工业系统组合**：
在保留 Qwen3-VL 原生 merger、M-RoPE 和 DeepStack 合同的前提下，只重训
merger 后 compressor；用双 Teacher 来源保护输出行为与多层视觉信息；并用
hard-label CE 作为不对称梯度预算锚点，解决垂类、小样本、类别不平衡和结构化输出
共同导致的训练崩塌。这个方向是否足够构成方法贡献，必须由严格消融、同合同基线和
真实 prefill/serving 数据决定。

内部数据可以支持工业应用论文，且不一定需要公开样本；但这只解决数据保密问题，
不会自动解决新意、可复现性和泛化证据问题。公开材料至少应给出匿名化数据统计、
固定切分协议、指标分母、置信区间、重复实验和可运行的 clean-room 实现。

## 1. Method under review

The public, architecture-level description being reviewed is:

1. Keep the pretrained vision tower, native visual merger, DeepStack mergers,
   and a large language-model backbone frozen.
2. Insert a lightweight spatial compressor after the native merger and before
   LLM prefill. Apply a consistent per-image spatial mapping to the main visual
   sequence and DeepStack side inputs; do not mix tokens across images.
3. Train only the compressor with a combination of:
   - hard-label task CE for the domain output and serialization contract;
   - response-distribution distillation on a mixture of student- and
     teacher-generated prefixes;
   - vision semantic distillation (VSD);
   - vision-language affinity distillation (VLAD); and
   - local, main-stream, and multi-level feature protection.
4. Compute per-objective gradients, track their moving norms, apply bounded
   reweighting, cap the weighted VSD gradient against the task-CE gradient, and
   then clip the joint update.

This is **compressor-only post-training**, not full vision-encoder SFT and not
an end-to-end retraining of the VLM.

## 2. Closest work: overlap and remaining distinction

| Area | Closest primary work | Material overlap | What is still different here | Safe interpretation |
| --- | --- | --- | --- | --- |
| Frozen-backbone visual compression followed by post-training | [FCoT-VL](https://arxiv.org/abs/2502.18512) freezes the ViT and LLM, learns a compression module/projector, uses output KL plus hard-label CE, and then post-trains the compressed VLM | Lightweight compressor training, frozen strong backbone, small-data re-alignment, KD + CE, and a later task-oriented stage | Qwen3-VL-specific post-native-merger placement, DeepStack-preserving mapping, private-domain structured output, and the proposed gradient controller | Strong prior overlap. Do not claim the broad framework of “distill then SFT a visual compressor” as new. |
| VSD and VLAD | [EM-KD](https://arxiv.org/abs/2511.21106), accepted at AAAI 2026 | The names and objectives are essentially direct matches: reverse KL over vocabulary-space vision logits for VSD; Smooth L1 between vision-language affinity matrices for VLAD. EM-KD also combines supervised, response, VSD, and VLAD losses | The present teacher/student use an already aligned compressed layout, so Hungarian token matching may be unnecessary; only the compressor is optimized; additional feature protection and gradient budgeting are used | **VSD and VLAD themselves are not novel claims.** Describe them as adopted/adapted from EM-KD and cite it prominently. |
| Response-level on-policy KD | [On-Policy Distillation / GKD](https://arxiv.org/abs/2306.13649); [MiniLLM](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8ac015d409635f196f9e3e9dcfb9a94e-Abstract-Conference.html) | Student-generated rollouts, mixed on/off-policy prefixes, alternative divergences, and reverse-KL-style generative KD are established | Applying mixed-rollout distribution retention while only a visual compressor can move, with a frozen shared LLM and a second non-thinking hard-label path | An application/adaptation of GKD, not a new on-policy KD algorithm. |
| Multimodal and relation distillation | [LLaVA-KD](https://arxiv.org/abs/2410.16236) | Multi-stage multimodal distillation, SFT, and relation transfer between visual representations | Same-sized frozen LLM and a tiny compressor student rather than a small end-to-end VLM; explicit DeepStack and task-CE gradient budget | Useful comparison for the “multi-level retention” claim; current work needs ablations to show why its added targets are needed. |
| Task-aware visual information retention | [ETC](https://arxiv.org/abs/2606.00543) | Task-aware visual token compression, auxiliary visual-information distillation, Qwen3-VL experiments, and direct efficiency/KV-cache motivation | ETC uses instruction-aware cross-attention, a variational information objective, and trains the projector plus LLM/LoRA; the present method uses a fixed spatial compressor, frozen LLM, response/feature teachers, and moderate rather than one-token compression | ETC is a high-risk closest work for the paper narrative, even though the loss and trainable-parameter boundary differ. |
| Learning a compression mapping with limited compute | [MetaCompress](https://arxiv.org/abs/2603.21701) ([official repository](https://github.com/MArSha1147/MetaCompress)) | Learnable prompt-agnostic compression and a data-efficient training paradigm | Current work is a fixed spatial mapping plus learned content transformation, targets single-turn domain retention, and preserves Qwen3-VL DeepStack inputs | “Learnable, data-efficient token compression” is already claimed; focus on frozen-backbone retention and vertical failure recovery. |
| Efficient projector / local spatial compression | [TokenPacker](https://arxiv.org/abs/2407.02392) ([official code](https://github.com/CircleRadon/TokenPacker)); [DeCo](https://arxiv.org/abs/2405.20985) | Local region aggregation, compressed visual projector output, and 2-D pooling/patch-level compression before the LLM are established | Compressor is inserted after an existing native merger and must preserve the original Qwen3-VL positional and DeepStack contracts; backbone replacement is avoided | A `2x2 -> 1` spatial compressor or post-encoder placement alone is not a sufficient novelty claim. |
| Training-free token selection/merging | [VisionZip](https://arxiv.org/abs/2412.04467) ([official code](https://github.com/dvlab-research/VisionZip)); [FastV](https://arxiv.org/abs/2403.06764) ([official code](https://github.com/pkunlp-icler/fastv)); [PixelPrune](https://arxiv.org/abs/2604.00886) | All reduce visual tokens to lower downstream cost, but at different pipeline locations and with different information-selection rules | Current method is training-based, does not prune the ViT input, and aims at preserving a private domain under aggressive post-merger compression | These remain required matched-budget baselines; architecture differences do not make them irrelevant. |
| Variable and very small token budgets | [Matryoshka Query Transformer](https://proceedings.neurips.cc/paper_files/paper/2024/hash/59c147c7d4fdb732daea3064eab949bf-Abstract-Conference.html); [Matryoshka Multimodal Models](https://arxiv.org/abs/2405.17430) | Learned compact visual representations and controllable visual-token counts | Present recipe uses a fixed budget and optimizes a pre-existing compressor for one vertical task | Do not claim flexible token budgeting; include multiple compression ratios if the paper claims a general compression recipe. |
| Multi-level visual feature preservation | [Vision Remember](https://arxiv.org/abs/2506.03928); the [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631) | Multi-level visual features are known to mitigate information loss; Qwen3-VL already introduces DeepStack for multi-level fusion | Compressing main and all DeepStack streams with one image-local mapping while preserving the frozen Qwen3-VL interface may be a useful systems contribution | Claim architecture compatibility and measured retention, not invention of multi-level visual fusion. |
| Dynamic multi-loss balancing | [GradNorm](https://proceedings.mlr.press/v80/chen18a.html); [MetaBalance](https://arxiv.org/abs/2203.06801) ([official code](https://github.com/facebookresearch/MetaBalance)); [EnCodec](https://arxiv.org/abs/2210.13438) ([official code](https://github.com/facebookresearch/encodec)) | Per-loss gradient norms, running statistics, and adaptive control of auxiliary gradient magnitudes are established | An asymmetric, task-CE-anchored cap designed to prevent VSD from dominating a compressor-only update may be the most specific algorithmic difference | Present as a safety-constrained adaptation unless a formal analysis and ablation establish an independent contribution. |
| Frozen pretrained components joined by a small trainable bridge | [BLIP-2](https://arxiv.org/abs/2301.12597) | Frozen image encoder and frozen LLM with a lightweight learned bridge is established | Retrofitting compression into an already aligned Qwen3-VL and preserving its output contract differs from bootstrapping a new multimodal bridge | “Freeze both towers and train a small bridge” is not new. |

## 3. What may still be paper-worthy

The strongest paper is likely an **industrial compression and retention study**,
not a claim that every component is novel.

### 3.1 Defensible contribution candidates

1. **Architecture-preserving retrofit.** A post-native-merger compressor that
   reduces LLM prefill tokens without retraining the vision tower, native merger,
   DeepStack mergers, or LLM, while preserving per-image boundaries, M-RoPE
   indexing, and all DeepStack injection shapes.
2. **Failure-driven retention recipe.** A documented case where task-only
   compressor SFT destroys recall and structured generation, followed by a
   controlled recipe that combines hard targets with behavior, visual-semantic,
   cross-modal, and feature-retention constraints.
3. **Task-anchored gradient safety.** An asymmetric constraint that treats hard
   task CE as the optimization anchor and prevents a large auxiliary VSD gradient
   from taking over the only trainable module.
4. **Low-data private-domain evidence.** Demonstrating that a frozen-backbone
   compressor can be adapted with limited, expensive labels while retaining or
   improving the predeclared business metric at a meaningful token budget.
5. **Evidence and deployment discipline.** Reporting not just token count, but
   prefill latency, TTFT, peak memory, throughput/capacity, structured-output
   validity, class-specific errors, and promotion gates under the same serving
   contract.

Each item is a **candidate** contribution. None is established merely because
the code path exists.

### 3.2 Claims that should not appear

- “The first visual-token compression method.”
- “The first trainable post-encoder/post-projector compressor.”
- “We introduce VSD/VLAD.”
- “We introduce on-policy or generalized KD.”
- “We introduce gradient-norm loss balancing.”
- “A 74% token reduction proves a 74% latency, memory, or capacity gain.”
- “The method is generally applicable to all vertical models” from one private
  dataset and one backbone.
- “Full-u64 is better than identity” unless both are measured on the exact same
  examples, prompts, preprocessing, checkpoints, decoding contract, denominator,
  and metric definition, and the paper names the metric on which it is better.
- “The custom training framework is a standalone research contribution” unless
  it exposes a capability not provided by existing trainers and that capability
  is isolated experimentally.

## 4. Current aggregate-result claim gate

### 4.1 Valid matched Dev256 comparison

One completed evaluation used the exact same 256 members, order, prompt,
request seed, parser, and invalid-row denominator for identity and Full-u64:

| Arm | Precision | Recall | F1 | Accuracy | FN | Valid outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Identity, no compression | 60.71% | **73.91%** | 66.67% | 91.02% | **6** | **250 / 256** |
| Full-u64 | **69.57%** | 69.57% | **69.57%** | 91.02% | 7 | 246 / 256 |
| Full minus identity | +8.85 pp | **-4.35 pp** | +2.90 pp | 0.00 pp | +1 | -4 rows |

The supported conclusion is narrow but positive:

> Full-u64 exceeded identity on F1 and precision on the first matched Dev256
> while preserving fixed-denominator accuracy, but it did not dominate identity
> on recall, false negatives, or structured-output validity.

This does not establish statistical significance or full-volume superiority.

### 4.2 Invalid cross-split comparison

The independent Full-u64 Dev256 result (Precision `58.06%`, Recall `75.00%`,
F1 `65.45%`, Accuracy `89.84%`) and the historical identity Dev2,141 result
(Precision `52.08%`, Recall `74.26%`, F1 `61.22%`, Accuracy `93.79%`) use
different members, prevalence, and denominators. Therefore `65.45% > 61.22%`
is **not** evidence that Full-u64 beats identity. A same-order Full Dev2,141
confirmation is still required.

Every published comparison must verify:

- identical sample IDs and order;
- identical positive prevalence and metric denominator;
- identical prompt, image preprocessing, tokenizer, generation, and parser;
- no invalid rows silently removed;
- paired error counts and confidence intervals.

If a newer full-cohort A/B/C evaluation exists, it should replace the historical
cross-split headlines rather than being blended with them. “Better overall” is
only defensible against a predeclared primary business metric or utility
function.

## 5. Minimum experimental program before submission

### 5.1 Causal ablations

Use the same training data, initialization, update count, and evaluation contract:

1. frozen identity baseline;
2. pre-retention compressed baseline;
3. hard CE only;
4. hard CE + feature protection;
5. add response GKD only;
6. add VSD only;
7. add VLAD only;
8. full objective with fixed weights;
9. full objective with gradient balancing;
10. full objective with balancing plus the CE-anchored VSD cap.

This is necessary to avoid assigning the entire Full-Control gain to any one of
GKD, VSD, VLAD, feature protection, or gradient control.

### 5.2 Architecture ablations

- compressor before versus after the native merger;
- main stream only versus main plus all DeepStack streams;
- spatially consistent mapping versus an otherwise matched generic pooler;
- at least three token budgets, including the deployed budget;
- fixed budget versus a simple adaptive-budget baseline if generality is claimed;
- compressed-teacher versus uncompressed-teacher supervision;
- learned compressor versus parameter-free 2-D pooling.

### 5.3 Matched baselines

At a matched visual-token budget, include feasible representatives from:

- parameter-free pooling / DeCo-style compression;
- a trainable projector such as TokenPacker;
- training-free selection/merging such as VisionZip or FastV;
- a task-aware learned compressor such as ETC or MetaCompress when compatible;
- task-only compressor SFT and the unmodified identity model.

If an implementation cannot be ported fairly to Qwen3-VL, document the exact
incompatibility instead of reporting a mismatched number.

### 5.4 Statistical and domain evidence

- predeclare one primary metric and all secondary metrics;
- report exact denominators, prevalence, invalid-output handling, and JSON/schema
  validity;
- report paired bootstrap intervals and exact McNemar tests on paired decisions;
- run multiple training seeds and report mean, dispersion, and worst seed;
- keep a terminal test split that is untouched by recipe selection;
- evaluate at least one temporal or source-shifted private holdout;
- ideally add a second vertical task or one public benchmark. If no public data
  can be used, narrow the paper to an industrial case study rather than claiming
  general-purpose superiority.

### 5.5 Efficiency evidence

Token reduction is a mechanism measurement, not a deployment result. Measure on
the actual serving stack:

- vision encoding time;
- prefill latency and TTFT at controlled concurrency;
- decode latency separately;
- peak GPU memory and KV-cache bytes;
- throughput and sustainable capacity;
- variance and tail latency;
- model-load and compressor overhead.

## 6. Private-data publication boundary

A private dataset can be a legitimate basis for an industrial paper. The public
artifact should still disclose enough aggregate information to make the evidence
auditable without exposing samples:

- task definition and prediction schema in generic terms;
- data collection window and inclusion/exclusion rules;
- label process, annotator agreement, and class prevalence;
- train/dev/test sizes and deduplication boundaries;
- frozen split/manifest hashes that reveal no record identifiers;
- model-selection policy and number of attempted recipes;
- complete aggregate metrics, confidence intervals, and failure categories;
- clean-room code, synthetic contract tests, and an exact pseudocode recipe.

The paper must state that raw data cannot be released and avoid claiming that an
unseen external party can reproduce the private-domain score. Venue eligibility
and artifact requirements must be checked against the official call for papers
for the chosen industry/application track.

## 7. Research-positioning recommendation

A cautious working title is:

> **Retention-Constrained Post-Merger Visual Token Compression for Frozen
> Vision-Language Models in a Low-Data Industrial Domain**

A defensible one-sentence claim, subject to the missing experiments, is:

> We study how to retrofit a lightweight visual-token compressor after the
> native merger of a frozen DeepStack VLM, and show that a task-anchored,
> multi-level retention recipe can recover the failure of task-only compressor
> SFT at a fixed prefill-token budget on a private industrial task.

This wording claims a studied setting and an empirical result. It does not claim
ownership of EM-KD losses, GKD, spatial pooling, frozen-backbone adaptation, or
gradient-norm balancing.

## 8. Primary-source registry

- Qwen Team, [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) and
  [official repository](https://github.com/QwenLM/Qwen3-VL).
- Feng et al., [EM-KD](https://ojs.aaai.org/index.php/AAAI/article/view/39254)
  (AAAI 2026) and [arXiv version](https://arxiv.org/abs/2511.21106).
- Li et al., [FCoT-VL](https://arxiv.org/abs/2502.18512).
- Agarwal et al., [On-Policy Distillation / GKD](https://arxiv.org/abs/2306.13649)
  (ICLR 2024).
- Gao et al., [ETC](https://arxiv.org/abs/2606.00543).
- Wang et al., [MetaCompress](https://arxiv.org/abs/2603.21701) and
  [official repository](https://github.com/MArSha1147/MetaCompress).
- Cai et al., [LLaVA-KD](https://arxiv.org/abs/2410.16236) (ICCV 2025).
- Li et al., [TokenPacker](https://arxiv.org/abs/2407.02392) and
  [official repository](https://github.com/CircleRadon/TokenPacker).
- Yang et al., [VisionZip](https://arxiv.org/abs/2412.04467) and
  [official repository](https://github.com/dvlab-research/VisionZip).
- Yao et al., [DeCo](https://arxiv.org/abs/2405.20985).
- Feng et al., [Vision Remember](https://arxiv.org/abs/2506.03928).
- Hu et al., [Matryoshka Query Transformer](https://proceedings.neurips.cc/paper_files/paper/2024/hash/59c147c7d4fdb732daea3064eab949bf-Abstract-Conference.html).
- Chen et al., [GradNorm](https://proceedings.mlr.press/v80/chen18a.html).
- He et al., [MetaBalance](https://arxiv.org/abs/2203.06801) and
  [official repository](https://github.com/facebookresearch/MetaBalance).
- Défossez et al., [High Fidelity Neural Audio Compression / EnCodec](https://arxiv.org/abs/2210.13438)
  and [official repository](https://github.com/facebookresearch/encodec).
- Li et al., [BLIP-2](https://arxiv.org/abs/2301.12597) (ICML 2023).
