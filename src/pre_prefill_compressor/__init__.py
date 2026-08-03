"""Clean-room visual pre-prefill compressor reference package."""

from .checkpoint import (
    CheckpointState,
    load_training_checkpoint,
    save_training_checkpoint,
    stable_config_digest,
)
from .compressor import (
    FeatureCompressionOutput,
    GridTokenCompressor,
    ImageTokenPlan,
    VisionCompressorConfig,
    build_image_token_plan,
    compress_feature_branches,
    masked_reconstruction_loss,
)
from .evaluation import (
    BinaryMetrics,
    PairedBinaryEvaluation,
    binary_classification_metrics,
    exact_mcnemar_p_value,
    paired_binary_evaluation,
)
from .gradient_budget import (
    EMABoundedGradientController,
    GradientBudgetConfig,
    GradientCapAudit,
    apply_ce_anchor_cap,
)
from .qwen3vl_adapter import CompactedPrompt, compact_image_placeholders
from .retention_kd import (
    EffectiveNumberWeights,
    FeatureDistillationLoss,
    RetentionKDConfig,
    effective_number_weights,
    generalized_jsd_loss,
    group_mean,
    normalized_feature_distillation_loss,
    vision_language_affinity_distillation_loss,
    vision_semantic_distillation_loss,
    weighted_classification_loss,
)

__all__ = [
    "BinaryMetrics",
    "CheckpointState",
    "CompactedPrompt",
    "EMABoundedGradientController",
    "EffectiveNumberWeights",
    "FeatureCompressionOutput",
    "FeatureDistillationLoss",
    "GradientBudgetConfig",
    "GradientCapAudit",
    "GridTokenCompressor",
    "ImageTokenPlan",
    "PairedBinaryEvaluation",
    "RetentionKDConfig",
    "VisionCompressorConfig",
    "apply_ce_anchor_cap",
    "binary_classification_metrics",
    "build_image_token_plan",
    "compact_image_placeholders",
    "compress_feature_branches",
    "effective_number_weights",
    "exact_mcnemar_p_value",
    "generalized_jsd_loss",
    "group_mean",
    "load_training_checkpoint",
    "masked_reconstruction_loss",
    "normalized_feature_distillation_loss",
    "paired_binary_evaluation",
    "save_training_checkpoint",
    "stable_config_digest",
    "vision_language_affinity_distillation_loss",
    "vision_semantic_distillation_loss",
    "weighted_classification_loss",
]
