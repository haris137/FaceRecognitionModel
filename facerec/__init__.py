"""facerec: face detection -> alignment -> quality -> embedding -> matching, tuned for Pakistani faces."""
from .config import PipelineConfig, QualityConfig, ADAFACE_WEIGHTS, TAG_GROUPS

__all__ = ["PipelineConfig", "QualityConfig", "ADAFACE_WEIGHTS", "TAG_GROUPS"]
