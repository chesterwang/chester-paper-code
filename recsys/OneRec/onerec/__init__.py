"""
OneRec: An End-to-End Generative Framework for Recommender Systems

This package implements the OneRec model architecture as described in the paper:
"OneRec: A Unified Generative Framework for Recommender Systems"
"""

from .model import OneRec
from .tokenizer import OneRecTokenizer, QFormer, RQKmeansTokenizer
from .encoder import OneRecEncoder, UserStaticPathway, ShortTermPathway, PositiveFeedbackPathway, LifelongPathway
from .decoder import OneRecDecoder, CausalSelfAttention, CrossAttention, MixtureOfExperts
from .reward_system import OneRecRewardSystem, PreferenceScoreTower, EarlyClippedGRPO, FormatReward, IndustrialReward
from .training import OneRecTrainer, OneRecDataset
from .config import config

__version__ = "1.0.0"
__author__ = "OneRec Team"
__all__ = [
    "OneRec",
    "OneRecTokenizer", "QFormer", "RQKmeansTokenizer",
    "OneRecEncoder", "UserStaticPathway", "ShortTermPathway", "PositiveFeedbackPathway", "LifelongPathway",
    "OneRecDecoder", "CausalSelfAttention", "CrossAttention", "MixtureOfExperts",
    "OneRecRewardSystem", "PreferenceScoreTower", "EarlyClippedGRPO", "FormatReward", "IndustrialReward",
    "OneRecTrainer", "OneRecDataset",
    "config"
]