"""engine/ — task-agnostic training and evaluation loops."""
from .trainer import train_one_epoch, build_optimizer, build_scheduler
from .evaluator import evaluate_classifier, evaluate_regressor

__all__ = [
    "train_one_epoch",
    "build_optimizer",
    "build_scheduler",
    "evaluate_classifier",
    "evaluate_regressor",
]
