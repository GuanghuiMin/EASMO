"""M4 (stretch) — TinyBERT-style classifier that predicts the source agent
from an oracle memory text. Optional; runs offline on the JSONL output.

Implementation note: to keep deps light, we fall back to a TF-IDF +
LogisticRegression baseline. If ``transformers`` is available we can
upgrade to a small BERT later — but per the spec the goal is just
"can a tiny model distinguish them?" and TF-IDF + LR is usually enough
when oracle memories really differ structurally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np

from .utils import setup_logging

_logger = setup_logging("motivation.classifier")


@dataclass
class ClassifierResult:
    test_accuracy: float
    n_train: int
    n_test: int
    confusion_matrix: List[List[int]]
    label_names: List[str]
    pass_70: bool

    def to_dict(self) -> dict:
        return asdict(self)


def train_eval_agent_classifier(
    memories: List[str],
    agent_labels: List[str],
    *,
    seed: int = 42,
    test_size: float = 0.2,
) -> ClassifierResult:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    assert len(memories) == len(agent_labels)
    if len(set(agent_labels)) < 2:
        raise ValueError("Need ≥2 distinct agent labels to train a classifier.")
    if len(memories) < 10:
        _logger.warning(
            "Only %d memories — test-set evaluation will be noisy.", len(memories)
        )

    Xtrain_txt, Xtest_txt, ytrain, ytest = train_test_split(
        memories, agent_labels, test_size=test_size, random_state=seed,
        stratify=agent_labels,
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10_000)
    Xtrain = vec.fit_transform(Xtrain_txt)
    Xtest = vec.transform(Xtest_txt)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=seed)
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    acc = float(accuracy_score(ytest, ypred))
    labels = sorted(set(agent_labels))
    cm = confusion_matrix(ytest, ypred, labels=labels).tolist()
    _logger.info("TF-IDF+LR agent classifier test accuracy = %.3f", acc)
    return ClassifierResult(
        test_accuracy=acc,
        n_train=len(Xtrain_txt),
        n_test=len(Xtest_txt),
        confusion_matrix=cm,
        label_names=labels,
        pass_70=acc > 0.70,
    )
