from __future__ import annotations

# Optional TensorFlow dependency with a numpy fallback so installs don't fail.
from typing import Callable
import numpy as np

try:  # pragma: no cover - optional path
    import tensorflow as tf  # type: ignore
    _TF_AVAILABLE = True
except Exception:  # pragma: no cover - optional path
    tf = None  # type: ignore
    _TF_AVAILABLE = False


class TensorflowBiasScorer:
    def __init__(self) -> None:
        if _TF_AVAILABLE:
            # Tiny placeholder: embedding -> dense -> sigmoid
            self._vectorizer = tf.keras.layers.TextVectorization(  # type: ignore[attr-defined]
                output_mode="int",
                output_sequence_length=16,
                standardize="lower_and_strip_punctuation",
            )
            # Fit vectorizer on minimal vocabulary for demo; in real use, adapt with dataset
            self._vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["sample vocabulary for bias scoring model"]))  # type: ignore[attr-defined]

            self._model = tf.keras.Sequential(  # type: ignore[attr-defined]
                [
                    self._vectorizer,
                    tf.keras.layers.Embedding(input_dim=1000, output_dim=8, mask_zero=True),  # type: ignore[attr-defined]
                    tf.keras.layers.GlobalAveragePooling1D(),  # type: ignore[attr-defined]
                    tf.keras.layers.Dense(8, activation="relu"),  # type: ignore[attr-defined]
                    tf.keras.layers.Dense(1, activation="sigmoid"),  # type: ignore[attr-defined]
                ]
            )
            self._scorer: Callable[[str], float] = self._score_tf
        else:
            self._scorer = self._score_np

    def _score_tf(self, text: str) -> float:
        assert _TF_AVAILABLE and tf is not None
        pred = self._model.predict(np.array([text]), verbose=0)[0][0]
        return float(pred)

    def _score_np(self, text: str) -> float:
        x = (abs(hash(text)) % 1000) / 1000.0
        return float(1.0 / (1.0 + np.exp(-(x * 10 - 5))))

    def score(self, text: str) -> float:
        return float(self._scorer(text))

