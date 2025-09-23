from __future__ import annotations

import tensorflow as tf
import numpy as np


class TensorflowBiasScorer:
    def __init__(self) -> None:
        # Tiny placeholder: embedding -> dense -> sigmoid
        self._vectorizer = tf.keras.layers.TextVectorization(
            output_mode="int",
            output_sequence_length=16,
            standardize="lower_and_strip_punctuation",
        )
        # Fit vectorizer on minimal vocabulary for demo; in real use, adapt with dataset
        self._vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["sample vocabulary for bias scoring model"]))

        self._model = tf.keras.Sequential(
            [
                self._vectorizer,
                tf.keras.layers.Embedding(input_dim=1000, output_dim=8, mask_zero=True),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        # Random weights are fine for a placeholder

    def score(self, text: str) -> float:
        pred = self._model.predict(np.array([text]), verbose=0)[0][0]
        return float(pred)


