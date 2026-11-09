"""Python/NumPy reference for the *translation* preprocessing.

Spec: TSControllerTranslationCoordinateTransform.forward in
training/components/ml/src/export/torch/preproc/controller_transform.py (lines 6-17):
subtract the first time sample from every feature, no rotation, no axis flip, no extra channel.

The other class in that file (TSControllerCoordinateTransform, PCA-based frame) needs the
reference quaternion of the recording; webxr/gesture_data_test.json has none, so it is not
reproduced here.
"""

import numpy as np


def translation_transform(x: np.ndarray) -> np.ndarray:
    """x: (F, T) -> (F, T), x[:, t] - x[:, 0]."""
    x = np.asarray(x, dtype=np.float32)
    return x - x[:, :1]
