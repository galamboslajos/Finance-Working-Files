"""One-fit, train-only probability estimators for the frozen development pilot.

Estimator libraries are imported only when explicitly fitting. Validation labels,
early stopping, searches, resampling, calibration and RET have no entry point here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import math
from numbers import Integral

import numpy as np
import pandas as pd

from .ret import CLASS_IDS, align_class_probabilities
from .universe import DataContractError


@dataclass(frozen=True)
class PilotArmResult:
    probabilities: np.ndarray
    metadata: dict
    estimator: object


def _matrix(value, name):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise DataContractError(f"{name} must be a finite numerical matrix") from error
    if array.ndim != 2 or array.shape[1] < 1 or array.dtype.kind not in "fiu":
        raise DataContractError(f"{name} must be a two-dimensional real numerical matrix, without boolean or text values")
    if not np.isfinite(array).all():
        raise DataContractError(f"{name} must already be finite after frozen train-only preprocessing")
    float32_bound = float(np.finfo(np.float32).max)
    if array.size and (float(array.max()) > float32_bound or float(array.min()) < -float32_bound):
        raise DataContractError(f"{name} cannot be represented by a finite float32 matrix")
    result = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise DataContractError(f"{name} overflowed during float32 preparation")
    return result


def _training_labels(value, n):
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) != n:
        raise DataContractError("Training classes must be a one-dimensional vector aligned with train_X")
    objects = raw.astype(object, copy=False)
    if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, (Integral, float, np.floating))
           or not math.isfinite(item) or int(item) != item or int(item) not in CLASS_IDS
           for item in objects):
        raise DataContractError("Selected training labels must be finite integer canonical classes 1 through 10; never impute a target")
    labels = raw.astype(np.int64)
    absent = sorted(set(CLASS_IDS) - set(labels.tolist()))
    if absent:
        raise DataContractError(f"Training partition is missing canonical classes: {absent}; no probability zero padding")
    return labels


def _constructor(kind):
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier
    from xgboost import XGBClassifier
    return XGBClassifier


def _json_parameters(value):
    if isinstance(value, dict):
        return {str(key): _json_parameters(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_parameters(item) for item in value]
    if isinstance(value, np.generic):
        return _json_parameters(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_package_default": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _sha(array):
    return "sha256:" + hashlib.sha256(memoryview(array).cast("B") if array.size else b"").hexdigest()


def _parameters(kind, parameters):
    if not isinstance(parameters, dict) or not parameters or any(not isinstance(key, str) for key in parameters):
        raise DataContractError("Every arm requires its explicit frozen parameter dictionary")
    result = dict(parameters)
    shared = {"n_estimators", "n_jobs", "random_state"}
    required = (shared | {"max_features", "max_depth", "min_samples_leaf", "bootstrap", "class_weight"}
                if kind == "rf" else
                shared | {"objective", "num_class", "tree_method", "max_depth", "learning_rate", "subsample", "colsample_bytree", "eval_metric"})
    if required - result.keys():
        raise DataContractError(f"Missing explicit {kind} parameters: {sorted(required - result.keys())}")
    if (isinstance(result["n_estimators"], bool) or not isinstance(result["n_estimators"], Integral)
            or result["n_estimators"] < 1 or result["n_jobs"] != 1
            or isinstance(result["random_state"], bool) or not isinstance(result["random_state"], Integral)):
        raise DataContractError("Pilot fits require positive fixed rounds/trees, one worker and an explicit integer seed")
    if kind == "rf":
        if result["class_weight"] is not None or result.get("warm_start", False) or result.get("oob_score", False):
            raise DataContractError("Pilot forest cannot use class weights, warm starts or out-of-bag selection")
    elif (result["objective"] != "multi:softprob" or result["num_class"] != 10
          or result.get("early_stopping_rounds") is not None or result.get("callbacks") is not None
          or result.get("device", "cpu") != "cpu"):
        raise DataContractError("Pilot XGBoost requires ten-class soft probabilities on CPU and no early stopping/callbacks")
    return result


def fit_predict_pilot_arm(kind, train_X, train_y, predict_X, *, parameters, package_versions):
    """Fit once on supplied supervised training rows and predict every supplied row.

    The caller enforces dated training/validation separation before supplying
    matrices. This worker accepts no validation targets or evaluation set. Both
    matrices are finite float32 copies; labels retain canonical meaning. XGBoost
    uses native labels 0..9, with the explicit 0->1, ..., 9->10 mapping persisted.
    Its objective softmax is evaluated from native margins in float64, avoiding
    float32 probability-sum drift without clipping or repairing probabilities.
    """
    if kind not in {"rf", "xgb"}:
        raise DataContractError("Pilot estimator kind must be 'rf' or 'xgb'")
    parameters = _parameters(kind, parameters)
    package = "scikit-learn" if kind == "rf" else "xgboost"
    if not isinstance(package_versions, dict) or not isinstance(package_versions.get(package), str):
        raise DataContractError(f"Explicit pinned package_versions must include {package}")
    installed = version(package)
    if package_versions[package] != installed:
        raise DataContractError(f"Installed {package} {installed} differs from frozen version {package_versions[package]}")
    if isinstance(train_X, pd.DataFrame):
        if not train_X.columns.is_unique or not train_X.index.is_unique:
            raise DataContractError("train_X columns and row index must be unique")
        if isinstance(train_y, pd.Series) and not train_X.index.equals(train_y.index):
            raise DataContractError("Training class Series index differs from train_X")
    if isinstance(predict_X, pd.DataFrame):
        if not predict_X.columns.is_unique or not predict_X.index.is_unique:
            raise DataContractError("predict_X columns and row index must be unique")
        if isinstance(train_X, pd.DataFrame) and not train_X.columns.equals(predict_X.columns):
            raise DataContractError("Prediction columns must equal frozen training columns in the same order")
    training, prediction = _matrix(train_X, "train_X"), _matrix(predict_X, "predict_X")
    if not len(training) or training.shape[1] != prediction.shape[1]:
        raise DataContractError("Pilot requires nonempty training rows and the same train/prediction matrix width")
    labels = _training_labels(train_y, len(training))
    encoded = labels if kind == "rf" else labels - 1
    estimator = _constructor(kind)(**parameters)
    estimator.fit(training, encoded)
    native_ids = list(estimator.classes_)
    expected_native = list(CLASS_IDS) if kind == "rf" else list(range(10))
    if (len(native_ids) != 10 or any(isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral) for item in native_ids)
            or set(native_ids) != set(expected_native)):
        raise DataContractError("Fitted estimator classes differ from the complete explicit native class mapping")
    canonical_ids = native_ids if kind == "rf" else [int(item) + 1 for item in native_ids]
    if not len(prediction):
        values = np.empty((0, 10), dtype=np.float64)
    elif kind == "rf":
        values = estimator.predict_proba(prediction)
    else:
        margins = np.asarray(estimator.predict(prediction, output_margin=True))
        if margins.ndim != 2 or margins.shape != (len(prediction), 10) or margins.dtype.kind not in "fiu" or not np.isfinite(margins).all():
            raise DataContractError("XGBoost must return finite ten-class native margins")
        shifted = margins.astype(np.float64) - margins.max(axis=1, keepdims=True).astype(np.float64)
        exponentials = np.exp(shifted)
        values = exponentials / exponentials.sum(axis=1, keepdims=True)
    probabilities = align_class_probabilities(values, class_ids=canonical_ids)
    metadata = {
        "estimator_kind": kind, "package": package, "package_version": installed,
        "explicit_parameters": _json_parameters(parameters),
        "effective_parameters": _json_parameters(estimator.get_params(deep=False)),
        "fit_count": 1, "validation_labels_supplied_to_estimator": False,
        "early_stopping_used": False, "parameter_search_performed": False,
        "class_weighting_applied": False, "probability_calibration_performed": False,
        "native_class_ids": [int(item) for item in native_ids],
        "native_to_canonical_mapping": [{"native_class_id": int(item), "canonical_class_id": int(canonical)}
                                        for item, canonical in zip(native_ids, canonical_ids)],
        "probability_class_ids": list(CLASS_IDS),
        "probability_extraction": ("native_predict_proba_float64" if kind == "rf" else "native_output_margins_then_objective_softmax_float64"),
        "probability_clipping_performed": False, "probability_repair_renormalization_performed": False,
        "prepared_matrix_dtype": "float32", "train_rows": len(training), "prediction_rows": len(prediction),
        "matrix_width": training.shape[1], "prepared_train_matrix_sha256": _sha(training),
        "prepared_prediction_matrix_sha256": _sha(prediction), "encoded_training_labels_sha256": _sha(np.ascontiguousarray(encoded)),
        "train_class_counts": {str(item): int(np.sum(labels == item)) for item in CLASS_IDS},
    }
    return PilotArmResult(probabilities, metadata, estimator)
