#!/usr/bin/env python3
"""
Method-aligned machine learning pipeline for OVCF screening.

This script rewrites the earlier Claude-generated pipeline so that the
implementation matches the current Methods section in Manuscript0314.docx.

Key alignments with the manuscript
----------------------------------
1. Uses pre-defined data splits only:
   - training cohort (n≈103)
   - internal validation cohort (n≈45)
   - external validation cohort (n≈56)
2. Builds the final feature vector from:
   - 22 LLM-derived continuous indicators
   - 7 binary modality-missingness indicators
   - 4 clinical variables (age, sex, weight, BMI)
3. Evaluates the seven manuscript-specified candidate classifiers:
   - Logistic Regression
   - SVM (RBF)
   - KNN
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Histogram-based Gradient Boosting
4. Uses median imputation fitted on the training cohort only.
5. Applies z-score standardization to Logistic Regression, SVM, and KNN only.
6. Tunes hyperparameters via five-fold cross-validated grid search.
7. Reports default-threshold (0.5) and Youden-optimized threshold metrics.
8. Computes ROC, PR, calibration, feature-importance, missingness, and optional
   explanation-based interpretability analyses.
9. Uses DeLong tests for within-cohort AUC comparisons between models.

Notes
-----
- Because the upstream LLM export format may vary across iterations, this file
  includes a BEST-EFFORT alias dictionary and a clearly isolated feature block
  that can be adjusted in one place if column names differ.
- The manuscript states that some statistical analyses were performed in R.
  Here the equivalent analyses are implemented directly in Python so that the
  pipeline is self-contained.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl  # noqa: F401  # used by pandas ExcelWriter engine
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Feature specification aligned to the current manuscript.
# -----------------------------------------------------------------------------
# 22 continuous indicators (best-effort implementation aligned to the written Methods).
# This selection also preserves the key feature names already referenced in the Results,
# such as RollR_HipShoulderCoordination, SupSit_ExecutionTime, SupSit_MotionCoordination,
# SitSup_TrunkControl, and Lateral_TrunkInclination.
SCORE_FEATURES_22: List[str] = [
    # posture images
    "Back_ParaspinalTension",
    "Back_PelvicRotation",
    "Back_ScapularSymmetry",
    "Back_SpinalVerticality",
    "Front_GlobalBalance",
    "Front_PelvicLeveling",
    "Front_ShoulderSymmetry",
    "Front_SpinalAlignment",
    "Lat_LumbarLordosis",
    "Lat_SagittalBalance",
    "Lat_ThoracicKyphosis",
    "Lat_TrunkInclination",
    # functional videos
    "RollL_HipShoulderCoordination",
    "RollL_PainDuringMovement",
    "RollR_HipShoulderCoordination",
    "RollR_PainDuringMovement",
    "SitSup_ArmAssistance",
    "SitSup_ExecutionTime",
    "SitSup_TrunkControl",
    "SupSit_ExecutionTime",
    "SupSit_MotionCoordination",
    "SupSit_TrunkControl",
]

CLINICAL_FEATURES: List[str] = ["age", "sex", "weight_kg", "bmi"]

MODALITY_TO_FEATURES: Dict[str, List[str]] = {
    "anterior": ["Front_GlobalBalance", "Front_PelvicLeveling", "Front_ShoulderSymmetry", "Front_SpinalAlignment"],
    "posterior": ["Back_ParaspinalTension", "Back_PelvicRotation", "Back_ScapularSymmetry", "Back_SpinalVerticality"],
    "lateral": ["Lat_LumbarLordosis", "Lat_SagittalBalance", "Lat_ThoracicKyphosis", "Lat_TrunkInclination"],
    "roll_left": ["RollL_HipShoulderCoordination", "RollL_PainDuringMovement"],
    "roll_right": ["RollR_HipShoulderCoordination", "RollR_PainDuringMovement"],
    "sit_to_supine": ["SitSup_ArmAssistance", "SitSup_ExecutionTime", "SitSup_TrunkControl"],
    "supine_to_sit": ["SupSit_ExecutionTime", "SupSit_MotionCoordination", "SupSit_TrunkControl"],
}

MISSINGNESS_FEATURES: List[str] = [
    "Missing_Anterior",
    "Missing_Posterior",
    "Missing_Lateral",
    "Missing_RollLeft",
    "Missing_RollRight",
    "Missing_SitToSupine",
    "Missing_SupineToSit",
]

FINAL_FEATURES: List[str] = SCORE_FEATURES_22 + MISSINGNESS_FEATURES + CLINICAL_FEATURES

# Raw-column -> harmonized-column aliases.
COLUMN_ALIASES: Dict[str, str] = {
    # back/posterior
    "image_back__ParaspinalTensionAsymmetry__score": "Back_ParaspinalTension",
    "image_back__PelvicRotation__score": "Back_PelvicRotation",
    "image_back__ScapularSymmetry__score": "Back_ScapularSymmetry",
    "image_back__SpinalVerticality__score": "Back_SpinalVerticality",
    # front/anterior
    "image_frontal__GlobalBalance__score": "Front_GlobalBalance",
    "image_frontal__PelvicLeveling__score": "Front_PelvicLeveling",
    "image_frontal__ShoulderSymmetry__score": "Front_ShoulderSymmetry",
    "image_frontal__SpinalAlignment__score": "Front_SpinalAlignment",
    "image_frontal__CoronalCurvature__score": "Front_CoronalCurvature",
    # lateral
    "image_lateral__LumbarLordosis__score": "Lat_LumbarLordosis",
    "image_lateral__SagittalBalance__score": "Lat_SagittalBalance",
    "image_lateral__ThoracicKyphosis__score": "Lat_ThoracicKyphosis",
    "image_lateral__TrunkInclination__score": "Lat_TrunkInclination",
    "image_lateral__HeadPosition__score": "Lat_HeadPosition",
    # roll left
    "video_roll_left__HipShoulderCoordination__score": "RollL_HipShoulderCoordination",
    "video_roll_left__PainDuringMovement__score": "RollL_PainDuringMovement",
    "video_roll_left__RollingSmoothness__score": "RollL_RollingSmoothness",
    "video_roll_left__SupportHandUsage__score": "RollL_SupportHandUsage",
    # roll right
    "video_roll_right__HipShoulderCoordination__score": "RollR_HipShoulderCoordination",
    "video_roll_right__PainDuringMovement__score": "RollR_PainDuringMovement",
    "video_roll_right__RollingSmoothness__score": "RollR_RollingSmoothness",
    "video_roll_right__SupportHandUsage__score": "RollR_SupportHandUsage",
    # sit to supine
    "video_sit_to_supine__ArmAssistance__score": "SitSup_ArmAssistance",
    "video_sit_to_supine__ExecutionTime__score": "SitSup_ExecutionTime",
    "video_sit_to_supine__MotionCoordination__score": "SitSup_MotionCoordination",
    "video_sit_to_supine__PainResponseLevel__score": "SitSup_PainResponseLevel",
    "video_sit_to_supine__TrunkControl__score": "SitSup_TrunkControl",
    # supine to sit
    "video_supine_to_sit__ArmAssistance__score": "SupSit_ArmAssistance",
    "video_supine_to_sit__ExecutionTime__score": "SupSit_ExecutionTime",
    "video_supine_to_sit__MotionCoordination__score": "SupSit_MotionCoordination",
    "video_supine_to_sit__PainResponseLevel__score": "SupSit_PainResponseLevel",
    "video_supine_to_sit__TrunkControl__score": "SupSit_TrunkControl",
    # clinical aliases
    "weight": "weight_kg",
    "Weight": "weight_kg",
    "body_weight": "weight_kg",
    "BMI": "bmi",
    "bmi_value": "bmi",
    "gender": "sex",
}

SEX_MAP = {
    "m": 1,
    "male": 1,
    "man": 1,
    "1": 1,
    1: 1,
    "f": 0,
    "female": 0,
    "woman": 0,
    "0": 0,
    0: 0,
}

TARGET_MAP = {
    "positive": 1,
    "ovcf": 1,
    "fracture": 1,
    "case": 1,
    "1": 1,
    1: 1,
    "negative": 0,
    "control": 0,
    "non-ovcf": 0,
    "non_ovcf": 0,
    "0": 0,
    0: 0,
}

# Optional lexicon for differential vocabulary analysis.
OVCF_LEXICON = [
    "kyphosis",
    "stooped",
    "guarding",
    "pain",
    "hesitant",
    "stiff",
    "asymmetric",
    "imbalance",
    "lean",
    "compensation",
    "slow",
    "unstable",
    "limited",
    "support",
    "difficulty",
    "postural",
    "deformity",
    "thoracic",
    "lumbar",
    "sagittal",
]


@dataclass
class SplitData:
    train: pd.DataFrame
    internal: pd.DataFrame
    external: pd.DataFrame


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path}")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def standardize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        unique = set(vals.dropna().astype(int).unique().tolist())
        if unique.issubset({0, 1}):
            return vals.astype("Int64")
    mapped = series.astype(str).str.strip().str.lower().map(TARGET_MAP)
    return mapped.astype("Int64")


def standardize_sex(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        unique = set(vals.dropna().astype(int).unique().tolist())
        if unique.issubset({0, 1}):
            return vals
    return series.astype(str).str.strip().str.lower().map(SEX_MAP)


def compute_bmi(df: pd.DataFrame) -> pd.Series:
    if "bmi" in df.columns and df["bmi"].notna().any():
        return pd.to_numeric(df["bmi"], errors="coerce")
    if "weight_kg" in df.columns and "height_cm" in df.columns:
        w = pd.to_numeric(df["weight_kg"], errors="coerce")
        h_cm = pd.to_numeric(df["height_cm"], errors="coerce")
        h_m = h_cm / 100.0
        bmi = w / (h_m ** 2)
        bmi[(h_m <= 0) | h_m.isna()] = np.nan
        return bmi
    return pd.Series(np.nan, index=df.index)


def rename_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    overlapping = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}
    return df.rename(columns=overlapping)


def maybe_decode_ordinal_text(df: pd.DataFrame) -> pd.DataFrame:
    mappings = {
        "RollL_SupportHandUsage": {"无需辅助": 0, "单手辅助": 1, "双手用力推床": 2},
        "RollR_SupportHandUsage": {"无需辅助": 0, "单手辅助": 1, "双手用力推床": 2},
        "SitSup_ArmAssistance": {"无需辅助": 0, "部分支撑": 1, "明显依赖支撑": 2},
        "SupSit_ArmAssistance": {"无需辅助": 0, "部分支撑": 1, "明显依赖支撑": 2},
    }
    for col, mapping in mappings.items():
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(mapping).where(df[col].isna(), df[col].map(mapping))
    return df


def build_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    indicator_names = {
        "anterior": "Missing_Anterior",
        "posterior": "Missing_Posterior",
        "lateral": "Missing_Lateral",
        "roll_left": "Missing_RollLeft",
        "roll_right": "Missing_RollRight",
        "sit_to_supine": "Missing_SitToSupine",
        "supine_to_sit": "Missing_SupineToSit",
    }
    for modality, cols in MODALITY_TO_FEATURES.items():
        available_cols = [c for c in cols if c in df.columns]
        if not available_cols:
            df[indicator_names[modality]] = 1
            continue
        all_missing = df[available_cols].isna().all(axis=1)
        df[indicator_names[modality]] = all_missing.astype(int)
    return df


def prepare_dataframe(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = df.copy()
    df = rename_known_columns(df)
    df = maybe_decode_ordinal_text(df)

    # Remove obvious header-artifact rows if present.
    if "group" in df.columns:
        df = df[~df["group"].astype(str).str.lower().isin(["group", "nan", "none"])]

    if "target" not in df.columns:
        if "group" in df.columns:
            df["target"] = standardize_target(df["group"])
        elif "label" in df.columns:
            df["target"] = standardize_target(df["label"])
        else:
            raise KeyError(f"{split_name}: could not find target column. Expected one of: target/group/label")
    else:
        df["target"] = standardize_target(df["target"])

    if "sex" in df.columns:
        df["sex"] = standardize_sex(df["sex"])
    else:
        df["sex"] = np.nan

    if "weight_kg" not in df.columns:
        df["weight_kg"] = np.nan
    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")

    if "age" not in df.columns:
        df["age"] = np.nan
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    if "height_cm" not in df.columns:
        df["height_cm"] = np.nan
    df["height_cm"] = pd.to_numeric(df["height_cm"], errors="coerce")

    df["bmi"] = compute_bmi(df)

    # Force all LLM-derived score columns to numeric.
    for col in SCORE_FEATURES_22:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = build_missingness_indicators(df)

    # Ensure all final feature columns exist.
    for col in FINAL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    return df


def load_splits(args: argparse.Namespace) -> SplitData:
    if args.train and args.internal and args.external:
        train_df = read_table(Path(args.train))
        internal_df = read_table(Path(args.internal))
        external_df = read_table(Path(args.external))
    elif args.input and args.split_col:
        full = read_table(Path(args.input))
        if args.split_col not in full.columns:
            raise KeyError(f"Split column '{args.split_col}' not found in {args.input}")
        split_norm = full[args.split_col].astype(str).str.strip().str.lower()
        train_df = full[split_norm.isin({"train", "training"})].copy()
        internal_df = full[split_norm.isin({"internal", "internal_validation", "val", "validation"})].copy()
        external_df = full[split_norm.isin({"external", "external_validation", "test_external"})].copy()
        if train_df.empty or internal_df.empty or external_df.empty:
            raise ValueError(
                "Could not recover train/internal/external splits from the split column. "
                "Expected values like train / internal / external."
            )
    else:
        raise ValueError(
            "Provide either --train/--internal/--external, or a single --input with --split-col."
        )

    train_df = prepare_dataframe(train_df, "train")
    internal_df = prepare_dataframe(internal_df, "internal")
    external_df = prepare_dataframe(external_df, "external")
    return SplitData(train=train_df, internal=internal_df, external=external_df)


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------
def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 2000, seed: int = RANDOM_STATE) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return (np.nan, np.nan)
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    thr = thresholds[idx]
    # roc_curve may return inf as the first threshold.
    if not np.isfinite(thr):
        thr = 0.5
    return float(thr)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    return {
        "Threshold": float(threshold),
        "Accuracy": float(acc),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "PPV": float(ppv),
        "NPV": float(npv),
        "F1": float(f1),
        "AUC": float(auc),
        "AP": float(ap),
        "Brier": float(brier),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    default_metrics = classification_metrics(y_true, y_prob, threshold=0.5)
    y_thr = youden_threshold(y_true, y_prob)
    youden_metrics = classification_metrics(y_true, y_prob, threshold=y_thr)
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob)
    out = {f"Default_{k}": v for k, v in default_metrics.items()}
    out.update({f"Youden_{k}": v for k, v in youden_metrics.items()})
    out["AUC_CI_Low"] = ci_low
    out["AUC_CI_High"] = ci_high
    return out


def fisher_ci_from_r(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 3 or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = z - z_crit * se, z + z_crit * se
    return (float(np.tanh(lo)), float(np.tanh(hi)))


# -----------------------------------------------------------------------------
# DeLong test implementation
# -----------------------------------------------------------------------------
def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_pvalue(y_true: np.ndarray, pred_one: np.ndarray, pred_two: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    pred_one = np.asarray(pred_one)
    pred_two = np.asarray(pred_two)
    order = np.argsort(-y_true)
    label_1_count = int(np.sum(y_true == 1))
    preds = np.vstack([pred_one, pred_two])[:, order]
    aucs, delongcov = fast_delong(preds, label_1_count)
    diff = np.array([[1, -1]])
    var = float(diff @ delongcov @ diff.T)
    if var <= 0:
        return 1.0
    z = float(np.abs(np.diff(aucs)) / np.sqrt(var))
    return float(2 * (1 - stats.norm.cdf(z)))


# -----------------------------------------------------------------------------
# Modelling
# -----------------------------------------------------------------------------
def build_model_specs() -> Dict[str, Dict[str, object]]:
    return {
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]),
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs"],
            },
        },
        "SVM (RBF)": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]),
            "param_grid": {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__gamma": ["scale", 0.1, 0.01],
            },
        },
        "KNN": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            "param_grid": {
                "clf__n_neighbors": [3, 5, 7, 9, 11],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        },
        "Decision Tree": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]),
            "param_grid": {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [2, 3, 4, 5, 6, None],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4, 6],
            },
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
            ]),
            "param_grid": {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 5, 7, None],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", 0.5],
            },
        },
        "Gradient Boosting": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 200, 300],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__max_depth": [2, 3, 4],
                "clf__subsample": [0.8, 1.0],
            },
        },
        "Histogram-based GBM": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]),
            "param_grid": {
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__max_depth": [None, 3, 5],
                "clf__max_leaf_nodes": [15, 31],
                "clf__min_samples_leaf": [10, 20],
                "clf__l2_regularization": [0.0, 0.01],
            },
        },
    }


def fit_models(X_train: pd.DataFrame, y_train: np.ndarray, out_dir: Path) -> Tuple[Dict[str, object], pd.DataFrame]:
    specs = build_model_specs()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fitted: Dict[str, object] = {}
    rows = []

    for name, spec in specs.items():
        print(f"Tuning {name} ...")
        grid = GridSearchCV(
            estimator=clone(spec["pipeline"]),
            param_grid=spec["param_grid"],
            scoring="roc_auc",
            n_jobs=-1,
            cv=cv,
            refit=True,
            return_train_score=False,
        )
        grid.fit(X_train, y_train)
        fitted[name] = grid.best_estimator_
        rows.append({
            "Model": name,
            "Best_CV_AUC": float(grid.best_score_),
            "Best_Params": json.dumps(grid.best_params_, ensure_ascii=False),
        })

    tuning_df = pd.DataFrame(rows).sort_values("Best_CV_AUC", ascending=False).reset_index(drop=True)
    tuning_df.to_csv(out_dir / "model_tuning_summary.csv", index=False)
    return fitted, tuning_df


def predict_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        # Min-max map only as a fallback; this path should rarely be used here.
        denom = scores.max() - scores.min()
        if denom <= 0:
            return np.repeat(0.5, len(scores))
        return (scores - scores.min()) / denom
    raise AttributeError("Model does not support probability-like prediction.")


def evaluate_models(
    fitted_models: Dict[str, object],
    X_internal: pd.DataFrame,
    y_internal: np.ndarray,
    X_external: pd.DataFrame,
    y_external: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    internal_rows = []
    external_rows = []
    internal_probs: Dict[str, np.ndarray] = {}
    external_probs: Dict[str, np.ndarray] = {}

    for name, model in fitted_models.items():
        p_int = predict_probabilities(model, X_internal)
        p_ext = predict_probabilities(model, X_external)
        internal_probs[name] = p_int
        external_probs[name] = p_ext

        metrics_int = evaluate_probabilities(y_internal, p_int)
        metrics_ext = evaluate_probabilities(y_external, p_ext)
        internal_rows.append({"Model": name, **metrics_int})
        external_rows.append({"Model": name, **metrics_ext})

    # soft-vote ensemble = mean predicted probability across the seven tuned models
    ensemble_int = np.column_stack(list(internal_probs.values())).mean(axis=1)
    ensemble_ext = np.column_stack(list(external_probs.values())).mean(axis=1)
    internal_probs["Soft-vote Ensemble"] = ensemble_int
    external_probs["Soft-vote Ensemble"] = ensemble_ext
    internal_rows.append({"Model": "Soft-vote Ensemble", **evaluate_probabilities(y_internal, ensemble_int)})
    external_rows.append({"Model": "Soft-vote Ensemble", **evaluate_probabilities(y_external, ensemble_ext)})

    internal_df = pd.DataFrame(internal_rows).sort_values("Default_AUC", ascending=False).reset_index(drop=True)
    external_df = pd.DataFrame(external_rows).sort_values("Default_AUC", ascending=False).reset_index(drop=True)
    return internal_df, external_df, internal_probs, external_probs


def delong_against_best(results_df: pd.DataFrame, y_true: np.ndarray, prob_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    best_model = results_df.iloc[0]["Model"]
    best_prob = prob_dict[best_model]
    rows = []
    m = len(results_df)
    bonf_n = max(m - 1, 1)
    for model in results_df["Model"]:
        if model == best_model:
            p = np.nan
            p_adj = np.nan
        else:
            p = delong_pvalue(y_true, prob_dict[model], best_prob)
            p_adj = min(p * bonf_n, 1.0)
        rows.append({
            "Reference_Best_Model": best_model,
            "Compared_Model": model,
            "DeLong_P": p,
            "DeLong_P_Bonferroni": p_adj,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Demographic and missingness analyses
# -----------------------------------------------------------------------------
def cohort_demographic_comparison(train_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["age", "weight_kg", "bmi", "height_cm"]:
        if col not in train_df.columns or col not in external_df.columns:
            continue
        x = pd.to_numeric(train_df[col], errors="coerce").dropna()
        y = pd.to_numeric(external_df[col], errors="coerce").dropna()
        if len(x) == 0 or len(y) == 0:
            continue
        stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append({
            "Variable": col,
            "Train_Mean_SD": f"{x.mean():.2f} ± {x.std(ddof=1):.2f}",
            "External_Mean_SD": f"{y.mean():.2f} ± {y.std(ddof=1):.2f}",
            "Test": "Mann-Whitney U",
            "P": p,
        })

    if "sex" in train_df.columns and "sex" in external_df.columns:
        train_counts = train_df["sex"].dropna().astype(int).value_counts().reindex([0, 1], fill_value=0)
        ext_counts = external_df["sex"].dropna().astype(int).value_counts().reindex([0, 1], fill_value=0)
        table = np.array([train_counts.values, ext_counts.values])
        chi2, p, _, _ = stats.chi2_contingency(table)
        rows.append({
            "Variable": "sex",
            "Train_Mean_SD": f"F={train_counts[0]}, M={train_counts[1]}",
            "External_Mean_SD": f"F={ext_counts[0]}, M={ext_counts[1]}",
            "Test": "Chi-squared",
            "P": p,
        })
    return pd.DataFrame(rows)


def missingness_summary(all_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_col = "target"
    rows = []
    for modality_feature in MISSINGNESS_FEATURES:
        g1 = all_df.loc[all_df[group_col] == 1, modality_feature].mean()
        g0 = all_df.loc[all_df[group_col] == 0, modality_feature].mean()
        table = pd.crosstab(all_df[group_col], all_df[modality_feature]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        try:
            _, p = stats.fisher_exact(table.values)
        except Exception:
            p = np.nan
        rows.append({
            "Indicator": modality_feature,
            "Control_Missing_Rate": g0,
            "OVCF_Missing_Rate": g1,
            "P_Fisher": p,
        })
    rate_df = pd.DataFrame(rows)

    all_df = all_df.copy()
    all_df["Missing_Modality_Count"] = all_df[MISSINGNESS_FEATURES].sum(axis=1)
    g1 = all_df.loc[all_df[group_col] == 1, "Missing_Modality_Count"]
    g0 = all_df.loc[all_df[group_col] == 0, "Missing_Modality_Count"]
    stat, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    count_df = pd.DataFrame([
        {
            "Control_Mean": g0.mean(),
            "OVCF_Mean": g1.mean(),
            "MannWhitneyU_P": p,
        }
    ])
    return rate_df, count_df


# -----------------------------------------------------------------------------
# Interpretability analyses
# -----------------------------------------------------------------------------
def get_estimator_from_pipeline(model) -> object:
    if isinstance(model, Pipeline):
        return model.named_steps["clf"]
    return model


def feature_importance_tables(model, X_external: pd.DataFrame, y_external: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    estimator = get_estimator_from_pipeline(model)
    if hasattr(estimator, "feature_importances_"):
        impurity = pd.DataFrame({
            "Feature": FINAL_FEATURES,
            "Impurity_Importance": estimator.feature_importances_,
        }).sort_values("Impurity_Importance", ascending=False).reset_index(drop=True)
    else:
        impurity = pd.DataFrame({"Feature": FINAL_FEATURES, "Impurity_Importance": np.nan})

    perm = permutation_importance(
        model,
        X_external,
        y_external,
        scoring="roc_auc",
        n_repeats=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    permutation_df = pd.DataFrame({
        "Feature": FINAL_FEATURES,
        "Permutation_Mean": perm.importances_mean,
        "Permutation_SD": perm.importances_std,
    }).sort_values("Permutation_Mean", ascending=False).reset_index(drop=True)
    return impurity, permutation_df


def extract_first_numeric_quantity(text: str) -> Optional[float]:
    if pd.isna(text):
        return None
    matches = re.findall(r"[-+]?\d*\.?\d+", str(text))
    if not matches:
        return None
    try:
        return float(matches[0])
    except Exception:
        return None


def run_score_explanation_concordance(explanations_df: pd.DataFrame) -> pd.DataFrame:
    required = {"indicator", "score", "explanation"}
    if not required.issubset(explanations_df.columns):
        raise KeyError("Explanation concordance file must contain columns: indicator, score, explanation")
    targets = {
        "SpinalVerticality",
        "ScapularSymmetry",
        "TrunkInclination",
        "SagittalBalance",
        "ExecutionTime",
        "ThoracicKyphosis",
    }
    df = explanations_df.copy()
    df = df[df["indicator"].isin(targets)].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["physical_quantity"] = df["explanation"].apply(extract_first_numeric_quantity)

    rows = []
    for indicator, sub in df.groupby("indicator"):
        sub = sub.dropna(subset=["score", "physical_quantity"])
        if len(sub) < 3:
            rows.append({"Indicator": indicator, "N": len(sub), "Pearson_r": np.nan, "CI_Low": np.nan, "CI_High": np.nan, "P": np.nan})
            continue
        r, p = stats.pearsonr(sub["score"], sub["physical_quantity"])
        ci_low, ci_high = fisher_ci_from_r(float(r), len(sub))
        rows.append({"Indicator": indicator, "N": len(sub), "Pearson_r": r, "CI_Low": ci_low, "CI_High": ci_high, "P": p})
    return pd.DataFrame(rows).sort_values("Pearson_r", ascending=False)


def run_differential_vocabulary(explanations_df: pd.DataFrame, lexicon: Sequence[str] = OVCF_LEXICON) -> pd.DataFrame:
    required = {"explanation", "target"}
    if not required.issubset(explanations_df.columns):
        raise KeyError("Vocabulary file must contain columns: explanation and target")
    df = explanations_df.copy()
    df["target"] = standardize_target(df["target"]).astype(int)
    text = df["explanation"].fillna("").astype(str).str.lower()

    rows = []
    for term in lexicon:
        present = text.str.contains(rf"\b{re.escape(term.lower())}\b", regex=True)
        a = int(((df["target"] == 1) & present).sum())
        b = int(((df["target"] == 1) & ~present).sum())
        c = int(((df["target"] == 0) & present).sum())
        d = int(((df["target"] == 0) & ~present).sum())
        table = np.array([[a, b], [c, d]])
        odds_ratio, p = stats.fisher_exact(table)
        log2_or = np.nan if odds_ratio <= 0 else np.log2(odds_ratio)
        rows.append({
            "Term": term,
            "OVCF_Present": a,
            "Control_Present": c,
            "Odds_Ratio": odds_ratio,
            "Log2_OR": log2_or,
            "NegLog10_P": -np.log10(max(p, 1e-300)),
            "P": p,
        })
    vocab_df = pd.DataFrame(rows).sort_values("P")
    return vocab_df


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_roc_curves(y_true: np.ndarray, prob_dict: Dict[str, np.ndarray], title: str, save_path: Path) -> None:
    plt.figure(figsize=(8, 7))
    for name, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curves(y_true: np.ndarray, prob_dict: Dict[str, np.ndarray], title: str, save_path: Path) -> None:
    plt.figure(figsize=(8, 7))
    for name, probs in prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        plt.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration(y_true: np.ndarray, prob_dict: Dict[str, np.ndarray], title: str, save_path: Path) -> None:
    plt.figure(figsize=(8, 7))
    for name, probs in prob_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=8, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=1.8, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_for_model(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, title: str, save_path: Path) -> None:
    thr = 0.5
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    plt.xticks([0, 1], ["Control", "OVCF"])
    plt.yticks([0, 1], ["Control", "OVCF"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{title}\n{model_name} @ threshold=0.50")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importance_bar(df: pd.DataFrame, value_col: str, title: str, save_path: Path, top_n: int = 20) -> None:
    top = df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(6, top_n * 0.28)))
    plt.barh(top["Feature"], top[value_col])
    plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_missingness_rates(df: pd.DataFrame, save_path: Path) -> None:
    plot_df = df.copy()
    plot_df["Group"] = plot_df["target"].map({0: "Control", 1: "OVCF"})
    rates = []
    for feature in MISSINGNESS_FEATURES:
        grp = plot_df.groupby("Group")[feature].mean().reset_index()
        grp["Indicator"] = feature
        rates.append(grp)
    rate_df = pd.concat(rates, axis=0, ignore_index=True)

    pivot = rate_df.pivot(index="Indicator", columns="Group", values=feature if False else 0)
    # easier manual bar plot
    ind = np.arange(len(MISSINGNESS_FEATURES))
    width = 0.35
    control_vals = [rate_df[(rate_df["Indicator"] == f) & (rate_df["Group"] == "Control")][feature].values[0] if False else plot_df.loc[plot_df["target"] == 0, f].mean() for f in MISSINGNESS_FEATURES]
    ovcf_vals = [plot_df.loc[plot_df["target"] == 1, f].mean() for f in MISSINGNESS_FEATURES]
    plt.figure(figsize=(10, 6))
    plt.bar(ind - width / 2, control_vals, width=width, label="Control")
    plt.bar(ind + width / 2, ovcf_vals, width=width, label="OVCF")
    plt.xticks(ind, MISSINGNESS_FEATURES, rotation=35, ha="right")
    plt.ylabel("Missing rate")
    plt.title("Modality missingness rates by group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_probability_histograms(y_true: np.ndarray, probs: np.ndarray, title: str, save_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.hist(probs[y_true == 0], bins=12, alpha=0.7, label="Control")
    plt.hist(probs[y_true == 1], bins=12, alpha=0.7, label="OVCF")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    out_dir = ensure_output_dir(Path(args.out_dir))
    splits = load_splits(args)

    print("Loaded splits:")
    print(f"  Train:    {len(splits.train)}")
    print(f"  Internal: {len(splits.internal)}")
    print(f"  External: {len(splits.external)}")

    X_train = splits.train[FINAL_FEATURES].copy()
    y_train = splits.train["target"].to_numpy()
    X_internal = splits.internal[FINAL_FEATURES].copy()
    y_internal = splits.internal["target"].to_numpy()
    X_external = splits.external[FINAL_FEATURES].copy()
    y_external = splits.external["target"].to_numpy()

    fitted_models, tuning_df = fit_models(X_train, y_train, out_dir)
    internal_df, external_df, internal_probs, external_probs = evaluate_models(
        fitted_models, X_internal, y_internal, X_external, y_external
    )

    delong_internal = delong_against_best(internal_df, y_internal, internal_probs)
    delong_external = delong_against_best(external_df, y_external, external_probs)

    # Demographic comparability (training vs external, per manuscript).
    demographics_df = cohort_demographic_comparison(splits.train, splits.external)

    # Missingness analysis over the full cohort.
    all_df = pd.concat([splits.train, splits.internal, splits.external], axis=0, ignore_index=True)
    missingness_rate_df, missingness_count_df = missingness_summary(all_df)

    # Gradient Boosting interpretability (as specified in the manuscript).
    gb_model = fitted_models["Gradient Boosting"]
    impurity_df, permutation_df = feature_importance_tables(gb_model, X_external, y_external)

    # Save core tables.
    with pd.ExcelWriter(out_dir / "ML_Results_Tables_method_aligned.xlsx", engine="openpyxl") as writer:
        tuning_df.to_excel(writer, sheet_name="CV_Tuning", index=False)
        internal_df.to_excel(writer, sheet_name="Internal_Performance", index=False)
        external_df.to_excel(writer, sheet_name="External_Performance", index=False)
        delong_internal.to_excel(writer, sheet_name="DeLong_Internal", index=False)
        delong_external.to_excel(writer, sheet_name="DeLong_External", index=False)
        demographics_df.to_excel(writer, sheet_name="Demographics", index=False)
        missingness_rate_df.to_excel(writer, sheet_name="Missingness_Rates", index=False)
        missingness_count_df.to_excel(writer, sheet_name="Missingness_Count", index=False)
        impurity_df.to_excel(writer, sheet_name="GB_Impurity_Importance", index=False)
        permutation_df.to_excel(writer, sheet_name="GB_Permutation_Importance", index=False)

    # Plots.
    plot_roc_curves(y_internal, internal_probs, "ROC curves: internal validation", out_dir / "Fig_ROC_Internal.png")
    plot_roc_curves(y_external, external_probs, "ROC curves: external validation", out_dir / "Fig_ROC_External.png")
    plot_pr_curves(y_internal, internal_probs, "Precision-recall curves: internal validation", out_dir / "Fig_PR_Internal.png")
    plot_pr_curves(y_external, external_probs, "Precision-recall curves: external validation", out_dir / "Fig_PR_External.png")
    plot_calibration(y_internal, internal_probs, "Calibration curves: internal validation", out_dir / "Fig_Calibration_Internal.png")
    plot_calibration(y_external, external_probs, "Calibration curves: external validation", out_dir / "Fig_Calibration_External.png")

    best_external_model = external_df.iloc[0]["Model"]
    plot_confusion_for_model(
        y_external,
        external_probs[best_external_model],
        best_external_model,
        "Confusion matrix on external validation",
        out_dir / "Fig_Confusion_BestExternal.png",
    )
    plot_feature_importance_bar(
        impurity_df,
        "Impurity_Importance",
        "Gradient Boosting impurity-based feature importance",
        out_dir / "Fig_GB_Impurity_Importance.png",
    )
    plot_feature_importance_bar(
        permutation_df,
        "Permutation_Mean",
        "Gradient Boosting permutation importance (external validation)",
        out_dir / "Fig_GB_Permutation_Importance.png",
    )
    plot_missingness_rates(all_df, out_dir / "Fig_Missingness_Rates.png")

    if "Gradient Boosting" in external_probs:
        plot_probability_histograms(
            y_external,
            external_probs["Gradient Boosting"],
            "Gradient Boosting predicted probabilities on external validation",
            out_dir / "Fig_GB_Probability_Distribution_External.png",
        )

    # Optional explanation-based modules.
    if args.explanations:
        expl_path = Path(args.explanations)
        if expl_path.exists():
            expl_df = read_table(expl_path)
            try:
                concordance_df = run_score_explanation_concordance(expl_df)
                with pd.ExcelWriter(out_dir / "Interpretability_Explanation_Results.xlsx", engine="openpyxl") as writer:
                    concordance_df.to_excel(writer, sheet_name="Score_Explanation", index=False)
                concordance_df.to_csv(out_dir / "Score_Explanation_Concordance.csv", index=False)
            except Exception as exc:
                print(f"[WARN] score–explanation concordance skipped: {exc}")
            try:
                vocab_df = run_differential_vocabulary(expl_df)
                vocab_df.to_csv(out_dir / "Differential_Clinical_Vocabulary.csv", index=False)
            except Exception as exc:
                print(f"[WARN] differential vocabulary analysis skipped: {exc}")

    # Save final feature manifest for easy checking.
    pd.DataFrame({
        "Feature": FINAL_FEATURES,
        "Feature_Group": (["LLM_Score"] * len(SCORE_FEATURES_22))
        + (["Missingness"] * len(MISSINGNESS_FEATURES))
        + (["Clinical"] * len(CLINICAL_FEATURES)),
    }).to_csv(out_dir / "feature_manifest.csv", index=False)

    print("\nDone.")
    print(f"Outputs written to: {out_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Method-aligned OVCF ML pipeline")
    parser.add_argument("--train", type=str, default=None, help="Path to training split file (.xlsx/.csv/.parquet)")
    parser.add_argument("--internal", type=str, default=None, help="Path to internal validation split file")
    parser.add_argument("--external", type=str, default=None, help="Path to external validation split file")
    parser.add_argument("--input", type=str, default=None, help="Single table containing all splits")
    parser.add_argument("--split-col", type=str, default=None, help="Column name identifying train/internal/external splits")
    parser.add_argument("--explanations", type=str, default=None, help="Optional explanation-level table for interpretability analysis")
    parser.add_argument("--out-dir", type=str, default="/mnt/data/ovcf_method_aligned_outputs", help="Directory for all result files")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
