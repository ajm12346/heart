"""Train the heart-risk prototype without scikit-learn.

The script uses the four processed UCI Heart Disease sites, fits a regularized
logistic regression with deterministic cross-validation, selects conservative
prototype tiers, and writes a portable JSON model consumed by the web app.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
FILES = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va_long_beach": "processed.va.data",
}
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = {
    "cp": [1, 2, 3, 4],
    "restecg": [0, 1, 2],
    "slope": [1, 2, 3],
    "thal": [3, 6, 7],
}
RAW_FEATURES = NUMERIC + BINARY + list(CATEGORICAL)
LABELS_KO = {
    "age": "연령", "sex": "성별", "cp": "흉통 유형",
    "trestbps": "안정 시 수축기 혈압", "chol": "혈청 콜레스테롤",
    "fbs": "공복혈당 120mg/dL 초과", "restecg": "안정 시 심전도",
    "thalach": "최대 심박수", "exang": "운동 유발 협심증",
    "oldpeak": "운동 유발 ST 저하", "slope": "ST 구간 기울기",
    "ca": "조영된 주요 혈관 수", "thal": "Thal 검사 결과",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data() -> pd.DataFrame:
    frames = []
    for site, filename in FILES.items():
        path = RAW / filename
        if not path.exists():
            raise FileNotFoundError(f"Official UCI file is missing: {path}")
        df = pd.read_csv(path, names=COLUMNS, na_values="?")
        df["site"] = site
        df["target_binary"] = (pd.to_numeric(df["target"], errors="coerce") > 0).astype(int)
        frames.append(df)
    all_data = pd.concat(frames, ignore_index=True)
    all_data["source_row"] = np.arange(1, len(all_data) + 1)
    return all_data


class Preprocessor:
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.medians = {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in NUMERIC}
        self.means = {}
        self.scales = {}
        for c in NUMERIC:
            values = pd.to_numeric(df[c], errors="coerce").fillna(self.medians[c]).astype(float)
            self.means[c] = float(values.mean())
            scale = float(values.std(ddof=0))
            self.scales[c] = scale if scale > 1e-9 else 1.0
        self.modes = {}
        for c in BINARY + list(CATEGORICAL):
            mode = pd.to_numeric(df[c], errors="coerce").mode(dropna=True)
            self.modes[c] = int(mode.iloc[0]) if len(mode) else 0
        self.feature_names = ["intercept"]
        self.feature_base = ["intercept"]
        for c in NUMERIC:
            self.feature_names.append(f"{c}_z")
            self.feature_base.append(c)
        for c in BINARY:
            self.feature_names.append(c)
            self.feature_base.append(c)
        for c, levels in CATEGORICAL.items():
            for level in levels[1:]:  # first level is the reference
                self.feature_names.append(f"{c}={level}")
                self.feature_base.append(c)
        for c in RAW_FEATURES:
            self.feature_names.append(f"{c}_missing")
            self.feature_base.append(c)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in df.iterrows():
            out = [1.0]
            missing = {c: bool(pd.isna(row[c])) for c in RAW_FEATURES}
            for c in NUMERIC:
                value = self.medians[c] if missing[c] else float(row[c])
                out.append((value - self.means[c]) / self.scales[c])
            for c in BINARY:
                value = self.modes[c] if missing[c] else int(float(row[c]))
                out.append(float(value))
            for c, levels in CATEGORICAL.items():
                value = self.modes[c] if missing[c] else int(float(row[c]))
                for level in levels[1:]:
                    out.append(1.0 if value == level else 0.0)
            out.extend(1.0 if missing[c] else 0.0 for c in RAW_FEATURES)
            rows.append(out)
        return np.asarray(rows, dtype=float)

    def to_json(self) -> dict:
        return {
            "numeric": NUMERIC,
            "binary": BINARY,
            "categorical": CATEGORICAL,
            "raw_features": RAW_FEATURES,
            "medians": self.medians,
            "means": self.means,
            "scales": self.scales,
            "modes": self.modes,
            "feature_names": self.feature_names,
            "feature_base": self.feature_base,
        }


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 0.12, max_iter: int = 120) -> np.ndarray:
    beta = np.zeros(X.shape[1], dtype=float)
    penalty = np.ones(X.shape[1]); penalty[0] = 0.0
    for _ in range(max_iter):
        p = sigmoid(X @ beta)
        w = np.maximum(p * (1.0 - p), 1e-5)
        grad = X.T @ (p - y) / len(y) + l2 * penalty * beta
        hessian = (X.T * w) @ X / len(y) + np.diag(l2 * penalty + 1e-8)
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ grad
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return beta


def stratified_folds(df: pd.DataFrame, k: int = 5, seed: int = 20260813) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    group_key = df["site"].astype(str) + "|" + df["target_binary"].astype(str)
    for _, idx in group_key.groupby(group_key).groups.items():
        shuffled = np.asarray(list(idx), dtype=int)
        rng.shuffle(shuffled)
        for i, value in enumerate(shuffled):
            folds[i % k].append(int(value))
    return [np.asarray(sorted(x), dtype=int) for x in folds]


def rank_auc(y: np.ndarray, p: np.ndarray) -> float:
    pos = y == 1; neg = y == 0
    if not pos.any() or not neg.any(): return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float); ranks[order] = np.arange(1, len(p) + 1)
    # Average tied ranks.
    for value in np.unique(p):
        mask = p == value
        if mask.sum() > 1: ranks[mask] = ranks[mask].mean()
    n1, n0 = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    ys = y[order]
    total = ys.sum()
    if total == 0: return float("nan")
    cum = np.cumsum(ys)
    precision = cum / (np.arange(len(ys)) + 1)
    return float((precision * ys).sum() / total)


def confusion_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    safe = lambda a, b: float(a / b) if b else None
    return {
        "threshold": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": safe(tp, tp + fn), "specificity": safe(tn, tn + fp),
        "ppv": safe(tp, tp + fp), "npv": safe(tn, tn + fn),
        "accuracy": safe(tp + tn, len(y)),
    }


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 8) -> list[dict]:
    result = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            result.append({"lower": float(lo), "upper": float(hi), "n": int(mask.sum()),
                           "mean_probability": float(p[mask].mean()), "observed_rate": float(y[mask].mean())})
    return result


def select_thresholds(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    high_candidates = []
    for t in np.linspace(0.15, 0.85, 141):
        m = confusion_metrics(y, p, float(t))
        if m["sensitivity"] is not None and m["sensitivity"] >= 0.85:
            high_candidates.append(float(t))
    high = max(high_candidates) if high_candidates else 0.50
    high = float(np.clip(high, 0.40, 0.72))

    low_candidates = []
    for t in np.linspace(0.10, max(0.11, high - 0.10), 100):
        low_mask = p < t
        if low_mask.sum() >= max(30, int(0.10 * len(y))):
            npv = float((y[low_mask] == 0).mean())
            if npv >= 0.90:
                low_candidates.append(float(t))
    low = max(low_candidates) if low_candidates else min(0.30, high - 0.12)
    low = float(np.clip(low, 0.18, high - 0.10))
    return round(low, 3), round(high, 3)


def metric_summary(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    m = confusion_metrics(y, p, threshold)
    m.update({
        "n": int(len(y)), "positive_rate": float(y.mean()),
        "auroc": rank_auc(y, p), "average_precision": average_precision(y, p),
        "brier": float(np.mean((p - y) ** 2)),
    })
    return m


def cross_validate(df: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    y_all = df["target_binary"].to_numpy(dtype=float)
    oof = np.zeros(len(df), dtype=float)
    fold_reports = []
    folds = stratified_folds(df)
    all_idx = np.arange(len(df))
    for fold_no, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(all_idx, test_idx)
        prep = Preprocessor().fit(df.iloc[train_idx])
        X_train = prep.transform(df.iloc[train_idx]); X_test = prep.transform(df.iloc[test_idx])
        beta = fit_logistic(X_train, y_all[train_idx])
        oof[test_idx] = sigmoid(X_test @ beta)
        fold_reports.append(metric_summary(y_all[test_idx], oof[test_idx], 0.5) | {"fold": fold_no})
    return oof, fold_reports


def leave_site_out(df: pd.DataFrame) -> list[dict]:
    reports = []
    y = df["target_binary"].to_numpy(dtype=float)
    for site in FILES:
        test = np.where(df["site"].to_numpy() == site)[0]
        train = np.where(df["site"].to_numpy() != site)[0]
        prep = Preprocessor().fit(df.iloc[train])
        beta = fit_logistic(prep.transform(df.iloc[train]), y[train])
        p = sigmoid(prep.transform(df.iloc[test]) @ beta)
        reports.append(metric_summary(y[test], p, 0.5) | {"held_out_site": site})
    return reports


def clean_number(value):
    if value is None or (isinstance(value, float) and math.isnan(value)): return None
    return float(value)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    export_cols = ["source_row", "site"] + COLUMNS + ["target_binary"]
    df[export_cols].to_csv(PROCESSED / "heart_disease_all_sites.csv", index=False, encoding="utf-8-sig")

    y = df["target_binary"].to_numpy(dtype=float)
    oof, fold_reports = cross_validate(df)
    low_t, high_t = select_thresholds(y, oof)
    prep = Preprocessor().fit(df)
    X = prep.transform(df)
    beta = fit_logistic(X, y)

    site_stats = {}
    for site, g in df.groupby("site"):
        site_stats[site] = {
            "rows": int(len(g)), "positive": int(g["target_binary"].sum()),
            "positive_rate": float(g["target_binary"].mean()),
            "missing_cells": int(g[RAW_FEATURES].isna().sum().sum()),
        }
    report = {
        "model_version": "uci-logistic-2026.08.13-v1",
        "created": "2026-08-13",
        "training_rows": int(len(df)),
        "target": "UCI angiographic heart-disease presence: 0 vs 1-4",
        "not_intended_for": ["clinical diagnosis", "emergency triage", "treatment decisions", "future absolute risk"],
        "sources": site_stats,
        "raw_sha256": {name: file_sha256(RAW / filename) for name, filename in FILES.items()},
        "oof_metrics_at_high_threshold": metric_summary(y, oof, high_t),
        "oof_metrics_at_0_5": metric_summary(y, oof, 0.5),
        "fold_metrics": fold_reports,
        "leave_one_site_out": leave_site_out(df),
        "calibration_bins": calibration_bins(y, oof),
        "thresholds": {"low_attention": low_t, "attention_high": high_t},
    }
    model = {
        "schema_version": 1,
        "model_version": report["model_version"],
        "created": report["created"],
        "purpose": "Research/education prototype for relative heart-disease likelihood classification",
        "preprocessor": prep.to_json(),
        "coefficients": [float(x) for x in beta],
        "thresholds": report["thresholds"],
        "labels_ko": LABELS_KO,
        "training_summary": {
            "rows": int(len(df)), "positive_rate": float(y.mean()),
            "auroc_oof": float(rank_auc(y, oof)), "average_precision_oof": float(average_precision(y, oof)),
            "brier_oof": float(np.mean((oof - y) ** 2)),
        },
    }
    (MODEL_DIR / "heart_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "rows": len(df), "positive_rate": round(float(y.mean()), 4),
        "auroc_oof": round(rank_auc(y, oof), 4), "ap_oof": round(average_precision(y, oof), 4),
        "brier_oof": round(float(np.mean((oof - y) ** 2)), 4),
        "thresholds": report["thresholds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
