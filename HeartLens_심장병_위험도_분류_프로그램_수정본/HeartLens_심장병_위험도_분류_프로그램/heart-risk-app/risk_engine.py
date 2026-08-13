"""Dependency-free inference and validation for the local web application."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "heart_model.json"

RANGES = {
    "age": (18, 100), "trestbps": (70, 250), "chol": (80, 700),
    "thalach": (40, 250), "oldpeak": (-2, 10), "ca": (0, 3),
}
VALID = {
    "sex": {0, 1}, "cp": {1, 2, 3, 4}, "fbs": {0, 1},
    "restecg": {0, 1, 2}, "exang": {0, 1}, "slope": {1, 2, 3},
    "thal": {3, 6, 7},
}
REQUIRED = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]
OPTIONAL = ["slope", "ca", "thal"]
ACUTE = {
    "chest_pain_now": "현재 지속되는 흉통 또는 가슴 압박감",
    "severe_breathlessness": "갑작스럽거나 심한 호흡곤란",
    "cold_sweat_fainting": "식은땀·실신 또는 실신 직전 느낌",
    "pain_spreading": "팔·턱·등으로 퍼지는 통증",
}


def load_model(path: Path = MODEL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_number(value: Any, field: str, *, integer: bool = False):
    if value is None or value == "": return None
    if isinstance(value, bool): return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field}: 숫자 형식이 아닙니다.")
    if not math.isfinite(number): raise ValueError(f"{field}: 유효한 숫자가 아닙니다.")
    return int(number) if integer else number


def validate_payload(payload: dict) -> tuple[dict, list[str], list[str]]:
    if not isinstance(payload, dict): raise ValueError("요청 데이터는 객체 형식이어야 합니다.")
    clean = {}
    errors, warnings = [], []
    for field in REQUIRED + OPTIONAL:
        try:
            integer = field in VALID or field in {"age", "ca"}
            clean[field] = _as_number(payload.get(field), field, integer=integer)
        except ValueError as exc:
            errors.append(str(exc)); clean[field] = None
    for field in REQUIRED:
        if clean.get(field) is None: errors.append(f"{field}: 필수 입력값입니다.")
    for field, (lo, hi) in RANGES.items():
        value = clean.get(field)
        if value is not None and not (lo <= value <= hi):
            errors.append(f"{field}: 허용범위 {lo}~{hi}를 벗어났습니다.")
    for field, allowed in VALID.items():
        value = clean.get(field)
        if value is not None and value not in allowed:
            errors.append(f"{field}: 허용값 {sorted(allowed)} 중 하나여야 합니다.")
    if clean.get("trestbps") and clean["trestbps"] >= 180:
        warnings.append("안정 시 수축기 혈압이 매우 높습니다. 측정 오류를 확인하고 의료진에게 상담하세요.")
    if clean.get("chol") and clean["chol"] >= 300:
        warnings.append("콜레스테롤 수치가 높은 범위입니다. 단위와 검사결과를 다시 확인하세요.")
    missing_optional = [f for f in OPTIONAL if clean.get(f) is None]
    if missing_optional:
        warnings.append("선택검사 일부가 없어 학습자료의 대표값으로 보완됩니다: " + ", ".join(missing_optional))
    return clean, errors, warnings


def _feature_vector(clean: dict, model: dict) -> list[float]:
    p = model["preprocessor"]
    missing = {c: clean.get(c) is None for c in p["raw_features"]}
    x = [1.0]
    for c in p["numeric"]:
        value = p["medians"][c] if missing[c] else float(clean[c])
        x.append((value - p["means"][c]) / p["scales"][c])
    for c in p["binary"]:
        value = p["modes"][c] if missing[c] else int(clean[c])
        x.append(float(value))
    for c, levels in p["categorical"].items():
        levels = [int(v) for v in levels]
        value = int(p["modes"][c]) if missing[c] else int(clean[c])
        x.extend(1.0 if value == level else 0.0 for level in levels[1:])
    x.extend(1.0 if missing[c] else 0.0 for c in p["raw_features"])
    return x


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def predict(payload: dict, model: dict | None = None) -> dict:
    model = model or load_model()
    clean, errors, warnings = validate_payload(payload)
    acute_selected = [label for key, label in ACUTE.items() if bool(payload.get("acute", {}).get(key))]
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "emergency": bool(acute_selected),
                "acute_symptoms": acute_selected}
    x = _feature_vector(clean, model)
    coefficients = model["coefficients"]
    logit = sum(a * b for a, b in zip(x, coefficients))
    probability = _sigmoid(logit)
    low = model["thresholds"]["low_attention"]
    high = model["thresholds"]["attention_high"]
    if probability < low:
        tier, tier_ko = "low", "낮음"
    elif probability < high:
        tier, tier_ko = "attention", "주의"
    else:
        tier, tier_ko = "high", "높음"

    grouped = {}
    bases = model["preprocessor"]["feature_base"]
    names = model["preprocessor"]["feature_names"]
    for i in range(1, len(x)):
        base = bases[i]
        grouped[base] = grouped.get(base, 0.0) + x[i] * coefficients[i]
    factors = []
    for base, contribution in sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]:
        factors.append({
            "feature": base,
            "label": model["labels_ko"].get(base, base),
            "contribution": round(contribution, 4),
            "direction": "위험 증가 방향" if contribution > 0 else "위험 감소 방향",
            "note": "모델 계산상의 연관성이며 인과관계를 의미하지 않습니다.",
        })
    if acute_selected:
        warnings.insert(0, "급성 경고 증상이 선택되었습니다. 이 분류 결과를 기다리거나 따르지 말고 즉시 119 또는 응급의료기관에 연락하세요.")
    return {
        "ok": True, "probability": round(probability, 6), "percentage": round(probability * 100, 1),
        "tier": tier, "tier_ko": tier_ko, "thresholds": model["thresholds"],
        "factors": factors, "warnings": warnings, "emergency": bool(acute_selected),
        "acute_symptoms": acute_selected, "model_version": model["model_version"],
        "training_summary": model["training_summary"],
        "disclaimer": "공개 UCI 자료에서 학습한 연구·교육용 상대적 분류입니다. 진단, 응급판단, 치료지시 또는 미래 절대위험이 아닙니다.",
        "input_quality": {"optional_missing": sum(clean.get(f) is None for f in OPTIONAL), "total_fields": len(REQUIRED) + len(OPTIONAL)},
        "normalized_input": clean,
    }
