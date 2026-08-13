"""Local, privacy-conscious HTTP server for the heart-risk prototype."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from risk_engine import ACUTE, load_model, predict


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MODEL = load_model()
EVALUATION = json.loads((ROOT / "model" / "evaluation.json").read_text(encoding="utf-8"))

FIELDS = [
    {"name": "age", "label": "연령", "type": "number", "min": 18, "max": 100, "step": 1, "unit": "세", "required": True, "help": "검사 시점의 만 나이"},
    {"name": "sex", "label": "성별", "type": "select", "required": True, "options": [{"value": 0, "label": "여성"}, {"value": 1, "label": "남성"}]},
    {"name": "cp", "label": "흉통 유형", "type": "select", "required": True, "options": [{"value": 1, "label": "전형적 협심증"}, {"value": 2, "label": "비전형적 협심증"}, {"value": 3, "label": "비협심증성 통증"}, {"value": 4, "label": "무증상"}], "help": "의료진이 분류한 흉통 범주"},
    {"name": "trestbps", "label": "안정 시 수축기 혈압", "type": "number", "min": 70, "max": 250, "step": 1, "unit": "mmHg", "required": True},
    {"name": "chol", "label": "혈청 콜레스테롤", "type": "number", "min": 80, "max": 700, "step": 1, "unit": "mg/dL", "required": True},
    {"name": "fbs", "label": "공복혈당 120mg/dL 초과", "type": "select", "required": True, "options": [{"value": 0, "label": "아니오"}, {"value": 1, "label": "예"}]},
    {"name": "restecg", "label": "안정 시 심전도", "type": "select", "required": True, "options": [{"value": 0, "label": "정상"}, {"value": 1, "label": "ST-T 이상"}, {"value": 2, "label": "좌심실 비대 가능"}]},
    {"name": "thalach", "label": "최대 심박수", "type": "number", "min": 40, "max": 250, "step": 1, "unit": "회/분", "required": True, "help": "운동부하검사에서 기록된 최대값"},
    {"name": "exang", "label": "운동 유발 협심증", "type": "select", "required": True, "options": [{"value": 0, "label": "아니오"}, {"value": 1, "label": "예"}]},
    {"name": "oldpeak", "label": "운동 유발 ST 저하", "type": "number", "min": -2, "max": 10, "step": 0.1, "unit": "mm", "required": True},
    {"name": "slope", "label": "ST 구간 기울기", "type": "select", "required": False, "options": [{"value": "", "label": "모름 / 미입력"}, {"value": 1, "label": "상승형"}, {"value": 2, "label": "평탄형"}, {"value": 3, "label": "하강형"}]},
    {"name": "ca", "label": "조영된 주요 혈관 수", "type": "select", "required": False, "options": [{"value": "", "label": "모름 / 미입력"}, {"value": 0, "label": "0개"}, {"value": 1, "label": "1개"}, {"value": 2, "label": "2개"}, {"value": 3, "label": "3개"}]},
    {"name": "thal", "label": "Thal 검사 결과", "type": "select", "required": False, "options": [{"value": "", "label": "모름 / 미입력"}, {"value": 3, "label": "정상"}, {"value": 6, "label": "고정 결손"}, {"value": 7, "label": "가역 결손"}]},
]


def metadata() -> dict:
    high = EVALUATION["oof_metrics_at_high_threshold"]
    site = [{"name": key, **value} for key, value in EVALUATION["sources"].items()]
    return {
        "app_name": "HeartLens",
        "subtitle": "설명 가능한 심장병 위험도 분류 연구용 시제품",
        "fields": FIELDS,
        "acute": [{"name": key, "label": value} for key, value in ACUTE.items()],
        "model": {
            "version": MODEL["model_version"], "training_rows": MODEL["training_summary"]["rows"],
            "auroc_oof": MODEL["training_summary"]["auroc_oof"],
            "average_precision_oof": MODEL["training_summary"]["average_precision_oof"],
            "brier_oof": MODEL["training_summary"]["brier_oof"],
            "sensitivity": high["sensitivity"], "specificity": high["specificity"],
            "thresholds": MODEL["thresholds"], "sites": site,
        },
        "safety": {
            "short": "연구·교육용 상대적 분류이며 의료 진단이 아닙니다.",
            "long": "UCI 공개자료에서 학습한 질환 존재 가능성의 상대적 분류입니다. 미래의 5년·10년 절대위험, 응급판단 또는 치료지시로 사용할 수 없습니다.",
            "privacy": "입력값은 이 기기에서만 계산되며 서버에 저장하지 않습니다.",
        },
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "HeartLens/1.0"

    def log_message(self, fmt, *args):
        # Do not log request bodies or query data.
        print(f"[{self.log_date_time_string()}] {self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}")

    def _security_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'")

    def _json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health": return self._json({"ok": True, "model_version": MODEL["model_version"]})
        if path == "/api/metadata": return self._json(metadata())
        if path == "/": path = "/index.html"
        safe = posixpath.normpath(unquote(path)).lstrip("/")
        target = (STATIC / safe).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self._json({"ok": False, "error": "파일을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
        content = target.read_bytes()
        kind = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK); self._security_headers()
        self.send_header("Content-Type", kind + ("; charset=utf-8" if kind.startswith("text/") or kind == "application/javascript" else ""))
        self.send_header("Content-Length", str(len(content))); self.end_headers(); self.wfile.write(content)

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self._json({"ok": False, "error": "지원하지 않는 주소입니다."}, HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024: raise ValueError("요청 크기가 올바르지 않습니다.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = predict(payload, MODEL)
            status = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
            return self._json(result, status)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return self._json({"ok": False, "errors": [str(exc)]}, HTTPStatus.BAD_REQUEST)


def main():
    parser = argparse.ArgumentParser(description="Run HeartLens locally")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"HeartLens running at {url}")
    print("Stop with Ctrl+C. Patient inputs are not persisted.")
    if not args.no_browser:
        try: webbrowser.open(url)
        except Exception: pass
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__":
    main()
