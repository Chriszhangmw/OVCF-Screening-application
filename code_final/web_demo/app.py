from __future__ import annotations

import base64
import csv
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
CODE_DIR = APP_DIR.parent
PROJECT_ROOT = CODE_DIR.parent
STATIC_DIR = APP_DIR / "static"
LOG_DIR = APP_DIR / "runtime_logs"
RUNTIME_ROOT = APP_DIR / "runtime_cases"
ACTIVE_CASES_DIR = RUNTIME_ROOT / f"session-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
SAMPLE_OUTPUT_DIR = CODE_DIR / "patient_prediction_outputs" / "positive1"
MODEL_RUNTIME = APP_DIR / "model_runtime.py"
DEFAULT_MODEL_PYTHON = Path(r"D:\software_installation\anaconda\python.exe")
APP_LOG = LOG_DIR / "app.log.jsonl"

HOST = os.environ.get("OVCF_WEB_HOST", "0.0.0.0")
PORT = int(os.environ.get("OVCF_WEB_PORT", "7860"))

ACTION_LABELS = {
    "frontal": "Frontal posture image",
    "back": "Back posture image",
    "lateral": "Lateral posture image",
    "roll_left": "Left roll video",
    "roll_right": "Right roll video",
    "supine_to_sit": "Supine-to-sit video",
    "sit_to_supine": "Sit-to-supine video",
}

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".mp4": "video/mp4",
}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    return []


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_log(log_path: Path, event: str, **data: Any) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **sanitize_for_json(data)}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    except Exception:
        pass


def read_log_file(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"time": "", "event": "raw_log", "message": line})
    except Exception as exc:
        rows.append({"time": "", "event": "log_read_error", "message": str(exc)})
    return rows


def collect_case_logs(case_dir: Path, limit: int = 240) -> list[dict[str, Any]]:
    paths = [
        case_dir / "analysis_log.jsonl",
        case_dir / "live_outputs" / "analysis_log.jsonl",
        case_dir / "runtime_stderr.log",
        case_dir / "runtime_stdout.log",
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        source = path.name
        for row in read_log_file(path, limit=limit):
            row.setdefault("source", source)
            rows.append(row)
    return rows[-limit:]


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", name or "upload.bin", flags=re.UNICODE)
    safe = safe.strip("._")
    return safe or "upload.bin"


def sanitize_case_id(raw: str) -> str:
    raw = re.sub(r"[^\w\-]+", "-", raw.strip())
    raw = raw.strip("-")
    if raw:
        return raw[:48]
    return f"case-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(sanitize_for_json(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler: BaseHTTPRequestHandler, message: str, status: int = 400, extra: dict[str, Any] | None = None) -> None:
    payload = {"ok": False, "error": message}
    if extra:
        payload.update(extra)
    json_response(handler, payload, status)


def parse_request_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def save_uploaded_media(case_dir: Path, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    upload_dir = case_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    case_log = case_dir / "analysis_log.jsonl"
    saved: list[dict[str, Any]] = []
    for index, item in enumerate(files):
        name = sanitize_filename(str(item.get("name") or f"media-{index + 1}.bin"))
        role = str(item.get("role") or "unassigned")
        mime = str(item.get("type") or "application/octet-stream")
        encoded = str(item.get("data") or "")
        if "," in encoded and encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        try:
            binary = base64.b64decode(encoded, validate=False)
        except Exception:
            binary = b""
        if not binary:
            append_log(case_log, "upload_skipped_empty", index=index, name=name, role=role, mime=mime)
            continue
        target = upload_dir / f"{index + 1:02d}_{name}"
        target.write_bytes(binary)
        item_saved = {
            "role": role,
            "label": ACTION_LABELS.get(role, role),
            "name": name,
            "path": str(target),
            "mime": mime,
            "size": len(binary),
        }
        append_log(case_log, "upload_saved", **item_saved)
        saved.append(item_saved)
    return saved


def get_cases_dir() -> Path:
    ACTIVE_CASES_DIR.mkdir(parents=True, exist_ok=True)
    return ACTIVE_CASES_DIR


def clear_cases() -> dict[str, Any]:
    global ACTIVE_CASES_DIR

    root = ACTIVE_CASES_DIR.resolve()
    app_root = APP_DIR.resolve()
    if app_root not in root.parents and root != app_root:
        raise RuntimeError("Cleanup directory validation failed.")
    warnings: list[str] = []

    def unlock_and_retry(func: Any, path: str, _exc_info: Any) -> None:
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if ACTIVE_CASES_DIR.exists():
        try:
            shutil.rmtree(ACTIVE_CASES_DIR, onerror=unlock_and_retry)
        except Exception as exc:
            warnings.append(f"{ACTIVE_CASES_DIR.name}: {exc}")
    ACTIVE_CASES_DIR = RUNTIME_ROOT / f"session-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    ACTIVE_CASES_DIR.mkdir(parents=True, exist_ok=True)
    return {"warnings": warnings[:8], "warning_count": len(warnings)}


def load_structured_outputs(sample_dir: Path) -> dict[str, Any]:
    structured_dir = sample_dir / "structured_json"
    out: dict[str, Any] = {}
    if not structured_dir.exists():
        return out
    for path in sorted(structured_dir.glob("*.json")):
        out[path.stem] = read_json(path, {})
    return out


def build_completeness(uploaded_media: list[dict[str, Any]]) -> dict[str, Any]:
    uploaded_roles = sorted({m["role"] for m in uploaded_media if m.get("role") and m.get("role") != "unassigned"})
    expected_roles = list(ACTION_LABELS)
    return {
        "uploaded_roles": uploaded_roles,
        "expected_roles": expected_roles,
        "uploaded_count": len(uploaded_roles),
        "expected_count": len(expected_roles),
        "ratio": round(len(uploaded_roles) / len(expected_roles), 3) if expected_roles else 0,
    }


def clone_report_for_case(report: dict[str, Any], meta: dict[str, Any], patient_id: str) -> dict[str, Any]:
    cloned = json.loads(json.dumps(report, ensure_ascii=False))
    cloned["patient"] = patient_id
    cloned["clinical_meta_used"] = {
        "age": meta.get("age"),
        "sex": meta.get("sex"),
        "height_cm": meta.get("height_cm"),
        "weight_kg": meta.get("weight_kg"),
        "smoking_history": meta.get("smoking_history"),
        "drinking_history": meta.get("drinking_history"),
        "injury_history": meta.get("injury_history"),
    }
    cloned["clinical_meta_used"] = {k: v for k, v in cloned["clinical_meta_used"].items() if v not in (None, "")}
    cloned["clinical_meta_source"] = "web_form"
    return cloned


def build_cached_result(case_id: str, meta: dict[str, Any], uploaded_media: list[dict[str, Any]], runtime_error: str | None = None) -> dict[str, Any]:
    report = read_json(SAMPLE_OUTPUT_DIR / "prediction_report.json", {})
    report = clone_report_for_case(report, meta, case_id)
    explanation_table = read_csv_rows(SAMPLE_OUTPUT_DIR / "prediction_explanation_table.csv")
    llm_explanations = read_csv_rows(SAMPLE_OUTPUT_DIR / "patient_llm_explanations.csv")
    structured = load_structured_outputs(SAMPLE_OUTPUT_DIR)
    return {
        "ok": True,
        "case_id": case_id,
        "analysis_mode": "cached_research_demo",
        "runtime_error": runtime_error,
        "report": report,
        "explanation_table": explanation_table,
        "llm_explanations": llm_explanations,
        "structured": structured,
        "media": uploaded_media,
        "completeness": build_completeness(uploaded_media),
        "logs": [],
        "note": "This is a cached demo result. Live analysis requires live mode plus an available model environment and API.",
    }


def try_live_analysis(case_dir: Path, case_id: str, meta: dict[str, Any], uploaded_media: list[dict[str, Any]]) -> dict[str, Any]:
    payload_path = case_dir / "runtime_payload.json"
    output_path = case_dir / "runtime_result.json"
    media_map = {}
    for item in uploaded_media:
        role = item.get("role")
        path = item.get("path")
        if role and role != "unassigned" and path and role not in media_map:
            media_map[role] = path

    case_log = case_dir / "analysis_log.jsonl"
    append_log(
        case_log,
        "live_analysis_prepare",
        case_id=case_id,
        upload_count=len(uploaded_media),
        media_roles=list(media_map.keys()),
        ignored_uploads=[m for m in uploaded_media if m.get("role") in {"", "unassigned"}],
    )

    write_json(
        payload_path,
        {
            "case_id": case_id,
            "patient_meta": meta,
            "media_map": media_map,
            "project_root": str(PROJECT_ROOT),
            "code_dir": str(CODE_DIR),
            "output_root": str(case_dir / "live_outputs"),
        },
    )

    model_python = os.environ.get("OVCF_MODEL_PYTHON")
    if not model_python:
        model_python = str(DEFAULT_MODEL_PYTHON) if DEFAULT_MODEL_PYTHON.exists() else sys.executable
    cmd = [model_python, str(MODEL_RUNTIME), "--payload", str(payload_path), "--result", str(output_path)]
    runtime_env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    for proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        runtime_env.pop(proxy_key, None)
    runtime_env["NO_PROXY"] = "dashscope.aliyuncs.com,localhost,127.0.0.1,::1"
    append_log(case_log, "runtime_start", python=model_python, media_map=media_map)
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=900, env=runtime_env)
    (case_dir / "runtime_stdout.log").write_text(completed.stdout or "", encoding="utf-8", errors="replace")
    (case_dir / "runtime_stderr.log").write_text(completed.stderr or "", encoding="utf-8", errors="replace")
    append_log(
        case_log,
        "runtime_finished",
        returncode=completed.returncode,
        stdout_tail=(completed.stdout or "")[-1000:],
        stderr_tail=(completed.stderr or "")[-2000:],
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "model runtime failed").strip()
        raise RuntimeError(detail[-2000:])
    result = read_json(output_path, None)
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError("model runtime did not produce a valid result")
    result.setdefault("completeness", build_completeness(uploaded_media))
    result["logs"] = collect_case_logs(case_dir)
    return result


class OVCFHandler(BaseHTTPRequestHandler):
    server_version = "OVCFWebDemo/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.serve_static(STATIC_DIR / "index.html")
            return
        if self.path == "/api/health":
            json_response(
                self,
                {
                    "ok": True,
                    "sample_output_available": (SAMPLE_OUTPUT_DIR / "prediction_report.json").exists(),
                    "model_runtime_available": MODEL_RUNTIME.exists(),
                },
            )
            return
        if self.path == "/api/logs":
            json_response(self, {"ok": True, "logs": read_log_file(APP_LOG, limit=300)})
            return
        if self.path.startswith("/static/"):
            rel = self.path.removeprefix("/static/").split("?", 1)[0]
            self.serve_static(STATIC_DIR / rel)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path == "/api/reset":
            try:
                result = clear_cases()
                message = "The current page has been cleared and switched to a new patient session."
                json_response(self, {"ok": True, "message": message, **result})
            except Exception as exc:
                error_response(self, f"Cleanup failed: {exc}", 500)
            return

        if self.path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            payload = parse_request_json(self)
        except Exception as exc:
            error_response(self, f"Invalid request data format: {exc}", 400)
            return

        meta = payload.get("patient") or {}
        case_id = sanitize_case_id(str(meta.get("case_id") or meta.get("patient_id") or ""))
        case_dir = get_cases_dir() / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        write_json(case_dir / "patient_meta.json", meta)
        append_log(APP_LOG, "analyze_request", case_id=case_id, demo_mode=payload.get("demo_mode"), allow_fallback=payload.get("allow_fallback"))
        append_log(case_dir / "analysis_log.jsonl", "request_received", case_id=case_id, demo_mode=payload.get("demo_mode"), file_count=len(payload.get("files") or []))

        try:
            uploaded_media = save_uploaded_media(case_dir, payload.get("files") or [])
        except Exception as exc:
            error_response(self, f"Unable to save uploaded files: {exc}", 500)
            return
        append_log(case_dir / "analysis_log.jsonl", "upload_summary", uploaded_media=uploaded_media, completeness=build_completeness(uploaded_media))

        demo_mode = bool(payload.get("demo_mode", True))
        allow_fallback = bool(payload.get("allow_fallback", True))
        if not demo_mode:
            try:
                live_result = try_live_analysis(case_dir, case_id, meta, uploaded_media)
                json_response(self, live_result)
                return
            except Exception as exc:
                append_log(case_dir / "analysis_log.jsonl", "live_analysis_error", error=str(exc))
                if not allow_fallback:
                    error_response(self, str(exc), 500, extra={"logs": collect_case_logs(case_dir)})
                    return
                result = build_cached_result(case_id, meta, uploaded_media, runtime_error=str(exc))
                result["logs"] = collect_case_logs(case_dir)
                json_response(self, result)
                return

        result = build_cached_result(case_id, meta, uploaded_media)
        result["logs"] = collect_case_logs(case_dir)
        json_response(self, result)

    def serve_static(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            static_root = STATIC_DIR.resolve()
            if static_root not in resolved.parents and resolved != static_root:
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            if not resolved.exists() or not resolved.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            body = resolved.read_bytes()
            content_type = CONTENT_TYPES.get(resolved.suffix.lower(), "application/octet-stream")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def main() -> None:
    get_cases_dir()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    append_log(APP_LOG, "server_start", host=HOST, port=PORT, project_root=str(PROJECT_ROOT))
    server = ThreadingHTTPServer((HOST, PORT), OVCFHandler)
    print(f"OVCF clinical web demo running at http://{HOST}:{PORT}")
    print(f"Project root: {PROJECT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
