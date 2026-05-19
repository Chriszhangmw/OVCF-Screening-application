from __future__ import annotations

import argparse
import ast
import base64
import importlib.util
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests


SCORE_COLUMNS = [
    "image_back__ParaspinalTensionAsymmetry__score",
    "image_back__PelvicRotation__score",
    "image_back__ScapularSymmetry__score",
    "image_back__SpinalVerticality__score",
    "image_frontal__CoronalCurvature__score",
    "image_frontal__GlobalBalance__score",
    "image_frontal__PelvicLeveling__score",
    "image_frontal__ShoulderSymmetry__score",
    "image_frontal__SpinalAlignment__score",
    "image_lateral__HeadPosition__score",
    "image_lateral__LumbarLordosis__score",
    "image_lateral__SagittalBalance__score",
    "image_lateral__ThoracicKyphosis__score",
    "image_lateral__TrunkInclination__score",
    "video_roll_left__HipShoulderCoordination__score",
    "video_roll_left__PainDuringMovement__score",
    "video_roll_left__RollingSmoothness__score",
    "video_roll_left__SupportHandUsage__score",
    "video_roll_right__HipShoulderCoordination__score",
    "video_roll_right__PainDuringMovement__score",
    "video_roll_right__RollingSmoothness__score",
    "video_roll_right__SupportHandUsage__score",
    "video_sit_to_supine__ArmAssistance__score",
    "video_sit_to_supine__ExecutionTime__score",
    "video_sit_to_supine__MotionCoordination__score",
    "video_sit_to_supine__PainResponseLevel__score",
    "video_sit_to_supine__TrunkControl__score",
    "video_supine_to_sit__ArmAssistance__score",
    "video_supine_to_sit__ExecutionTime__score",
    "video_supine_to_sit__MotionCoordination__score",
    "video_supine_to_sit__PainResponseLevel__score",
    "video_supine_to_sit__TrunkControl__score",
]

DEMO_COLUMNS = ["age", "sex", "height_cm", "weight_kg", "smoking_history", "drinking_history", "injury_history"]
FEATURE_COLUMNS = DEMO_COLUMNS + SCORE_COLUMNS

ORDINAL_COLUMNS = {
    "video_roll_left__SupportHandUsage__score": "SupportHandUsage",
    "video_roll_right__SupportHandUsage__score": "SupportHandUsage",
    "video_sit_to_supine__ArmAssistance__score": "ArmAssistance",
    "video_supine_to_sit__ArmAssistance__score": "ArmAssistance",
}

ORDINAL_VALUE_MAPS = {
    "SupportHandUsage": {
        "无使用": 0.0,
        "无辅助": 0.0,
        "无需辅助": 0.0,
        "单手辅助": 1.0,
        "单手支撑": 1.0,
        "双手用力推床": 2.0,
        "双手发力辅助": 2.0,
        "双手辅助": 2.0,
    },
    "ArmAssistance": {
        "无支撑": 0.0,
        "无辅助": 0.0,
        "无需辅助": 0.0,
        "部分支撑": 1.0,
        "部分辅助": 1.0,
        "明显依赖支撑": 2.0,
        "明显依赖": 2.0,
        "双手支撑": 2.0,
    },
}

ACTION_TO_PREFIX = {
    "frontal": "image_frontal",
    "lateral": "image_lateral",
    "back": "image_back",
    "roll_left": "video_roll_left",
    "roll_right": "video_roll_right",
    "sit_to_supine": "video_sit_to_supine",
    "supine_to_sit": "video_supine_to_sit",
}

ACTION_TO_PROMPT_FILENAME = {
    "frontal": "image_frontal.txt",
    "lateral": "image_lateral.txt",
    "back": "image_back.txt",
    "supine_to_sit": "video_supine_to_sit.txt",
    "sit_to_supine": "video_sit_to_supine.txt",
    "roll_left": "video_roll.txt",
    "roll_right": "video_roll.txt",
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
NULL_STRINGS = {"", "nan", "none", "null", "信息缺失", "缺失", "na", "n/a", "-"}
DASHSCOPE_API_KEY = ""  # Set your DashScope API key here or via environment variable
LOG_ROWS: list[dict[str, Any]] = []


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def log_event(output_root: Path, event: str, **data: Any) -> None:
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **json_safe(data)}
    LOG_ROWS.append(row)
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with (output_root / "analysis_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    except Exception:
        pass


def is_missing_value(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and x.strip().lower() in NULL_STRINGS:
        return True
    return False


def normalize_sex(x: Any) -> float:
    if is_missing_value(x):
        return np.nan
    mapping = {"m": 1, "male": 1, "男": 1, "1": 1, "f": 0, "female": 0, "女": 0, "0": 0}
    return float(mapping.get(str(x).strip().lower(), np.nan))


def encode_binary_history(x: Any) -> float:
    if is_missing_value(x):
        return 0.0
    s = str(x).strip().lower()
    if s in {"无", "否", "no", "0", "false", "无明显", "无吸烟史", "无饮酒史"}:
        return 0.0
    if "无" in s and "有" not in s:
        return 0.0
    return 1.0


def encode_injury_history(x: Any) -> float:
    if is_missing_value(x):
        return 0.0
    s = str(x).strip().lower()
    if "高能量" in s or "车祸" in s:
        return 2.0
    if "低能量" in s or "平地" in s or "扭伤" in s:
        return 1.0
    return 0.0


def encode_ordinal_value(x: Any, kind: str) -> float:
    if is_missing_value(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    try:
        return float(s)
    except Exception:
        pass
    if s in ORDINAL_VALUE_MAPS[kind]:
        return ORDINAL_VALUE_MAPS[kind][s]
    for key, val in ORDINAL_VALUE_MAPS[kind].items():
        if key in s:
            return val
    return np.nan


def prepare_model_matrix(raw_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    df = raw_df.copy()
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    df["sex"] = df["sex"].apply(normalize_sex)
    df["smoking_history"] = df["smoking_history"].apply(encode_binary_history)
    df["drinking_history"] = df["drinking_history"].apply(encode_binary_history)
    df["injury_history"] = df["injury_history"].apply(encode_injury_history)
    for col, kind in ORDINAL_COLUMNS.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda v, k=kind: encode_ordinal_value(v, k))
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[feature_columns]


def parse_indicator_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        s = payload.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {"value": s}
    return {"value": payload}


def value_for_model(indicator: str, item: dict[str, Any]) -> Any:
    if indicator in {"ArmAssistance", "SupportHandUsage"} and "value" in item:
        return item.get("value")
    if "score" in item:
        return item.get("score")
    if "value" in item:
        return item.get("value")
    return np.nan


def load_prompt_content(prompt_py: Path) -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("ovcf_prompt_module", str(prompt_py))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return dict(module.PROMPTS_CONTENT)


def file_to_base64_data_url(file_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        mime_type = "video/mp4" if file_path.suffix.lower() == ".mp4" else "image/jpeg"
    data = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def extract_json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise


def parse_dashscope_response(resp: dict[str, Any]) -> dict[str, Any]:
    choices = resp.get("output", {}).get("choices", [])
    if not choices:
        return {"error": "empty choices", "raw_response": resp}
    content = choices[0].get("message", {}).get("content", [])
    text = None
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text = part["text"]
                break
    elif isinstance(content, str):
        text = content
    if not text:
        return {"error": "no text content", "raw_response": resp}
    try:
        return extract_json_from_text(text)
    except Exception:
        return {"raw_content": text, "raw_response": resp}


def call_dashscope(media_path: Path, prompt_text: str, api_key: str, model_name: str = "qwen-vl-max") -> dict[str, Any]:
    is_video = media_path.suffix.lower() in VIDEO_EXTS
    media_type = "video" if is_video else "image"
    payload = {
        "model": model_name,
        "input": {
            "messages": [{"role": "user", "content": [{"text": prompt_text}, {media_type: file_to_base64_data_url(media_path)}]}],
            "video_frame_sampling": {"strategy": "uniform", "sample_frame_count": 16},
        },
        "parameters": {"result_format": "message"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    session = requests.Session()
    session.trust_env = False
    resp = session.post(
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        headers=headers,
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()
    return parse_dashscope_response(resp.json())


def flatten_structured_json(structured: dict[str, dict[str, Any]], patient_id: str, patient_meta: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    row: dict[str, Any] = {"patient": patient_id}
    row.update(patient_meta)
    explanation_rows = []
    for action, data in structured.items():
        prefix = ACTION_TO_PREFIX.get(action)
        if not prefix or not isinstance(data, dict) or "error" in data:
            continue
        for indicator, raw_item in data.items():
            if indicator in {"error", "raw_content", "raw_response"}:
                continue
            item = parse_indicator_payload(raw_item)
            score_col = f"{prefix}__{indicator}__score"
            json_col = f"{prefix}__{indicator}__json"
            row[score_col] = value_for_model(indicator, item)
            row[json_col] = json.dumps(item, ensure_ascii=False)
            explanation_rows.append(
                {
                    "patient": patient_id,
                    "action": action,
                    "prefix": prefix,
                    "indicator": indicator,
                    "score_column": score_col,
                    "value_for_model": row[score_col],
                    "score": item.get("score", np.nan),
                    "value": item.get("value", np.nan),
                    "explanation": item.get("explanation", ""),
                    "json": json.dumps(item, ensure_ascii=False),
                }
            )
    return row, pd.DataFrame(explanation_rows)


def json_col_from_score_col(score_col: str) -> str:
    return score_col.replace("__score", "__json")


def feature_display_name(score_col: str) -> str:
    if "__" not in score_col:
        return score_col
    prefix, indicator, _ = score_col.split("__", 2)
    return f"{prefix} / {indicator}"


def get_json_explanation(row: dict[str, Any], score_col: str) -> tuple[str, Any, Any]:
    raw = row.get(json_col_from_score_col(score_col))
    if is_missing_value(raw):
        return "", np.nan, np.nan
    item = parse_indicator_payload(raw)
    return str(item.get("explanation", "")), item.get("score", np.nan), item.get("value", np.nan)


def build_interpretability_table(row: dict[str, Any], x_numeric: pd.DataFrame, bundle: dict[str, Any], top_n: int = 10) -> pd.DataFrame:
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    importances = bundle.get("feature_importances", {})
    ref = bundle.get("reference_stats", {})
    x_imputed = model.named_steps["imputer"].transform(x_numeric)[0]
    rows = []
    for i, feature in enumerate(feature_columns):
        raw_value = row.get(feature, np.nan)
        if is_missing_value(raw_value):
            continue
        stats = ref.get(feature, {})
        pos_mean = stats.get("positive_mean", np.nan)
        ctrl_mean = stats.get("control_mean", np.nan)
        median = stats.get("median", np.nan)
        iqr = stats.get("iqr", 1.0) or 1.0
        direction = np.sign(pos_mean - ctrl_mean) if np.isfinite(pos_mean) and np.isfinite(ctrl_mean) else 0.0
        z_like = (x_imputed[i] - median) / iqr if np.isfinite(median) else 0.0
        risk_signal = float(importances.get(feature, 0.0) * z_like * direction)
        if feature in SCORE_COLUMNS:
            explanation, llm_score, llm_value = get_json_explanation(row, feature)
            feature_type = "LLM_score"
        else:
            explanation, llm_score, llm_value = "Clinical metadata feature; no LLM explanation.", np.nan, raw_value
            feature_type = "clinical"
        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "display": feature_display_name(feature),
                "raw_value": raw_value,
                "numeric_value_after_encoding": float(x_imputed[i]) if np.isfinite(x_imputed[i]) else np.nan,
                "model_importance": float(importances.get(feature, 0.0)),
                "positive_mean": pos_mean,
                "control_mean": ctrl_mean,
                "risk_signal": risk_signal,
                "llm_score": llm_score,
                "llm_value": llm_value,
                "llm_explanation": explanation,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_signal"] = out["risk_signal"].abs()
    return out.sort_values(["risk_signal", "abs_signal"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def force_single_thread_estimator(model: Any) -> None:
    estimator = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]
    if hasattr(estimator, "set_params") and "n_jobs" in estimator.get_params():
        estimator.set_params(n_jobs=1)


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(payload["project_root"])
    code_dir = Path(payload["code_dir"])
    output_root = Path(payload["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    case_id = payload["case_id"]
    patient_meta = payload.get("patient_meta") or {}
    media_map = {k: Path(v) for k, v in (payload.get("media_map") or {}).items()}
    log_event(output_root, "runtime_payload_loaded", case_id=case_id, media_roles=list(media_map.keys()), media_paths=media_map)

    api_key = DASHSCOPE_API_KEY
    if not api_key:
        raise RuntimeError("DashScope API key was not found, so live image/video analysis cannot run.")
    log_event(output_root, "api_key_ready", source="environment_or_embedded", key_present=bool(api_key))

    prompts = load_prompt_content(code_dir / "prompt.py")
    log_event(output_root, "prompts_loaded", prompt_count=len(prompts), prompt_keys=sorted(prompts.keys()))
    structured_dir = output_root / "structured_json"
    structured_dir.mkdir(parents=True, exist_ok=True)
    structured = {}
    for action, path in media_map.items():
        if action not in ACTION_TO_PREFIX:
            log_event(output_root, "media_role_ignored", action=action, path=path, reason="unknown_role")
            continue
        prompt = prompts[ACTION_TO_PROMPT_FILENAME[action]].strip()
        log_event(output_root, "dashscope_action_start", action=action, path=path, exists=path.exists(), size=path.stat().st_size if path.exists() else None)
        result = None
        last_error = None
        for attempt in range(1, 4):
            try:
                result = call_dashscope(path, prompt, api_key)
                indicators = [k for k, v in result.items() if isinstance(v, dict) and ("score" in v or "value" in v)]
                log_event(output_root, "dashscope_action_success", action=action, attempt=attempt, result_keys=list(result.keys()), indicator_count=len(indicators), indicators=indicators)
                break
            except Exception as exc:
                last_error = exc
                log_event(output_root, "dashscope_action_error", action=action, attempt=attempt, error=str(exc))
                time.sleep(2)
        if result is None:
            result = {"error": str(last_error)}
            log_event(output_root, "dashscope_action_failed", action=action, error=str(last_error))
        (structured_dir / f"{action}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        structured[action] = result

    row, llm_explanations = flatten_structured_json(structured, case_id, patient_meta)
    present_score_cols = [c for c in SCORE_COLUMNS if c in row and not is_missing_value(row.get(c))]
    log_event(output_root, "structured_flattened", present_score_count=len(present_score_cols), present_score_cols=present_score_cols, explanation_rows=len(llm_explanations))
    feature_df = pd.DataFrame([row])
    feature_df.to_csv(output_root / "patient_structured_features.csv", index=False, encoding="utf-8-sig")
    llm_explanations.to_csv(output_root / "patient_llm_explanations.csv", index=False, encoding="utf-8-sig")

    bundle = joblib.load(code_dir / "ovcf_rf_model_bundle.joblib")
    log_event(output_root, "model_bundle_loaded", feature_count=len(bundle.get("feature_columns", [])), threshold=bundle.get("threshold"))
    x = prepare_model_matrix(feature_df, bundle["feature_columns"])
    model = bundle["model"]
    force_single_thread_estimator(model)
    probability = float(model.predict_proba(x)[0, 1])
    threshold = float(bundle.get("threshold", 0.5))
    pred = int(probability >= threshold)
    interp = build_interpretability_table(row, x, bundle, top_n=10)

    missing_modalities = []
    for action, prefix in ACTION_TO_PREFIX.items():
        cols = [c for c in SCORE_COLUMNS if c.startswith(prefix + "__")]
        if cols and all(is_missing_value(row.get(c, np.nan)) for c in cols):
            missing_modalities.append(action)
    log_event(output_root, "missing_modalities_computed", missing_modalities=missing_modalities)

    report = {
        "patient": case_id,
        "clinical_meta_used": {c: row.get(c) for c in DEMO_COLUMNS if not is_missing_value(row.get(c))},
        "clinical_meta_source": "web_form",
        "ovcf_probability": probability,
        "threshold": threshold,
        "prediction": "OVCF" if pred == 1 else "control/non-OVCF",
        "missing_modalities": missing_modalities,
        "model_training_summary": bundle.get("training_summary", {}),
        "top_explanatory_features": interp.to_dict(orient="records") if not interp.empty else [],
        "note": "Research model output; not a substitute for clinical diagnosis.",
    }
    (output_root / "prediction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    interp.to_csv(output_root / "prediction_explanation_table.csv", index=False, encoding="utf-8-sig")
    log_event(output_root, "prediction_done", probability=probability, threshold=threshold, prediction=report["prediction"], top_feature_count=len(report["top_explanatory_features"]))
    return {
        "ok": True,
        "case_id": case_id,
        "analysis_mode": "live_model",
        "report": report,
        "explanation_table": interp.to_dict(orient="records") if not interp.empty else [],
        "llm_explanations": llm_explanations.to_dict(orient="records") if not llm_explanations.empty else [],
        "structured": structured,
        "media": [{"role": k, "path": str(v), "name": v.name} for k, v in media_map.items()],
        "logs": LOG_ROWS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    result = analyze(payload)
    Path(args.result).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
