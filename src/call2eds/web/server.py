"""Serveur FastAPI + UI minimal pour uploader des audios, suivre les runs et télécharger les artefacts."""

import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import base64
import hashlib
import hmac
import secrets
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, urlunsplit
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
from fastapi import FastAPI, UploadFile, Form, Request, File, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from call2eds.config.settings import settings
from call2eds.pipeline.pipeline import (
    PIPELINE_VERSION,
    doctor,
    export_artifacts,
    ingest_call,
    ingest_call_with_run_id,
    prepare_ingest,
    list_runs,
    list_runs_recent,
)
from call2eds.db.session import get_session, init_db
from call2eds.db import models
from call2eds.storage.minio_client import get_minio

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError
except Exception:  # pragma: no cover - optional at runtime
    snapshot_download = None
    LocalEntryNotFoundError = Exception

try:
    from faster_whisper import utils as fw_utils
except Exception:  # pragma: no cover - optional at runtime
    fw_utils = None

ASYNC_INGEST = os.getenv("CALL2EDS_ASYNC_INGEST", "true").lower() in ("1", "true", "yes")
ASYNC_WORKERS = int(os.getenv("CALL2EDS_ASYNC_WORKERS", "2"))
ASYNC_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, ASYNC_WORKERS))


API_DESCRIPTION = """
Call2EDS HTTP API.

Auth
- Session cookie: `POST /api/auth/login` (sets `call2eds_session`)
- API key: `X-API-Key` or `Authorization: Bearer <key>`
- Basic auth (UI only): `Authorization: Basic base64(user:pass)`

Examples
```bash
# Login (cookie session)
curl -s -X POST http://HOST:8000/api/auth/login \\
  -H 'Content-Type: application/json' \\
  -d '{"username":"admin","password":"secret"}'

# Active runs
curl -s http://HOST:8000/api/runs/active

# Models cache status
curl -s http://HOST:8000/api/models

# Download a model
curl -s -X POST http://HOST:8000/api/models/download \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"small"}'

# Read config
curl -s http://HOST:8000/api/config

# Update config (optionally persist to .env.secrets)
curl -s -X POST http://HOST:8000/api/config \\
  -H 'Content-Type: application/json' \\
  -d '{"values":{"CALL2EDS_MODEL":"medium"},"persist":true}'
```

Notes
- If auth is enabled, admin rights are required for `/api/auth/*`.
- Inputs are validated server-side (never trust the input).
"""

OPENAPI_TAGS = [
    {"name": "auth", "description": "Users, sessions and API keys."},
    {"name": "runs", "description": "Runtime status and active runs."},
    {"name": "system", "description": "System resources, checks and restart."},
    {"name": "models", "description": "ASR model cache and downloads."},
    {"name": "config", "description": "Runtime configuration (env-backed)."},
]

app = FastAPI(
    title="Call2EDS API",
    version=PIPELINE_VERSION,
    description=API_DESCRIPTION,
    openapi_tags=OPENAPI_TAGS,
    docs_url=None,
    redoc_url=None,
)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=OPENAPI_TAGS,
    )
    components = schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes.update(
        {
            "basicAuth": {"type": "http", "scheme": "basic"},
            "apiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
            "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "API key"},
        }
    )
    schema["security"] = [{"apiKeyAuth": []}, {"bearerAuth": []}, {"basicAuth": []}]
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def _startup_init() -> None:
    init_db()


@app.get("/swagger", include_in_schema=False)
async def swagger_redirect():
    return RedirectResponse(url="/docs")


@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
async def swagger_page(request: Request):
    return templates.TemplateResponse(
        "swagger.html",
        {
            "request": request,
            "openapi_url": app.openapi_url,
        },
    )

AUTH_EXEMPT_PREFIXES = (
    "/static/",
    "/healthz",
    "/login",
    "/logout",
    "/setup",
    "/api/auth/login",
    "/api/auth/setup",
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not settings.auth_enabled:
        return await call_next(request)

    path = request.url.path
    if path.startswith(AUTH_EXEMPT_PREFIXES):
        return await call_next(request)

    if not _has_admin():
        if path.startswith("/setup") or path.startswith("/login"):
            return await call_next(request)
        return RedirectResponse(url="/setup")

    auth_header = request.headers.get(AUTH_HEADER)
    session_token = request.cookies.get(AUTH_COOKIE)
    user = None
    role = None
    set_cookie = None

    # Session cookie
    if session_token:
        payload = _unsign_session(session_token)
        if payload and int(payload.get("exp", 0)) >= int(datetime.utcnow().timestamp()):
            uid = payload.get("uid")
            role = payload.get("role")
            if isinstance(uid, int):
                user = _get_user_by_id(uid)
            elif isinstance(uid, str) and uid.isdigit():
                user = _get_user_by_id(int(uid))
            if user:
                request.state.user = user
                request.state.role = role or user.role
                return await call_next(request)

    # API key header
    api_key = request.headers.get(API_KEY_HEADER) or _parse_bearer(auth_header or "")
    if api_key:
        user = _authenticate_api_key(api_key)
        if user:
            request.state.user = user
            request.state.role = user.role
            return await call_next(request)

    # Basic auth
    basic = _parse_basic_auth(auth_header or "")
    if basic:
        with get_session() as session:
            db_user = session.query(models.User).filter(models.User.username == basic["username"]).first()
            if db_user and _verify_password(basic["password"], db_user.password_hash):
                db_user.last_login_at = datetime.utcnow()
                session.add(db_user)
                request.state.user = db_user
                request.state.role = db_user.role
                set_cookie = _session_cookie_for(db_user.id, db_user.role)
                _log_auth_event(db_user.id, "login_basic", request)
            else:
                _log_auth_event(None, "login_failed", request, {"username": basic["username"]})

    if not getattr(request.state, "user", None):
        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        accept = request.headers.get("accept", "")
        if "text/html" in accept or "*/*" in accept:
            return RedirectResponse(url="/login")
        return JSONResponse(status_code=401, content={"detail": "unauthorized"})

    response = await call_next(request)
    if set_cookie:
        response.set_cookie(AUTH_COOKIE, set_cookie, httponly=True, samesite="lax")
    return response

# Configuration éditable depuis l'UI
EDITABLE_PREFIXES = ("CALL2EDS_DIAR_", "CALL2EDS_ECAPA_")
MODEL_CHOICES = ("tiny", "small", "medium", "large")
LANG_CHOICES = ("auto", "fr", "en", "es", "de", "it", "pt", "ar", "ru", "zh")
EDITABLE_KEYS = {
    "CALL2EDS_MODEL": "Modèle ASR par défaut",
    "CALL2EDS_LANG": "Langue ASR par défaut",
    "CALL2EDS_DEVICE": "Device ASR (auto/cuda/cpu)",
    "CALL2EDS_COMPUTE_TYPE": "Compute type (int8_float32, float16, etc.)",
    "CALL2EDS_NO_SPEECH_THRESHOLD": "ASR no_speech_threshold",
    "CALL2EDS_LOGPROB_THRESHOLD": "ASR log_prob_threshold",
    "CALL2EDS_VAD_FILTER": "ASR vad_filter (true/false)",
}
HF_TOKEN_ENV = "HUGGINGFACE_HUB_TOKEN"
HF_TOKEN_ENV_FALLBACK = "HF_TOKEN"
CONFIG_PATH = Path(os.getenv("CALL2EDS_CONFIG_PATH", "/workspace/.env.secrets"))
MODEL_REPO_TEMPLATE = "Systran/faster-whisper-{model}"
MODEL_STATE: Dict[str, Dict[str, str]] = {}
MODEL_LOCK = threading.Lock()
AUTH_COOKIE = "call2eds_session"
AUTH_HEADER = "authorization"
API_KEY_HEADER = "x-api-key"
DEFAULT_ROLE = "admin"
VALID_ROLES = ("admin", "viewer")
AUTH_HASH_ITERATIONS = 120_000
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _hf_cache_dirs() -> List[Path]:
    candidates: List[Path] = []
    seen = set()

    def add(path: Optional[Path]) -> None:
        if not path:
            return
        p = Path(path)
        if p in seen:
            return
        seen.add(p)
        candidates.append(p)

    custom = os.getenv("HUGGINGFACE_HUB_CACHE")
    if custom:
        add(Path(custom))
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        add(Path(hf_home) / "hub")
    xdg = os.getenv("XDG_CACHE_HOME")
    if xdg:
        add(Path(xdg) / "huggingface" / "hub")
    add(Path.home() / ".cache" / "huggingface" / "hub")
    add(Path("/cache/hf"))
    add(Path("/root/.cache/huggingface/hub"))
    return candidates


def _model_repo_id(model: str) -> str:
    return MODEL_REPO_TEMPLATE.format(model=model)


def _model_repo_dirs(model: str) -> List[Path]:
    repo_id = _model_repo_id(model).replace("/", "--")
    return [cache_dir / f"models--{repo_id}" for cache_dir in _hf_cache_dirs()]


def _latest_snapshot_dir(repo_dir: Path) -> Optional[Path]:
    snap_dir = repo_dir / "snapshots"
    try:
        if not snap_dir.exists():
            return None
        snaps = sorted([p for p in snap_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        return snaps[0] if snaps else None
    except OSError:
        return None


def _dir_size_bytes(path: Path) -> Optional[int]:
    if not path or not path.exists():
        return None
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def _model_local_path(model: str) -> Optional[Path]:
    if fw_utils and hasattr(fw_utils, "download_model"):
        try:
            path = fw_utils.download_model(model, local_files_only=True)
            return Path(path)
        except Exception:
            pass
    for repo_dir in _model_repo_dirs(model):
        snap = _latest_snapshot_dir(repo_dir)
        if snap:
            return snap
    if snapshot_download:
        try:
            path = snapshot_download(_model_repo_id(model), local_files_only=True)
            return Path(path)
        except LocalEntryNotFoundError:
            return None
        except Exception:
            return None
    return None


def _model_cached(model: str) -> bool:
    return _model_local_path(model) is not None


def _model_cache_size(model: str) -> Optional[int]:
    path = _model_local_path(model)
    if path:
        return _dir_size_bytes(path)
    return None


def _set_model_state(model: str, state: str, error: Optional[str] = None) -> None:
    with MODEL_LOCK:
        MODEL_STATE[model] = {
            "state": state,
            "error": error or "",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }


def _download_model_task(model: str) -> None:
    _set_model_state(model, "downloading")
    try:
        if fw_utils and hasattr(fw_utils, "download_model"):
            fw_utils.download_model(model, local_files_only=False)
        elif snapshot_download:
            snapshot_download(_model_repo_id(model), resume_download=True)
        else:
            from faster_whisper import WhisperModel

            WhisperModel(model, device="cpu", compute_type="int8")
        _set_model_state(model, "installed")
    except Exception as exc:  # noqa: BLE001
        _set_model_state(model, "error", str(exc))


def _human_time(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


def _manifest_public_links(manifest: Dict) -> List[Dict]:
    base = settings.minio_public_endpoint.rstrip("/")
    bucket = settings.minio_bucket
    items = []
    for art in manifest.get("artifacts", []):
        key = art.get("key")
        if not key:
            continue
        url = f"{base}/{bucket}/{key}"
        items.append({"key": key, "size": art.get("size_bytes", 0), "sha256": art.get("sha256", ""), "url": url})
    return items


def _is_editable_key(key: str) -> bool:
    if key in EDITABLE_KEYS:
        return True
    return key.startswith(EDITABLE_PREFIXES)


def _list_editable_keys() -> List[str]:
    keys = set(EDITABLE_KEYS.keys())
    for k in os.environ.keys():
        if _is_editable_key(k):
            keys.add(k)
    return sorted(keys)


def _meta_for_key(key: str) -> Dict[str, object]:
    if key in EDITABLE_KEYS:
        if key == "CALL2EDS_MODEL":
            return {
                "label": EDITABLE_KEYS[key],
                "type": "select",
                "choices": list(MODEL_CHOICES),
                "placeholder": "tiny/small/medium/large",
            }
        if key == "CALL2EDS_LANG":
            return {
                "label": EDITABLE_KEYS[key],
                "type": "select",
                "choices": list(LANG_CHOICES),
                "placeholder": "auto/fr/en/...",
            }
        if key == "CALL2EDS_DEVICE":
            return {
                "label": EDITABLE_KEYS[key],
                "type": "text",
                "placeholder": "auto/cuda/cpu",
            }
        if key == "CALL2EDS_COMPUTE_TYPE":
            return {
                "label": EDITABLE_KEYS[key],
                "type": "text",
                "placeholder": "int8_float32 / float16 / int8",
            }
        if key == "CALL2EDS_NO_SPEECH_THRESHOLD":
            return {
                "label": EDITABLE_KEYS[key],
                "type": "number",
                "placeholder": "0.0–1.0",
                "hint": "0.0–1.0",
            }
        if key == "CALL2EDS_LOGPROB_THRESHOLD":
            return {
                "label": EDITABLE_KEYS[key],
                "type": "number",
                "placeholder": "ex: -1.0",
            }
        if key == "CALL2EDS_VAD_FILTER":
            return {
                "label": EDITABLE_KEYS[key],
                "type": "bool",
                "placeholder": "true/false",
            }
        return {"label": EDITABLE_KEYS[key], "type": "text"}

    # Heuristique pour les variables diarisation/ECAPA
    key_upper = key.upper()
    if key_upper.endswith(("_ENABLED", "_FORCE", "_STRICT", "_REFINE", "_DIVERSITY")):
        return {"type": "bool", "hint": "0/1 (bool)", "placeholder": "0/1"}
    if "_TOKENS" in key_upper:
        return {"type": "number", "hint": "integer", "placeholder": "int"}
    if any(x in key_upper for x in ("_THRESHOLD", "_CONF", "_OV", "_RATIO", "_DROP", "_CHANGE")):
        return {"type": "number", "hint": "0.0–1.0", "placeholder": "0.0–1.0"}
    if any(x in key_upper for x in ("_WIN_", "_WINDOW_", "_HOP_", "_GAP_", "_MIN_", "_MAX_", "_S", "_DUR")):
        return {"type": "number", "hint": "seconds", "placeholder": "seconds"}
    if key_upper.endswith("_METHOD"):
        return {"type": "text", "hint": "method name", "placeholder": "method name"}
    return {"type": "text"}


def _read_env_file(path: Path) -> Tuple[List[str], Dict[str, int]]:
    if not path.exists():
        return [], {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index: Dict[str, int] = {}
    for idx, line in enumerate(lines):
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key:
            index[key] = idx
    return lines, index


def _write_env_file(path: Path, updates: Dict[str, Optional[str]]) -> None:
    lines, index = _read_env_file(path)
    for key, value in updates.items():
        if value is None or value == "":
            if key in index:
                lines.pop(index[key])
                lines, index = _read_env_file(path)
            continue
        line = f"{key}={value}"
        if key in index:
            lines[index[key]] = line
        else:
            lines.append(line)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _env_has_key(path: Path, key: str) -> bool:
    lines, index = _read_env_file(path)
    return key in index


def _get_hf_token() -> str:
    return os.getenv(HF_TOKEN_ENV) or os.getenv(HF_TOKEN_ENV_FALLBACK, "")


def _auth_secret() -> str:
    if settings.auth_secret:
        return settings.auth_secret
    return os.getenv("CALL2EDS_AUTH_SECRET", "")


def _ensure_auth_secret() -> str:
    secret = _auth_secret()
    if secret:
        return secret
    secret = secrets.token_urlsafe(32)
    try:
        _write_env_file(CONFIG_PATH, {"CALL2EDS_AUTH_SECRET": secret})
        os.environ["CALL2EDS_AUTH_SECRET"] = secret
        settings.auth_secret = secret
    except Exception:
        os.environ["CALL2EDS_AUTH_SECRET"] = secret
        settings.auth_secret = secret
    return secret


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, AUTH_HASH_ITERATIONS)
    return f"pbkdf2_sha256${AUTH_HASH_ITERATIONS}${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(dk).decode()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters_str, salt_b64, hash_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iters = int(iters_str)
        salt = base64.urlsafe_b64decode(salt_b64.encode())
        expected = base64.urlsafe_b64decode(hash_b64.encode())
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def _sign_session(payload: Dict[str, object]) -> str:
    secret = _ensure_auth_secret().encode("utf-8")
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(secret, raw, hashlib.sha256).digest()
    return f"{base64.urlsafe_b64encode(raw).decode()}.{base64.urlsafe_b64encode(sig).decode()}"


def _unsign_session(token: str) -> Optional[Dict[str, object]]:
    try:
        raw_b64, sig_b64 = token.split(".", 1)
        raw = base64.urlsafe_b64decode(raw_b64.encode())
        sig = base64.urlsafe_b64decode(sig_b64.encode())
        secret = _ensure_auth_secret().encode("utf-8")
        expected = hmac.new(secret, raw, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(raw.decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _session_cookie_for(user_id: int, role: str) -> str:
    exp = int((datetime.utcnow() + timedelta(seconds=settings.auth_ttl_s)).timestamp())
    return _sign_session({"uid": user_id, "role": role, "exp": exp})


def _get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for") or ""
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host
    return ""


def _log_auth_event(user_id: Optional[int], event: str, request: Request, details: Optional[Dict] = None) -> None:
    try:
        with get_session() as session:
            session.add(
                models.AuthEvent(
                    user_id=user_id,
                    event=event,
                    ip=_get_client_ip(request),
                    user_agent=request.headers.get("user-agent"),
                    details=details or {},
                )
            )
    except Exception:
        pass


def _get_user_by_id(user_id: int) -> Optional[models.User]:
    with get_session() as session:
        return session.get(models.User, user_id)


def _get_user_by_username(username: str) -> Optional[models.User]:
    with get_session() as session:
        return session.query(models.User).filter(models.User.username == username).first()


def _has_users() -> bool:
    with get_session() as session:
        return session.query(models.User.id).first() is not None


def _admin_count() -> int:
    with get_session() as session:
        return session.query(models.User.id).filter(models.User.role == "admin").count()


def _has_admin() -> bool:
    return _admin_count() > 0


def _public_user(user: models.User) -> Dict[str, object]:
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "can_api": bool(user.can_api),
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
    }


def _parse_basic_auth(header_val: str) -> Optional[Dict[str, str]]:
    if not header_val or not header_val.lower().startswith("basic "):
        return None
    try:
        raw = base64.b64decode(header_val.split(" ", 1)[1]).decode("utf-8")
        if ":" not in raw:
            return None
        username, password = raw.split(":", 1)
        return {"username": username, "password": password}
    except Exception:
        return None


def _parse_bearer(header_val: str) -> Optional[str]:
    if not header_val or not header_val.lower().startswith("bearer "):
        return None
    return header_val.split(" ", 1)[1].strip()


def _authenticate_api_key(key: str) -> Optional[models.User]:
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    with get_session() as session:
        api_key = (
            session.query(models.ApiKey)
            .filter(models.ApiKey.key_hash == key_hash, models.ApiKey.revoked_at.is_(None))
            .first()
        )
        if not api_key:
            return None
        user = session.get(models.User, api_key.user_id)
        if not user or not user.can_api:
            return None
        api_key.last_used_at = datetime.utcnow()
        return user


def _require_admin(request: Request) -> None:
    if not settings.auth_enabled:
        return
    user = getattr(request.state, "user", None)
    if not user or getattr(user, "role", "") != "admin":
        raise HTTPException(status_code=403, detail="admin required")


def _normalize_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _safe_id(value: Optional[str], label: str) -> str:
    if value is None:
        raise HTTPException(status_code=400, detail=f"{label} required")
    value = str(value).strip()
    if not value or not SAFE_ID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"invalid {label}")
    return value


def _start_background_ingest(
    audio_path: str,
    call_id: Optional[str],
    language: str,
    model_size: str,
    user_timestamp: Optional[str] = None,
) -> Tuple[str, str]:
    run_id, call_id_final, params_json = prepare_ingest(
        audio_path=audio_path,
        call_id=call_id,
        language=language,
        model_size=model_size,
        user_timestamp=user_timestamp,
        status="queued",
    )

    def _task():
        try:
            ingest_call_with_run_id(
                run_id,
                call_id_final,
                audio_path,
                language,
                model_size,
                user_timestamp=user_timestamp,
                params_json=params_json,
            )
        except Exception:
            # errors are tracked in DB by ingest_call_with_run_id
            return

    ASYNC_EXECUTOR.submit(_task)
    return run_id, call_id_final


def _get_disk_usage(path: Path) -> Dict:
    try:
        usage = shutil.disk_usage(path)
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        free_gb = usage.free / (1024**3)
        pct = (used_gb / total_gb * 100) if total_gb else 0.0
        return {
            "path": str(path),
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "used_pct": round(pct, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "error": str(exc)}


def _get_meminfo() -> Dict:
    mem = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                mem[key.strip()] = val.strip()
        total_kb = int(mem.get("MemTotal", "0 kB").split()[0])
        avail_kb = int(mem.get("MemAvailable", "0 kB").split()[0])
        used_kb = max(0, total_kb - avail_kb)
        pct = (used_kb / total_kb * 100) if total_kb else 0.0
        return {
            "total_gb": round(total_kb / (1024**2), 2),
            "available_gb": round(avail_kb / (1024**2), 2),
            "used_gb": round(used_kb / (1024**2), 2),
            "used_pct": round(pct, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _get_cpuinfo() -> Dict:
    info = {"count": os.cpu_count() or 0}
    try:
        load1, load5, load15 = os.getloadavg()
        info.update({"load1": round(load1, 2), "load5": round(load5, 2), "load15": round(load15, 2)})
    except Exception:
        pass
    return info


def _get_gpuinfo() -> Dict:
    info: Dict = {"available": False, "gpus": [], "driver": None}
    info["cuda_visible_devices"] = os.getenv("CUDA_VISIBLE_DEVICES")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["available"] = True
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    info["gpus"].append(
                        {
                            "name": parts[0],
                            "memory_total_mb": int(float(parts[1])),
                            "memory_used_mb": int(float(parts[2])),
                            "util_gpu_pct": int(float(parts[3])),
                        }
                    )
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    if not info["available"]:
        nvidia_path = Path("/proc/driver/nvidia/gpus")
        if nvidia_path.exists():
            info["available"] = True
            info["gpus"] = [{"name": "nvidia (details indisponibles)"}]
    return info


def _ffmpeg_bin() -> str:
    from shutil import which
    import imageio_ffmpeg

    sys_ff = which("ffmpeg")
    return sys_ff or imageio_ffmpeg.get_ffmpeg_exe()


def _guess_console_url(public_endpoint: str) -> str:
    try:
        parts = urlsplit(public_endpoint)
        port = parts.port
        if port is None:
            return public_endpoint
        host = parts.hostname or parts.netloc
        if host is None:
            return public_endpoint
        new_netloc = f"{host}:{port + 1}"
        return urlunsplit((parts.scheme, new_netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return public_endpoint


def _minio_console_url() -> str:
    console = os.getenv("MINIO_CONSOLE_PUBLIC_ENDPOINT")
    if console:
        return console.rstrip("/")
    return _guess_console_url(settings.minio_public_endpoint.rstrip("/"))


def _get_normalized_audio(call_id: str, run_id: str) -> Path:
    key = f"calls/{call_id}/runs/{run_id}/audio/normalized.flac"
    cache_dir = Path("/tmp/call2eds_audio") / call_id / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / "normalized.flac"
    if not local_path.exists():
        minio = get_minio()
        minio.download_file(key, local_path)
    return local_path


def _make_snippet(call_id: str, run_id: str, t0: float, t1: float) -> Path:
    if t1 <= t0:
        raise HTTPException(status_code=400, detail="invalid time range")
    max_dur = 120.0
    if (t1 - t0) > max_dur:
        t1 = t0 + max_dur
    cache_dir = Path("/tmp/call2eds_audio") / call_id / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_ms = int(max(0.0, t0) * 1000)
    end_ms = int(max(0.0, t1) * 1000)
    out_path = cache_dir / f"seg_{start_ms}_{end_ms}.mp3"
    if out_path.exists():
        return out_path
    src = _get_normalized_audio(call_id, run_id)
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-ss",
        f"{max(0.0, t0):.3f}",
        "-to",
        f"{max(0.0, t1):.3f}",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-vn",
        "-f",
        "mp3",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def _download_manifest(call_id: str, run_id: str) -> Path:
    key = f"calls/{call_id}/runs/{run_id}/manifest.json"
    tmp = Path("/tmp") / f"manifest_{run_id}.json"
    minio = get_minio()
    minio.download_file(key, tmp)
    return tmp


def _load_parquet_from_minio(key: str) -> pd.DataFrame:
    tmp = Path("/tmp") / key.replace("/", "_")
    minio = get_minio()
    minio.download_file(key, tmp)
    return pd.read_parquet(tmp)


def _metric_latest(session, run_id: str, key: str):
    return (
        session.query(models.Metric)
        .filter(models.Metric.run_id == run_id, models.Metric.key == key)
        .order_by(models.Metric.id.desc())
        .first()
    )


def _set_metric_value(session, run_id: str, key: str, value_num: Optional[float] = None, value_json: Optional[Dict] = None):
    session.query(models.Metric).filter(
        models.Metric.run_id == run_id, models.Metric.key == key
    ).delete()
    session.add(models.Metric(run_id=run_id, key=key, value_num=value_num, value_json=value_json))


def _arousal_label(row: Dict) -> Dict:
    # Heuristique simple : combinaison f0, énergie, voisement
    f0 = float(row.get("f0_mean", 0.0) or 0.0)
    energy_db = float(row.get("energy_mean", 0.0) or 0.0)
    voiced = float(row.get("voiced_ratio", 0.0) or 0.0)
    f0_norm = max(0.0, min(1.0, (f0 - 50.0) / 250.0))
    energy_norm = max(0.0, min(1.0, (energy_db + 80.0) / 80.0))
    score = 0.5 * f0_norm + 0.4 * energy_norm + 0.1 * voiced
    if score < 0.35:
        return {"label": "calme", "color": "is-success", "score": score}
    if score > 0.65:
        return {"label": "énergique", "color": "is-danger", "score": score}
    return {"label": "neutre", "color": "is-warning", "score": score}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if not settings.auth_enabled:
        return RedirectResponse(url="/")
    if getattr(request.state, "user", None):
        return RedirectResponse(url="/")
    if not _has_admin():
        return RedirectResponse(url="/setup")
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "auth_enabled": settings.auth_enabled,
        },
    )


@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    if _has_admin():
        return RedirectResponse(url="/login" if settings.auth_enabled else "/")
    return templates.TemplateResponse(
        "setup.html",
        {
            "request": request,
            "auth_enabled": settings.auth_enabled,
        },
    )


@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/login" if settings.auth_enabled else "/")
    response.delete_cookie(AUTH_COOKIE)
    user = getattr(request.state, "user", None)
    if user:
        _log_auth_event(user.id, "logout", request)
    return response


@app.get("/access", response_class=HTMLResponse)
async def access_page(request: Request):
    _require_admin(request)
    with get_session() as session:
        users_count = session.query(models.User.id).count()
        keys_count = session.query(models.ApiKey.id).filter(models.ApiKey.revoked_at.is_(None)).count()
    return templates.TemplateResponse(
        "access.html",
        {
            "request": request,
            "auth_enabled": settings.auth_enabled,
            "auth_ttl_s": settings.auth_ttl_s,
            "users_count": users_count,
            "keys_count": keys_count,
        },
    )


@app.post(
    "/api/auth/enable",
    tags=["auth"],
    summary="Enable authentication",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "ok": True,
                        "auth_enabled": True,
                        "persisted": True,
                        "needs_setup": True,
                        "redirect": "/setup",
                    }
                }
            }
        }
    },
)
async def auth_enable(request: Request):
    if settings.auth_enabled:
        _require_admin(request)
    payload = await request.json()
    persist = bool((payload or {}).get("persist", True))
    os.environ["CALL2EDS_AUTH_ENABLED"] = "true"
    settings.auth_enabled = True
    _ensure_auth_secret()
    if persist:
        if not CONFIG_PATH.parent.exists():
            raise HTTPException(status_code=500, detail="persist path unavailable")
        _write_env_file(CONFIG_PATH, {"CALL2EDS_AUTH_ENABLED": "true"})
    needs_setup = not _has_admin()
    _log_auth_event(None, "auth_enabled", request, {"persist": persist})
    return {
        "ok": True,
        "auth_enabled": True,
        "persisted": persist,
        "needs_setup": needs_setup,
        "redirect": "/setup" if needs_setup else "/login",
    }


@app.post(
    "/api/auth/setup",
    tags=["auth"],
    summary="Create first admin account",
    responses={
        200: {"content": {"application/json": {"example": {"ok": True}}}},
        409: {"content": {"application/json": {"example": {"detail": "admin already exists"}}}},
    },
)
async def auth_setup(request: Request):
    payload = await request.json()
    username = str((payload or {}).get("username", "")).strip()
    password = str((payload or {}).get("password", "")).strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")
    if _has_admin():
        raise HTTPException(status_code=409, detail="admin already exists")
    user = models.User(
        username=username,
        password_hash=_hash_password(password),
        role="admin",
        can_api=True,
    )
    with get_session() as session:
        session.add(user)
    _ensure_auth_secret()
    _log_auth_event(user.id, "setup", request)
    return {"ok": True}


@app.post(
    "/api/auth/login",
    tags=["auth"],
    summary="Login and receive session cookie",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {"username": "admin", "password": "secret"}
                }
            }
        }
    },
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "ok": True,
                        "user": {"id": 1, "username": "admin", "role": "admin", "can_api": True},
                    }
                }
            }
        },
        401: {"content": {"application/json": {"example": {"detail": "invalid credentials"}}}},
    },
)
async def auth_login(request: Request):
    payload = await request.json()
    username = str((payload or {}).get("username", "")).strip()
    password = str((payload or {}).get("password", "")).strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")
    with get_session() as session:
        user = session.query(models.User).filter(models.User.username == username).first()
        if not user or not _verify_password(password, user.password_hash):
            _log_auth_event(None, "login_failed", request, {"username": username})
            raise HTTPException(status_code=401, detail="invalid credentials")
        user.last_login_at = datetime.utcnow()
        session.add(user)
        _log_auth_event(user.id, "login_form", request)
        cookie = _session_cookie_for(user.id, user.role)
    response = JSONResponse({"ok": True, "user": _public_user(user)})
    response.set_cookie(AUTH_COOKIE, cookie, httponly=True, samesite="lax")
    return response


@app.get(
    "/api/auth/users",
    tags=["auth"],
    summary="List users",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "items": [
                            {
                                "id": 1,
                                "username": "admin",
                                "role": "admin",
                                "can_api": True,
                                "created_at": "2025-12-22T12:00:00Z",
                                "last_login_at": "2025-12-22T12:05:00Z",
                                "api_keys": 2,
                            }
                        ]
                    }
                }
            }
        }
    },
)
async def auth_users(request: Request):
    _require_admin(request)
    with get_session() as session:
        users = session.query(models.User).order_by(models.User.created_at.asc()).all()
        items = []
        for user in users:
            item = _public_user(user)
            item["api_keys"] = len([k for k in user.api_keys if k.revoked_at is None])
            items.append(item)
    return {"items": items}


@app.post(
    "/api/auth/users",
    tags=["auth"],
    summary="Create user",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {"username": "viewer", "password": "changeme", "role": "viewer", "can_api": True}
                }
            }
        }
    },
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "ok": True,
                        "user": {"id": 2, "username": "viewer", "role": "viewer", "can_api": False},
                    }
                }
            }
        }
    },
)
async def auth_users_create(request: Request):
    _require_admin(request)
    payload = await request.json()
    username = str((payload or {}).get("username", "")).strip()
    password = str((payload or {}).get("password", "")).strip()
    role = str((payload or {}).get("role", "admin")).strip() or "admin"
    if role not in VALID_ROLES:
        role = DEFAULT_ROLE
    can_api = bool((payload or {}).get("can_api", True))
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")
    with get_session() as session:
        existing = session.query(models.User).filter(models.User.username == username).first()
        if existing:
            raise HTTPException(status_code=409, detail="username exists")
        user = models.User(
            username=username,
            password_hash=_hash_password(password),
            role=role,
            can_api=can_api,
        )
        session.add(user)
        session.flush()
        _log_auth_event(user.id, "user_create", request, {"username": username})
        return {"ok": True, "user": _public_user(user)}


@app.patch(
    "/api/auth/users/{user_id}",
    tags=["auth"],
    summary="Update user",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {"role": "viewer", "can_api": False}
                }
            }
        }
    },
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "ok": True,
                        "user": {"id": 2, "username": "viewer", "role": "viewer", "can_api": True},
                    }
                }
            }
        }
    },
)
async def auth_users_update(user_id: int, request: Request):
    _require_admin(request)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    with get_session() as session:
        user = session.get(models.User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="user not found")
        role = (payload or {}).get("role")
        if role is not None:
            role = str(role).strip() or user.role
            if role not in VALID_ROLES:
                raise HTTPException(status_code=400, detail="invalid role")
            if user.role == "admin" and role != "admin" and _admin_count() <= 1:
                raise HTTPException(status_code=400, detail="cannot remove last admin")
            user.role = role
        if "can_api" in (payload or {}):
            user.can_api = bool(payload.get("can_api"))
        if payload.get("password"):
            user.password_hash = _hash_password(str(payload.get("password")).strip())
        session.add(user)
        _log_auth_event(user.id, "user_update", request)
        return {"ok": True, "user": _public_user(user)}


@app.delete(
    "/api/auth/users/{user_id}",
    tags=["auth"],
    summary="Delete user",
    responses={200: {"content": {"application/json": {"example": {"ok": True}}}}},
)
async def auth_users_delete(user_id: int, request: Request):
    _require_admin(request)
    current = getattr(request.state, "user", None)
    if current and getattr(current, "id", None) == user_id:
        raise HTTPException(status_code=400, detail="cannot delete current user")
    with get_session() as session:
        user = session.get(models.User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="user not found")
        if user.role == "admin" and _admin_count() <= 1:
            raise HTTPException(status_code=400, detail="cannot delete last admin")
        # Remove dependent rows to avoid FK violations.
        session.query(models.ApiKey).filter(models.ApiKey.user_id == user_id).delete(synchronize_session=False)
        session.query(models.AuthEvent).filter(models.AuthEvent.user_id == user_id).delete(synchronize_session=False)
        session.delete(user)
        _log_auth_event(None, "user_delete", request, {"user_id": user_id})
    return {"ok": True}


@app.get(
    "/api/auth/keys",
    tags=["auth"],
    summary="List API keys",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "items": [
                            {
                                "id": 1,
                                "user_id": 1,
                                "username": "admin",
                                "prefix": "k_1234",
                                "name": "automation",
                                "created_at": "2025-12-22T12:00:00Z",
                                "last_used_at": None,
                                "revoked_at": None,
                            }
                        ]
                    }
                }
            }
        }
    },
)
async def auth_keys(request: Request):
    _require_admin(request)
    with get_session() as session:
        rows = (
            session.query(models.ApiKey, models.User.username)
            .join(models.User, models.ApiKey.user_id == models.User.id)
            .order_by(models.ApiKey.created_at.desc())
            .all()
        )
        items = []
        for key, username in rows:
            items.append(
                {
                    "id": key.id,
                    "user_id": key.user_id,
                    "username": username,
                    "name": key.name,
                    "prefix": key.prefix,
                    "created_at": key.created_at.isoformat() if key.created_at else None,
                    "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
                    "revoked_at": key.revoked_at.isoformat() if key.revoked_at else None,
                }
            )
    return {"items": items}


@app.post(
    "/api/auth/keys",
    tags=["auth"],
    summary="Create API key",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {"user_id": 1, "name": "automation"}
                }
            }
        }
    },
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "ok": True,
                        "key": "k_live_xxxxxxxxxxxxx",
                        "item": {"id": 1, "prefix": "k_live", "name": "automation"},
                    }
                }
            }
        }
    },
)
async def auth_keys_create(request: Request):
    _require_admin(request)
    payload = await request.json()
    user_id = int((payload or {}).get("user_id", 0) or 0)
    name = str((payload or {}).get("name", "")).strip() or None
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    raw_key = f"ck_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    prefix = raw_key[:8]
    with get_session() as session:
        user = session.get(models.User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="user not found")
        if not user.can_api:
            raise HTTPException(status_code=400, detail="user cannot access api")
        api_key = models.ApiKey(user_id=user_id, name=name, key_hash=key_hash, prefix=prefix)
        session.add(api_key)
        session.flush()
        _log_auth_event(user.id, "api_key_create", request, {"prefix": prefix})
        return {
            "ok": True,
            "key": raw_key,
            "item": {
                "id": api_key.id,
                "user_id": api_key.user_id,
                "name": api_key.name,
                "prefix": api_key.prefix,
                "created_at": api_key.created_at.isoformat() if api_key.created_at else None,
            },
        }


@app.post("/api/auth/keys/{key_id}/revoke", tags=["auth"], summary="Revoke API key")
async def auth_keys_revoke(key_id: int, request: Request):
    _require_admin(request)
    with get_session() as session:
        api_key = session.get(models.ApiKey, key_id)
        if not api_key:
            raise HTTPException(status_code=404, detail="key not found")
        api_key.revoked_at = datetime.utcnow()
        session.add(api_key)
        _log_auth_event(api_key.user_id, "api_key_revoke", request, {"prefix": api_key.prefix})
    return {"ok": True}


@app.get("/api/auth/events", tags=["auth"], summary="List auth events")
async def auth_events(request: Request, limit: int = 50):
    _require_admin(request)
    with get_session() as session:
        rows = (
            session.query(models.AuthEvent, models.User.username)
            .outerjoin(models.User, models.AuthEvent.user_id == models.User.id)
            .order_by(models.AuthEvent.created_at.desc())
            .limit(limit)
            .all()
        )
        items = []
        for event, username in rows:
            items.append(
                {
                    "id": event.id,
                    "user_id": event.user_id,
                    "username": username,
                    "event": event.event,
                    "ip": event.ip,
                    "user_agent": event.user_agent,
                    "details": event.details,
                    "created_at": event.created_at.isoformat() if event.created_at else None,
                }
            )
    return {"items": items}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, call_id: Optional[str] = None):
    if call_id:
        call_id = _safe_id(call_id, "call_id")
    runs = list_runs(call_id) if call_id else []
    for r in runs:
        r["created_at_h"] = _human_time(r["created_at"])
    try:
        history_limit = int(os.getenv("CALL2EDS_WEB_HISTORY_LIMIT", "200"))
    except Exception:
        history_limit = 200
    recent_runs = list_runs_recent(limit=history_limit)
    for r in recent_runs:
        r["created_at_h"] = _human_time(r["created_at"])
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "user": getattr(request.state, "user", None),
            "call_id": call_id,
            "runs": runs,
            "recent_runs": recent_runs,
            "minio_console_url": _minio_console_url(),
            "minio_bucket": settings.minio_bucket,
            "pipeline_version": PIPELINE_VERSION,
            "history_limit": history_limit,
            "default_lang": settings.call2eds_lang,
            "default_model": settings.call2eds_model,
        },
    )


@app.get("/api/runs/active", tags=["runs"], summary="List active runs")
async def runs_active(limit: int = 20):
    init_db()
    with get_session() as session:
        runs = (
            session.query(models.Run)
            .filter(models.Run.status != "completed")
            .order_by(models.Run.created_at.desc())
            .limit(limit)
            .all()
        )
        items = []
        for r in runs:
            pct = None
            stage = None
            m_pct = _metric_latest(session, r.run_id, "progress_pct")
            m_stage = _metric_latest(session, r.run_id, "progress_stage")
            if m_pct and m_pct.value_num is not None:
                pct = float(m_pct.value_num)
            if m_stage and isinstance(m_stage.value_json, dict):
                stage = m_stage.value_json.get("stage")
            items.append(
                {
                    "run_id": r.run_id,
                    "call_id": r.call_id,
                    "status": r.status,
                    "stage": stage,
                    "progress": pct,
                    "created_at": r.created_at.isoformat(),
                }
            )
    return {"status": "ok", "items": items}


@app.post("/api/runs/{run_id}/cancel", tags=["runs"], summary="Cancel a run")
async def runs_cancel(run_id: str, request: Request):
    run_id = _safe_id(run_id, "run_id")
    if settings.auth_enabled:
        _require_admin(request)
    init_db()
    with get_session() as session:
        run = session.get(models.Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")
        if run.status in ("completed", "failed", "canceled"):
            return {"ok": True, "status": run.status}
        run.status = "canceling"
        _set_metric_value(
            session,
            run_id,
            "cancel_requested",
            value_num=1.0,
            value_json={
                "requested": True,
                "requested_at": datetime.utcnow().isoformat(),
                "by": getattr(getattr(request, "state", None), "user", None).username
                if getattr(request.state, "user", None)
                else None,
            },
        )
        _set_metric_value(session, run_id, "progress_stage", value_json={"stage": "cancel"})
    return {"ok": True, "status": "canceling"}


@app.get("/run/{call_id}/{run_id}", response_class=HTMLResponse)
async def run_detail(call_id: str, run_id: str, request: Request):
    call_id = _safe_id(call_id, "call_id")
    run_id = _safe_id(run_id, "run_id")
    manifest_path = _download_manifest(call_id, run_id)
    manifest = json.loads(manifest_path.read_text())
    artifacts = _manifest_public_links(manifest)
    timeline: List[Dict] = []
    summary = {
        "turns": 0,
        "speakers": [],
        "duration_s": 0.0,
        "speaker_stats": {},
    }
    try:
        turns_df = _load_parquet_from_minio(f"calls/{call_id}/runs/{run_id}/eds/turns.parquet")
        prosody_turns_df = _load_parquet_from_minio(f"calls/{call_id}/runs/{run_id}/eds/prosody_turns.parquet")
        merged = turns_df.merge(prosody_turns_df, on="turn_id", how="left", suffixes=("", "_p"))
        merged = merged.sort_values("t_start")
        for _, row in merged.iterrows():
            ar = _arousal_label(row)
            t_start = float(row.get("t_start", 0.0))
            t_end = float(row.get("t_end", 0.0))
            pad = 0.15
            clip_start = max(0.0, t_start - pad)
            clip_end = t_end + pad
            clip_duration_s = max(0.0, clip_end - clip_start)
            speaker_id = int(row.get("speaker_id", 0) or 0)
            overlap_ratio = float(row.get("overlap_ratio", 0.0) or 0.0)
            timeline.append(
                {
                    "turn_id": int(row.get("turn_id", -1)),
                    "speaker_id": speaker_id,
                    "speaker_label": f"spk {speaker_id}" if speaker_id >= 0 else "overlap",
                    "t_start": t_start,
                    "t_end": t_end,
                    "text": row.get("text", ""),
                    "asr_conf_turn": float(row.get("asr_conf_turn", 0.0)),
                    "f0_mean": float(row.get("f0_mean", 0.0)),
                    "energy_mean": float(row.get("energy_mean", 0.0)),
                    "voiced_ratio": float(row.get("voiced_ratio", 0.0)),
                    "speech_rate_wps": float(row.get("speech_rate_wps", 0.0)),
                    "arousal": ar,
                    "overlap_ratio": overlap_ratio,
                    "audio_url": f"/audio/{call_id}/{run_id}?t0={clip_start:.2f}&t1={clip_end:.2f}",
                    "clip_duration_s": clip_duration_s,
                }
            )
        if timeline:
            summary["turns"] = len(timeline)
            summary["duration_s"] = max(t.get("t_end", 0.0) for t in timeline)
            speakers = sorted({int(t["speaker_id"]) for t in timeline if int(t["speaker_id"]) >= 0})
            summary["speakers"] = speakers
            speaker_stats: Dict[int, Dict[str, float]] = {}
            for t in timeline:
                spk = int(t["speaker_id"])
                if spk < 0:
                    continue
                speaker_stats.setdefault(spk, {"turns": 0, "duration_s": 0.0})
                speaker_stats[spk]["turns"] += 1
                speaker_stats[spk]["duration_s"] += float(t.get("t_end", 0.0)) - float(t.get("t_start", 0.0))
            summary["speaker_stats"] = speaker_stats
    except Exception:
        timeline = []
    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "user": getattr(request.state, "user", None),
            "call_id": call_id,
            "run_id": run_id,
            "manifest": manifest,
            "artifacts": artifacts,
            "timeline": timeline,
            "summary": summary,
            "minio_console_url": _minio_console_url(),
            "pipeline_version": PIPELINE_VERSION,
        },
    )


@app.post("/ingest")
async def ingest(
    request: Request,
    audio: UploadFile = File(...),
    call_id: Optional[str] = Form(None),
    lang: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    when: Optional[str] = Form(None),
):
    tmp_path = Path("/tmp") / Path(audio.filename).name
    with tmp_path.open("wb") as f:
        f.write(await audio.read())
    safe_call_id = _safe_id(call_id, "call_id") if call_id else None
    if ASYNC_INGEST:
        _start_background_ingest(
            audio_path=str(tmp_path),
            call_id=safe_call_id,
            language=lang or settings.call2eds_lang,
            model_size=model or settings.call2eds_model,
            user_timestamp=when,
        )
    else:
        _ = ingest_call(
            audio_path=str(tmp_path),
            call_id=safe_call_id,
            language=lang or settings.call2eds_lang,
            model_size=model or settings.call2eds_model,
            user_timestamp=when,
        )
    return RedirectResponse(url="/" + (f"?call_id={safe_call_id}" if safe_call_id else ""), status_code=303)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}


@app.get("/diagnostic", response_class=HTMLResponse)
async def diagnostic(request: Request, format: Optional[str] = None):
    try:
        report = doctor()
    except Exception as exc:  # noqa: BLE001
        report = {"error": str(exc)}
    diar_env = {
        k: v
        for k, v in os.environ.items()
        if k.startswith("CALL2EDS_DIAR_") or k.startswith("CALL2EDS_ECAPA_")
    }
    with get_session() as session:
        users_count = session.query(models.User.id).count()
        keys_count = session.query(models.ApiKey.id).filter(models.ApiKey.revoked_at.is_(None)).count()
    diag = {
        "status": "ok",
        "pipeline_version": PIPELINE_VERSION,
        "minio_endpoint": settings.minio_endpoint,
        "minio_public_endpoint": settings.minio_public_endpoint,
        "minio_console_public_endpoint": _minio_console_url(),
        "minio_bucket": settings.minio_bucket,
        "auth": {
            "enabled": settings.auth_enabled,
            "ttl_s": settings.auth_ttl_s,
            "users": users_count,
            "api_keys": keys_count,
        },
        "report": report,
        "diarization_env": diar_env,
    }
    if format == "json":
        return diag
    accept = request.headers.get("accept", "")
    if "text/html" in accept or "*/*" in accept:
        return templates.TemplateResponse(
            "diagnostic.html",
            {"request": request, "diag": diag, "user": getattr(request.state, "user", None)},
        )
    return diag


@app.get("/api/system", tags=["system"], summary="System status and checks")
async def system_status():
    disk_path = Path("/workspace") if Path("/workspace").exists() else Path("/")
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cpu": _get_cpuinfo(),
        "ram": _get_meminfo(),
        "disk": _get_disk_usage(disk_path),
        "gpu": _get_gpuinfo(),
    }


@app.get(
    "/api/models",
    tags=["models"],
    summary="ASR model cache status",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "items": [
                            {"name": "tiny", "state": "installed", "size_bytes": 75231234},
                            {"name": "small", "state": "missing", "size_bytes": None},
                        ]
                    }
                }
            }
        }
    },
)
async def models_status(request: Request):
    _require_admin(request)
    items = []
    for model in MODEL_CHOICES:
        cached = _model_cached(model)
        size_bytes = _model_cache_size(model) if cached else None
        with MODEL_LOCK:
            state_info = MODEL_STATE.get(model, {})
        state = "installed" if cached else "missing"
        if state_info.get("state") == "downloading":
            state = "downloading"
        elif state_info.get("state") == "error" and not cached:
            state = "error"
        items.append(
            {
                "name": model,
                "state": state,
                "size_bytes": size_bytes,
                "updated_at": state_info.get("updated_at"),
                "error": state_info.get("error", ""),
            }
        )
    return {"items": items}


@app.post(
    "/api/models/download",
    tags=["models"],
    summary="Download ASR model",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {"ok": True, "state": "downloading"}
                }
            }
        }
    },
)
async def models_download(request: Request):
    _require_admin(request)
    payload = await request.json()
    model = (payload or {}).get("model")
    if model not in MODEL_CHOICES:
        raise HTTPException(status_code=400, detail="invalid model")
    if _model_cached(model):
        _set_model_state(model, "installed")
        return {"ok": True, "state": "installed"}
    with MODEL_LOCK:
        current = MODEL_STATE.get(model, {}).get("state")
        if current == "downloading":
            return {"ok": True, "state": "downloading"}
        MODEL_STATE[model] = {"state": "downloading", "error": "", "updated_at": datetime.utcnow().isoformat() + "Z"}
    thread = threading.Thread(target=_download_model_task, args=(model,), daemon=True)
    thread.start()
    return {"ok": True, "state": "downloading"}


@app.post("/api/system/restart", tags=["system"], summary="Restart application")
async def system_restart(request: Request):
    _require_admin(request)
    sock = Path("/var/run/docker.sock")
    if not sock.exists():
        raise HTTPException(status_code=501, detail="docker socket not available")
    last_err = None
    for cmd in (["docker", "compose", "restart", "app"], ["docker-compose", "restart", "app"]):
        try:
            result = subprocess.run(
                cmd,
                cwd=Path("/workspace"),
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if result.returncode == 0:
                return {"ok": True, "stdout": result.stdout, "stderr": result.stderr}
            last_err = result.stderr or result.stdout
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
    raise HTTPException(status_code=500, detail=f"restart failed: {last_err}")


@app.get("/api/config", tags=["config"], summary="Get runtime config")
async def config_get(request: Request):
    _require_admin(request)
    keys = _list_editable_keys()
    values = {k: os.getenv(k, "") for k in keys}
    if not values.get("CALL2EDS_MODEL"):
        values["CALL2EDS_MODEL"] = settings.call2eds_model
    if not values.get("CALL2EDS_LANG"):
        values["CALL2EDS_LANG"] = settings.call2eds_lang
    if not values.get("CALL2EDS_DEVICE"):
        values["CALL2EDS_DEVICE"] = "auto"
    meta = {key: _meta_for_key(key) for key in keys}
    return {
        "keys": keys,
        "values": values,
        "meta": meta,
        "default_model": settings.call2eds_model,
        "default_lang": settings.call2eds_lang,
        "persist_path": str(CONFIG_PATH),
        "persist_supported": CONFIG_PATH.exists() or CONFIG_PATH.parent.exists(),
    }


@app.get("/api/hf-token", tags=["config"], summary="Get HF token status")
async def hf_token_status(request: Request):
    _require_admin(request)
    token = _get_hf_token()
    return {
        "set": bool(token),
        "persisted": _env_has_key(CONFIG_PATH, HF_TOKEN_ENV) or _env_has_key(CONFIG_PATH, HF_TOKEN_ENV_FALLBACK),
        "persist_path": str(CONFIG_PATH),
    }


@app.post("/api/hf-token", tags=["config"], summary="Set HF token")
async def hf_token_update(request: Request):
    _require_admin(request)
    payload = await request.json()
    token = (payload or {}).get("token", "")
    persist = bool((payload or {}).get("persist", False))
    if token is None:
        token = ""
    token = str(token).strip()
    if token:
        os.environ[HF_TOKEN_ENV] = token
        os.environ[HF_TOKEN_ENV_FALLBACK] = token
    else:
        os.environ.pop(HF_TOKEN_ENV, None)
        os.environ.pop(HF_TOKEN_ENV_FALLBACK, None)
    if persist:
        updates = {HF_TOKEN_ENV: token if token else None, HF_TOKEN_ENV_FALLBACK: None}
        _write_env_file(CONFIG_PATH, updates)
    return {"ok": True, "set": bool(token), "persisted": persist}


@app.post(
    "/api/config",
    tags=["config"],
    summary="Update runtime config",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {"ok": True, "applied": {"CALL2EDS_MODEL": "medium"}}
                }
            }
        }
    },
)
async def config_update(request: Request):
    _require_admin(request)
    payload = await request.json()
    updates = payload.get("updates", [])
    persist = bool(payload.get("persist", False))
    if not isinstance(updates, list):
        raise HTTPException(status_code=400, detail="updates must be a list")
    applied: Dict[str, Optional[str]] = {}
    for item in updates:
        key = (item or {}).get("key")
        if not key or not _is_editable_key(key):
            continue
        value = _normalize_value((item or {}).get("value"))
        applied[key] = value
    if not applied:
        raise HTTPException(status_code=400, detail="no valid keys to update")

    for key, value in applied.items():
        if value == "":
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
        if key == "CALL2EDS_MODEL" and value:
            settings.call2eds_model = value
        if key == "CALL2EDS_LANG" and value:
            settings.call2eds_lang = value

    if persist:
        if not CONFIG_PATH.parent.exists():
            raise HTTPException(status_code=500, detail="persist path unavailable")
        _write_env_file(CONFIG_PATH, applied)

    return {"ok": True, "applied": applied, "persisted": persist}


@app.get("/export/{call_id}")
async def export(call_id: str):
    call_id = _safe_id(call_id, "call_id")
    out_dir = Path("/tmp/export") / call_id
    manifest_path = export_artifacts(call_id=call_id, run_id=None, out_dir=out_dir)
    return FileResponse(path=manifest_path, filename=manifest_path.name)


@app.get("/export-zip/{call_id}/{run_id}")
async def export_zip(call_id: str, run_id: str):
    call_id = _safe_id(call_id, "call_id")
    run_id = _safe_id(run_id, "run_id")
    # télécharge manifest + artefacts du run dans /tmp puis zip
    out_dir = Path("/tmp/export") / call_id / run_id
    manifest_path = export_artifacts(call_id=call_id, run_id=run_id, out_dir=out_dir)
    zip_path = out_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(out_dir), "zip", out_dir)
    return FileResponse(path=zip_path, filename=zip_path.name)


@app.get("/audio/{call_id}/{run_id}")
async def audio_snippet(call_id: str, run_id: str, t0: float, t1: float):
    call_id = _safe_id(call_id, "call_id")
    run_id = _safe_id(run_id, "run_id")
    try:
        out_path = _make_snippet(call_id, run_id, float(t0), float(t1))
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="ffmpeg failed")
    return FileResponse(path=out_path, media_type="audio/mpeg", filename=out_path.name)


@app.post("/ingest-batch")
async def ingest_batch(
    file_csv: UploadFile = File(...),
    audios: Optional[List[UploadFile]] = None,
):
    def _csv_val(row, key):
        val = row.get(key)
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass
        return val

    tmp_dir = Path("/tmp/batch")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    audio_map: Dict[str, Path] = {}
    if audios:
        for a in audios:
            safe_name = Path(a.filename).name
            p = tmp_dir / safe_name
            with p.open("wb") as f:
                f.write(await a.read())
            audio_map[a.filename] = p
            audio_map[safe_name] = p

    df = pd.read_csv(io.BytesIO(await file_csv.read()))
    run_ids: List[str] = []
    for _, row in df.iterrows():
        apath_val = _csv_val(row, "audio_path")
        if not apath_val:
            raise HTTPException(status_code=400, detail="audio_path missing in csv")
        apath = str(apath_val)
        if apath in audio_map:
            apath = str(audio_map[apath])
        else:
            rel = Path(apath)
            if rel.is_absolute() or ".." in rel.parts:
                raise HTTPException(status_code=400, detail="invalid audio_path in csv")
            candidate = (tmp_dir / rel).resolve()
            if not str(candidate).startswith(str(tmp_dir.resolve())):
                raise HTTPException(status_code=400, detail="invalid audio_path in csv")
            if not candidate.exists():
                raise HTTPException(status_code=400, detail="audio_path not found in upload")
            apath = str(candidate)
        csv_call_id = _csv_val(row, "call_id")
        if csv_call_id:
            csv_call_id = _safe_id(csv_call_id, "call_id")
        if ASYNC_INGEST:
            run_id, _ = _start_background_ingest(
                audio_path=apath,
                call_id=csv_call_id,
                language=_csv_val(row, "lang") or settings.call2eds_lang,
                model_size=_csv_val(row, "model") or settings.call2eds_model,
                user_timestamp=_csv_val(row, "when"),
            )
            run_ids.append(run_id)
        else:
            run_id = ingest_call(
                audio_path=apath,
                call_id=csv_call_id,
                language=_csv_val(row, "lang") or settings.call2eds_lang,
                model_size=_csv_val(row, "model") or settings.call2eds_model,
                user_timestamp=_csv_val(row, "when"),
            )
            run_ids.append(run_id)
    return {"status": "accepted" if ASYNC_INGEST else "ok", "rows": len(df), "run_ids": run_ids, "async": ASYNC_INGEST}


@app.post("/upload-api")
async def upload_api(
    audio: UploadFile = File(...),
    call_id: Optional[str] = Form(None),
    lang: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    when: Optional[str] = Form(None),
):
    tmp_path = Path("/tmp") / Path(audio.filename).name
    with tmp_path.open("wb") as f:
        f.write(await audio.read())
    safe_call_id = _safe_id(call_id, "call_id") if call_id else None
    if ASYNC_INGEST:
        run_id, call_id_final = _start_background_ingest(
            audio_path=str(tmp_path),
            call_id=safe_call_id,
            language=lang or settings.call2eds_lang,
            model_size=model or settings.call2eds_model,
            user_timestamp=when,
        )
        return {"status": "accepted", "run_id": run_id, "call_id": call_id_final, "async": True}
    run_id = ingest_call(
        audio_path=str(tmp_path),
        call_id=safe_call_id,
        language=lang or settings.call2eds_lang,
        model_size=model or settings.call2eds_model,
        user_timestamp=when,
    )
    return {"status": "ok", "run_id": run_id, "call_id": safe_call_id or Path(audio.filename).stem, "async": False}
