"""Serveur FastAPI + UI minimal pour uploader des audios, suivre les runs et télécharger les artefacts."""

import io
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
from fastapi import FastAPI, UploadFile, Form, Request, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from call2eds.config.settings import settings
from call2eds.pipeline.pipeline import ingest_call, list_runs, export_artifacts
from call2eds.storage.minio_client import get_minio


app = FastAPI(title="Call2EDS Web")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


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


def _ffmpeg_bin() -> str:
    from shutil import which
    import imageio_ffmpeg

    sys_ff = which("ffmpeg")
    return sys_ff or imageio_ffmpeg.get_ffmpeg_exe()


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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, call_id: Optional[str] = None):
    runs = list_runs(call_id) if call_id else []
    for r in runs:
        r["created_at_h"] = _human_time(r["created_at"])
    return templates.TemplateResponse("home.html", {"request": request, "call_id": call_id, "runs": runs})


@app.get("/run/{call_id}/{run_id}", response_class=HTMLResponse)
async def run_detail(call_id: str, run_id: str, request: Request):
    manifest_path = _download_manifest(call_id, run_id)
    manifest = json.loads(manifest_path.read_text())
    artifacts = _manifest_public_links(manifest)
    timeline: List[Dict] = []
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
    except Exception:
        timeline = []
    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "call_id": call_id,
            "run_id": run_id,
            "manifest": manifest,
            "artifacts": artifacts,
            "timeline": timeline,
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
    tmp_path = Path("/tmp") / audio.filename
    with tmp_path.open("wb") as f:
        f.write(await audio.read())
    _ = ingest_call(
        audio_path=str(tmp_path),
        call_id=call_id,
        language=lang or settings.call2eds_lang,
        model_size=model or settings.call2eds_model,
        user_timestamp=when,
    )
    return RedirectResponse(url="/" + (f"?call_id={call_id}" if call_id else ""), status_code=303)


@app.get("/export/{call_id}")
async def export(call_id: str):
    out_dir = Path("/tmp/export") / call_id
    manifest_path = export_artifacts(call_id=call_id, run_id=None, out_dir=out_dir)
    return FileResponse(path=manifest_path, filename=manifest_path.name)


@app.get("/export-zip/{call_id}/{run_id}")
async def export_zip(call_id: str, run_id: str):
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
    tmp_dir = Path("/tmp/batch")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    audio_map: Dict[str, Path] = {}
    if audios:
        for a in audios:
            p = tmp_dir / a.filename
            with p.open("wb") as f:
                f.write(await a.read())
            audio_map[a.filename] = p

    df = pd.read_csv(io.BytesIO(await file_csv.read()))
    for _, row in df.iterrows():
        apath = str(row.get("audio_path"))
        if apath in audio_map:
            apath = str(audio_map[apath])
        ingest_call(
            audio_path=apath,
            call_id=row.get("call_id") or None,
            language=row.get("lang") or settings.call2eds_lang,
            model_size=row.get("model") or settings.call2eds_model,
            user_timestamp=row.get("when") or None,
        )
    return {"status": "ok", "rows": len(df)}


@app.post("/upload-api")
async def upload_api(
    audio: UploadFile = File(...),
    call_id: Optional[str] = Form(None),
    lang: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    when: Optional[str] = Form(None),
):
    tmp_path = Path("/tmp") / audio.filename
    with tmp_path.open("wb") as f:
        f.write(await audio.read())
    run_id = ingest_call(
        audio_path=str(tmp_path),
        call_id=call_id,
        language=lang or settings.call2eds_lang,
        model_size=model or settings.call2eds_model,
        user_timestamp=when,
    )
    return {"status": "ok", "run_id": run_id, "call_id": call_id or Path(audio.filename).stem}
