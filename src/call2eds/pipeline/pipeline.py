import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shortuuid

from call2eds.config.settings import settings
from call2eds.db.session import get_session, init_db
from call2eds.db import models
from call2eds.pipeline import audio, asr, prosody
from call2eds.pipeline.diarization import diarize_pyannote, diarize_segments_pyannote
from call2eds.storage.minio_client import get_minio
from call2eds.utils.logger import logger
from call2eds.utils.manifest import build_manifest, save_manifest
from sqlalchemy import text
import os
import librosa
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

# Version du pipeline (distincte de call2eds __version__)
PIPELINE_VERSION = "0.1.8"


class RunCanceled(Exception):
    pass


def _upsert_run(call_id: str, run_id: str, status: str, params_json: Optional[Dict] = None) -> None:
    init_db()
    with get_session() as session:
        call = session.get(models.Call, call_id)
        if not call:
            call = models.Call(call_id=call_id, meta_json={})
            session.add(call)
        run = session.get(models.Run, run_id)
        if not run:
            run = models.Run(
                run_id=run_id,
                call_id=call_id,
                pipeline_version=PIPELINE_VERSION,
                params_json=params_json or {},
                status=status,
            )
            session.add(run)
        else:
            run.status = status
            run.pipeline_version = PIPELINE_VERSION
            if params_json:
                run.params_json = params_json


def _set_metric(run_id: str, key: str, value_num: Optional[float] = None, value_json: Optional[Dict] = None) -> None:
    init_db()
    with get_session() as session:
        session.query(models.Metric).filter(
            models.Metric.run_id == run_id, models.Metric.key == key
        ).delete()
        session.add(models.Metric(run_id=run_id, key=key, value_num=value_num, value_json=value_json))


def _cancel_requested(run_id: str) -> bool:
    init_db()
    with get_session() as session:
        metric = (
            session.query(models.Metric)
            .filter(models.Metric.run_id == run_id, models.Metric.key == "cancel_requested")
            .order_by(models.Metric.id.desc())
            .first()
        )
    if not metric:
        return False
    if metric.value_num and metric.value_num > 0:
        return True
    if isinstance(metric.value_json, dict):
        return bool(metric.value_json.get("requested"))
    return False


def _raise_if_canceled(run_id: str, call_id: str, params_json: Optional[Dict] = None) -> None:
    if not _cancel_requested(run_id):
        return
    _upsert_run(call_id, run_id, status="canceled", params_json=params_json)
    _set_progress(run_id, "canceled", 1.0)
    logger.info("Ingestion canceled run_id=%s", run_id)
    try:
        _purge_run_prefix(call_id, run_id)
    except Exception:  # noqa: BLE001
        logger.warning("Cancel purge prefix failed run_id=%s", run_id)
    try:
        purge_run(call_id, run_id)
    except Exception:  # noqa: BLE001
        logger.warning("Cancel purge DB failed run_id=%s", run_id)
    raise RunCanceled()


def _set_progress(run_id: str, stage: str, pct: float) -> None:
    _set_metric(run_id, "progress_stage", value_json={"stage": stage})
    _set_metric(run_id, "progress_pct", value_num=pct)


def prepare_ingest(
    audio_path: str,
    call_id: Optional[str],
    language: str,
    model_size: str,
    user_timestamp: Optional[str] = None,
    status: str = "running",
) -> tuple[str, str, Dict]:
    call_id_final = call_id or Path(audio_path).stem
    run_id = shortuuid.uuid()
    params_json = {"model": model_size, "lang": language}
    if user_timestamp:
        params_json["user_timestamp"] = user_timestamp
    _upsert_run(call_id_final, run_id, status=status, params_json=params_json)
    if status == "queued":
        _set_progress(run_id, "queued", 0.01)
    return run_id, call_id_final, params_json


def ingest_call_with_run_id(
    run_id: str,
    call_id_final: str,
    audio_path: str,
    language: str,
    model_size: str,
    user_timestamp: Optional[str] = None,
    params_json: Optional[Dict] = None,
) -> str:
    tmpdir = Path(tempfile.mkdtemp(prefix=f"call2eds_{run_id}_"))
    params_json = params_json or {"model": model_size, "lang": language}
    if user_timestamp:
        params_json["user_timestamp"] = user_timestamp
    try:
        _upsert_run(call_id_final, run_id, status="running", params_json=params_json)
        _set_progress(run_id, "start", 0.02)
        _raise_if_canceled(run_id, call_id_final, params_json=params_json)
        logger.info("Ingestion start call_id=%s run_id=%s", call_id_final, run_id)
        wav_stereo, flac_path, channel_wavs, is_stereo, wav_mono = audio.convert_audio(Path(audio_path), tmpdir)
        quality_audio = audio.basic_quality(wav_stereo)
        _set_progress(run_id, "audio", 0.1)
        _raise_if_canceled(run_id, call_id_final, params_json=params_json)

        tokens_all: List[Dict] = []
        turns_all: List[Dict] = []
        frames_all: List[pd.DataFrame] = []
        total_conf_sum = 0.0
        total_words = 0

        for speaker_id, ch_wav in enumerate(channel_wavs):
            tokens_raw, turns_raw, avg_conf, words_count = asr.transcribe_audio(
                ch_wav, model_size=model_size, language=language, speaker_id=speaker_id
            )
            base_turn = len(turns_all)
            for t in turns_raw:
                t["turn_id"] += base_turn
            for tok in tokens_raw:
                tok["turn_id"] += base_turn

            tokens_all.extend(tokens_raw)
            turns_all.extend(turns_raw)
            if words_count > 0:
                total_conf_sum += avg_conf * words_count
                total_words += words_count

            frames_df = prosody.extract_frames(ch_wav, speaker_id=speaker_id)
            frames_all.append(frames_df)
            _raise_if_canceled(run_id, call_id_final, params_json=params_json)
        _set_progress(run_id, "asr", 0.4)
        _raise_if_canceled(run_id, call_id_final, params_json=params_json)

        tokens_df = pd.DataFrame(tokens_all)
        if tokens_df.empty:
            tokens_df = pd.DataFrame(columns=["turn_id", "token_id", "word", "t_start", "t_end", "asr_conf_word", "speaker_id"])
        turns_df = pd.DataFrame(turns_all)
        if turns_df.empty:
            turns_df = pd.DataFrame(columns=["turn_id", "speaker_id", "t_start", "t_end", "text", "asr_conf_turn"])
        frames_df = pd.concat(frames_all, ignore_index=True) if frames_all else pd.DataFrame()

        # Diarisation : mono → essayer ECAPA; stéréo → concat canaux puis ECAPA; fallback MFCC
        if not turns_df.empty:
            max_speakers = int(os.getenv("CALL2EDS_MAX_SPK", "6"))
            min_speakers = int(os.getenv("CALL2EDS_MIN_SPK", "1"))
            smooth_s = float(os.getenv("CALL2EDS_DIAR_SMOOTH_S", "1.0"))
            gap_s = float(os.getenv("CALL2EDS_DIAR_TURN_GAP_S", "0.6"))
            mix = wav_mono if wav_mono.exists() else channel_wavs[0]

            # 1) pyannote overlap-aware -> assign speakers to tokens + rebuild turns
            segments = diarize_segments_pyannote(mix, min_spk=min_speakers, max_spk=max_speakers)
            if segments:
                tokens_df = _assign_speakers_to_tokens(tokens_df, segments)
                frames_df = _assign_speakers_to_frames(frames_df, segments)
                tokens_df, frames_df, refined = _maybe_refine_early_with_ecapa(
                    tokens_df,
                    frames_df,
                    mix,
                    max_spk=max_speakers,
                    min_spk=min_speakers,
                )
                if refined:
                    logger.info("Diarisation: correction ECAPA appliquée sur le début du call.")
                tokens_df, frames_df, refined_global = _refine_with_prototype_windows_global(
                    tokens_df, frames_df, mix, max_spk=max_speakers
                )
                if refined_global:
                    logger.info("Diarisation: resegmentation globale par prototypes appliquée.")
                tokens_df, frames_df, _ = _refine_with_ecapa_microturns(
                    tokens_df, frames_df, mix, max_spk=max_speakers
                )
                tokens_df, frames_df, _ = _merge_short_speaker_flips(tokens_df, frames_df)
                tokens_df, turns_df = _rebuild_turns_from_tokens(tokens_df, gap_s=gap_s)
            else:
                # 2) pyannote turn-level fallback
                new_labels = diarize_pyannote(mix, turns_df, max_spk=max_speakers, min_spk=min_speakers)
                if new_labels is not None:
                    try:
                        nuniq = pd.Series(new_labels).nunique(dropna=True)
                    except Exception:  # noqa: BLE001
                        nuniq = 0
                    if nuniq < max(1, min_speakers):
                        logger.warning(
                            "Diarisation pyannote retourne %s speaker(s) (< min_speakers=%s). Fallback.",
                            nuniq,
                            min_speakers,
                        )
                        new_labels = None
                # 3) ECAPA clustering
                if new_labels is None:
                    new_labels = _diarize_speechbrain(mix, turns_df, k_max=max_speakers)
                    if new_labels is not None:
                        try:
                            nuniq = pd.Series(new_labels).nunique(dropna=True)
                        except Exception:  # noqa: BLE001
                            nuniq = 0
                        if nuniq < max(1, min_speakers):
                            logger.warning(
                                "Diarisation speechbrain retourne %s speaker(s) (< min_speakers=%s). Fallback.",
                                nuniq,
                                min_speakers,
                            )
                            new_labels = None
                # 4) fallback MFCC
                if new_labels is None:
                    new_labels = _cluster_speakers_mono(mix, turns_df, k=2, k_max=max_speakers)

                if new_labels is None:
                    new_labels = pd.Series([0] * len(turns_df), index=turns_df.index, dtype=int)
                elif not isinstance(new_labels, pd.Series):
                    new_labels = pd.Series(new_labels, index=turns_df.index, dtype=int)
                else:
                    new_labels = new_labels.reindex(turns_df.index).fillna(0).astype(int)

                # optional smoothing: fix short flip-flops (A-B-A)
                if smooth_s > 0 and len(new_labels) >= 3:
                    labels = new_labels.to_numpy().copy()
                    durations = (turns_df["t_end"] - turns_df["t_start"]).to_numpy()
                    for i in range(1, len(labels) - 1):
                        if (
                            labels[i - 1] == labels[i + 1]
                            and labels[i] != labels[i - 1]
                            and durations[i] <= smooth_s
                        ):
                            labels[i] = labels[i - 1]
                    new_labels = pd.Series(labels, index=turns_df.index, dtype=int)

                turns_df["speaker_id"] = new_labels.values
                if not tokens_df.empty:
                    tokens_df = tokens_df.merge(
                        turns_df[["turn_id", "speaker_id"]], on="turn_id", how="left", suffixes=("", "_t")
                    )
                    tokens_df["speaker_id"] = tokens_df["speaker_id_t"].fillna(tokens_df["speaker_id"]).astype(int)
                    tokens_df = tokens_df.drop(columns=["speaker_id_t"])
        _set_progress(run_id, "diar", 0.6)
        _raise_if_canceled(run_id, call_id_final, params_json=params_json)

        words_df = prosody.aggregate_words(tokens_df, frames_df)
        turns_prosody_df = prosody.aggregate_turns(turns_df, tokens_df, frames_df)
        _set_progress(run_id, "prosody", 0.75)
        _raise_if_canceled(run_id, call_id_final, params_json=params_json)

        # enrich identifiers
        for df in [tokens_df, turns_df, frames_df, words_df, turns_prosody_df]:
            df["call_id"] = call_id_final
            df["run_id"] = run_id

        # quality metrics
        pct_voiced = float((frames_df.voicing > 0.5).mean()) if not frames_df.empty else 0.0
        # pitch confidence: compute on "speech-like" frames (exclude very low energy)
        # to avoid penalizing long silences. If energy is missing/constant, fallback to global mean.
        pitch_conf_mean = float(frames_df.pitch_conf.mean()) if not frames_df.empty else 0.0
        if not frames_df.empty and "energy" in frames_df.columns:
            try:
                energy = pd.to_numeric(frames_df["energy"], errors="coerce").fillna(0.0)
                if float(energy.max()) > float(energy.min()):
                    thr = float(energy.quantile(0.3))
                    mask = energy > thr
                    if bool(mask.any()):
                        pitch_conf_mean = float(pd.to_numeric(frames_df.loc[mask, "pitch_conf"], errors="coerce").mean())
            except Exception:  # noqa: BLE001
                pass

        warnings_list = []
        if frames_df.empty:
            warnings_list.append("prosody: frames missing (extraction failed)")
        if pct_voiced == 0.0 and frames_df.shape[0] > 0:
            warnings_list.append("prosody: no voiced frames detected (check audio or opensmile)")
        if pitch_conf_mean < 0.4 and frames_df.shape[0] > 0:
            warnings_list.append("prosody: low pitch confidence (<0.4) - pitch extraction likely failed")

        quality = {
            **quality_audio,
            "asr_conf_mean": (total_conf_sum / total_words) if total_words else 0.0,
            "asr_conf_p10": float(tokens_df.asr_conf_word.quantile(0.1)) if not tokens_df.empty else 0.0,
            "asr_conf_p90": float(tokens_df.asr_conf_word.quantile(0.9)) if not tokens_df.empty else 0.0,
            "pct_voiced": pct_voiced,
            "pct_pitch_invalid": float(1 - pitch_conf_mean) if not frames_df.empty else 0.0,
            "warnings": warnings_list,
        }
        quality_df = pd.DataFrame([quality | {"call_id": call_id_final, "run_id": run_id}])

        # write parquet files
        eds_dir = tmpdir / "eds"
        eds_dir.mkdir(parents=True, exist_ok=True)
        file_map = {
            "turns": turns_df,
            "tokens": tokens_df,
            "prosody_frames": frames_df,
            "prosody_words": words_df,
            "prosody_turns": turns_prosody_df,
            "quality": quality_df,
        }
        parquet_paths: Dict[str, Path] = {}
        for name, df in file_map.items():
            path = eds_dir / f"{name}.parquet"
            table = pa.Table.from_pandas(df)
            pq.write_table(table, path)
            parquet_paths[name] = path
        _set_progress(run_id, "bundle", 0.85)
        _raise_if_canceled(run_id, call_id_final, params_json=params_json)

        manifest = build_manifest(
            call_id=call_id_final,
            run_id=run_id,
            model_name=model_size,
            language=language,
            artifacts=[],  # filled later
            params={
                "sample_rate": 16000,
                "hop_ms": 10,
                "asr_language": language,
                "asr_model": model_size,
                "user_timestamp": user_timestamp,
            },
            stats=quality,
            ffmpeg_version=_ffmpeg_version(),
        )
        manifest_path = tmpdir / "manifest.json"
        save_manifest(manifest, manifest_path)
        _raise_if_canceled(run_id, call_id_final, params_json=params_json)

        minio = get_minio()
        artifacts: List[Dict] = []

        # upload audio FLAC
        audio_key = f"calls/{call_id_final}/runs/{run_id}/audio/normalized.flac"
        audio_uri = minio.upload_file(flac_path, audio_key)
        artifacts.append(_artifact_entry(audio_key, audio_uri, flac_path))

        # upload parquet files
        for name, path in parquet_paths.items():
            key = f"calls/{call_id_final}/runs/{run_id}/eds/{name}.parquet"
            uri = minio.upload_file(path, key)
            artifacts.append(_artifact_entry(key, uri, path))
        _set_progress(run_id, "upload", 0.9)
        _raise_if_canceled(run_id, call_id_final, params_json=params_json)

        # upload manifest
        manifest["artifacts"] = artifacts
        save_manifest(manifest, manifest_path)
        manifest_key = f"calls/{call_id_final}/runs/{run_id}/manifest.json"
        manifest_uri = minio.upload_file(manifest_path, manifest_key)
        artifacts.append(_artifact_entry(manifest_key, manifest_uri, manifest_path))

        # record DB
        _record_db(
            call_id_final,
            run_id,
            model_size,
            language,
            artifacts,
            quality,
            extra_params={"user_timestamp": user_timestamp} if user_timestamp else None,
        )
        _upsert_run(call_id_final, run_id, status="completed", params_json=params_json)
        _set_progress(run_id, "completed", 1.0)

        logger.info("Ingestion terminé run_id=%s", run_id)
        return run_id
    except RunCanceled:
        return run_id
    except Exception as exc:  # noqa: BLE001
        _upsert_run(call_id_final, run_id, status="failed", params_json=params_json)
        _set_progress(run_id, "failed", 1.0)
        _set_metric(run_id, "error", value_json={"error": str(exc)})
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def ingest_call(
    audio_path: str,
    call_id: Optional[str],
    language: str,
    model_size: str,
    user_timestamp: Optional[str] = None,
) -> str:
    run_id, call_id_final, params_json = prepare_ingest(
        audio_path=audio_path,
        call_id=call_id,
        language=language,
        model_size=model_size,
        user_timestamp=user_timestamp,
        status="running",
    )
    return ingest_call_with_run_id(
        run_id,
        call_id_final,
        audio_path,
        language,
        model_size,
        user_timestamp=user_timestamp,
        params_json=params_json,
    )


def _artifact_entry(key: str, s3_uri: str, path: Path) -> Dict[str, object]:
    from call2eds.storage.minio_client import MinioClient

    sha = MinioClient.sha256_file(path)
    return {"key": key, "s3_uri": s3_uri, "sha256": sha, "size_bytes": path.stat().st_size}


def _record_db(
    call_id: str,
    run_id: str,
    model: str,
    lang: str,
    artifacts: List[Dict],
    quality: Dict,
    extra_params: Optional[Dict] = None,
):
    init_db()
    with get_session() as session:
        call = session.get(models.Call, call_id)
        if not call:
            call = models.Call(call_id=call_id, meta_json={})
            session.add(call)
        params_json = {"model": model, "lang": lang}
        if extra_params:
            params_json.update(extra_params)
        run = session.get(models.Run, run_id)
        if not run:
            run = models.Run(
                run_id=run_id,
                call_id=call_id,
                pipeline_version=PIPELINE_VERSION,
                params_json=params_json,
                status="completed",
            )
            session.add(run)
        else:
            run.status = "completed"
            run.pipeline_version = PIPELINE_VERSION
            run.params_json = params_json
        for art in artifacts:
            session.add(
                models.Artifact(
                    run_id=run_id,
                    kind=_kind_from_key(art["key"]),
                    s3_uri=art.get("s3_uri", f"s3://{settings.minio_bucket}/{art['key']}"),
                    sha256=art["sha256"],
                    size_bytes=art["size_bytes"],
                )
            )
        for k, v in quality.items():
            if k == "warnings":
                session.add(models.Metric(run_id=run_id, key=k, value_json=v))
            else:
                try:
                    val = float(v)
                except Exception:
                    val = None
                session.add(models.Metric(run_id=run_id, key=k, value_num=val, value_json=None))


def _ffmpeg_version() -> str:
    """Retourne la première ligne de `ffmpeg -version`, ou 'unknown'."""
    try:
        out = subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=5)
        return out.splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Impossible de récupérer la version ffmpeg: %s", exc)
        return "unknown"


def _kind_from_key(key: str) -> str:
    if key.endswith(".parquet"):
        return Path(key).stem
    if key.endswith(".flac"):
        return "audio"
    if key.endswith("manifest.json"):
        return "manifest"
    return "other"


def list_runs(call_id: str) -> List[Dict]:
    init_db()
    with get_session() as session:
        runs = (
            session.query(models.Run)
            .filter(models.Run.call_id == call_id)
            .order_by(models.Run.created_at.desc())
            .all()
        )
        return [
            {
                "run_id": r.run_id,
                "created_at": r.created_at.isoformat(),
                "status": r.status,
                "model": (r.params_json or {}).get("model"),
            }
            for r in runs
        ]


def list_runs_recent(limit: int = 200) -> List[Dict]:
    init_db()
    with get_session() as session:
        runs = (
            session.query(models.Run)
            .order_by(models.Run.created_at.desc())
            .limit(limit)
            .all()
        )
        run_ids = [r.run_id for r in runs]
        duration_map: Dict[str, float] = {}
        if run_ids:
            rows = (
                session.query(models.Metric.run_id, models.Metric.value_num)
                .filter(models.Metric.run_id.in_(run_ids), models.Metric.key == "duration_s")
                .order_by(models.Metric.id.desc())
                .all()
            )
            for run_id, value in rows:
                if run_id not in duration_map:
                    duration_map[run_id] = float(value) if value is not None else 0.0
        return [
            {
                "call_id": r.call_id,
                "run_id": r.run_id,
                "created_at": r.created_at.isoformat(),
                "status": r.status,
                "model": (r.params_json or {}).get("model"),
                "user_timestamp": (r.params_json or {}).get("user_timestamp"),
                "pipeline_version": r.pipeline_version,
                "duration_s": duration_map.get(r.run_id),
            }
            for r in runs
        ]


def export_artifacts(call_id: str, out_dir: Path, run_id: Optional[str] = None) -> Path:
    init_db()
    with get_session() as session:
        query = session.query(models.Run).filter(models.Run.call_id == call_id)
        if run_id:
            query = query.filter(models.Run.run_id == run_id)
        run = query.order_by(models.Run.created_at.desc()).first()
        if not run:
            raise RuntimeError("Run introuvable")
        artifacts = session.query(models.Artifact).filter(models.Artifact.run_id == run.run_id).all()
    minio = get_minio()
    manifest_path: Optional[Path] = None
    for art in artifacts:
        key = art.s3_uri.split(f"s3://{settings.minio_bucket}/")[-1]
        dest = out_dir / Path(key).name
        minio.download_file(key, dest)
        # simple sha check
        sha = minio.sha256_file(dest)
        if sha != art.sha256:
            raise RuntimeError(f"Checksum mismatch for {dest}")
        if dest.name == "manifest.json":
            manifest_path = dest
    return manifest_path or out_dir / "manifest.json"


def _purge_run_prefix(call_id: str, run_id: str) -> None:
    minio = get_minio()
    prefix = f"calls/{call_id}/runs/{run_id}/"
    paginator = minio.client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.minio_bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue
        objects = [{"Key": obj["Key"]} for obj in contents]
        minio.client.delete_objects(Bucket=settings.minio_bucket, Delete={"Objects": objects})


def purge_run(call_id: str, run_id: Optional[str]):
    init_db()
    with get_session() as session:
        runs = session.query(models.Run).filter(models.Run.call_id == call_id)
        if run_id:
            runs = runs.filter(models.Run.run_id == run_id)
        runs = runs.all()
        minio = get_minio()
        for r in runs:
            artifacts = session.query(models.Artifact).filter(models.Artifact.run_id == r.run_id).all()
            for art in artifacts:
                key = art.s3_uri.split(f"s3://{settings.minio_bucket}/")[-1]
                try:
                    minio.client.delete_object(Bucket=settings.minio_bucket, Key=key)
                except Exception:
                    logger.warning("Impossible de supprimer %s", key)
            session.query(models.Metric).filter(models.Metric.run_id == r.run_id).delete()
            session.query(models.Artifact).filter(models.Artifact.run_id == r.run_id).delete()
            session.delete(r)
        session.commit()


def doctor() -> Dict[str, Dict[str, object]]:
    report: Dict[str, Dict[str, object]] = {}
    # ffmpeg
    try:
        from call2eds.pipeline.audio import _ffmpeg_bin

        bin_path = _ffmpeg_bin()
        ffmpeg_ok = bin_path is not None
        report["ffmpeg"] = {"ok": ffmpeg_ok, "bin": bin_path}
    except Exception as e:
        report["ffmpeg"] = {"ok": False, "error": str(e)}
    # minio
    minio = get_minio()
    try:
        buckets = minio.client.list_buckets()
        report["minio"] = {"ok": True, "buckets": [b["Name"] for b in buckets.get("Buckets", [])]}
    except Exception as e:
        report["minio"] = {"ok": False, "error": str(e)}
    # postgres
    try:
        init_db()
        with get_session() as session:
            session.execute(text("SELECT 1"))
        report["postgres"] = {"ok": True}
    except Exception as e:
        report["postgres"] = {"ok": False, "error": str(e)}
    return report


def _assign_speakers_to_tokens(tokens_df: pd.DataFrame, segments: List[Dict[str, float]]) -> pd.DataFrame:
    """Assign speaker_id to tokens by maximal overlap with diarization segments."""
    if tokens_df.empty or not segments:
        return tokens_df
    segs = sorted(segments, key=lambda s: s["start"])
    tokens = tokens_df.sort_values("t_start").copy()
    speaker_ids = []
    speaker_conf = []
    overlap_ratio = []
    speaker_alt = []
    overlap_min = float(os.getenv("CALL2EDS_DIAR_OVERLAP_MIN", "0.25"))
    overlap_strict = os.getenv("CALL2EDS_DIAR_OVERLAP_STRICT", "0").strip().lower() in ("1", "true", "yes")
    idx = 0
    for _, tok in tokens.iterrows():
        t0 = float(tok.t_start)
        t1 = float(tok.t_end)
        if t1 <= t0:
            speaker_ids.append(int(tok.get("speaker_id", 0) or 0))
            speaker_conf.append(0.0)
            overlap_ratio.append(0.0)
            speaker_alt.append(-1)
            continue
        while idx < len(segs) and segs[idx]["end"] <= t0:
            idx += 1
        j = idx
        dur_by_spk: Dict[int, float] = {}
        while j < len(segs) and segs[j]["start"] < t1:
            s0 = segs[j]["start"]
            s1 = segs[j]["end"]
            overlap = max(0.0, min(t1, s1) - max(t0, s0))
            if overlap > 0:
                spk = int(segs[j]["speaker"])
                dur_by_spk[spk] = dur_by_spk.get(spk, 0.0) + overlap
            j += 1
        if dur_by_spk:
            ranked = sorted(dur_by_spk.items(), key=lambda kv: kv[1], reverse=True)
            spk, best = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            conf = best / max(1e-6, (t1 - t0))
            ov_ratio = second / max(1e-6, (t1 - t0))
            if overlap_strict and ov_ratio >= overlap_min:
                speaker_ids.append(-1)
            else:
                speaker_ids.append(int(spk))
            speaker_conf.append(float(conf))
            overlap_ratio.append(float(ov_ratio))
            speaker_alt.append(int(ranked[1][0]) if len(ranked) > 1 else -1)
        else:
            speaker_ids.append(int(tok.get("speaker_id", 0) or 0))
            speaker_conf.append(0.0)
            overlap_ratio.append(0.0)
            speaker_alt.append(-1)
    tokens["speaker_id"] = speaker_ids
    tokens["speaker_conf"] = speaker_conf
    tokens["overlap_ratio"] = overlap_ratio
    tokens["speaker_alt"] = speaker_alt

    # smooth low-confidence tokens with local majority vote
    min_conf = float(os.getenv("CALL2EDS_DIAR_MIN_CONF", "0.6"))
    win = int(os.getenv("CALL2EDS_DIAR_SMOOTH_TOKENS", "2"))
    if win > 0 and len(tokens) > 1:
        ids = tokens["speaker_id"].to_numpy().copy()
        confs = tokens["speaker_conf"].to_numpy().copy()
        overlaps = tokens["overlap_ratio"].to_numpy().copy()
        for i in range(len(ids)):
            if confs[i] >= min_conf:
                continue
            if overlaps[i] >= overlap_min:
                continue
            if ids[i] < 0:
                continue
            left = max(0, i - win)
            right = min(len(ids), i + win + 1)
            window = ids[left:right]
            if len(window) == 0:
                continue
            # majority vote in window
            vals, counts = np.unique(window, return_counts=True)
            ids[i] = int(vals[np.argmax(counts)])
        tokens["speaker_id"] = ids
    return tokens


def _assign_speakers_to_frames(frames_df: pd.DataFrame, segments: List[Dict[str, float]]) -> pd.DataFrame:
    """Assign speaker_id to frames by diarization segments."""
    if frames_df.empty or not segments or "t" not in frames_df.columns:
        return frames_df
    segs = sorted(segments, key=lambda s: s["start"])
    frames = frames_df.sort_values("t").copy()
    speaker_ids = []
    overlap_flags = []
    active: List[Dict[str, float]] = []
    idx = 0
    for _, fr in frames.iterrows():
        t = float(fr.t)
        while idx < len(segs) and segs[idx]["start"] <= t:
            active.append(segs[idx])
            idx += 1
        if active:
            active = [s for s in active if s["end"] > t]
        if not active:
            speaker_ids.append(int(fr.get("speaker_id", 0) or 0))
            overlap_flags.append(0)
            continue
        if len(active) == 1:
            speaker_ids.append(int(active[0]["speaker"]))
            overlap_flags.append(0)
        else:
            # overlapping speech: mark as unknown to avoid mixing prosody
            speaker_ids.append(-1)
            overlap_flags.append(1)
    frames["speaker_id"] = speaker_ids
    frames["overlap"] = overlap_flags
    return frames


def _merge_short_speaker_flips(
    tokens_df: pd.DataFrame, frames_df: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame | None, int]:
    """
    Merge short A-B-A speaker flips (same speaker split by a short segment).

    This targets cases like: spk1 -> spk0 ("oui") -> spk1 where the middle
    segment is very short and non-overlapping, typically a diarization wobble.
    """
    enabled = os.getenv("CALL2EDS_DIAR_FLIP_ENABLED", "1").strip().lower() in ("1", "true", "yes")
    if not enabled or tokens_df.empty or "speaker_id" not in tokens_df.columns:
        return tokens_df, frames_df, 0

    max_s_env = os.getenv("CALL2EDS_DIAR_FLIP_MAX_S")
    if max_s_env:
        try:
            max_s = float(max_s_env)
        except Exception:
            max_s = 1.3
    else:
        try:
            smooth_fallback = os.getenv("CALL2EDS_DIAR_SMOOTH_S")
            max_s = float(smooth_fallback) if smooth_fallback else 1.3
        except Exception:
            max_s = 1.3
    try:
        max_tokens = int(os.getenv("CALL2EDS_DIAR_FLIP_MAX_TOKENS", "6"))
    except Exception:
        max_tokens = 6
    try:
        max_ov = float(os.getenv("CALL2EDS_DIAR_FLIP_MAX_OV", "0.2"))
    except Exception:
        max_ov = 0.2
    max_conf = None
    max_conf_raw = os.getenv("CALL2EDS_DIAR_FLIP_MAX_CONF")
    if max_conf_raw is None or max_conf_raw.strip() == "":
        max_conf_raw = "0.8"
    max_conf_raw = max_conf_raw.strip()
    if max_conf_raw:
        try:
            max_conf = float(max_conf_raw)
        except Exception:
            max_conf = None
    gap_raw = os.getenv("CALL2EDS_DIAR_FLIP_MAX_GAP_S")
    if gap_raw is None or gap_raw.strip() == "":
        gap_raw = os.getenv("CALL2EDS_DIAR_TURN_GAP_S", "0.6")
    gap_raw = gap_raw.strip()
    try:
        max_gap = float(gap_raw) if gap_raw else None
    except Exception:
        max_gap = None

    if max_s <= 0 and max_tokens <= 0:
        return tokens_df, frames_df, 0

    tokens = tokens_df.sort_values("t_start").reset_index(drop=False)
    if tokens.empty:
        return tokens_df, frames_df, 0

    def _build_runs(df: pd.DataFrame) -> list[dict]:
        runs: list[dict] = []
        cur_spk = int(df.loc[0, "speaker_id"]) if pd.notna(df.loc[0, "speaker_id"]) else -1
        cur_start = float(df.loc[0, "t_start"])
        cur_end = float(df.loc[0, "t_end"])
        cur_idxs = [int(df.loc[0, "index"])]
        cur_ov = [float(df.loc[0, "overlap_ratio"]) if "overlap_ratio" in df.columns else 0.0]
        cur_conf = [float(df.loc[0, "speaker_conf"]) if "speaker_conf" in df.columns else 0.0]
        for i in range(1, len(df)):
            spk = int(df.loc[i, "speaker_id"]) if pd.notna(df.loc[i, "speaker_id"]) else -1
            if spk == cur_spk:
                cur_end = max(cur_end, float(df.loc[i, "t_end"]))
                cur_idxs.append(int(df.loc[i, "index"]))
                cur_ov.append(float(df.loc[i, "overlap_ratio"]) if "overlap_ratio" in df.columns else 0.0)
                if "speaker_conf" in df.columns:
                    cur_conf.append(float(df.loc[i, "speaker_conf"]))
            else:
                runs.append(
                    {
                        "speaker_id": cur_spk,
                        "start": cur_start,
                        "end": cur_end,
                        "idxs": cur_idxs,
                        "overlap_mean": float(np.mean(cur_ov)) if cur_ov else 0.0,
                        "conf_mean": float(np.mean(cur_conf)) if cur_conf else None,
                        "token_count": len(cur_idxs),
                    }
                )
                cur_spk = spk
                cur_start = float(df.loc[i, "t_start"])
                cur_end = float(df.loc[i, "t_end"])
                cur_idxs = [int(df.loc[i, "index"])]
                cur_ov = [float(df.loc[i, "overlap_ratio"]) if "overlap_ratio" in df.columns else 0.0]
                cur_conf = [float(df.loc[i, "speaker_conf"]) if "speaker_conf" in df.columns else 0.0]
        runs.append(
            {
                "speaker_id": cur_spk,
                "start": cur_start,
                "end": cur_end,
                "idxs": cur_idxs,
                "overlap_mean": float(np.mean(cur_ov)) if cur_ov else 0.0,
                "conf_mean": float(np.mean(cur_conf)) if cur_conf else None,
                "token_count": len(cur_idxs),
            }
        )
        return runs

    total_merged = 0
    changes: list[dict] = []
    for _ in range(2):  # two passes max to catch chained flips
        runs = _build_runs(tokens)
        pending: list[tuple[dict, int]] = []
        # edge merge (start): short first run -> merge into next
        edge_enabled = os.getenv("CALL2EDS_DIAR_EDGE_ENABLED", "1").strip().lower() in ("1", "true", "yes")
        if edge_enabled and len(runs) >= 2:
            edge_window_raw = os.getenv("CALL2EDS_DIAR_EDGE_WINDOW_S", "8.0")
            edge_max_s_raw = os.getenv("CALL2EDS_DIAR_EDGE_MAX_S")
            edge_max_tokens_raw = os.getenv("CALL2EDS_DIAR_EDGE_MAX_TOKENS")
            edge_max_ov_raw = os.getenv("CALL2EDS_DIAR_EDGE_MAX_OV")
            edge_min_next_raw = os.getenv("CALL2EDS_DIAR_EDGE_MIN_NEXT_S", "3.0")
            edge_gap_raw = os.getenv("CALL2EDS_DIAR_EDGE_MAX_GAP_S")
            if edge_max_s_raw is None or edge_max_s_raw.strip() == "":
                edge_max_s_raw = str(max(max_s, 2.0))
            if edge_max_tokens_raw is None or edge_max_tokens_raw.strip() == "":
                edge_max_tokens_raw = str(max_tokens)
            if edge_max_ov_raw is None or edge_max_ov_raw.strip() == "":
                edge_max_ov_raw = str(max_ov)
            if edge_gap_raw is None or edge_gap_raw.strip() == "":
                edge_gap_raw = gap_raw
            try:
                edge_window_s = float(edge_window_raw)
            except Exception:
                edge_window_s = 8.0
            try:
                edge_max_s = float(edge_max_s_raw)
            except Exception:
                edge_max_s = max_s
            try:
                edge_max_tokens = int(edge_max_tokens_raw)
            except Exception:
                edge_max_tokens = max_tokens
            try:
                edge_max_ov = float(edge_max_ov_raw)
            except Exception:
                edge_max_ov = max_ov
            try:
                edge_min_next_s = float(edge_min_next_raw)
            except Exception:
                edge_min_next_s = 3.0
            edge_gap = None
            if edge_gap_raw is not None and edge_gap_raw.strip() != "":
                try:
                    edge_gap = float(edge_gap_raw.strip())
                except Exception:
                    edge_gap = None

            first = runs[0]
            second = runs[1]
            if (
                first["speaker_id"] >= 0
                and second["speaker_id"] >= 0
                and first["start"] <= edge_window_s
            ):
                dur = float(first["end"] - first["start"])
                next_dur = float(second["end"] - second["start"])
                if edge_max_s > 0 and dur <= edge_max_s:
                    if edge_max_tokens > 0 and first["token_count"] <= edge_max_tokens:
                        if first["overlap_mean"] < edge_max_ov:
                            if max_conf is None or first["conf_mean"] is None or first["conf_mean"] < max_conf:
                                if edge_min_next_s <= 0 or next_dur >= edge_min_next_s:
                                    if edge_gap is None or (second["start"] - first["end"]) <= edge_gap:
                                        pending.append((first, second["speaker_id"]))

        for i in range(1, len(runs) - 1):
            prev = runs[i - 1]
            cur = runs[i]
            nxt = runs[i + 1]
            if prev["speaker_id"] < 0 or cur["speaker_id"] < 0 or nxt["speaker_id"] < 0:
                continue
            if prev["speaker_id"] != nxt["speaker_id"]:
                continue
            dur = float(cur["end"] - cur["start"])
            if max_s > 0 and dur > max_s:
                continue
            if max_tokens > 0 and cur["token_count"] > max_tokens:
                continue
            if cur["overlap_mean"] >= max_ov:
                continue
            if max_conf is not None and cur["conf_mean"] is not None and cur["conf_mean"] >= max_conf:
                continue
            if max_gap is not None:
                if (cur["start"] - prev["end"]) > max_gap or (nxt["start"] - cur["end"]) > max_gap:
                    continue
            pending.append((cur, prev["speaker_id"]))

        if not pending:
            break
        for cur, new_spk in pending:
            tokens.loc[tokens["index"].isin(cur["idxs"]), "speaker_id"] = new_spk
            changes.append({"start": cur["start"], "end": cur["end"], "old": cur["speaker_id"], "new": new_spk})
            total_merged += 1

    if total_merged > 0:
        # write back to original tokens_df
        tokens_df.loc[tokens["index"], "speaker_id"] = tokens["speaker_id"].to_numpy()
        # update frames to stay consistent for prosody aggregation
        if frames_df is not None and not frames_df.empty and "t" in frames_df.columns:
            frames = frames_df.copy()
            for ch in changes:
                mask = (frames["t"] >= ch["start"]) & (frames["t"] <= ch["end"])
                if "overlap" in frames.columns:
                    mask &= frames["overlap"] == 0
                if "speaker_id" in frames.columns:
                    mask &= frames["speaker_id"] == int(ch["old"])
                frames.loc[mask, "speaker_id"] = int(ch["new"])
            frames_df = frames
        logger.info("Diarisation: fusion de %d flip(s) A-B-A.", total_merged)

    return tokens_df, frames_df, total_merged


def _refine_with_ecapa_microturns(
    tokens_df: pd.DataFrame, frames_df: pd.DataFrame | None, wav_path: Path, max_spk: int
) -> tuple[pd.DataFrame, pd.DataFrame | None, bool]:
    """Global ECAPA micro-turn refinement to recover quick alternations."""
    enabled = os.getenv("CALL2EDS_DIAR_MICRO_REFINE", "1").strip().lower() in ("1", "true", "yes")
    if not enabled or tokens_df.empty:
        return tokens_df, frames_df, False

    mode = os.getenv("CALL2EDS_DIAR_MICRO_MODE", "hybrid").strip().lower()
    candidates: list[tuple[str, List[Dict[str, float]]]] = []

    if mode in ("micro", "microturn", "tokens", "hybrid"):
        segments_micro = _ecapa_microturn_segments(tokens_df, wav_path, k_max=max_spk)
        if segments_micro:
            candidates.append(("micro", segments_micro))
    if mode in ("window", "hybrid"):
        try:
            win_s = float(os.getenv("CALL2EDS_DIAR_MICRO_WIN_S", "0.6"))
        except Exception:
            win_s = 0.6
        try:
            hop_s = float(os.getenv("CALL2EDS_DIAR_MICRO_HOP_S", "0.3"))
        except Exception:
            hop_s = 0.3
        try:
            smooth = int(os.getenv("CALL2EDS_DIAR_MICRO_SMOOTH", "0"))
        except Exception:
            smooth = 0
        segments_win = _ecapa_window_segments(
            wav_path,
            k_max=max_spk,
            win_s=win_s,
            hop_s=hop_s,
            smooth=smooth,
        )
        if segments_win:
            candidates.append(("window", segments_win))

    if not candidates:
        return tokens_df, frames_df, False

    try:
        min_delta = int(os.getenv("CALL2EDS_DIAR_MICRO_MIN_DELTA", "0"))
    except Exception:
        min_delta = 0

    def _transitions(ids: np.ndarray) -> int:
        seq = [int(x) for x in ids if int(x) >= 0]
        if len(seq) < 2:
            return 0
        cnt = 0
        prev = seq[0]
        for v in seq[1:]:
            if v != prev:
                cnt += 1
                prev = v
        return cnt

    def _speaker_count(ids: np.ndarray) -> int:
        return len({int(x) for x in ids if int(x) >= 0})

    def _map_labels(tokens_micro: pd.DataFrame, tokens_tmp: pd.DataFrame) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        try:
            ref = tokens_micro.merge(
                tokens_tmp[["_tok_idx", "speaker_id"]].rename(columns={"speaker_id": "orig_speaker_id"}),
                on="_tok_idx",
                how="left",
            )
            overlap_map: Dict[int, list[tuple[int, int]]] = {}
            for spk in ref["speaker_id"].dropna().unique():
                subset = ref[ref["speaker_id"] == spk]
                counts = subset["orig_speaker_id"].value_counts()
                overlap_map[int(spk)] = [(int(k), int(v)) for k, v in counts.items()]
                if counts.size > 0:
                    mapping[int(spk)] = int(counts.idxmax())

            # If mapping collapses all labels to the same speaker, force diversity when possible
            force_div = os.getenv("CALL2EDS_DIAR_MICRO_FORCE_DIVERSITY", "1").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            orig_speakers = [int(s) for s in ref["orig_speaker_id"].dropna().unique() if int(s) >= 0]
            if force_div and len(set(mapping.values())) < min(2, len(orig_speakers)) and len(overlap_map) >= 2:
                # pick the micro label with the strongest 2nd-best overlap and remap it
                best_candidate = None
                best_score = -1
                for micro_lab, pairs in overlap_map.items():
                    if len(pairs) < 2:
                        continue
                    pairs_sorted = sorted(pairs, key=lambda kv: kv[1], reverse=True)
                    second = pairs_sorted[1]
                    if second[1] > best_score and mapping.get(micro_lab) != second[0]:
                        best_score = second[1]
                        best_candidate = (micro_lab, second[0])
                if best_candidate is not None:
                    mapping[best_candidate[0]] = best_candidate[1]
        except Exception:
            mapping = {}
        return mapping

    base_ids = tokens_df["speaker_id"].to_numpy()
    base_trans = _transitions(base_ids)
    base_spk = _speaker_count(base_ids)

    scored: list[dict] = []
    for name, segments in candidates:
        tokens_tmp = tokens_df.copy()
        tokens_tmp["_tok_idx"] = range(len(tokens_tmp))
        tokens_micro = _assign_speakers_to_tokens(tokens_tmp, segments)
        mapping = _map_labels(tokens_micro, tokens_tmp)
        if mapping:
            tokens_micro["speaker_id"] = tokens_micro["speaker_id"].map(
                lambda x: mapping.get(int(x), int(x))
            )
        micro_ids = tokens_micro["speaker_id"].to_numpy()
        micro_trans = _transitions(micro_ids)
        micro_spk = _speaker_count(micro_ids)
        delta = micro_trans - base_trans
        scored.append(
            {
                "name": name,
                "segments": segments,
                "mapping": mapping,
                "tokens": tokens_micro,
                "delta": delta,
                "transitions": micro_trans,
                "speakers": micro_spk,
            }
        )

    accepted: list[dict] = []
    for cand in scored:
        if cand["transitions"] < base_trans:
            continue
        if cand["speakers"] < base_spk:
            continue
        if cand["speakers"] == base_spk and min_delta > 0 and cand["delta"] < min_delta:
            continue
        accepted.append(cand)
    if not accepted:
        return tokens_df, frames_df, False

    def _pref(name: str) -> int:
        return 1 if name == "window" else 0

    best = max(accepted, key=lambda c: (c["delta"], c["speakers"], _pref(c["name"])))

    mapped_series = best["tokens"].set_index("_tok_idx")["speaker_id"]
    tokens_df.loc[mapped_series.index, "speaker_id"] = mapped_series.to_numpy()

    if frames_df is not None and not frames_df.empty and "t" in frames_df.columns:
        segs = []
        mapping = best["mapping"]
        for s in best["segments"]:
            spk = int(s["speaker"])
            if mapping:
                spk = mapping.get(spk, spk)
            segs.append({"start": float(s["start"]), "end": float(s["end"]), "speaker": spk})
        frames_df = _assign_speakers_to_frames(frames_df, segs)

    logger.info(
        "Diarisation: raffinement ECAPA micro-tours appliqué (%s, Δ=%s, spk=%s).",
        best["name"],
        best["delta"],
        best["speakers"],
    )
    return tokens_df, frames_df, True


def _pyannote_window_segments(
    wav_path: Path, start_s: float, end_s: float, min_spk: int, max_spk: int
) -> Optional[List[Dict[str, float]]]:
    """Run pyannote diarization on a time window and offset segments."""
    if end_s <= start_s:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="call2eds_diar_") as td:
            clip = Path(td) / "clip.wav"
            audio.run_ffmpeg(
                [
                    "-ss",
                    f"{max(0.0, start_s):.3f}",
                    "-to",
                    f"{max(0.0, end_s):.3f}",
                    "-i",
                    str(wav_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(clip),
                ]
            )
            segs = diarize_segments_pyannote(clip, min_spk=min_spk, max_spk=max_spk)
            if not segs:
                return None
            for s in segs:
                s["start"] = float(s["start"]) + float(start_s)
                s["end"] = float(s["end"]) + float(start_s)
            return segs
    except Exception as exc:  # noqa: BLE001
        logger.warning("Diarisation fenêtre pyannote échouée: %s", exc)
        return None


def _maybe_refine_early_with_ecapa(
    tokens_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    wav_path: Path,
    max_spk: int,
    min_spk: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """If the beginning looks like 1 speaker but later has more, refine early tokens with ECAPA."""
    enabled = os.getenv("CALL2EDS_DIAR_EARLY_REPAIR", "1").strip().lower() in ("1", "true", "yes")
    if not enabled or tokens_df.empty:
        return tokens_df, frames_df, False
    try:
        early_s = float(os.getenv("CALL2EDS_DIAR_EARLY_WINDOW_S", "15"))
    except Exception:
        early_s = 15.0
    if early_s <= 0:
        return tokens_df, frames_df, False
    if "t_start" not in tokens_df.columns:
        return tokens_df, frames_df, False

    # Dynamic early block: if the first speaker lasts "too long", refine that block only.
    try:
        block_min_s = float(os.getenv("CALL2EDS_DIAR_EARLY_BLOCK_S", "6"))
    except Exception:
        block_min_s = 6.0
    tokens_sorted = tokens_df.sort_values("t_start")
    valid = tokens_sorted[tokens_sorted["speaker_id"] >= 0] if "speaker_id" in tokens_sorted.columns else tokens_sorted
    block_start = float(valid.iloc[0].t_start) if not valid.empty else 0.0
    block_end = None
    first_spk = int(valid.iloc[0].speaker_id) if not valid.empty and "speaker_id" in valid.columns else None
    if first_spk is not None:
        for _, row in valid.iterrows():
            if int(row.speaker_id) != first_spk:
                block_end = float(row.t_start)
                break
    if block_end is not None and (block_end - block_start) >= block_min_s:
        target_start, target_end = block_start, block_end
    else:
        target_start, target_end = 0.0, early_s

    mask = (tokens_df["t_start"] >= target_start) & (tokens_df["t_start"] < target_end)
    if not mask.any():
        return tokens_df, frames_df, False

    def _nuniq(df: pd.DataFrame) -> int:
        if df.empty or "speaker_id" not in df.columns:
            return 0
        return int(df.loc[df["speaker_id"] >= 0, "speaker_id"].nunique())

    total_spk = _nuniq(tokens_df)
    early_spk = _nuniq(tokens_df.loc[mask])
    target_min = max(2, int(min_spk))

    force_early = os.getenv("CALL2EDS_DIAR_EARLY_FORCE", "0").strip().lower() in ("1", "true", "yes")
    if force_early:
        try:
            early_kmax = int(os.getenv("CALL2EDS_DIAR_EARLY_MAX_SPK", "2"))
        except Exception:
            early_kmax = 2
        early_kmax = max(target_min, min(early_kmax, max_spk))
        tokens_df, frames_df, forced = _force_early_reseg_with_prototypes(
            tokens_df,
            frames_df,
            wav_path,
            start_s=target_start,
            end_s=target_end,
            min_spk=target_min,
            max_spk=early_kmax,
        )
        if forced:
            logger.info("Diarisation: resegmentation early forcée (pyannote+ECAPA).")
            return tokens_df, frames_df, True

    if total_spk < target_min or early_spk >= min(target_min, total_spk):
        return tokens_df, frames_df, False

    early_segments = _pyannote_window_segments(
        wav_path, start_s=target_start, end_s=target_end, min_spk=target_min, max_spk=target_min
    )
    if early_segments:
        tokens_py = _assign_speakers_to_tokens(tokens_df, early_segments)
        early_spk_py = _nuniq(tokens_py.loc[mask])
        if early_spk_py > early_spk:
            for col in ("speaker_id", "speaker_conf", "overlap_ratio", "speaker_alt"):
                if col in tokens_py.columns:
                    tokens_df.loc[mask, col] = tokens_py.loc[mask, col]
            if frames_df is not None and not frames_df.empty and "t" in frames_df.columns:
                fmask = (frames_df["t"] >= target_start) & (frames_df["t"] < target_end)
                if fmask.any():
                    frames_py = _assign_speakers_to_frames(frames_df, early_segments)
                    for col in ("speaker_id", "overlap"):
                        if col in frames_py.columns:
                            frames_df.loc[fmask, col] = frames_py.loc[fmask, col]
            return tokens_df, frames_df, True

    try:
        early_win = float(os.getenv("CALL2EDS_ECAPA_EARLY_WIN_S", "0.8"))
        early_hop = float(os.getenv("CALL2EDS_ECAPA_EARLY_HOP_S", "0.4"))
        early_smooth = int(os.getenv("CALL2EDS_ECAPA_EARLY_SMOOTH", "0"))
    except Exception:
        early_win, early_hop, early_smooth = 0.8, 0.4, 0
    try:
        early_kmax = int(os.getenv("CALL2EDS_DIAR_EARLY_MAX_SPK", "2"))
    except Exception:
        early_kmax = 2
    early_kmax = max(target_min, min(early_kmax, max_spk))
    ecapa_segments = _ecapa_window_segments(
        wav_path, k_max=early_kmax, win_s=early_win, hop_s=early_hop, smooth=early_smooth
    )
    if not ecapa_segments:
        ecapa_segments = None

    tokens_tmp = tokens_df.copy()
    tokens_tmp["_tok_idx"] = range(len(tokens_tmp))
    tokens_ecapa = _assign_speakers_to_tokens(tokens_tmp, ecapa_segments)
    # align ECAPA labels with existing labels using majority vote outside early window
    try:
        ref = tokens_ecapa.merge(
            tokens_tmp[["_tok_idx", "speaker_id"]].rename(columns={"speaker_id": "orig_speaker_id"}),
            on="_tok_idx",
            how="left",
        )
        ref_out = ref.loc[~mask]
        mapping: Dict[int, int] = {}
        if not ref_out.empty and "speaker_id" in ref_out.columns:
            for spk in ref_out["speaker_id"].dropna().unique():
                subset = ref_out[ref_out["speaker_id"] == spk]
                counts = subset["orig_speaker_id"].value_counts()
                if not counts.empty:
                    mapping[int(spk)] = int(counts.idxmax())
        if mapping:
            tokens_ecapa["speaker_id"] = tokens_ecapa["speaker_id"].map(
                lambda x: mapping.get(int(x), int(x))
            )
    except Exception:
        pass
    early_spk_ecapa = _nuniq(tokens_ecapa.loc[mask])
    if early_spk_ecapa > early_spk:
        try:
            min_ratio = float(os.getenv("CALL2EDS_DIAR_EARLY_MINORITY_RATIO", "0.15"))
        except Exception:
            min_ratio = 0.15
        subset = tokens_ecapa.loc[mask]
        subset = subset[subset["speaker_id"] >= 0]
        counts = subset["speaker_id"].value_counts()
        if len(counts) >= 2:
            minority_ratio = float(counts.iloc[-1]) / float(counts.sum())
            if minority_ratio < min_ratio:
                early_spk_ecapa = early_spk
    if early_spk_ecapa <= early_spk:
        # micro-turn ECAPA fallback (more sensitive to quick alternations)
        try:
            micro_gap = float(os.getenv("CALL2EDS_ECAPA_MICRO_EARLY_GAP_S", "0.2"))
            micro_max = float(os.getenv("CALL2EDS_ECAPA_MICRO_EARLY_MAX_S", "1.6"))
            micro_min = float(os.getenv("CALL2EDS_ECAPA_MICRO_EARLY_MIN_S", "0.5"))
        except Exception:
            micro_gap, micro_max, micro_min = 0.2, 1.6, 0.5
        micro_segments = _ecapa_microturn_segments(
            tokens_df,
            wav_path,
            k_max=early_kmax,
            t_max=target_end,
            micro_gap=micro_gap,
            max_dur=micro_max,
            min_dur=micro_min,
        )
        if not micro_segments:
            return tokens_df, frames_df, False
        tokens_micro = _assign_speakers_to_tokens(tokens_df, micro_segments)
        early_spk_micro = _nuniq(tokens_micro.loc[mask])
        if early_spk_micro <= early_spk:
            return tokens_df, frames_df, False
        for col in ("speaker_id", "speaker_conf", "overlap_ratio", "speaker_alt"):
            if col in tokens_micro.columns:
                tokens_df.loc[mask, col] = tokens_micro.loc[mask, col]
        if frames_df is not None and not frames_df.empty and "t" in frames_df.columns:
            fmask = (frames_df["t"] >= target_start) & (frames_df["t"] < target_end)
            if fmask.any():
                frames_micro = _assign_speakers_to_frames(frames_df, micro_segments)
                for col in ("speaker_id", "overlap"):
                    if col in frames_micro.columns:
                        frames_df.loc[fmask, col] = frames_micro.loc[fmask, col]
        return tokens_df, frames_df, True

    for col in ("speaker_id", "speaker_conf", "overlap_ratio", "speaker_alt"):
        if col in tokens_ecapa.columns:
            tokens_df.loc[mask, col] = tokens_ecapa.loc[mask, col]

    if frames_df is not None and not frames_df.empty and "t" in frames_df.columns:
        fmask = (frames_df["t"] >= target_start) & (frames_df["t"] < target_end)
        if fmask.any():
            frames_ecapa = _assign_speakers_to_frames(frames_df, ecapa_segments)
            for col in ("speaker_id", "overlap"):
                if col in frames_ecapa.columns:
                    frames_df.loc[fmask, col] = frames_ecapa.loc[fmask, col]

    return tokens_df, frames_df, True


def _force_early_reseg_with_prototypes(
    tokens_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    wav_path: Path,
    start_s: float,
    end_s: float,
    min_spk: int,
    max_spk: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Force early resegmentation with pyannote, map to global speaker prototypes via ECAPA."""
    segments = _pyannote_window_segments(wav_path, start_s=start_s, end_s=end_s, min_spk=min_spk, max_spk=max_spk)
    if not segments:
        return tokens_df, frames_df, False

    def _transitions(ids: np.ndarray) -> int:
        seq = [int(x) for x in ids if int(x) >= 0]
        if len(seq) < 2:
            return 0
        cnt = 0
        prev = seq[0]
        for v in seq[1:]:
            if v != prev:
                cnt += 1
                prev = v
        return cnt

    def _nuniq_ids(ids: np.ndarray) -> int:
        return len({int(x) for x in ids if int(x) >= 0})

    # Build speaker prototypes outside the early window if possible
    try:
        min_proto_s = float(os.getenv("CALL2EDS_DIAR_PROTO_MIN_S", "1.0"))
    except Exception:
        min_proto_s = 1.0
    try:
        max_proto = int(os.getenv("CALL2EDS_DIAR_PROTO_MAX_PER_SPK", "3"))
    except Exception:
        max_proto = 3
    prototypes, ecapa_ctx = _speaker_prototypes_from_tokens(
        tokens_df,
        wav_path,
        exclude_window=(start_s, end_s),
        min_dur=min_proto_s,
        max_per_spk=max_proto,
    )
    if len(prototypes) < 2:
        # fallback: use all tokens (may be noisier but better than nothing)
        prototypes, ecapa_ctx = _speaker_prototypes_from_tokens(
            tokens_df,
            wav_path,
            exclude_window=None,
            min_dur=min_proto_s,
            max_per_spk=max_proto,
        )

    def _speaker_durations(exclude: tuple[float, float] | None) -> Dict[int, float]:
        if tokens_df.empty:
            return {}
        toks = tokens_df.sort_values("t_start")
        durations: Dict[int, float] = {}
        cur_spk = int(toks.iloc[0].speaker_id) if pd.notna(toks.iloc[0].speaker_id) else -1
        cur_start = float(toks.iloc[0].t_start)
        cur_end = float(toks.iloc[0].t_end)
        for i in range(1, len(toks)):
            spk = int(toks.iloc[i].speaker_id) if pd.notna(toks.iloc[i].speaker_id) else -1
            if spk == cur_spk:
                cur_end = max(cur_end, float(toks.iloc[i].t_end))
            else:
                if cur_spk >= 0:
                    if exclude is None or (cur_end <= exclude[0] or cur_start >= exclude[1]):
                        durations[cur_spk] = durations.get(cur_spk, 0.0) + (cur_end - cur_start)
                cur_spk = spk
                cur_start = float(toks.iloc[i].t_start)
                cur_end = float(toks.iloc[i].t_end)
        if cur_spk >= 0:
            if exclude is None or (cur_end <= exclude[0] or cur_start >= exclude[1]):
                durations[cur_spk] = durations.get(cur_spk, 0.0) + (cur_end - cur_start)
        return durations

    # limit prototypes to top speakers by duration
    if len(prototypes) > max_spk:
        dur_map = _speaker_durations((start_s, end_s))
        if not dur_map:
            dur_map = _speaker_durations(None)
        keep = sorted(prototypes.keys(), key=lambda k: dur_map.get(k, 0.0), reverse=True)[:max_spk]
        prototypes = {k: prototypes[k] for k in keep if k in prototypes}

    mapped_segments = segments
    if prototypes and ecapa_ctx:
        try:
            win_s = float(os.getenv("CALL2EDS_DIAR_EARLY_PROTO_WIN_S", "0.4"))
        except Exception:
            win_s = 0.4
        try:
            hop_s = float(os.getenv("CALL2EDS_DIAR_EARLY_PROTO_HOP_S", "0.2"))
        except Exception:
            hop_s = 0.2
        try:
            smooth = int(os.getenv("CALL2EDS_DIAR_EARLY_PROTO_SMOOTH", "0"))
        except Exception:
            smooth = 0
        proto_segments = _prototype_window_segments(
            start_s=start_s,
            end_s=end_s,
            win_s=win_s,
            hop_s=hop_s,
            smooth=smooth,
            prototypes=prototypes,
            ecapa_ctx=ecapa_ctx,
        )
        if proto_segments:
            mapped_segments = proto_segments
        else:
            mapped_segments = _map_segments_to_prototypes(segments, prototypes, ecapa_ctx)

    tokens_new = _assign_speakers_to_tokens(tokens_df, mapped_segments)
    mask = (tokens_df["t_start"] >= start_s) & (tokens_df["t_start"] < end_s)
    base_ids = tokens_df.loc[mask, "speaker_id"].to_numpy()
    new_ids = tokens_new.loc[mask, "speaker_id"].to_numpy()

    if _transitions(new_ids) <= _transitions(base_ids) and _nuniq_ids(new_ids) <= _nuniq_ids(base_ids):
        return tokens_df, frames_df, False

    for col in ("speaker_id", "speaker_conf", "overlap_ratio", "speaker_alt"):
        if col in tokens_new.columns:
            tokens_df.loc[mask, col] = tokens_new.loc[mask, col]

    if frames_df is not None and not frames_df.empty and "t" in frames_df.columns:
        fmask = (frames_df["t"] >= start_s) & (frames_df["t"] < end_s)
        if fmask.any():
            frames_new = _assign_speakers_to_frames(frames_df, mapped_segments)
            for col in ("speaker_id", "overlap"):
                if col in frames_new.columns:
                    frames_df.loc[fmask, col] = frames_new.loc[fmask, col]
    return tokens_df, frames_df, True


def _refine_with_prototype_windows_global(
    tokens_df: pd.DataFrame,
    frames_df: pd.DataFrame | None,
    wav_path: Path,
    max_spk: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None, bool]:
    """Global resegmentation with ECAPA prototypes, applied selectively on unstable zones."""
    enabled = os.getenv("CALL2EDS_DIAR_GLOBAL_RESEG", "1").strip().lower() in ("1", "true", "yes")
    if not enabled or tokens_df.empty:
        return tokens_df, frames_df, False

    if "t_start" not in tokens_df.columns or "t_end" not in tokens_df.columns:
        return tokens_df, frames_df, False

    tokens_tmp = tokens_df.copy()
    tokens_tmp["_tok_idx"] = range(len(tokens_tmp))
    tokens_sorted = tokens_tmp.sort_values("t_start").reset_index(drop=True)
    if tokens_sorted.empty:
        return tokens_df, frames_df, False

    start_s = float(tokens_sorted["t_start"].min())
    end_s = float(tokens_sorted["t_end"].max())
    if end_s <= start_s:
        return tokens_df, frames_df, False

    try:
        min_proto_s = float(os.getenv("CALL2EDS_DIAR_PROTO_MIN_S", "1.0"))
    except Exception:
        min_proto_s = 1.0
    try:
        max_proto = int(os.getenv("CALL2EDS_DIAR_PROTO_MAX_PER_SPK", "3"))
    except Exception:
        max_proto = 3

    prototypes, ecapa_ctx = _speaker_prototypes_from_tokens(
        tokens_tmp,
        wav_path,
        exclude_window=None,
        min_dur=min_proto_s,
        max_per_spk=max_proto,
    )
    if len(prototypes) < 2 or ecapa_ctx is None:
        return tokens_df, frames_df, False

    try:
        max_spk_env = int(os.getenv("CALL2EDS_DIAR_GLOBAL_MAX_SPK", str(max_spk)))
    except Exception:
        max_spk_env = max_spk
    max_spk_env = max(1, min(max_spk_env, max_spk))

    def _speaker_durations() -> Dict[int, float]:
        durations: Dict[int, float] = {}
        toks = tokens_sorted
        cur_spk = int(toks.loc[0, "speaker_id"]) if pd.notna(toks.loc[0, "speaker_id"]) else -1
        cur_start = float(toks.loc[0, "t_start"])
        cur_end = float(toks.loc[0, "t_end"])
        for i in range(1, len(toks)):
            spk = int(toks.loc[i, "speaker_id"]) if pd.notna(toks.loc[i, "speaker_id"]) else -1
            if spk == cur_spk:
                cur_end = max(cur_end, float(toks.loc[i, "t_end"]))
            else:
                if cur_spk >= 0:
                    durations[cur_spk] = durations.get(cur_spk, 0.0) + (cur_end - cur_start)
                cur_spk = spk
                cur_start = float(toks.loc[i, "t_start"])
                cur_end = float(toks.loc[i, "t_end"])
        if cur_spk >= 0:
            durations[cur_spk] = durations.get(cur_spk, 0.0) + (cur_end - cur_start)
        return durations

    if len(prototypes) > max_spk_env:
        dur_map = _speaker_durations()
        keep = sorted(prototypes.keys(), key=lambda k: dur_map.get(k, 0.0), reverse=True)[:max_spk_env]
        prototypes = {k: prototypes[k] for k in keep if k in prototypes}

    try:
        win_s = float(os.getenv("CALL2EDS_DIAR_GLOBAL_PROTO_WIN_S", "0.4"))
    except Exception:
        win_s = 0.4
    try:
        hop_s = float(os.getenv("CALL2EDS_DIAR_GLOBAL_PROTO_HOP_S", "0.2"))
    except Exception:
        hop_s = 0.2
    try:
        smooth = int(os.getenv("CALL2EDS_DIAR_GLOBAL_PROTO_SMOOTH", "0"))
    except Exception:
        smooth = 0

    proto_segments = _prototype_window_segments(
        start_s=start_s,
        end_s=end_s,
        win_s=win_s,
        hop_s=hop_s,
        smooth=smooth,
        prototypes=prototypes,
        ecapa_ctx=ecapa_ctx,
    )
    if not proto_segments:
        return tokens_df, frames_df, False

    tokens_proto = _assign_speakers_to_tokens(tokens_tmp, proto_segments)
    tokens_proto = tokens_proto.set_index("_tok_idx").sort_index()

    n = len(tokens_df)
    orig_ids = tokens_df["speaker_id"].to_numpy() if "speaker_id" in tokens_df.columns else np.zeros(n, dtype=int)
    new_ids = tokens_proto["speaker_id"].to_numpy()
    changed = new_ids != orig_ids

    force = os.getenv("CALL2EDS_DIAR_GLOBAL_FORCE", "0").strip().lower() in ("1", "true", "yes")
    if force:
        update_mask = changed
    else:
        try:
            conf_max = float(os.getenv("CALL2EDS_DIAR_GLOBAL_MAX_CONF", "0.85"))
        except Exception:
            conf_max = 0.85
        try:
            ov_max = float(os.getenv("CALL2EDS_DIAR_GLOBAL_MAX_OV", "0.2"))
        except Exception:
            ov_max = 0.2
        try:
            short_s = float(os.getenv("CALL2EDS_DIAR_GLOBAL_SHORT_S", "0.6"))
        except Exception:
            short_s = 0.6
        try:
            short_tokens = int(os.getenv("CALL2EDS_DIAR_GLOBAL_SHORT_TOKENS", "4"))
        except Exception:
            short_tokens = 4

        confs = (
            tokens_df["speaker_conf"].to_numpy()
            if "speaker_conf" in tokens_df.columns
            else np.ones(n, dtype=float)
        )
        overlaps = (
            tokens_df["overlap_ratio"].to_numpy()
            if "overlap_ratio" in tokens_df.columns
            else np.zeros(n, dtype=float)
        )
        low_conf = confs < conf_max
        high_ov = overlaps >= ov_max
        unknown = orig_ids < 0

        short_mask = np.zeros(n, dtype=bool)
        if short_s > 0 or short_tokens > 0:
            cur_spk = int(tokens_sorted.loc[0, "speaker_id"]) if pd.notna(tokens_sorted.loc[0, "speaker_id"]) else -1
            cur_start = float(tokens_sorted.loc[0, "t_start"])
            cur_end = float(tokens_sorted.loc[0, "t_end"])
            cur_idxs = [int(tokens_sorted.loc[0, "_tok_idx"])]
            for i in range(1, len(tokens_sorted)):
                spk = int(tokens_sorted.loc[i, "speaker_id"]) if pd.notna(tokens_sorted.loc[i, "speaker_id"]) else -1
                if spk == cur_spk:
                    cur_end = max(cur_end, float(tokens_sorted.loc[i, "t_end"]))
                    cur_idxs.append(int(tokens_sorted.loc[i, "_tok_idx"]))
                else:
                    dur = cur_end - cur_start
                    if (short_s > 0 and dur <= short_s) or (short_tokens > 0 and len(cur_idxs) <= short_tokens):
                        short_mask[cur_idxs] = True
                    cur_spk = spk
                    cur_start = float(tokens_sorted.loc[i, "t_start"])
                    cur_end = float(tokens_sorted.loc[i, "t_end"])
                    cur_idxs = [int(tokens_sorted.loc[i, "_tok_idx"])]
            dur = cur_end - cur_start
            if (short_s > 0 and dur <= short_s) or (short_tokens > 0 and len(cur_idxs) <= short_tokens):
                short_mask[cur_idxs] = True

        update_mask = changed & (low_conf | high_ov | short_mask | unknown)

        if not update_mask.any():
            return tokens_df, frames_df, False

        try:
            max_change = float(os.getenv("CALL2EDS_DIAR_GLOBAL_MAX_CHANGE", "0.45"))
        except Exception:
            max_change = 0.45
        if max_change > 0:
            change_ratio = float(update_mask.sum()) / float(max(1, n))
            if change_ratio > max_change:
                return tokens_df, frames_df, False

        try:
            max_drop = int(os.getenv("CALL2EDS_DIAR_GLOBAL_MAX_DROP", "1"))
        except Exception:
            max_drop = 1
        base_spk = len({int(x) for x in orig_ids if int(x) >= 0})
        final_ids = orig_ids.copy()
        final_ids[update_mask] = new_ids[update_mask]
        new_spk = len({int(x) for x in final_ids if int(x) >= 0})
        if base_spk >= 2 and new_spk < 2:
            return tokens_df, frames_df, False
        if base_spk - new_spk > max_drop:
            return tokens_df, frames_df, False

    for col in ("speaker_id", "speaker_conf", "overlap_ratio", "speaker_alt"):
        if col in tokens_proto.columns:
            base_vals = tokens_df[col].to_numpy() if col in tokens_df.columns else np.zeros(n)
            new_vals = tokens_proto[col].to_numpy()
            base_vals[update_mask] = new_vals[update_mask]
            tokens_df[col] = base_vals

    if frames_df is not None and not frames_df.empty and "t" in frames_df.columns:
        frames_new = _assign_speakers_to_frames(frames_df, proto_segments)
        if force:
            frames_df = frames_new
        else:
            try:
                merge_gap = float(os.getenv("CALL2EDS_DIAR_GLOBAL_FRAME_GAP_S", "0.05"))
            except Exception:
                merge_gap = 0.05
            spans: list[tuple[float, float]] = []
            updated_tokens = tokens_tmp.loc[update_mask].sort_values("t_start")
            for _, row in updated_tokens.iterrows():
                s0 = float(row.t_start)
                s1 = float(row.t_end)
                if not spans:
                    spans.append((s0, s1))
                    continue
                last_s0, last_s1 = spans[-1]
                if s0 <= last_s1 + merge_gap:
                    spans[-1] = (last_s0, max(last_s1, s1))
                else:
                    spans.append((s0, s1))
            frames = frames_df.copy()
            for s0, s1 in spans:
                fmask = (frames["t"] >= s0) & (frames["t"] <= s1)
                for col in ("speaker_id", "overlap"):
                    if col in frames_new.columns:
                        frames.loc[fmask, col] = frames_new.loc[fmask, col]
            frames_df = frames

    logger.info(
        "Diarisation: resegmentation globale prototypes appliquée (%d tokens).",
        int(update_mask.sum()),
    )
    return tokens_df, frames_df, True


def _speaker_prototypes_from_tokens(
    tokens_df: pd.DataFrame,
    wav_path: Path,
    exclude_window: tuple[float, float] | None,
    min_dur: float,
    max_per_spk: int,
) -> tuple[Dict[int, np.ndarray], dict | None]:
    """Compute ECAPA speaker prototypes from token runs."""
    if tokens_df.empty:
        return {}, None

    tokens = tokens_df.sort_values("t_start").reset_index(drop=True)
    runs: list[dict] = []
    cur_spk = int(tokens.loc[0, "speaker_id"]) if pd.notna(tokens.loc[0, "speaker_id"]) else -1
    cur_start = float(tokens.loc[0, "t_start"])
    cur_end = float(tokens.loc[0, "t_end"])
    cur_ov = [float(tokens.loc[0, "overlap_ratio"]) if "overlap_ratio" in tokens.columns else 0.0]
    cur_conf = [float(tokens.loc[0, "speaker_conf"]) if "speaker_conf" in tokens.columns else 0.0]
    for i in range(1, len(tokens)):
        spk = int(tokens.loc[i, "speaker_id"]) if pd.notna(tokens.loc[i, "speaker_id"]) else -1
        if spk == cur_spk:
            cur_end = max(cur_end, float(tokens.loc[i, "t_end"]))
            if "overlap_ratio" in tokens.columns:
                cur_ov.append(float(tokens.loc[i, "overlap_ratio"]))
            if "speaker_conf" in tokens.columns:
                cur_conf.append(float(tokens.loc[i, "speaker_conf"]))
        else:
            runs.append(
                {
                    "speaker_id": cur_spk,
                    "start": cur_start,
                    "end": cur_end,
                    "overlap_mean": float(np.mean(cur_ov)) if cur_ov else 0.0,
                    "conf_mean": float(np.mean(cur_conf)) if cur_conf else None,
                }
            )
            cur_spk = spk
            cur_start = float(tokens.loc[i, "t_start"])
            cur_end = float(tokens.loc[i, "t_end"])
            cur_ov = [float(tokens.loc[i, "overlap_ratio"]) if "overlap_ratio" in tokens.columns else 0.0]
            cur_conf = [float(tokens.loc[i, "speaker_conf"]) if "speaker_conf" in tokens.columns else 0.0]
    runs.append(
        {
            "speaker_id": cur_spk,
            "start": cur_start,
            "end": cur_end,
            "overlap_mean": float(np.mean(cur_ov)) if cur_ov else 0.0,
            "conf_mean": float(np.mean(cur_conf)) if cur_conf else None,
        }
    )

    if exclude_window is not None:
        s0, s1 = exclude_window
        runs = [r for r in runs if r["end"] <= s0 or r["start"] >= s1]

    try:
        max_ov = float(os.getenv("CALL2EDS_DIAR_PROTO_MAX_OV", "0.2"))
    except Exception:
        max_ov = 0.2
    try:
        min_conf = float(os.getenv("CALL2EDS_DIAR_PROTO_MIN_CONF", "0.6"))
    except Exception:
        min_conf = 0.6

    ecapa_ctx = _ecapa_prepare(wav_path)
    if ecapa_ctx is None:
        return {}, None

    by_spk: Dict[int, list[tuple[float, float]]] = {}
    for r in runs:
        spk = int(r["speaker_id"])
        if spk < 0:
            continue
        dur = float(r["end"] - r["start"])
        if dur < min_dur:
            continue
        if r["overlap_mean"] >= max_ov:
            continue
        if r["conf_mean"] is not None and r["conf_mean"] < min_conf:
            continue
        by_spk.setdefault(spk, []).append((float(r["start"]), float(r["end"])))

    prototypes: Dict[int, np.ndarray] = {}
    for spk, spans in by_spk.items():
        # prefer longest runs
        spans_sorted = sorted(spans, key=lambda se: (se[1] - se[0]), reverse=True)[: max_per_spk]
        embs = _ecapa_embeddings_for_segments(ecapa_ctx, spans_sorted)
        if embs:
            prototypes[spk] = np.mean(np.stack(embs), axis=0)
    return prototypes, ecapa_ctx


def _ecapa_prepare(wav_path: Path) -> dict | None:
    """Load ECAPA classifier and waveform once for embedding extraction."""
    try:
        import inspect
        import torch
        import torchaudio
        import huggingface_hub
        from huggingface_hub import snapshot_download
        from speechbrain.pretrained import EncoderClassifier
    except Exception:
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        token = os.getenv("HF_TOKEN")
        if not token:
            token_file = Path.home() / ".cache" / "hf_token"
            if token_file.exists():
                token = token_file.read_text().strip()
        if "use_auth_token" not in inspect.signature(huggingface_hub.hf_hub_download).parameters:
            _orig_hf_hub_download = huggingface_hub.hf_hub_download

            def _hf_hub_download_compat(*args, **kwargs):
                if "use_auth_token" in kwargs and "token" not in kwargs:
                    kwargs["token"] = kwargs.pop("use_auth_token")
                else:
                    kwargs.pop("use_auth_token", None)
                return _orig_hf_hub_download(*args, **kwargs)

            huggingface_hub.hf_hub_download = _hf_hub_download_compat

        local_dir = snapshot_download(repo_id="speechbrain/spkrec-ecapa-voxceleb", token=token)
        classifier = EncoderClassifier.from_hparams(
            source=local_dir,
            savedir=local_dir,
            run_opts={"device": device},
        )
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        # normalize amplitude
        try:
            rms = torch.sqrt(torch.mean(waveform**2))
            if float(rms) > 0:
                waveform = waveform / rms
        except Exception:
            pass
        return {"classifier": classifier, "waveform": waveform, "sr": sr}
    except Exception:
        return None


def _ecapa_embeddings_for_segments(ecapa_ctx: dict, segments: list[tuple[float, float]]) -> list[np.ndarray]:
    """Compute ECAPA embeddings for segments using a prepared context."""
    if not ecapa_ctx or not segments:
        return []
    classifier = ecapa_ctx["classifier"]
    waveform = ecapa_ctx["waveform"]
    sr = int(ecapa_ctx["sr"])
    total_dur = waveform.shape[1] / sr
    try:
        min_dur = float(os.getenv("CALL2EDS_DIAR_PROTO_MIN_DUR", "0.6"))
    except Exception:
        min_dur = 0.6

    embs: list[np.ndarray] = []
    for s0, s1 in segments:
        dur = float(s1 - s0)
        if dur < min_dur:
            pad = 0.5 * (min_dur - dur)
            s0 = max(0.0, s0 - pad)
            s1 = min(total_dur, s1 + pad)
        start = int(max(0.0, s0) * sr)
        end = int(min(total_dur, s1) * sr)
        if end - start < int(0.3 * sr):
            continue
        seg = waveform[:, start:end]
        emb = classifier.encode_batch(seg).detach().cpu().squeeze().numpy().reshape(-1)
        embs.append(emb)
    return embs


def _map_segments_to_prototypes(
    segments: List[Dict[str, float]],
    prototypes: Dict[int, np.ndarray],
    ecapa_ctx: dict | None,
) -> List[Dict[str, float]]:
    """Assign each segment to nearest prototype (cosine similarity)."""
    if not segments or not prototypes or ecapa_ctx is None:
        return segments
    protos = {k: v for k, v in prototypes.items() if v is not None}
    if not protos:
        return segments
    proto_ids = list(protos.keys())
    proto_vecs = [protos[k] for k in proto_ids]

    seg_spans = [(float(s["start"]), float(s["end"])) for s in segments]
    seg_embs = _ecapa_embeddings_for_segments(ecapa_ctx, seg_spans)
    if len(seg_embs) != len(segments):
        return segments

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den <= 1e-8:
            return -1.0
        return float(np.dot(a, b) / den)

    mapped: List[Dict[str, float]] = []
    for seg, emb in zip(segments, seg_embs):
        sims = [_cos(emb, pv) for pv in proto_vecs]
        best_idx = int(np.argmax(sims)) if sims else 0
        mapped.append({"start": float(seg["start"]), "end": float(seg["end"]), "speaker": int(proto_ids[best_idx])})
    return mapped


def _prototype_window_segments(
    start_s: float,
    end_s: float,
    win_s: float,
    hop_s: float,
    smooth: int,
    prototypes: Dict[int, np.ndarray],
    ecapa_ctx: dict,
) -> List[Dict[str, float]] | None:
    """Sliding-window ECAPA assignment to known speaker prototypes."""
    if not prototypes or ecapa_ctx is None:
        return None
    win_s = max(0.3, float(win_s))
    hop_s = max(0.1, float(hop_s))
    if end_s - start_s < win_s * 2:
        return None

    proto_ids = list(prototypes.keys())
    proto_vecs = [prototypes[k] for k in proto_ids]

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den <= 1e-8:
            return -1.0
        return float(np.dot(a, b) / den)

    times: list[tuple[float, float]] = []
    labels: list[int | None] = []
    t = float(start_s)
    while t + win_s <= end_s:
        seg = (t, t + win_s)
        emb = _ecapa_embeddings_for_segments(ecapa_ctx, [seg])
        if emb:
            sims = [_cos(emb[0], pv) for pv in proto_vecs]
            lab = int(np.argmax(sims)) if sims else 0
        else:
            lab = None
        times.append(seg)
        labels.append(lab)
        t += hop_s
    if len(labels) < 2 or all(l is None for l in labels):
        return None

    # fill missing labels by nearest known (forward/backfill)
    last = None
    for i in range(len(labels)):
        if labels[i] is None:
            labels[i] = last
        else:
            last = labels[i]
    last = None
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] is None:
            labels[i] = last
        else:
            last = labels[i]
    labels = [0 if l is None else int(l) for l in labels]

    if smooth > 0:
        new_labels = labels.copy()
        for i in range(len(labels)):
            left = max(0, i - smooth)
            right = min(len(labels), i + smooth + 1)
            window = labels[left:right]
            vals, counts = np.unique(window, return_counts=True)
            new_labels[i] = int(vals[np.argmax(counts)])
        labels = new_labels

    centers = [0.5 * (t0 + t1) for t0, t1 in times]
    boundaries = [times[0][0]]
    for i in range(1, len(centers)):
        boundaries.append(0.5 * (centers[i - 1] + centers[i]))
    boundaries.append(times[-1][1])

    segments: List[Dict[str, float]] = []
    for i, lab in enumerate(labels):
        s0 = float(boundaries[i])
        s1 = float(boundaries[i + 1])
        if s1 <= s0:
            continue
        spk = int(proto_ids[lab])
        segments.append({"start": s0, "end": s1, "speaker": spk})
    return segments


def _rebuild_turns_from_tokens(tokens_df: pd.DataFrame, gap_s: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild turns by grouping consecutive tokens with same speaker and small gaps."""
    if tokens_df.empty:
        return tokens_df, pd.DataFrame(columns=["turn_id", "speaker_id", "t_start", "t_end", "text", "asr_conf_turn"])
    tokens = tokens_df.sort_values("t_start").copy()
    new_tokens = []
    turns = []
    current_words: List[str] = []
    current_start = None
    current_end = None
    current_speaker = None
    current_conf = []
    current_overlap = []
    turn_id = 0
    token_id = 0

    def _flush():
        nonlocal turn_id, token_id, current_words, current_start, current_end, current_speaker, current_conf, current_overlap
        if current_start is None or current_end is None:
            return
        text = " ".join(w for w in current_words if w)
        conf = float(np.mean(current_conf)) if current_conf else 0.0
        ov = float(np.mean(current_overlap)) if current_overlap else 0.0
        turns.append(
            {
                "turn_id": turn_id,
                "speaker_id": int(current_speaker or 0),
                "t_start": float(current_start),
                "t_end": float(current_end),
                "text": text.strip(),
                "asr_conf_turn": conf,
                "overlap_ratio": ov,
            }
        )
        turn_id += 1
        token_id = 0

    prev_end = None
    for _, tok in tokens.iterrows():
        spk = int(tok.get("speaker_id", 0) or 0)
        t0 = float(tok.t_start)
        t1 = float(tok.t_end)
        gap = (t0 - prev_end) if prev_end is not None else 0.0
        new_turn = (
            current_speaker is None
            or spk != current_speaker
            or (gap_s is not None and gap > gap_s)
        )
        if new_turn:
            _flush()
            current_words = []
            current_conf = []
            current_overlap = []
            current_start = t0
            current_end = t1
            current_speaker = spk
        else:
            current_end = max(current_end, t1)
        new_tokens.append(
            {
                **tok.to_dict(),
                "turn_id": turn_id,
                "token_id": token_id,
            }
        )
        token_id += 1
        current_words.append(str(tok.word))
        current_conf.append(float(tok.asr_conf_word) if "asr_conf_word" in tok else 0.0)
        if "overlap_ratio" in tok:
            try:
                current_overlap.append(float(tok.overlap_ratio))
            except Exception:
                current_overlap.append(0.0)
        prev_end = t1
    _flush()
    return pd.DataFrame(new_tokens), pd.DataFrame(turns)
def _cluster_speakers_mono(wav_path: Path, turns_df: pd.DataFrame, k: int = 2, k_max: int = 6) -> pd.Series:
    """
    Heuristique légère de diarisation mono :
    - extrait un vecteur MFCC moyen par tour
    - clusterise en k groupes (par défaut 2)
    - renvoie une série des labels alignés sur turns_df.index
    Si trop peu de tours ou échec, renvoie des zéros.
    """
    if turns_df.empty or len(turns_df) < 2:
        return pd.Series([0] * len(turns_df), index=turns_df.index, dtype=int)
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        feats = []
        idx = []
        for i, row in turns_df.iterrows():
            s = int(row.t_start * sr)
            e = int(row.t_end * sr)
            if e - s < int(0.3 * sr):  # ignore segments <300 ms
                continue
            seg = y[s:e]
            if seg.size == 0:
                continue
            mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
            feats.append(mfcc.mean(axis=1))
            idx.append(i)
        if len(feats) < 2:
            return pd.Series([0] * len(turns_df), index=turns_df.index, dtype=int)
        # auto k (<=k_max) via silhouette
        feats_np = np.stack(feats)
        labels = _auto_cluster(feats_np, k_max=min(k_max, len(feats)))
        speaker_labels = pd.Series([0] * len(turns_df), index=turns_df.index, dtype=int)
        for i, lab in zip(idx, labels):
            speaker_labels.loc[i] = int(lab)
        return speaker_labels
    except Exception as exc:  # noqa: BLE001
        logger.warning("Diarisation mono échouée: %s", exc)
        return pd.Series([0] * len(turns_df), index=turns_df.index, dtype=int)


def _auto_cluster(embeddings: np.ndarray, k_max: int = 6) -> np.ndarray:
    """Choisit automatiquement k (2..k_max) via silhouette cosine."""
    if embeddings.shape[0] < 2:
        return np.zeros(len(embeddings), dtype=int)
    best_score = -1
    best_labels = None
    for k in range(2, min(k_max, len(embeddings)) + 1):
        labels = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(embeddings)
        try:
            score = silhouette_score(embeddings, labels, metric="cosine")
        except Exception:
            score = -1
        if score > best_score:
            best_score = score
            best_labels = labels
    if best_labels is None:
        best_labels = np.zeros(len(embeddings), dtype=int)
    return best_labels


def _ecapa_window_segments(
    wav_path: Path,
    k_max: int = 6,
    start_s: float | None = None,
    end_s: float | None = None,
    win_s: float | None = None,
    hop_s: float | None = None,
    smooth: int | None = None,
) -> List[Dict[str, float]] | None:
    """Segments dérivés d'ECAPA (fenêtres glissantes + clustering)."""
    try:
        import inspect
        import torch
        import torchaudio
        import huggingface_hub
        from huggingface_hub import snapshot_download
        from speechbrain.pretrained import EncoderClassifier
    except Exception as exc:  # noqa: BLE001
        logger.warning("ECAPA indisponible: %s", exc)
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        token = os.getenv("HF_TOKEN")
        if not token:
            token_file = Path.home() / ".cache" / "hf_token"
            if token_file.exists():
                token = token_file.read_text().strip()
        if "use_auth_token" not in inspect.signature(huggingface_hub.hf_hub_download).parameters:
            _orig_hf_hub_download = huggingface_hub.hf_hub_download

            def _hf_hub_download_compat(*args, **kwargs):
                if "use_auth_token" in kwargs and "token" not in kwargs:
                    kwargs["token"] = kwargs.pop("use_auth_token")
                else:
                    kwargs.pop("use_auth_token", None)
                return _orig_hf_hub_download(*args, **kwargs)

            huggingface_hub.hf_hub_download = _hf_hub_download_compat

        local_dir = snapshot_download(repo_id="speechbrain/spkrec-ecapa-voxceleb", token=token)
        classifier = EncoderClassifier.from_hparams(
            source=local_dir,
            savedir=local_dir,
            run_opts={"device": device},
        )
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        total_dur = waveform.shape[1] / sr
        start_s = max(0.0, float(start_s)) if start_s is not None else 0.0
        end_s = min(float(end_s), total_dur) if end_s is not None else total_dur
        if end_s <= start_s:
            return None

        if win_s is None:
            try:
                win_s = float(os.getenv("CALL2EDS_ECAPA_WIN_S", "1.2"))
            except Exception:
                win_s = 1.2
        if hop_s is None:
            try:
                hop_s = float(os.getenv("CALL2EDS_ECAPA_HOP_S", "0.6"))
            except Exception:
                hop_s = 0.6
        win_s = max(0.4, float(win_s))
        hop_s = max(0.2, float(hop_s))
        win = int(win_s * sr)
        hop = int(hop_s * sr)
        start_idx = int(start_s * sr)
        end_idx = int(end_s * sr)
        if end_idx - start_idx < win * 2:
            return None
        waveform = waveform[:, start_idx:end_idx]
        # normalize amplitude to stabilize embeddings on low volume audio
        try:
            rms = torch.sqrt(torch.mean(waveform**2))
            if float(rms) > 0:
                waveform = waveform / rms
        except Exception:
            pass
        embeddings = []
        times = []
        for start in range(0, waveform.shape[1] - win, hop):
            seg = waveform[:, start : start + win]
            emb = classifier.encode_batch(seg).detach().cpu().squeeze().numpy().reshape(-1)
            embeddings.append(emb)
            times.append((start / sr + start_s, (start + win) / sr + start_s))
        if len(embeddings) < 2:
            return None
        embeddings = np.stack(embeddings)
        labels = _auto_cluster(embeddings, k_max=k_max)

        if smooth is None:
            smooth = int(os.getenv("CALL2EDS_ECAPA_SMOOTH", "1"))
        if int(smooth) > 0:
            new_labels = labels.copy()
            for i in range(len(labels)):
                left = max(0, i - smooth)
                right = min(len(labels), i + smooth + 1)
                window = labels[left:right]
                vals, counts = np.unique(window, return_counts=True)
                new_labels[i] = int(vals[np.argmax(counts)])
            labels = new_labels

        centers = [0.5 * (t0 + t1) for t0, t1 in times]
        boundaries = [times[0][0]]
        for i in range(1, len(centers)):
            boundaries.append(0.5 * (centers[i - 1] + centers[i]))
        boundaries.append(times[-1][1])
        segments: List[Dict[str, float]] = []
        for i, lab in enumerate(labels):
            s0 = float(boundaries[i])
            s1 = float(boundaries[i + 1])
            if s1 <= s0:
                continue
            segments.append({"start": s0, "end": s1, "speaker": int(lab)})
        return segments
    except Exception as exc:  # noqa: BLE001
        logger.warning("Diarisation ECAPA échouée: %s", exc)
        return None


def _ecapa_microturn_segments(
    tokens_df: pd.DataFrame,
    wav_path: Path,
    k_max: int = 6,
    t_max: float | None = None,
    micro_gap: float | None = None,
    max_dur: float | None = None,
    min_dur: float | None = None,
) -> List[Dict[str, float]] | None:
    """Segments dérivés d'ECAPA sur micro-tours (tokens regroupés)."""
    if tokens_df.empty or "t_start" not in tokens_df.columns or "t_end" not in tokens_df.columns:
        return None
    if micro_gap is None:
        try:
            micro_gap = float(os.getenv("CALL2EDS_ECAPA_MICRO_GAP_S", "0.25"))
        except Exception:
            micro_gap = 0.25
    if max_dur is None:
        try:
            max_dur = float(os.getenv("CALL2EDS_ECAPA_MICRO_MAX_S", "3.0"))
        except Exception:
            max_dur = 3.0
    if min_dur is None:
        try:
            min_dur = float(os.getenv("CALL2EDS_ECAPA_MICRO_MIN_S", "0.6"))
        except Exception:
            min_dur = 0.6
    micro_gap = float(micro_gap)
    max_dur = float(max_dur)
    min_dur = float(min_dur)

    toks = tokens_df.sort_values("t_start")
    if t_max is not None:
        toks = toks[toks["t_start"] < t_max]
    if len(toks) < 2:
        return None

    microturns: List[tuple[float, float]] = []
    cur_start = None
    cur_end = None
    prev_end = None
    for _, tok in toks.iterrows():
        t0 = float(tok.t_start)
        t1 = float(tok.t_end)
        if cur_start is None:
            cur_start, cur_end = t0, t1
        else:
            gap = t0 - (prev_end if prev_end is not None else cur_end)
            dur = (cur_end - cur_start) if cur_start is not None else 0.0
            if gap > micro_gap or dur >= max_dur:
                microturns.append((cur_start, cur_end))
                cur_start, cur_end = t0, t1
            else:
                cur_end = max(cur_end, t1)
        prev_end = t1
    if cur_start is not None and cur_end is not None:
        microturns.append((cur_start, cur_end))
    if len(microturns) < 2:
        return None

    try:
        import inspect
        import torch
        import torchaudio
        import huggingface_hub
        from huggingface_hub import snapshot_download
        from speechbrain.pretrained import EncoderClassifier
    except Exception as exc:  # noqa: BLE001
        logger.warning("ECAPA indisponible: %s", exc)
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        token = os.getenv("HF_TOKEN")
        if not token:
            token_file = Path.home() / ".cache" / "hf_token"
            if token_file.exists():
                token = token_file.read_text().strip()
        if "use_auth_token" not in inspect.signature(huggingface_hub.hf_hub_download).parameters:
            _orig_hf_hub_download = huggingface_hub.hf_hub_download

            def _hf_hub_download_compat(*args, **kwargs):
                if "use_auth_token" in kwargs and "token" not in kwargs:
                    kwargs["token"] = kwargs.pop("use_auth_token")
                else:
                    kwargs.pop("use_auth_token", None)
                return _orig_hf_hub_download(*args, **kwargs)

            huggingface_hub.hf_hub_download = _hf_hub_download_compat

        local_dir = snapshot_download(repo_id="speechbrain/spkrec-ecapa-voxceleb", token=token)
        classifier = EncoderClassifier.from_hparams(
            source=local_dir,
            savedir=local_dir,
            run_opts={"device": device},
        )
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        # normalize amplitude to stabilize embeddings on low volume audio
        try:
            rms = torch.sqrt(torch.mean(waveform**2))
            if float(rms) > 0:
                waveform = waveform / rms
        except Exception:
            pass
        total_dur = waveform.shape[1] / sr

        embeddings = []
        spans = []
        for (s0, s1) in microturns:
            dur = s1 - s0
            if dur < min_dur:
                pad = 0.5 * (min_dur - dur)
                s0 = max(0.0, s0 - pad)
                s1 = min(total_dur, s1 + pad)
            start = int(max(0.0, s0) * sr)
            end = int(min(total_dur, s1) * sr)
            if end - start < int(0.3 * sr):
                continue
            seg = waveform[:, start:end]
            emb = classifier.encode_batch(seg).detach().cpu().squeeze().numpy().reshape(-1)
            embeddings.append(emb)
            spans.append((s0, s1))
        if len(embeddings) < 2:
            return None
        embeddings = np.stack(embeddings)
        labels = _auto_cluster(embeddings, k_max=k_max)
        segments: List[Dict[str, float]] = []
        for (s0, s1), lab in zip(spans, labels):
            if s1 <= s0:
                continue
            segments.append({"start": float(s0), "end": float(s1), "speaker": int(lab)})
        return segments
    except Exception as exc:  # noqa: BLE001
        logger.warning("Diarisation ECAPA micro-tours échouée: %s", exc)
        return None


def _diarize_speechbrain(wav_path: Path, turns_df: pd.DataFrame, k_max: int = 2) -> pd.Series | None:
    """
    Diarisation plus robuste via embeddings ECAPA (speechbrain) + clustering agglomératif.
    On découpe l'audio en fenêtres glissantes, clusterise, puis assigne chaque tour au label majoritaire.
    Retourne None si échec (lib non dispo, etc.).
    """
    try:
        import inspect
        import torch
        import torchaudio
        import huggingface_hub
        from huggingface_hub import snapshot_download
        from speechbrain.pretrained import EncoderClassifier
    except Exception as exc:  # noqa: BLE001
        logger.warning("speechbrain indisponible: %s", exc)
        return None
    if turns_df.empty:
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        token = os.getenv("HF_TOKEN")
        if not token:
            token_file = Path.home() / ".cache" / "hf_token"
            if token_file.exists():
                token = token_file.read_text().strip()
        # compat HF Hub: map legacy use_auth_token -> token
        if "use_auth_token" not in inspect.signature(huggingface_hub.hf_hub_download).parameters:
            _orig_hf_hub_download = huggingface_hub.hf_hub_download

            def _hf_hub_download_compat(*args, **kwargs):
                if "use_auth_token" in kwargs and "token" not in kwargs:
                    kwargs["token"] = kwargs.pop("use_auth_token")
                else:
                    kwargs.pop("use_auth_token", None)
                return _orig_hf_hub_download(*args, **kwargs)

            huggingface_hub.hf_hub_download = _hf_hub_download_compat

        try:
            local_dir = snapshot_download(repo_id="speechbrain/spkrec-ecapa-voxceleb", token=token)
        except Exception as exc:  # noqa: BLE001
            logger.warning("speechbrain download échoué: %s", exc)
            return None
        classifier = EncoderClassifier.from_hparams(
            source=local_dir,
            savedir=local_dir,
            run_opts={"device": device},
        )
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        win = int(1.5 * sr)
        hop = int(0.75 * sr)
        embeddings = []
        times = []
        for start in range(0, waveform.shape[1] - win, hop):
            seg = waveform[:, start : start + win]
            emb = classifier.encode_batch(seg).detach().cpu().squeeze().numpy().reshape(-1)
            embeddings.append(emb)
            times.append((start / sr, (start + win) / sr))
        if len(embeddings) < 2:
            return None
        embeddings = np.stack(embeddings)
        labels = _auto_cluster(embeddings, k_max=k_max)

        # assignation par majorité des fenêtres qui chevauchent le tour
        speaker_labels = []
        for _, row in turns_df.iterrows():
            overlaps = []
            for (t0, t1), lab in zip(times, labels):
                if t1 < row.t_start or t0 > row.t_end:
                    continue
                overlaps.append(lab)
            if overlaps:
                counts = np.bincount(overlaps)
                speaker_labels.append(int(np.argmax(counts)))
            else:
                speaker_labels.append(0)
        return pd.Series(speaker_labels, index=turns_df.index, dtype=int)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Diarisation speechbrain échouée: %s", exc)
        return None
