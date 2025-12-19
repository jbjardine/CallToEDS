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
from call2eds.pipeline.diarization import diarize_pyannote
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
PIPELINE_VERSION = "0.1.1"


def ingest_call(
    audio_path: str,
    call_id: Optional[str],
    language: str,
    model_size: str,
    user_timestamp: Optional[str] = None,
) -> str:
    init_db()
    call_id_final = call_id or Path(audio_path).stem
    run_id = shortuuid.uuid()
    tmpdir = Path(tempfile.mkdtemp(prefix=f"call2eds_{run_id}_"))
    try:
        logger.info("Ingestion start call_id=%s run_id=%s", call_id_final, run_id)
        wav_stereo, flac_path, channel_wavs, is_stereo, wav_mono = audio.convert_audio(Path(audio_path), tmpdir)
        quality_audio = audio.basic_quality(wav_stereo)

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
            mix = wav_mono if wav_mono.exists() else channel_wavs[0]

            # 1) pyannote overlap-aware
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
            # 2) ECAPA clustering
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
            # 3) fallback MFCC
            if new_labels is None:
                new_labels = _cluster_speakers_mono(mix, turns_df, k=2, k_max=max_speakers)

            if new_labels is None:
                new_labels = pd.Series([0] * len(turns_df), index=turns_df.index, dtype=int)
            elif not isinstance(new_labels, pd.Series):
                new_labels = pd.Series(new_labels, index=turns_df.index, dtype=int)
            else:
                new_labels = new_labels.reindex(turns_df.index).fillna(0).astype(int)

            turns_df["speaker_id"] = new_labels.values
            if not tokens_df.empty:
                tokens_df = tokens_df.merge(turns_df[["turn_id", "speaker_id"]], on="turn_id", how="left", suffixes=("", "_t"))
                tokens_df["speaker_id"] = tokens_df["speaker_id_t"].fillna(tokens_df["speaker_id"]).astype(int)
                tokens_df = tokens_df.drop(columns=["speaker_id_t"])

        words_df = prosody.aggregate_words(tokens_df, frames_df)
        turns_prosody_df = prosody.aggregate_turns(turns_df, tokens_df, frames_df)

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

        logger.info("Ingestion terminé run_id=%s", run_id)
        return run_id
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
        run = models.Run(
            run_id=run_id,
            call_id=call_id,
            pipeline_version=PIPELINE_VERSION,
            params_json=params_json,
            status="completed",
        )
        session.add(run)
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
