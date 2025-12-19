import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.polynomial.polyutils import RankWarning

from call2eds.pipeline.pitch import fuse_pitch, smooth_series

from call2eds.utils.logger import logger


def _empty_frames(speaker_id: int = 0) -> pd.DataFrame:
    """Return an empty frames DataFrame with expected columns."""
    return pd.DataFrame(columns=["t", "f0_hz", "energy", "voicing", "pitch_conf", "speaker_id"]).assign(
        speaker_id=pd.Series(dtype=int)
    )


def _semitone_to_hz(semi: float) -> float:
    # Reference 27.5 Hz (A0)
    return 27.5 * (2 ** (semi / 12.0)) if semi > 0 else 0.0


def _get_smile_column(df: pd.DataFrame, candidates: List[str]) -> pd.Series | None:
    """
    Return the first matching column from openSMILE output.

    openSMILE column names vary by FeatureSet and often use capitalized names
    (e.g. `Loudness_sma3`). We accept exact and case-insensitive matches.
    """
    if df.empty:
        return None
    lower_to_original = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return df[cand]
        lowered = cand.lower()
        if lowered in lower_to_original:
            return df[lower_to_original[lowered]]
    return None


def _extract_frames_inner(wav_path: str, speaker_id: int) -> pd.DataFrame:
    import opensmile
    import soundfile as sf

    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # pyworld + yin fusion
    pitch_df = fuse_pitch(y, sr)
    pitch_df["f0_hz"] = smooth_series(pitch_df["f0_hz"].to_numpy(), window=5)
    pitch_df["pitch_conf"] = smooth_series(pitch_df["pitch_conf"].to_numpy(), window=5)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    df_sm = smile.process_file(wav_path).reset_index()
    if "start" in df_sm.columns:
        df_sm = df_sm.rename(columns={"start": "t"})
    elif "time" in df_sm.columns:
        df_sm = df_sm.rename(columns={"time": "t"})
    if pd.api.types.is_timedelta64_dtype(df_sm["t"]):
        df_sm["t"] = df_sm["t"].dt.total_seconds()

    energy_col = _get_smile_column(df_sm, ["Loudness_sma3", "loudness_sma3", "pcm_RMSenergy_sma"])
    energy = pd.to_numeric(energy_col, errors="coerce").fillna(0.0) if energy_col is not None else pd.Series([0.0] * len(df_sm))
    voicing_col = _get_smile_column(
        df_sm,
        [
            "voicingFinalUnclipped_sma3nz",
            "voicingFinal_sma3nz",
            "voicingFinal_sma3",
            "VoicingFinalUnclipped_sma3nz",
        ],
    )
    if voicing_col is not None:
        voicing = pd.to_numeric(voicing_col, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        voicing = (pitch_df["f0_hz"] > 0).astype(float)

    # align pitch and smile times by merge nearest
    merged = pd.merge_asof(pitch_df.sort_values("t"), df_sm.sort_values("t"), on="t", direction="nearest")
    merged["energy"] = pd.to_numeric(energy, errors="coerce").fillna(0.0)
    merged["voicing"] = voicing.values if len(voicing) == len(merged) else voicing.reindex(merged.index, fill_value=0.0)

    # pitch validity
    energy_norm = merged["energy"].to_numpy()
    try:
        p10 = float(np.percentile(energy_norm, 10))
        p90 = float(np.percentile(energy_norm, 90))
        if p90 > p10:
            energy_norm = (energy_norm - p10) / (p90 - p10)
        else:
            energy_norm *= 0
    except Exception:
        energy_norm *= 0
    energy_norm = np.clip(energy_norm, 0.0, 1.0)
    pitch_valid = 0.5 * merged["pitch_conf"].to_numpy() + 0.3 * merged["voicing"].to_numpy() + 0.2 * energy_norm

    return pd.DataFrame(
        {
            "t": merged["t"],
            "f0_hz": merged["f0_hz"],
            "energy": merged["energy"],
            "voicing": merged["voicing"],
            "pitch_conf": np.clip(pitch_valid, 0, 1),
            "speaker_id": speaker_id,
        }
    )


def extract_frames(wav_path: Path, speaker_id: int = 0, timeout_s: int = 120) -> pd.DataFrame:
    """
    Extract frame-level prosodic features inline (single process).
    We previously sandboxed in a worker to avoid native crashes, but that triggered
    import issues inside containers; inline is more reliable here. Failures return
    an empty frame DataFrame so the pipeline can continue.
    """
    logger.info("Extraction prosodie (openSMILE)")
    try:
        return _extract_frames_inner(str(wav_path), speaker_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prosody extraction failed: %s", exc)
        return _empty_frames(speaker_id)


def aggregate_words(tokens: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    tokens_sorted = tokens.sort_values(["turn_id", "token_id"]).reset_index(drop=True)
    for _, tok in tokens_sorted.iterrows():
        mask = (frames["t"] >= tok.t_start) & (frames["t"] <= tok.t_end)
        sub = frames[mask]
        f0_mean = sub.f0_hz.mean() if not sub.empty else 0.0
        f0_std = sub.f0_hz.std(ddof=0) if not sub.empty else 0.0
        # slope via simple linear regression (skip ill-conditioned cases to avoid RankWarning spam)
        if len(sub) >= 2 and sub.t.std(ddof=0) > 1e-6 and sub.f0_hz.std(ddof=0) > 1e-3:
            with np.errstate(all="ignore"):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RankWarning)
                    slope = np.polyfit(sub.t, sub.f0_hz, 1)[0]
        else:
            slope = 0.0
        energy_mean = sub.energy.mean() if not sub.empty else 0.0
        energy_std = sub.energy.std(ddof=0) if not sub.empty else 0.0
        voiced_ratio = (sub.voicing > 0.5).mean() if not sub.empty else 0.0
        rows.append(
            {
                "token_id": tok.token_id,
                "turn_id": tok.turn_id,
                "f0_mean": f0_mean,
                "f0_std": f0_std,
                "f0_slope": slope,
                "f0_range": (sub.f0_hz.quantile(0.95) - sub.f0_hz.quantile(0.05)) if not sub.empty else 0.0,
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "energy_range": (sub.energy.quantile(0.95) - sub.energy.quantile(0.05)) if not sub.empty else 0.0,
                "voiced_ratio": voiced_ratio,
                "pitch_valid_ratio": (sub.pitch_conf > 0.5).mean() if not sub.empty else 0.0,
                "pause_before_ms": 0.0,
                "pause_after_ms": 0.0,
            }
        )
    # pauses par mot dans le même tour
    for i, tok in tokens_sorted.iterrows():
        if i > 0 and tok.turn_id == tokens_sorted.iloc[i - 1].turn_id:
            rows[i]["pause_before_ms"] = (tok.t_start - tokens_sorted.iloc[i - 1].t_end) * 1000
        if i < len(tokens_sorted) - 1 and tok.turn_id == tokens_sorted.iloc[i + 1].turn_id:
            rows[i]["pause_after_ms"] = (tokens_sorted.iloc[i + 1].t_start - tok.t_end) * 1000
    return pd.DataFrame(rows)


def aggregate_turns(turns: pd.DataFrame, tokens: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, turn in turns.iterrows():
        duration = turn.t_end - turn.t_start
        tok_turn = tokens[tokens.turn_id == turn.turn_id]
        frames_turn = frames[(frames.t >= turn.t_start) & (frames.t <= turn.t_end)]
        speech_rate = len(tok_turn) / duration if duration > 0 else 0
        silence_ratio = 1 - ((frames_turn.voicing > 0.5).mean() if not frames_turn.empty else 0)
        pitch_valid_ratio = (frames_turn.pitch_conf > 0.5).mean() if not frames_turn.empty else 0
        rows.append(
            {
                "turn_id": turn.turn_id,
                "speech_rate_wps": speech_rate,
                "silence_ratio": silence_ratio,
                "voiced_ratio": 1 - silence_ratio,
                "f0_mean": frames_turn.f0_hz.mean() if not frames_turn.empty else 0,
                "f0_std": frames_turn.f0_hz.std(ddof=0) if not frames_turn.empty else 0,
                "f0_range": (frames_turn.f0_hz.quantile(0.95) - frames_turn.f0_hz.quantile(0.05)) if not frames_turn.empty else 0,
                "energy_mean": frames_turn.energy.mean() if not frames_turn.empty else 0,
                "energy_std": frames_turn.energy.std(ddof=0) if not frames_turn.empty else 0,
                "energy_range": (frames_turn.energy.quantile(0.95) - frames_turn.energy.quantile(0.05)) if not frames_turn.empty else 0,
                "pitch_valid_ratio": pitch_valid_ratio,
            }
        )
    return pd.DataFrame(rows)
