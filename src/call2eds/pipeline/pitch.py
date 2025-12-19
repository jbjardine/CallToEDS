from __future__ import annotations

import numpy as np
import pandas as pd
import pyworld as pw
import librosa


def fuse_pitch(wav: np.ndarray, sr: int) -> pd.DataFrame:
    """Compute pitch with pyworld + librosa yin, fuse and return frame dataframe (t, f0_hz, conf)."""
    # pyworld frame hop 5 ms
    _f0, t = pw.harvest(wav.astype(np.float64), sr, frame_period=5.0)
    f0_pw = pw.stonemask(wav.astype(np.float64), _f0, t, sr)
    # librosa yin (hop 5 ms)
    f0_yin = librosa.yin(wav.astype(float), fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=int(sr*0.005))
    # align lengths
    m = min(len(f0_pw), len(f0_yin))
    f0_pw = f0_pw[:m]
    f0_yin = f0_yin[:m]
    t = t[:m]
    # confidence : agreement + nonzero
    agree = (np.abs(f0_pw - f0_yin) / (f0_pw + 1e-6)) < 0.2
    conf = 0.5 * agree.astype(float) + 0.25 * (f0_pw > 0) + 0.25 * (f0_yin > 0)
    f0 = np.where(conf >= 0.5, 0.5 * (f0_pw + f0_yin), np.maximum(f0_pw, f0_yin))
    return pd.DataFrame({
        "t": t,
        "f0_hz": f0,
        "pitch_conf": conf.clip(0,1)
    })


def smooth_series(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return x
    pad = window // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(xpad, kernel, mode="valid")

