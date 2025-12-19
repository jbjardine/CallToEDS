import subprocess
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import soundfile as sf
import imageio_ffmpeg

from call2eds.utils.logger import logger


def _ffmpeg_bin() -> str:
    """
    Returns a usable ffmpeg binary path:
    - prefer system ffmpeg if available
    - fallback to imageio-ffmpeg embedded binary (works on Windows)
    """
    from shutil import which

    sys_ff = which("ffmpeg")
    if sys_ff:
        return sys_ff
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args):
    cmd = [_ffmpeg_bin(), "-y"] + args
    logger.debug("ffmpeg command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def convert_audio(input_path: Path, workdir: Path) -> Tuple[Path, Path, List[Path], bool, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    wav_stereo = workdir / "normalized_stereo.wav"
    wav_mono = workdir / "normalized_mono.wav"
    flac_path = workdir / "normalized.flac"
    # force 16kHz, 2 canaux (si source mono, ffmpeg duplique)
    run_ffmpeg(["-i", str(input_path), "-ar", "16000", "-ac", "2", "-acodec", "pcm_s16le", str(wav_stereo)])
    # version mono pour la diarisation robuste
    run_ffmpeg(["-i", str(input_path), "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", str(wav_mono)])

    audio_data, sr = sf.read(str(wav_stereo))
    # encode FLAC via soundfile pour éviter un 2ème appel ffmpeg (plus robuste en container)
    sf.write(str(flac_path), audio_data, sr, format="FLAC")
    channel_paths: List[Path] = []
    is_stereo = False
    if audio_data.ndim == 1:
        mono_path = workdir / "channel_0.wav"
        sf.write(str(mono_path), audio_data, sr)
        channel_paths.append(mono_path)
    else:
        if audio_data.shape[1] == 2:
            # Détecte le "dual mono" (canaux très corrélés) et replie en mono
            ch0, ch1 = audio_data[:, 0], audio_data[:, 1]
            rms0 = np.sqrt(np.mean(ch0**2) + 1e-9)
            rms1 = np.sqrt(np.mean(ch1**2) + 1e-9)
            corr = np.corrcoef(ch0, ch1)[0, 1] if rms0 > 0 and rms1 > 0 else 1.0
            delta = np.max(np.abs(ch0 - ch1))
            if corr > 0.9 and delta < 5e-2:
                mono_path = workdir / "channel_0.wav"
                sf.write(str(mono_path), (ch0 + ch1) / 2.0, sr)
                channel_paths.append(mono_path)
                is_stereo = False
            else:
                is_stereo = True
                for ch in range(audio_data.shape[1]):
                    ch_path = workdir / f"channel_{ch}.wav"
                    sf.write(str(ch_path), audio_data[:, ch], sr)
                    channel_paths.append(ch_path)
        else:
            is_stereo = audio_data.shape[1] == 2
            for ch in range(audio_data.shape[1]):
                ch_path = workdir / f"channel_{ch}.wav"
                sf.write(str(ch_path), audio_data[:, ch], sr)
                channel_paths.append(ch_path)
    return wav_stereo, flac_path, channel_paths, is_stereo, wav_mono


def basic_quality(wav_path: Path) -> Dict[str, float]:
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    peak = float(np.max(np.abs(audio)))
    clip_rate = float(np.mean(np.abs(audio) > 0.99))
    # Silence estimate: frames of 0.1s
    frame = int(sr * 0.1)
    energies = [np.mean(np.square(audio[i : i + frame])) for i in range(0, len(audio), frame)]
    energies = np.array(energies)
    silence_thresh = np.percentile(energies, 10)
    pct_silence = float(np.mean(energies < silence_thresh))
    # SNR approx: ratio between 90th and 10th percentile energy
    snr = 10 * np.log10((np.percentile(energies, 90) + 1e-9) / (np.percentile(energies, 10) + 1e-9))
    return {
        "duration_s": duration,
        "peak": peak,
        "clip_rate": clip_rate,
        "pct_silence": pct_silence,
        "snr_est": float(snr),
    }
