import os
import platform
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from faster_whisper import WhisperModel

from call2eds.utils.logger import logger


def transcribe_audio(
    wav_path: Path,
    model_size: str,
    language: Optional[str] = None,
    speaker_id: int = 0,
) -> Tuple[List[Dict], List[Dict], float, int]:
    # Auto device, with graceful fallback to CPU if CUDA/cuDNN not usable.
    force_cpu = os.getenv("C_TRANSLATE2_FORCE_CPU", "").lower() in {"1", "true", "yes"}
    default_device = "cpu" if platform.system().lower().startswith("win") else "auto"
    requested = os.getenv("CALL2EDS_DEVICE") or ("cpu" if force_cpu else default_device)
    user_compute = os.getenv("CALL2EDS_COMPUTE_TYPE")

    def compute_for(device: str) -> str:
        if user_compute:
            return user_compute
        # int8 on GPU is lighter and more stable on consumer cards; caller can override via env.
        return "int8_float32" if device.startswith("cuda") else "int8_float32"

    devices_to_try = []
    if requested == "auto":
        devices_to_try = ["auto", "cuda", "cpu"]
    elif requested == "cuda":
        devices_to_try = ["cuda", "cpu"]
    else:
        devices_to_try = [requested]

    last_err = None
    model = None
    chosen_device = None
    for dev in devices_to_try:
        try:
            ct = compute_for(dev)
            logger.info("Chargement du modèle ASR %s (device=%s, compute=%s)", model_size, dev, ct)
            model = WhisperModel(model_size, device=dev, compute_type=ct)
            chosen_device = dev
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("Echec chargement modèle sur device=%s : %s", dev, e)
            continue
    if model is None:
        raise RuntimeError(f"Impossible de charger le modèle {model_size} (dernière erreur: {last_err})")
    # transcribe avec fallback CPU si besoin
    segments = None
    info = None
    last_run_err = None
    for dev in (chosen_device, "cpu"):
        if dev != chosen_device:
            ct = compute_for(dev)
            logger.info("Fallback ASR sur device=%s (compute=%s)", dev, ct)
            model = WhisperModel(model_size, device=dev, compute_type=ct)
        try:
            segments, info = model.transcribe(
                str(wav_path),
                beam_size=1,
                language=language if language and language != "auto" else None,
                word_timestamps=True,
                condition_on_previous_text=False,  # réduit les répétitions / boucles
            )
            break
        except Exception as e:  # noqa: BLE001
            last_run_err = e
            logger.warning("Echec transcribe sur device=%s : %s", dev, e)
            continue

    if segments is None:
        raise RuntimeError(f"Echec ASR (dernière erreur: {last_run_err})")
    tokens: List[Dict] = []
    turns: List[Dict] = []
    total_conf = 0.0
    total_words = 0
    for turn_id, seg in enumerate(segments):
        seg_conf = 0.0
        word_count = 0
        if seg.words:
            for wid, w in enumerate(seg.words):
                tokens.append(
                    {
                        "turn_id": turn_id,
                        "token_id": wid,
                        "word": w.word.strip(),
                        "t_start": w.start,
                        "t_end": w.end,
                        "asr_conf_word": w.probability,
                        "speaker_id": speaker_id,
                    }
                )
                seg_conf += w.probability
                total_conf += w.probability
                total_words += 1
                word_count += 1
        avg_seg_conf = seg_conf / word_count if word_count else 0.0
        turns.append(
            {
                "turn_id": turn_id,
                "speaker_id": speaker_id,
                "t_start": seg.start,
                "t_end": seg.end,
                "text": seg.text.strip(),
                "asr_conf_turn": avg_seg_conf,
            }
        )
    avg_conf = total_conf / total_words if total_words else 0.0
    logger.info("ASR terminé (%d mots, conf=%.3f)", total_words, avg_conf)
    return tokens, turns, avg_conf, total_words
