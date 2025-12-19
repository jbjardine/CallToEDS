import os
from pathlib import Path
from typing import Optional
import re

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from call2eds.utils.logger import logger


def _auto_cluster(embeddings: np.ndarray, k_max: int = 8) -> np.ndarray:
    if embeddings.shape[0] < 2:
        return np.zeros(len(embeddings), dtype=int)
    best_score = -1
    best_labels = None
    for k in range(2, min(k_max, len(embeddings)) + 1):
        labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(
            embeddings
        )
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


def diarize_pyannote(
    wav_path: Path,
    turns_df: pd.DataFrame,
    max_spk: int = 8,
    min_spk: int = 1,
) -> Optional[pd.Series]:
    """
    Diarisation robuste via pyannote (overlap géré). Retourne une Série speaker_id alignée sur turns_df.
    Nécessite HF_TOKEN dans l'environnement ou ~/.cache/hf_token.
    """
    if turns_df.empty:
        return None
    token = os.getenv("HF_TOKEN")
    if not token:
        token_file = Path.home() / ".cache" / "hf_token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        logger.warning("Aucun HF_TOKEN pour pyannote; skip")
        return None
    try:
        from pyannote.audio import Pipeline
    except Exception as exc:  # noqa: BLE001
        logger.warning("pyannote indisponible: %s", exc)
        return None

    try:
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        diar = pipeline(str(wav_path), min_speakers=min_spk, max_speakers=max_spk)
        # pyannote >= 3.1 returns DiarizeOutput dataclass
        if hasattr(diar, "speaker_diarization"):
            diar = diar.speaker_diarization
        elif hasattr(diar, "diarization"):
            diar = diar.diarization
        # Construire embeddings frame-level -> on prend les segments de diarisation
        # Assignation au tour : speaker majoritaire en durée
        try:
            from pyannote.core import Segment
        except Exception:  # noqa: BLE001
            Segment = None
        speaker_labels = []
        label_map: dict[str, int] = {}

        def _label_to_int(lbl: object) -> int:
            lbl_str = str(lbl)
            match = re.search(r"(\d+)", lbl_str)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
            if lbl_str not in label_map:
                label_map[lbl_str] = len(label_map)
            return label_map[lbl_str]
        for _, row in turns_df.iterrows():
            if Segment is not None:
                segment = diar.crop(Segment(float(row.t_start), float(row.t_end)))
            else:
                segment = diar.crop(float(row.t_start), float(row.t_end))
            dur_by_spk = {}
            for turn in segment.itertracks(yield_label=True):
                if len(turn) == 3:
                    span, _, spk = turn
                else:
                    span, spk = turn
                dur_by_spk[spk] = dur_by_spk.get(spk, 0.0) + (span.end - span.start)
            if dur_by_spk:
                spk_major = max(dur_by_spk.items(), key=lambda kv: kv[1])[0]
                speaker_labels.append(_label_to_int(spk_major))
            else:
                speaker_labels.append(0)
        return pd.Series(speaker_labels, index=turns_df.index, dtype=int)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        lowered = msg.lower()
        if "403" in msg or "gated" in lowered or "access to model" in lowered:
            logger.warning(
                "Diarisation pyannote bloquée (modèle gated / accès refusé). "
                "Cause probable: conditions Hugging Face non acceptées pour "
                "pyannote/speaker-diarization-3.1 ou token invalide. "
                "Action: ouvrir la page du modèle, accepter les conditions "
                "avec le compte lié au token, puis relancer. "
                "Vérifier aussi HF_TOKEN ou ~/.cache/hf_token. Détail: %s",
                msg,
            )
        else:
            logger.warning("Diarisation pyannote échouée: %s", msg)
        return None
