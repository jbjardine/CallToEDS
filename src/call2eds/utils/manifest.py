import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from call2eds import __version__


def build_manifest(
    call_id: str,
    run_id: str,
    model_name: str,
    language: str,
    artifacts: List[Dict[str, Any]],
    params: Dict[str, Any],
    stats: Dict[str, Any],
    ffmpeg_version: str = "unknown",
) -> Dict[str, Any]:
    return {
        "call_id": call_id,
        "run_id": run_id,
        # timestamp en UTC avec offset explicite pour éviter toute ambiguïté
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "call2eds": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "faster_whisper_model": model_name,
            "opensmile": "2.x",
            "ffmpeg": ffmpeg_version,
        },
        "params": params,
        "language": language,
        "artifacts": artifacts,
        "quality": stats,
    }


def save_manifest(manifest: Dict[str, Any], path: Path):
    # Force LF et ajoute un newline final pour limiter les diffs CRLF/Unix
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(manifest, indent=2))
        f.write("\n")
