# Call2EDS (MVP)

Pipeline CLI pour transformer automatiquement un enregistrement d'appel (wav/mp3/m4a/aac/ogg/flac…) en paquet EDS prêt pour l'entraînement IA : audio normalisé, transcription horodatée, prosodie frame-level, agrégats, métriques qualité, manifest et index DB/MinIO.

## Fonctionnalités clés
- Conversion audio via ffmpeg (16 kHz); mono traité en speaker_0, stéréo splitté L/R → speaker_0/1; FLAC lossless archivé
- ASR Faster-Whisper (CTranslate2) avec timestamps mot à mot
- Prosodie frame-level via openSMILE (f0/voicing/energy) + agrégats mots/tours
- Stockage objet MinIO (S3) des artefacts Parquet + manifest
- Index Postgres minimal : calls, runs, artifacts, metrics
- CLI Typer : `ingest`, `show`, `export`, `doctor`, `purge`

## Prérequis
- Docker + Docker Compose
- Ports libres : 5432 (Postgres), 9000/9001 (MinIO)
- Espace disque pour modèles ASR (~200–600 Mo selon le modèle)
- ffmpeg disponible dans le PATH : sous Windows, `choco install ffmpeg` (ou ajouter le binaire ffmpeg.exe au PATH)

## Démarrage rapide (CLI)
```bash
git clone <repo> && cd <repo>
cp .env.example .env
# Lancer les services de base
docker compose up -d
# Ingestion d'un audio démo
docker compose run --rm app call2eds ingest ./samples/demo.wav --call-id DEMO001
# Inspecter les runs d'un call_id
docker compose run --rm app call2eds show DEMO001
# Exporter les artefacts localement
docker compose run --rm app call2eds export DEMO001 --out ./out
```

## Interface Web + mini-API
- UI intégrée (FastAPI + Jinja). Après `docker compose up -d`, ouvrir http://localhost:8000
- Upload direct (call_id auto ou saisi), choix langue/modèle, horodatage manuel optionnel.
- Liste des runs par call_id, détail run avec liens directs MinIO, export manifest ou ZIP.
- Batch CSV (formulaire) : colonnes `audio_path,call_id,lang,model,when` ; possibilité de joindre les audios pour résoudre les chemins.
- API upload sans token (MVP) : `POST /upload-api` (form-data `audio`, `call_id` opt, `lang`, `model`, `when`).

### Mode GPU (Linux / Docker Desktop avec support GPU)
- Activer le support GPU dans Docker Desktop.
- Le Dockerfile est basé sur CUDA 12.6 + cuDNN et `docker-compose` demande un device GPU ; sinon fallback CPU automatique.
- Exécution identique : `docker compose run --rm app call2eds ingest ...` (le GPU sera utilisé s'il est dispo).

## Commandes CLI
- `call2eds ingest <audio_path> [--call-id ID] [--lang fr] [--model small]`
- `call2eds run <audio_path> [...]` (alias de ingest)
- `call2eds show <call_id>`
- `call2eds export <call_id> --run-id <id?> --out <dir>`
- `call2eds doctor` (ffmpeg, MinIO, Postgres)
- `call2eds purge <call_id> [--run-id <id>]` (optionnel, supprime objets S3 + DB)

## Stockage S3 (MinIO)
Bucket `call2eds`
```
calls/{call_id}/runs/{run_id}/audio/normalized.flac
calls/{call_id}/runs/{run_id}/eds/turns.parquet
calls/{call_id}/runs/{run_id}/eds/tokens.parquet
calls/{call_id}/runs/{run_id}/eds/prosody_frames.parquet
calls/{call_id}/runs/{run_id}/eds/prosody_words.parquet
calls/{call_id}/runs/{run_id}/eds/prosody_turns.parquet
calls/{call_id}/runs/{run_id}/eds/quality.parquet
calls/{call_id}/runs/{run_id}/manifest.json
```

## Schéma Parquet
- `turns.parquet` : call_id, run_id, turn_id, speaker_id, t_start, t_end, text, asr_conf_turn
- `tokens.parquet` : call_id, run_id, turn_id, token_id, word, t_start, t_end, asr_conf_word
- `prosody_frames.parquet` : call_id, run_id, speaker_id, t, f0_hz, voicing, energy, pitch_conf
- `prosody_words.parquet` : call_id, run_id, token_id, f0_mean, f0_std, f0_slope, energy_mean, energy_std, voiced_ratio, pause_before_ms, pause_after_ms
- `prosody_turns.parquet` : call_id, run_id, turn_id, speech_rate_wps, silence_ratio, voiced_ratio, f0_mean, f0_std, energy_mean, energy_std
- `quality.parquet` : call_id, run_id, duration_s, snr_est, clip_rate, pct_silence, pct_voiced, pct_pitch_invalid, asr_conf_mean, asr_conf_p10, asr_conf_p90, warnings
Notes :
- `energy_*` est exprimée en dBFS (0 = pleine échelle, valeurs négatives = plus faible).

## Base Postgres
Tables :
- `calls(call_id, created_at, meta_json)`
- `runs(run_id, call_id, created_at, pipeline_version, params_json, status)`
- `artifacts(artifact_id, run_id, kind, s3_uri, sha256, size_bytes)`
- `metrics(run_id, key, value_num, value_json)`

## Limitations / RGPD
- Aucune dé-identification automatique n'est réalisée.
- Diarisation avancée: pyannote si `HF_TOKEN` est fourni et l'accès au modèle est accepté; sinon fallback SpeechBrain (moins robuste).
- Pipeline CPU par défaut ; si un GPU CUDA fonctionnel est présent, faster-whisper l’utilise automatiquement. En cas d’échec GPU, le code retombe en CPU sans variable à régler.
- Pour forcer explicitement : `CALL2EDS_DEVICE=cpu` (ou `C_TRANSLATE2_FORCE_CPU=1`) ; `CALL2EDS_DEVICE=cuda` si vous voulez obliger le GPU.

## Dépannage
- Vérifier les services : `docker compose ps`
- ffmpeg manquant : `call2eds doctor`
- Modèle ASR trop lourd : utiliser `--model small` ou `tiny`.
- MinIO UI : http://localhost:9001 (minio / minio123 par défaut)
- Erreur 403 / "gated model" sur pyannote : accepter les conditions du modèle sur Hugging Face avec le compte lié au token, puis relancer. Assurez-vous que `HF_TOKEN` est défini (ou via `~/.cache/hf_token`).
- Diarisation avec un seul speaker alors que l'audio est multi-speakers : définir `CALL2EDS_MIN_SPK=2` (et éventuellement `CALL2EDS_MAX_SPK`) puis relancer.
- Pitch/prosodie : `CALL2EDS_PITCH_BACKEND=parselmouth` (par défaut) ou `fuse` si besoin. Ajuster `CALL2EDS_PITCH_FLOOR` / `CALL2EDS_PITCH_CEIL` si voix atypiques.
- Overlap (paroles simultanées) : `CALL2EDS_DIAR_OVERLAP_STRICT=1` pour marquer les segments chevauchés comme “overlap” et éviter la contamination prosodique. Seuil via `CALL2EDS_DIAR_OVERLAP_MIN`.
- Début de transcription manquant : désactiver le VAD (`CALL2EDS_VAD_FILTER=false`) et augmenter `CALL2EDS_NO_SPEECH_THRESHOLD` (ex: `0.95` ou `1.0`) et/ou baisser `CALL2EDS_LOGPROB_THRESHOLD` (ex: `-2.0`) puis relancer.
- Diarisation trop “collée” au début : activer la correction ECAPA (`CALL2EDS_DIAR_EARLY_REPAIR=1`) et ajuster la fenêtre (`CALL2EDS_DIAR_EARLY_WINDOW_S`) ou la durée minimale du premier bloc (`CALL2EDS_DIAR_EARLY_BLOCK_S`). Si la séparation existe mais reste marginale, augmenter `CALL2EDS_DIAR_EARLY_MINORITY_RATIO` pour forcer le fallback micro‑tours. Pour un début plus sensible, ajuster `CALL2EDS_ECAPA_EARLY_WIN_S` / `CALL2EDS_ECAPA_EARLY_HOP_S` / `CALL2EDS_ECAPA_EARLY_SMOOTH` et la limite `CALL2EDS_DIAR_EARLY_MAX_SPK` (par défaut 2). Pour forcer un re‑découpage avec prototypes ECAPA, activer `CALL2EDS_DIAR_EARLY_FORCE=1` et régler `CALL2EDS_DIAR_EARLY_PROTO_WIN_S` / `CALL2EDS_DIAR_EARLY_PROTO_HOP_S` / `CALL2EDS_DIAR_EARLY_PROTO_SMOOTH`. Les micro‑tours dédiés sont aussi réglables (`CALL2EDS_ECAPA_MICRO_EARLY_GAP_S`, `CALL2EDS_ECAPA_MICRO_EARLY_MAX_S`, `CALL2EDS_ECAPA_MICRO_EARLY_MIN_S`). Vous pouvez aussi ajuster (`CALL2EDS_ECAPA_WIN_S`, `CALL2EDS_ECAPA_HOP_S`) et le fallback micro‑tours global (`CALL2EDS_ECAPA_MICRO_GAP_S`, `CALL2EDS_ECAPA_MICRO_MAX_S`, `CALL2EDS_ECAPA_MICRO_MIN_S`). Enfin, régler `CALL2EDS_DIAR_MIN_ON` / `CALL2EDS_DIAR_MIN_OFF` (ex: `0.0` / `0.1`), `CALL2EDS_DIAR_CLUSTER_THRESHOLD` (ex: `0.6`) et `CALL2EDS_DIAR_CLUSTER_MIN_SIZE` pour mieux séparer les speakers.
- Flip A‑B‑A (même speaker séparé par un micro‑segment) : activer `CALL2EDS_DIAR_FLIP_ENABLED=1` (défaut) et ajuster `CALL2EDS_DIAR_FLIP_MAX_S` / `CALL2EDS_DIAR_FLIP_MAX_TOKENS` / `CALL2EDS_DIAR_FLIP_MAX_OV` / `CALL2EDS_DIAR_FLIP_MAX_GAP_S`. Pour éviter de fusionner un vrai speaker, limiter via `CALL2EDS_DIAR_FLIP_MAX_CONF` (ex: `0.8`) si la diarisation est confiante.
- Début découpé (petit segment initial puis long bloc) : activer `CALL2EDS_DIAR_EDGE_ENABLED=1` (défaut) et ajuster `CALL2EDS_DIAR_EDGE_WINDOW_S` / `CALL2EDS_DIAR_EDGE_MAX_S` / `CALL2EDS_DIAR_EDGE_MAX_TOKENS` / `CALL2EDS_DIAR_EDGE_MAX_OV` / `CALL2EDS_DIAR_EDGE_MIN_NEXT_S` / `CALL2EDS_DIAR_EDGE_MAX_GAP_S`.
- Alternances rapides mal détectées (ex: dialogue serré) : activer `CALL2EDS_DIAR_MICRO_REFINE=1` (défaut) et ajuster `CALL2EDS_DIAR_MICRO_MIN_DELTA` pour forcer le raffinement ECAPA. Le mode `CALL2EDS_DIAR_MICRO_MODE=hybrid` essaie à la fois micro‑tours et fenêtres glissantes courtes (paramètres `CALL2EDS_DIAR_MICRO_WIN_S` / `CALL2EDS_DIAR_MICRO_HOP_S` / `CALL2EDS_DIAR_MICRO_SMOOTH`). Option `CALL2EDS_DIAR_MICRO_FORCE_DIVERSITY=1` pour éviter que tous les micro‑labels se recollent au même speaker.
- Resegmentation globale (toute la conversation) : activer `CALL2EDS_DIAR_GLOBAL_RESEG=1` (défaut). Réglages principaux `CALL2EDS_DIAR_GLOBAL_PROTO_WIN_S` / `CALL2EDS_DIAR_GLOBAL_PROTO_HOP_S` / `CALL2EDS_DIAR_GLOBAL_PROTO_SMOOTH`. Pour limiter l'impact, ajuster `CALL2EDS_DIAR_GLOBAL_MAX_CHANGE`, `CALL2EDS_DIAR_GLOBAL_MAX_DROP`, et les zones ciblées via `CALL2EDS_DIAR_GLOBAL_MAX_CONF` / `CALL2EDS_DIAR_GLOBAL_MAX_OV` / `CALL2EDS_DIAR_GLOBAL_SHORT_S` / `CALL2EDS_DIAR_GLOBAL_SHORT_TOKENS`. En cas de besoin, `CALL2EDS_DIAR_GLOBAL_FORCE=1` applique la reseg sur tous les tokens.
- Prototypes ECAPA (qualité des références) : régler `CALL2EDS_DIAR_PROTO_MIN_S` / `CALL2EDS_DIAR_PROTO_MAX_PER_SPK` / `CALL2EDS_DIAR_PROTO_MIN_CONF` / `CALL2EDS_DIAR_PROTO_MAX_OV` / `CALL2EDS_DIAR_PROTO_MIN_DUR`.

## Développement
- Lint : `ruff .`
- Tests : `pytest`
- Typage léger : `mypy src`
