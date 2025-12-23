# Call2EDS

[![Release](https://img.shields.io/github/v/release/jbjardine/CallToESD)](https://github.com/jbjardine/CallToESD/releases)
[![Release CI](https://github.com/jbjardine/CallToESD/actions/workflows/release.yml/badge.svg)](https://github.com/jbjardine/CallToESD/actions/workflows/release.yml)
[![GHCR](https://img.shields.io/badge/ghcr.io-jbjardine%2Fcall2eds-1f2937)](https://github.com/jbjardine/CallToEDS/pkgs/container/call2eds)
[![License](https://img.shields.io/github/license/jbjardine/CallToESD)](https://github.com/jbjardine/CallToESD/blob/main/LICENSE)

Call2EDS transforme un appel audio en paquet EDS (Entrepot de Donnees de Sante) pret pour l’IA.
Le nom est un clin d’oeil bilingue: “call to EDS” en anglais et “appel vers l’EDS” en francais.
Bref, on passe du call au data warehouse.

This repo turns phone calls into EDS-ready training packages.
The name is a bilingual wink: “call to EDS” in English and “appel vers l’EDS” in French.
In short, it bridges the call and the data warehouse.

<br/>
If this project is useful to you, you can support its development here:

[![Support](https://img.shields.io/badge/Support-Buy%20Me%20a%20Coffee-1f2937)](https://www.buymeacoffee.com/jbjardine)

---

## Francais

### Pourquoi
Call2EDS met en place un flux reproductible pour:
- normaliser l’audio,
- transcrire avec horodatage mot-a-mot,
- extraire la prosodie (frame, mot, tour),
- consolider les artefacts dans MinIO + indexer dans Postgres,
- fournir une UI interne et une API simple.

L’objectif est un format propre, stable et “trainable” pour des modeles downstream (ASR, diarisation, emotion, QA, etc.).

### Fonctionnalites
- Normalisation audio via ffmpeg (16 kHz). Mono = speaker_0; stereo = split L/R.
- ASR Faster-Whisper (CTranslate2) avec timestamps mot-a-mot.
- Prosodie frame-level via openSMILE + agregats mots/tours.
- Stockage S3 MinIO + index Postgres minimal.
- UI Web (FastAPI + Jinja) pour ingestion, suivi des runs, historique, diagnostics.
- API HTTP pour ingestion et operations systeme.
- Auth optionnelle: sessions, Basic Auth, API keys.

### Demarrage rapide (Docker)
```bash
git clone <repo>
cd <repo>
cp .env.example .env

docker compose up -d
# Ingestion rapide
docker compose run --rm app ingest /path/to/audio.wav --call-id DEMO001
# Inspecter les runs
docker compose run --rm app show DEMO001
# Exporter les artefacts localement
docker compose run --rm app export DEMO001 --out ./out
```

### Installation depuis une release
- Tarball source: telecharger `call2eds-vX.Y.Z.tar.gz` depuis GitHub Releases, puis:
```
tar -xzf call2eds-vX.Y.Z.tar.gz
cd call2eds
cp .env.example .env
docker compose up -d
```
- Image Docker (GHCR):
```
docker pull ghcr.io/jbjardine/call2eds:vX.Y.Z
```

### Interface Web & API
- UI: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Upload direct, choix langue/modele, horodatage manuel optionnel.
- Batch CSV: colonnes `audio_path,call_id,lang,model,when` (audios en pieces jointes si besoin).

### Diagnostic, securite, admin
- Diagnostic UI: `/diagnostic` (HTML) ou `/diagnostic?format=json`.
- Ressources systeme: `/api/system` (CPU/RAM/disque/GPU).
- Config runtime: `/api/config` (lecture / ecriture). Option `persist` -> `.env.secrets`.
- Annuler un traitement: `POST /api/runs/{run_id}/cancel` (soft cancel, arret au prochain checkpoint).
- Auth (optionnelle):
  - Activer: `CALL2EDS_AUTH_ENABLED=true`
  - Session cookie via `/api/auth/login`
  - API key via header `X-API-Key` ou `Authorization: Bearer <key>`
  - Basic Auth possible pour UI/API
  - Gestion comptes/keys: `/access`
- Swagger permet de tester avec vos identifiants (bouton Authorize).
 - Avant production: changer les mots de passe par defaut (MinIO, Postgres) et activer l'auth.
 - Pour un usage intranet: mettre un reverse proxy TLS devant l'UI/API.

### GPU
- Image base CUDA + cuDNN.
- Si GPU dispo, faster-whisper l’utilise automatiquement.
- Forcer: `CALL2EDS_DEVICE=cuda` ou `CALL2EDS_DEVICE=cpu`.

### Stockage MinIO
Bucket `call2eds`:
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

### Base Postgres
- `calls(call_id, created_at, meta_json)`
- `runs(run_id, call_id, created_at, pipeline_version, params_json, status)`
- `artifacts(artifact_id, run_id, kind, s3_uri, sha256, size_bytes)`
- `metrics(run_id, key, value_num, value_json)`

### Limitations / RGPD
- Pas de de-identification automatique.
- Diarisation: pyannote si HF_TOKEN valide, sinon SpeechBrain fallback.
- Input always validated server-side. Never trust the input.

### Depannage
- `docker compose ps` pour l’etat services.
- ffmpeg manquant: `call2eds doctor`.
- Modeles lourds: utiliser `tiny` ou `small`.
- Pyannote gated: accepter conditions HF + `HF_TOKEN`.


---

## English

### Why
Call2EDS provides a repeatable pipeline to:
- normalize audio,
- transcribe with word-level timestamps,
- extract prosody (frame/word/turn),
- store artifacts in MinIO + index in Postgres,
- expose a clean UI and HTTP API.

The goal is a stable, trainable format for downstream ML.

### Features
- ffmpeg normalization (16 kHz). Mono -> speaker_0; stereo split L/R.
- Faster-Whisper ASR (CTranslate2) with word timestamps.
- Prosody via openSMILE + word/turn aggregates.
- MinIO S3 storage + minimal Postgres index.
- Web UI for ingestion, runs, history, diagnostics.
- HTTP API for ingestion and ops.
- Optional auth: sessions, Basic Auth, API keys.

### Quickstart (Docker)
```bash
git clone <repo>
cd <repo>
cp .env.example .env

docker compose up -d
# Ingest
docker compose run --rm app ingest /path/to/audio.wav --call-id DEMO001
# Show runs
docker compose run --rm app show DEMO001
# Export artifacts
docker compose run --rm app export DEMO001 --out ./out
```

### Install from a release
- Source tarball: download `call2eds-vX.Y.Z.tar.gz` from GitHub Releases, then:
```
tar -xzf call2eds-vX.Y.Z.tar.gz
cd call2eds
cp .env.example .env
docker compose up -d
```
- Docker image (GHCR):
```
docker pull ghcr.io/jbjardine/call2eds:vX.Y.Z
```

### Web UI & API
- UI: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Upload, language/model selection, manual timestamp.
- CSV batch: `audio_path,call_id,lang,model,when`.

### Diagnostics, security, admin
- Diagnostic UI: `/diagnostic` or `/diagnostic?format=json`.
- System resources: `/api/system`.
- Runtime config: `/api/config` (persist to `.env.secrets`).
- Cancel a running job: `POST /api/runs/{run_id}/cancel` (soft cancel, stops at next checkpoint).
- Auth:
  - Enable: `CALL2EDS_AUTH_ENABLED=true`
  - Session cookie via `/api/auth/login`
  - API key via `X-API-Key` or `Authorization: Bearer <key>`
  - Basic Auth supported
  - Manage users/keys: `/access`
- Swagger “Authorize” supports testing with credentials.
 - Before production: change default passwords (MinIO, Postgres) and enable auth.
 - For intranet use: put a TLS reverse proxy in front of UI/API.

### GPU
- CUDA image base.
- Auto GPU use if available.
- Force with `CALL2EDS_DEVICE=cuda` or `CALL2EDS_DEVICE=cpu`.

### Storage and DB
See French section (paths and tables identical).

### Limitations
- No automatic de-identification.
- Diarization: pyannote with HF_TOKEN or SpeechBrain fallback.
- Inputs validated server-side. Never trust the input.


---

## Credits & third-party
Not a fork. Built with open-source components including:
- Faster-Whisper (CTranslate2)
- openSMILE
- pyannote.audio
- SpeechBrain
- FastAPI, MinIO, Postgres

Please review upstream licenses before redistribution.

## Security
See `SECURITY.md` for the vulnerability disclosure policy.

## Disclaimer
FR: Ce projet est un outil technique. Il ne remplace pas un avis medical, ne doit pas etre utilise pour des decisions cliniques ou d'urgence, et n'offre aucune garantie de fiabilite sans validation locale. Il n'est pas un dispositif medical. Il est fourni "en l'etat", sans garantie, et son usage est sous la seule responsabilite de l'utilisateur. Il est construit pour aider la recherche libre et gratuite.

EN: This project is a technical tool. It does not replace medical advice, must not be used for clinical or emergency decisions, and provides no reliability guarantee without local validation. It is not a medical device. It is provided "as is", without warranty, and use is at the user's sole responsibility. It is built to support free and open research.

## License
MIT (see `pyproject.toml`).
