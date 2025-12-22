# Changelog

## [Unreleased]
### Changed
- Removed typographic em-dashes in UI strings (replace with simple hyphen or "n/a").
- Swagger UI now supports Authorize with Basic/API key/Bearer.
- README refreshed and bilingual (FR/EN), with project name explanation and credits.
- Removed local build artifacts from repo root.
- Disabled run detail/zip actions while processing; clearer health/GPU labels.
- Added soft cancel for running jobs (API + UI).

## [0.1.8] - 2025-12-21
### Added
- Global ECAPA prototype resegmentation with safeguards (targeting low-confidence/short/overlap zones).
- Early forced resegmentation path mapped to speaker prototypes.
- New diarization tuning knobs in `.env.example` and README.

### Changed
- Default early prototype window/hop/smoothing tuned for tighter splits.
- Pipeline version bump to 0.1.8.

## [0.1.0] - 2025-12-19
### Added
- Docker Compose stack (app + Postgres + MinIO) with persistent volumes and web UI.
- End-to-end ingestion pipeline: audio normalization (FFmpeg), ASR (faster-whisper), diarization (pyannote + SpeechBrain fallback), prosody extraction (openSMILE + pyworld), Parquet outputs, manifest, and MinIO storage.
- CLI commands: `ingest`, `show`, `export`, `doctor`, `purge`, `web`.
- Web UI: upload, run list, run detail with conversation view, prosody tags, and per-turn audio snippets.
- Batch CSV ingest, public artifact download links, and export ZIP.
- Tests for manifest generation.
