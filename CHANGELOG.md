# Changelog

## [0.1.0] - 2025-12-19
### Added
- Docker Compose stack (app + Postgres + MinIO) with persistent volumes and web UI.
- End-to-end ingestion pipeline: audio normalization (FFmpeg), ASR (faster-whisper), diarization (pyannote + SpeechBrain fallback), prosody extraction (openSMILE + pyworld), Parquet outputs, manifest, and MinIO storage.
- CLI commands: `ingest`, `show`, `export`, `doctor`, `purge`, `web`.
- Web UI: upload, run list, run detail with conversation view, prosody tags, and per‑turn audio snippets.
- Batch CSV ingest, public artifact download links, and export ZIP.
- Tests for manifest generation.

