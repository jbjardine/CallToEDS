import json
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.table import Table

from call2eds.pipeline.pipeline import ingest_call, list_runs, export_artifacts, purge_run
from call2eds.pipeline.pipeline import doctor as doctor_fn
from call2eds.config.settings import settings

app = typer.Typer(help="Call2EDS CLI")


@app.command()
def web(host: str = typer.Option(None, help="Host binding"), port: int = typer.Option(None, help="Port")):
    """Démarre le serveur web (FastAPI + UI)."""
    host = host or settings.web_host
    port = port or settings.web_port
    import uvicorn

    uvicorn.run("call2eds.web.server:app", host=host, port=port, reload=False)


@app.command()
def ingest(
    audio_path: Path = typer.Argument(..., exists=True, readable=True, help="Fichier audio"),
    call_id: Optional[str] = typer.Option(None, help="Identifiant appel (sinon auto)"),
    lang: Optional[str] = typer.Option(None, help="Langue ASR (fr/en/auto)", metavar="LANG"),
    model: Optional[str] = typer.Option(None, help="Modèle faster-whisper (tiny/small/medium/large)"),
):
    """Traite un fichier audio et stocke les artefacts EDS."""
    run_id = ingest_call(
        audio_path=str(audio_path),
        call_id=call_id,
        language=lang or settings.call2eds_lang,
        model_size=model or settings.call2eds_model,
    )
    print(f"[green]OK[/green] run_id={run_id} call_id={call_id or 'auto'}")


@app.command(name="run")
def run_cmd(
    audio_path: Path = typer.Argument(..., exists=True, readable=True, help="Fichier audio"),
    call_id: Optional[str] = typer.Option(None, help="Identifiant appel (sinon auto)"),
    lang: Optional[str] = typer.Option(None, help="Langue ASR (fr/en/auto)", metavar="LANG"),
    model: Optional[str] = typer.Option(None, help="Modèle faster-whisper (tiny/small/medium/large)"),
):
    """Alias de ingest pour compatibilité UX."""
    ingest(audio_path=audio_path, call_id=call_id, lang=lang, model=model)


@app.command()
def show(call_id: str = typer.Argument(..., help="Identifiant appel")):
    """Liste les runs et artefacts d'un call_id."""
    runs = list_runs(call_id)
    if not runs:
        print(f"[yellow]Aucun run pour {call_id}[/yellow]")
        raise typer.Exit(code=0)
    table = Table(title=f"Runs pour {call_id}")
    table.add_column("run_id")
    table.add_column("created_at")
    table.add_column("status")
    table.add_column("model")
    for r in runs:
        table.add_row(r["run_id"], r["created_at"], r["status"], r.get("model", ""))
    print(table)


@app.command()
def export(
    call_id: str = typer.Argument(..., help="Identifiant appel"),
    out: Path = typer.Option(..., "--out", file_okay=False, dir_okay=True, writable=True, help="Dossier sortie"),
    run_id: Optional[str] = typer.Option(None, help="run_id (dernier par défaut)"),
):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = export_artifacts(call_id=call_id, run_id=run_id, out_dir=out_dir)
    print(f"[green]Exporté[/green] manifest={manifest_path}")


@app.command()
def doctor():
    """Vérifie les dépendances et connexions (ffmpeg, MinIO, Postgres)."""
    report = doctor_fn()
    print(json.dumps(report, indent=2))
    if not all(v.get("ok", False) for v in report.values()):
        raise typer.Exit(code=1)


@app.command()
def purge(
    call_id: str = typer.Argument(...),
    run_id: Optional[str] = typer.Option(None, help="run_id spécifique (sinon tous)"),
    yes: bool = typer.Option(False, "--yes", help="Confirmer suppression"),
):
    if not yes:
        print("Ajoutez --yes pour confirmer")
        raise typer.Exit(code=1)
    purge_run(call_id, run_id)
    print("[red]Supprimé[/red]")


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
