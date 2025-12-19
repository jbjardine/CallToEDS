from call2eds.utils.manifest import build_manifest


def test_manifest_basic():
    manifest = build_manifest(
        call_id="C1",
        run_id="R1",
        model_name="small",
        language="fr",
        artifacts=[{"key": "a", "sha256": "x", "size_bytes": 1}],
        params={"p": 1},
        stats={"duration_s": 1.0},
        ffmpeg_version="ffmpeg 6.1",
    )
    assert manifest["call_id"] == "C1"
    assert manifest["run_id"] == "R1"
    assert manifest["artifacts"][0]["key"] == "a"
