import json

import pytest

from boldt_posttrain.bootstrap import BootstrapState, derive_state, run_bootstrap
from boldt_posttrain.data_pipeline import IntegrityError, canonical_json, sha256_bytes


def hashed(body):
    return {**body, "artifact_hash": sha256_bytes(canonical_json(body))}


def test_empty_fixture_reaches_research_ready(tmp_path):
    cfg = {"data": {"org": "openeurollm", "allowed_licenses": ["apache-2.0"]}}

    def discover():
        return hashed(
            {
                "status": "ok",
                "mode": "real",
                "run_id": "disc-1",
                "candidates": [
                    {
                        "dataset": "d",
                        "config": "de",
                        "split": "train",
                        "schema": "sft",
                        "org": "openeurollm",
                        "license": "apache-2.0",
                        "training_usable": True,
                        "german_sample_ratio": 1.0,
                        "row_count": 1,
                    }
                ],
            }
        )

    def prepare(_discovery, selection):
        body = {"status": "trainable", "mode": "real", "selection": selection["run_id"]}
        manifest = hashed(body)
        data = tmp_path / "data"
        (data / "manifest.json").write_text(json.dumps(manifest))
        (data / "leakage_report.json").write_text(json.dumps({"status": "verified_clean"}))
        return manifest

    def baseline():
        value = hashed({"status": "ok", "mode": "real", "technical_error_count": 0})
        path = tmp_path / "baseline/dev/current.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
        return value

    result = run_bootstrap(
        output_root=tmp_path, config=cfg, discover=discover, prepare=prepare, baseline=baseline
    )
    assert result["state"] == "RESEARCH_READY"
    assert derive_state(tmp_path) == BootstrapState.RESEARCH_READY


def test_bootstrap_never_overwrites_a_tampered_authoritative_artifact(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "discovery.json").write_text(
        json.dumps({"status": "ok", "mode": "real", "artifact_hash": "tampered"})
    )
    with pytest.raises(IntegrityError):
        derive_state(tmp_path)
