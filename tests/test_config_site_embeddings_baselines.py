from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch as t

from methylation_latent import cli
from methylation_latent import config as protocol_config
from methylation_latent.artifacts import Eligibility
from methylation_latent.baselines import SequenceAgeBaseline, fit_sequence_age_baseline
from methylation_latent.config import load_protocol_config
from methylation_latent.embeddings import (
    CADUCEUS_CHECKPOINT_SHA256,
    EmbeddingInferenceConfig,
    InferencePrecision,
    LoadedCaduceus,
    embed_all_sequences,
    embed_center_tokens,
    embed_sequence_batches,
    load_pinned_caduceus,
)
from methylation_latent.evaluation import PairPopulation
from methylation_latent.site import (
    AgeMetricPoint,
    DistanceMetricPoint,
    SiteData,
    SiteProvenance,
    WindowSweepPoint,
    _finite_number,
    _positive_integer,
    _string,
    build_static_site,
    load_site_data,
)
from methylation_latent.targets import CorrelationVector

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_protocol_configuration_is_strict_and_reviewed() -> None:
    config = load_protocol_config(REPOSITORY_ROOT / "configs/protocol-v1.toml")
    assert tuple(map(int, config.splits.window_sizes)) == (1024, 4096, 16384, 65536)
    assert tuple(map(int, config.sweep.latent_dimensions)) == (16, 32, 64, 128, 256)
    assert tuple(map(float, config.sweep.lambda_age)) == (0.1, 1.0, 10.0)
    assert int(config.data.expected_samples) == 656
    assert int(config.reference_audit.eligible_probes) == 403573
    assert config.reference_audit.boundary_exclusions == 15
    assert config.status == "draft"


def test_protocol_configuration_rejects_unknown_fields(tmp_path: Path) -> None:
    text = (REPOSITORY_ROOT / "configs/protocol-v1.toml").read_text(encoding="utf-8")
    path = tmp_path / "bad.toml"
    path.write_text(f"{text}\nunknown = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_protocol_config(path)
    path.write_text(text.replace("window_sizes = [1024", "window_sizes = [2048"), encoding="utf-8")
    with pytest.raises(ValueError, match="window sweep"):
        load_protocol_config(path)


def test_protocol_configuration_rejects_every_frozen_identity_and_sweep_drift() -> None:
    config = load_protocol_config(REPOSITORY_ROOT / "configs/protocol-v1.toml")
    invalid: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: replace(config.data, accession="GSE0"), "accession/build"),
        (lambda: replace(config.data, expected_samples=1), "exactly 656"),
        (lambda: replace(config.data, manifest_sha256="0" * 64), "manifest fingerprint"),
        (
            lambda: replace(config.data, series_matrix_sha256="0" * 64),
            "series.*fingerprint",
        ),
        (lambda: replace(config.data, reference_archive_md5="bad"), "reference archive"),
        (
            lambda: replace(config.reference_audit, eligible_probes=1),
            "reference-audit result",
        ),
        (lambda: replace(config.qc, processor="unknown"), "processor"),
        (lambda: replace(config.qc, parity_arrays=11), "at least 12"),
        (
            lambda: replace(config.qc, reference_sesame_version="devel"),
            "parity-reference identity",
        ),
        (lambda: replace(config.masks, chen_sha256="0" * 64), "probe-mask identity"),
        (lambda: replace(config.model, revision="main"), "Caduceus identity"),
        (lambda: replace(config.splits, primary_seed=config.splits.validation_seed), "must differ"),
        (
            lambda: replace(
                config.sweep,
                latent_dimensions=(
                    *config.sweep.latent_dimensions,
                    config.sweep.latent_dimensions[-1],
                ),
            ),
            "unique and increasing",
        ),
        (
            lambda: replace(
                config.sweep,
                latent_dimensions=(*config.sweep.latent_dimensions[:-1], 257),
            ),
            "exceeds Caduceus",
        ),
        (lambda: replace(config.sweep, lambda_age=(0.0,)), "positive"),
        (lambda: replace(config, schema="unknown"), "unknown protocol"),
        (lambda: replace(config, status="complete"), "draft.*frozen"),
        (lambda: replace(config, protocol_id=""), "must not be empty"),
    )
    for constructor, message in invalid:
        with pytest.raises(ValueError, match=message):
            constructor()


def test_protocol_parser_rejects_wrong_primitive_shapes() -> None:
    with pytest.raises(TypeError, match="table"):
        protocol_config._table({"data": 1}, "data")
    with pytest.raises(TypeError, match="integer"):
        protocol_config._integer(True, "value")
    with pytest.raises(TypeError, match="numeric"):
        protocol_config._number("1", "value")
    with pytest.raises(TypeError, match="non-empty string"):
        protocol_config._string("", "value")
    with pytest.raises(TypeError, match="non-empty array"):
        protocol_config._list([], "value")


def test_sequence_feature_baseline_fits_float64_ols_without_test_leakage() -> None:
    features = t.tensor([[0.1, 0.2], [0.2, 0.5], [0.4, 0.3], [0.7, 0.8]], dtype=t.float64)
    coefficients = t.tensor([0.05, 0.4, -0.2], dtype=t.float64)
    design = t.cat((t.ones((4, 1), dtype=t.float64), features), dim=1)
    rho = CorrelationVector(design @ coefficients)
    baseline = fit_sequence_age_baseline(features, rho)
    assert t.allclose(baseline.coefficients, coefficients, atol=1e-12)
    assert t.allclose(baseline.predict(features), rho.tensor, atol=1e-12)


def test_sequence_feature_baseline_rejects_bad_designs() -> None:
    rho = CorrelationVector(t.tensor([0.1, 0.2, 0.3], dtype=t.float64))
    with pytest.raises(ValueError, match="rank deficient"):
        fit_sequence_age_baseline(t.ones((3, 2), dtype=t.float64) / 2, rho)
    with pytest.raises(ValueError, match="exactly"):
        fit_sequence_age_baseline(t.ones((3, 3), dtype=t.float64) / 2, rho)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fit_sequence_age_baseline(
            t.tensor([[0.1, 0.2], [1.2, 0.1], [0.3, 0.4]], dtype=t.float64), rho
        )
    with pytest.raises(ValueError, match="length three"):
        SequenceAgeBaseline(t.ones(2, dtype=t.float64))
    with pytest.raises(ValueError, match="finite"):
        SequenceAgeBaseline(t.tensor([0.0, float("nan"), 0.0], dtype=t.float64))
    with pytest.raises(ValueError, match="finite"):
        fit_sequence_age_baseline(
            t.tensor([[0.1, 0.2], [0.2, float("nan")], [0.3, 0.4]], dtype=t.float64),
            rho,
        )


def test_sequence_baseline_fails_if_prediction_or_normal_equation_is_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enormous = SequenceAgeBaseline(t.full((3,), t.finfo(t.float64).max, dtype=t.float64))
    with pytest.raises(RuntimeError, match="non-finite"):
        enormous.predict(t.ones((1, 2), dtype=t.float64))

    features = t.tensor([[0.1, 0.2], [0.2, 0.5], [0.4, 0.3]], dtype=t.float64)
    rho = CorrelationVector(t.tensor([0.1, 0.2, 0.3], dtype=t.float64))
    monkeypatch.setattr(
        "methylation_latent.baselines.t.linalg.lstsq",
        lambda *_: SimpleNamespace(solution=t.zeros((3, 1), dtype=t.float64)),
    )
    with pytest.raises(RuntimeError, match="normal-equation audit"):
        fit_sequence_age_baseline(features, rho)


class FakeTokenizer:
    def __init__(self, *, extra_token: bool = False) -> None:
        self.extra_token = extra_token

    def __call__(
        self,
        sequences: Sequence[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        return_tensors: str,
    ) -> Mapping[str, t.Tensor]:
        assert not add_special_tokens
        assert not padding
        assert return_tensors == "pt"
        width = len(sequences[0]) + int(self.extra_token)
        return {"input_ids": t.zeros((len(sequences), width), dtype=t.int64)}


@dataclass
class FakeOutput:
    last_hidden_state: t.Tensor


class FakeModel:
    def __init__(self, *, hidden_delta: int = 0, nonfinite: bool = False) -> None:
        self.hidden_delta = hidden_delta
        self.nonfinite = nonfinite

    def __call__(self, **inputs: t.Tensor) -> FakeOutput:
        input_ids = inputs["input_ids"]
        batch, width = input_ids.shape
        hidden_width = 512 + self.hidden_delta
        hidden = t.arange(batch * width * hidden_width, dtype=t.float32).reshape(
            batch, width, hidden_width
        )
        if self.nonfinite:
            hidden[0, 0, 0] = float("nan")
        return FakeOutput(hidden)


def _embedding_config(batch_size: int = 2) -> EmbeddingInferenceConfig:
    return EmbeddingInferenceConfig(
        device="cpu",
        precision=InferencePrecision.FLOAT32,
        batch_size=batch_size,
    )


def test_center_token_embedding_disables_special_tokens_and_never_pools() -> None:
    sequences = ("AAACGT", "TTTCGA")
    result = embed_center_tokens(
        sequences,
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        config=_embedding_config(),
    )
    full = FakeModel()(input_ids=t.zeros((2, 6), dtype=t.int64)).last_hidden_state
    assert result.dtype == t.float16
    assert t.equal(result, full[:, 3, :256].to(t.float16))
    with pytest.raises(ValueError, match="token/base identity"):
        embed_center_tokens(
            sequences,
            tokenizer=FakeTokenizer(extra_token=True),
            model=FakeModel(),
            config=_embedding_config(),
        )
    with pytest.raises(ValueError, match="shape differs"):
        embed_center_tokens(
            sequences,
            tokenizer=FakeTokenizer(),
            model=FakeModel(hidden_delta=1),
            config=_embedding_config(),
        )


@pytest.mark.parametrize("sequence", ["AACATT", "AACNTT", "AACGT", ""])
def test_embedding_sequence_contract_rejects_invalid_windows(sequence: str) -> None:
    with pytest.raises(ValueError):
        embed_center_tokens(
            (sequence,),
            tokenizer=FakeTokenizer(),
            model=FakeModel(),
            config=_embedding_config(),
        )


def test_embedding_batches_cover_exact_probe_order() -> None:
    loaded = LoadedCaduceus(FakeTokenizer(), FakeModel(), Path("/snapshot"))
    sequences = ("AAACGT", "TTTCGA", "GGGCGC")
    matrix = embed_all_sequences(sequences, loaded=loaded, config=_embedding_config(2))
    assert matrix.tensor.shape == (3, 256)
    streamed = embed_sequence_batches(
        (("AAACGT", "TTTCGA"), ("GGGCGC",)),
        expected_probe_count=3,
        loaded=loaded,
        config=_embedding_config(2),
    )
    assert t.equal(streamed.tensor, matrix.tensor)
    with pytest.raises(ValueError, match="coverage differs"):
        embed_sequence_batches(
            (("AAACGT",),),
            expected_probe_count=2,
            loaded=loaded,
            config=_embedding_config(2),
        )


def test_fp16_embedding_config_rejects_cpu_execution() -> None:
    config = EmbeddingInferenceConfig(
        device="cpu",
        precision=InferencePrecision.FLOAT16,
        batch_size=1,
    )
    with pytest.raises(ValueError, match="only on CUDA"):
        embed_center_tokens(
            ("AAACGT",), tokenizer=FakeTokenizer(), model=FakeModel(), config=config
        )
    with pytest.raises(ValueError, match="frozen protocol"):
        EmbeddingInferenceConfig(
            device="cpu",
            precision=InferencePrecision.FLOAT32,
            batch_size=1,
            revision="main",
        )
    with pytest.raises(ValueError, match="batch size"):
        replace(_embedding_config(), batch_size=0)
    with pytest.raises(ValueError, match="checkpoint hash"):
        replace(_embedding_config(), checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError, match="exactly 256"):
        replace(_embedding_config(), embedding_width=255)


def test_embedding_pipeline_rejects_malformed_model_io_and_streams() -> None:
    loaded = LoadedCaduceus(FakeTokenizer(), FakeModel(), Path("/snapshot"))
    with pytest.raises(ValueError, match="same width"):
        embed_center_tokens(
            ("AAACGT", "AAAACGTT"),
            tokenizer=loaded.tokenizer,
            model=loaded.model,
            config=_embedding_config(),
        )

    class MissingIdsTokenizer(FakeTokenizer):
        def __call__(self, *args: object, **kwargs: object) -> Mapping[str, t.Tensor]:
            del args, kwargs
            return {}

    with pytest.raises(ValueError, match="lacks input_ids"):
        embed_center_tokens(
            ("AAACGT",),
            tokenizer=MissingIdsTokenizer(),
            model=loaded.model,
            config=_embedding_config(),
        )
    with pytest.raises(ValueError, match="non-finite"):
        embed_center_tokens(
            ("AAACGT",),
            tokenizer=loaded.tokenizer,
            model=FakeModel(nonfinite=True),
            config=_embedding_config(),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        embed_all_sequences((), loaded=loaded, config=_embedding_config())
    with pytest.raises(ValueError, match="expected probe count"):
        embed_sequence_batches(
            (), expected_probe_count=0, loaded=loaded, config=_embedding_config()
        )
    with pytest.raises(ValueError, match="invalid size"):
        embed_sequence_batches(
            ((),), expected_probe_count=1, loaded=loaded, config=_embedding_config()
        )
    with pytest.raises(ValueError, match="exceed"):
        embed_sequence_batches(
            (("AAACGT", "TTTCGA"),),
            expected_probe_count=1,
            loaded=loaded,
            config=_embedding_config(),
        )


def test_pinned_loader_verifies_snapshot_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"checkpoint")

    class AutoTokenizerStub:
        @classmethod
        def from_pretrained(cls, *_: object, **__: object) -> FakeTokenizer:
            return FakeTokenizer()

    class LoadedModelStub(FakeModel):
        config = SimpleNamespace(d_model=256, rcps=True)

        def to(self, _: str) -> LoadedModelStub:
            return self

        def eval(self) -> LoadedModelStub:
            return self

    class AutoModelStub:
        @classmethod
        def from_pretrained(cls, *_: object, **__: object) -> LoadedModelStub:
            return LoadedModelStub()

    import huggingface_hub
    import transformers

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: str(snapshot))
    monkeypatch.setattr(transformers, "AutoTokenizer", AutoTokenizerStub)
    monkeypatch.setattr(transformers, "AutoModel", AutoModelStub)
    monkeypatch.setattr(
        "methylation_latent.embeddings.sha256_file", lambda _: CADUCEUS_CHECKPOINT_SHA256
    )
    loaded = load_pinned_caduceus(
        cache_directory=tmp_path / "cache",
        config=_embedding_config(),
    )
    assert loaded.snapshot_path == snapshot
    (snapshot / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        load_pinned_caduceus(cache_directory=tmp_path / "cache", config=_embedding_config())
    (snapshot / "model.safetensors").write_bytes(b"checkpoint")
    monkeypatch.setattr("methylation_latent.embeddings.sha256_file", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="checkpoint hash differs"):
        load_pinned_caduceus(cache_directory=tmp_path / "cache", config=_embedding_config())


def test_empty_site_is_visible_and_nonvalidated_metrics_are_forbidden(tmp_path: Path) -> None:
    data = load_site_data(REPOSITORY_ROOT / "results/registry.json")
    assert data.status.startswith("blocked:")
    output = tmp_path / "site"
    build_static_site(
        data_path=REPOSITORY_ROOT / "results/registry.json",
        template_directory=REPOSITORY_ROOT / "site-template",
        output_directory=output,
    )
    assert (output / "index.html").is_file()
    assert json.loads((output / "results.json").read_text())["eligibility"] == "audit_only"
    with pytest.raises(FileExistsError):
        build_static_site(
            data_path=REPOSITORY_ROOT / "results/registry.json",
            template_directory=REPOSITORY_ROOT / "site-template",
            output_directory=output,
        )

    raw = json.loads((REPOSITORY_ROOT / "results/registry.json").read_text())
    raw["window_sweep"] = [
        {
            "window_size": 1024,
            "population": "held_out_by_held_out",
            "latent_dimension": 16,
            "lambda_age": 1.0,
            "mse": 0.1,
            "pearson": 0.2,
            "r_squared": 0.0,
        }
    ]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="non-validated"):
        load_site_data(invalid)


def _primary_site_payload() -> dict[str, object]:
    return {
        "schema": "methylation-latent.site-data.v1",
        "protocol_id": "protocol-v1",
        "eligibility": "primary_validated",
        "status": "validated",
        "artifact_ids": ["evaluation-one"],
        "provenance": {
            "split_family": "diverse_blocks",
            "data_sha256": "a" * 64,
            "split_sha256": "b" * 64,
            "retained_probe_count": 100,
            "retained_sample_count": 656,
        },
        "window_sweep": [
            {
                "window_size": 1024,
                "population": population,
                "latent_dimension": 16,
                "lambda_age": 1.0,
                "mse": 0.1,
                "pearson": 0.2,
                "r_squared": -0.1,
            }
            for population in ("seen_by_held_out", "held_out_by_held_out")
        ],
        "distance_metrics": [
            {
                "window_size": 1024,
                "population": population,
                "distance_class": "trans" if population == "seen_by_held_out" else "cis_1_4kb",
                "count": 10,
                "target_mean": 0.1,
                "prediction_mean": 0.08,
                "distance_baseline_mean": 0.03,
                "mse": 0.1,
                "pearson": 0.2,
                "r_squared": -0.1,
            }
            for population in ("seen_by_held_out", "held_out_by_held_out")
        ],
        "age_metrics": [
            {"window_size": 1024, "stage": stage, "count": 20, "mse": 0.1, "pearson": 0.2}
            for stage in ("sequence_features", "caduceus_age_only", "full_latent_metric")
        ],
        "projection": [
            {
                "probe_id": f"cg{index:08d}",
                "x": float(index),
                "y": -float(index),
                "context": context,
            }
            for index, context in enumerate(("island", "shore", "shelf", "open_sea"), start=1)
        ],
    }


def test_primary_validated_site_parses_every_panel(tmp_path: Path) -> None:
    path = tmp_path / "primary.json"
    path.write_text(json.dumps(_primary_site_payload()), encoding="utf-8")
    data = load_site_data(path)
    assert len(data.window_sweep) == 2
    assert len(data.distance_metrics) == 2
    assert len(data.age_metrics) == 3
    assert len(data.projection) == 4
    missing_stage = _primary_site_payload()
    missing_stage["age_metrics"] = cast(list[object], missing_stage["age_metrics"])[:-1]
    path.write_text(json.dumps(missing_stage), encoding="utf-8")
    with pytest.raises(ValueError, match="all three"):
        load_site_data(path)


def test_site_scalar_and_panel_types_reject_plausible_but_invalid_values() -> None:
    with pytest.raises(TypeError, match="numeric"):
        _finite_number(True, "value")
    with pytest.raises(ValueError, match="finite"):
        _finite_number(float("nan"), "value")
    with pytest.raises(TypeError, match="positive integer"):
        _positive_integer(0, "value")
    with pytest.raises(TypeError, match="non-empty string"):
        _string("", "value")

    population = PairPopulation.SEEN_BY_HELD_OUT
    with pytest.raises(ValueError, match="positive"):
        WindowSweepPoint(0, population, 16, 1.0, 0.1, 0.2, 0.0)
    with pytest.raises(ValueError, match="bounded"):
        WindowSweepPoint(1024, population, 16, 1.0, -0.1, 0.2, 0.0)
    with pytest.raises(ValueError, match="positive"):
        DistanceMetricPoint(0, population, "cis_0_1kb", 1, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0)
    with pytest.raises(ValueError, match="unknown distance"):
        DistanceMetricPoint(1024, population, "near", 1, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0)
    with pytest.raises(ValueError, match="bounded"):
        DistanceMetricPoint(1024, population, "cis_0_1kb", 1, 1.1, 0.0, 0.0, 0.1, 0.2, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        AgeMetricPoint(1024, "sequence_features", 1, -0.1, 0.2)
    with pytest.raises(ValueError, match="unknown age"):
        AgeMetricPoint(1024, "unknown", 1, 0.1, 0.2)


def test_site_envelope_rejects_identity_duplicates_and_incomplete_primary_panels() -> None:
    empty = SiteData(
        "methylation-latent.site-data.v1",
        "protocol-v1",
        Eligibility.AUDIT_ONLY,
        "blocked",
        (),
        None,
        (),
        (),
        (),
        (),
    )
    with pytest.raises(ValueError, match="schema"):
        replace(empty, schema="unknown")
    with pytest.raises(ValueError, match="unique"):
        replace(empty, artifact_ids=("same", "same"))
    with pytest.raises(ValueError, match="requires provenance"):
        replace(empty, eligibility=Eligibility.PRIMARY_VALIDATED)
    provenance = SiteProvenance("diverse_blocks", "a" * 64, "b" * 64, 100, 656)
    with pytest.raises(ValueError, match="cannot claim"):
        replace(empty, provenance=provenance)
    with pytest.raises(ValueError, match="split family"):
        replace(provenance, split_family="random_rows")
    with pytest.raises(ValueError, match="SHA-256"):
        replace(provenance, data_sha256="bad")
    with pytest.raises(ValueError, match="at least two"):
        replace(provenance, retained_sample_count=1)


def test_primary_site_rejects_duplicate_or_cross_window_incomplete_panels(tmp_path: Path) -> None:
    raw = _primary_site_payload()
    raw["window_sweep"] = [*cast(list[object], raw["window_sweep"])] * 2
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate window"):
        load_site_data(path)

    raw = _primary_site_payload()
    extra = cast(dict[str, object], cast(list[object], raw["window_sweep"])[0]).copy()
    extra["window_size"] = 4096
    cast(list[object], raw["window_sweep"]).append(extra)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="every window"):
        load_site_data(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_ids", "not-a-list", "artifact IDs"),
        ("window_sweep", "not-a-list", "array"),
        ("window_sweep", [1], "entries"),
    ],
)
def test_site_parser_rejects_wrong_json_shapes(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    raw = json.loads((REPOSITORY_ROOT / "results/registry.json").read_text())
    raw[field] = value
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(TypeError, match=message):
        load_site_data(path)


def test_cli_check_config_and_build_site(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "methylation-latent",
            "check-config",
            "--config",
            str(REPOSITORY_ROOT / "configs/protocol-v1.toml"),
        ],
    )
    cli.main()
    assert '"status": "draft"' in capsys.readouterr().out
    output = tmp_path / "cli-site"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "methylation-latent",
            "build-site",
            "--registry",
            str(REPOSITORY_ROOT / "results/registry.json"),
            "--template",
            str(REPOSITORY_ROOT / "site-template"),
            "--output",
            str(output),
        ],
    )
    cli.main()
    assert output.is_dir()
    assert '"eligibility": "audit_only"' in capsys.readouterr().out


def test_cli_public_audit_dispatch_and_hash_gate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = REPOSITORY_ROOT / "configs/protocol-v1.toml"
    manifest_path = Path("manifest.gz")
    arguments = [
        "methylation-latent",
        "audit-public-inputs",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest_path),
        "--series-matrix",
        "series.gz",
        "--sample-key",
        "key.gz",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(cli, "sha256_file", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="manifest hash differs"):
        cli.main()

    config = load_protocol_config(config_path)
    hashes = {
        "manifest.gz": config.data.manifest_sha256,
        "series.gz": config.data.series_matrix_sha256,
        "key.gz": config.data.sample_key_sha256,
    }
    monkeypatch.setattr(cli, "sha256_file", lambda path: hashes[path.name])
    monkeypatch.setattr(cli, "assert_gzip_integrity", lambda _: 1)
    monkeypatch.setattr(
        cli,
        "parse_gpl13534_manifest",
        lambda _: SimpleNamespace(probes=(1, 2), exclusions=(1,), total_assay_rows=3),
    )
    monkeypatch.setattr(cli, "parse_series_matrix_metadata", lambda *_args, **_kwargs: ("series",))

    # SimpleNamespace does not make a special method visible; use a tiny concrete class.
    class Samples:
        order_sha256 = config.data.sample_order_sha256

        def __len__(self) -> int:
            return 1

    class WrongSamples:
        order_sha256 = "f" * 64

        def __len__(self) -> int:
            return 1

    monkeypatch.setattr(cli, "join_sample_key", lambda *_: WrongSamples())
    with pytest.raises(ValueError, match="sample order hash differs"):
        cli.main()
    monkeypatch.setattr(cli, "join_sample_key", lambda *_: Samples())
    cli.main()
    assert '"manifest_rows": 3' in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="unreachable"):
        cli._unreachable("future-command")


def test_cli_exclusion_audit_reports_the_applied_union(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = REPOSITORY_ROOT / "configs/protocol-v1.toml"
    config = load_protocol_config(config_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "methylation-latent",
            "audit-exclusion-lists",
            "--config",
            str(config_path),
            "--manifest",
            "manifest.gz",
            "--chen",
            "chen.csv",
            "--zhou",
            "zhou.tsv.gz",
        ],
    )
    monkeypatch.setattr(cli, "sha256_file", lambda _: config.data.manifest_sha256)
    monkeypatch.setattr(
        cli,
        "load_chen2013_cross_reactive",
        lambda _: SimpleNamespace(sha256=config.masks.chen_sha256, probe_ids=(1, 2)),
    )
    monkeypatch.setattr(
        cli,
        "load_zhou2017_mask_general",
        lambda _: SimpleNamespace(sha256=config.masks.zhou_sha256, probe_ids=(2, 3, 4)),
    )
    monkeypatch.setattr(
        cli,
        "parse_gpl13534_manifest",
        lambda _: SimpleNamespace(probes=(1, 2, 3, 4, 5)),
    )
    monkeypatch.setattr(
        cli,
        "apply_published_exclusions",
        lambda *_: SimpleNamespace(
            retained=(5,),
            ledger=(1, 2, 2, 3, 4),
            unmatched_by_source={"Chen2013": (6,), "Zhou2017": ()},
        ),
    )
    cli.main()
    output = json.loads(capsys.readouterr().out)
    assert output["eligible_autosomal_cpgs_before_masks"] == 5
    assert output["exclusion_ledger_rows"] == 5
    assert output["retained_autosomal_cpgs"] == 1
    assert output["unmatched_by_source"] == {"Chen2013": 1, "Zhou2017": 0}


def test_cli_reference_audit_verifies_every_external_identity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = REPOSITORY_ROOT / "configs/protocol-v1.toml"
    config = load_protocol_config(config_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "methylation-latent",
            "audit-reference",
            "--config",
            str(config_path),
            "--manifest",
            "manifest.gz",
            "--chen",
            "chen.csv",
            "--zhou",
            "zhou.tsv.gz",
            "--reference-archive",
            "hg19.fa.gz",
            "--fasta",
            "hg19.fa",
            "--fasta-index",
            "hg19.fa.fai",
        ],
    )
    monkeypatch.setattr(cli, "sha256_file", lambda _: config.data.manifest_sha256)
    monkeypatch.setattr(cli, "md5_file", lambda _: config.data.reference_archive_md5)
    monkeypatch.setattr(
        cli,
        "assert_gzip_payload_matches_file",
        lambda *_: config.reference_audit.payload_sha256,
    )
    chen = SimpleNamespace(sha256=config.masks.chen_sha256)
    zhou = SimpleNamespace(sha256=config.masks.zhou_sha256)
    monkeypatch.setattr(cli, "load_chen2013_cross_reactive", lambda _: chen)
    monkeypatch.setattr(cli, "load_zhou2017_mask_general", lambda _: zhou)
    monkeypatch.setattr(cli, "parse_gpl13534_manifest", lambda _: SimpleNamespace(probes=(1, 2)))
    monkeypatch.setattr(
        cli,
        "apply_published_exclusions",
        lambda *_: SimpleNamespace(retained=(1, 2)),
    )

    class Reference:
        def __init__(self, *_: object) -> None:
            pass

        def __enter__(self) -> Reference:
            return self

        def __exit__(self, *_: object) -> None:
            pass

    monkeypatch.setattr(cli, "IndexedFasta", Reference)
    monkeypatch.setattr(
        cli,
        "audit_reference_windows",
        lambda *_: SimpleNamespace(
            boundary_exclusions=config.reference_audit.boundary_exclusions,
            coordinate_cpgs_verified=int(config.reference_audit.coordinate_cpgs_verified),
            eligible_probe_order_sha256=config.reference_audit.eligible_probe_order_sha256,
            eligible_probes=int(config.reference_audit.eligible_probes),
            input_probes=int(config.reference_audit.coordinate_cpgs_verified),
            non_acgt_exclusions=config.reference_audit.non_acgt_exclusions,
        ),
    )
    cli.main()
    output = json.loads(capsys.readouterr().out)
    assert output["coordinate_cpgs_verified"] == 405610
    assert output["eligible_probes"] == 403573
    assert output["maximum_window_size"] == 65536
