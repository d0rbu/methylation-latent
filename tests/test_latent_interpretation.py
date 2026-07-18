from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
from jaxtyping import TypeCheckError

import methylation_latent.latent_interpretation as interpretation
import methylation_latent.latent_interpretation_protocol as interpretation_protocol
from methylation_latent.artifacts import sha256_file
from methylation_latent.domain import (
    GenomicContext,
    InfiniumDesign,
    ManifestStrand,
    NonEmptyProbeSet,
    ProbeLocus,
    parse_autosome,
    parse_one_based_position,
    parse_probe_id,
)
from methylation_latent.latent_interpretation import (
    ManifestAnnotationAudit,
    PairDecomposition,
    age_stratified_half_indices,
    deterministic_indices,
    exact_pair_decomposition,
    fit_linear_surrogate,
    gather_pair_products,
    linear_cka,
    load_manifest_interpretation_annotations,
    mean_neighbour_overlap,
    normalized_frobenius_cosine,
    participation_rank,
    spearman_brown,
    split_half_targets,
    stable_extreme_indices,
    standardize_rows,
    standardize_vector,
    stratified_sample_indices,
    strict_pearson,
)
from methylation_latent.latent_interpretation_protocol import (
    DisplayProtocol,
    GeometryProtocol,
    InterpretationParent,
    ReliabilityProtocol,
    load_latent_interpretation_protocol,
)


def test_frozen_interpretation_protocol_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = load_latent_interpretation_protocol(
        root / "configs" / "latent-interpretation-v1.toml"
    )
    assert protocol.windows == (1024, 4096)
    assert protocol.splits == ("diverse-blocks", "held-out-chromosome")
    assert protocol.parent("held-out-chromosome", 4096).model_sha256.startswith("86a0")


def test_age_stratified_halves_are_exact_deterministic_partitions() -> None:
    age = t.tensor((19.0, 20.0, 30.0, 31.0, 50.0, 51.0, 70.0, 71.0), dtype=t.float64)
    first, second = age_stratified_half_indices(age, seed=17)
    repeated = age_stratified_half_indices(age, seed=17)
    assert t.equal(first, repeated[0])
    assert t.equal(second, repeated[1])
    assert t.equal(t.sort(t.cat((first, second))).values, t.arange(8))
    assert not bool(t.isin(first, second).any().item())
    ordered = t.argsort(age, stable=True).reshape(-1, 2)
    assert all(int(t.isin(pair, first).sum().item()) == 1 for pair in ordered)


def test_distance_sample_is_capped_per_class_without_target_access() -> None:
    classes = t.tensor((0,) * 10 + (1,) * 3 + (7,) * 8, dtype=t.int64)
    selected = stratified_sample_indices(classes, maximum_per_class=4, seed=31)
    counts = t.bincount(classes.index_select(0, selected), minlength=8)
    assert counts.tolist() == [4, 3, 0, 0, 0, 0, 0, 4]
    assert t.equal(
        selected,
        stratified_sample_indices(classes, maximum_per_class=4, seed=31),
    )


def test_exact_age_decomposition_reconstructs_both_gram_values() -> None:
    generator = t.Generator().manual_seed(41)
    empirical_rows = standardize_rows(t.randn((7, 9), generator=generator, dtype=t.float64))
    empirical_age = standardize_vector(t.randn(9, generator=generator, dtype=t.float64))
    empirical_rho = empirical_rows @ empirical_age
    latent = t.randn((7, 4), generator=generator, dtype=t.float32)
    age_direction = t.randn(4, generator=generator, dtype=t.float32)
    left = t.tensor((0, 0, 1, 2, 3, 4), dtype=t.int64)
    right = t.tensor((1, 2, 2, 5, 6, 6), dtype=t.int64)
    target = t.sum(
        empirical_rows.index_select(0, left) * empirical_rows.index_select(0, right), dim=1
    )
    decomposition = exact_pair_decomposition(
        latent,
        age_direction,
        empirical_rows,
        empirical_age,
        empirical_rho,
        target,
        left,
        right,
        chunk_size=2,
    )
    assert t.allclose(
        decomposition.target_total,
        decomposition.target_age + decomposition.target_residual,
        atol=2.0e-12,
        rtol=0.0,
    )
    assert t.allclose(
        decomposition.prediction_total,
        decomposition.prediction_age + decomposition.prediction_residual,
        atol=2.0e-6,
        rtol=0.0,
    )


def test_rotation_invariant_representation_metrics() -> None:
    generator = t.Generator().manual_seed(43)
    values = t.randn((30, 5), generator=generator, dtype=t.float64)
    rotation, _ = t.linalg.qr(t.randn((5, 5), generator=generator, dtype=t.float64))
    rotated = values @ rotation
    assert linear_cka(values, rotated) == pytest.approx(1.0, abs=1.0e-12)
    assert mean_neighbour_overlap(values.float(), rotated.float(), neighbours=4) == 1.0
    assert normalized_frobenius_cosine(values, values) == pytest.approx(1.0)


def test_spectrum_and_correlation_helpers_fail_loudly() -> None:
    assert participation_rank(t.tensor((2.0, 2.0, 0.0), dtype=t.float64)) == 2.0
    assert spearman_brown(0.5) == pytest.approx(2.0 / 3.0)
    assert strict_pearson(t.tensor((1.0, 2.0, 3.0)), t.tensor((2.0, 4.0, 6.0))) == pytest.approx(
        1.0
    )
    with pytest.raises(ValueError, match="constant"):
        strict_pearson(t.ones(3), t.arange(3, dtype=t.float32))


def test_prediction_only_candidate_ranking_and_deterministic_sample() -> None:
    first = t.tensor((0.7, 0.5, -0.8, -0.3, 0.2), dtype=t.float64)
    second = t.tensor((0.6, 0.9, -0.4, -0.7, -0.1), dtype=t.float64)
    assert stable_extreme_indices(first, second, positive=True, maximum=2).tolist() == [0, 1]
    assert stable_extreme_indices(first, second, positive=False, maximum=2).tolist() == [2, 3]
    sample = deterministic_indices(100, 10, seed=47)
    assert sample.numel() == 10
    assert t.equal(sample, t.unique(sample, sorted=True))


def test_linear_surrogate_recovers_full_rank_solution() -> None:
    features = t.tensor(((1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.0, 3.0)), dtype=t.float64)
    target = features @ t.tensor((2.0, -0.5), dtype=t.float64)
    coefficients = fit_linear_surrogate(features, target)
    assert t.allclose(coefficients, t.tensor((2.0, -0.5), dtype=t.float64), atol=1.0e-12)


def test_manifest_annotation_join_requires_exact_locus_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "[Assay]\n"
        "IlmnID,Name,Infinium_Design_Type,Forward_Sequence,Genome_Build,CHR,MAPINFO,Strand,"
        "UCSC_CpG_Islands_Name,Relation_to_UCSC_CpG_Island,UCSC_RefGene_Name,"
        "UCSC_RefGene_Group,Enhancer,Regulatory_Feature_Group,DHS\n"
        "cg00000001,cg00000001,II,AAA[CG]AAA,37,1,100,F,chr1:90-110,Island,GENE1,"
        "TSS200,TRUE,Promoter_Associated,TRUE\n",
        encoding="utf-8",
    )
    probes = NonEmptyProbeSet(
        (
            ProbeLocus(
                probe_id=parse_probe_id("cg00000001"),
                chromosome=parse_autosome("1"),
                position=parse_one_based_position("100"),
                context=GenomicContext.ISLAND,
                design=InfiniumDesign.TYPE_II,
                manifest_strand=ManifestStrand.FORWARD,
            ),
        )
    )
    annotations, audit = load_manifest_interpretation_annotations(
        manifest,
        probes,
        expected_sha256=sha256_file(manifest),
    )
    assert audit.matched_probes == 1
    assert annotations["cg00000001"].refgene_names == ("GENE1",)
    assert annotations["cg00000001"].enhancer
    assert annotations["cg00000001"].regulatory
    assert annotations["cg00000001"].dhs


@pytest.mark.parametrize(
    ("values", "error"),
    (
        (t.empty(0, dtype=t.float64), TypeError),
        (t.tensor((1, 2), dtype=t.int64), TypeError),
        (t.tensor((1.0, float("nan")), dtype=t.float64), ValueError),
    ),
)
def test_interpretation_vector_contract_rejects_invalid_values(
    values: t.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        interpretation._require_vector(values, "values")


def test_interpretation_index_contracts_reject_invalid_values() -> None:
    with pytest.raises(TypeError, match="int64"):
        interpretation._require_indices(t.tensor((0.0,)), 2, "indices")
    with pytest.raises(IndexError, match="outside"):
        interpretation._require_indices(t.tensor((2,), dtype=t.int64), 2, "indices")
    with pytest.raises(ValueError, match="duplicates"):
        interpretation._require_indices(t.tensor((0, 0), dtype=t.int64), 2, "indices")
    with pytest.raises(TypeError, match="int64"):
        interpretation._require_pair_axis(t.empty(0, dtype=t.int64), 2, "pairs")
    with pytest.raises(IndexError, match="outside"):
        interpretation._require_pair_axis(t.tensor((-1,), dtype=t.int64), 2, "pairs")


def test_standardization_and_pearson_reject_invalid_inputs() -> None:
    with pytest.raises(TypeCheckError, match="right"):
        strict_pearson(t.tensor((1.0, 2.0)), t.tensor((1.0, 2.0, 3.0)))
    with pytest.raises(ValueError, match="finite"):
        standardize_rows(t.tensor(((1.0, float("inf")), (2.0, 3.0))))
    with pytest.raises(ValueError, match="constant row"):
        standardize_rows(t.ones((2, 3), dtype=t.float64))
    with pytest.raises(ValueError, match="constant vector"):
        standardize_vector(t.ones(3, dtype=t.float64))


def test_sampling_and_pair_gather_fail_loudly() -> None:
    age = t.arange(5, dtype=t.float64)
    with pytest.raises(ValueError, match="even subject"):
        age_stratified_half_indices(age, seed=1)
    with pytest.raises(ValueError, match="positive seed"):
        age_stratified_half_indices(t.arange(4, dtype=t.float64), seed=0)
    with pytest.raises(ValueError, match="invalid"):
        stratified_sample_indices(t.tensor((0, 1), dtype=t.int64), maximum_per_class=0, seed=1)
    with pytest.raises(ValueError, match="unknown class"):
        stratified_sample_indices(t.tensor((0, 8), dtype=t.int64), maximum_per_class=1, seed=1)
    rows = t.tensor(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))
    left = t.tensor((0, 1), dtype=t.int64)
    right = t.tensor((1, 2), dtype=t.int64)
    assert t.equal(gather_pair_products(rows, left, right, chunk_size=1), t.tensor((11.0, 39.0)))
    with pytest.raises(TypeError, match="floating matrix"):
        gather_pair_products(rows.to(t.int64), left, right, chunk_size=1)
    with pytest.raises(ValueError, match="indices or chunk"):
        gather_pair_products(rows, left, right[:1], chunk_size=1)


def _valid_decomposition() -> PairDecomposition:
    target_age = t.tensor((0.1, 0.2), dtype=t.float64)
    target_residual = t.tensor((0.2, 0.1), dtype=t.float64)
    prediction_age = t.tensor((0.05, 0.1), dtype=t.float64)
    prediction_residual = t.tensor((0.15, 0.2), dtype=t.float64)
    return PairDecomposition(
        target_total=target_age + target_residual,
        target_age=target_age,
        target_residual=target_residual,
        prediction_total=prediction_age + prediction_residual,
        prediction_age=prediction_age,
        prediction_residual=prediction_residual,
    )


def test_pair_decomposition_dataclass_rejects_axis_and_identity_drift() -> None:
    valid = _valid_decomposition()
    with pytest.raises(ValueError, match="aligned finite float64"):
        replace(valid, target_age=valid.target_age.float())
    with pytest.raises(ValueError, match="reconstruct"):
        replace(valid, target_total=valid.target_total + 0.1)


def test_exact_pair_decomposition_rejects_all_axis_drift() -> None:
    latent = t.randn((3, 2), generator=t.Generator().manual_seed(1), dtype=t.float32)
    rows = standardize_rows(
        t.randn((3, 4), generator=t.Generator().manual_seed(2), dtype=t.float64)
    )
    age = standardize_vector(t.arange(4, dtype=t.float64))
    rho = rows @ age
    left = t.tensor((0, 1), dtype=t.int64)
    right = t.tensor((1, 2), dtype=t.int64)
    target = gather_pair_products(rows, left, right, chunk_size=2)
    arguments = (latent, t.ones(2), rows, age, rho, target, left, right)
    with pytest.raises(TypeError, match="float32 latents"):
        exact_pair_decomposition(*((latent.double(),) + arguments[1:]), chunk_size=2)
    with pytest.raises(ValueError, match="probe axes"):
        exact_pair_decomposition(
            latent,
            arguments[1],
            rows,
            age,
            rho[:2],
            target,
            left,
            right,
            chunk_size=2,
        )
    with pytest.raises(ValueError, match="age axis"):
        exact_pair_decomposition(
            latent,
            arguments[1],
            rows,
            age[:3],
            rho,
            target,
            left,
            right,
            chunk_size=2,
        )
    with pytest.raises(ValueError, match="aligned float64"):
        exact_pair_decomposition(
            latent,
            arguments[1],
            rows,
            age,
            rho,
            target.float(),
            left,
            right,
            chunk_size=2,
        )


def test_split_half_target_contract_and_spearman_brown_edges() -> None:
    beta = t.tensor(
        ((0.1, 0.2, 0.4, 0.8), (0.8, 0.4, 0.2, 0.1), (0.2, 0.4, 0.3, 0.7)),
        dtype=t.float64,
    )
    age = t.tensor((20.0, 30.0, 40.0, 50.0), dtype=t.float64)
    probes = t.tensor((0, 2), dtype=t.int64)
    samples = t.tensor((0, 1, 2, 3), dtype=t.int64)
    rows, rho = split_half_targets(beta, age, probes, samples)
    assert rows.shape == (2, 4)
    assert rho.shape == (2,)
    with pytest.raises(TypeError, match="float64"):
        split_half_targets(beta.float(), age, probes, samples)
    with pytest.raises(ValueError, match="sample axes"):
        split_half_targets(beta, age[:3], probes, samples)
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        spearman_brown(float("nan"))
    with pytest.raises(ValueError, match="undefined"):
        spearman_brown(-1.0)


def test_rotation_invariant_helpers_reject_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        participation_rank(t.tensor((1.0, -0.1), dtype=t.float64))
    with pytest.raises(ValueError, match="zero spectrum"):
        participation_rank(t.zeros(2, dtype=t.float64))
    with pytest.raises(ValueError, match="aligned"):
        normalized_frobenius_cosine(t.ones(2), t.ones(3))
    with pytest.raises(ValueError, match="matching finite"):
        normalized_frobenius_cosine(t.ones(2), t.ones(2, dtype=t.float64))
    with pytest.raises(ValueError, match="zero tensor"):
        normalized_frobenius_cosine(t.zeros(2), t.zeros(2))
    with pytest.raises(ValueError, match="aligned floating matrices"):
        linear_cka(t.ones((2, 2)), t.ones((3, 2)))
    with pytest.raises(ValueError, match="constant representation"):
        linear_cka(t.ones((3, 2)), t.ones((3, 2)))
    with pytest.raises(ValueError, match="inputs or count"):
        mean_neighbour_overlap(t.ones((3, 2)), t.ones((3, 2)), neighbours=3)


def test_deterministic_ranking_and_surrogate_failure_contracts() -> None:
    with pytest.raises(ValueError, match="positive"):
        deterministic_indices(0, 2, seed=1)
    assert t.equal(deterministic_indices(2, 3, seed=1), t.arange(2))
    with pytest.raises(ValueError, match="predictions or maximum"):
        stable_extreme_indices(t.ones(2), t.ones(3), positive=True, maximum=1)
    with pytest.raises(ValueError, match="not enough"):
        stable_extreme_indices(t.ones(2), -t.ones(2), positive=True, maximum=1)
    features = t.tensor(((1.0, 0.0), (1.0, 1.0), (1.0, 2.0)), dtype=t.float64)
    target = t.ones(3, dtype=t.float64)
    with pytest.raises(ValueError, match="axes or dtypes"):
        fit_linear_surrogate(features.float(), target)
    with pytest.raises(ValueError, match="finite"):
        fit_linear_surrogate(features, t.tensor((1.0, float("nan"), 1.0), dtype=t.float64))
    with pytest.raises(ValueError, match="full column rank"):
        fit_linear_surrogate(t.ones((3, 2), dtype=t.float64), target)


def test_manifest_annotation_helpers_and_audit_reject_drift() -> None:
    assert interpretation._semicolon_values("A;;B") == ("A", "B")
    assert not interpretation._manifest_boolean("", "field")
    with pytest.raises(ValueError, match="unexpected"):
        interpretation._manifest_boolean("FALSE", "field")
    with pytest.raises(ValueError, match="incomplete"):
        ManifestAnnotationAudit("a" * 64, requested_probes=2, matched_probes=1)
    with pytest.raises(ValueError, match="source hash"):
        ManifestAnnotationAudit("short", requested_probes=1, matched_probes=1)


def _manifest_text(*rows: str) -> str:
    header = (
        "IlmnID,Name,Infinium_Design_Type,Forward_Sequence,Genome_Build,CHR,MAPINFO,Strand,"
        "UCSC_CpG_Islands_Name,Relation_to_UCSC_CpG_Island,UCSC_RefGene_Name,"
        "UCSC_RefGene_Group,Enhancer,Regulatory_Feature_Group,DHS\n"
    )
    return "[Assay]\n" + header + "".join(rows)


def test_manifest_annotation_join_rejects_hash_missing_duplicate_and_identity(
    tmp_path: Path,
) -> None:
    row = "cg00000001,cg00000001,II,AAA[CG]AAA,37,1,100,F,chr1:90-110,Island,G,Body,TRUE,,TRUE\n"
    probe = ProbeLocus(
        probe_id=parse_probe_id("cg00000001"),
        chromosome=parse_autosome("1"),
        position=parse_one_based_position("100"),
        context=GenomicContext.ISLAND,
        design=InfiniumDesign.TYPE_II,
        manifest_strand=ManifestStrand.FORWARD,
    )
    probes = NonEmptyProbeSet((probe,))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(_manifest_text(row), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash differs"):
        load_manifest_interpretation_annotations(manifest, probes, expected_sha256="0" * 64)
    missing = tmp_path / "missing.csv"
    missing.write_text(_manifest_text(row.replace("cg00000001", "cg00000002")), encoding="utf-8")
    with pytest.raises(ValueError, match="join differs"):
        load_manifest_interpretation_annotations(
            missing,
            probes,
            expected_sha256=sha256_file(missing),
        )
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(_manifest_text(row, row), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest_interpretation_annotations(
            duplicate,
            probes,
            expected_sha256=sha256_file(duplicate),
        )
    drift = tmp_path / "drift.csv"
    drift.write_text(_manifest_text(row.replace(",100,F,", ",101,F,")), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        load_manifest_interpretation_annotations(
            drift,
            probes,
            expected_sha256=sha256_file(drift),
        )


def test_interpretation_protocol_helpers_and_dataclasses_reject_drift() -> None:
    sha = "a" * 64
    with pytest.raises(TypeError, match="positive integer"):
        interpretation_protocol._positive(True, "value")
    with pytest.raises(TypeError, match="SHA-256"):
        interpretation_protocol._sha256("bad", "value")
    with pytest.raises(ValueError, match="fields differ"):
        interpretation_protocol._object({"table": {"extra": 1}}, "table", {"expected"})
    with pytest.raises(TypeError, match="non-empty integer array"):
        interpretation_protocol._positive_tuple([], "values")
    with pytest.raises(ValueError, match="duplicates"):
        interpretation_protocol._positive_tuple([1, 1], "values")
    with pytest.raises(ValueError, match="at least two"):
        ReliabilityProtocol((1,), 1, 1)
    with pytest.raises(ValueError, match="unique"):
        ReliabilityProtocol((1, 1), 1, 1)
    with pytest.raises(ValueError, match="positive"):
        ReliabilityProtocol((1, 2), 0, 1)
    with pytest.raises(ValueError, match="positive"):
        GeometryProtocol(1, 1, 2, 1, 0)
    with pytest.raises(ValueError, match="smaller"):
        GeometryProtocol(1, 1, 2, 2, 1)
    with pytest.raises(ValueError, match="positive"):
        DisplayProtocol(1, 1, 1, (1,), 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        DisplayProtocol(1, 1, 1, (2, 1), 1)
    with pytest.raises(ValueError, match="cannot exceed"):
        DisplayProtocol(1, 1, 1, (1,), 2)
    with pytest.raises(ValueError, match="split or window"):
        InterpretationParent("bad", 1024, sha, sha, sha)
    with pytest.raises(ValueError, match="hashes"):
        InterpretationParent("diverse-blocks", 1024, "bad", sha, sha)


def test_interpretation_protocol_identity_grid_and_lookup_fail_loudly() -> None:
    root = Path(__file__).resolve().parents[1]
    valid = load_latent_interpretation_protocol(root / "configs" / "latent-interpretation-v1.toml")
    with pytest.raises(ValueError, match="identity"):
        replace(valid, protocol_id="")
    with pytest.raises(ValueError, match="GPL13534"):
        replace(valid, gpl13534_sha256="bad")
    with pytest.raises(ValueError, match="frozen split/window grid"):
        replace(valid, parents=valid.parents[::-1])
    with pytest.raises(KeyError, match="no unique"):
        valid.parent("diverse-blocks", 16_384)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ('status = "post_hoc_hypothesis_generating"', 'status = "wrong"', "schema or status"),
        ("windows = [1024, 4096]", "windows = [1024]", "split/window grid"),
        ("[sources]", "[sources]\nextra = 1", "sources fields"),
        ('protocol_id = "gse87571-hg19-caduceus-ps-v2"', 'protocol_id = ""', "protocol_id"),
    ),
)
def test_interpretation_protocol_loader_rejects_frozen_config_drift(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    source = Path("configs/latent-interpretation-v1.toml").read_text(encoding="utf-8")
    assert source.count(old) == 1
    path = tmp_path / "config.toml"
    path.write_text(source.replace(old, new), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=message):
        load_latent_interpretation_protocol(path)
