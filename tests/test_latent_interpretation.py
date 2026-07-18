from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

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
    age_stratified_half_indices,
    deterministic_indices,
    exact_pair_decomposition,
    fit_linear_surrogate,
    linear_cka,
    load_manifest_interpretation_annotations,
    mean_neighbour_overlap,
    normalized_frobenius_cosine,
    participation_rank,
    spearman_brown,
    stable_extreme_indices,
    standardize_rows,
    standardize_vector,
    stratified_sample_indices,
    strict_pearson,
)
from methylation_latent.latent_interpretation_protocol import (
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
