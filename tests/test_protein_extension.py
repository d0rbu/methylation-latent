from __future__ import annotations

import pytest
import torch as t
from hypothesis import given
from hypothesis import strategies as st

from methylation_latent.domain import (
    GenomicContext,
    InfiniumDesign,
    ManifestStrand,
    NonNegativeWeight,
    ProbeLocus,
    parse_autosome,
    parse_one_based_position,
    parse_probe_id,
    parse_window_size,
)
from methylation_latent.model import normalize_rows_strict
from methylation_latent.protein_extension import (
    CorrelationRectangle,
    FreeProteinVectors,
    GeneLocus,
    LinearProteinMapper,
    ProteinObjectiveTerms,
    ProteinPanel,
    ProteinSplit,
    ProteinTargets,
    build_protein_targets,
    cpg_protein_windows_overlap,
    protein_chunk_ranges,
    protein_objective,
    standardize_measurement_rows,
    target_blind_protein_split,
    tss_window_interval,
)
from methylation_latent.targets import (
    CorrelationMatrix,
    CorrelationVector,
    UnitNormRows,
    UnitNormVector,
    standardize_rows,
    standardize_vector,
)


def _probe(position: int = 10_001) -> ProbeLocus:
    return ProbeLocus(
        probe_id=parse_probe_id("cg00000001"),
        chromosome=parse_autosome(1),
        position=parse_one_based_position(position),
        context=GenomicContext.ISLAND,
        design=InfiniumDesign.TYPE_I,
        manifest_strand=ManifestStrand.FORWARD,
    )


def test_protein_panel_rejects_misaligned_and_nonfinite_values() -> None:
    values = t.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=t.float64)
    panel = ProteinPanel(("a", "b"), ("A", "B"), ("P1", "P2"), ("g1", "g2", "g3"), values)
    assert panel.values.shape == (2, 3)
    try:
        ProteinPanel(("a", "b"), ("A",), ("P1", "P2"), ("g1", "g2", "g3"), values)
    except ValueError as error:
        assert "identity axes" in str(error)
    else:
        raise AssertionError("misaligned identities were accepted")
    values[0, 0] = float("nan")
    try:
        ProteinPanel(("a", "b"), ("A", "B"), ("P1", "P2"), ("g1", "g2", "g3"), values)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite protein values were accepted")


def test_protein_targets_equal_torch_correlations() -> None:
    beta = t.tensor(
        [[0.1, 0.3, 0.8, 0.2], [0.7, 0.2, 0.4, 0.9], [0.3, 0.4, 0.2, 0.8]],
        dtype=t.float64,
    )
    proteins = t.tensor([[2.0, -1.0, 0.5, 3.0], [-2.0, 4.0, 1.0, 0.0]], dtype=t.float64)
    age_raw = t.tensor([30.0, 41.0, 55.0, 70.0], dtype=t.float64)
    targets = build_protein_targets(
        standardize_rows(beta),
        proteins,
        standardize_vector(age_raw),
    )
    joined = t.cat((beta, proteins, age_raw[None, :]), dim=0)
    expected = t.corrcoef(joined)
    assert t.allclose(targets.probe_protein.tensor, expected[:3, 3:5], atol=1e-12, rtol=0)
    assert t.allclose(targets.protein_gram.tensor, expected[3:5, 3:5], atol=1e-12, rtol=0)
    assert t.allclose(targets.direct_age.tensor, expected[3:5, 5], atol=1e-12, rtol=0)


def test_unbounded_standardization_and_correlation_rectangle_fail_loudly() -> None:
    standardized = standardize_measurement_rows(
        t.tensor([[-3.0, 0.0, 4.0], [8.0, 3.0, -2.0]], dtype=t.float64)
    )
    assert t.allclose(standardized.tensor.mean(1), t.zeros(2, dtype=t.float64), atol=1e-12)
    assert t.allclose(
        t.linalg.vector_norm(standardized.tensor, dim=1),
        t.ones(2, dtype=t.float64),
        atol=1e-12,
    )
    for invalid in (
        t.ones((2, 3), dtype=t.float32),
        t.tensor([[1.0, 1.0], [3.0, 4.0]], dtype=t.float64),
    ):
        try:
            standardize_measurement_rows(invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid measurement matrix was accepted")
    try:
        CorrelationRectangle(t.tensor([[1.1]], dtype=t.float64))
    except ValueError as error:
        assert "exceeds one" in str(error)
    else:
        raise AssertionError("out-of-range correlation was accepted")


def test_target_blind_split_is_exact_and_matches_frozen_partition() -> None:
    genes = (
        "ADM",
        "CCL24",
        "CCL19",
        "CXCL10",
        "CXCL11",
        "CXCL13",
        "CXCL5",
        "SELE",
        "EPCAM",
        "FLT3LG",
        "GDF15",
        "LGALS3",
        "HBEGF",
        "WFDC2",
        "IL12B",
        "IL2RA",
        "IL6R",
        "CCL2",
        "MDK",
        "PECAM1",
        "PGF",
        "KITLG",
        "TEK",
        "TNFRSF1A",
        "VEGFA",
        "KDR",
        "SRC",
        "TNFSF10",
        "IL27",
        "CXCL1",
        "FGF23",
        "IL18",
        "RETN",
        "CHI3L1",
        "IL1RL1",
        "HAVCR1",
        "THBD",
        "MMP10",
        "CCL4",
        "AGER",
        "MMP7",
        "CXCL6",
        "DKK1",
        "GAL",
        "AGRP",
        "CD40",
        "PLAT",
        "ESM1",
        "MMP12",
        "CX3CL1",
        "IKBKG",
        "RNASE3",
    )
    split = target_blind_protein_split(genes, validation_count=8, test_count=8, seed=902017)
    assert tuple(genes[index] for index in split.test_indices.tolist()) == (
        "IL2RA",
        "CCL24",
        "THBD",
        "KITLG",
        "MMP7",
        "HAVCR1",
        "IKBKG",
        "VEGFA",
    )
    assert tuple(genes[index] for index in split.validation_indices.tolist()) == (
        "CXCL1",
        "RETN",
        "MDK",
        "CCL2",
        "AGRP",
        "IL27",
        "CCL19",
        "IL18",
    )
    assert split.refit_indices.numel() == 44
    try:
        ProteinSplit(3, t.tensor([0]), t.tensor([1]), t.tensor([1], dtype=t.int64))
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("overlapping protein split was accepted")


def test_sequence_mappers_and_free_vectors_are_unit_sphere_points() -> None:
    t.manual_seed(3)
    mapper = LinearProteinMapper(feature_dimension=5, latent_dimension=3)
    latent = mapper.latent(t.randn((4, 5)))
    assert latent.shape == (4, 3)
    assert t.allclose(t.linalg.vector_norm(latent, dim=1), t.ones(4), atol=1e-6)
    free = FreeProteinVectors(protein_count=4, latent_dimension=3).latent()
    assert t.allclose(t.linalg.vector_norm(free, dim=1), t.ones(4), atol=1e-6)
    mapper.projection.weight.data.zero_()
    try:
        mapper.latent(t.ones((2, 5)))
    except ValueError as error:
        assert "zero latent" in str(error)
    else:
        raise AssertionError("zero protein latents were accepted")


def test_protein_objective_averages_blocks_separately_and_excludes_diagonal() -> None:
    probes = normalize_rows_strict(t.tensor([[1.0, 0.0], [0.0, 1.0]]))
    proteins = normalize_rows_strict(t.tensor([[1.0, 1.0], [1.0, -1.0]]))
    cross_target = t.zeros((2, 2))
    gram_target = t.tensor([[100.0, 0.5], [0.5, -100.0]])
    terms = protein_objective(
        probes,
        proteins,
        cross_target,
        gram_target,
        lambda_protein_pairs=NonNegativeWeight(2.0),
    )
    expected_cross = t.square(probes @ proteins.mT).mean()
    expected_pair = t.tensor(0.25)
    assert t.allclose(terms.cross_mse, expected_cross)
    assert t.allclose(terms.protein_pair_mse, expected_pair)
    assert t.allclose(terms.total, expected_cross + 2.0 * expected_pair)


def test_tss_windows_use_grch37_coordinates_and_plus_reference_strand() -> None:
    width = parse_window_size(1024)
    plus = GeneLocus("A", "1", 10_001, 12_000, 1)
    minus = GeneLocus("B", "1", 8_001, 10_001, -1)
    plus_interval = tss_window_interval(plus, width, chromosome_length=20_000)
    minus_interval = tss_window_interval(minus, width, chromosome_length=20_000)
    assert plus_interval == minus_interval
    assert plus_interval.start == 9_488
    assert cpg_protein_windows_overlap(
        _probe(10_001), plus, width, chromosome_length=20_000
    )
    assert not cpg_protein_windows_overlap(
        _probe(15_001), plus, width, chromosome_length=20_000
    )


@given(
    residue_count=st.integers(min_value=1, max_value=5_000),
    maximum_residues=st.integers(min_value=1, max_value=1_022),
)
def test_protein_chunking_never_truncates(
    residue_count: int,
    maximum_residues: int,
) -> None:
    chunks = protein_chunk_ranges(residue_count, maximum_residues=maximum_residues)
    flattened = tuple(index for chunk in chunks for index in chunk)
    assert flattened == tuple(range(residue_count))
    assert all(len(chunk) <= maximum_residues for chunk in chunks)


def test_direct_age_target_is_empirical_not_implied_by_latent_proximity() -> None:
    protein_rows = standardize_measurement_rows(
        t.tensor([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0]], dtype=t.float64)
    )
    age = UnitNormVector(t.tensor([-0.5, -0.5, 0.5, 0.5], dtype=t.float64))
    direct = protein_rows.tensor @ age.tensor
    arbitrary_age_direction = t.tensor([1.0, 0.0])
    arbitrary_protein_latent = t.eye(2)
    model_cosines = arbitrary_protein_latent @ arbitrary_age_direction
    assert not t.allclose(direct, model_cosines.to(t.float64))


@pytest.mark.parametrize(
    ("names", "genes", "accessions", "gsms", "values", "error_type"),
    [
        (("a", "a"), ("A", "B"), ("P1", "P2"), ("g1", "g2", "g3"), t.ones((2, 3), dtype=t.float64), ValueError),
        (("a", "b"), ("A", "B"), ("P1", "P2"), ("g1", "g1", "g3"), t.ones((2, 3), dtype=t.float64), ValueError),
        (("a", "b"), ("A", "B"), ("P1", "P2"), ("g1", "g2", "g3"), t.ones((2, 2), dtype=t.float64), ValueError),
        (("a", "b"), ("A", "B"), ("P1", "P2"), ("g1", "g2", "g3"), t.ones((2, 3), dtype=t.float32), TypeError),
    ],
)
def test_protein_panel_rejects_each_identity_axis_failure(
    names: tuple[str, ...],
    genes: tuple[str, ...],
    accessions: tuple[str, ...],
    gsms: tuple[str, ...],
    values: t.Tensor,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        ProteinPanel(names, genes, accessions, gsms, values)


@pytest.mark.parametrize(
    "values",
    [
        t.empty((0, 2), dtype=t.float64),
        t.ones(2, dtype=t.float64),
        t.ones((2, 2), dtype=t.int64),
        t.tensor([[float("nan"), 0.0]], dtype=t.float64),
    ],
)
def test_correlation_rectangle_rejects_invalid_semantics(values: t.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        CorrelationRectangle(values)


def test_protein_target_axes_are_checked_independently() -> None:
    rows = UnitNormRows(t.tensor([[-0.5, 0.5], [0.5, -0.5]], dtype=t.float64) * 2**0.5)
    gram = CorrelationMatrix(rows.tensor @ rows.tensor.mT)
    direct = CorrelationVector(t.tensor([0.0, 0.0], dtype=t.float64))
    rectangle = CorrelationRectangle(t.zeros((3, 2), dtype=t.float64))
    ProteinTargets(rows, gram, direct, rectangle)
    with pytest.raises(ValueError, match="Gram"):
        ProteinTargets(rows, CorrelationMatrix(t.eye(3, dtype=t.float64)), direct, rectangle)
    with pytest.raises(ValueError, match="protein-age"):
        ProteinTargets(rows, gram, CorrelationVector(t.zeros(3, dtype=t.float64)), rectangle)
    with pytest.raises(ValueError, match="probe-protein"):
        ProteinTargets(
            rows,
            gram,
            direct,
            CorrelationRectangle(t.zeros((3, 3), dtype=t.float64)),
        )


def test_measurement_and_target_axis_failures_are_not_coerced() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        standardize_measurement_rows(t.empty((0, 3), dtype=t.float64))
    with pytest.raises(ValueError, match="finite"):
        standardize_measurement_rows(t.tensor([[0.0, float("inf")]], dtype=t.float64))
    with pytest.raises(ValueError, match="subject axes"):
        build_protein_targets(
            standardize_rows(t.tensor([[0.0, 1.0, 0.5]], dtype=t.float64)),
            t.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=t.float64),
            standardize_vector(t.tensor([1.0, 2.0], dtype=t.float64)),
        )


@pytest.mark.parametrize(
    ("protein_count", "train", "validation", "test", "error_type"),
    [
        (2, t.tensor([0]), t.tensor([1]), t.tensor([0]), ValueError),
        (3, t.tensor([0], dtype=t.int32), t.tensor([1]), t.tensor([2]), TypeError),
        (3, t.empty(0, dtype=t.int64), t.tensor([1]), t.tensor([2]), ValueError),
    ],
)
def test_protein_split_rejects_invalid_partition_contracts(
    protein_count: int,
    train: t.Tensor,
    validation: t.Tensor,
    test: t.Tensor,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        ProteinSplit(protein_count, train, validation, test)


def test_target_blind_split_rejects_duplicate_names_and_invalid_counts() -> None:
    with pytest.raises(ValueError, match="unique"):
        target_blind_protein_split(("A", "A", "B"), validation_count=1, test_count=1, seed=1)
    with pytest.raises(ValueError, match="non-empty train"):
        target_blind_protein_split(("A", "B", "C"), validation_count=1, test_count=2, seed=1)


def test_protein_mapper_rejects_invalid_dimensions_features_and_free_vectors() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        LinearProteinMapper(feature_dimension=0, latent_dimension=2)
    mapper = LinearProteinMapper(feature_dimension=3, latent_dimension=2)
    for features in (
        t.empty((0, 3)),
        t.ones((2, 4)),
        t.ones((2, 3), dtype=t.float64),
        t.tensor([[1.0, 2.0, float("nan")]], dtype=t.float32),
    ):
        with pytest.raises((TypeError, ValueError)):
            mapper.latent(features)
    with pytest.raises(ValueError, match="protein count"):
        FreeProteinVectors(protein_count=0, latent_dimension=2)


def test_protein_objective_rejects_every_axis_and_scalar_failure() -> None:
    probes = t.eye(2)
    proteins = t.eye(2)
    cross = t.zeros((2, 2))
    gram = t.eye(2)
    failures = (
        (t.ones((2, 3)), proteins, cross, gram),
        (probes, proteins, t.zeros((2, 3)), gram),
        (probes, proteins, cross, t.eye(3)),
        (probes[:1], proteins[:1], t.zeros((1, 1)), t.ones((1, 1))),
        (probes, proteins, cross.to(t.float64), gram),
        (probes, proteins, t.tensor([[float("nan"), 0.0], [0.0, 0.0]]), gram),
        (probes, proteins, cross, t.tensor([[1.0, float("inf")], [0.0, 1.0]])),
    )
    for probe_values, protein_values, cross_values, gram_values in failures:
        with pytest.raises((TypeError, ValueError)):
            protein_objective(
                probe_values,
                protein_values,
                cross_values,
                gram_values,
                lambda_protein_pairs=NonNegativeWeight(1.0),
            )
    with pytest.raises(ValueError, match="finite floating scalar"):
        ProteinObjectiveTerms(t.tensor(float("nan")), t.tensor(0.0), t.tensor(0.0))


@pytest.mark.parametrize(
    "gene",
    [
        ("", "1", 1, 2, 1),
        ("A", "1", 0, 2, 1),
        ("A", "1", 1, 2, 0),
    ],
)
def test_gene_locus_and_tss_boundary_failures(gene: tuple[str, str, int, int, int]) -> None:
    with pytest.raises(ValueError):
        GeneLocus(*gene)
    valid = GeneLocus("A", "1", 2, 3, 1)
    with pytest.raises(ValueError, match="complete TSS window"):
        tss_window_interval(valid, parse_window_size(1024), chromosome_length=2_000)


@pytest.mark.parametrize(("residues", "maximum"), [(0, 10), (10, 0), (-1, 2)])
def test_protein_chunking_rejects_nonpositive_dimensions(residues: int, maximum: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        protein_chunk_ranges(residues, maximum_residues=maximum)
