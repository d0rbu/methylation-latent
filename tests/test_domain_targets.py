from __future__ import annotations

from collections.abc import Callable

import pytest
import torch as t
from hypothesis import given
from hypothesis import strategies as st
from phantom import BoundError

from methylation_latent.domain import (
    AgeYears,
    Autosome,
    BetaValue,
    Correlation,
    Fraction,
    GenomicContext,
    LatentDimension,
    NonEmptyProbeSet,
    ProbeId,
    ProbeLocus,
    SentrixIdentity,
    WindowSize,
    parse_age_years,
    parse_autosome,
    parse_beta,
    parse_correlation,
    parse_fraction,
    parse_genomic_context,
    parse_gsm_accession,
    parse_latent_dimension,
    parse_non_negative_weight,
    parse_one_based_position,
    parse_positive_int,
    parse_probe_id,
    parse_sentrix_identity,
    parse_window_size,
)
from methylation_latent.targets import (
    CorrelationMatrix,
    CorrelationVector,
    TargetGeometry,
    UnitNormRows,
    UnitNormVector,
    assert_correlation_identity,
    build_target_geometry,
    correlation_block,
    standardize_rows,
    standardize_vector,
    target_rank_upper_bound,
    validate_latent_dimension,
)


@pytest.mark.parametrize(
    ("parser", "raw", "expected_type"),
    [
        (parse_beta, "0.25", BetaValue),
        (parse_correlation, -1, Correlation),
        (parse_fraction, 1, Fraction),
        (parse_autosome, "chr22", Autosome),
        (parse_window_size, 1024, WindowSize),
        (parse_latent_dimension, 16, LatentDimension),
        (parse_probe_id, "cg00000001", ProbeId),
        (parse_age_years, 67, AgeYears),
        (parse_sentrix_identity, "5815284001_R01C01", SentrixIdentity),
    ],
)
def test_domain_parsers_accept_valid_values(
    parser: Callable[..., object], raw: object, expected_type: type
) -> None:
    assert isinstance(parser(raw), expected_type)


@pytest.mark.parametrize(
    ("parser", "raw"),
    [
        (parse_beta, -0.01),
        (parse_beta, 1.01),
        (parse_correlation, 1.1),
        (parse_fraction, float("nan")),
        (parse_autosome, 0),
        (parse_autosome, 23),
        (parse_window_size, 3),
        (parse_window_size, 0),
        (parse_probe_id, "rs00000001"),
        (parse_age_years, 0),
        (parse_age_years, 131),
        (parse_gsm_accession, "GSE40279"),
        (parse_sentrix_identity, "5815284001_R07C01"),
    ],
)
def test_domain_parsers_reject_invalid_values(parser: Callable[..., object], raw: object) -> None:
    with pytest.raises((TypeError, ValueError, BoundError)):
        parser(raw)


@pytest.mark.parametrize(
    "parser",
    [
        parse_autosome,
        parse_one_based_position,
        parse_positive_int,
        parse_latent_dimension,
        parse_window_size,
    ],
)
def test_integer_domain_parsers_reject_boolean(parser: Callable[..., object]) -> None:
    with pytest.raises(TypeError, match="boolean"):
        parser(True)


def test_non_negative_weight_and_accession_parsers() -> None:
    assert float(parse_non_negative_weight(0)) == 0.0
    assert parse_gsm_accession("GSM989827") == "GSM989827"
    with pytest.raises((TypeError, ValueError, BoundError)):
        parse_non_negative_weight(-1)


@pytest.mark.parametrize(
    ("relation", "island", "expected"),
    [
        ("Island", "chr1:1-100", GenomicContext.ISLAND),
        ("N_Shore", "chr1:1-100", GenomicContext.SHORE),
        ("S_Shore", "chr1:1-100", GenomicContext.SHORE),
        ("N_Shelf", "chr1:1-100", GenomicContext.SHELF),
        ("S_Shelf", "chr1:1-100", GenomicContext.SHELF),
        ("", "", GenomicContext.OPEN_SEA),
    ],
)
def test_genomic_context_mapping(relation: str, island: str, expected: GenomicContext) -> None:
    assert parse_genomic_context(relation, island) == expected


@pytest.mark.parametrize(("relation", "island"), [("Island", ""), ("", "chr1:1-2"), ("other", "")])
def test_genomic_context_mapping_rejects_contradictions(relation: str, island: str) -> None:
    with pytest.raises(ValueError, match="unrecognized|requires"):
        parse_genomic_context(relation, island)


def test_non_empty_probe_set_rejects_empty_and_duplicate(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        NonEmptyProbeSet(())
    probe = make_probe(1)
    with pytest.raises(ValueError, match="duplicate"):
        NonEmptyProbeSet((probe, probe))
    assert len(NonEmptyProbeSet.from_iterable((probe,))) == 1


@pytest.mark.property
@given(
    st.lists(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=8,
        ),
        min_size=1,
        max_size=6,
    ).filter(
        lambda rows: (
            len({len(row) for row in rows}) == 1
            and all(max(row) - min(row) >= 1.0e-6 for row in rows)
        )
    )
)
def test_standardized_dot_product_is_pearson(raw: list[list[float]]) -> None:
    beta = t.tensor(raw, dtype=t.float64)
    rows = standardize_rows(beta)
    indices = t.arange(beta.shape[0], dtype=t.int64)
    assert assert_correlation_identity(beta, rows, indices, atol=1.0e-9) <= 1.0e-9


def test_target_geometry_exact_contract() -> None:
    beta = t.tensor(
        [[0.1, 0.4, 0.9, 0.2], [0.8, 0.6, 0.3, 0.1], [0.2, 0.3, 0.5, 0.7]],
        dtype=t.float64,
    )
    age = t.tensor([20.0, 40.0, 60.0, 80.0], dtype=t.float64)
    geometry = build_target_geometry(beta, age)
    assert t.allclose(
        geometry.methylation.tensor.mean(dim=1), t.zeros(3, dtype=t.float64), atol=1e-15
    )
    assert t.allclose(
        t.linalg.vector_norm(geometry.methylation.tensor, dim=1), t.ones(3, dtype=t.float64)
    )
    assert t.allclose(geometry.rho.tensor, geometry.methylation.tensor @ geometry.age.tensor)
    block = correlation_block(geometry.methylation, t.tensor([0, 2], dtype=t.int64))
    assert t.allclose(block.tensor.diagonal(), t.ones(2, dtype=t.float64), atol=1e-15)
    assert geometry.rank_upper_bound == 3


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (t.tensor([[0.1, 0.2]], dtype=t.float32), "Float64"),
        (t.tensor([[0.1, float("nan")]], dtype=t.float64), "finite"),
        (t.tensor([[0.1, 1.2]], dtype=t.float64), r"\[0, 1\]"),
        (t.tensor([[0.2, 0.2]], dtype=t.float64), "constant"),
        (t.empty((0, 2), dtype=t.float64), "non-empty"),
    ],
)
def test_standardize_rows_rejects_invalid_beta(values: t.Tensor, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        standardize_rows(values)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (t.tensor([1.0, 2.0], dtype=t.float32), "Float64"),
        (t.tensor([1.0, float("inf")], dtype=t.float64), "finite"),
        (t.tensor([2.0, 2.0], dtype=t.float64), "constant"),
        (t.empty(0, dtype=t.float64), "non-empty"),
    ],
)
def test_standardize_vector_rejects_invalid_age(values: t.Tensor, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        standardize_vector(values)


def test_semantic_tensor_wrappers_reject_illegal_states() -> None:
    with pytest.raises(ValueError, match="non-empty rank-2"):
        UnitNormRows(t.empty((0, 2), dtype=t.float64))
    with pytest.raises(ValueError, match="finite"):
        UnitNormRows(t.tensor([[float("nan"), 0.0]], dtype=t.float64))
    with pytest.raises(TypeError, match="float32 or float64"):
        UnitNormRows(t.tensor([[-0.707, 0.707]], dtype=t.float16))
    with pytest.raises(ValueError, match="zero-mean"):
        UnitNormRows(t.tensor([[1.0, 0.0]], dtype=t.float64))
    with pytest.raises(ValueError, match="unit norm"):
        UnitNormRows(t.tensor([[-1.0, 1.0]], dtype=t.float64))
    with pytest.raises(ValueError, match="non-empty rank-1"):
        UnitNormVector(t.empty(0, dtype=t.float64))
    with pytest.raises(ValueError, match="zero-mean"):
        UnitNormVector(t.tensor([0.0, 1.0], dtype=t.float64))
    with pytest.raises(ValueError, match="unit norm"):
        UnitNormVector(t.tensor([-1.0, 1.0], dtype=t.float64))
    with pytest.raises(ValueError, match="non-empty rank-1"):
        CorrelationVector(t.empty(0, dtype=t.float64))
    with pytest.raises(ValueError, match="finite"):
        CorrelationVector(t.tensor([float("inf")], dtype=t.float64))
    with pytest.raises(ValueError, match="exceeds one"):
        CorrelationVector(t.tensor([1.01], dtype=t.float64))
    with pytest.raises(ValueError, match="non-empty rank-2"):
        CorrelationMatrix(t.empty((0, 0), dtype=t.float64))
    with pytest.raises(ValueError, match="square"):
        CorrelationMatrix(t.ones((2, 3), dtype=t.float64))
    with pytest.raises(ValueError, match="exceeds one"):
        CorrelationMatrix(t.tensor([[1.0, 1.1], [1.1, 1.0]], dtype=t.float64))
    with pytest.raises(ValueError, match="symmetric"):
        CorrelationMatrix(t.tensor([[1.0, 0.2], [0.1, 1.0]], dtype=t.float64))
    with pytest.raises(ValueError, match="diagonal"):
        CorrelationMatrix(t.tensor([[0.9, 0.0], [0.0, 1.0]], dtype=t.float64))


def test_target_geometry_rejects_axis_mismatch() -> None:
    rows = UnitNormRows(t.tensor([[-(2**-0.5), 2**-0.5]], dtype=t.float64))
    age = UnitNormVector(t.tensor([-(2**-0.5), 2**-0.5], dtype=t.float64))
    with pytest.raises(ValueError, match="rho probe"):
        TargetGeometry(rows, age, CorrelationVector(t.tensor([1.0, 0.0], dtype=t.float64)))
    longer_age = UnitNormVector(t.tensor([-(6**-0.5), -(6**-0.5), 2 * (6**-0.5)], dtype=t.float64))
    with pytest.raises(ValueError, match="sample axes"):
        TargetGeometry(rows, longer_age, CorrelationVector(t.tensor([0.0], dtype=t.float64)))


def test_correlation_block_rejects_bad_indices() -> None:
    rows = standardize_rows(t.tensor([[0.1, 0.2, 0.3]], dtype=t.float64))
    with pytest.raises(TypeError, match="Int64"):
        correlation_block(rows, t.tensor([0], dtype=t.int32))
    with pytest.raises(IndexError):
        correlation_block(rows, t.tensor([1], dtype=t.int64))
    with pytest.raises(ValueError, match="non-empty"):
        correlation_block(rows, t.empty(0, dtype=t.int64))


def test_correlation_identity_rejects_misalignment_and_detects_error() -> None:
    raw = t.tensor([[0.1, 0.3, 0.8]], dtype=t.float64)
    rows = standardize_rows(raw)
    with pytest.raises(ValueError, match="identical shapes"):
        assert_correlation_identity(raw[:, :2], rows, t.tensor([0], dtype=t.int64))
    wrong = UnitNormRows(-rows.tensor)
    assert assert_correlation_identity(raw, wrong, t.tensor([0], dtype=t.int64)) == pytest.approx(
        0.0
    )
    raw_two = t.tensor([[0.1, 0.3, 0.8], [0.8, 0.4, 0.2]], dtype=t.float64)
    correct = standardize_rows(raw_two)
    corrupted = UnitNormRows(t.stack((-correct.tensor[0], correct.tensor[1])))
    with pytest.raises(ValueError, match="identity failed"):
        assert_correlation_identity(
            raw_two,
            corrupted,
            t.tensor([0, 1], dtype=t.int64),
            atol=1.0e-12,
        )


def test_rank_ceiling_and_latent_dimension_contract() -> None:
    assert target_rank_upper_bound(656) == 655
    assert (
        validate_latent_dimension(
            parse_latent_dimension(256), embedding_dimension=256, n_samples=656
        )
        == 256
    )
    with pytest.raises(ValueError, match="target ceiling=655"):
        validate_latent_dimension(
            parse_latent_dimension(656), embedding_dimension=1024, n_samples=656
        )
    with pytest.raises(ValueError, match="embedding ceiling=256"):
        validate_latent_dimension(
            parse_latent_dimension(257), embedding_dimension=256, n_samples=656
        )
    with pytest.raises(ValueError, match="at least two"):
        target_rank_upper_bound(1)


def test_empirical_gram_rank_is_bounded_by_centered_sample_dimension() -> None:
    generator = t.Generator().manual_seed(44)
    beta = t.rand((12, 8), generator=generator, dtype=t.float64)
    x = standardize_rows(beta).tensor
    gram = x @ x.mT
    assert int(t.linalg.matrix_rank(gram).item()) <= 7
