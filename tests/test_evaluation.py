from __future__ import annotations

from collections.abc import Callable

import pytest
import torch as t
from jaxtyping import TypeCheckError

from methylation_latent.domain import NonEmptyProbeSet, ProbeLocus
from methylation_latent.evaluation import (
    DISTANCE_CLASS_LABELS,
    DistanceBaseline,
    DistanceClass,
    PairIndices,
    PairPopulation,
    Projection2D,
    _combination_from_rank,
    _floyd_sample,
    _pair_limit,
    build_cross_partition_pairs,
    build_within_partition_pairs,
    classify_distances,
    fit_distance_baseline,
    fit_projection_2d,
    gather_pair_predictions,
    gather_pair_targets,
    metrics_by_distance,
    regression_metrics,
)
from methylation_latent.targets import standardize_rows


def _distance_probe_universe(make_probe: Callable[..., ProbeLocus]) -> NonEmptyProbeSet:
    positions = (
        1_000_000,
        1_000_999,
        1_001_000,
        1_004_000,
        1_016_000,
        1_064_000,
        1_256_000,
        2_000_000,
    )
    probes = [make_probe(index + 1, position=position) for index, position in enumerate(positions)]
    probes.append(make_probe(9, chromosome=2, position=1_000_000))
    return NonEmptyProbeSet(tuple(probes))


def test_distance_classes_respect_half_open_boundaries_and_trans(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _distance_probe_universe(make_probe)
    left = t.zeros(8, dtype=t.int64)
    right = t.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=t.int64)
    classes = classify_distances(probes, left, right)
    assert classes.tolist() == [
        DistanceClass.CIS_0_1KB,
        DistanceClass.CIS_1_4KB,
        DistanceClass.CIS_4_16KB,
        DistanceClass.CIS_16_64KB,
        DistanceClass.CIS_64_256KB,
        DistanceClass.CIS_256KB_1MB,
        DistanceClass.CIS_1MB_PLUS,
        DistanceClass.TRANS,
    ]
    assert len(DISTANCE_CLASS_LABELS) == 8


def test_cross_partition_pair_artifact_is_exact_or_uniformly_capped(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _distance_probe_universe(make_probe)
    seen = t.tensor([0, 1, 2, 3], dtype=t.int64)
    held_out = t.tensor([4, 5, 6], dtype=t.int64)
    complete = build_cross_partition_pairs(probes, seen, held_out, maximum_pairs=None, seed=5)
    assert complete.count == 12
    assert complete.total_possible_pairs == 12
    capped = build_cross_partition_pairs(probes, seen, held_out, maximum_pairs=5, seed=5)
    repeated = build_cross_partition_pairs(probes, seen, held_out, maximum_pairs=5, seed=5)
    assert capped.count == 5
    assert t.equal(capped.left, repeated.left)
    assert t.equal(capped.right, repeated.right)
    assert set(capped.left.tolist()) <= set(seen.tolist())
    assert set(capped.right.tolist()) <= set(held_out.tolist())


def test_within_partition_rank_mapping_matches_all_combinations(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _distance_probe_universe(make_probe)
    indices = t.tensor([0, 2, 4, 6, 8], dtype=t.int64)
    complete = build_within_partition_pairs(
        probes,
        indices,
        population=PairPopulation.HELD_OUT_BY_HELD_OUT,
        maximum_pairs=None,
        seed=2,
    )
    expected = {
        (left, right)
        for offset, left in enumerate(indices.tolist())
        for right in indices.tolist()[offset + 1 :]
    }
    assert set(zip(complete.left.tolist(), complete.right.tolist(), strict=True)) == expected
    capped = build_within_partition_pairs(
        probes,
        indices,
        population=PairPopulation.TRAINING_BY_TRAINING,
        maximum_pairs=4,
        seed=8,
    )
    assert capped.count == 4
    assert bool(t.all(capped.left < capped.right).item())


def test_pair_builders_reject_invalid_partitions(make_probe: Callable[..., ProbeLocus]) -> None:
    probes = _distance_probe_universe(make_probe)
    with pytest.raises(ValueError, match="disjoint"):
        build_cross_partition_pairs(
            probes,
            t.tensor([0, 1]),
            t.tensor([1, 2]),
            maximum_pairs=2,
            seed=1,
        )
    with pytest.raises(ValueError, match="invalid for within"):
        build_within_partition_pairs(
            probes,
            t.tensor([0, 1]),
            population=PairPopulation.SEEN_BY_HELD_OUT,
            maximum_pairs=None,
            seed=1,
        )
    with pytest.raises(ValueError, match="at least two"):
        build_within_partition_pairs(
            probes,
            t.tensor([0]),
            population=PairPopulation.TRAINING_BY_TRAINING,
            maximum_pairs=None,
            seed=1,
        )
    with pytest.raises(IndexError):
        classify_distances(probes, t.tensor([99]), t.tensor([0]))
    with pytest.raises(ValueError, match="non-empty int64"):
        build_cross_partition_pairs(probes, t.tensor([0.0]), t.tensor([1]), maximum_pairs=1, seed=1)
    with pytest.raises(ValueError, match="unique"):
        build_cross_partition_pairs(
            probes, t.tensor([0, 0]), t.tensor([1]), maximum_pairs=1, seed=1
        )
    with pytest.raises(ValueError, match="identical shapes"):
        classify_distances(probes, t.tensor([0, 1]), t.tensor([2]))


def test_pair_indices_reject_diagonal_duplicate_and_noncanonical_records() -> None:
    distances = t.tensor([0, 0], dtype=t.int64)
    with pytest.raises(ValueError, match="diagonal"):
        PairIndices(
            PairPopulation.SEEN_BY_HELD_OUT,
            t.tensor([1, 2]),
            t.tensor([1, 3]),
            distances,
            2,
            1,
        )
    with pytest.raises(ValueError, match="duplicate"):
        PairIndices(
            PairPopulation.SEEN_BY_HELD_OUT,
            t.tensor([1, 1]),
            t.tensor([2, 2]),
            distances,
            2,
            1,
        )
    with pytest.raises(ValueError, match="left < right"):
        PairIndices(
            PairPopulation.TRAINING_BY_TRAINING,
            t.tensor([3, 1]),
            t.tensor([2, 4]),
            distances,
            2,
            1,
        )


def test_pair_target_and_prediction_gathering() -> None:
    beta = t.tensor(
        [[0.1, 0.2, 0.8, 0.7], [0.7, 0.6, 0.2, 0.1], [0.1, 0.4, 0.2, 0.9]],
        dtype=t.float64,
    )
    rows = standardize_rows(beta)
    pairs = PairIndices(
        PairPopulation.HELD_OUT_BY_HELD_OUT,
        t.tensor([0, 0, 1]),
        t.tensor([1, 2, 2]),
        t.tensor([0, 0, 0]),
        3,
        1,
    )
    targets = gather_pair_targets(rows, pairs)
    expected = t.tensor(
        [
            rows.tensor[0] @ rows.tensor[1],
            rows.tensor[0] @ rows.tensor[2],
            rows.tensor[1] @ rows.tensor[2],
        ]
    )
    assert t.allclose(targets, expected)
    latent = t.nn.functional.normalize(
        t.randn((3, 4), generator=t.Generator().manual_seed(4)), dim=1
    )
    predictions = gather_pair_predictions(latent, pairs)
    assert predictions.shape == (3,)
    with pytest.raises(ValueError, match="unit norm"):
        gather_pair_predictions(latent * 2, pairs)


def _all_distance_pair_artifact() -> PairIndices:
    left = t.arange(0, 16, 2, dtype=t.int64).repeat_interleave(2)
    right = t.arange(1, 17, 2, dtype=t.int64).repeat_interleave(2) + t.tensor([0, 16] * 8)
    classes = t.arange(8, dtype=t.int64).repeat_interleave(2)
    return PairIndices(
        PairPopulation.TRAINING_BY_TRAINING,
        left,
        right,
        classes,
        16,
        9,
    )


def test_distance_baseline_is_training_only_and_requires_every_bin() -> None:
    pairs = _all_distance_pair_artifact()
    targets = t.linspace(-0.8, 0.7, 16, dtype=t.float64)
    baseline = fit_distance_baseline(pairs, targets)
    assert baseline.counts.tolist() == [2] * 8
    assert t.allclose(baseline.predict(pairs.distance_class), baseline.means.repeat_interleave(2))
    missing = PairIndices(
        PairPopulation.TRAINING_BY_TRAINING,
        t.tensor([0, 2]),
        t.tensor([1, 3]),
        t.tensor([0, 1]),
        2,
        1,
    )
    with pytest.raises(ValueError, match="do not cover"):
        fit_distance_baseline(missing, t.tensor([0.1, 0.2]))
    wrong_population = PairIndices(
        PairPopulation.HELD_OUT_BY_HELD_OUT,
        t.tensor([0, 2]),
        t.tensor([1, 3]),
        t.tensor([0, 1]),
        2,
        1,
    )
    with pytest.raises(ValueError, match="training-by-training"):
        fit_distance_baseline(wrong_population, t.tensor([0.1, 0.2]))
    with pytest.raises(ValueError, match="at least one"):
        DistanceBaseline(t.zeros(8), t.tensor([1, 1, 1, 1, 1, 1, 1, 0]))


def test_regression_metrics_and_distance_stratification() -> None:
    target = t.tensor([-1.0, -0.2, 0.3, 0.9], dtype=t.float64)
    prediction = t.tensor([-0.8, -0.1, 0.4, 0.7], dtype=t.float64)
    report = regression_metrics(target, prediction)
    assert report.count == 4
    assert report.mse == pytest.approx(float(t.mean((target - prediction) ** 2)))
    assert report.pearson > 0.9
    pairs = PairIndices(
        PairPopulation.SEEN_BY_HELD_OUT,
        t.tensor([0, 1, 2, 3]),
        t.tensor([4, 5, 6, 7]),
        t.tensor([0, 0, 7, 7]),
        4,
        1,
    )
    reports = metrics_by_distance(pairs, target, prediction)
    assert set(reports) == {DistanceClass.CIS_0_1KB, DistanceClass.TRANS}
    with pytest.raises(ValueError, match="constant target"):
        regression_metrics(t.ones(3), t.arange(3, dtype=t.float32))
    with pytest.raises(ValueError, match="constant prediction"):
        regression_metrics(t.arange(3, dtype=t.float32), t.ones(3))


def test_projection_is_fit_on_training_and_has_deterministic_orientation() -> None:
    train = t.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.1], [0.0, 1.0, 0.0], [0.0, -1.0, -0.1]],
        dtype=t.float64,
    )
    projection = fit_projection_2d(train)
    assert projection.axes.shape == (2, 3)
    assert all(float(axis[t.argmax(axis.abs())].item()) >= 0.0 for axis in projection.axes)
    held_out = projection.transform(t.tensor([[0.5, 0.5, 0.0]], dtype=t.float64))
    assert held_out.shape == (1, 2)
    with pytest.raises(ValueError, match="rank below two"):
        fit_projection_2d(t.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    with pytest.raises(ValueError, match="orthonormal"):
        Projection2D(t.zeros(2), t.ones((2, 2)))


@pytest.mark.parametrize(
    ("left", "right", "classes", "total", "message"),
    [
        (t.tensor([0], dtype=t.int32), t.tensor([1]), t.tensor([0]), 1, "int64"),
        (
            t.empty(0, dtype=t.int64),
            t.empty(0, dtype=t.int64),
            t.empty(0, dtype=t.int64),
            0,
            "empty",
        ),
        (t.tensor([0, 1]), t.tensor([2]), t.tensor([0, 1]), 2, "identical shapes"),
        (t.tensor([0]), t.tensor([1]), t.tensor([8]), 1, "unknown"),
        (t.tensor([-1]), t.tensor([1]), t.tensor([0]), 1, "non-negative"),
        (t.tensor([0, 1]), t.tensor([2, 3]), t.tensor([0, 0]), 1, "exceeds"),
    ],
)
def test_pair_index_artifact_rejects_structural_corruption(
    left: t.Tensor,
    right: t.Tensor,
    classes: t.Tensor,
    total: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PairIndices(PairPopulation.SEEN_BY_HELD_OUT, left, right, classes, total, 1)


def test_pair_sampling_helpers_reject_impossible_requests() -> None:
    generator = t.Generator().manual_seed(1)
    with pytest.raises(ValueError, match="sample size"):
        _floyd_sample(10, 0, generator)
    with pytest.raises(IndexError, match="rank"):
        _combination_from_rank(10, 3)
    with pytest.raises(ValueError, match="at least one"):
        _pair_limit(0, None)
    with pytest.raises(ValueError, match="explicit"):
        _pair_limit(10_000_001, None)
    with pytest.raises(ValueError, match="positive"):
        _pair_limit(10, 0)


def test_pair_gathering_and_distance_baseline_reject_corruption() -> None:
    pairs = PairIndices(
        PairPopulation.SEEN_BY_HELD_OUT,
        t.tensor([0]),
        t.tensor([2]),
        t.tensor([0]),
        1,
        1,
    )
    rows = standardize_rows(t.tensor([[0.1, 0.2], [0.2, 0.1]], dtype=t.float64))
    with pytest.raises(IndexError, match="outside"):
        gather_pair_targets(rows, pairs)
    with pytest.raises(TypeCheckError, match="normalized_latent"):
        gather_pair_predictions(t.ones(2), pairs)
    with pytest.raises(ValueError, match="finite"):
        gather_pair_predictions(t.full((3, 2), float("nan")), pairs)
    unit = t.nn.functional.normalize(t.ones((2, 2)), dim=1)
    with pytest.raises(IndexError, match="outside"):
        gather_pair_predictions(unit, pairs)
    valid = DistanceBaseline(t.zeros(8), t.ones(8, dtype=t.int64))
    with pytest.raises(ValueError, match="int64"):
        valid.predict(t.tensor([0], dtype=t.int32))
    with pytest.raises(ValueError, match="unknown"):
        valid.predict(t.tensor([8], dtype=t.int64))
    with pytest.raises(ValueError, match="shape or dtype"):
        DistanceBaseline(t.zeros(7), t.ones(8, dtype=t.int64))
    with pytest.raises(ValueError, match="finite"):
        DistanceBaseline(t.full((8,), float("nan")), t.ones(8, dtype=t.int64))
    with pytest.raises(ValueError, match="counts"):
        DistanceBaseline(t.zeros(8), t.ones(8, dtype=t.int32))
    training_pairs = _all_distance_pair_artifact()
    with pytest.raises(ValueError, match="shape"):
        fit_distance_baseline(training_pairs, t.ones(15))
    with pytest.raises(ValueError, match="finite"):
        fit_distance_baseline(training_pairs, t.full((16,), float("nan")))


def test_metrics_reject_nonfinite_small_and_misaligned_vectors() -> None:
    with pytest.raises(ValueError, match="at least two"):
        regression_metrics(t.tensor([0.1]), t.tensor([0.1]))
    with pytest.raises(ValueError, match="finite"):
        regression_metrics(t.tensor([0.1, float("nan")]), t.tensor([0.1, 0.2]))
    with pytest.raises(TypeError, match="dtypes differ"):
        regression_metrics(
            t.tensor([0.1, 0.2], dtype=t.float64), t.tensor([0.1, 0.2], dtype=t.float32)
        )
    pairs = PairIndices(
        PairPopulation.SEEN_BY_HELD_OUT,
        t.tensor([0, 1]),
        t.tensor([2, 3]),
        t.tensor([0, 0]),
        2,
        1,
    )
    with pytest.raises(ValueError, match="align"):
        metrics_by_distance(pairs, t.tensor([0.1]), t.tensor([0.1]))


def test_projection_rejects_incompatible_nonfinite_and_low_dimensional_inputs() -> None:
    projection = Projection2D(t.zeros(2), t.eye(2))
    with pytest.raises(ValueError, match="incompatible"):
        projection.transform(t.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        projection.transform(t.tensor([[float("nan"), 0.0]]))
    with pytest.raises(ValueError, match="three rows"):
        fit_projection_2d(t.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        fit_projection_2d(t.tensor([[1.0, 0.0], [0.0, 1.0], [float("nan"), 0.0]]))
    with pytest.raises(ValueError, match="shapes"):
        Projection2D(t.zeros(2), t.eye(3)[:2])
    with pytest.raises(TypeError, match="floating dtype"):
        Projection2D(t.zeros(2, dtype=t.int64), t.eye(2, dtype=t.int64))
    with pytest.raises(ValueError, match="finite"):
        Projection2D(t.zeros(2), t.tensor([[1.0, 0.0], [0.0, float("inf")]]))
