from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    build_stratified_cross_partition_pairs,
    build_stratified_within_partition_pairs,
    build_within_partition_pairs,
    classify_distances,
    fit_distance_baseline,
    fit_projection_2d,
    gather_pair_predictions,
    gather_pair_predictions_chunked,
    gather_pair_targets,
    gather_pair_targets_chunked,
    metrics_by_distance,
    regression_metrics,
)
from methylation_latent.evaluation_cache import (
    CachedPairSet,
    EvaluationPairCache,
    EvaluationPairCacheIdentity,
    PairSetName,
    build_evaluation_pair_cache,
    load_evaluation_pair_cache,
    save_evaluation_pair_cache_exclusive,
)
from methylation_latent.targets import UnitNormRows, standardize_rows


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


def test_distance_stratified_pair_sampling_matches_brute_force_when_uncapped(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _distance_probe_universe(make_probe)
    all_indices = t.arange(len(probes), dtype=t.int64)
    expected_within = build_within_partition_pairs(
        probes,
        all_indices,
        population=PairPopulation.TRAINING_BY_TRAINING,
        maximum_pairs=None,
        seed=1,
    )
    observed_within = build_stratified_within_partition_pairs(
        probes,
        all_indices,
        population=PairPopulation.TRAINING_BY_TRAINING,
        maximum_pairs_per_distance_class=100,
        seed=1,
    )
    expected_pairs = set(
        zip(expected_within.left.tolist(), expected_within.right.tolist(), strict=True)
    )
    observed_pairs = set(
        zip(observed_within.left.tolist(), observed_within.right.tolist(), strict=True)
    )
    assert observed_pairs == expected_pairs
    assert set(observed_within.distance_class.tolist()) == set(range(8))

    seen = t.tensor([0, 2, 4, 8], dtype=t.int64)
    held_out = t.tensor([1, 3, 5, 6, 7], dtype=t.int64)
    expected_cross = build_cross_partition_pairs(
        probes,
        seen,
        held_out,
        maximum_pairs=None,
        seed=2,
    )
    observed_cross = build_stratified_cross_partition_pairs(
        probes,
        seen,
        held_out,
        maximum_pairs_per_distance_class=100,
        seed=2,
    )
    assert set(
        zip(observed_cross.left.tolist(), observed_cross.right.tolist(), strict=True)
    ) == set(zip(expected_cross.left.tolist(), expected_cross.right.tolist(), strict=True))


def test_distance_stratified_pair_sampling_is_capped_and_deterministic(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _distance_probe_universe(make_probe)
    indices = t.arange(len(probes), dtype=t.int64)
    first = build_stratified_within_partition_pairs(
        probes,
        indices,
        population=PairPopulation.HELD_OUT_BY_HELD_OUT,
        maximum_pairs_per_distance_class=1,
        seed=9,
    )
    second = build_stratified_within_partition_pairs(
        probes,
        indices,
        population=PairPopulation.HELD_OUT_BY_HELD_OUT,
        maximum_pairs_per_distance_class=1,
        seed=9,
    )
    assert t.equal(first.left, second.left)
    assert t.equal(first.right, second.right)
    assert int(t.bincount(first.distance_class, minlength=8).max().item()) == 1
    with pytest.raises(ValueError, match="must be positive"):
        build_stratified_within_partition_pairs(
            probes,
            indices,
            population=PairPopulation.TRAINING_BY_TRAINING,
            maximum_pairs_per_distance_class=0,
            seed=1,
        )


def test_evaluation_pair_cache_round_trips_fixed_pairs_and_targets(
    make_probe: Callable[..., ProbeLocus],
    tmp_path: Path,
) -> None:
    probes = _distance_probe_universe(make_probe)
    extended = NonEmptyProbeSet(
        (
            *probes.probes,
            make_probe(10, chromosome=1, position=1_002_000),
            make_probe(11, chromosome=2, position=2_000_000),
            make_probe(12, chromosome=3, position=3_000_000),
        )
    )
    generator = t.Generator().manual_seed(41)
    rows = standardize_rows(t.rand((len(extended), 7), generator=generator, dtype=t.float64))
    built = build_evaluation_pair_cache(
        extended,
        rows,
        t.arange(9, dtype=t.int64),
        t.arange(9, 12, dtype=t.int64),
        maximum_uniform_pairs=5,
        maximum_pairs_per_distance_class=3,
        uniform_seed=17,
        distance_seed=23,
        target_chunk_size=2,
    )
    assert built.pair_sets[PairSetName.SEEN_UNIFORM].pairs.count == 5
    assert built.pair_sets[PairSetName.HELD_OUT_UNIFORM].pairs.count == 3
    assert set(
        built.pair_sets[PairSetName.TRAINING_STRATIFIED].pairs.distance_class.tolist()
    ) == set(range(8))
    with pytest.raises(ValueError, match="global probe universe"):
        build_evaluation_pair_cache(
            extended,
            UnitNormRows(rows.tensor[:-1]),
            t.arange(9, dtype=t.int64),
            t.arange(9, 12, dtype=t.int64),
            maximum_uniform_pairs=5,
            maximum_pairs_per_distance_class=3,
            uniform_seed=17,
            distance_seed=23,
            target_chunk_size=2,
        )
    for uniform_cap, distance_cap, chunk_size in (
        (0, 3, 2),
        (5, 0, 2),
        (5, 3, 0),
    ):
        with pytest.raises(ValueError, match="must be positive"):
            build_evaluation_pair_cache(
                extended,
                rows,
                t.arange(9, dtype=t.int64),
                t.arange(9, 12, dtype=t.int64),
                maximum_uniform_pairs=uniform_cap,
                maximum_pairs_per_distance_class=distance_cap,
                uniform_seed=17,
                distance_seed=23,
                target_chunk_size=chunk_size,
            )
    example = built.pair_sets[PairSetName.SEEN_UNIFORM]
    with pytest.raises(TypeError, match="float64 vector"):
        CachedPairSet(example.pairs, example.targets.float())
    with pytest.raises(ValueError, match="different lengths"):
        CachedPairSet(example.pairs, t.zeros(example.pairs.count + 1, dtype=t.float64))
    nonfinite = example.targets.clone()
    nonfinite[0] = t.nan
    with pytest.raises(ValueError, match="finite"):
        CachedPairSet(example.pairs, nonfinite)
    unbounded = example.targets.clone()
    unbounded[0] = 1.1
    with pytest.raises(ValueError, match="correlation bounds"):
        CachedPairSet(example.pairs, unbounded)
    missing_pair_set = dict(built.pair_sets)
    del missing_pair_set[PairSetName.SEEN_UNIFORM]
    with pytest.raises(ValueError, match="five frozen"):
        EvaluationPairCache(missing_pair_set, built.distance_baseline)
    wrong_population = dict(built.pair_sets)
    wrong_population[PairSetName.SEEN_UNIFORM] = built.pair_sets[PairSetName.HELD_OUT_UNIFORM]
    with pytest.raises(ValueError, match="wrong pair population"):
        EvaluationPairCache(wrong_population, built.distance_baseline)
    with pytest.raises(ValueError, match="distance baseline differs"):
        EvaluationPairCache(
            built.pair_sets,
            replace(
                built.distance_baseline,
                means=built.distance_baseline.means + 0.01,
            ),
        )
    with pytest.raises(ValueError, match="protocol and split"):
        replace(
            EvaluationPairCacheIdentity(
                protocol_id="test",
                protocol_sha256="a" * 64,
                data_sha256="b" * 64,
                target_sha256="c" * 64,
                split_name="split",
                split_sha256="d" * 64,
                probe_order_sha256="e" * 64,
            ),
            protocol_id="",
        )
    with pytest.raises(ValueError, match="SHA-256"):
        EvaluationPairCacheIdentity(
            protocol_id="test",
            protocol_sha256="bad",
            data_sha256="b" * 64,
            target_sha256="c" * 64,
            split_name="split",
            split_sha256="d" * 64,
            probe_order_sha256="e" * 64,
        )
    identity = EvaluationPairCacheIdentity(
        protocol_id="test-protocol",
        protocol_sha256="a" * 64,
        data_sha256="b" * 64,
        target_sha256="c" * 64,
        split_name="test-split",
        split_sha256="d" * 64,
        probe_order_sha256="e" * 64,
    )
    directory = tmp_path / "pairs"
    save_evaluation_pair_cache_exclusive(directory, built, identity)
    loaded = load_evaluation_pair_cache(directory, identity)
    for name in PairSetName:
        assert t.equal(loaded.pair_sets[name].pairs.left, built.pair_sets[name].pairs.left)
        assert t.equal(loaded.pair_sets[name].pairs.right, built.pair_sets[name].pairs.right)
        assert t.equal(loaded.pair_sets[name].targets, built.pair_sets[name].targets)
    assert t.equal(loaded.distance_baseline.means, built.distance_baseline.means)
    with pytest.raises(FileExistsError):
        save_evaluation_pair_cache_exclusive(directory, built, identity)
    with pytest.raises(ValueError, match="identity differs"):
        load_evaluation_pair_cache(
            directory,
            EvaluationPairCacheIdentity(
                protocol_id="other-protocol",
                protocol_sha256="a" * 64,
                data_sha256="b" * 64,
                target_sha256="c" * 64,
                split_name="test-split",
                split_sha256="d" * 64,
                probe_order_sha256="e" * 64,
            ),
        )
    extra = tmp_path / "extra-file"
    shutil.copytree(directory, extra)
    (extra / "unknown").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="file inventory"):
        load_evaluation_pair_cache(extra, identity)

    def corrupted_copy(name: str) -> tuple[Path, dict[str, Any]]:
        destination = tmp_path / name
        shutil.copytree(directory, destination)
        metadata_path = destination / "metadata.json"
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        return destination, raw

    malformed, raw = corrupted_copy("malformed-envelope")
    raw["unknown"] = 1
    (malformed / "metadata.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata envelope"):
        load_evaluation_pair_cache(malformed, identity)

    fingerprint, raw = corrupted_copy("fingerprint")
    raw["payload_sha256"] = "0" * 64
    (fingerprint / "metadata.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="payload fingerprint"):
        load_evaluation_pair_cache(fingerprint, identity)

    short, raw = corrupted_copy("short-manifest")
    raw["pair_sets"] = raw["pair_sets"][:-1]
    (short / "metadata.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest length"):
        load_evaluation_pair_cache(short, identity)

    bad_fields, raw = corrupted_copy("record-fields")
    del raw["pair_sets"][0]["seed"]
    (bad_fields / "metadata.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="record fields"):
        load_evaluation_pair_cache(bad_fields, identity)

    duplicate, raw = corrupted_copy("duplicate-record")
    raw["pair_sets"][1]["name"] = raw["pair_sets"][0]["name"]
    (duplicate / "metadata.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="contains duplicates"):
        load_evaluation_pair_cache(duplicate, identity)

    wrong_count, raw = corrupted_copy("wrong-count")
    raw["pair_sets"][0]["count"] += 1
    (wrong_count / "metadata.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs"):
        load_evaluation_pair_cache(wrong_count, identity)

    wrong_type, raw = corrupted_copy("wrong-type")
    raw["pair_sets"][0]["total_possible_pairs"] = True
    (wrong_type / "metadata.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(TypeError, match="must be an integer"):
        load_evaluation_pair_cache(wrong_type, identity)


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
    assert t.equal(
        gather_pair_targets_chunked(rows, pairs, chunk_size=1),
        targets,
    )
    latent = t.nn.functional.normalize(
        t.randn((3, 4), generator=t.Generator().manual_seed(4)), dim=1
    )
    predictions = gather_pair_predictions(latent, pairs)
    assert predictions.shape == (3,)
    assert t.equal(
        gather_pair_predictions_chunked(latent, pairs, chunk_size=2),
        predictions,
    )
    with pytest.raises(ValueError, match="chunk size"):
        gather_pair_targets_chunked(rows, pairs, chunk_size=0)
    with pytest.raises(ValueError, match="chunk size"):
        gather_pair_predictions_chunked(latent, pairs, chunk_size=0)
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
