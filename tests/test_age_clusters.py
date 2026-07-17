from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch as t

from methylation_latent.age_clusters import (
    CategoricalAssociation,
    ContinuousContrast,
    KMeansTwoConfig,
    KMeansTwoResult,
    SubgroupCorrelations,
    adjusted_rand_index,
    categorical_association,
    continuous_contrast,
    fit_kmeans_two,
    permutation_invariant_label_agreement,
    subgroup_age_correlations,
)


def _config() -> KMeansTwoConfig:
    return KMeansTwoConfig(restarts=12, maximum_iterations=100, seed=17)


def test_kmeans_two_is_deterministic_standardized_and_orders_empirical_centers() -> None:
    low = t.tensor(((-3.0, -0.2), (-2.5, -0.1), (-2.0, -0.3)), dtype=t.float64)
    high = t.tensor(((1.0, 0.2), (2.0, 0.4), (3.0, 0.3)), dtype=t.float64)
    values = t.cat((low, high))
    first = fit_kmeans_two(values, config=_config())
    second = fit_kmeans_two(values, config=_config())
    assert t.equal(first.labels, t.tensor((0, 0, 0, 1, 1, 1)))
    assert t.equal(first.labels, second.labels)
    assert t.equal(first.centers, second.centers)
    assert first.inertia == second.inertia
    assert first.centers[0, 0] < first.centers[1, 0]


@pytest.mark.parametrize(
    ("values", "message"),
    (
        (t.ones((4, 2), dtype=t.float32), "float64"),
        (t.ones((3, 2), dtype=t.float64), "at least four"),
        (t.ones((4, 3), dtype=t.float64), "shape"),
        (t.tensor(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0))), "float64"),
        (
            t.tensor(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)), dtype=t.float64),
            "constant column",
        ),
    ),
)
def test_kmeans_two_rejects_invalid_geometry(values: t.Tensor, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        fit_kmeans_two(values, config=_config())


def test_kmeans_two_rejects_nonfinite_input_and_nonconvergence() -> None:
    values = t.tensor(((0.0, 0.0), (1.0, 1.0), (2.0, 3.0), (4.0, 5.0)), dtype=t.float64)
    invalid = values.clone()
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_kmeans_two(invalid, config=_config())
    with pytest.raises(RuntimeError, match="converge"):
        fit_kmeans_two(values, config=replace(_config(), maximum_iterations=1))


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"centers": t.ones((2, 3), dtype=t.float64)}, "shape"),
        ({"inertia": -1.0}, "inertia"),
        ({"iterations": 0}, "iteration"),
        ({"centers": t.tensor(((2.0, 0.0), (1.0, 0.0)), dtype=t.float64)}, "ordered"),
    ),
)
def test_kmeans_result_rejects_invalid_state(change: dict[str, object], message: str) -> None:
    valid = KMeansTwoResult(
        labels=t.tensor((0, 0, 1, 1), dtype=t.int64),
        centers=t.tensor(((0.0, 0.0), (1.0, 1.0)), dtype=t.float64),
        standardized_centers=t.tensor(((-1.0, -1.0), (1.0, 1.0)), dtype=t.float64),
        inertia=1.0,
        iterations=2,
    )
    with pytest.raises(ValueError, match=message):
        replace(valid, **change)


@pytest.mark.parametrize("change", ({"restarts": 0}, {"maximum_iterations": 0}))
def test_kmeans_config_rejects_nonpositive_counts(change: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        replace(_config(), **change)


def test_categorical_and_continuous_cluster_associations_are_exact() -> None:
    labels = t.tensor((0, 0, 0, 1, 1, 1), dtype=t.int64)
    categories = t.tensor((0, 0, 1, 1, 2, 2), dtype=t.int64)
    categorical = categorical_association(labels, categories)
    assert t.equal(categorical.contingency, t.tensor(((2, 1, 0), (0, 1, 2))))
    assert categorical.cramer_v == pytest.approx((2.0 / 3.0) ** 0.5)
    contrast = continuous_contrast(
        labels,
        t.tensor((0.0, 1.0, 2.0, 4.0, 5.0, 6.0), dtype=t.float64),
    )
    assert contrast.means == (1.0, 5.0)
    assert contrast.standard_deviations == (1.0, 1.0)
    assert contrast.standardized_mean_difference == 4.0


def test_association_functions_reject_invalid_labels_categories_and_values() -> None:
    labels = t.tensor((0, 0, 1, 1), dtype=t.int64)
    with pytest.raises(TypeError, match="labels"):
        categorical_association(labels.to(t.float64), t.tensor((0, 0, 1, 1)))
    with pytest.raises(ValueError, match="exactly"):
        categorical_association(t.zeros(4, dtype=t.int64), t.tensor((0, 0, 1, 1)))
    with pytest.raises(TypeError, match="categories"):
        categorical_association(labels, t.tensor((0.0, 0.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match="align"):
        categorical_association(labels, t.tensor((0, 0, 1), dtype=t.int64))
    singleton = categorical_association(labels, t.zeros(4, dtype=t.int64))
    assert singleton.cramer_v == 0.0
    with pytest.raises(TypeError, match="float64"):
        continuous_contrast(labels, t.ones(4, dtype=t.float32))
    with pytest.raises(ValueError, match="aligned"):
        continuous_contrast(labels, t.tensor((0.0, 1.0, 2.0), dtype=t.float64))
    with pytest.raises(ValueError, match="at least two"):
        continuous_contrast(
            t.tensor((0, 0, 0, 1), dtype=t.int64),
            t.arange(4, dtype=t.float64),
        )
    with pytest.raises(ValueError, match="constant"):
        continuous_contrast(labels, t.ones(4, dtype=t.float64))


@pytest.mark.parametrize(
    "constructor",
    (
        lambda: CategoricalAssociation(t.tensor(((1, -1), (1, 1))), 0.2),
        lambda: CategoricalAssociation(t.ones((2, 2), dtype=t.int64), 2.0),
        lambda: ContinuousContrast((float("nan"), 1.0), (1.0, 1.0), 0.0),
        lambda: ContinuousContrast((0.0, 1.0), (-1.0, 1.0), 0.0),
        lambda: SubgroupCorrelations(t.ones(2, dtype=t.float32), 3),
        lambda: SubgroupCorrelations(t.ones(2, dtype=t.float64), 2),
        lambda: SubgroupCorrelations(t.tensor((0.0, float("nan"))), 3),
    ),
)
def test_analysis_result_types_reject_invalid_state(constructor: Callable[[], object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        constructor()


def test_partition_agreement_is_invariant_to_binary_label_permutation() -> None:
    first = t.tensor((0, 0, 1, 1, 1, 0), dtype=t.int64)
    second = 1 - first
    assert permutation_invariant_label_agreement(first, second) == 1.0
    assert adjusted_rand_index(first, second) == 1.0


def test_partition_agreement_rejects_misaligned_or_degenerate_labels() -> None:
    first = t.tensor((0, 0, 1, 1), dtype=t.int64)
    with pytest.raises(ValueError, match="align"):
        permutation_invariant_label_agreement(first, t.tensor((0, 1), dtype=t.int64))
    with pytest.raises(ValueError, match="undefined"):
        adjusted_rand_index(
            t.tensor((0, 1), dtype=t.int64),
            t.tensor((0, 1), dtype=t.int64),
        )


def test_subgroup_age_correlations_restandardize_both_axes() -> None:
    raw = t.tensor(
        (
            (0.1, 0.2, 0.4, 0.8, 0.5, 0.7),
            (0.8, 0.7, 0.5, 0.1, 0.4, 0.2),
        ),
        dtype=t.float64,
    )
    globally_centered = raw - raw.mean(dim=1, keepdim=True)
    globally_standardized = globally_centered / t.linalg.vector_norm(
        globally_centered, dim=1, keepdim=True
    )
    raw_age = t.tensor((20.0, 30.0, 40.0, 50.0, 60.0, 70.0), dtype=t.float64)
    age = raw_age - raw_age.mean()
    age /= t.linalg.vector_norm(age)
    mask = t.tensor((True, False, True, False, True, True))
    observed = subgroup_age_correlations(globally_standardized, age, mask)
    selected_raw = raw[:, mask]
    selected_raw -= selected_raw.mean(dim=1, keepdim=True)
    selected_raw /= t.linalg.vector_norm(selected_raw, dim=1, keepdim=True)
    selected_age = raw_age[mask] - raw_age[mask].mean()
    selected_age /= t.linalg.vector_norm(selected_age)
    assert observed.sample_count == 4
    assert t.allclose(observed.correlations, selected_raw @ selected_age, atol=1e-15, rtol=0.0)


def test_subgroup_age_correlations_reject_constant_probe_in_subgroup() -> None:
    methylation = t.tensor(((0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 2.0, 3.0)), dtype=t.float64)
    age = t.tensor((-0.5, -0.5, 0.5, 0.5), dtype=t.float64)
    mask = t.tensor((True, True, True, False))
    with pytest.raises(ValueError, match="constant methylation"):
        subgroup_age_correlations(methylation, age, mask)


def test_subgroup_age_correlations_reject_invalid_axes_and_constant_age() -> None:
    methylation = t.tensor(((0.0, 1.0, 2.0, 3.0),), dtype=t.float64)
    age = t.tensor((-1.0, -0.5, 0.5, 1.0), dtype=t.float64)
    mask = t.ones(4, dtype=t.bool)
    with pytest.raises(TypeError, match="standardized age"):
        subgroup_age_correlations(methylation, age.to(t.float32), mask)
    with pytest.raises(TypeError, match="subgroup mask"):
        subgroup_age_correlations(methylation, age, mask.to(t.int64))
    with pytest.raises(ValueError, match="three"):
        subgroup_age_correlations(
            methylation,
            age,
            t.tensor((True, True, False, False)),
        )
    with pytest.raises(ValueError, match="age is constant"):
        subgroup_age_correlations(
            methylation,
            t.ones(4, dtype=t.float64),
            mask,
        )
