from __future__ import annotations

from dataclasses import replace

import pytest
import torch as t

from methylation_latent.age_clusters import (
    KMeansTwoConfig,
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


def test_partition_agreement_is_invariant_to_binary_label_permutation() -> None:
    first = t.tensor((0, 0, 1, 1, 1, 0), dtype=t.int64)
    second = 1 - first
    assert permutation_invariant_label_agreement(first, second) == 1.0
    assert adjusted_rand_index(first, second) == 1.0


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
