from __future__ import annotations

from collections.abc import Callable

import pytest
import torch as t

from methylation_latent.distance_integration import (
    KernelDesign,
    SimplexKernelFit,
    append_sequence_kernel,
    distance_kernel_design,
    fit_simplex_kernel,
    predict_kernel_mixture,
)
from methylation_latent.domain import NonEmptyProbeSet, ProbeLocus


def _probes(make_probe: Callable[..., ProbeLocus]) -> NonEmptyProbeSet:
    return NonEmptyProbeSet(
        (
            make_probe(1, chromosome=1, position=1_000),
            make_probe(2, chromosome=1, position=2_000),
            make_probe(3, chromosome=2, position=1_000),
        )
    )


def test_distance_design_separates_trans_and_has_unit_diagonal(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _probes(make_probe)
    left = t.tensor([0, 0, 0, 1, 2], dtype=t.int64)
    right = t.tensor([0, 1, 2, 1, 2], dtype=t.int64)
    design = distance_kernel_design(probes, left, right, length_scales=(1_000, 4_000))
    assert design.names == ("global", "cis_laplacian_1000", "cis_laplacian_4000")
    assert t.equal(design.values[[0, 3, 4]], t.ones((3, 3), dtype=t.float64))
    assert t.equal(design.values[2], t.tensor([1.0, 0.0, 0.0], dtype=t.float64))
    assert design.values[1, 1].item() == pytest.approx(t.exp(t.tensor(-1.0)).item())


def test_simplex_fit_recovers_exact_mixture_and_boundary_solution() -> None:
    first = t.linspace(-0.5, 0.5, 101, dtype=t.float64)
    second = t.square(first)
    design = KernelDesign(t.stack((first, second), dim=1), ("first", "second"))
    target = 0.25 * first + 0.75 * second
    fit = fit_simplex_kernel(design, target)
    assert fit.weights == pytest.approx((0.25, 0.75), abs=1.0e-10)
    assert fit.mse == pytest.approx(0.0, abs=1.0e-20)
    assert t.allclose(predict_kernel_mixture(design, fit), target, atol=1.0e-12, rtol=0.0)

    boundary = fit_simplex_kernel(design, 1.5 * first - 0.5 * second)
    assert boundary.weights == pytest.approx((1.0, 0.0), abs=1.0e-10)


def test_convex_distance_sequence_mixture_is_a_unit_diagonal_psd_gram(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _probes(make_probe)
    grid = t.cartesian_prod(t.arange(3, dtype=t.int64), t.arange(3, dtype=t.int64))
    sequence_latent = t.tensor(
        [[1.0, 0.0], [0.6, 0.8], [-0.8, 0.6]],
        dtype=t.float64,
    )
    sequence = t.sum(
        sequence_latent.index_select(0, grid[:, 0])
        * sequence_latent.index_select(0, grid[:, 1]),
        dim=1,
    )
    distance = distance_kernel_design(
        probes,
        grid[:, 0],
        grid[:, 1],
        length_scales=(1_000, 4_000),
    )
    design = append_sequence_kernel(distance, sequence)
    fit = SimplexKernelFit(design.names, (0.1, 0.2, 0.3, 0.4), 0.0)
    gram = predict_kernel_mixture(design, fit).reshape(3, 3)
    assert t.allclose(t.diag(gram), t.ones(3, dtype=t.float64), atol=1.0e-12)
    assert float(t.linalg.eigvalsh(gram).min().item()) >= -1.0e-12


def test_distance_integration_rejects_misaligned_or_invalid_state(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _probes(make_probe)
    indices = t.tensor([0, 1], dtype=t.int64)
    with pytest.raises(ValueError, match="length scales"):
        distance_kernel_design(probes, indices, indices, length_scales=())
    with pytest.raises(ValueError, match="unique and increasing"):
        distance_kernel_design(probes, indices, indices, length_scales=(4_000, 1_000))
    design = distance_kernel_design(probes, indices, indices, length_scales=(1_000,))
    with pytest.raises(ValueError, match="misaligned"):
        append_sequence_kernel(design, t.ones(1, dtype=t.float64))
    with pytest.raises(ValueError, match="names differ"):
        predict_kernel_mixture(design, SimplexKernelFit(("x", "y"), (0.5, 0.5), 0.0))
    with pytest.raises(ValueError, match="at least two"):
        fit_simplex_kernel(
            KernelDesign(t.ones((1, 1), dtype=t.float64), ("one",)),
            t.ones(1, dtype=t.float64),
        )


@pytest.mark.parametrize(
    ("values", "names", "message"),
    [
        (t.ones((2, 2), dtype=t.float32), ("a", "b"), "rank-2 float64"),
        (t.empty((0, 2), dtype=t.float64), ("a", "b"), "non-empty"),
        (t.ones((2, 2), dtype=t.float64), ("a",), "width and names"),
        (t.ones((2, 2), dtype=t.float64), ("a", "a"), "non-empty and unique"),
        (t.tensor([[1.0, t.nan]], dtype=t.float64), ("a", "b"), "finite"),
        (t.tensor([[1.0, 1.1]], dtype=t.float64), ("a", "b"), "lie in"),
    ],
)
def test_kernel_design_rejects_invalid_axes(
    values: t.Tensor,
    names: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        KernelDesign(values, names)


def test_fit_and_prediction_reject_invalid_numeric_inputs(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _probes(make_probe)
    indices = t.tensor([0, 1], dtype=t.int64)
    design = distance_kernel_design(probes, indices, indices, length_scales=(1_000,))
    with pytest.raises(TypeError, match="Type-check error"):
        append_sequence_kernel(design, t.ones(2, dtype=t.float32))
    nonfinite = t.tensor([0.0, t.nan], dtype=t.float64)
    with pytest.raises(ValueError, match="finite"):
        append_sequence_kernel(design, nonfinite)
    with pytest.raises(ValueError, match="cosine bounds"):
        append_sequence_kernel(design, t.tensor([0.0, 1.1], dtype=t.float64))
    with pytest.raises(TypeError, match="Type-check error"):
        fit_simplex_kernel(design, t.ones(2, dtype=t.float32))
    with pytest.raises(ValueError, match="finite"):
        fit_simplex_kernel(design, nonfinite)
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        fit_simplex_kernel(design, t.tensor([0.0, 1.1], dtype=t.float64))


@pytest.mark.parametrize(
    ("names", "weights", "mse", "message"),
    [
        (("a",), (0.5, 0.5), 0.0, "aligned"),
        (("a", "b"), (-0.1, 1.1), 0.0, "non-negative"),
        (("a", "b"), (0.2, 0.2), 0.0, "sum to one"),
        (("a", "b"), (0.5, 0.5), -1.0, "finite and non-negative"),
    ],
)
def test_simplex_fit_record_rejects_invalid_weights(
    names: tuple[str, ...],
    weights: tuple[float, ...],
    mse: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SimplexKernelFit(names, weights, mse)


def test_distance_design_rejects_bad_pair_indices(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _probes(make_probe)
    with pytest.raises(TypeError, match="Type-check error"):
        distance_kernel_design(
            probes,
            t.tensor([0.0]),
            t.tensor([1]),
            length_scales=(1_000,),
        )
    with pytest.raises(TypeError, match="Type-check error"):
        distance_kernel_design(
            probes,
            t.tensor([0], dtype=t.int64),
            t.tensor([1, 2], dtype=t.int64),
            length_scales=(1_000,),
        )
    with pytest.raises(ValueError, match="aligned and non-empty"):
        distance_kernel_design(
            probes,
            t.empty(0, dtype=t.int64),
            t.empty(0, dtype=t.int64),
            length_scales=(1_000,),
        )
    with pytest.raises(IndexError, match="outside"):
        distance_kernel_design(
            probes,
            t.tensor([0], dtype=t.int64),
            t.tensor([3], dtype=t.int64),
            length_scales=(1_000,),
        )
