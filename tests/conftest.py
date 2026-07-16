from __future__ import annotations

from collections.abc import Callable

import pytest

from methylation_latent.domain import (
    GenomicContext,
    InfiniumDesign,
    ManifestStrand,
    ProbeLocus,
    parse_autosome,
    parse_one_based_position,
    parse_probe_id,
)


@pytest.fixture
def make_probe() -> Callable[..., ProbeLocus]:
    def factory(
        index: int,
        *,
        chromosome: int = 1,
        position: int = 1_000,
        context: GenomicContext = GenomicContext.ISLAND,
        strand: ManifestStrand = ManifestStrand.FORWARD,
        design: InfiniumDesign = InfiniumDesign.TYPE_II,
    ) -> ProbeLocus:
        return ProbeLocus(
            probe_id=parse_probe_id(f"cg{index:08d}"),
            chromosome=parse_autosome(chromosome),
            position=parse_one_based_position(position),
            context=context,
            design=design,
            manifest_strand=strand,
        )

    return factory
