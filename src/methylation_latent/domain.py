"""Scalar domain types and immutable locus records."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from beartype import beartype
from phantom import Phantom

_PROBE_ID = re.compile(r"^cg[0-9]{8}$", flags=re.ASCII)
_GSM_ACCESSION = re.compile(r"^GSM[0-9]+$", flags=re.ASCII)
_SENTRIX_IDENTITY = re.compile(r"^[0-9]{10}_R0[1-6]C0[1-2]$", flags=re.ASCII)


def _is_beta(value: float) -> bool:
    return 0.0 <= value <= 1.0


def _is_correlation(value: float) -> bool:
    return -1.0 <= value <= 1.0


def _is_fraction(value: float) -> bool:
    return 0.0 <= value <= 1.0


def _is_non_negative(value: float) -> bool:
    return value >= 0.0


def _is_positive_int(value: int) -> bool:
    return not isinstance(value, bool) and value > 0


def _is_autosome(value: int) -> bool:
    return not isinstance(value, bool) and 1 <= value <= 22


def _is_even_window(value: int) -> bool:
    return not isinstance(value, bool) and value >= 2 and value % 2 == 0


def _is_probe_id(value: str) -> bool:
    return _PROBE_ID.fullmatch(value) is not None


def _is_age_years(value: float) -> bool:
    return math.isfinite(value) and 0.0 < value <= 130.0


def _is_gsm_accession(value: str) -> bool:
    return _GSM_ACCESSION.fullmatch(value) is not None


def _is_sentrix_identity(value: str) -> bool:
    return _SENTRIX_IDENTITY.fullmatch(value) is not None


class BetaValue(float, Phantom[float], predicate=_is_beta, bound=float):
    """A methylation beta value in the closed interval [0, 1]."""


class Correlation(float, Phantom[float], predicate=_is_correlation, bound=float):
    """A Pearson/cosine correlation in the closed interval [-1, 1]."""


class Fraction(float, Phantom[float], predicate=_is_fraction, bound=float):
    """A dimensionless fraction in the closed interval [0, 1]."""


class NonNegativeWeight(float, Phantom[float], predicate=_is_non_negative, bound=float):
    """A loss weight greater than or equal to zero."""


class PositiveInt(int, Phantom[int], predicate=_is_positive_int, bound=int):
    """A strictly positive integer."""


class Autosome(int, Phantom[int], predicate=_is_autosome, bound=int):
    """An autosomal chromosome number in [1, 22]."""


class OneBasedPosition(int, Phantom[int], predicate=_is_positive_int, bound=int):
    """A positive one-based genomic coordinate."""


class LatentDimension(int, Phantom[int], predicate=_is_positive_int, bound=int):
    """A positive latent-space dimension before model-specific ceiling validation."""


class WindowSize(int, Phantom[int], predicate=_is_even_window, bound=int):
    """A positive, even sequence-window width in bases."""


class ProbeId(str, Phantom[str], predicate=_is_probe_id, bound=str):
    """An Illumina CpG probe identifier of the form cg plus eight digits."""


class AgeYears(float, Phantom[float], predicate=_is_age_years, bound=float):
    """A finite human chronological age in the interval (0, 130]."""


class GsmAccession(str, Phantom[str], predicate=_is_gsm_accession, bound=str):
    """A GEO sample accession."""


class SentrixIdentity(str, Phantom[str], predicate=_is_sentrix_identity, bound=str):
    """An HM450 Sentrix barcode and array position, e.g. 5815284001_R01C01."""


class GenomicContext(StrEnum):
    """Collapsed UCSC CpG-island relation used for split stratification."""

    ISLAND = "island"
    SHORE = "shore"
    SHELF = "shelf"
    OPEN_SEA = "open_sea"


class InfiniumDesign(StrEnum):
    """Illumina Infinium probe chemistry."""

    TYPE_I = "I"
    TYPE_II = "II"


class ManifestStrand(StrEnum):
    """Assay strand retained only for coordinate audits."""

    FORWARD = "F"
    REVERSE = "R"


@beartype
@dataclass(frozen=True, slots=True)
class ProbeLocus:
    """A validated hg19 autosomal CpG locus."""

    probe_id: ProbeId
    chromosome: Autosome
    position: OneBasedPosition
    context: GenomicContext
    design: InfiniumDesign
    manifest_strand: ManifestStrand


@dataclass(frozen=True, slots=True)
class NonEmptyProbeSet:
    """An ordered, non-empty collection of loci with unique probe IDs."""

    probes: tuple[ProbeLocus, ...]

    def __post_init__(self) -> None:
        if not self.probes:
            raise ValueError("probe set must not be empty")
        probe_ids = tuple(probe.probe_id for probe in self.probes)
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError("probe set contains duplicate probe IDs")

    @classmethod
    def from_iterable(cls, probes: Iterable[ProbeLocus]) -> NonEmptyProbeSet:
        return cls(tuple(probes))

    def __len__(self) -> int:
        return len(self.probes)


@beartype
def parse_beta(value: float | int | str) -> BetaValue:
    return BetaValue.parse(float(value))


@beartype
def parse_correlation(value: float | int | str) -> Correlation:
    return Correlation.parse(float(value))


@beartype
def parse_fraction(value: float | int | str) -> Fraction:
    return Fraction.parse(float(value))


@beartype
def parse_non_negative_weight(value: float | int | str) -> NonNegativeWeight:
    return NonNegativeWeight.parse(float(value))


@beartype
def parse_positive_int(value: int | str) -> PositiveInt:
    if isinstance(value, bool):
        raise TypeError("boolean is not a positive integer")
    return PositiveInt.parse(int(value))


@beartype
def parse_autosome(value: int | str) -> Autosome:
    if isinstance(value, bool):
        raise TypeError("boolean is not an autosome")
    text = str(value)
    normalized = text[3:] if text.lower().startswith("chr") else text
    return Autosome.parse(int(normalized))


@beartype
def parse_one_based_position(value: int | str) -> OneBasedPosition:
    if isinstance(value, bool):
        raise TypeError("boolean is not a genomic position")
    return OneBasedPosition.parse(int(value))


@beartype
def parse_latent_dimension(value: int | str) -> LatentDimension:
    if isinstance(value, bool):
        raise TypeError("boolean is not a latent dimension")
    return LatentDimension.parse(int(value))


@beartype
def parse_window_size(value: int | str) -> WindowSize:
    if isinstance(value, bool):
        raise TypeError("boolean is not a sequence-window width")
    return WindowSize.parse(int(value))


@beartype
def parse_probe_id(value: str) -> ProbeId:
    return ProbeId.parse(value)


@beartype
def parse_age_years(value: float | int | str) -> AgeYears:
    return AgeYears.parse(float(value))


@beartype
def parse_gsm_accession(value: str) -> GsmAccession:
    return GsmAccession.parse(value)


@beartype
def parse_sentrix_identity(value: str) -> SentrixIdentity:
    return SentrixIdentity.parse(value)


@beartype
def parse_genomic_context(raw_relation: str, island_name: str) -> GenomicContext:
    relation = raw_relation.strip()
    if relation == "Island":
        if not island_name.strip():
            raise ValueError("island relation requires a CpG-island name")
        return GenomicContext.ISLAND
    if relation in {"N_Shore", "S_Shore"}:
        return GenomicContext.SHORE
    if relation in {"N_Shelf", "S_Shelf"}:
        return GenomicContext.SHELF
    if relation == "" and island_name.strip() == "":
        return GenomicContext.OPEN_SEA
    raise ValueError(
        f"unrecognized or contradictory CpG-island context: relation={raw_relation!r}, "
        f"island={island_name!r}"
    )
