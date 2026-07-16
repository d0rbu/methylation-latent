from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from methylation_latent.domain import (
    GenomicContext,
    ManifestStrand,
    NonEmptyProbeSet,
    ProbeLocus,
    WindowSize,
    parse_autosome,
    parse_window_size,
)
from methylation_latent.genome import (
    FastaIndexEntry,
    GenomicInterval,
    IndexedFasta,
    assert_probe_windows,
    audit_reference_windows,
    chromosome_lengths,
    cpg_window_interval,
    encode_dna,
    extract_cpg_window,
    parse_fasta_index,
    sequence_features,
)
from methylation_latent.splits import (
    GenomicSplit,
    NestedGenomicSplit,
    OverlapAudit,
    SplitKind,
    _block_for_probe,
    _choose_blocks,
    _nearest_distances,
    _require_context_coverage,
    _torch_permutation,
    add_diverse_validation_split,
    apply_maximum_window_buffer,
    assert_no_window_overlap,
    build_diverse_block_split,
    build_held_out_chromosome_split,
)

PRIMARY_WINDOWS = tuple(map(parse_window_size, (1024, 4096, 16384, 65536)))


def _write_single_line_fasta(path: Path, chromosome: str, sequence: str) -> None:
    path.write_text(f">{chromosome}\n{sequence}\n", encoding="ascii")
    Path(f"{path}.fai").write_text(
        f"{chromosome}\t{len(sequence)}\t{len(chromosome) + 2}\t{len(sequence)}\t{len(sequence) + 1}\n",
        encoding="ascii",
    )


def test_genomic_interval_half_open_overlap_semantics() -> None:
    interval = GenomicInterval("chr1", 10, 20)
    assert interval.width == 10
    assert interval.overlaps(GenomicInterval("chr1", 19, 30))
    assert not interval.overlaps(GenomicInterval("chr1", 20, 30))
    assert not interval.overlaps(GenomicInterval("chr2", 10, 20))
    with pytest.raises(ValueError):
        GenomicInterval("", 0, 1)
    with pytest.raises(ValueError):
        GenomicInterval("chr1", -1, 1)
    with pytest.raises(ValueError):
        GenomicInterval("chr1", 1, 1)


def test_fasta_index_entry_and_parser_contract(tmp_path: Path) -> None:
    index = tmp_path / "reference.fai"
    index.write_text("chr1\t10\t6\t10\t11\n", encoding="ascii")
    assert parse_fasta_index(index)["chr1"] == FastaIndexEntry("chr1", 10, 6, 10, 11)
    index.write_text("chr1\t10\t6\t10\n", encoding="ascii")
    with pytest.raises(ValueError, match="invalid"):
        parse_fasta_index(index)
    index.write_text("chr1\t10\t6\t10\t11\nchr1\t10\t6\t10\t11\n", encoding="ascii")
    with pytest.raises(ValueError, match="duplicate"):
        parse_fasta_index(index)
    index.write_text("", encoding="ascii")
    with pytest.raises(ValueError, match="at least one"):
        parse_fasta_index(index)
    with pytest.raises(ValueError):
        FastaIndexEntry("chr1", 10, 0, 10, 9)
    with pytest.raises(ValueError, match="name"):
        FastaIndexEntry("", 10, 1, 10, 11)
    with pytest.raises(ValueError, match="smaller"):
        FastaIndexEntry("chr1", 10, 1, 10, 9)


def test_plus_strand_cpg_extraction_ignores_assay_strand(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    sequence = "A" * 49 + "CG" + "T" * 49
    fasta = tmp_path / "hg19.fa"
    _write_single_line_fasta(fasta, "chr1", sequence)
    probe = make_probe(1, position=50, strand=ManifestStrand.REVERSE)
    window = parse_window_size(20)
    interval = cpg_window_interval(probe, window, chromosome_length=len(sequence))
    assert interval == GenomicInterval("chr1", 39, 59)
    with IndexedFasta(fasta) as reference:
        extracted = extract_cpg_window(reference, probe, window)
        assert extracted[10:12] == "CG"
        assert assert_probe_windows(reference, (probe,), (window,)) == 1
        assert reference.contig_length("chr1") == len(sequence)
        assert reference.read_interval(GenomicInterval("chr1", 49, 51)) == "CG"
    assert extracted == sequence[39:59]


def test_fasta_access_fails_loudly_on_coordinate_and_sequence_errors(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    fasta = tmp_path / "hg19.fa"
    _write_single_line_fasta(fasta, "chr1", "A" * 40 + "CG" + "N" + "A" * 57)
    reference = IndexedFasta(fasta)
    with pytest.raises(RuntimeError, match="context manager"):
        reference.read_interval(GenomicInterval("chr1", 0, 1))
    with reference:
        with pytest.raises(KeyError):
            reference.contig_length("chr2")
        with pytest.raises(ValueError, match="exceeds"):
            reference.read_interval(GenomicInterval("chr1", 99, 101))
        with pytest.raises(ValueError, match="plus-strand CG"):
            extract_cpg_window(reference, make_probe(1, position=40), parse_window_size(10))
        with pytest.raises(ValueError, match="non-ACGT"):
            extract_cpg_window(reference, make_probe(2, position=41), parse_window_size(10))
        with pytest.raises(ValueError, match="full"):
            cpg_window_interval(
                make_probe(3, position=2),
                parse_window_size(20),
                chromosome_length=100,
            )
    with pytest.raises(RuntimeError, match="already open"), IndexedFasta(fasta) as duplicate:
        duplicate.__enter__()
    unopened = IndexedFasta(fasta)
    with pytest.raises(RuntimeError, match="not open"):
        unopened.__exit__(None, None, None)
    with pytest.raises(ValueError, match="positive"):
        cpg_window_interval(make_probe(4, position=20), parse_window_size(10), chromosome_length=0)


def test_fasta_index_corruption_never_returns_partial_or_newline_sequence(tmp_path: Path) -> None:
    short = tmp_path / "short.fa"
    short.write_bytes(b">chr1\nAC")
    Path(f"{short}.fai").write_text("chr1\t10\t6\t10\t11\n", encoding="ascii")
    with IndexedFasta(short) as reference, pytest.raises(ValueError, match="ended before"):
        reference.read_interval(GenomicInterval("chr1", 0, 4))

    newline = tmp_path / "newline.fa"
    newline.write_bytes(b">chr1\nACGT\n")
    Path(f"{newline}.fai").write_text("chr1\t4\t5\t4\t5\n", encoding="ascii")
    with IndexedFasta(newline) as reference, pytest.raises(ValueError, match="line terminator"):
        reference.read_interval(GenomicInterval("chr1", 0, 1))


def test_wrapped_fasta_random_access(tmp_path: Path) -> None:
    fasta = tmp_path / "wrapped.fa"
    fasta.write_text(">chr1\nACGT\nTGCA\n", encoding="ascii")
    Path(f"{fasta}.fai").write_text("chr1\t8\t6\t4\t5\n", encoding="ascii")
    with IndexedFasta(fasta) as reference:
        assert reference.read_interval(GenomicInterval("chr1", 2, 7)) == "GTTGC"


def test_sequence_encoding_and_handcrafted_features() -> None:
    encoded = encode_dna("acgc")
    assert encoded.dtype == t.uint8
    assert encoded.tolist() == list(b"ACGC")
    features = sequence_features("ACGCGT")
    assert float(features.gc_content) == pytest.approx(4 / 6)
    assert float(features.cpg_density) == pytest.approx(2 / 5)
    assert features.as_float64_tensor().dtype == t.float64
    with pytest.raises(ValueError, match="non-ACGT"):
        encode_dna("ACN")
    with pytest.raises(ValueError, match="must not be empty"):
        encode_dna("")
    with pytest.raises(ValueError, match="at least two"):
        sequence_features("A")


def test_probe_window_audit_requires_both_probes_and_windows(
    tmp_path: Path, make_probe: Callable[..., ProbeLocus]
) -> None:
    fasta = tmp_path / "reference.fa"
    _write_single_line_fasta(fasta, "chr1", "A" * 49 + "CG" + "A" * 49)
    with IndexedFasta(fasta) as reference:
        with pytest.raises(ValueError, match="at least one sequence-window"):
            assert_probe_windows(reference, (make_probe(1, position=50),), ())
        with pytest.raises(ValueError, match="at least one probe"):
            assert_probe_windows(reference, (), (parse_window_size(10),))


def test_exhaustive_reference_audit_partitions_boundary_and_ambiguous_windows(
    tmp_path: Path, make_probe: Callable[..., ProbeLocus]
) -> None:
    bases = list("A" * 120)
    for cytosine_zero in (1, 49, 79):
        bases[cytosine_zero : cytosine_zero + 2] = "CG"
    bases[75] = "N"
    fasta = tmp_path / "reference.fa"
    _write_single_line_fasta(fasta, "chr1", "".join(bases))
    probes = NonEmptyProbeSet(
        (
            make_probe(1, position=50),
            make_probe(2, position=2),
            make_probe(3, position=80),
        )
    )
    with IndexedFasta(fasta) as reference:
        audit = audit_reference_windows(reference, probes, parse_window_size(20))
    assert audit.input_probes == audit.coordinate_cpgs_verified == 3
    assert audit.eligible_probes == 1
    assert audit.boundary_exclusions == 1
    assert audit.non_acgt_exclusions == 1
    assert len(audit.eligible_probe_order_sha256) == 64

    with pytest.raises(ValueError, match="plus-strand CpG"), IndexedFasta(fasta) as reference:
        audit_reference_windows(
            reference,
            NonEmptyProbeSet((make_probe(4, position=30),)),
            parse_window_size(20),
        )
    with pytest.raises(ValueError, match="partition"):
        replace(audit, eligible_probes=2)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(audit, eligible_probe_order_sha256="bad")


@pytest.mark.parametrize("window", PRIMARY_WINDOWS)
def test_overlap_assertion_for_every_primary_window(
    window: WindowSize,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    width = int(window)
    test = NonEmptyProbeSet((make_probe(1, position=100_000),))
    exact = NonEmptyProbeSet((make_probe(2, position=100_000 + width),))
    audit = assert_no_window_overlap(exact, test, (window,))
    assert audit[0].minimum_cytosine_distance == width
    overlapping = NonEmptyProbeSet((make_probe(3, position=100_000 + width - 1),))
    with pytest.raises(ValueError, match="overlap"):
        assert_no_window_overlap(overlapping, test, (window,))


@pytest.mark.property
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    test_position=st.integers(min_value=70_000, max_value=1_000_000),
    offset=st.integers(min_value=-80_000, max_value=80_000).filter(lambda value: value != 0),
    width=st.sampled_from([1024, 4096, 16384, 65536]),
)
def test_optimized_overlap_assertion_matches_interval_oracle(
    test_position: int,
    offset: int,
    width: int,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    train_position = test_position + offset
    if train_position <= width:
        return
    window = parse_window_size(width)
    train_probe = make_probe(1, position=train_position)
    test_probe = make_probe(2, position=test_position)
    train = NonEmptyProbeSet((train_probe,))
    test = NonEmptyProbeSet((test_probe,))
    train_interval = cpg_window_interval(train_probe, window, chromosome_length=2_000_000)
    test_interval = cpg_window_interval(test_probe, window, chromosome_length=2_000_000)
    oracle_overlap = train_interval.overlaps(test_interval)
    if oracle_overlap:
        with pytest.raises(ValueError, match="overlap"):
            assert_no_window_overlap(train, test, (window,))
    else:
        assert_no_window_overlap(train, test, (window,))


def test_maximum_window_buffer_is_deliberately_one_base_conservative(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    test = NonEmptyProbeSet((make_probe(1, position=100_000),))
    candidates = NonEmptyProbeSet(
        (
            make_probe(2, position=100_000 + 65_536),
            make_probe(3, position=100_000 + 65_537),
            make_probe(4, chromosome=2, position=100_000),
        )
    )
    retained, excluded = apply_maximum_window_buffer(candidates, test, parse_window_size(65_536))
    assert {probe.probe_id for probe in excluded} == {"cg00000002"}
    assert {probe.probe_id for probe in retained.probes} == {"cg00000003", "cg00000004"}
    assert_no_window_overlap(retained, test, PRIMARY_WINDOWS)


def _diverse_universe(make_probe: Callable[..., ProbeLocus]) -> NonEmptyProbeSet:
    probes: list[ProbeLocus] = []
    index = 1
    contexts = tuple(GenomicContext)
    for context_index, context in enumerate(contexts):
        for chromosome in (context_index * 2 + 1, context_index * 2 + 2):
            for position in (100_000, 300_000, 500_000):
                probes.append(
                    make_probe(
                        index,
                        chromosome=chromosome,
                        position=position,
                        context=context,
                    )
                )
                index += 1
    return NonEmptyProbeSet(tuple(probes))


def test_diverse_and_nested_splits_are_target_blind_and_context_complete(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _diverse_universe(make_probe)
    lengths = dict.fromkeys(range(1, 9), 1000000)
    windows = (parse_window_size(1024), parse_window_size(4096))
    primary = build_diverse_block_split(
        probes,
        chromosome_lengths=lengths,
        block_width=20_000,
        anchors_per_context=1,
        window_sizes=windows,
        seed=11,
    )
    assert primary.kind == SplitKind.DIVERSE_BLOCKS
    assert {probe.context for probe in primary.test.probes} == set(GenomicContext)
    assert {probe.context for probe in primary.train.probes} == set(GenomicContext)
    assert len(primary.blocks) == 4
    assert primary == build_diverse_block_split(
        probes,
        chromosome_lengths=lengths,
        block_width=20_000,
        anchors_per_context=1,
        window_sizes=windows,
        seed=11,
    )
    nested = add_diverse_validation_split(
        primary,
        chromosome_lengths=lengths,
        block_width=20_000,
        anchors_per_context=1,
        seed=22,
    )
    assert len(nested.optimization) > 0
    assert len(nested.validation) > 0
    assert nested.test == primary.test


def test_held_out_chromosome_split_has_no_shared_chromosome(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = NonEmptyProbeSet(
        (
            make_probe(1, chromosome=1),
            make_probe(2, chromosome=2),
            make_probe(3, chromosome=2, position=2_000),
        )
    )
    split = build_held_out_chromosome_split(
        probes,
        held_out_chromosome=parse_autosome(2),
        window_sizes=(parse_window_size(1024),),
        seed=4,
    )
    assert split.kind == SplitKind.HELD_OUT_CHROMOSOME
    assert {int(probe.chromosome) for probe in split.test.probes} == {2}
    assert split.overlap_audits[0].shared_chromosomes == 0


def test_split_semantic_wrappers_reject_inconsistent_records(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    train = NonEmptyProbeSet((make_probe(1),))
    test = NonEmptyProbeSet((make_probe(2, chromosome=2),))
    audit = assert_no_window_overlap(train, test, (parse_window_size(1024),))
    with pytest.raises(ValueError, match="unique"):
        GenomicSplit(
            SplitKind.HELD_OUT_CHROMOSOME,
            train,
            test,
            (),
            (),
            (parse_window_size(1024), parse_window_size(1024)),
            1,
            (*audit, *audit),
        )


def test_split_envelopes_reject_missing_audits_overlap_and_bad_repartition(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    window = parse_window_size(1024)
    train = NonEmptyProbeSet((make_probe(1, chromosome=1), make_probe(2, chromosome=2)))
    test = NonEmptyProbeSet((make_probe(3, chromosome=3),))
    audit = assert_no_window_overlap(train, test, (window,))
    primary = GenomicSplit(SplitKind.HELD_OUT_CHROMOSOME, train, test, (), (), (window,), 1, audit)
    with pytest.raises(ValueError, match="at least one"):
        replace(primary, window_sizes=(), overlap_audits=())
    with pytest.raises(ValueError, match="disjoint"):
        replace(primary, buffer_excluded=(train.probes[0],))
    with pytest.raises(ValueError, match="exactly match"):
        replace(primary, overlap_audits=())

    validation_train = NonEmptyProbeSet((train.probes[0],))
    validation_test = NonEmptyProbeSet((train.probes[1],))
    validation = GenomicSplit(
        SplitKind.DIVERSE_BLOCKS,
        validation_train,
        validation_test,
        (),
        (),
        (window,),
        2,
        assert_no_window_overlap(validation_train, validation_test, (window,)),
    )
    assert NestedGenomicSplit(primary, validation).optimization == validation_train
    other_window = parse_window_size(4096)
    with pytest.raises(ValueError, match="identical window"):
        NestedGenomicSplit(
            primary,
            replace(
                validation,
                window_sizes=(other_window,),
                overlap_audits=(OverlapAudit(other_window, 0, None),),
            ),
        )
    alien = NonEmptyProbeSet((make_probe(4, chromosome=4),))
    wrong_repartition = replace(
        validation,
        test=alien,
        overlap_audits=assert_no_window_overlap(validation_train, alien, (window,)),
    )
    with pytest.raises(ValueError, match="repartition"):
        NestedGenomicSplit(primary, wrong_repartition)


def test_split_distance_and_buffer_helpers_fail_closed(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    with pytest.raises(ValueError, match="non-empty vectors"):
        _nearest_distances(t.empty(0, dtype=t.int64), t.tensor([1], dtype=t.int64))
    with pytest.raises(TypeError, match="int64"):
        _nearest_distances(t.tensor([1.0]), t.tensor([1], dtype=t.int64))
    with pytest.raises(ValueError, match="query positions"):
        _nearest_distances(t.tensor([2, 1]), t.tensor([1], dtype=t.int64))
    with pytest.raises(ValueError, match="reference positions"):
        _nearest_distances(t.tensor([1]), t.tensor([2, 1], dtype=t.int64))

    first = NonEmptyProbeSet((make_probe(1),))
    same = NonEmptyProbeSet((make_probe(1),))
    with pytest.raises(ValueError, match="at least one window"):
        assert_no_window_overlap(first, same, ())
    with pytest.raises(ValueError, match="duplicate"):
        assert_no_window_overlap(first, same, (parse_window_size(1024),))
    with pytest.raises(ValueError, match="must be disjoint"):
        apply_maximum_window_buffer(first, same, parse_window_size(1024))
    near = NonEmptyProbeSet((make_probe(2, position=1001),))
    with pytest.raises(ValueError, match="removed every"):
        apply_maximum_window_buffer(near, first, parse_window_size(1024))


def test_diverse_block_selection_rejects_unrepresentative_or_invalid_universes(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    window = (parse_window_size(1024),)
    island_only = NonEmptyProbeSet((make_probe(1, context=GenomicContext.ISLAND),))
    with pytest.raises(ValueError, match="positive"):
        build_diverse_block_split(
            island_only,
            chromosome_lengths={1: 1_000_000},
            block_width=0,
            anchors_per_context=1,
            window_sizes=window,
            seed=1,
        )
    with pytest.raises(ValueError, match="at least one window"):
        build_diverse_block_split(
            island_only,
            chromosome_lengths={1: 1_000_000},
            block_width=1000,
            anchors_per_context=1,
            window_sizes=(),
            seed=1,
        )
    with pytest.raises(ValueError, match="chromosome lengths"):
        build_diverse_block_split(
            island_only,
            chromosome_lengths={},
            block_width=1000,
            anchors_per_context=1,
            window_sizes=window,
            seed=1,
        )
    with pytest.raises(KeyError, match="missing chromosome"):
        build_diverse_block_split(
            island_only,
            chromosome_lengths={2: 1_000_000},
            block_width=1000,
            anchors_per_context=1,
            window_sizes=window,
            seed=1,
        )
    with pytest.raises(ValueError, match="eligible anchors"):
        build_diverse_block_split(
            island_only,
            chromosome_lengths={1: 1_000_000},
            block_width=1000,
            anchors_per_context=1,
            window_sizes=window,
            seed=1,
        )
    with pytest.raises(ValueError, match="lacks required contexts"):
        _require_context_coverage(island_only, "test")


def test_block_helpers_reject_edges_and_cross_context_overlap(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    edge = make_probe(1, position=2, context=GenomicContext.ISLAND)
    assert _block_for_probe(edge, block_width=100, chromosome_length=1000) is None
    generator = t.Generator().manual_seed(1)
    assert _torch_permutation((), generator) == ()
    collocated = NonEmptyProbeSet(
        tuple(
            make_probe(index, position=100_000, context=context)
            for index, context in enumerate(GenomicContext, start=1)
        )
    )
    with pytest.raises(ValueError, match="could not select"):
        _choose_blocks(
            collocated,
            chromosome_lengths={1: 1_000_000},
            block_width=10_000,
            anchors_per_context=1,
            generator=generator,
        )


def test_chromosome_lengths_requires_all_autosomes(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "one.fa"
    _write_single_line_fasta(fasta, "chr1", "A" * 10)
    with IndexedFasta(fasta) as reference, pytest.raises(KeyError, match="chr2"):
        chromosome_lengths(reference)
