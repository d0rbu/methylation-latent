from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch as t

from methylation_latent.domain import (
    GenomicContext,
    NonEmptyProbeSet,
    ProbeLocus,
    parse_autosome,
    parse_positive_int,
    parse_window_size,
)
from methylation_latent.experiment_data import (
    SequenceFeatureArtifact,
    SplitArtifact,
    build_primary_splits,
    compute_sequence_feature_artifact,
    load_sequence_features,
    load_split_artifact,
    save_sequence_features_exclusive,
    save_split_artifact_exclusive,
)
from methylation_latent.genome import IndexedFasta, sequence_features


def _split_universe(make_probe: Callable[..., ProbeLocus]) -> NonEmptyProbeSet:
    probes: list[ProbeLocus] = []
    index = 1
    for context_index, context in enumerate(GenomicContext):
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


def _write_single_line_fasta(path: Path, chromosome: str, sequence: str) -> None:
    path.write_text(f">{chromosome}\n{sequence}\n", encoding="ascii")
    Path(f"{path}.fai").write_text(
        f"{chromosome}\t{len(sequence)}\t{len(chromosome) + 2}\t"
        f"{len(sequence)}\t{len(sequence) + 1}\n",
        encoding="ascii",
    )


def test_primary_split_artifacts_exactly_partition_and_round_trip(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _split_universe(make_probe)
    windows = (parse_window_size(1024), parse_window_size(4096))
    built = build_primary_splits(
        probes,
        chromosome_lengths=dict.fromkeys(range(1, 9), 1_000_000),
        block_width=parse_positive_int(20_000),
        anchors_per_context=parse_positive_int(1),
        held_out_chromosome=parse_autosome(8),
        window_sizes=windows,
        primary_seed=11,
        validation_seed=22,
    )
    for name, split, nested in (
        ("diverse", built.diverse, built.diverse_nested),
        (
            "chromosome",
            built.held_out_chromosome,
            built.held_out_chromosome_nested,
        ),
    ):
        assert split.probe_set(probes, "optimization")
        assert split.probe_set(probes, "validation")
        assert split.probe_set(probes, "test")
        expected_primary_train = t.tensor(
            sorted(
                set(range(len(probes)))
                - set(split.test_indices.tolist())
                - set(split.primary_buffer_indices.tolist())
            ),
            dtype=t.int64,
        )
        assert t.equal(split.primary_train_indices, expected_primary_train)
        tensor_path = tmp_path / f"{name}.safetensors"
        metadata_path = tmp_path / f"{name}.json"
        save_split_artifact_exclusive(
            tensor_path,
            metadata_path,
            split,
            nested,
            probes,
        )
        loaded = load_split_artifact(
            tensor_path,
            metadata_path,
            probes=probes,
        )
        assert loaded.name == split.name
        assert loaded.universe_size == split.universe_size
        assert t.equal(loaded.optimization_indices, split.optimization_indices)
        assert t.equal(loaded.validation_indices, split.validation_indices)
        assert t.equal(loaded.test_indices, split.test_indices)
        assert t.equal(loaded.primary_buffer_indices, split.primary_buffer_indices)
        assert t.equal(loaded.validation_buffer_indices, split.validation_buffer_indices)


def test_nested_sequence_features_share_the_same_plus_strand_cpg(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    left_flank = ("ACGTCGGC" * 5)[:35]
    right_flank = ("GCCGATAC" * 5)[:35]
    sequence = left_flank + "CG" + right_flank
    fasta = tmp_path / "hg19.fa"
    _write_single_line_fasta(fasta, "chr1", sequence)
    probes = NonEmptyProbeSet((make_probe(1, position=36),))
    windows = (parse_window_size(10), parse_window_size(20))
    with IndexedFasta(fasta) as reference:
        features = compute_sequence_feature_artifact(reference, probes, windows)
    assert set(features.by_window) == {10, 20}
    assert features.by_window[10].shape == (1, 2)
    assert not t.equal(features.by_window[10], features.by_window[20])
    maximum_sequence = sequence[25:45]
    expected_10 = sequence_features(maximum_sequence[5:15]).as_float64_tensor()
    expected_20 = sequence_features(maximum_sequence).as_float64_tensor()
    assert t.equal(features.by_window[10][0], expected_10)
    assert t.equal(features.by_window[20][0], expected_20)

    tensor_path = tmp_path / "features.safetensors"
    metadata_path = tmp_path / "features.json"
    save_sequence_features_exclusive(
        tensor_path,
        metadata_path,
        features,
        probes,
    )
    loaded = load_sequence_features(
        tensor_path,
        metadata_path,
        probes=probes,
    )
    assert t.equal(loaded.by_window[10], features.by_window[10])
    assert t.equal(loaded.by_window[20], features.by_window[20])


def _valid_split_artifact() -> SplitArtifact:
    return SplitArtifact(
        name="split",
        universe_size=5,
        optimization_indices=t.tensor([0], dtype=t.int64),
        validation_indices=t.tensor([1], dtype=t.int64),
        test_indices=t.tensor([2], dtype=t.int64),
        primary_buffer_indices=t.tensor([3], dtype=t.int64),
        validation_buffer_indices=t.tensor([4], dtype=t.int64),
    )


def test_split_artifact_rejects_every_partition_drift(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    valid = _valid_split_artifact()
    probes = NonEmptyProbeSet(tuple(make_probe(index + 1) for index in range(5)))
    assert valid.probe_set(probes, "test").probes == (probes.probes[2],)
    with pytest.raises(ValueError, match="positive universe"):
        SplitArtifact(
            "",
            0,
            valid.optimization_indices,
            valid.validation_indices,
            valid.test_indices,
            valid.primary_buffer_indices,
            valid.validation_buffer_indices,
        )
    with pytest.raises(ValueError, match="int64 vectors"):
        SplitArtifact(
            "split",
            5,
            valid.optimization_indices.float(),
            valid.validation_indices,
            valid.test_indices,
            valid.primary_buffer_indices,
            valid.validation_buffer_indices,
        )
    with pytest.raises(ValueError, match="must be non-empty"):
        SplitArtifact(
            "split",
            4,
            t.empty(0, dtype=t.int64),
            valid.validation_indices,
            valid.test_indices,
            valid.primary_buffer_indices,
            t.empty(0, dtype=t.int64),
        )
    with pytest.raises(IndexError, match="outside"):
        SplitArtifact(
            "split",
            5,
            t.tensor([5]),
            valid.validation_indices,
            valid.test_indices,
            valid.primary_buffer_indices,
            valid.validation_buffer_indices,
        )
    with pytest.raises(ValueError, match="duplicates"):
        SplitArtifact(
            "split",
            6,
            t.tensor([0, 0]),
            valid.validation_indices,
            valid.test_indices,
            valid.primary_buffer_indices,
            valid.validation_buffer_indices,
        )
    with pytest.raises(ValueError, match="exactly and disjointly"):
        SplitArtifact(
            "split",
            6,
            valid.optimization_indices,
            valid.validation_indices,
            valid.test_indices,
            valid.primary_buffer_indices,
            valid.validation_buffer_indices,
        )
    with pytest.raises(ValueError, match="universe size differs"):
        valid.probe_set(NonEmptyProbeSet(probes.probes[:-1]), "test")
    with pytest.raises(ValueError, match="unknown non-empty"):
        valid.probe_set(probes, "buffer")


def test_split_and_sequence_metadata_reject_identity_drift(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _split_universe(make_probe)
    windows = (parse_window_size(1024), parse_window_size(4096))
    built = build_primary_splits(
        probes,
        chromosome_lengths=dict.fromkeys(range(1, 9), 1_000_000),
        block_width=parse_positive_int(20_000),
        anchors_per_context=parse_positive_int(1),
        held_out_chromosome=parse_autosome(8),
        window_sizes=windows,
        primary_seed=11,
        validation_seed=22,
    )
    tensor = tmp_path / "split.safetensors"
    metadata = tmp_path / "split.json"
    save_split_artifact_exclusive(
        tensor,
        metadata,
        built.diverse,
        built.diverse_nested,
        probes,
    )
    with pytest.raises(ValueError, match="sizes differ"):
        save_split_artifact_exclusive(
            tmp_path / "wrong.safetensors",
            tmp_path / "wrong.json",
            built.diverse,
            built.diverse_nested,
            NonEmptyProbeSet(probes.probes[:-1]),
        )
    raw = json.loads(metadata.read_text(encoding="utf-8"))
    raw["unknown"] = 1
    malformed = tmp_path / "split-malformed.json"
    malformed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata envelope"):
        load_split_artifact(tensor, malformed, probes=probes)
    del raw["unknown"]
    raw["schema"] = "other"
    wrong_identity = tmp_path / "split-identity.json"
    wrong_identity.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata identity"):
        load_split_artifact(tensor, wrong_identity, probes=probes)

    feature_values = t.full((len(probes), 2), 0.5, dtype=t.float64)
    features = SequenceFeatureArtifact(len(probes), {1024: feature_values})
    feature_tensor = tmp_path / "features.safetensors"
    feature_metadata = tmp_path / "features.json"
    save_sequence_features_exclusive(
        feature_tensor,
        feature_metadata,
        features,
        probes,
    )
    with pytest.raises(ValueError, match="sizes differ"):
        save_sequence_features_exclusive(
            tmp_path / "feature-wrong.safetensors",
            tmp_path / "feature-wrong.json",
            features,
            NonEmptyProbeSet(probes.probes[:-1]),
        )
    feature_raw = json.loads(feature_metadata.read_text(encoding="utf-8"))
    feature_raw["window_sizes"] = [True]
    malformed_windows = tmp_path / "features-windows.json"
    malformed_windows.write_text(json.dumps(feature_raw), encoding="utf-8")
    with pytest.raises(ValueError, match="window list"):
        load_sequence_features(feature_tensor, malformed_windows, probes=probes)
    feature_raw["window_sizes"] = [1024]
    feature_raw["columns"] = ["wrong"]
    feature_identity = tmp_path / "features-identity.json"
    feature_identity.write_text(json.dumps(feature_raw), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        load_sequence_features(feature_tensor, feature_identity, probes=probes)
    feature_raw["unknown"] = 1
    feature_envelope = tmp_path / "features-envelope.json"
    feature_envelope.write_text(json.dumps(feature_raw), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata envelope"):
        load_sequence_features(feature_tensor, feature_envelope, probes=probes)


def test_sequence_feature_artifact_rejects_invalid_domains(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        SequenceFeatureArtifact(0, {})
    with pytest.raises(ValueError, match="positive and even"):
        SequenceFeatureArtifact(1, {3: t.zeros((1, 2), dtype=t.float64)})
    with pytest.raises(ValueError, match=r"shape \[probes, 2\]"):
        SequenceFeatureArtifact(1, {2: t.zeros((2, 2), dtype=t.float64)})
    with pytest.raises(ValueError, match=r"values in \[0, 1\]"):
        SequenceFeatureArtifact(1, {2: t.tensor([[t.nan, 0.0]], dtype=t.float64)})

    fasta = tmp_path / "hg19.fa"
    _write_single_line_fasta(fasta, "chr1", "A" * 20 + "CG" + "A" * 20)
    probes = NonEmptyProbeSet((make_probe(1, position=21),))
    with IndexedFasta(fasta) as reference:
        with pytest.raises(ValueError, match="non-empty and unique"):
            compute_sequence_feature_artifact(reference, probes, ())
        with pytest.raises(ValueError, match="non-empty and unique"):
            compute_sequence_feature_artifact(
                reference,
                probes,
                (parse_window_size(10), parse_window_size(10)),
            )
