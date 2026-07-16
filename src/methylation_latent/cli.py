"""Command-line entry points for audits and static result publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn

from methylation_latent.artifacts import sha256_file
from methylation_latent.config import load_protocol_config
from methylation_latent.genome import IndexedFasta, audit_reference_windows
from methylation_latent.hashing import assert_gzip_payload_matches_file, md5_file
from methylation_latent.manifest import (
    apply_published_exclusions,
    load_chen2013_cross_reactive,
    load_zhou2017_mask_general,
    parse_gpl13534_manifest,
)
from methylation_latent.site import build_static_site


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="methylation-latent")
    subcommands = parser.add_subparsers(dest="command", required=True)

    check_config = subcommands.add_parser("check-config")
    check_config.add_argument("--config", type=Path, required=True)

    mask_audit = subcommands.add_parser("audit-exclusion-lists")
    mask_audit.add_argument("--config", type=Path, required=True)
    mask_audit.add_argument("--manifest", type=Path, required=True)
    mask_audit.add_argument("--chen", type=Path, required=True)
    mask_audit.add_argument("--zhou", type=Path, required=True)

    reference_audit = subcommands.add_parser("audit-reference")
    reference_audit.add_argument("--config", type=Path, required=True)
    reference_audit.add_argument("--manifest", type=Path, required=True)
    reference_audit.add_argument("--chen", type=Path, required=True)
    reference_audit.add_argument("--zhou", type=Path, required=True)
    reference_audit.add_argument("--reference-archive", type=Path, required=True)
    reference_audit.add_argument("--fasta", type=Path, required=True)
    reference_audit.add_argument("--fasta-index", type=Path, required=True)

    build_site = subcommands.add_parser("build-site")
    build_site.add_argument("--registry", type=Path, required=True)
    build_site.add_argument("--template", type=Path, default=Path("site-template"))
    build_site.add_argument("--output", type=Path, required=True)
    return parser


def _json_print(value: dict[str, object]) -> None:
    print(json.dumps(value, allow_nan=False, sort_keys=True, indent=2))


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "check-config":
        config = load_protocol_config(arguments.config)
        _json_print(
            {
                "protocol_id": config.protocol_id,
                "schema": config.schema,
                "status": config.status,
            }
        )
        return
    if arguments.command == "audit-exclusion-lists":
        config = load_protocol_config(arguments.config)
        observed_manifest_hash = sha256_file(arguments.manifest)
        if observed_manifest_hash != config.data.manifest_sha256:
            raise ValueError(
                "manifest hash differs: "
                f"expected={config.data.manifest_sha256}, observed={observed_manifest_hash}"
            )
        chen = load_chen2013_cross_reactive(arguments.chen)
        zhou = load_zhou2017_mask_general(arguments.zhou)
        if chen.sha256 != config.masks.chen_sha256 or zhou.sha256 != config.masks.zhou_sha256:
            raise ValueError("loaded exclusion-list identity differs from protocol configuration")
        manifest = parse_gpl13534_manifest(arguments.manifest)
        filtered = apply_published_exclusions(manifest.probes, (chen, zhou))
        _json_print(
            {
                "chen_cross_reactive_cpgs": len(chen.probe_ids),
                "chen_sha256": chen.sha256,
                "eligible_autosomal_cpgs_before_masks": len(manifest.probes),
                "exclusion_ledger_rows": len(filtered.ledger),
                "retained_autosomal_cpgs": len(filtered.retained),
                "unmatched_by_source": {
                    name: len(probes) for name, probes in filtered.unmatched_by_source.items()
                },
                "zhou_mask_general_cpgs": len(zhou.probe_ids),
                "zhou_sha256": zhou.sha256,
            }
        )
        return
    if arguments.command == "audit-reference":
        config = load_protocol_config(arguments.config)
        observed_manifest_hash = sha256_file(arguments.manifest)
        if observed_manifest_hash != config.data.manifest_sha256:
            raise ValueError(
                "manifest hash differs: "
                f"expected={config.data.manifest_sha256}, observed={observed_manifest_hash}"
            )
        observed_reference_md5 = md5_file(arguments.reference_archive)
        if observed_reference_md5 != config.data.reference_archive_md5:
            raise ValueError(
                "reference archive MD5 differs: "
                f"expected={config.data.reference_archive_md5}, observed={observed_reference_md5}"
            )
        reference_payload_sha256 = assert_gzip_payload_matches_file(
            arguments.reference_archive,
            arguments.fasta,
        )
        chen = load_chen2013_cross_reactive(arguments.chen)
        zhou = load_zhou2017_mask_general(arguments.zhou)
        if chen.sha256 != config.masks.chen_sha256 or zhou.sha256 != config.masks.zhou_sha256:
            raise ValueError("loaded exclusion-list identity differs from protocol configuration")
        manifest = parse_gpl13534_manifest(arguments.manifest)
        filtered = apply_published_exclusions(manifest.probes, (chen, zhou))
        with IndexedFasta(arguments.fasta, arguments.fasta_index) as reference:
            audit = audit_reference_windows(
                reference,
                filtered.retained,
                max(config.splits.window_sizes, key=int),
            )
        observed_audit = (
            reference_payload_sha256,
            audit.coordinate_cpgs_verified,
            max(map(int, config.splits.window_sizes)),
            audit.eligible_probes,
            audit.boundary_exclusions,
            audit.non_acgt_exclusions,
            audit.eligible_probe_order_sha256,
        )
        expected_audit = (
            config.reference_audit.payload_sha256,
            int(config.reference_audit.coordinate_cpgs_verified),
            int(config.reference_audit.maximum_window_size),
            int(config.reference_audit.eligible_probes),
            config.reference_audit.boundary_exclusions,
            config.reference_audit.non_acgt_exclusions,
            config.reference_audit.eligible_probe_order_sha256,
        )
        if observed_audit != expected_audit:
            raise ValueError(
                "exhaustive reference audit differs from protocol configuration: "
                f"expected={expected_audit}, observed={observed_audit}"
            )
        _json_print(
            {
                "boundary_exclusions": audit.boundary_exclusions,
                "coordinate_cpgs_verified": audit.coordinate_cpgs_verified,
                "eligible_probe_order_sha256": audit.eligible_probe_order_sha256,
                "eligible_probes": audit.eligible_probes,
                "input_probes": audit.input_probes,
                "maximum_window_size": max(map(int, config.splits.window_sizes)),
                "non_acgt_exclusions": audit.non_acgt_exclusions,
                "reference_archive_md5": observed_reference_md5,
                "reference_payload_sha256": reference_payload_sha256,
            }
        )
        return
    if arguments.command == "build-site":
        data = build_static_site(
            data_path=arguments.registry,
            template_directory=arguments.template,
            output_directory=arguments.output,
        )
        _json_print(
            {
                "eligibility": data.eligibility,
                "output": str(arguments.output.resolve()),
                "protocol_id": data.protocol_id,
                "status": data.status,
            }
        )
        return
    _unreachable(arguments.command)


def _unreachable(command: str) -> NoReturn:
    raise RuntimeError(f"unreachable command dispatch: {command}")


if __name__ == "__main__":
    main()
