"""Pinned methylprep parity runner with exact, unrounded binary exports."""

from __future__ import annotations

import sys
from array import array
from importlib.metadata import version
from pathlib import Path
from re import fullmatch

from methylprep.processing import run_pipeline  # ty: ignore[unresolved-import]


def _write_lines_exclusive(path: Path, values: tuple[str, ...]) -> None:
    with path.open(mode="x", encoding="utf-8", newline="") as handle:
        handle.writelines(f"{value}\n" for value in values)
        handle.flush()


def _write_f64_exclusive(path: Path, values: tuple[float, ...]) -> None:
    payload = array("d", values)
    if payload.itemsize != 8 or sys.byteorder != "little":
        raise RuntimeError("methylprep binary export requires little-endian float64")
    with path.open(mode="xb") as handle:
        payload.tofile(handle)
        handle.flush()


def _write_u8_exclusive(path: Path, values: tuple[bool, ...]) -> None:
    payload = array("B", (int(value) for value in values))
    if payload.itemsize != 1:
        raise RuntimeError("methylprep binary export requires one-byte unsigned values")
    with path.open(mode="xb") as handle:
        payload.tofile(handle)
        handle.flush()


def main(arguments: list[str]) -> None:
    if len(arguments) != 4:
        raise SystemExit(
            "usage: preprocess.py IDAT_DIRECTORY SAMPLE_SHEET MANIFEST OUTPUT_DIRECTORY"
        )
    idat_directory, sample_sheet, manifest, output_directory = map(Path, arguments)
    if not idat_directory.is_dir():
        raise ValueError(f"IDAT directory does not exist: {idat_directory}")
    if not sample_sheet.is_file() or not manifest.is_file():
        raise ValueError("methylprep sample sheet and manifest must both exist")
    if output_directory.exists():
        raise FileExistsError(f"output path already exists: {output_directory}")

    containers = run_pipeline(
        data_dir=idat_directory,
        array_type="450k",
        export=False,
        manifest_filepath=manifest,
        sample_sheet_filepath=sample_sheet,
        betas=False,
        m_value=False,
        save_uncorrected=False,
        save_control=False,
        meta_data_frame=False,
        bit="float64",
        poobah=True,
        export_poobah=False,
        poobah_decimals=16,
        poobah_sig=0.01,
        low_memory=True,
        sesame=True,
        quality_mask=True,
    )
    if not isinstance(containers, list) or not containers:
        raise RuntimeError("methylprep returned no sample containers")

    output_directory.mkdir(parents=True)
    expected_probe_ids: tuple[str, ...] | None = None
    sample_ids: list[str] = []
    for container in containers:
        sample_id = str(container.sample.name)
        if sample_id != str(container.sample):
            raise ValueError(
                f"methylprep sample identity changed: name={sample_id}, sentrix={container.sample}"
            )
        frame = container._SampleDataContainer__data_frame
        required_columns = {"beta_value", "poobah_pval", "quality_mask"}
        if not required_columns.issubset(frame.columns):
            raise ValueError(
                "methylprep output lacks required columns: "
                f"{sorted(required_columns - set(frame.columns))}"
            )
        frame = frame.loc[
            [fullmatch(r"cg[0-9]{8}", str(probe_id)) is not None for probe_id in frame.index]
        ]
        probe_ids = tuple(map(str, frame.index))
        beta = tuple(map(float, frame["beta_value"]))
        detection_p = tuple(map(float, frame["poobah_pval"]))
        quality_excluded = tuple(float(value) == 0.0 for value in frame["quality_mask"])
        if (
            not probe_ids
            or len(set(probe_ids)) != len(probe_ids)
            or len(
                {
                    len(probe_ids),
                    len(beta),
                    len(detection_p),
                    len(quality_excluded),
                }
            )
            != 1
        ):
            raise ValueError(f"methylprep emitted malformed probe axes for {sample_id}")
        if expected_probe_ids is None:
            expected_probe_ids = probe_ids
            _write_lines_exclusive(output_directory / "probe_ids.txt", probe_ids)
        elif probe_ids != expected_probe_ids:
            raise ValueError(f"methylprep changed probe order for {sample_id}")
        _write_f64_exclusive(output_directory / f"{sample_id}.beta.f64", beta)
        _write_f64_exclusive(output_directory / f"{sample_id}.detection_p.f64", detection_p)
        _write_u8_exclusive(
            output_directory / f"{sample_id}.quality_excluded.u8",
            quality_excluded,
        )
        sample_ids.append(sample_id)

    _write_lines_exclusive(output_directory / "sample_order.txt", tuple(sample_ids))
    _write_lines_exclusive(
        output_directory / "environment.txt",
        (
            f"python={sys.version.split()[0]}",
            f"methylprep={version('methylprep')}",
            "pipeline=infer-channel-pOOBAH@0.01-quality-noob-nonlinear-dye",
            "probe_filter=^cg[0-9]{8}$",
            "poobah_decimals=16",
            "beta_mask=false",
            f"samples={len(sample_ids)}",
            f"probes={len(expected_probe_ids or ())}",
        ),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
