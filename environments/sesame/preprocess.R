suppressPackageStartupMessages({
    library(BiocParallel)
    library(sesame)
})

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 3) {
    stop("usage: preprocess.R INPUT_TSV OUTPUT_DIRECTORY WORKERS")
}

input_path <- arguments[[1]]
output_directory <- arguments[[2]]
workers <- as.integer(arguments[[3]])
if (is.na(workers) || workers <= 0) {
    stop("worker count must be a positive integer")
}
if (!file.exists(input_path)) {
    stop(sprintf("input TSV does not exist: %s", input_path))
}
if (dir.exists(output_directory) || file.exists(output_directory)) {
    stop(sprintf("output path already exists: %s", output_directory))
}
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

samples <- read.delim(
    input_path,
    header = TRUE,
    sep = "\t",
    quote = "",
    stringsAsFactors = FALSE,
    check.names = FALSE
)
if (!identical(colnames(samples), c("sentrix_identity", "prefix"))) {
    stop("input TSV must have exactly sentrix_identity and prefix columns")
}
if (nrow(samples) == 0) {
    stop("input TSV must contain at least one sample")
}
if (anyDuplicated(samples$sentrix_identity) || anyDuplicated(samples$prefix)) {
    stop("input TSV contains duplicate Sentrix identities or prefixes")
}
if (any(!grepl("^[0-9]{10}_R0[1-6]C0[1-2]$", samples$sentrix_identity))) {
    stop("input TSV contains an invalid Sentrix identity")
}

write_binary_exclusive <- function(path, values, size) {
    if (file.exists(path)) {
        stop(sprintf("refusing to overwrite preprocessing output: %s", path))
    }
    temporary <- sprintf("%s.tmp.%d", path, Sys.getpid())
    if (file.exists(temporary)) {
        stop(sprintf("temporary preprocessing path already exists: %s", temporary))
    }
    connection <- file(temporary, open = "wb")
    on.exit(close(connection), add = TRUE)
    writeBin(values, connection, size = size, endian = "little")
    close(connection)
    on.exit(NULL, add = FALSE)
    observed_bytes <- file.info(temporary)$size
    expected_bytes <- length(values) * size
    if (observed_bytes != expected_bytes) {
        unlink(temporary)
        stop(
            sprintf(
                "binary output size differs for %s: expected=%d observed=%d",
                path,
                expected_bytes,
                observed_bytes
            )
        )
    }
    if (!file.rename(temporary, path)) {
        unlink(temporary)
        stop(sprintf("failed to atomically publish preprocessing output: %s", path))
    }
}

process_sample <- function(sentrix_identity, prefix) {
    signal <- readIDATpair(prefix, min_beads = 1)
    platform <- sdfPlatform(signal)
    if (!identical(platform, "HM450")) {
        stop(sprintf("%s has platform %s, expected HM450", sentrix_identity, platform))
    }

    corrected <- prepSesame(signal, prep = "QCD")
    quality_excluded <- setNames(as.logical(corrected$mask), corrected$Probe_ID)
    detection_p <- pOOBAH(corrected, return.pval = TRUE)
    corrected <- pOOBAH(corrected, pval.threshold = 0.05)
    corrected <- noob(corrected)
    beta <- getBetas(corrected, mask = FALSE)
    probe_ids <- names(beta)

    if (
        is.null(probe_ids) ||
        anyDuplicated(probe_ids) ||
        !setequal(probe_ids, names(detection_p)) ||
        !setequal(probe_ids, names(quality_excluded))
    ) {
        stop(sprintf("%s has malformed or misaligned probe identities", sentrix_identity))
    }
    detection_p <- detection_p[probe_ids]
    quality_excluded <- quality_excluded[probe_ids]
    canonical_cpg <- grepl("^cg[0-9]{8}$", probe_ids)
    if (!any(canonical_cpg)) {
        stop(sprintf("%s produced no canonical CpG probes", sentrix_identity))
    }
    probe_ids <- probe_ids[canonical_cpg]
    beta <- beta[canonical_cpg]
    detection_p <- detection_p[canonical_cpg]
    quality_excluded <- quality_excluded[canonical_cpg]
    if (
        anyNA(beta) ||
        anyNA(detection_p) ||
        anyNA(quality_excluded) ||
        any(!is.finite(beta)) ||
        any(!is.finite(detection_p)) ||
        any(beta < 0 | beta > 1) ||
        any(detection_p < 0 | detection_p > 1)
    ) {
        stop(sprintf("%s produced invalid beta, p-value, or mask values", sentrix_identity))
    }
    list(
        probe_ids = probe_ids,
        beta = as.double(beta),
        detection_p = as.double(detection_p),
        quality_excluded = as.logical(quality_excluded)
    )
}

write_sample <- function(sentrix_identity, prefix, expected_probe_ids) {
    result <- process_sample(sentrix_identity, prefix)
    if (!identical(result$probe_ids, expected_probe_ids)) {
        stop(sprintf("%s changed the probe order", sentrix_identity))
    }
    write_binary_exclusive(
        file.path(output_directory, sprintf("%s.beta.f64", sentrix_identity)),
        result$beta,
        8
    )
    write_binary_exclusive(
        file.path(output_directory, sprintf("%s.detection_p.f64", sentrix_identity)),
        result$detection_p,
        8
    )
    write_binary_exclusive(
        file.path(
            output_directory,
            sprintf("%s.quality_excluded.u8", sentrix_identity)
        ),
        as.raw(as.integer(result$quality_excluded)),
        1
    )
    sentrix_identity
}

first <- process_sample(samples$sentrix_identity[[1]], samples$prefix[[1]])
probe_id_path <- file.path(output_directory, "probe_ids.txt")
writeLines(first$probe_ids, probe_id_path, sep = "\n", useBytes = TRUE)
if (length(readLines(probe_id_path, warn = FALSE)) != length(first$probe_ids)) {
    stop("probe-ID output row count changed during serialization")
}
write_binary_exclusive(
    file.path(
        output_directory,
        sprintf("%s.beta.f64", samples$sentrix_identity[[1]])
    ),
    first$beta,
    8
)
write_binary_exclusive(
    file.path(
        output_directory,
        sprintf("%s.detection_p.f64", samples$sentrix_identity[[1]])
    ),
    first$detection_p,
    8
)
write_binary_exclusive(
    file.path(
        output_directory,
        sprintf("%s.quality_excluded.u8", samples$sentrix_identity[[1]])
    ),
    as.raw(as.integer(first$quality_excluded)),
    1
)

if (nrow(samples) > 1) {
    indices <- seq.int(2, nrow(samples))
    completed <- bplapply(
        indices,
        function(index) {
            write_sample(
                samples$sentrix_identity[[index]],
                samples$prefix[[index]],
                first$probe_ids
            )
        },
        BPPARAM = MulticoreParam(workers = min(workers, length(indices)), stop.on.error = TRUE)
    )
    completed <- unlist(completed, use.names = FALSE)
    if (!identical(completed, samples$sentrix_identity[indices])) {
        stop("parallel preprocessing completion order differs from sample order")
    }
}

writeLines(
    samples$sentrix_identity,
    file.path(output_directory, "sample_order.txt"),
    sep = "\n",
    useBytes = TRUE
)
writeLines(
    c(
        sprintf("R=%s", as.character(getRversion())),
        sprintf("Bioconductor=%s", as.character(BiocManager::version())),
        sprintf("sesame=%s", as.character(packageVersion("sesame"))),
        sprintf("sesameData=%s", as.character(packageVersion("sesameData"))),
        "pipeline=QCD-pOOBAH@0.05-B",
        "probe_filter=^cg[0-9]{8}$",
        "beta_mask=false",
        sprintf("samples=%d", nrow(samples)),
        sprintf("probes=%d", length(first$probe_ids))
    ),
    file.path(output_directory, "environment.txt"),
    sep = "\n",
    useBytes = TRUE
)
