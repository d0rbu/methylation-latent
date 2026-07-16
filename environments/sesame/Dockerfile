FROM bioconductor/bioconductor_docker@sha256:b10002b39efa30c3779ad839549806ebdbb29b3266f0d2428478b04426e55929

ENV RENV_CONFIG_AUTOLOADER_ENABLED=false
ENV R_LIBS_USER=/opt/methylation-latent/renv-library

WORKDIR /opt/methylation-latent
COPY renv.lock /opt/methylation-latent/renv.lock

RUN R --vanilla --slave -e \
    'install.packages("https://cran.r-project.org/src/contrib/Archive/renv/renv_1.2.3.tar.gz", repos = NULL, type = "source")' \
    && R --vanilla --slave -e \
    'renv::restore(lockfile = "/opt/methylation-latent/renv.lock", library = Sys.getenv("R_LIBS_USER"), prompt = FALSE)' \
    && R --vanilla --slave -e \
    'stopifnot(as.character(getRversion()) == "4.6.0", as.character(BiocManager::version()) == "3.23", as.character(packageVersion("sesame")) == "1.30.1", as.character(packageVersion("sesameData")) == "1.30.0")'

