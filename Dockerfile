# Pin by digest once built: docker pull condaforge/miniforge3:latest
#   && docker inspect --format '{{index .RepoDigests 0}}' condaforge/miniforge3:latest
FROM condaforge/miniforge3:latest

LABEL maintainer="Theo Portlock"
LABEL description="Hybrid Python/R Metatoolkit for Microbiome Analysis"

# git is needed for the pip git+https installs (metatoolkit, workforce)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libgl1 \
    && rm -rf /var/lib/apt/lists/*

# conda env from the lock file (includes the pip layer)
WORKDIR /app
COPY conda-lock.yml .
RUN mamba install -y -n base -c conda-forge conda-lock \
    && conda-lock install -n metatoolkit conda-lock.yml \
    && mamba clean -afy
ENV PATH=/opt/conda/envs/metatoolkit/bin:${PATH}

# R packages that aren't on conda, pinned by CRAN snapshot date.
# install.packages() only warns on failure, so stopifnot makes the build fail instead.
RUN Rscript -e 'options(repos = c(CRAN = "https://packagemanager.posit.co/cran/2024-03-01")); install.packages("anthro"); stopifnot(requireNamespace("anthro", quietly = TRUE))'

# Smoke test: fails the build if either command is missing
RUN mtk --help > /dev/null && which workforce

CMD ["/bin/bash"]
