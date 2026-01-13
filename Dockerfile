# Use Mambaforge as base (lightweight, fast, pre-configured for conda-forge)
FROM condaforge/mambaforge:latest

# Label metadata
LABEL maintainer="Theo Portlock"
LABEL description="Hybrid Python/R Metatoolkit for Microbiome Analysis"

# 1. Install system tools required for plotting (matplotlib/R) and Ollama connectivity
# libgl1 is often needed for Python graphics backends
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy environment file first (better caching)
COPY environment.yml .

# 4. Create the environment
# We install into the base environment to avoid needing 'conda activate' inside Docker
RUN mamba env update -n base -f environment.yml && \
    mamba clean --all -f -y

# 5. Copy your scripts into the container
# Assuming your scripts are in a folder named 'scripts' or 'metatoolkit'
COPY . /app/metatoolkit

# 6. Make scripts executable and add to PATH
# This allows you to run 'abund.py' from anywhere in the container
RUN chmod +x /app/metatoolkit/*.py /app/metatoolkit/*.R /app/metatoolkit/*.sh
ENV PATH="/app/metatoolkit:${PATH}"

# 7. Default command (optional, can be a shell)
CMD ["/bin/bash"]
