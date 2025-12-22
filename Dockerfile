FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONDA_DIR=/opt/conda \
    MAMBA_ROOT_PREFIX=/opt/conda

# MAJ APT (IPv4 + retries) + bzip2/certs uniquement
RUN apt-get -o Acquire::Retries=5 -o Acquire::ForceIPv4=true update && \
    apt-get install -y --no-install-recommends \
      bzip2 \
      ca-certificates \
      curl \
      build-essential \
    && \
    rm -rf /var/lib/apt/lists/*

# micromamba binaire
RUN curl -L --retry 5 --retry-delay 2 --retry-all-errors \
    https://micro.mamba.pm/api/micromamba/linux-64/latest \
    -o /tmp/micromamba.tar.bz2
RUN mkdir -p /tmp/micromamba && \
    tar -xvjf /tmp/micromamba.tar.bz2 -C /tmp/micromamba && \
    mkdir -p "$CONDA_DIR" && \
    /tmp/micromamba/bin/micromamba shell init -s bash -r "$CONDA_DIR" > /etc/profile.d/micromamba.sh && \
    rm /tmp/micromamba.tar.bz2

ENV PATH="/opt/conda/bin:/tmp/micromamba/bin:$PATH"

RUN micromamba create -y -n call2eds -c conda-forge \
      python=3.11 \
      pip=24.0 \
      numpy=2.3.5 \
      pandas=2.2.3 \
      pyarrow=15.0.2 \
      ffmpeg \
      libsndfile \
      openblas \
      setuptools wheel \
    && micromamba clean -ay

ENV PATH="/opt/conda/envs/call2eds/bin:$PATH" \
    CONDA_DEFAULT_ENV=call2eds

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY samples ./samples

RUN pip install --no-cache-dir .

RUN useradd -m worker && chown -R worker:worker /app
USER worker

ENV PATH="/home/worker/.local/bin:$PATH"

ENTRYPOINT ["call2eds"]
CMD ["--help"]
