# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Some RUN statements are combined together to make Docker build run faster.
# Get latest package listing, install software-properties-common, git, wget,
# compilers and libraries.
# git is required for pyproject.toml toolchain's use of CMakeLists.txt.
# gcc, g++, make are required for compiling hmmer and AlphaFold 3 libaries.
# zlib is a required dependency of AlphaFold 3.
RUN apt-get update --quiet \
    && apt-get install --yes --quiet software-properties-common \
    && apt-get install --yes --quiet git wget gcc g++ make zlib1g-dev zstd 

# Get apt repository of specific Python versions. Then install Python. Tell APT
# this isn't an interactive TTY to avoid timezone prompt when installing.
RUN add-apt-repository ppa:deadsnakes/ppa \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet python3.11 python3-pip python3.11-venv python3.11-dev
RUN python3.11 -m venv /alphafold3_venv
ENV PATH="/hmmer/bin:/alphafold3_venv/bin:$PATH"
# Update pip to the latest version. Not necessary in Docker, but good to do when
# this is used as a recipe for local installation since we rely on new pip
# features for secure installs.
RUN pip3 install --upgrade pip

# Install HMMER. Do so before copying the source code, so that docker can cache
# the image layer containing HMMER. Alternatively, you could also install it
# using `apt-get install hmmer` instead of bulding it from source, but we want
# to have control over the exact version of HMMER and also apply the sequence
# limit patch. Also note that eddylab.org unfortunately doesn't support HTTPS
# and the tar file published on GitHub is explicitly not recommended to be used
# for building from source.

# Download, check hash, and extract the HMMER source code.
RUN mkdir /hmmer_build /hmmer ; \
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /hmmer_build ; \
    (cd /hmmer_build && echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz" | sha256sum --check) && \
    (cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz)

# Apply the --seq_limit patch to HMMER.
COPY docker/jackhmmer_seq_limit.patch /hmmer_build/
RUN (cd /hmmer_build && patch -p0 < jackhmmer_seq_limit.patch)

# Build HMMER.
RUN (cd /hmmer_build/hmmer-3.4 && CC=gcc-10 CXX=g++-10 ./configure --prefix /hmmer) ; \
    (cd /hmmer_build/hmmer-3.4 && CC=gcc-10 CXX=g++-10 make -j) ; \
    (cd /hmmer_build/hmmer-3.4 && CC=gcc-10 CXX=g++-10 make install) ; \
    (cd /hmmer_build/hmmer-3.4/easel && CC=gcc-10 CXX=g++-10  make install) ; \
    rm -R /hmmer_build

# Copy the AlphaFold 3 source code from the local machine to the container and
# set the working directory to there.
COPY . /app/alphafold
WORKDIR /app/alphafold

# Install the Python dependencies AlphaFold 3 needs.
RUN pip3 install -r dev-requirements.txt
RUN pip3 install --no-deps .
# Build chemical components database (this binary was installed by pip).
RUN build_data

# To work around a known XLA issue causing the compilation time to greatly
# increase, the following environment variable setting XLA flags must be enabled
# when running AlphaFold 3. Note that if using CUDA capability 7 GPUs, it is
# necessary to set the following XLA_FLAGS value instead:
# ENV XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
# (no need to disable gemm in that case as it is not supported for such GPU).
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
# Memory settings used for folding up to 5,120 tokens on A100 80 GB.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95

CMD ["python3", "run_alphafold.py"]