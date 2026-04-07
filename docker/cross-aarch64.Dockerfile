# Custom cross-compilation image for aarch64-unknown-linux-gnu
# Built on Ubuntu 22.04 for GLIBC 2.35 compatibility with modern GitHub runners
#
# This image provides:
# - GLIBC 2.35 (compatible with GitHub runners that have GLIBC 2.39)
# - LLVM 14 for bindgen/librocksdb-sys (clang_Type_getValueType)
# - ARM64 cross-compilation toolchain

FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install base development tools and ARM64 cross-compilation toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential build tools
    build-essential \
    curl \
    ca-certificates \
    pkg-config \
    # ARM64 cross-compilation
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    libc6-dev-arm64-cross \
    # LLVM 14 for bindgen (RocksDB)
    llvm-14 \
    llvm-14-dev \
    libclang-14-dev \
    clang-14 \
    # Additional tools
    cmake \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Add ARM64 target
RUN rustup target add aarch64-unknown-linux-gnu

# Configure cargo for cross-compilation
RUN mkdir -p /root/.cargo && echo '\
[target.aarch64-unknown-linux-gnu]\n\
linker = "aarch64-linux-gnu-gcc"\n\
' > /root/.cargo/config.toml

# Set environment variables for bindgen to find LLVM 14
ENV LLVM_CONFIG_PATH=/usr/bin/llvm-config-14
ENV LIBCLANG_PATH=/usr/lib/llvm-14/lib
ENV CLANG_PATH=/usr/bin/clang-14

# Set cross-compilation environment variables
ENV CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
ENV CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
ENV AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc

# Verify setup
RUN clang-14 --version && \
    llvm-config-14 --version && \
    aarch64-linux-gnu-gcc --version && \
    rustc --version && \
    cargo --version

WORKDIR /project
