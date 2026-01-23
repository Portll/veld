# Custom cross-compilation image for aarch64-unknown-linux-gnu
# Extends the cross-rs base image with LLVM 14 for bindgen compatibility
#
# The default cross image has clang-10, but librocksdb-sys uses bindgen which
# requires clang_Type_getValueType (added in clang-14). This Dockerfile installs
# LLVM 14 and sets up the environment correctly.

FROM ghcr.io/cross-rs/aarch64-unknown-linux-gnu:main

# Install LLVM 14 from official LLVM apt repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    software-properties-common \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main" >> /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        llvm-14 \
        llvm-14-dev \
        libclang-14-dev \
        clang-14 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for bindgen to find LLVM 14
ENV LLVM_CONFIG_PATH=/usr/bin/llvm-config-14
ENV LIBCLANG_PATH=/usr/lib/llvm-14/lib
ENV CLANG_PATH=/usr/bin/clang-14

# Verify clang version
RUN clang-14 --version && llvm-config-14 --version
