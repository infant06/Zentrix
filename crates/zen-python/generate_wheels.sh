#!/bin/bash
# Wheel generation commands for each target machine.
# Uses scripts/build_wheels.py which auto-detects platform and builds appropriate wheels.
#
# Build method:
# - Docker manylinux: ONLY for CPU-only builds on Linux (no features)
# - Native maturin: For CUDA, MKL, Metal, Accelerate builds

###############################################################################
# BOX 1: Linux aarch64 + CUDA
###############################################################################

# zentrix: CPU-only, uses Docker manylinux with RUSTFLAGS="-C target-cpu=generic"
# zentrix-cuda: uses native maturin (not Docker)
python scripts/build_wheels.py -p zentrix zentrix-cuda

###############################################################################
# BOX 2: Linux x86_64 + CUDA + MKL
###############################################################################

# zentrix: has MKL, uses native maturin (not Docker, because MKL feature)
# zentrix-cuda: uses native maturin
# zentrix-mkl: uses native maturin
python scripts/build_wheels.py -p zentrix zentrix-cuda zentrix-mkl

###############################################################################
# BOX 2: Windows x86_64 + CUDA + MKL
###############################################################################

# All use native maturin (no Docker on Windows)
python scripts/build_wheels.py -p zentrix zentrix-cuda zentrix-mkl

###############################################################################
# BOX 3: macOS aarch64 + Metal
###############################################################################

# All use native maturin with MACOSX_DEPLOYMENT_TARGET=15.0 for Metal
python scripts/build_wheels.py --all

###############################################################################
# UPLOADING
###############################################################################

# Collect all wheels from all boxes to a single directory, then:

# Dry run to verify:
# python scripts/upload_wheels.py ./all_wheels --dry-run

# Upload to TestPyPI first:
# python scripts/upload_wheels.py ./all_wheels --test --token $TESTPYPI_TOKEN

# Upload to PyPI:
# python scripts/upload_wheels.py ./all_wheels --token $PYPI_TOKEN

###############################################################################
# PACKAGE SUMMARY
###############################################################################
#
# Package              | Features    | Platforms                      | Build Method
# ---------------------|-------------|--------------------------------|------------------
# zentrix            | (none)      | Linux aarch64                  | Docker manylinux
# zentrix            | mkl         | Linux/Windows x86_64           | Native maturin
# zentrix            | metal       | macOS aarch64                  | Native maturin
# zentrix-cuda       | cuda        | Linux + Windows (x86_64/arm64) | Native maturin
# zentrix-metal      | metal       | macOS aarch64                  | Native maturin
# zentrix-accelerate | accelerate  | macOS aarch64                  | Native maturin
# zentrix-mkl        | mkl         | Linux + Windows x86_64         | Native maturin
#
# Python version: 3.10 only (abi3 provides forward compatibility to 3.11+)
