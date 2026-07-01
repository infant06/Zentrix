# Contributing to ZenLLM

Thanks for your interest in contributing! This guide covers how to build, test,
and submit changes.

## Prerequisites

- **Rust** ≥ 1.88 (`rustup` recommended)
- **protobuf compiler** (`protoc`) — required by some transitive build steps
  - Ubuntu/Debian: `sudo apt-get install protobuf-compiler`
  - macOS: `brew install protobuf`
  - Windows: install from https://github.com/protocolbuffers/protobuf/releases and add to `PATH`
- **CUDA Toolkit** (optional, for GPU builds) + a matching NVIDIA driver
- **Python** ≥ 3.10 and `maturin` (only if building the `zentrix` bindings)

## Building

CPU-only:

```bash
cargo build --release -p zenllm            # CLI/server
```

With CUDA — pin your GPU's compute capability so kernels compile correctly
(Turing `75`, Ampere `80`/`86`, Ada `89`, Hopper `90`):

```bash
CUDA_COMPUTE_CAP=86 ./scripts/build.sh cli          # Linux/macOS
powershell -File scripts\build.ps1 -ComputeCap 75   # Windows
```

Python bindings (inside a virtualenv):

```bash
cd crates/zen-python && maturin develop --release --features "cuda,code-execution"
```

## Before submitting a PR

Run the same checks CI runs:

```bash
cargo fmt --all
cargo clippy --workspace --exclude zen-python
cargo check --workspace --exclude zen-python
cargo test --workspace            # where applicable
```

For CUDA/kernel changes, please note in the PR which **compute capability** you
built and tested against, and the GPU model.

## Coding guidelines

- Keep changes focused; one logical change per PR.
- Match the existing style; `cargo fmt` is authoritative for Rust.
- Prefer clear, minimal, well-documented code over cleverness.
- Add or update tests when fixing bugs or adding features.
- Don't commit large binaries, model weights (`*.gguf`), or generated artifacts
  (`target/`, logs). See `.gitignore`.

## Commit and PR conventions

- Write descriptive commit messages (imperative mood, e.g. "Fix rotary FFI width").
- Push to a feature branch and open a PR against `main`.
- Describe what changed, why, and how you verified it.

## Reporting bugs

Open an issue with: your OS, GPU + driver + CUDA version (`zenllm doctor`
output is ideal), the model and mode used, and the exact error. See
`docs/WINDOWS_SETUP_AND_TROUBLESHOOTING.md` for common fixes.

## License

By contributing, you agree that your contributions are licensed under the
project's MIT License.
