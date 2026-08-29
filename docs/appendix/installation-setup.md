# Appendix A: Installation & Setup

## Local machine with an NVIDIA GPU

Every kernel in this book is written and explained against a compute-capability 8.0 (Ampere, e.g. A100) target and compiled with `-arch=sm_80` throughout — that's the one thing to match if you want the exact `nvcc`/`cuobjdump`/`nm` output this book's own chapters and appendices show. Anything Turing (`sm_75`) or newer will compile and run every example correctly; only the SASS disassembly listings in chapters that inspect compiled instructions directly may look slightly different on a different architecture.

Install the CUDA Toolkit for your distribution from NVIDIA's own installer (the exact steps vary by OS and package manager, so this book intentionally doesn't try to replace [NVIDIA's own CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads)). Once installed, confirm both halves of the toolchain are visible before going any further:

```bash
nvidia-smi
nvcc --version
```

`nvidia-smi` reports the driver and the GPU the driver sees; `nvcc --version` reports the CUDA compiler's own version, which is not always the same number `nvidia-smi` shows (the driver can support a newer CUDA version than the toolkit installed alongside it, and vice versa within limits). If either command fails, fix that before compiling anything in this book — every chapter assumes both are already working.

Compile and run this book's kernels the same way every chapter's own compile block shows:

```bash
nvcc -arch=sm_80 some_example.cu -o some_example
./some_example
```

Chapters that inspect compiled machine code directly (register counts, spill bytes, SASS instructions) add extra flags on top of this baseline — `-Xptxas -v` for register/spill reporting, `-lineinfo` plus `cuobjdump --dump-sass` for disassembly — each introduced inline, at the point in the book where it first matters.

## No local NVIDIA GPU: cloud instances and honest fallback behavior

Any CUDA-enabled cloud instance works — a single consumer or datacenter GPU is enough for every example in this book. A typical setup on a fresh Ubuntu instance with the CUDA Toolkit already provisioned by the image:

```bash
nvidia-smi
nvcc --version
git clone https://github.com/MaximumBeings/cuda-from-first-principles.git
cd cuda-from-first-principles
```

If you're reading this appendix with no GPU at all — the exact situation this book's own later chapters and its Appendix C were written under — every Runtime API call this book makes (`cudaMalloc`, `cudaMallocManaged`, `cudaMemcpyToSymbol`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, and so on) still compiles and runs; it simply, honestly reports `cudaErrorNoDevice` instead of succeeding. That failure mode is deliberate and load-bearing for this book's own methodology from Chapter 18 onward: rather than skip examples that need a device, later chapters run them anyway and show you the real, unedited error a Runtime API call returns with no GPU present, alongside independently hand-verified or host-computed arithmetic proving the surrounding logic is correct regardless. You do not need a GPU to read or follow this book end to end — you need one only to see a kernel's numbers actually come back from a device instead of a documented, cited reference figure.

## Version notes

This book targets CUDA 12.x and compute capability 8.0 (`-arch=sm_80`) throughout, chosen because it's old enough to be broadly available on cheap cloud instances and new enough to support every feature this book uses, including the WMMA tensor-core API in Appendix C.7. Nothing here depends on a version-specific CUDA language feature introduced after CUDA 11 — if your toolkit is newer, everything still compiles; if you're on an older toolkit, check `nvcc --version` against a specific chapter's compile block before assuming a mismatch is the problem, since most build failures at that point come from a missing `-arch` flag rather than a genuine version incompatibility.

## Building this book's own documentation site

```bash
git clone https://github.com/MaximumBeings/cuda-from-first-principles.git
cd cuda-from-first-principles
python3 -m venv .venv && source .venv/bin/activate
pip install mkdocs-material
mkdocs serve
```

`mkdocs serve` builds no CUDA code — it only renders the markdown in `docs/` into the site you're reading now. None of the commands above need a GPU; only the compile-and-run steps earlier in this appendix do.
