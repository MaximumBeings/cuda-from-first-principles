# Getting Started

This book targets CUDA C++ compiled with `nvcc` for real NVIDIA GPU architectures — `sm_70` (Volta) through `sm_90` (Hopper), since Part 6's tensor-core chapters need at least `sm_70` for `wmma` support. A real NVIDIA GPU with a matching driver is what you need to *run* the code in this book; `nvcc` itself only needs the CUDA toolkit and will happily compile for architectures your current machine doesn't have.

## Environment

```bash
# Ubuntu/Debian, using NVIDIA's own apt repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

Confirm the driver and toolkit are both visible:

```bash
nvidia-smi      # real driver + GPU, needed to run anything
nvcc --version  # compiler, needed to build anything
```

## Verify the toolchain

```cpp
// hello_cuda.cu
#include <cstdio>

__global__ void hello_kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

```bash
nvcc -arch=sm_80 hello_cuda.cu -o hello_cuda   # pick the arch flag matching your GPU
./hello_cuda
```

A working setup prints 8 lines, one per thread across both blocks, in whatever order the scheduler happens to run them — that non-determinism in the ordering is itself the first real lesson in Chapter 4.

## A note on how this book was written

This book's reference implementations were authored and compiled in an environment with a full CUDA toolkit (`nvcc`, `ptxas`, `cuobjdump`) but no NVIDIA GPU. Every kernel in this book is genuinely compiled for real target architectures, and tensor-core kernels are genuinely disassembled to show real `HMMA` machine instructions and real `ptxas`-reported register/shared-memory usage — none of that is fabricated. What that environment cannot do is *execute* a kernel, so any chapter's runtime numbers, timing comparisons, or numerically-checked output were captured in a separate real-hardware pass rather than alongside the prose. Each chapter says explicitly which of its claims rest on compilation evidence and which rest on an actual run.

## Building this book locally

```bash
pip install mkdocs-material
mkdocs build --strict
mkdocs serve
```

`mkdocs serve` runs a local preview server; `mkdocs build --strict` is what every chapter in this book is checked against before it's committed.

The rendered book will be published at <https://maximumbeings.github.io/cuda-from-first-principles/> once the first deploy goes out.

See [Appendix A](appendix/installation-setup.md) for cloud-GPU-specific setup notes (Lambda, AWS, and similar providers).
