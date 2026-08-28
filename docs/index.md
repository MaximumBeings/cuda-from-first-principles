# CUDA From First Principles — Under Development

**Building a GPU-Native Tensor and Autodiff Framework in CUDA C++, One Layer at a Time**

!!! info "This book is just getting started"
    This is the second sibling project to [*Mojo From First Principles*](https://maximumbeings.github.io/mojo-from-first-principles/), alongside [*Triton From First Principles*](https://maximumbeings.github.io/triton-from-first-principles/), built the same way: one chapter at a time, in page order, each one carrying real code and honestly labeled output. This environment has no NVIDIA GPU, so every kernel in this book is genuinely compiled with `nvcc` for real target architectures (and, for tensor-core kernels, genuinely disassembled to show the actual machine instructions `ptxas` emits) — but not executed here. Runtime numbers are captured on real hardware as a separate pass once the book is otherwise complete, and every chapter says plainly which of its claims rest on compilation evidence versus a real run.

> "A `torch.Tensor` and Mojo's `Tensor[dtype]` both arrive at a kernel boundary already built. CUDA C++ gives you neither — a `malloc`, a `cudaMalloc`, and a raw pointer, with the shape, the stride, and the device all left for you to invent. This book is about building that invention once, correctly, and then never touching a bare pointer again."

This is a practitioner's guide to CUDA C++ that follows the same arc as its Mojo sibling rather than its Triton one: instead of consuming an existing tensor library, it builds one — a small, real `Tensor` class with its own shape and stride bookkeeping, its own device abstraction, and its own memory manager — and then builds a genuine reverse-mode automatic differentiation engine on top of it, node by node, exactly the way the Mojo book does. Where it necessarily departs from Mojo is where CUDA itself is different: an explicit host/device split from the first line of code, warp-level SIMT execution as a first-class concept, and, from Part 6 onward, NVIDIA's tensor cores — the dedicated matrix-multiply hardware behind the `wmma` API — as the engine behind this book's neural-network layers and fused kernels.

The design principles carried through the whole book:

- **Build the tensor before you build anything on top of it.** Shape, stride, dtype, and device ownership are modeled explicitly as a `Tensor` class in Part 1, the same way Mojo's `Tensor[dtype]` is — not assumed from a framework, the way the Triton book assumes them from PyTorch.
- **Your own autodiff engine, not someone else's.** Parts 3 and 4 build a real computational graph and a real backward traversal in C++ — no `torch.autograd` anywhere in this book's core chapters.
- **Tensor cores as a first-class citizen.** Part 6 is built around NVIDIA's `wmma` tensor-core API, with real compiled and disassembled `HMMA` machine instructions shown wherever a kernel claims to use them.
- **Honest about what "verified" means without a GPU.** Every kernel genuinely compiles for real architectures (`sm_70` through `sm_90`) and, where relevant, genuinely disassembles to real SASS — but this book never fabricates a captured runtime number it didn't actually measure.
- **Financial computing ready.** The closing chapter validates these kernels against the same Black-Scholes, rolling-volatility, and Monte Carlo option-pricing problems the Mojo and Triton books end on.

<div class="grid cards" markdown>

- :material-book-open-page-variant:{ .lg .middle } **22 chapters**

    Part 0 through Part 7, from raw pointers to a Monte Carlo option pricer.

- :material-memory:{ .lg .middle } **A real Tensor class**

    Shape, stride, dtype, and device ownership, built from scratch in Part 1.

- :material-chip:{ .lg .middle } **Tensor cores, genuinely compiled**

    Real `wmma` kernels, disassembled to real `HMMA` SASS instructions.

- :material-finance:{ .lg .middle } **Real financial models**

    Black-Scholes, rolling volatility, and Monte Carlo pricing, GPU-native.

</div>

## How the book is organized

| Part | Focus |
|---|---|
| **Part 0 — CUDA Foundations** | C++ types across the host/device boundary, struct design patterns, memory layout strategies, the kernel launch model, warp-level SIMD |
| **Part 1 — Core Tensor Infrastructure** | The `Tensor` class itself, memory layout design, creation functions, specialized (half-precision) tensor types, the device abstraction layer, the memory management system |
| **Part 2 — Basic Tensor Operations** | Element-wise operations, matrix operations, reduction operations |
| **Part 3 — Computational Graph Foundation** | A real graph-node architecture, built from scratch |
| **Part 4 — Automatic Differentiation Engine** | Backward function implementation, the gradient computation engine (topological traversal, fan-in accumulation) |
| **Part 5 — GPU Acceleration and Performance** | GPU kernel implementation and tiling, performance optimization techniques (occupancy, bank conflicts, register pressure) |
| **Part 6 — Neural Network Building Blocks** | Neural network layers built on tensor-core `wmma` GEMM, advanced fused features (Flash-Attention-style fusion, MoE gating) |
| **Part 7 — Financial Computing Applications** | Black-Scholes, rolling volatility, and Monte Carlo option pricing |

Start with [Getting Started](getting-started.md) to stand up an `nvcc` toolchain, or jump straight into [Part 0: CUDA Foundations](part0/01-variables-and-types.md) if your compiler is already installed.

!!! note "How this book relates to its siblings"
    Mojo builds its own `Tensor[dtype]` struct and its own autodiff engine because Mojo is a systems language with nothing else to lean on. Triton deliberately does neither — it owns only the kernel, and leans on PyTorch for the tensor, the device, and the autograd graph. CUDA C++ is, in this respect, much closer to Mojo's position than Triton's: it has no built-in tensor abstraction and no autograd engine of its own, so this book follows Mojo's arc rather than Triton's, building both from scratch. What's genuinely new here, with no equivalent in either sibling, is the explicit host/device compilation split from Chapter 1 onward and NVIDIA's tensor cores as Part 6's central hardware feature.
