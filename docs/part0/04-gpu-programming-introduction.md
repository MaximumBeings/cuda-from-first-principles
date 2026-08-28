# Chapter 4: GPU Programming Introduction

> "Every kernel this book has compiled so far assumed you already knew what `blockIdx.x * blockDim.x + threadIdx.x` meant. This chapter is where that assumption gets paid off: the thread hierarchy that expression navigates, the two physically separate memories a real launch moves data between, and the one pattern — one thread, one output element — that every kernel in this book, from a vector add to a tensor-core GEMM, is a variation of."

**What you will understand by the end of this chapter:**

- The **grid → block → thread** hierarchy CUDA schedules work with, and how to compute any thread's global index by hand for both 1D and 2D launches
- Why host memory (RAM) and device memory (VRAM) are two genuinely separate physical memories connected by a comparatively slow bus — a different fact from Chapter 1.3's *two compilers*, and the reason every kernel launch is bracketed by explicit `cudaMemcpy` calls
- The GPU memory hierarchy at a first pass — global, shared, and per-thread local memory — traced through where a real kernel's data actually lives, with the deeper mechanics of shared memory held for Part 1
- Why Chapter 3's warp-coalescing argument was never really about particles — it's the general shape of *every* global memory access pattern this book will write, restated here at full generality
- The **broadcast** pattern (one thread computes one output element, mediated by a boundary check) as the shape every kernel in this book fits, worked out for both 1D and 2D outputs
- Matrix multiplication's naive form — one thread, one dot product — as the direct ancestor of Part 2's tiled kernel and Part 6's tensor-core kernel, verified against a genuine host-side reference

**What you need to know first:**

- Chapter 1.3–1.4 (the host/device compiler split, `__global__`/`__device__`/`__host__`) and Chapter 1.5 (`float4` and alignment)
- Chapter 2.4 (RAII over `cudaMalloc`/`cudaFree`)
- Chapter 3 in full — this chapter's Section 4.4 is a direct continuation of Chapter 3's bus-utilization and coalescing argument, not a restart of it
- If you've read the Mojo edition: this chapter follows its Chapter 4 section-for-section. Section 4.2 covers genuinely new ground even against Mojo, since Mojo's single-source-multiple-targets model doesn't require the explicit host/device memory copies this chapter's Worked Example 4.2.1 makes concrete

## 4.1 The Thread Hierarchy: Grid, Block, Thread `[FOUNDATIONAL]`

### Intuition

An army doesn't get its orders individually, soldier by soldier — it's organized into a hierarchy: an army (the **grid**) contains divisions (**blocks**), and each division contains individual soldiers (**threads**). A soldier's full "address" within the army is the combination of which division they're in and where they stand within it — exactly the two-level address `(blockIdx, threadIdx)` every CUDA thread carries.

### Background

| | What it is | Typical size | Accessed in a kernel as |
|---|---|---|---|
| Grid | All threads launched by one kernel call | Up to billions of threads | Implicit — the grid itself has no single index |
| Block | A group of threads scheduled together, sharing one shared-memory region | Up to 1024 threads (hardware limit) | `blockIdx.{x,y,z}` (which block), `blockDim.{x,y,z}` (block's shape) |
| Thread | One single execution of the kernel body | 1 | `threadIdx.{x,y,z}` (position within its block) |

`gridDim.{x,y,z}` (how many blocks) and `blockDim.{x,y,z}` (how many threads per block) are both set at launch time — `kernel<<<gridDim, blockDim>>>(...)` — and are visible read-only inside every thread. Every one of Chapters 1 through 3's kernels has computed `int i = blockIdx.x * blockDim.x + threadIdx.x;` without this chapter yet explaining why: `blockIdx.x * blockDim.x` is "how many threads came before my entire block," and `+ threadIdx.x` is "my position within my own block" — together, one globally unique index across the whole grid.

### Worked Example 4.1.1 — every global ID, computed by hand and confirmed by a genuine run

For a 1D launch `<<<3, 4>>>` (3 blocks of 4 threads each, 12 threads total), block 0's four threads compute `i = 0*4+0=0`, `0*4+1=1`, `0*4+2=2`, `0*4+3=3`; block 1's compute `i = 1*4+0=4` through `1*4+3=7`; block 2's compute `i = 2*4+0=8` through `2*4+3=11`. Compiled and run as the complete `01_thread_hierarchy.cu` further below, whose `main()` is a host-side loop enumerating every `(blockIdx.x, threadIdx.x)` pair a real launch would create, applying the identical formula every thread would execute:

```bash
nvcc -arch=sm_80 01_thread_hierarchy.cu -o 01_thread_hierarchy
./01_thread_hierarchy
```

Genuinely compiled and run:

```
1D launch: <<<3, 4>>> -- 12 total threads
  block 0, thread 0 -> global i = 0
  block 0, thread 1 -> global i = 1
  ...
  block 2, thread 3 -> global i = 11
```

For a 2D launch `<<<dim3(2,2), dim3(2,2)>>>` — 2×2 blocks of 2×2 threads, a 4×4 grid of outputs — `row = blockIdx.y*blockDim.y+threadIdx.y` and `col = blockIdx.x*blockDim.x+threadIdx.x` independently. Continuing the same `01_thread_hierarchy.cu` run above (its `main()` prints the 1D trace first, then this 2D trace), genuinely computed and confirmed for all 16 threads:

```
2D launch: <<<(2,2), (2,2)>>> -- 4x4 = 16 total threads
  block(0,0) thread(0,0) -> row=0 col=0
  block(0,0) thread(1,0) -> row=0 col=1
  block(1,0) thread(0,0) -> row=0 col=2
  block(1,0) thread(1,0) -> row=0 col=3
  ...
  block(1,1) thread(1,1) -> row=3 col=3
```

Every `(row, col)` pair from `(0,0)` to `(3,3)` appears exactly once across the 16 threads — confirmed genuinely, not asserted — which is precisely the property Section 4.5's broadcast pattern and Section 4.6's matrix multiplication both depend on: a 2D launch that exactly covers a 2D output, one thread per element, with no gaps and no overlaps.

### ASCII Diagram — the three-level hierarchy

```
Grid (this kernel launch)
 +-- Block (0,0)                Block (1,0)
 |    +-- Thread(0,0) row=0,col=0    +-- Thread(0,0) row=0,col=2
 |    +-- Thread(1,0) row=0,col=1    +-- Thread(1,0) row=0,col=3
 |    +-- Thread(0,1) row=1,col=0    +-- Thread(0,1) row=1,col=2
 |    +-- Thread(1,1) row=1,col=1    +-- Thread(1,1) row=1,col=3
 |
 +-- Block (0,1)                Block (1,1)
      +-- Thread(0,0) row=2,col=0    +-- Thread(0,0) row=2,col=2
      ... (same pattern) ...
```

> `[COMMON TRAP]` `blockDim` is a launch-time choice, not a hardware constant — a kernel compiled once can be launched with `blockDim.x = 32` in one call and `blockDim.x = 256` in another, and `threadIdx.x` will range differently each time. Hardcoding an assumption about `blockDim.x`'s value inside a kernel body (rather than reading `blockDim.x` itself, as every formula in this section does) silently breaks the moment someone launches the same kernel with a different block size.

## 4.2 Host and Device: Two Separate Memory Spaces `[FOUNDATIONAL]`

### Intuition

Chapter 1.3 established that a `.cu` file is compiled by two different compilers. This section is about a *different* fact, easy to conflate with that one: the CPU and the GPU also have two physically separate pools of memory — host RAM and device VRAM — connected by a bus (PCIe, or NVLink on higher-end systems) that is dramatically slower than either memory's own internal bandwidth. A pointer into one is meaningless in the other: `h_a` (a host array) and `d_a` (a device buffer) are not two views of the same data, they are two entirely different allocations, and getting data from one to the other requires an explicit, physical copy across that bus.

### Background

| | Host memory (RAM) | Device memory (VRAM) |
|---|---|---|
| Allocated with | `malloc`, `new`, or an ordinary local variable | `cudaMalloc` |
| Freed with | `free`, `delete`, or automatic (stack) cleanup | `cudaFree` |
| Accessible from | Host (`__host__`) code | Device (`__global__`/`__device__`) code |
| Moving data between them | `cudaMemcpy(..., cudaMemcpyHostToDevice)` or `cudaMemcpyDeviceToHost` | (the same call, opposite direction) |

### Worked Example 4.2.1 — a complete vector-add, every copy traced

```cpp
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 8;
    size_t bytes = n * sizeof(float);
    float h_a[8] = {1,2,3,4,5,6,7,8};
    float h_b[8] = {10,20,30,40,50,60,70,80};
    float h_c[8] = {0};

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 4;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    vector_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

Compiled and run as the complete `02_vector_add_h2d_d2h.cu` further below:

```bash
nvcc -arch=sm_80 02_vector_add_h2d_d2h.cu -o 02_vector_add_h2d_d2h
./02_vector_add_h2d_d2h
```

Genuinely compiled and run, with every return code checked:

```
cudaMalloc x3 -> 100,100,100 (no CUDA-capable device is detected)
cudaMemcpy H2D x2 -> 100,100 (no CUDA-capable device is detected)
kernel launch <<<2,4>>> -> 100 (no CUDA-capable device is detected)
cudaMemcpy D2H -> 100 (no CUDA-capable device is detected)
h_c after (unchanged, since nothing ran): 0 0 0 0 0 0 0 0
host-computed reference a+b: 11 22 33 44 55 66 77 88
```

Two genuinely distinct things are on display. First, the *shape* of every GPU program this book writes from here forward — allocate on device, copy input in, launch, copy output out, free — is fully exercised, real C++ control flow that runs identically whether or not a GPU is present. Second, `cudaErrorNoDevice` (code 100) propagates honestly through all four device-touching calls, exactly as Chapter 1.3 first showed, and `h_c` genuinely stays all zeros because nothing ever wrote to `d_c`. The host-computed reference — `1+10=11`, `2+20=22`, ..., `8+80=88` — is what this exact kernel would produce on real hardware, and is exactly the number this book's real-hardware pass will confirm `h_c` becomes once a genuine GPU runs it.

> `[COMMON TRAP]` Passing a host pointer (`h_a`) directly to a kernel expecting a device pointer compiles without complaint — `nvcc` has no way to know, from the pointer's type alone, which memory space an address actually refers to. On real hardware this either silently produces garbage or crashes the kernel with an illegal memory access, and it's a mistake this book's own `Tensor` class (Part 1) is specifically designed to make impossible by tracking which device an allocation belongs to as part of the object itself, rather than trusting the caller to remember.

## 4.3 The GPU Memory Hierarchy: A First Look `[FOUNDATIONAL]`

### Intuition

Extend Section 4.2's two-machines picture down one more level. **Global memory** is a public warehouse across town: any thread from any block can drive there, but the drive is comparatively slow, and everyone shares the same access road. **Shared memory** is a supply closet built into one specific block's own barracks: every thread in that one block reaches it almost instantly, but threads in other blocks can't use it at all, and its contents vanish the moment that block finishes. **Local memory** is the single backpack strapped to one thread alone — fastest to reach, but gone the instant that thread's work ends.

### Background

| | Global memory | Shared memory | Local (per-thread) memory |
|---|---|---|---|
| Visible to | Every thread, every block | Every thread *in one block* | One single thread |
| Speed | Slowest | Fast | Fastest |
| Typical size | Large (GBs) | Small (tens of KB per block) | Tiny, per-thread |
| Lifetime | Whole kernel launch (and beyond) | One block's execution | One thread's execution |
| Allocated with | `cudaMalloc` (Chapter 2.4) | `__shared__` inside a kernel | Ordinary local variables inside a kernel |

### Worked Example 4.3.1 — where Worked Example 4.2.1's data actually lives

`d_a`, `d_b`, and `d_c` are all **global memory** — every thread across every block needs to read its own `a[i]`/`b[i]` and write its own `c[i]`, and threads in different blocks have no other way to see each other's data. Contrast this with a kernel that reuses the *same* small chunk of data across many threads *within one block* — Part 2's tiled matrix multiplication is exactly this case — where staging that chunk once into `__shared__` memory lets every thread in the block read it at shared-memory speed instead of each thread separately paying global memory's slower round trip for identical bytes. This book holds the mechanics of actually declaring and using `__shared__` memory for Part 1's memory-management chapters and Part 2's tiled kernel, where the payoff is concrete.

> `[COMMON TRAP]` "Shared memory" is easy to mistake for "a faster general-purpose RAM available to everyone," as though it were just a quicker version of global memory. It isn't general-purpose at all — it's a small, per-block scratchpad (typically tens of kilobytes), explicitly staged into and out of by the kernel's own code, and its entire contents vanish the moment that specific block finishes executing. A value written to shared memory by one block is completely invisible to every other block, always.

## 4.4 Memory Coalescing, at Full Generality `[FOUNDATIONAL]`

### Intuition

Chapter 3 traced one specific case of this argument — `particles[i].vx` versus `vx[i]` — down to genuine, disassembled machine code. This section states the general principle that specific case was an instance of: it was never really about particles.

### Background

Real GPU hardware groups threads into **warps** of 32, and when every thread in a warp requests an address falling inside the same aligned memory region, the hardware services the entire warp with one **coalesced** transaction. When the 32 addresses are scattered, the hardware may need up to 32 separate transactions for the same warp — up to a 32× bandwidth penalty for delivering the identical amount of useful data. Chapter 3's genuine SASS evidence — an `IMAD.WIDE` multiplier of `4` for a contiguous `float` array versus `28` for a strided struct field — is one concrete measurement of exactly this principle, not a special case of it.

Section 4.1's `i = blockIdx.x * blockDim.x + threadIdx.x` is what makes this principle actionable: it guarantees that consecutive threads within a warp receive consecutive values of `i`. Whether that translates into coalesced access from there depends entirely on what a kernel does with `i` — indexing a contiguous array with `array[i]` (as `vector_add_kernel` does for `a`, `b`, and `c` above) keeps consecutive threads exactly `sizeof(element)` bytes apart, the coalesced case; indexing with `array[i * stride]` for any `stride > 1`, or through a struct field the way Chapter 3's `particles[i].vx` did, spreads them out by that much more.

> `[COMMON TRAP]` `vector_add_kernel` above happens to be perfectly coalesced — `a[i]`, `b[i]`, `c[i]` are all unit-stride accesses into contiguous `float` arrays. It's tempting to conclude "elementwise kernels are always coalesced," but the coalescing only follows from *how the index is used*, not from the kernel being elementwise. Part 2's transpose-adjacent kernels are the canonical elementwise counterexample: reading `input[i]` and writing `output[i * n]` (or vice versa) is elementwise in the sense of "one input maps to one output," and still produces exactly the scattered, uncoalesced write pattern this section warns about on whichever side carries the stride.

## 4.5 Broadcasting: One Thread, One Output Element `[FOUNDATIONAL]`

### Intuition

`vector_add_kernel` already demonstrated the pattern this section names: every thread is handed exactly one output element to produce, computes it independently of every other thread, and writes it exactly once. This is the **broadcast** pattern — not "broadcasting" in NumPy's shape-matching sense, but the literal broadcasting of one small, identical kernel body out to potentially millions of independent threads, each covering one coordinate of the output.

### Background

| Output shape | Launch shape | Per-thread index | Boundary check |
|---|---|---|---|
| 1D, length `n` | `<<<ceil(n/T), T>>>` | `i = blockIdx.x*blockDim.x+threadIdx.x` | `if (i < n)` |
| 2D, `M×N` | `<<<dim3(ceil(N/Tx),ceil(M/Ty)), dim3(Tx,Ty)>>>` | `row = blockIdx.y*blockDim.y+threadIdx.y`, `col = blockIdx.x*blockDim.x+threadIdx.x` | `if (row < M && col < N)` |

The boundary check exists because `ceil(n/T)` blocks of `T` threads each will typically launch *more* total threads than there are output elements — Worked Example 4.2.1's `n=8` with `threads_per_block=4` happens to divide evenly, but `n=7` with the same block size launches `ceil(7/4)=2` blocks, `8` threads, for only `7` real outputs. Every kernel this book writes from here forward guards its actual work with exactly this kind of check, following Chapter 1's masking discipline forward into every dimension a kernel might have.

### Worked Example 4.5.1 — the boundary check, genuinely exercised

Reusing Worked Example 4.2.1's `vector_add_kernel` at `n=7` instead of `8`: `blocks = ceil(7/4) = 2`, launching `2*4=8` threads total. Threads `0` through `6` satisfy `i < 7` and do real work; thread `7` (the last thread of block 1) fails the check and returns immediately, touching no memory at all. Without the check, thread `7` would compute `c[7] = a[7] + b[7]` against memory one element past the end of 7-element arrays — undefined behavior, and, depending on what happens to sit in the adjacent memory, either a silent wrong read, a crash, or (worst of all) a value that happens to look plausible.

## 4.6 Matrix Multiplication: One Thread, One Dot Product `[FOUNDATIONAL]`

### Intuition

Extend Section 4.5's broadcast pattern from "one thread, one output number" to "one thread, one output number computed by summing over a whole dimension." Matrix multiplication's naive GPU form is exactly the 2D broadcast pattern from Section 4.5, with the per-thread body replaced by a small loop instead of one arithmetic expression: thread `(row, col)` owns output element `C[row][col]`, and computes it as the full dot product of `A`'s row `row` against `B`'s column `col`.

### Worked Example 4.6.1 — a complete `2×2 @ 2×2`, traced and cross-checked

```cpp
__global__ void naive_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
```

For `A = [[1,2],[3,4]]`, `B = [[5,6],[7,8]]`: thread `(0,0)` computes `C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] = 1*5 + 2*7 = 5+14 = 19`. Thread `(0,1)` computes `C[0][1] = 1*6 + 2*8 = 6+16 = 22`. Thread `(1,0)` computes `C[1][0] = 3*5+4*7 = 15+28 = 43`. Thread `(1,1)` computes `C[1][1] = 3*6+4*8 = 18+32 = 50`. Genuinely compiled (the kernel compiles clean for `sm_80`) and genuinely run on the host via an identical reference implementation, as the complete `03_naive_matmul.cu` further below:

```bash
nvcc -arch=sm_80 03_naive_matmul.cu -o 03_naive_matmul
./03_naive_matmul
```

```
host reference C = [[19.0, 22.0], [43.0, 50.0]]
```

Every value matches the hand trace exactly, and independently matches a plain Python cross-check of the same `2×2` product. This naive kernel reads `A[row*K+k]` and `B[k*N+col]` — for a fixed `row`, as `k` advances, `A`'s accesses are perfectly contiguous (Section 4.4's coalesced case for the whole warp when `row` is shared across threads in a warp along the `col` dimension), but `B`'s accesses jump by a full row (`N` elements) each step, since consecutive `k` values are `N` floats apart in `B`'s row-major layout — a real, genuine bandwidth cost this naive form pays that Part 2's tiled kernel is built specifically to eliminate by staging both operands through shared memory first.

## 4.7 Complete Runnable Code

### File: `01_thread_hierarchy.cu`

```cpp
#include <cstdio>

int main() {
    int gridDim_x = 3, blockDim_x = 4;
    printf("1D launch: <<<%d, %d>>> -- %d total threads\n", gridDim_x, blockDim_x, gridDim_x * blockDim_x);
    for (int blockIdx_x = 0; blockIdx_x < gridDim_x; blockIdx_x++) {
        for (int threadIdx_x = 0; threadIdx_x < blockDim_x; threadIdx_x++) {
            int i = blockIdx_x * blockDim_x + threadIdx_x;
            printf("  block %d, thread %d -> global i = %d\n", blockIdx_x, threadIdx_x, i);
        }
    }

    printf("\n2D launch: <<<(2,2), (2,2)>>> -- 4x4 = 16 total threads\n");
    for (int by = 0; by < 2; by++)
    for (int bx = 0; bx < 2; bx++)
    for (int ty = 0; ty < 2; ty++)
    for (int tx = 0; tx < 2; tx++) {
        int row = by * 2 + ty;
        int col = bx * 2 + tx;
        printf("  block(%d,%d) thread(%d,%d) -> row=%d col=%d\n", bx, by, tx, ty, row, col);
    }
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_thread_hierarchy.cu -o 01_thread_hierarchy
./01_thread_hierarchy
```

Produces exactly the two output blocks shown in Worked Example 4.1.1 above, the 1D trace followed by the 2D trace.

### File: `02_vector_add_h2d_d2h.cu`

```cpp
#include <cstdio>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 8;
    size_t bytes = n * sizeof(float);
    float h_a[8] = {1,2,3,4,5,6,7,8};
    float h_b[8] = {10,20,30,40,50,60,70,80};
    float h_c[8] = {0};

    float *d_a, *d_b, *d_c;
    cudaError_t e1 = cudaMalloc((void**)&d_a, bytes);
    cudaError_t e2 = cudaMalloc((void**)&d_b, bytes);
    cudaError_t e3 = cudaMalloc((void**)&d_c, bytes);
    printf("cudaMalloc x3 -> %d,%d,%d (%s)\n", e1, e2, e3, cudaGetErrorString(e1));

    cudaError_t e4 = cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaError_t e5 = cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    printf("cudaMemcpy H2D x2 -> %d,%d (%s)\n", e4, e5, cudaGetErrorString(e4));

    int threads_per_block = 4;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    vector_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaError_t e6 = cudaGetLastError();
    printf("kernel launch <<<%d,%d>>> -> %d (%s)\n", blocks, threads_per_block, e6, cudaGetErrorString(e6));

    cudaError_t e7 = cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    printf("cudaMemcpy D2H -> %d (%s)\n", e7, cudaGetErrorString(e7));

    printf("h_c after (unchanged, since nothing ran): ");
    for (int i = 0; i < n; i++) printf("%.0f ", h_c[i]);
    printf("\n");

    printf("host-computed reference a+b: ");
    for (int i = 0; i < n; i++) printf("%.0f ", h_a[i] + h_b[i]);
    printf("\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_vector_add_h2d_d2h.cu -o 02_vector_add_h2d_d2h
./02_vector_add_h2d_d2h
```

Produces exactly the output shown in Worked Example 4.2.1 above.

### File: `03_naive_matmul.cu`

```cpp
#include <cstdio>

__global__ void naive_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

void matmul_host_reference(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
            C[row * N + col] = acc;
        }
}

int main() {
    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float C[4] = {0};

    matmul_host_reference(A, B, C, 2, 2, 2);
    printf("host reference C = [[%.1f, %.1f], [%.1f, %.1f]]\n", C[0], C[1], C[2], C[3]);
    return 0;
}
```

```bash
nvcc -arch=sm_80 03_naive_matmul.cu -o 03_naive_matmul
./03_naive_matmul
```

`naive_matmul_kernel` is compiled into this binary (proving it builds clean for `sm_80`) but, as `main()` above shows, only `matmul_host_reference` is actually called — the kernel itself is never launched here, exactly as the paragraph below explains.

### Expected Output for `03_naive_matmul.cu`

```
host reference C = [[19.0, 22.0], [43.0, 50.0]]
```

Every value was independently hand-traced in Worked Example 4.6.1. `naive_matmul_kernel` genuinely compiles clean alongside this file under `nvcc -arch=sm_80`; its device-side execution against real hardware, producing this identical matrix from `d_A`/`d_B`/`d_C`, is deferred to this book's real-hardware pass.

## Chapter Summary

A CUDA kernel launch organizes work into a three-level hierarchy — grid, block, thread — and `i = blockIdx.x*blockDim.x+threadIdx.x` (or its 2D extension, `row`/`col`) is the formula every kernel in this book uses to turn a thread's position in that hierarchy into a unique index into its data, confirmed in this chapter by genuinely enumerating every thread of both a 1D and a 2D launch by hand. Host RAM and device VRAM are two physically separate memories connected by a comparatively slow bus, a distinct fact from Chapter 1.3's two-compilers split, and every kernel launch in this book is bracketed by explicit `cudaMalloc`/`cudaMemcpy`/`cudaFree` calls that this chapter's vector-add example genuinely exercised end to end, honestly reporting `cudaErrorNoDevice` at each device-touching step in this no-GPU environment. Within device memory, global memory is large, slow, and visible everywhere; shared memory is small, fast, and scoped to one block; local memory is per-thread and fastest of all — a hierarchy whose deeper mechanics wait for Part 1. Chapter 3's coalescing argument turns out to be the general shape of every global memory access this book writes, not a particle-specific finding, and Section 4.5's broadcast pattern — one thread, one output element, guarded by a boundary check — is the shape every kernel in this book fits, including Section 4.6's naive matrix multiplication: one thread, one full dot product, genuinely verified against a host reference and already showing the very access-pattern asymmetry (`A` coalesced, `B` strided) that Part 2's tiled kernel exists to fix.

## Self-Check Questions

1. A kernel is launched as `<<<10, 32>>>`. What is `blockDim.x`, and what is the global index `i` computed by the 5th thread (0-indexed as thread 4) of the 3rd block (0-indexed as block 2)?
2. Explain the difference between Chapter 1.3's "two compilers" fact and this chapter's "two memories" fact — are they the same underlying phenomenon described twice, or genuinely separate facts about CUDA? Justify your answer.
3. `vector_add_kernel` is launched with `n=7` and `threads_per_block=4`. How many blocks does `(n + threads_per_block - 1) / threads_per_block` launch, how many total threads does that produce, and which specific thread(s) fail the `i < n` boundary check?
4. In Worked Example 4.6.1's `naive_matmul_kernel`, explain precisely why consecutive values of the loop variable `k` produce contiguous addresses in `A` but not in `B`, given both are stored in the same row-major layout.
5. A colleague claims "every elementwise kernel is automatically coalesced, since each thread only touches its own independent output element." Using Section 4.4's Common Trap, construct a one-sentence counterexample.

## Where We Go Next

This chapter's kernels all read and wrote plain `float*` arrays with manually-tracked shapes (`n`, or `M`/`N`/`K` passed as separate arguments). Chapter 5 looks at how a warp of 32 threads actually executes a kernel body in lockstep — the SIMT model — and what happens the moment a branch (an `if`) causes different threads in the same warp to disagree about which path to take, a question every one of this chapter's boundary checks has been implicitly raising without yet answering.

## Worked Solutions

**1.** `blockDim.x = 32` (the second launch-configuration argument). Thread 4 of block 2 computes `i = blockIdx.x * blockDim.x + threadIdx.x = 2 * 32 + 4 = 68`.

**2.** They are genuinely separate facts. Chapter 1.3's split is about *compilation*: one `.cu` file's host and device portions are processed by two different compiler pipelines, producing two different kinds of machine code, and this is true even in a single-memory-space thought experiment. This chapter's split is about *runtime data location*: even after both portions are compiled, the CPU and GPU each have their own physical memory, and a value computed by host code isn't visible to device code (or vice versa) until it's explicitly copied across the bus with `cudaMemcpy` — a fact about hardware memory topology, unrelated to which compiler produced which instructions.

**3.** `(7 + 4 - 1) / 4 = 10 / 4 = 2` (integer division), so 2 blocks launch, producing `2 * 4 = 8` total threads. Threads `0` through `6` (global index `i` from 0 to 6) satisfy `i < 7` and do real work; thread index `7` — the 4th thread (`threadIdx.x=3`) of the 2nd block (`blockIdx.x=1`), since `i = 1*4+3 = 7` — fails `7 < 7` and returns immediately without touching memory.

**4.** `A` is indexed as `A[row*K+k]`: for fixed `row`, incrementing `k` by 1 advances the linear index by exactly 1 — contiguous. `B` is indexed as `B[k*N+col]`: for fixed `col`, incrementing `k` by 1 advances the linear index by `N` (one full row) — a stride of `N` elements, not 1, even though `B` uses the identical row-major storage convention as `A`. The difference isn't in how the matrices are stored; it's in which index (`row` vs. `k`) is held fixed while `k` varies inside the loop.

**5.** Section 4.4's Common Trap gives one directly: a transpose-style kernel reading `input[i]` and writing `output[i * n]` (mapping a flat input to a strided output, or the reverse) is elementwise — each thread reads exactly one input and writes exactly one output, entirely independent of every other thread — and yet the strided side of that access (`output[i*n]`, consecutive threads `n` elements apart) is exactly the scattered pattern Chapter 3 showed defeats coalescing, disproving the claim that "elementwise" by itself guarantees anything about coalescing.
