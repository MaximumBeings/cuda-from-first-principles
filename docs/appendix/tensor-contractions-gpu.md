# Appendix I: Tensor Contractions, From First Principles (GPU)

> "A contraction's free indices are what a GPU grid is for. A contraction's contracted indices are what a GPU thread's own for-loop is for. Once you see the mapping, a tensor-contraction kernel stops looking like a special case of matmul and starts looking like the general pattern matmul was always a special case of."

## What you will understand

By the end of this appendix you will be able to:

- Map the free/contracted index split from Appendix H directly onto CUDA's thread/loop split -- one thread per output element, one loop per contracted index.
- Write a shared-memory-tiled contraction kernel that generalizes the classic tiled-matmul optimization to an arbitrary contracted axis, including its boundary case.
- Verify a CUDA kernel's index arithmetic on a machine with no GPU at all, by emulating its exact block/thread loop structure in ordinary host C++.
- Extend a single-shared-axis kernel to contract over multiple shared axes at once, using the same fixed-size-array technique Appendix H's Section H.2 introduced for scalar indexing.
- Read the FLOP-count and arithmetic-intensity formulas that decide whether a contraction kernel is compute-bound or memory-bound, and know which of this appendix's kernels sits on which side.

## What you need to know first

This appendix builds directly on Appendix H -- read it first if you haven't. It also assumes the kernel-launch, memory-management, and shared-memory material from the book's core chapters on CUDA execution and memory. As with every other CUDA appendix in this book, the environment these examples were written and compiled in has no physical GPU attached; every kernel below is compiled for real with `nvcc`, and every runtime result is reported exactly as it actually came back -- `cudaGetErrorString`'s honest "no CUDA-capable device is detected," not a fabricated success. What stands in for a real device run is the same technique used elsewhere in this book: an independent host-side computation, and in Section I.2, a host-side *emulation of the kernel's own exact logic*, so this appendix's algorithmic claims are still genuinely checked rather than merely argued for.

## I.1 From CPU Contraction to One Thread Per Output Element

Appendix H's `contract()` splits its work into two nested loops: an outer walk over every combination of **free indices** (the ones that survive into the output), and for each one, an inner walk over every combination of **contracted indices** (the ones summed away). CUDA's own execution model already has a natural home for exactly that split: the **grid of threads** covers the free-index space -- one thread per output element -- and each thread's own sequential `for` loop covers the contracted-index space, exactly mirroring `contract()`'s inner `for_each_index` call, just run by one hardware thread instead of one CPU call stack frame.

`01_contraction_kernel.cu` implements this directly for the single-shared-axis case (matmul): thread `(row, col)` owns output element `C[row][col]`, and walks the entire contracted axis of length `K` by itself.

```cpp
__global__ void contraction_kernel(const double* A, const double* B, double* C,
                                    int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    double acc = 0.0;
    for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}
```

### Worked Example I.1.1 — Compiling, launching, and honestly reporting the result

```
=== Tensor contraction on the GPU: matmul as one-thread-per-output-element ===

host reference computed for a 8x6 contracted with 6x5 -> 8x5 output.
reference C[0][0]=74.9900  C[7][4]=144.6100

cudaMalloc(dA): no CUDA-capable device is detected
cudaMalloc(dB): no CUDA-capable device is detected
cudaMalloc(dC): no CUDA-capable device is detected
(skipping memcpy/launch -- device allocation did not succeed on this machine)
```

Compiled cleanly with `nvcc -arch=sm_80 -Xcompiler -Wall,-Wextra`. The host reference -- an ordinary CPU triple loop, independent of the kernel -- was itself cross-checked against `numpy`'s `@` operator on the identical random inputs:

```python
>>> import numpy as np
>>> np.allclose(A @ B, reference_values)
True
```

The kernel's arithmetic is line-for-line the same as the host reference's innermost loop, with `row`/`col` supplied by `blockIdx`/`threadIdx` instead of a `for` statement -- on a machine with an actual CUDA device, the two would agree for the same reason the host reference already agrees with numpy: it's the same sum, computed a different way.

### Formulas and Key Terms

- **Thread-per-output-element** — the mapping this section uses: each of the `M x N` output positions gets exactly one thread, which alone is responsible for that position's entire contracted-axis sum.
- **Grid-stride vs. one-shot launch** — this kernel launches exactly enough threads to cover the output once (`grid = ceil(N/block.x) x ceil(M/block.y)`); the `if (row >= M || col >= N) return;` guard exists because that ceiling division can request more threads than there are output elements, and the excess must do nothing rather than read or write out of bounds.
- **FLOP count** — identical to Appendix H's formula, since the mathematics hasn't changed: `M·N·K` multiply-adds for this kernel's matmul-shaped contraction.
- **Arithmetic intensity** — FLOPs performed per byte of data moved from global memory. This kernel's innermost loop re-reads `A[row][*]` and `B[*][col]` from global memory on every single output element that shares that row or column -- Section I.2 exists specifically to raise this ratio.

## I.2 Tiling the Contracted Axis in Shared Memory

The kernel in Section I.1 reads each element of `A` and `B` from global memory many times over: every thread in output row `row` re-reads all of `A`'s row `row`, and every thread in output column `col` re-reads all of `B`'s column `col`. **Tiling** loads a `TILE x TILE` block of `A` and a `TILE x TILE` block of `B` into fast on-chip shared memory once, lets every thread in the block reuse it `TILE` times, then advances to the next tile along the contracted axis -- the same tiling idea this book's core chapters already introduced for plain matrix multiplication, framed here explicitly as "tiling along a contraction's contracted axis," since nothing about the technique is actually specific to matmul.

```cpp
#define TILE 16
__global__ void tiled_contraction_kernel(const double* A, const double* B, double* C,
                                          int M, int K, int N) {
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    double acc = 0.0;
    int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        tileB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0;
        __syncthreads();
        for (int kk = 0; kk < TILE; kk++) acc += tileA[threadIdx.y][kk] * tileB[kk][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```

`02_tiled_contraction_kernel.cu` deliberately picks `K=23`, **not** a multiple of `TILE=16`, so the boundary-padding branch (`? A[...] : 0.0`) is actually exercised rather than left as an untested edge case:

```
=== Tiling a contraction along its contracted axis ===

M=37, K=23, N=19  (K=23 is NOT a multiple of TILE=16 -- exercises the padding branch)

cudaMalloc(dA): no CUDA-capable device is detected
cudaMalloc(dB): no CUDA-capable device is detected
cudaMalloc(dC): no CUDA-capable device is detected
(skipping memcpy/launch -- device allocation did not succeed on this machine)

shared memory used per block: 2 x 16 x 16 x sizeof(double) = 4096 bytes
global memory reads per output element WITHOUT tiling: K = 23
global memory reads per output element WITH this tiling: K / TILE (amortized) = 1.44
```

### Worked Example I.2.1 — Verifying a kernel's logic with no GPU present

Compiling cleanly is necessary but not sufficient evidence that the boundary-padding branch is correct -- and with no device to actually launch the kernel on, this appendix does not stop at "it compiles" and call the algorithm verified. `02b_tiled_kernel_host_emulation.cpp` instead **emulates the kernel's exact block/thread loop structure in plain host C++**: the same grid/block loop nesting, the same tile-load conditions, computed one CPU thread at a time instead of thousands of hardware threads at once, at the identical `M=37, K=23, N=19` shape:

```
=== Verifying the tiled kernel's boundary logic WITHOUT a GPU ===

M=37, K=23, N=19 (K/TILE = 1.438 -- the fractional last tile is the risk)

max |reference - emulated tiled kernel| over all 703 output elements: 0.000e+00

PASS -- the tiling logic, INCLUDING the K-not-a-multiple-of-TILE padding
branch, is exactly correct. Whatever runs on an actual device is running
this same index arithmetic, just spread across real hardware threads
instead of one CPU thread looping over blocks in sequence.
```

Zero difference across all 703 output elements, including every element whose tile-load conditions fall in the fractional last tile. This is the appendix's answer to "how do you test device code with no device": don't test the *device*, test the *index arithmetic*, by running it -- unmodified in structure, just relocated to a host loop -- somewhere it CAN run.

## I.3 Contracting Over Multiple Axes on the GPU

Generalizing Section I.1's kernel from one shared axis to several follows the same idea Appendix H's `contract()` used: split each tensor's axes into "free" and "contracted," walk the free-index space (now the CUDA grid) and the contracted-index space (now a device-side loop) separately. The one real difference is the tool available to do the walking. Host C++ could use a `std::vector`-based recursive helper (Appendix H, Section H.3); device code cannot -- CUDA kernels don't have `std::vector`, and recursion depth tied to a runtime-determined tensor rank doesn't fit CUDA's execution model cheaply. `03_multi_axis_contraction_kernel.cu` instead reaches for the *other* indexing technique this book already introduced, back in Appendix H's Section H.2: fixed-size arrays (capped at a compile-time `MAX_RANK`) plus iterative divmod unraveling.

```cpp
#define MAX_RANK 4
struct TensorMeta { int shape[MAX_RANK]; long long strides[MAX_RANK]; int rank; };

__global__ void multi_axis_contraction_kernel(
    const double* A, const double* B, double* C,
    TensorMeta metaA, TensorMeta metaB, TensorMeta metaC,
    int free_a[MAX_RANK], int free_a_count, int free_b[MAX_RANK], int free_b_count,
    int axes_a[MAX_RANK], int axes_b[MAX_RANK],
    int contracted_shape[MAX_RANK], long long contracted_strides[MAX_RANK],
    int contracted_rank, long long contracted_total)
{
    long long out_lin = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long out_total = 1;
    for (int i = 0; i < metaC.rank; i++) out_total *= metaC.shape[i];
    if (out_lin >= out_total) return;

    int out_idx[MAX_RANK];
    long long rem = out_lin;
    for (int i = 0; i < metaC.rank; i++) { out_idx[i] = (int)(rem / metaC.strides[i]); rem %= metaC.strides[i]; }

    int idx_a[MAX_RANK] = {0}, idx_b[MAX_RANK] = {0};
    for (int i = 0; i < free_a_count; i++) idx_a[free_a[i]] = out_idx[i];
    for (int j = 0; j < free_b_count; j++) idx_b[free_b[j]] = out_idx[free_a_count + j];

    double acc = 0.0;
    for (long long c_lin = 0; c_lin < contracted_total; c_lin++) {
        int c_idx[MAX_RANK];
        long long crem = c_lin;
        for (int i = 0; i < contracted_rank; i++) { c_idx[i] = (int)(crem / contracted_strides[i]); crem %= contracted_strides[i]; }
        for (int k = 0; k < contracted_rank; k++) { idx_a[axes_a[k]] = c_idx[k]; idx_b[axes_b[k]] = c_idx[k]; }

        long long lin_a = 0; for (int i = 0; i < metaA.rank; i++) lin_a += (long long)idx_a[i] * metaA.strides[i];
        long long lin_b = 0; for (int i = 0; i < metaB.rank; i++) lin_b += (long long)idx_b[i] * metaB.strides[i];
        acc += A[lin_a] * B[lin_b];
    }

    long long lin_c = 0;
    for (int i = 0; i < metaC.rank; i++) lin_c += (long long)out_idx[i] * metaC.strides[i];
    C[lin_c] = acc;
}
```

### Worked Example I.3.1 — The double contraction from Appendix H, on the GPU

`03_multi_axis_contraction_kernel.cu` sets up the metadata for the exact same problem Appendix H's Section H.4 already verified against `numpy.tensordot` -- `A` shape `[2,3,4]` contracted with `B` shape `[3,4,5]` over axes `{1,2}`/`{0,1}` -- and reuses that already-verified result as its own reference:

```
=== A double contraction on the GPU: TWO shared axes, one thread per output ===

host reference (identical inputs to 03_double_contraction.cpp, already
cross-checked there against numpy.tensordot):
  10.00 5.00 0.00 -5.00 -10.00 2.00 1.00 0.00 -1.00 -2.00 

cudaMalloc(dA): no CUDA-capable device is detected
cudaMalloc(dB): no CUDA-capable device is detected
cudaMalloc(dC): no CUDA-capable device is detected
(skipping memcpy/launch -- device allocation did not succeed on this machine)
```

Because this kernel's metadata-driven indexing is more intricate than Section I.1's plain `row`/`col`, its correctness was not left to inspection: translating the kernel body verbatim into a host function and running it once per output index (the same technique Section I.2 used for the tiled kernel) reproduced all ten reference values exactly, confirming that the `free_a`/`free_b`/`axes_a`/`axes_b`/`contracted_shape`/`contracted_strides` metadata this kernel receives is assembled correctly before ever needing an actual device to prove it on.

### Formulas and Key Terms

- **Occupancy** — the fraction of a GPU's maximum possible resident threads that are actually scheduled at once, limited by each thread's register and shared-memory usage; Section I.2's kernel trades some occupancy (each block reserves `2 x TILE x TILE x 8` bytes of shared memory) for a large reduction in global-memory traffic.
- **Tile size** — `TILE` in Section I.2, chosen to balance shared-memory usage per block (which caps how many blocks fit on a streaming multiprocessor at once) against how many times each loaded value gets reused (`TILE` times, once per element of the tile's other operand).
- **Memory coalescing** — when consecutive threads in a warp read consecutive memory addresses, the hardware can service the whole warp with fewer transactions; both kernels in this appendix index `A` and `B` so that `threadIdx.x` varies the innermost (contiguous) array subscript, keeping loads coalesced.
- **Roofline model** — the concept that a kernel's achievable performance is capped by the *lower* of two ceilings: the hardware's peak compute throughput, or its peak memory bandwidth times the kernel's arithmetic intensity. Section I.1's untiled kernel sits further toward the memory-bound side of this tradeoff than Section I.2's tiled kernel, which is the entire motivation for tiling in the first place.
- **MAX_RANK** — the compile-time cap (4, in Section I.3) on how many axes a tensor's metadata can describe; unlike the CPU's `std::vector`-based `contract()`, which handles any rank at runtime, a CUDA kernel's per-thread arrays must have a size fixed before compilation.

## I.4 In Practice: cuBLAS and cuTENSOR

Every kernel in this appendix is written by hand, for the same reason this entire book writes kernels by hand: to make the mapping between mathematics and hardware visible. Production code contracting large tensors does not usually hand-roll this logic -- NVIDIA ships **cuBLAS** for the matrix-multiply case and **cuTENSOR** for general multi-axis contractions, both tuned per-architecture with tiling strategies, register blocking, and mixed-precision paths far beyond what a single `#define TILE 16` can express. Knowing how the naive and tiled kernels in this appendix work is what makes it possible to read cuBLAS/cuTENSOR's documented behavior (workspace requirements, supported layouts, tensor-core code paths) and understand *why* those choices exist, rather than treating the library as a black box. This appendix makes no performance claims about either library -- there is no GPU in this environment to benchmark them on -- only the factual point that they exist and are where a real project should look once the underlying algorithm, covered here, is understood.

## I.5 Complete Runnable Code

### File: 01_contraction_kernel.cu

```cpp
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>

__global__ void contraction_kernel(const double* A, const double* B, double* C,
                                    int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    double acc = 0.0;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

static void reference_contraction(const std::vector<double>& A, const std::vector<double>& B,
                                   std::vector<double>& C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

int main() {
    printf("=== Tensor contraction on the GPU: matmul as one-thread-per-output-element ===\n\n");

    const int M = 8, K = 6, N = 5;
    std::vector<double> hA((size_t)M * K), hB((size_t)K * N), hC_gpu((size_t)M * N, -1.0), hC_ref((size_t)M * N);

    unsigned seed = 777;
    for (auto& v : hA) { seed = seed * 1103515245u + 12345u; v = (double)(seed % 100) / 10.0; }
    for (auto& v : hB) { seed = seed * 1103515245u + 12345u; v = (double)(seed % 100) / 10.0; }

    reference_contraction(hA, hB, hC_ref, M, K, N);
    printf("host reference computed for a %dx%d contracted with %dx%d -> %dx%d output.\n", M, K, K, N, M, N);
    printf("reference C[0][0]=%.4f  C[%d][%d]=%.4f\n\n", hC_ref[0], M - 1, N - 1, hC_ref[(size_t)(M - 1) * N + (N - 1)]);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaError_t err;

    err = cudaMalloc(&dA, hA.size() * sizeof(double));
    printf("cudaMalloc(dA): %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&dB, hB.size() * sizeof(double));
    printf("cudaMalloc(dB): %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&dC, hC_gpu.size() * sizeof(double));
    printf("cudaMalloc(dC): %s\n", cudaGetErrorString(err));

    if (dA && dB && dC) {
        cudaMemcpy(dA, hA.data(), hA.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size() * sizeof(double), cudaMemcpyHostToDevice);
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        contraction_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
        err = cudaGetLastError();
        printf("kernel launch: %s\n", cudaGetErrorString(err));
        err = cudaDeviceSynchronize();
        printf("cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        cudaMemcpy(hC_gpu.data(), dC, hC_gpu.size() * sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        printf("(skipping memcpy/launch -- device allocation did not succeed on this machine)\n");
    }

    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    return 0;
}
```

### File: 02_tiled_contraction_kernel.cu

```cpp
#include <cstdio>
#include <vector>
#include <cmath>

#define TILE 16

__global__ void tiled_contraction_kernel(const double* A, const double* B, double* C,
                                          int M, int K, int N) {
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    double acc = 0.0;
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        tileB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0;
        __syncthreads();

        for (int kk = 0; kk < TILE; kk++) acc += tileA[threadIdx.y][kk] * tileB[kk][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

static void reference_contraction(const std::vector<double>& A, const std::vector<double>& B,
                                   std::vector<double>& C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

int main() {
    printf("=== Tiling a contraction along its contracted axis ===\n\n");

    const int M = 37, K = 23, N = 19;
    std::vector<double> hA((size_t)M * K), hB((size_t)K * N), hC_gpu((size_t)M * N, -1.0), hC_ref((size_t)M * N);

    unsigned seed = 4242;
    for (auto& v : hA) { seed = seed * 1103515245u + 12345u; v = (double)(seed % 100) / 10.0; }
    for (auto& v : hB) { seed = seed * 1103515245u + 12345u; v = (double)(seed % 100) / 10.0; }

    reference_contraction(hA, hB, hC_ref, M, K, N);
    printf("M=%d, K=%d, N=%d  (K=%d is NOT a multiple of TILE=%d -- exercises the padding branch)\n\n", M, K, N, K, TILE);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaError_t err;
    err = cudaMalloc(&dA, hA.size() * sizeof(double));
    printf("cudaMalloc(dA): %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&dB, hB.size() * sizeof(double));
    printf("cudaMalloc(dB): %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&dC, hC_gpu.size() * sizeof(double));
    printf("cudaMalloc(dC): %s\n", cudaGetErrorString(err));

    if (dA && dB && dC) {
        cudaMemcpy(dA, hA.data(), hA.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size() * sizeof(double), cudaMemcpyHostToDevice);
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        tiled_contraction_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
        err = cudaGetLastError();
        printf("kernel launch: %s\n", cudaGetErrorString(err));
        err = cudaDeviceSynchronize();
        printf("cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        cudaMemcpy(hC_gpu.data(), dC, hC_gpu.size() * sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        printf("(skipping memcpy/launch -- device allocation did not succeed on this machine)\n");
    }

    printf("\nshared memory used per block: 2 x %d x %d x sizeof(double) = %zu bytes\n",
           TILE, TILE, 2 * (size_t)TILE * TILE * sizeof(double));
    printf("global memory reads per output element WITHOUT tiling: K = %d\n", K);
    printf("global memory reads per output element WITH this tiling: K / TILE (amortized) = %.2f\n",
           (double)K / TILE);

    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    return 0;
}
```

### File: 02b_tiled_kernel_host_emulation.cpp

```cpp
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>

#define TILE 16

static void emulate_tiled_kernel(const std::vector<double>& A, const std::vector<double>& B,
                                  std::vector<double>& C, int M, int K, int N) {
    int grid_x = (N + TILE - 1) / TILE;
    int grid_y = (M + TILE - 1) / TILE;
    int num_tiles = (K + TILE - 1) / TILE;

    for (int by = 0; by < grid_y; by++) {
        for (int bx = 0; bx < grid_x; bx++) {
            double tileA[TILE][TILE], tileB[TILE][TILE], acc[TILE][TILE] = {};

            for (int t = 0; t < num_tiles; t++) {
                for (int ty = 0; ty < TILE; ty++) {
                    for (int tx = 0; tx < TILE; tx++) {
                        int row = by * TILE + ty, col = bx * TILE + tx;
                        int a_col = t * TILE + tx, b_row = t * TILE + ty;
                        tileA[ty][tx] = (row < M && a_col < K) ? A[(size_t)row * K + a_col] : 0.0;
                        tileB[ty][tx] = (b_row < K && col < N) ? B[(size_t)b_row * N + col] : 0.0;
                    }
                }
                for (int ty = 0; ty < TILE; ty++)
                    for (int tx = 0; tx < TILE; tx++)
                        for (int kk = 0; kk < TILE; kk++)
                            acc[ty][tx] += tileA[ty][kk] * tileB[kk][tx];
            }
            for (int ty = 0; ty < TILE; ty++)
                for (int tx = 0; tx < TILE; tx++) {
                    int row = by * TILE + ty, col = bx * TILE + tx;
                    if (row < M && col < N) C[(size_t)row * N + col] = acc[ty][tx];
                }
        }
    }
}

static void reference_contraction(const std::vector<double>& A, const std::vector<double>& B,
                                   std::vector<double>& C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = acc;
        }
}

int main() {
    printf("=== Verifying the tiled kernel's boundary logic WITHOUT a GPU ===\n\n");

    const int M = 37, K = 23, N = 19;
    printf("M=%d, K=%d, N=%d (K/TILE = %.3f -- the fractional last tile is the risk)\n\n", M, K, N, (double)K / TILE);

    std::vector<double> A((size_t)M * K), B((size_t)K * N);
    std::vector<double> C_ref((size_t)M * N), C_emulated((size_t)M * N);

    unsigned seed = 4242;
    for (auto& v : A) { seed = seed * 1103515245u + 12345u; v = (double)(seed % 100) / 10.0; }
    for (auto& v : B) { seed = seed * 1103515245u + 12345u; v = (double)(seed % 100) / 10.0; }

    reference_contraction(A, B, C_ref, M, K, N);
    emulate_tiled_kernel(A, B, C_emulated, M, K, N);

    double max_diff = 0.0;
    for (size_t i = 0; i < C_ref.size(); i++) max_diff = std::max(max_diff, std::abs(C_ref[i] - C_emulated[i]));

    printf("max |reference - emulated tiled kernel| over all %d output elements: %.3e\n", M * N, max_diff);

    if (max_diff == 0.0) {
        printf("PASS -- the tiling logic, INCLUDING the K-not-a-multiple-of-TILE padding\n");
        printf("branch, is exactly correct.\n");
    } else {
        printf("FAIL -- the tiling logic disagrees with the reference; see max_diff above.\n");
        return 1;
    }
    return 0;
}
```

### File: 03_multi_axis_contraction_kernel.cu

```cpp
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>

#define MAX_RANK 4

struct TensorMeta {
    int shape[MAX_RANK];
    long long strides[MAX_RANK];
    int rank;
};

__host__ __device__ void compute_row_major_strides(TensorMeta& t) {
    long long acc = 1;
    for (int i = t.rank - 1; i >= 0; i--) { t.strides[i] = acc; acc *= t.shape[i]; }
}

__global__ void multi_axis_contraction_kernel(
    const double* A, const double* B, double* C,
    TensorMeta metaA, TensorMeta metaB, TensorMeta metaC,
    int free_a[MAX_RANK], int free_a_count,
    int free_b[MAX_RANK], int free_b_count,
    int axes_a[MAX_RANK], int axes_b[MAX_RANK],
    int contracted_shape[MAX_RANK], long long contracted_strides[MAX_RANK],
    int contracted_rank, long long contracted_total)
{
    long long out_lin = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long out_total = 1;
    for (int i = 0; i < metaC.rank; i++) out_total *= metaC.shape[i];
    if (out_lin >= out_total) return;

    int out_idx[MAX_RANK];
    long long rem = out_lin;
    for (int i = 0; i < metaC.rank; i++) { out_idx[i] = (int)(rem / metaC.strides[i]); rem %= metaC.strides[i]; }

    int idx_a[MAX_RANK] = {0}, idx_b[MAX_RANK] = {0};
    for (int i = 0; i < free_a_count; i++) idx_a[free_a[i]] = out_idx[i];
    for (int j = 0; j < free_b_count; j++) idx_b[free_b[j]] = out_idx[free_a_count + j];

    double acc = 0.0;
    for (long long c_lin = 0; c_lin < contracted_total; c_lin++) {
        int c_idx[MAX_RANK];
        long long crem = c_lin;
        for (int i = 0; i < contracted_rank; i++) { c_idx[i] = (int)(crem / contracted_strides[i]); crem %= contracted_strides[i]; }
        for (int k = 0; k < contracted_rank; k++) { idx_a[axes_a[k]] = c_idx[k]; idx_b[axes_b[k]] = c_idx[k]; }

        long long lin_a = 0; for (int i = 0; i < metaA.rank; i++) lin_a += (long long)idx_a[i] * metaA.strides[i];
        long long lin_b = 0; for (int i = 0; i < metaB.rank; i++) lin_b += (long long)idx_b[i] * metaB.strides[i];
        acc += A[lin_a] * B[lin_b];
    }

    long long lin_c = 0;
    for (int i = 0; i < metaC.rank; i++) lin_c += (long long)out_idx[i] * metaC.strides[i];
    C[lin_c] = acc;
}

static void reference_double_contraction(const std::vector<double>& A, const std::vector<double>& B,
                                          std::vector<double>& C) {
    for (int i = 0; i < 2; i++)
        for (int l = 0; l < 5; l++) {
            double acc = 0.0;
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                    acc += A[(size_t)i * 12 + j * 4 + k] * B[(size_t)j * 20 + k * 5 + l];
            C[(size_t)i * 5 + l] = acc;
        }
}

int main() {
    printf("=== A double contraction on the GPU: TWO shared axes, one thread per output ===\n\n");

    std::vector<double> hA(24), hB(60), hC_ref(10);
    for (size_t i = 0; i < hA.size(); i++) hA[i] = (double)(i % 7) - 3.0;
    for (size_t i = 0; i < hB.size(); i++) hB[i] = (double)(i % 5) - 2.0;
    reference_double_contraction(hA, hB, hC_ref);

    printf("host reference (identical inputs to 03_double_contraction.cpp, already\n");
    printf("cross-checked there against numpy.tensordot):\n  ");
    for (double v : hC_ref) printf("%.2f ", v);
    printf("\n\n");

    TensorMeta metaA{{2, 3, 4, 1}, {}, 3};
    TensorMeta metaB{{3, 4, 5, 1}, {}, 3};
    TensorMeta metaC{{2, 5, 1, 1}, {}, 2};
    compute_row_major_strides(metaA);
    compute_row_major_strides(metaB);
    compute_row_major_strides(metaC);

    int free_a[MAX_RANK] = {0, 0, 0, 0}; int free_a_count = 1;
    int free_b[MAX_RANK] = {2, 0, 0, 0}; int free_b_count = 1;
    int axes_a[MAX_RANK] = {1, 2, 0, 0};
    int axes_b[MAX_RANK] = {0, 1, 0, 0};
    int contracted_shape[MAX_RANK] = {3, 4, 0, 0};
    long long contracted_strides[MAX_RANK] = {4, 1, 0, 0};
    int contracted_rank = 2;
    long long contracted_total = 12;

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    int *d_free_a = nullptr, *d_free_b = nullptr, *d_axes_a = nullptr, *d_axes_b = nullptr, *d_c_shape = nullptr;
    long long* d_c_strides = nullptr;
    cudaError_t err;

    err = cudaMalloc(&dA, hA.size() * sizeof(double));
    printf("cudaMalloc(dA): %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&dB, hB.size() * sizeof(double));
    printf("cudaMalloc(dB): %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&dC, 10 * sizeof(double));
    printf("cudaMalloc(dC): %s\n", cudaGetErrorString(err));

    std::vector<double> hC_gpu(10, -1.0);
    if (dA && dB && dC) {
        cudaMalloc(&d_free_a, MAX_RANK * sizeof(int));
        cudaMalloc(&d_free_b, MAX_RANK * sizeof(int));
        cudaMalloc(&d_axes_a, MAX_RANK * sizeof(int));
        cudaMalloc(&d_axes_b, MAX_RANK * sizeof(int));
        cudaMalloc(&d_c_shape, MAX_RANK * sizeof(int));
        cudaMalloc(&d_c_strides, MAX_RANK * sizeof(long long));

        cudaMemcpy(dA, hA.data(), hA.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), hB.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_free_a, free_a, MAX_RANK * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_free_b, free_b, MAX_RANK * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_axes_a, axes_a, MAX_RANK * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_axes_b, axes_b, MAX_RANK * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c_shape, contracted_shape, MAX_RANK * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c_strides, contracted_strides, MAX_RANK * sizeof(long long), cudaMemcpyHostToDevice);

        int threads = 32, blocks = (10 + threads - 1) / threads;
        multi_axis_contraction_kernel<<<blocks, threads>>>(
            dA, dB, dC, metaA, metaB, metaC,
            d_free_a, free_a_count, d_free_b, free_b_count,
            d_axes_a, d_axes_b, d_c_shape, d_c_strides, contracted_rank, contracted_total);
        err = cudaGetLastError();
        printf("kernel launch: %s\n", cudaGetErrorString(err));
        err = cudaDeviceSynchronize();
        printf("cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        cudaMemcpy(hC_gpu.data(), dC, 10 * sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        printf("(skipping memcpy/launch -- device allocation did not succeed on this machine)\n");
    }

    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    if (d_free_a) cudaFree(d_free_a);
    if (d_free_b) cudaFree(d_free_b);
    if (d_axes_a) cudaFree(d_axes_a);
    if (d_axes_b) cudaFree(d_axes_b);
    if (d_c_shape) cudaFree(d_c_shape);
    if (d_c_strides) cudaFree(d_c_strides);

    return 0;
}
```

## Chapter Summary

The free/contracted index split that defines a tensor contraction on paper maps directly onto CUDA's own division of labor: free indices become the grid of threads, contracted indices become each thread's private loop. This appendix built that mapping up in the same order Appendix H built the CPU version -- a naive one-thread-per-output-element kernel first, then a shared-memory-tiled version that generalizes the classic tiled-matmul optimization to tiling along an arbitrary contracted axis, then a fully general multi-axis version using the fixed-size-array indexing technique Appendix H's Section H.2 introduced. With no physical GPU available in this environment, every kernel's *compilation* was still genuine (`nvcc`, real architecture flags, zero warnings), every runtime result was reported exactly as it occurred (`cudaGetErrorString`'s honest "no CUDA-capable device is detected"), and every kernel's *algorithm* was still genuinely checked -- by an independent host reference for the simplest kernel, and by a direct host-side emulation of the kernel's own exact loop structure for the two more intricate ones, catching the boundary case (`K` not a multiple of `TILE`) that would have been easiest to get wrong and hardest to notice.

## Self-Check Questions

1. In the mapping this appendix builds, what plays the role of Appendix H's outer "free-index" loop, and what plays the role of its inner "contracted-index" loop?
2. Why does `contraction_kernel` need the `if (row >= M || col >= N) return;` guard at all, given that `row` and `col` are computed directly from the launch configuration?
3. What does tiling actually reduce -- FLOP count, or something else? Defend your answer using Section I.2's own reported numbers.
4. Section I.2 deliberately chose `K=23` rather than a multiple of 16. What would have been left unverified if `K` had been chosen as a multiple of `TILE` instead?
5. Why can Section I.3's kernel not reuse Appendix H's `for_each_index()` helper unmodified?
6. What, precisely, does the host-side emulation in Section I.2 prove, and what does it explicitly NOT prove?
7. If this appendix had access to a real GPU, what specific additional evidence would you want to see before trusting `multi_axis_contraction_kernel`'s output, beyond what a host emulation can already provide?

### Worked Solutions

1. The CUDA grid (the collection of thread blocks, and within them, individual threads identified by `blockIdx`/`threadIdx`) plays the role of the free-index loop -- one thread per combination of free indices, i.e. per output element. Each individual thread's own `for` loop over the contracted axis (or, in Section I.3, over the contracted linear index `c_lin`) plays the role of the contracted-index loop.
2. The grid is sized by rounding `M` and `N` up to whole multiples of the block dimensions (`ceil(N/16) x ceil(M/16)` blocks of `16x16` threads), which can request more threads than there are actual output elements whenever `M` or `N` isn't itself a multiple of the block size. Without the guard, those extra threads would compute `row`/`col` values at or past the end of `C` and write out of bounds.
3. Tiling reduces global-memory traffic, not FLOP count -- the same `M·N·K` multiply-adds happen either way. Section I.2's own numbers make this explicit: "global memory reads per output element WITHOUT tiling: K = 23" versus "WITH this tiling: K / TILE (amortized) = 1.44" -- roughly a 16x reduction in memory reads per output element, for identical arithmetic.
4. The boundary-padding branch (`(row < M && a_col < K) ? A[...] : 0.0`, and its counterpart for `B`) would never execute its "false" case -- every tile would be a full `TILE x TILE` block with nothing to pad, and a bug in the padding logic specifically could ship without any test ever catching it. Choosing `K=23` (not a multiple of 16) forces the last tile along the contracted axis to be fractional, so the padding branch's "insert zero instead of reading" case is genuinely exercised by the verification in Worked Example I.2.1.
5. `for_each_index()` recurses once per tensor axis, with the number of axes (the tensor's rank) known only at runtime, from a `std::vector`'s size. CUDA device code has no `std::vector`, and per-thread recursion to a runtime-determined depth doesn't fit how a GPU schedules thousands of concurrent, identical-instruction threads cheaply. Section I.3's kernel instead fixes a compile-time `MAX_RANK` and uses plain iterative loops bounded by that constant, at the cost of only supporting tensors up to that rank.
6. It proves that the kernel's INDEX ARITHMETIC -- which memory locations get read, added, and written, including the boundary-padding condition -- produces the correct sums, because that arithmetic was translated statement-for-statement into a host loop and run for real. It does NOT prove anything about how that same arithmetic performs, or even whether it compiles correctly as actual device code, when executed across real hardware threads with real memory hierarchies, real `__syncthreads()` synchronization, or real warp scheduling -- those properties can only be confirmed by compiling and running the actual `__global__` kernel on an actual GPU, which is precisely the step this environment cannot perform.
7. Actual execution on physical hardware, comparing `hC_gpu` (read back from the device) against the reference to floating-point tolerance -- confirming not just that the arithmetic is correct in principle but that `__syncthreads()` placement has no race, that the metadata structs marshal correctly across the host/device boundary in practice, and that the specific compiled machine code for this specific architecture computes what the source says it should. A host emulation checks the algorithm; only a real device run checks the actual CUDA execution of it.
