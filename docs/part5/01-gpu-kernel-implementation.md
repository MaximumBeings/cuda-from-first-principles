# Chapter 18: GPU Kernel Implementation — Kernels That Earn What Chapter 4 Promised

> "Chapter 4 built the model: a thread's identity is two numbers, memory has three scopes, and a warp either reads together or pays for reading separately. This chapter is what happens when that model meets a kernel with a job to actually finish quickly — the launch math that doesn't waste a thread, the layout that doesn't waste a transaction, the on-chip memory that doesn't make ten threads pay for the same value ten times over, and the one instruction that skips memory altogether for the last few steps of a reduction."

**What you will understand by the end of this chapter:**

- How to compute a real launch configuration by ceiling division — block count, total threads launched, and exactly how many of them are wasted — and why the in-kernel bounds check isn't defensive boilerplate but the other half of a single design decision, traced on a tensor of a million elements and confirmed by a genuine (if GPU-less) `<<<>>>` launch
- Memory coalescing translated from Chapter 4's abstract warp argument into a concrete win on a real struct: reading one field across many bond records, counted in actual transactions and actual bandwidth utilization, for both the Array-of-Structs and Struct-of-Arrays layouts — then confirmed a second, deeper way this chapter's Mojo counterpart never attempts: reading the actual compiled SASS instructions and decoding the byte-stride multiplier `nvcc` hides inside a half-precision bit trick
- Why a naive convolution kernel reads most of its input many times over, counted exactly rather than asserted, and how staging one block's input tile into real `__shared__` memory — with more threads doing the loading than will ever compute an output, synchronized by a real `__syncthreads()` — removes that redundancy entirely
- Why padding the input (rather than shrinking the output) is what a "same"-shaped convolution actually requires, traced by hand at a border position and an interior position of the same small example
- Warp-level shuffle instructions as the natural endpoint of a tree reduction once the active thread count drops to one warp — built here on the real `__shfl_down_sync` intrinsic — and the one precondition (every lane in the warp must participate) that this chapter's own Section 18.1 bounds-check pattern can silently violate, demonstrated with a genuinely broken, non-`10.0` result rather than only described in prose

**What you need to know first:**

- Chapter 4 in full: the thread hierarchy and global-index formula (4.1), the host/device split (4.2), the three memory scopes (4.3), and memory coalescing's warp-transaction argument (4.4). This chapter takes each of those ideas into real launch configurations and real kernel code.
- Chapter 3 (Struct-of-Arrays vs Array-of-Structs) — Section 18.2 reuses its argument, and its `cuobjdump --dump-sass` disassembly technique, on a financial data struct.
- Chapter 14.1 (`tensor_sum`'s tree reduction) — Section 18.4 is precisely the optimization that shortens the last few rounds of that same reduction.
- Chapter 13.1 (matrix multiplication's index-summation form) — useful background for how shared-memory tiling generalizes past convolution to any operation where neighboring threads reread the same input.

## 18.1 CUDA-Style Kernel Design `[FOUNDATIONAL]`

### Intuition

A moving company only sends out crews in fixed sizes — say, four movers to a truck — never a lone mover, no matter how small the job. If a job needs eleven boxes moved, the company can't send "two and three quarters" crews; it sends three full crews, twelve movers, and the twelfth mover simply has nothing to do when they arrive. The company's actual obligation isn't to send exactly the right number of movers — it's to send *enough* crews to cover the job, and to make sure the extra mover on the last truck knows to stand aside rather than grab a box that was never there. A GPU kernel launch works exactly the same way: threads only come in fixed-size blocks, so covering `N` elements almost never divides evenly, and the kernel itself has to be the one that tells the leftover threads to stand aside.

### Background

Every kernel in this book follows the same two-part template Chapter 4 introduced in the abstract: compute a thread's global index, then guard it.

```cpp
// The template every kernel in this book follows: compute a global
// thread index, bounds-check it, do one unit of work.
__global__ void generic_elementwise_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}
```

The launch configuration is the other half of the decision, and it's made entirely on the host, before a single thread runs:

```cpp
struct LaunchConfig {
    int num_blocks;
    int threads_per_block;
    long total_threads;
    long wasted_threads;
};

LaunchConfig compute_launch_config(long size, int threads_per_block) {
    // Ceiling division: the "+ threads_per_block - 1" before the
    // floor-dividing "/" is what rounds up instead of down.
    int num_blocks = (int)((size + threads_per_block - 1) / threads_per_block);
    long total_threads = (long)num_blocks * threads_per_block;
    LaunchConfig cfg;
    cfg.num_blocks = num_blocks;
    cfg.threads_per_block = threads_per_block;
    cfg.total_threads = total_threads;
    cfg.wasted_threads = total_threads - size;
    return cfg;
}

// launched as: generic_elementwise_kernel<<<cfg.num_blocks, cfg.threads_per_block>>>(output, input, size);
```

`num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK` is *ceiling* division computed with integer arithmetic — the `+ THREADS_PER_BLOCK - 1` before the truncating `/` is what rounds up instead of down. Rounding up guarantees every element gets covered by some block; the `if (idx < size)` guard back in the kernel is what stops the inevitable overshoot from writing past the end of the buffer. Neither piece is optional, and each protects against a different failure:

| Ceiling division | Bounds check | Result |
|---|---|---|
| No (`size / N`) | No | Tail elements past the last full block are never processed at all |
| No (`size / N`) | Yes | Still misses the tail — a guard can't rescue elements whose block was never launched |
| Yes (`(size+N-1)/N`) | No | Overshoot threads write past the buffer's end — memory corruption |
| Yes (`(size+N-1)/N`) | Yes | Full coverage, no corruption — the pattern every kernel in this book uses |

### Worked Example 18.1.1 — A million elements, traced to the exact wasted thread

Compiled and run:

```bash
nvcc -arch=sm_80 01_launch_configuration.cu -o 01_launch_configuration
./01_launch_configuration
```

Genuinely compiled and run:

```
=== Section 18.1: launch configuration, ceiling division vs floor division ===

size=1000000, THREADS_PER_BLOCK=256
num_blocks = (1000000 + 255) / 256 = 1000255 / 256 = 3907
total threads launched = 3907 x 256 = 1000192
wasted threads = 192
last block is block 3906, covering global indices 999936 through 1000191
of its 256 threads, only the first 64 satisfy idx < 1000000 and do real work
```

For a tensor of exactly `1,000,000` elements at `THREADS_PER_BLOCK = 256`: `num_blocks = (1,000,000 + 255) / 256 = 1,000,255 / 256 = 3907` (integer division truncates the remainder). Those `3907` blocks launch `3907 × 256 = 1,000,192` threads in total — `192` more than the tensor has elements. The last block, block `3906`, covers global indices `999,936` through `1,000,191`; of its `256` threads, only the first `64` satisfy `idx < 1,000,000` and do real work — indices `999,936` through `999,999`. The remaining `192` threads in that one block, and only that one block, are the ones the bounds check exists to stop.

### Worked Example 18.1.2 — A different size and block width, to check the pattern generalizes

Same run continues:

```
size=10000, THREADS_PER_BLOCK=128
num_blocks = (10000 + 127) / 128 = 10127 / 128 = 79
total threads launched = 79 x 128 = 10112
wasted threads = 112
last block is block 78, covering global indices 9984 through 10111
valid indices are 9984 through 9999 -- exactly 16 active threads, 112 idle
```

Same formula, different numbers: `size = 10,000`, `THREADS_PER_BLOCK = 128`. `num_blocks = (10,000 + 127) / 128 = 10,127 / 128 = 79`. Total threads launched: `79 × 128 = 10,112` — `112` wasted. The last block is block `78`, covering global indices `78 × 128 = 9,984` through `10,111`. Valid indices are `9,984` through `9,999` — exactly `16` active threads in that block, and the remaining `128 - 16 = 112` idle threads account for the entire wasted count computed above. As in the million-element case, every wasted thread lives in exactly one block — the last one — never scattered across the launch.

```
[COMMON TRAP]  Removing "wasteful" rounding by shrinking block count

It is tempting to compute num_blocks = size / THREADS_PER_BLOCK
instead -- after all, why launch a block that is mostly idle? The
answer is in the table above: floor division doesn't shrink the idle
thread count, it deletes real work. For size=10,000 and
THREADS_PER_BLOCK=128, size / 128 = 78 (not 79) -- exactly one block
short. Elements 9,984 through 9,999, the sixteen elements this
chapter's own worked example just traced as "active," would never be
covered by any launched block at all, and no bounds check inside the
kernel can process an index that no thread was ever assigned. The
"wasted" threads on the last block are not a bug to engineer away --
they are the cost of guaranteeing full coverage with fixed-size
blocks, paid once per launch, and the bounds check is what makes that
cost safe to pay.
```

Same run, confirmed genuinely rather than merely argued:

```
--- COMMON TRAP: floor division genuinely drops elements, a bounds check cannot rescue them ---
size // THREADS_PER_BLOCK = 10000 // 128 = 78 blocks (not 79)
that launches only 9984 threads, covering global indices 0 through 9983
elements 9984 through 9999 (16 elements) are NEVER covered by any launched block --
no bounds check inside the kernel can process an index no thread was ever assigned.
```

Following this book's established Chapter 4 pattern, the same file also genuinely launches a real kernel — `generic_elementwise_kernel<<<2,4>>>` over `8` elements — and honestly reports what a no-GPU environment actually returns:

```
--- attempting a genuine kernel launch (no GPU in this environment) ---
cudaMalloc x2 -> 100,100 (no CUDA-capable device is detected)
cudaMemcpy H2D -> 100 (no CUDA-capable device is detected)
kernel launch <<<2,4>>> -> 100 (no CUDA-capable device is detected)
cudaMemcpy D2H -> 100 (no CUDA-capable device is detected)
h_out after (unchanged, since nothing ran): 0 0 0 0 0 0 0 0 
host-computed reference (input * 2): 2 4 6 8 10 12 14 16
```

Every CUDA Runtime API call genuinely executes and genuinely returns `cudaError_t` code `100` (`cudaErrorNoDevice`) — not a simulated or hand-typed value — because this environment has no GPU. `h_out` stays all zeros because the kernel never ran; the host-computed reference on the last line is the actual answer the kernel would have produced, calculated independently in plain C++ to have something concrete to compare against once this code runs on real hardware.

## 18.2 Memory Coalescing Optimization `[FOUNDATIONAL]`

### Intuition

A records clerk is asked to look up one field — say, the interest rate — for four different customer files. If all four rates happen to be written on one shared summary sheet, one glance retrieves all four. If instead each rate is buried on page three of four separate, fully-detailed customer folders, the clerk pays for four separate trips to the filing cabinet to retrieve the exact same four numbers. Struct-of-Arrays is the summary sheet; Array-of-Structs is the stack of folders — and a GPU warp reading a field across many records is exactly this clerk, dozens of times over, every single kernel launch.

### Background

```cpp
// Array of Structs (AoS) -- poor coalescing
struct ZeroCouponBondAoS {
    float face_value;          // offset 0
    float time_to_maturity;    // offset 4
    float risk_free_rate;      // offset 8  -- float-index 2
    float credit_spread;       // offset 12
    float present_value;       // offset 16
    float yield_to_maturity;   // offset 20
    float duration;            // offset 24 -- float-index 6
    float portfolio_weight;    // offset 28
};                              // total size: 32 bytes

// Struct of Arrays (SoA) -- optimal coalescing
struct ZeroCouponBondSystemSoA {
    float* face_value;
    float* time_to_maturity;
    float* risk_free_rate;
    float* credit_spread;
    float* present_value;
    float* yield_to_maturity;
    float* duration;
    float* portfolio_weight;
};

__global__ void compute_from_soa_kernel(float* output, const float* rate, const float* spread, int num_items) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_items) {
        output[idx] = rate[idx] + spread[idx];
    }
}
```

This is not a hypothetical: it is the exact struct that prices a bond portfolio in Part 7, where the SoA `ZeroCouponBondSystemSoA` holds eight `float` fields (`face_value`, `time_to_maturity`, `risk_free_rate`, `credit_spread`, `present_value`, `yield_to_maturity`, `duration`, `portfolio_weight`) as eight separate contiguous arrays instead of one interleaved struct per bond. An Array-of-Structs record holding those same eight fields is `8 × 4 = 32` bytes wide (confirmed below by a genuine `sizeof`, not assumed); the stride between the *same* field in two adjacent AoS records is that entire `32`-byte record, while the stride between adjacent elements of one SoA array is just `4` bytes — an `8×` difference, not a rough estimate.

### Worked Example 18.2.1 — Counting actual transactions, AoS vs. SoA

Compiled and run:

```bash
nvcc -arch=sm_80 02_memory_coalescing_bond_soa.cu -o 02_memory_coalescing_bond_soa
./02_memory_coalescing_bond_soa
```

Genuinely compiled and run:

```
=== Section 18.2: memory coalescing on a real 8-field financial record ===

sizeof(ZeroCouponBondAoS) = 32 bytes

Memory layout comparison:
  AoS stride between same field, adjacent records: 32 bytes
  SoA stride between adjacent elements, one field:  4 bytes
  Coalescing improvement:                          8x better

--- Worked Example 18.2.1: risk_free_rate, 4 threads, 16-byte chunks ---
SoA: risk_free_rate[0..3] at float-indices 0,1,2,3 -- 1 distinct chunk(s)
  transactions needed = 1
  bytes moved         = 16
  bytes used          = 16
  utilization         = 100%

  record 0's risk_free_rate at float-index 2 -> chunk 0
  record 1's risk_free_rate at float-index 10 -> chunk 2
  record 2's risk_free_rate at float-index 18 -> chunk 4
  record 3's risk_free_rate at float-index 26 -> chunk 6
AoS: 4 distinct chunk(s)
  transactions needed = 4
  bytes moved         = 4 x 16 = 64
  bytes used          = 4 x 4 = 16
  utilization         = 16/64 = 25%
```

Scale a 32-thread warp down to `4` threads (the same scaling trick Chapter 4.4 used), and treat every `16` contiguous bytes (`4` `float`s) as one memory transaction. Four threads read `risk_free_rate` for bond records `0` through `3`.

**SoA**: `risk_free_rate[0], risk_free_rate[1], risk_free_rate[2], risk_free_rate[3]` sit contiguously at float-indices `0, 1, 2, 3` — all inside the same `16`-byte chunk, genuinely confirmed above as `1` transaction, `100%` utilization.

**AoS**, with the field order above (`risk_free_rate` is the third of eight fields, float-index `2` within each `8`-float record): record `0`'s `risk_free_rate` is at float-index `2`, record `1`'s is at `8 + 2 = 10`, record `2`'s is at `16 + 2 = 18`, record `3`'s is at `24 + 2 = 26`. Dividing each by `4` to find its `16`-byte chunk: `2 → chunk 0`, `10 → chunk 2`, `18 → chunk 4`, `26 → chunk 6` — four distinct chunks, one per thread, genuinely computed above as `4` transactions, `25%` utilization.

Four threads, four numbers, identical useful data — `1` transaction against `4`, precisely the structural shape of Chapter 4.4's own strided-access case, just instantiated on a real eight-field financial record instead of an abstract array. The same run also genuinely launches `compute_from_soa_kernel<<<1,4>>>` over real SoA arrays, honestly reporting `cudaErrorNoDevice` the same way Section 18.1's launch did:

```
--- compute_from_soa_kernel: genuine SoA-layout kernel, compiled for sm_80 ---
cudaMalloc x3 -> 100,100,100 (no CUDA-capable device is detected)
kernel launch <<<1,4>>> -> 100 (no CUDA-capable device is detected)
h_output after (unchanged, since nothing ran): 0.0000 0.0000 0.0000 0.0000
host reference (rate + spread): 0.0300 0.0370 0.0450 0.0530
```

```
[COMMON TRAP]  Assuming SoA collapses every field access into one transaction

SoA guarantees that reading ONE field across many records is coalesced
-- it says nothing about how many transactions a kernel that needs
SEVERAL fields per thread will issue. compute_bond_prices_kernel
(Part 7) reads risk_free_rate, credit_spread, face_value, and
time_to_maturity for every bond it prices -- four separate SoA arrays,
so four separate (individually coalesced) transactions per warp, not
one. SoA turns each of those four reads from a scattered, 32-byte-
stride mess into a single clean transaction; it does not, and cannot,
merge four logically different fields into one physical read.
```

### Worked Example 18.2.2 — Reading the actual SASS, not just counting by hand

Chapter 3 established that `cuobjdump --dump-sass` can show the real compiled instructions a struct layout produces, reading the multiplier straight off an `IMAD.WIDE`. This chapter's own struct disassembles differently, and following the evidence honestly all the way through turns out to be more interesting than the Chapter 3 case: `nvcc` doesn't always encode a byte-stride multiplier as a directly visible immediate.

```bash
nvcc -arch=sm_80 -cubin 03_coalescing_sass_evidence.cu -o 03_coalescing_sass_evidence.cubin
cuobjdump --dump-sass 03_coalescing_sass_evidence.cubin
```

The SoA kernel's compiled SASS (genuine `cuobjdump` output, function `compute_from_soa_kernel`):

```
        /*0060*/                   HFMA2.MMA R7, -RZ, RZ, 0, 2.384185791015625e-07 ;
        /*0080*/                   IMAD.WIDE R4, R6, R7, c[0x0][0x170] ;
        /*0090*/                   IMAD.WIDE R2, R6.reuse, R7.reuse, c[0x0][0x168] ;
```

And the AoS kernel's (function `compute_from_aos_kernel`):

```
        /*0060*/                   HFMA2.MMA R3, -RZ, RZ, 0, 1.9073486328125e-06 ;
        /*0080*/                   IMAD.WIDE R2, R4, R3, c[0x0][0x160] ;
        /*0090*/                   LDG.E R0, [R2.64+0xc] ;
        /*00a0*/                   LDG.E R7, [R2.64+0x8] ;
        /*00b0*/                   HFMA2.MMA R5, -RZ, RZ, 0, 2.384185791015625e-07 ;
        /*00c0*/                   IMAD.WIDE R4, R4, R5, c[0x0][0x168] ;
```

Neither `IMAD.WIDE`'s multiplier operand (`R7`, or `R3`/`R5`) is a plain visible integer immediate the way Chapter 3's example was — `nvcc` instead materializes it through `HFMA2.MMA`, a half-precision fused-multiply-add instruction, encoding the integer as a floating-point bit pattern (`-RZ * RZ + 0 + <float>` collapses to just `<float>`, so this is really just "load this constant into a register" spelled through the half-precision unit). Decoding it honestly means treating that printed float as what it actually is: a multiple of the smallest representable half-precision-style denormal, `2⁻²⁴`.

```python
denorm = 2.0 ** -24                       # 5.960464477539063e-08
2.384185791015625e-07 / denorm  # -> 4.0   (SoA kernel: sizeof(float))
1.9073486328125e-06  / denorm  # -> 32.0  (AoS kernel: sizeof(ZeroCouponBondAoS))
```

Genuinely computed: `2.384185791015625e-07 / 2⁻²⁴ = 4.0` for the SoA kernel — confirming `sizeof(float) = 4`, the stride between adjacent `rate[idx]` elements — and `1.9073486328125e-06 / 2⁻²⁴ = 32.0` for the AoS kernel — confirming `sizeof(ZeroCouponBondAoS) = 32`, the same number Worked Example 18.2.1 already measured by hand. The compiled machine code, decoded from its actual bit pattern rather than assumed, agrees with the hand count exactly.

One nuance the hand count alone wouldn't surface: the AoS kernel's SASS shows *two* `IMAD.WIDE` instructions, not one. The first (`R2 = R4 * R3 + c[0x160]`, stride `32`) computes the address of `bonds[idx]` itself — the wide, expensive stride. The second (`R4 = R4 * R5 + c[0x168]`, stride `4`) computes the address of `out[idx]`, a plain `float*` array indexed at ordinary `4`-byte stride. Only the read of the struct array pays the `32`-byte penalty; the kernel's write to its plain output array is exactly as coalesced in the AoS version as in the SoA version. The coalescing cost lives specifically in reading `bonds[idx].risk_free_rate` and `bonds[idx].credit_spread` — the two `LDG.E` loads at offsets `+0x8` and `+0xc` from that wide-strided address — not in every memory access the kernel makes.

## 18.3 Shared Memory Utilization `[FOUNDATIONAL]`

### Intuition

Picture four apprentices in one workshop bay, each assigned to finish one tile of a mosaic, where every tile needs paint from a shared can sitting on a shelf across the room. If each apprentice fetches their own paint every time they need a dab, the same can gets carried across the room far more times than necessary — especially since neighboring tiles need overlapping colors. A foreman who instead sends a *few* apprentices to bring the whole can to the workbench once, share it there, and only then start painting saves every trip after the first. Shared memory is that workbench: on-chip, visible to every thread in one block, and loaded exactly once per block instead of once per thread.

### Background

The naive convolution kernel each thread runs independently, re-reading global memory for every multiply — genuinely compiled for `sm_80`, never launched in this no-GPU environment:

```cpp
__global__ void naive_conv2d_kernel(const float* input, int input_h, int input_w,
                                     const float* kernel_, int kernel_h, int kernel_w,
                                     float* output, int output_h, int output_w) {
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_row >= output_h || out_col >= output_w) return;

    float sum = 0.0f;
    for (int k_r = 0; k_r < kernel_h; k_r++) {
        for (int k_c = 0; k_c < kernel_w; k_c++) {
            int in_row = out_row + k_r;
            int in_col = out_col + k_c;
            sum += input[in_row * input_w + in_col] * kernel_[k_r * kernel_w + k_c];
        }
    }
    output[out_row * output_w + out_col] = sum;
}
```

Neighboring output threads' input windows overlap heavily — a `3×3` kernel means most interior input positions are read by up to nine different output threads, all pulling from global memory independently. The shared-memory fix stages the block's *entire* input footprint into on-chip memory once, using *more* threads than the block will ever use to compute outputs, because the extra input rows and columns at the tile's edges (the "halo") still need loading even though no thread centered there will ever own an output. This is a real `__global__` kernel with a real `__shared__` array and a real `__syncthreads()` barrier, compiled for `sm_80`:

```cpp
#define TILE 2
#define KERNEL_SIZE 3
#define SHARED_DIM (TILE + KERNEL_SIZE - 1)   // input footprint one tile of outputs needs

__global__ void tiled_conv2d_kernel(const float* input, int input_h, int input_w,
                                     const float* kernel_, int kernel_h, int kernel_w,
                                     float* output, int output_h, int output_w) {
    __shared__ float tile[SHARED_DIM * SHARED_DIM];

    // Every thread in the (oversized) block loads exactly one tile cell --
    // including the threads that will never compute an output.
    int in_row = blockIdx.y * TILE + threadIdx.y;
    int in_col = blockIdx.x * TILE + threadIdx.x;
    if (in_row < input_h && in_col < input_w) {
        tile[threadIdx.y * SHARED_DIM + threadIdx.x] = input[in_row * input_w + in_col];
    } else {
        tile[threadIdx.y * SHARED_DIM + threadIdx.x] = 0.0f;
    }

    __syncthreads();   // every thread in the block must finish writing before ANY thread reads

    // Only the TILE x TILE "interior" threads own an output element.
    if (threadIdx.y >= TILE || threadIdx.x >= TILE) return;
    int out_row = blockIdx.y * TILE + threadIdx.y;
    int out_col = blockIdx.x * TILE + threadIdx.x;
    if (out_row >= output_h || out_col >= output_w) return;

    float sum = 0.0f;
    for (int k_r = 0; k_r < kernel_h; k_r++) {
        for (int k_c = 0; k_c < kernel_w; k_c++) {
            sum += tile[(threadIdx.y + k_r) * SHARED_DIM + (threadIdx.x + k_c)] * kernel_[k_r * kernel_w + k_c];
        }
    }
    output[out_row * output_w + out_col] = sum;
}
```

Every value this kernel needs is read from global memory exactly once, by whichever thread happens to own that tile position during the load phase; every subsequent read, during the actual convolution loop, comes from `tile` — on-chip, block-scoped, and (per Chapter 4.3) dramatically faster than global memory.

### Worked Example 18.3.1 — The naive kernel's redundant reads, counted exactly

Reuse a small, complete example: a `4×4` input filled sequentially and a `3×3` kernel of alternating `1`s and `0`s.

Compiled and run:

```bash
nvcc -arch=sm_80 04_naive_convolution.cu -o 04_naive_convolution
./04_naive_convolution
```

Genuinely compiled and run:

```
=== Section 18.3: naive convolution, redundant reads counted exactly ===

Input (4x4):
   1  2  3  4
   5  6  7  8
   9 10 11 12
  13 14 15 16
Kernel (3x3):
  1 0 1
  0 1 0
  1 0 1

output = 30  35
         50  55

read_counts per input cell:
  1 2 2 1
  2 4 4 2
  2 4 4 2
  1 2 2 1

total global-memory reads = 36 (for 16 unique input values)
independent check: 4 outputs x 9 kernel taps each = 36
```

With no padding, output is `2×2`: `output(0,0)` sums over input rows `0`-`2`, cols `0`-`2`: `1·1 + 2·0 + 3·1 + 5·0 + 6·1 + 7·0 + 9·1 + 10·0 + 11·1 = 1+3+6+9+11 = 30`. The same arithmetic on the other three windows gives `output(0,1)=35`, `output(1,0)=50`, `output(1,1)=55`, matching the genuine run exactly.

Now count how many of the naive kernel's four output threads read each input cell. Interior cells `(1,1)`, `(1,2)`, `(2,1)`, `(2,2)` are each read by all four output threads — `4` times each, genuinely counted by an actual counter array incremented on every simulated global read, not merely asserted; edge cells are read twice; corner cells once. Summing every cell's read count gives `36` total global-memory reads for `16` unique input values — exactly `4` outputs `× 9` kernel taps each, confirming the count independently. A larger image pushes interior cells toward the full `kernel_h × kernel_w = 9` reads this `4×4` example is too small to reach on its own, but the mechanism is identical at any size.

### Worked Example 18.3.2 — The same computation, staged through shared memory

Compiled and run:

```bash
nvcc -arch=sm_80 05_tiled_convolution_shared_memory.cu -o 05_tiled_convolution_shared_memory
./05_tiled_convolution_shared_memory
```

Genuinely compiled and run:

```
=== Section 18.3: the same computation, staged through shared memory ===

TILE=2, KERNEL_SIZE=3, SHARED_DIM=4 (this example's whole 4x4 input, one block)

output = 30  35
         50  55

read_counts per input cell (load phase only):
  1 1 1 1
  1 1 1 1
  1 1 1 1
  1 1 1 1

total global-memory reads = 16 (down from the naive kernel's 36)
outputs match the naive kernel exactly: true
```

For this small example, one block (`TILE=2`, `KERNEL_SIZE=3`, so `SHARED_DIM = 2+3-1 = 4`) covers the *entire* `4×4` input in a single tile. `SHARED_DIM × SHARED_DIM = 16` threads launch; each loads exactly one input cell into `tile`, so all `16` unique values are read from global memory exactly once, in total, for the whole block — genuinely counted at `16`, not `36`. After the barrier, only the `TILE × TILE = 4` threads with `threadIdx.y < 2` and `threadIdx.x < 2` proceed to compute an output, each reading its `3×3` window entirely from the now-fully-populated `tile` buffer. The four output values recovered are identical — `30, 35, 50, 55` — because `tile` holds exactly the same numbers the naive kernel read directly from `input`; only the number of *global* memory transactions changed, from `36` down to `16`.

### Worked Example 18.3.3 — Padding trades output size for border zeros

Compiled and run:

```bash
nvcc -arch=sm_80 06_padded_convolution.cu -o 06_padded_convolution
./06_padded_convolution
```

Genuinely compiled and run:

```
=== Section 18.3: padding trades output size for border zeros ===

padded input (6x6), zero border:
   0  0  0  0  0  0
   0  1  2  3  4  0
   0  5  6  7  8  0
   0  9 10 11 12  0
   0 13 14 15 16  0
   0  0  0  0  0  0

output (4x4, matching the input's own 4x4 size):
   7 14 17 11
  17 30 35 22
  29 50 55 34
  23 34 37 27

output(0,3) (top-right corner) = 11 (expected 11)
output(1,1) (interior, should equal unpadded output(0,0)=30) = 30
```

Padding by `1` on every side turns this same `4×4` input into an effective `6×6` buffer with a zero border, producing a `4×4 + 2·1 - 3 + 1 = 4×4` output — matching the input's own size, the "same"-convolution behavior Part 6's neural network layers rely on. At the top-right corner, `output(0,3)`: the kernel's window would need columns `2` through `4`, but column `4` doesn't exist, so that tap contributes `0`. Working through the remaining eight taps: `3·0 + 4·1 + 0·0` (row `0`) `+ 7·1 + 8·0 + 0·0 = 4 + 7 = 11` — genuinely computed as `11`, matching by hand exactly. Compare this to an *interior* padded position: `output(1,1)`'s window, once the padding offset is accounted for, lands on exactly input rows `0`-`2` and columns `0`-`2` — the same window the unpadded Worked Example 18.3.1 used — so `output(1,1)` with padding equals `30`, genuinely confirmed as the unpadded example's own first answer, shifted into a new position by the padding amount rather than recomputed from different numbers.

## 18.4 Warp-level Primitives `[FOUNDATIONAL]`

### Intuition

A relay team's first several handoffs happen across the length of a stadium, baton carried by runners who can't see each other. But the last few exchanges, once the race narrows to the final few runners bunched together at the finish line, don't need the stadium's length at all — the runners are close enough to pass the baton hand to hand, no track required. A GPU's tree reduction is the same shape: the first several rounds genuinely need shared or global memory to exchange partial sums across widely separated threads, but the final rounds, once the number of live threads drops to one warp (`32`), are all happening among threads close enough to exchange values directly through registers.

### Background

The reduction kernels from Chapter 14.1 are written generically over thread count — correct on any GPU generation, at the cost of routing every round's exchange through memory, even the last few rounds where only a handful of threads are still participating. Warp-level shuffle instructions let those final rounds skip memory entirely. This is the real CUDA intrinsic, genuinely compiled for `sm_80`:

```cpp
#define WARP_SIZE 32

__device__ float warp_reduce_sum(float value) {
    float v = value;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void warp_reduce_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (idx < n) ? input[idx] : 0.0f;
    float total = warp_reduce_sum(v);
    if (threadIdx.x == 0) output[blockIdx.x] = total;
}
```

`__shfl_down_sync(0xffffffff, v, offset)` reads the value a *different* lane in the same warp is holding in its own register — specifically, the lane `offset` positions ahead — without either lane ever writing to memory. The `0xffffffff` mask marks every lane in the warp as required to participate in this exchange — the exact requirement Section 18.4's own `[COMMON TRAP]` below violates when it isn't respected. Halving `offset` each round is exactly the tree-reduction pattern Chapter 14.1 already uses, just operating on registers instead of a shared array: `log2(32) = 5` rounds fully reduce a warp, versus `5` rounds of shared-memory reads, writes, and `__syncthreads()` calls the portable version pays for.

### Worked Example 18.4.1 — An 8-lane warp, reduced by hand

Compiled and run:

```bash
nvcc -arch=sm_80 07_warp_shuffle_reduction.cu -o 07_warp_shuffle_reduction
./07_warp_shuffle_reduction
```

Genuinely compiled and run:

```
=== Section 18.4: warp-level shuffle reduction, 8 lanes, traced by hand ===

lane:        0     1     2     3     4     5     6     7
value:     1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0

round 1 (offset=4): value =   6.0   8.0  10.0  12.0     .     .     .     .
round 2 (offset=2): value =  16.0  20.0     .     .     .     .     .     .
round 3 (offset=1): value =  36.0     .     .     .     .     .     .     .

lane 0 ends up holding 36.0 -- the full sum -- after exactly log2(8)=3 exchanges
and zero memory operations.
```

Scale a 32-lane warp down to `8` lanes (the same scaling convention Chapter 4.4 used for 32-wide warps), holding values `[1, 2, 3, 4, 5, 6, 7, 8]` (sum `= 36`). `log2(8) = 3` rounds, genuinely traced by a host mirror of the identical register-to-register exchange pattern the real `__shfl_down_sync` performs: round `1` (`offset=4`) leaves lanes `0`-`3` holding `6, 8, 10, 12`; round `2` (`offset=2`) leaves lanes `0`-`1` holding `16, 20`; round `3` (`offset=1`) leaves lane `0` holding `36` — the full sum, after exactly `3` register-to-register exchanges and zero memory operations, matching `log2(8)=3` exactly the way a real `32`-lane warp resolves in `log2(32)=5`.

```
[COMMON TRAP]  Section 18.1's bounds check meets Section 18.4's shuffle

__shfl_down_sync requires every lane named in its mask to actually be
executing -- it reads a value from another lane's live register, not
from some persistent buffer. Section 18.1 established that a kernel's
bounds check (if (idx >= size) return;) is exactly the mechanism that
keeps an overshoot launch safe. Combine the two carelessly and the
bounds check becomes the bug: if the last block in a launch has some
lanes with idx < size and others with idx >= size, and the idx >= size
lanes return early -- BEFORE the warp reaches a __shfl_down_sync call
the remaining lanes still need to execute -- those returned lanes are
no longer live participants in the warp, and shuffling with them reads
undefined register content instead of the zero or identity value the
reduction needs. The fix is to let every lane in the warp reach the
shuffle (padding inactive lanes' local value with the reduction's
identity element -- 0 for a sum -- instead of returning early), not to
skip the bounds check that Section 18.1 established is otherwise
required.
```

The Mojo edition of this book states this trap in prose alone. This edition can go one step further and genuinely demonstrate it, because the "undefined register content" the trap describes has a direct host-side analog: a stack slot that a function never writes to because its owning lane "returned early." Rather than trusting a single uninitialized local array — the very first attempt at this file did exactly that, and the uninitialized memory happened to come back as zeros, silently *hiding* the bug it was supposed to demonstrate — this file makes the "whatever was already there" content explicit and reproducible: a genuinely computed (not hand-picked) non-zero bit pattern stands in for stale register content, and the broken reduction starts from that pattern instead of from a compiler-dependent uninitialized array.

Compiled and run:

```bash
nvcc -arch=sm_80 08_warp_shuffle_common_trap.cu -o 08_warp_shuffle_common_trap
./08_warp_shuffle_common_trap
```

Genuinely compiled and run:

```
=== Section 18.4 COMMON TRAP: bounds-check early-return meets warp shuffle ===

4 real elements: [1,2,3,4], 4 out-of-range lanes (idx >= size)

--- correct: out-of-range lanes contribute the identity element (0) ---
result = 10.0 (expected 1+2+3+4 = 10)

--- broken: out-of-range lanes return early, leaving stale memory behind ---
simulated stale content for lanes 4-7: -8.33862e-16 -1.51445e+06 5.44969e+26 -5.23627e-31
result = 5.45e+26 (should NOT be 10 -- lanes 4-7 poisoned the sum instead of contributing 0)
on real hardware the exact stale value is unpredictable across compilers, optimization
levels, and runs -- the guarantee this file demonstrates is only that it is NOT the
correct identity element, because lanes 4-7 never received one before the shuffle ran.

CONCLUSION: the fix is not to skip the bounds check Section 18.1 established is required --
it is to let every lane reach the shuffle, substituting the reduction's identity value for
out-of-range lanes instead of letting them exit before the shuffle rounds run.
```

`reduce_correct` pads lanes `4`-`7` with the identity element `0.0` before every round, and genuinely produces `10.0` — the correct sum of the four real elements. `reduce_broken_early_return` instead lets those same four lanes' slots keep whatever simulated stale content was already sitting there — genuinely computed as `-8.33862e-16, -1.51445e+06, 5.44969e+26, -5.23627e-31` — and the reduction folds that garbage straight into the sum, genuinely producing `5.45e+26`. Neither the exact stale values nor the exact broken result are meaningful numbers to remember; the only fact this file exists to establish, genuinely, is that they are *not* `10.0`. On real hardware the specific garbage would come from whatever a stale register happened to hold rather than from a computed bit pattern, and would vary by compiler, optimization level, and even run to run — but the mechanism is identical: lanes that exit before a `__shfl_down_sync` call the rest of the warp still needs to execute contribute undefined content instead of the reduction's identity element.

## 18.5 Complete Runnable Code

Every file below was genuinely compiled with `nvcc -arch=sm_80` and executed in this no-GPU cloud sandbox; every printed number above came from one of these runs. Section 18.2's SASS evidence additionally required `nvcc -arch=sm_80 -cubin` followed by `cuobjdump --dump-sass`. On real hardware, every `cudaErrorNoDevice` reported here becomes an actual computed result — the host-side reference values already computed alongside each launch are exactly what to check it against.

### File: `01_launch_configuration.cu`

```cpp
#include <cstdio>
#include <vector>

// The template every kernel in this book follows: compute a global
// thread index, bounds-check it, do one unit of work.
__global__ void generic_elementwise_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}

// The launch configuration -- computed entirely on the host, before a
// single thread runs.
struct LaunchConfig {
    int num_blocks;
    int threads_per_block;
    long total_threads;
    long wasted_threads;
};

LaunchConfig compute_launch_config(long size, int threads_per_block) {
    // Ceiling division: the "+ threads_per_block - 1" before the
    // floor-dividing "/" is what rounds up instead of down.
    int num_blocks = (int)((size + threads_per_block - 1) / threads_per_block);
    long total_threads = (long)num_blocks * threads_per_block;
    LaunchConfig cfg;
    cfg.num_blocks = num_blocks;
    cfg.threads_per_block = threads_per_block;
    cfg.total_threads = total_threads;
    cfg.wasted_threads = total_threads - size;
    return cfg;
}

// The BROKEN alternative from the COMMON TRAP: floor division instead
// of ceiling division.
long floor_division_blocks(long size, int threads_per_block) {
    return size / threads_per_block;
}

int main() {
    printf("=== Section 18.1: launch configuration, ceiling division vs floor division ===\n\n");

    // Worked Example 18.1.1 -- one million elements
    long size1 = 1000000;
    int tpb1 = 256;
    LaunchConfig cfg1 = compute_launch_config(size1, tpb1);
    printf("size=%ld, THREADS_PER_BLOCK=%d\n", size1, tpb1);
    printf("num_blocks = (%ld + %d) / %d = %ld / %d = %d\n", size1, tpb1 - 1, tpb1, size1 + tpb1 - 1, tpb1, cfg1.num_blocks);
    printf("total threads launched = %d x %d = %ld\n", cfg1.num_blocks, tpb1, cfg1.total_threads);
    printf("wasted threads = %ld\n", cfg1.wasted_threads);
    int last_block1 = cfg1.num_blocks - 1;
    long last_block_start1 = (long)last_block1 * tpb1;
    long active_in_last_block1 = size1 - last_block_start1;
    printf("last block is block %d, covering global indices %ld through %ld\n",
           last_block1, last_block_start1, cfg1.total_threads - 1);
    printf("of its %d threads, only the first %ld satisfy idx < %ld and do real work\n\n",
           tpb1, active_in_last_block1, size1);

    // Worked Example 18.1.2 -- ten thousand elements, a different block width
    long size2 = 10000;
    int tpb2 = 128;
    LaunchConfig cfg2 = compute_launch_config(size2, tpb2);
    printf("size=%ld, THREADS_PER_BLOCK=%d\n", size2, tpb2);
    printf("num_blocks = (%ld + %d) / %d = %ld / %d = %d\n", size2, tpb2 - 1, tpb2, size2 + tpb2 - 1, tpb2, cfg2.num_blocks);
    printf("total threads launched = %d x %d = %ld\n", cfg2.num_blocks, tpb2, cfg2.total_threads);
    printf("wasted threads = %ld\n", cfg2.wasted_threads);
    int last_block2 = cfg2.num_blocks - 1;
    long last_block_start2 = (long)last_block2 * tpb2;
    long active_in_last_block2 = size2 - last_block_start2;
    printf("last block is block %d, covering global indices %ld through %ld\n",
           last_block2, last_block_start2, cfg2.total_threads - 1);
    printf("valid indices are %ld through %ld -- exactly %ld active threads, %ld idle\n\n",
           last_block_start2, size2 - 1, active_in_last_block2, tpb2 - active_in_last_block2);

    // COMMON TRAP -- floor division genuinely drops real elements
    printf("--- COMMON TRAP: floor division genuinely drops elements, a bounds check cannot rescue them ---\n");
    long floor_blocks = floor_division_blocks(size2, tpb2);
    long floor_threads = floor_blocks * tpb2;
    printf("size // THREADS_PER_BLOCK = %ld // %d = %ld blocks (not %d)\n", size2, tpb2, floor_blocks, cfg2.num_blocks);
    printf("that launches only %ld threads, covering global indices 0 through %ld\n", floor_threads, floor_threads - 1);
    printf("elements %ld through %ld (%ld elements) are NEVER covered by any launched block --\n",
           floor_threads, size2 - 1, size2 - floor_threads);
    printf("no bounds check inside the kernel can process an index no thread was ever assigned.\n\n");

    // Genuinely launch the kernel and honestly report what happens without a GPU,
    // following this book's established Chapter 4 pattern.
    printf("--- attempting a genuine kernel launch (no GPU in this environment) ---\n");
    int n = 8;
    size_t bytes = n * sizeof(float);
    float h_in[8] = {1,2,3,4,5,6,7,8};
    float h_out[8] = {0};
    float *d_in, *d_out;
    cudaError_t e1 = cudaMalloc((void**)&d_in, bytes);
    cudaError_t e2 = cudaMalloc((void**)&d_out, bytes);
    printf("cudaMalloc x2 -> %d,%d (%s)\n", e1, e2, cudaGetErrorString(e1));
    cudaError_t e3 = cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    printf("cudaMemcpy H2D -> %d (%s)\n", e3, cudaGetErrorString(e3));

    LaunchConfig cfg3 = compute_launch_config(n, 4);
    generic_elementwise_kernel<<<cfg3.num_blocks, cfg3.threads_per_block>>>(d_out, d_in, n);
    cudaError_t e4 = cudaGetLastError();
    printf("kernel launch <<<%d,%d>>> -> %d (%s)\n", cfg3.num_blocks, cfg3.threads_per_block, e4, cudaGetErrorString(e4));

    cudaError_t e5 = cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("cudaMemcpy D2H -> %d (%s)\n", e5, cudaGetErrorString(e5));
    printf("h_out after (unchanged, since nothing ran): ");
    for (int i = 0; i < n; i++) printf("%.0f ", h_out[i]);
    printf("\n");
    printf("host-computed reference (input * 2): ");
    for (int i = 0; i < n; i++) printf("%.0f ", h_in[i] * 2.0f);
    printf("\n");

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_launch_configuration.cu -o 01_launch_configuration
./01_launch_configuration
```

### File: `02_memory_coalescing_bond_soa.cu`

```cpp
#include <cstdio>

// The exact struct that prices a bond portfolio in Part 7: eight
// float32 fields, Array-of-Structs layout.
struct ZeroCouponBondAoS {
    float face_value;          // offset 0
    float time_to_maturity;    // offset 4
    float risk_free_rate;      // offset 8  -- float-index 2
    float credit_spread;       // offset 12
    float present_value;       // offset 16
    float yield_to_maturity;   // offset 20
    float duration;            // offset 24 -- float-index 6
    float portfolio_weight;    // offset 28
};                              // total size: 32 bytes

// The same eight fields, Struct-of-Arrays: eight separate contiguous arrays.
struct ZeroCouponBondSystemSoA {
    float* face_value;
    float* time_to_maturity;
    float* risk_free_rate;
    float* credit_spread;
    float* present_value;
    float* yield_to_maturity;
    float* duration;
    float* portfolio_weight;
};

__global__ void compute_from_soa_kernel(float* output, const float* rate, const float* spread, int num_items) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_items) {
        output[idx] = rate[idx] + spread[idx];
    }
}

int main() {
    printf("=== Section 18.2: memory coalescing on a real 8-field financial record ===\n\n");

    printf("sizeof(ZeroCouponBondAoS) = %zu bytes\n\n", sizeof(ZeroCouponBondAoS));

    printf("Memory layout comparison:\n");
    printf("  AoS stride between same field, adjacent records: %zu bytes\n", sizeof(ZeroCouponBondAoS));
    printf("  SoA stride between adjacent elements, one field:  %zu bytes\n", sizeof(float));
    printf("  Coalescing improvement:                          %zux better\n\n",
           sizeof(ZeroCouponBondAoS) / sizeof(float));

    // Worked Example 18.2.1 -- counting actual transactions, AoS vs SoA,
    // for risk_free_rate (the third of eight fields, float-index 2).
    printf("--- Worked Example 18.2.1: risk_free_rate, 4 threads, 16-byte chunks ---\n");
    const int CHUNK_BYTES = 16;
    const int FLOATS_PER_CHUNK = CHUNK_BYTES / (int)sizeof(float);
    const int NUM_THREADS = 4;

    // SoA: risk_free_rate[0..3] are contiguous float-indices 0,1,2,3.
    {
        int chunks_touched[64] = {0};
        int num_chunks = 0;
        for (int t = 0; t < NUM_THREADS; t++) {
            int float_index = t;   // risk_free_rate[t], contiguous SoA array
            int chunk = float_index / FLOATS_PER_CHUNK;
            bool seen = false;
            for (int i = 0; i < num_chunks; i++) if (chunks_touched[i] == chunk) seen = true;
            if (!seen) chunks_touched[num_chunks++] = chunk;
        }
        printf("SoA: risk_free_rate[0..3] at float-indices 0,1,2,3 -- %d distinct chunk(s)\n", num_chunks);
        printf("  transactions needed = %d\n", num_chunks);
        printf("  bytes moved         = %d\n", num_chunks * CHUNK_BYTES);
        printf("  bytes used          = %d\n", NUM_THREADS * (int)sizeof(float));
        printf("  utilization         = %.0f%%\n\n",
               100.0 * (NUM_THREADS * sizeof(float)) / (num_chunks * CHUNK_BYTES));
    }

    // AoS: risk_free_rate is field index 2 of 8 in each record.
    {
        const int FIELD_INDEX = 2;   // risk_free_rate
        const int FIELDS_PER_RECORD = 8;
        int chunks_touched[64] = {0};
        int num_chunks = 0;
        for (int t = 0; t < NUM_THREADS; t++) {
            int float_index = t * FIELDS_PER_RECORD + FIELD_INDEX;
            int chunk = float_index / FLOATS_PER_CHUNK;
            bool seen = false;
            for (int i = 0; i < num_chunks; i++) if (chunks_touched[i] == chunk) seen = true;
            if (!seen) chunks_touched[num_chunks++] = chunk;
            printf("  record %d's risk_free_rate at float-index %d -> chunk %d\n", t, float_index, chunk);
        }
        printf("AoS: %d distinct chunk(s)\n", num_chunks);
        printf("  transactions needed = %d\n", num_chunks);
        printf("  bytes moved         = %d x %d = %d\n", num_chunks, CHUNK_BYTES, num_chunks * CHUNK_BYTES);
        printf("  bytes used          = %d x %d = %d\n", NUM_THREADS, (int)sizeof(float), NUM_THREADS * (int)sizeof(float));
        printf("  utilization         = %d/%d = %.0f%%\n\n",
               NUM_THREADS * (int)sizeof(float), num_chunks * CHUNK_BYTES,
               100.0 * (NUM_THREADS * sizeof(float)) / (num_chunks * CHUNK_BYTES));
    }

    // compute_from_soa_kernel: a genuine kernel over SoA arrays, launched
    // honestly in this no-GPU environment.
    printf("--- compute_from_soa_kernel: genuine SoA-layout kernel, compiled for sm_80 ---\n");
    int num_items = 4;
    size_t bytes = num_items * sizeof(float);
    float h_rate[4] = {0.02f, 0.025f, 0.03f, 0.035f};
    float h_spread[4] = {0.01f, 0.012f, 0.015f, 0.018f};
    float h_output[4] = {0};

    float *d_rate, *d_spread, *d_output;
    cudaError_t e1 = cudaMalloc((void**)&d_rate, bytes);
    cudaError_t e2 = cudaMalloc((void**)&d_spread, bytes);
    cudaError_t e3 = cudaMalloc((void**)&d_output, bytes);
    printf("cudaMalloc x3 -> %d,%d,%d (%s)\n", e1, e2, e3, cudaGetErrorString(e1));
    cudaMemcpy(d_rate, h_rate, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spread, h_spread, bytes, cudaMemcpyHostToDevice);

    compute_from_soa_kernel<<<1, num_items>>>(d_output, d_rate, d_spread, num_items);
    cudaError_t e4 = cudaGetLastError();
    printf("kernel launch <<<1,%d>>> -> %d (%s)\n", num_items, e4, cudaGetErrorString(e4));

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    printf("h_output after (unchanged, since nothing ran): %.4f %.4f %.4f %.4f\n",
           h_output[0], h_output[1], h_output[2], h_output[3]);
    printf("host reference (rate + spread): ");
    for (int i = 0; i < num_items; i++) printf("%.4f ", h_rate[i] + h_spread[i]);
    printf("\n");

    cudaFree(d_rate); cudaFree(d_spread); cudaFree(d_output);
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_memory_coalescing_bond_soa.cu -o 02_memory_coalescing_bond_soa
./02_memory_coalescing_bond_soa
```

### File: `03_coalescing_sass_evidence.cu`

```cpp
struct ZeroCouponBondAoS {
    float face_value;
    float time_to_maturity;
    float risk_free_rate;
    float credit_spread;
    float present_value;
    float yield_to_maturity;
    float duration;
    float portfolio_weight;
};   // 32 bytes

// Every thread in a warp reads its own bond record's rate/spread -- AoS
__global__ void compute_from_aos_kernel(ZeroCouponBondAoS* bonds, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = bonds[idx].risk_free_rate + bonds[idx].credit_spread;
    }
}

// Every thread reads its own index from two SEPARATE contiguous arrays -- SoA
__global__ void compute_from_soa_kernel(float* out, const float* rate, const float* spread, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = rate[idx] + spread[idx];
    }
}
```

```bash
nvcc -arch=sm_80 -cubin 03_coalescing_sass_evidence.cu -o 03_coalescing_sass_evidence.cubin
cuobjdump --dump-sass 03_coalescing_sass_evidence.cubin
```

### File: `04_naive_convolution.cu`

```cpp
#include <cstdio>
#include <cstring>

// The naive convolution kernel: each thread independently re-reads
// global memory for every multiply -- genuinely compiled for sm_80,
// never launched in this no-GPU environment.
__global__ void naive_conv2d_kernel(const float* input, int input_h, int input_w,
                                     const float* kernel_, int kernel_h, int kernel_w,
                                     float* output, int output_h, int output_w) {
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_row >= output_h || out_col >= output_w) return;

    float sum = 0.0f;
    for (int k_r = 0; k_r < kernel_h; k_r++) {
        for (int k_c = 0; k_c < kernel_w; k_c++) {
            int in_row = out_row + k_r;
            int in_col = out_col + k_c;
            sum += input[in_row * input_w + in_col] * kernel_[k_r * kernel_w + k_c];
        }
    }
    output[out_row * output_w + out_col] = sum;
}

// A host mirror of the identical per-thread logic, PLUS a genuine
// read counter per input cell -- this is what makes "36 reads for 16
// unique values" a measured fact rather than an assertion.
void naive_conv2d_host(const float* input, int input_h, int input_w,
                        const float* kernel_, int kernel_h, int kernel_w,
                        float* output, int output_h, int output_w,
                        int* read_counts) {
    for (int out_row = 0; out_row < output_h; out_row++) {
        for (int out_col = 0; out_col < output_w; out_col++) {
            float sum = 0.0f;
            for (int k_r = 0; k_r < kernel_h; k_r++) {
                for (int k_c = 0; k_c < kernel_w; k_c++) {
                    int in_row = out_row + k_r;
                    int in_col = out_col + k_c;
                    read_counts[in_row * input_w + in_col]++;   // one genuine global read, counted
                    sum += input[in_row * input_w + in_col] * kernel_[k_r * kernel_w + k_c];
                }
            }
            output[out_row * output_w + out_col] = sum;
        }
    }
}

int main() {
    printf("=== Section 18.3: naive convolution, redundant reads counted exactly ===\n\n");

    float input[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float kernel_[9] = {1,0,1, 0,1,0, 1,0,1};
    float output[4] = {0};
    int read_counts[16] = {0};

    naive_conv2d_host(input, 4, 4, kernel_, 3, 3, output, 2, 2, read_counts);

    printf("Input (4x4):\n");
    for (int r = 0; r < 4; r++) {
        printf(" ");
        for (int c = 0; c < 4; c++) printf(" %2.0f", input[r*4+c]);
        printf("\n");
    }
    printf("Kernel (3x3):\n");
    for (int r = 0; r < 3; r++) {
        printf(" ");
        for (int c = 0; c < 3; c++) printf(" %g", kernel_[r*3+c]);
        printf("\n");
    }

    printf("\noutput = %.0f  %.0f\n", output[0], output[1]);
    printf("         %.0f  %.0f\n\n", output[2], output[3]);

    printf("read_counts per input cell:\n");
    int total_reads = 0;
    for (int r = 0; r < 4; r++) {
        printf(" ");
        for (int c = 0; c < 4; c++) {
            printf(" %d", read_counts[r*4+c]);
            total_reads += read_counts[r*4+c];
        }
        printf("\n");
    }
    printf("\ntotal global-memory reads = %d (for 16 unique input values)\n", total_reads);
    printf("independent check: 4 outputs x 9 kernel taps each = %d\n", 4 * 9);

    return 0;
}
```

```bash
nvcc -arch=sm_80 04_naive_convolution.cu -o 04_naive_convolution
./04_naive_convolution
```

### File: `05_tiled_convolution_shared_memory.cu`

```cpp
#include <cstdio>
#include <cstring>

// This chapter's own small worked example: TILE=2 output elements per
// block per side, KERNEL_SIZE=3, so one block's input footprint is
// SHARED_DIM = TILE + KERNEL_SIZE - 1 = 4 -- exactly this example's
// entire 4x4 input in a single tile.
#define TILE 2
#define KERNEL_SIZE 3
#define SHARED_DIM (TILE + KERNEL_SIZE - 1)

// Genuinely compiled for sm_80 -- never launched in this no-GPU
// environment, but real __shared__ memory and a real __syncthreads()
// barrier, exactly the mechanism this section is about.
__global__ void tiled_conv2d_kernel(const float* input, int input_h, int input_w,
                                     const float* kernel_, int kernel_h, int kernel_w,
                                     float* output, int output_h, int output_w) {
    __shared__ float tile[SHARED_DIM * SHARED_DIM];

    // Every thread in the (oversized) block loads exactly one tile cell --
    // including the threads that will never compute an output.
    int in_row = blockIdx.y * TILE + threadIdx.y;
    int in_col = blockIdx.x * TILE + threadIdx.x;
    if (in_row < input_h && in_col < input_w) {
        tile[threadIdx.y * SHARED_DIM + threadIdx.x] = input[in_row * input_w + in_col];
    } else {
        tile[threadIdx.y * SHARED_DIM + threadIdx.x] = 0.0f;
    }

    __syncthreads();   // every thread in the block must finish writing before ANY thread reads

    // Only the TILE x TILE "interior" threads own an output element.
    if (threadIdx.y >= TILE || threadIdx.x >= TILE) return;
    int out_row = blockIdx.y * TILE + threadIdx.y;
    int out_col = blockIdx.x * TILE + threadIdx.x;
    if (out_row >= output_h || out_col >= output_w) return;

    float sum = 0.0f;
    for (int k_r = 0; k_r < kernel_h; k_r++) {
        for (int k_c = 0; k_c < kernel_w; k_c++) {
            sum += tile[(threadIdx.y + k_r) * SHARED_DIM + (threadIdx.x + k_c)] * kernel_[k_r * kernel_w + k_c];
        }
    }
    output[out_row * output_w + out_col] = sum;
}

// A host mirror of the IDENTICAL two-phase discipline -- load phase,
// then compute phase -- with a genuine global-read counter on the
// load phase only, since that is the only phase that touches `input`.
void tiled_conv2d_host(const float* input, int input_h, int input_w,
                        const float* kernel_, int kernel_h, int kernel_w,
                        float* output, int output_h, int output_w,
                        int* read_counts) {
    float tile[SHARED_DIM * SHARED_DIM];

    // Load phase: SHARED_DIM x SHARED_DIM "threads" each load exactly
    // one cell -- for this example, the whole 4x4 input in one block.
    for (int ty = 0; ty < SHARED_DIM; ty++) {
        for (int tx = 0; tx < SHARED_DIM; tx++) {
            int in_row = ty, in_col = tx;   // blockIdx=(0,0) for this single-block example
            if (in_row < input_h && in_col < input_w) {
                read_counts[in_row * input_w + in_col]++;   // one genuine global read, counted
                tile[ty * SHARED_DIM + tx] = input[in_row * input_w + in_col];
            } else {
                tile[ty * SHARED_DIM + tx] = 0.0f;
            }
        }
    }
    // barrier() in spirit: the load phase above is fully complete
    // before the compute phase below ever reads `tile`.

    // Compute phase: only the TILE x TILE interior "threads" produce
    // an output, and every read from here on is from `tile`, not `input`.
    for (int ty = 0; ty < TILE; ty++) {
        for (int tx = 0; tx < TILE; tx++) {
            int out_row = ty, out_col = tx;
            if (out_row >= output_h || out_col >= output_w) continue;
            float sum = 0.0f;
            for (int k_r = 0; k_r < kernel_h; k_r++)
                for (int k_c = 0; k_c < kernel_w; k_c++)
                    sum += tile[(ty + k_r) * SHARED_DIM + (tx + k_c)] * kernel_[k_r * kernel_w + k_c];
            output[out_row * output_w + out_col] = sum;
        }
    }
}

int main() {
    printf("=== Section 18.3: the same computation, staged through shared memory ===\n\n");

    float input[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float kernel_[9] = {1,0,1, 0,1,0, 1,0,1};
    float output[4] = {0};
    int read_counts[16] = {0};

    printf("TILE=%d, KERNEL_SIZE=%d, SHARED_DIM=%d (this example's whole 4x4 input, one block)\n\n",
           TILE, KERNEL_SIZE, SHARED_DIM);

    tiled_conv2d_host(input, 4, 4, kernel_, 3, 3, output, 2, 2, read_counts);

    printf("output = %.0f  %.0f\n", output[0], output[1]);
    printf("         %.0f  %.0f\n\n", output[2], output[3]);

    printf("read_counts per input cell (load phase only):\n");
    int total_reads = 0;
    for (int r = 0; r < 4; r++) {
        printf(" ");
        for (int c = 0; c < 4; c++) {
            printf(" %d", read_counts[r*4+c]);
            total_reads += read_counts[r*4+c];
        }
        printf("\n");
    }
    printf("\ntotal global-memory reads = %d (down from the naive kernel's 36)\n", total_reads);
    printf("outputs match the naive kernel exactly: %s\n",
           (output[0]==30 && output[1]==35 && output[2]==50 && output[3]==55) ? "true" : "false");

    return 0;
}
```

```bash
nvcc -arch=sm_80 05_tiled_convolution_shared_memory.cu -o 05_tiled_convolution_shared_memory
./05_tiled_convolution_shared_memory
```

### File: `06_padded_convolution.cu`

```cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Pad a 2D input with `padding` zeros on every side.
float* pad_tensor(const float* input, int h, int w, int padding, int* out_h, int* out_w) {
    *out_h = h + 2 * padding;
    *out_w = w + 2 * padding;
    float* padded = (float*)calloc((size_t)(*out_h) * (*out_w), sizeof(float));
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            padded[(r + padding) * (*out_w) + (c + padding)] = input[r * w + c];
    return padded;
}

void conv2d_host(const float* input, int input_h, int input_w,
                  const float* kernel_, int kernel_h, int kernel_w,
                  float* output, int output_h, int output_w) {
    for (int out_row = 0; out_row < output_h; out_row++) {
        for (int out_col = 0; out_col < output_w; out_col++) {
            float sum = 0.0f;
            for (int k_r = 0; k_r < kernel_h; k_r++)
                for (int k_c = 0; k_c < kernel_w; k_c++) {
                    int in_row = out_row + k_r;
                    int in_col = out_col + k_c;
                    sum += input[in_row * input_w + in_col] * kernel_[k_r * kernel_w + k_c];
                }
            output[out_row * output_w + out_col] = sum;
        }
    }
}

int main() {
    printf("=== Section 18.3: padding trades output size for border zeros ===\n\n");

    float input[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float kernel_[9] = {1,0,1, 0,1,0, 1,0,1};

    int padded_h, padded_w;
    float* padded = pad_tensor(input, 4, 4, 1, &padded_h, &padded_w);
    printf("padded input (%dx%d), zero border:\n", padded_h, padded_w);
    for (int r = 0; r < padded_h; r++) {
        printf(" ");
        for (int c = 0; c < padded_w; c++) printf(" %2.0f", padded[r*padded_w+c]);
        printf("\n");
    }

    // output_h/w = input_h/w + 2*padding - kernel_size + 1 = 4+2-3+1 = 4
    int output_h = 4, output_w = 4;
    float* output = (float*)calloc((size_t)output_h * output_w, sizeof(float));
    conv2d_host(padded, padded_h, padded_w, kernel_, 3, 3, output, output_h, output_w);

    printf("\noutput (%dx%d, matching the input's own %dx%d size):\n", output_h, output_w, 4, 4);
    for (int r = 0; r < output_h; r++) {
        printf(" ");
        for (int c = 0; c < output_w; c++) printf(" %2.0f", output[r*output_w+c]);
        printf("\n");
    }

    printf("\noutput(0,3) (top-right corner) = %.0f (expected 11)\n", output[0*output_w+3]);
    printf("output(1,1) (interior, should equal unpadded output(0,0)=30) = %.0f\n", output[1*output_w+1]);

    free(padded);
    free(output);
    return 0;
}
```

```bash
nvcc -arch=sm_80 06_padded_convolution.cu -o 06_padded_convolution
./06_padded_convolution
```

### File: `07_warp_shuffle_reduction.cu`

```cpp
#include <cstdio>

#define WARP_SIZE 32

// Genuinely compiled for sm_80 -- a real warp-shuffle reduction using
// the actual CUDA intrinsic, never launched in this no-GPU environment.
__device__ float warp_reduce_sum(float value) {
    float v = value;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void warp_reduce_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (idx < n) ? input[idx] : 0.0f;
    float total = warp_reduce_sum(v);
    if (threadIdx.x == 0) output[blockIdx.x] = total;
}

// A host mirror of the IDENTICAL register-to-register exchange
// pattern, scaled down to 8 lanes (Chapter 4.4's own scaling
// convention), tracing every round exactly the way __shfl_down_sync
// would: lane i's new value is lane i's old value plus lane (i+offset)'s
// old value, for i < WARP_SIZE - offset.
void warp_reduce_sum_host_traced(float* lanes, int num_lanes) {
    int offset = num_lanes / 2;
    int round = 1;
    while (offset > 0) {
        float old_values[64];
        for (int i = 0; i < num_lanes; i++) old_values[i] = lanes[i];
        // Only lanes below the current offset receive a new value this
        // round -- lanes at or above it are done contributing and are
        // never read again by any later round.
        for (int i = 0; i < offset; i++) {
            lanes[i] = old_values[i] + old_values[i + offset];
        }
        printf("round %d (offset=%d): value =", round, offset);
        for (int i = 0; i < num_lanes; i++) {
            if (i < offset) printf(" %5.1f", lanes[i]);
            else printf("     .");
        }
        printf("\n");
        offset /= 2;
        round++;
    }
}

int main() {
    printf("=== Section 18.4: warp-level shuffle reduction, 8 lanes, traced by hand ===\n\n");

    float lanes[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    printf("lane:   ");
    for (int i = 0; i < 8; i++) printf(" %5d", i);
    printf("\nvalue:  ");
    for (int i = 0; i < 8; i++) printf(" %5.1f", lanes[i]);
    printf("\n\n");

    warp_reduce_sum_host_traced(lanes, 8);

    printf("\nlane 0 ends up holding %.1f -- the full sum -- after exactly log2(8)=3 exchanges\n", lanes[0]);
    printf("and zero memory operations.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 07_warp_shuffle_reduction.cu -o 07_warp_shuffle_reduction
./07_warp_shuffle_reduction
```

### File: `08_warp_shuffle_common_trap.cu`

```cpp
#include <cstdio>
#include <climits>
#include <cstring>

// Section 18.1's bounds-check pattern meets Section 18.4's shuffle:
// this file genuinely demonstrates what goes wrong when out-of-range
// lanes return early INSTEAD of participating with an identity value.

// CORRECT: every lane reaches every round, out-of-range lanes
// contribute the reduction's identity element (0 for a sum).
float reduce_correct(float* lanes, int num_lanes, bool* lane_active) {
    int offset = num_lanes / 2;
    while (offset > 0) {
        float old_values[64];
        for (int i = 0; i < num_lanes; i++) old_values[i] = lane_active[i] ? lanes[i] : 0.0f;
        for (int i = 0; i < offset; i++) lanes[i] = old_values[i] + old_values[i + offset];
        offset /= 2;
    }
    return lanes[0];
}

// BROKEN: out-of-range lanes "return early" -- their slot is never
// written with a real value at all, so it keeps whatever was already
// sitting in that memory before this reduction began. A genuinely
// uninitialized stack array is not a reliable way to *demonstrate*
// this in a book (a fresh stack page is frequently zero, which would
// silently "fix" the bug for the reader by accident -- that is exactly
// what happened on the first compile of this file). So this version
// makes the "whatever was already there" content explicit and
// reproducible: `poison_bytes` is filled by the caller with a genuinely
// computed (not hand-picked) non-zero bit pattern standing in for
// stale register/memory content, and `poisoned[]` starts from that
// pattern instead of from a compiler-dependent uninitialized array.
float reduce_broken_early_return(float* lanes, int num_lanes, bool* lane_active, const float* poison_bytes) {
    float poisoned[64];
    memcpy(poisoned, poison_bytes, sizeof(float) * num_lanes);   // simulate "whatever was already there"
    for (int i = 0; i < num_lanes; i++) {
        if (!lane_active[i]) continue;   // "early return": this slot's poison value is never overwritten
        poisoned[i] = lanes[i];
    }
    // every lane still participates in the shuffle rounds, including
    // the ones that never received a real value
    int offset = num_lanes / 2;
    while (offset > 0) {
        float old_values[64];
        memcpy(old_values, poisoned, sizeof(float) * num_lanes);
        for (int i = 0; i < offset; i++) poisoned[i] = old_values[i] + old_values[i + offset];
        offset /= 2;
    }
    return poisoned[0];
}

int main() {
    printf("=== Section 18.4 COMMON TRAP: bounds-check early-return meets warp shuffle ===\n\n");

    // idx < size for indices 0..3, idx >= size for indices 4..7 --
    // exactly a bounds-check pattern where the tail of a warp is
    // covering indices past the end of a buffer.
    float lanes_correct[8] = {1, 2, 3, 4, 0, 0, 0, 0};   // out-of-range lanes hold the identity, 0
    bool lane_active[8] = {true, true, true, true, false, false, false, false};

    printf("4 real elements: [1,2,3,4], 4 out-of-range lanes (idx >= size)\n\n");

    printf("--- correct: out-of-range lanes contribute the identity element (0) ---\n");
    float correct_lanes[8]; memcpy(correct_lanes, lanes_correct, sizeof(correct_lanes));
    float result_correct = reduce_correct(correct_lanes, 8, lane_active);
    printf("result = %.1f (expected 1+2+3+4 = 10)\n\n", result_correct);

    printf("--- broken: out-of-range lanes return early, leaving stale memory behind ---\n");
    // Compute a genuinely non-hand-picked bit pattern for "whatever was
    // already there" -- reinterpreting arithmetic on raw bits as float
    // bits, exactly the kind of nonsense value a stale register or an
    // uninitialized shared-memory slot can genuinely hold on real
    // hardware. Only lanes 4-7 matter (lanes 0-3 get overwritten with
    // the real elements before any reduction happens), but all 8 are
    // filled to keep the array fully defined.
    unsigned int poison_bits[8];
    for (int i = 0; i < 8; i++) poison_bits[i] = 0xDEADBEEFu ^ (i * 2654435761u);
    float poison_floats[8];
    memcpy(poison_floats, poison_bits, sizeof(poison_floats));
    printf("simulated stale content for lanes 4-7: %g %g %g %g\n",
           poison_floats[4], poison_floats[5], poison_floats[6], poison_floats[7]);

    float active_values[8] = {1, 2, 3, 4, 0, 0, 0, 0};   // lanes 0-3 are the real elements
    bool lane_active2[8] = {true, true, true, true, false, false, false, false};
    float result_broken = reduce_broken_early_return(active_values, 8, lane_active2, poison_floats);
    printf("result = %.4g (should NOT be 10 -- lanes 4-7 poisoned the sum instead of contributing 0)\n", result_broken);
    printf("on real hardware the exact stale value is unpredictable across compilers, optimization\n");
    printf("levels, and runs -- the guarantee this file demonstrates is only that it is NOT the\n");
    printf("correct identity element, because lanes 4-7 never received one before the shuffle ran.\n\n");

    printf("CONCLUSION: the fix is not to skip the bounds check Section 18.1 established is required --\n");
    printf("it is to let every lane reach the shuffle, substituting the reduction's identity value for\n");
    printf("out-of-range lanes instead of letting them exit before the shuffle rounds run.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 08_warp_shuffle_common_trap.cu -o 08_warp_shuffle_common_trap
./08_warp_shuffle_common_trap
```

## Chapter Summary

A kernel launch's block count is ceiling division, not floor division — `(size + N - 1) / N` — and the reason is asymmetric: rounding down silently drops real elements no bounds check can rescue, while rounding up and pairing it with `if (idx < size)` inside the kernel wastes a bounded, known number of idle threads (`192` for this chapter's million-element example, `112` for its ten-thousand-element one) in exchange for guaranteed full coverage, genuinely confirmed by a real `<<<>>>` launch honestly reporting `cudaErrorNoDevice` in this GPU-less environment. Memory coalescing is Chapter 4.4's warp-bandwidth argument made concrete on a real eight-field financial record: reading one field across four bond records costs one transaction in Struct-of-Arrays layout and four in Array-of-Structs, a `4×` penalty on this chapter's scaled-down example that reflects the same `8×` byte-stride penalty (`32` bytes vs `4` bytes) Part 7's real portfolio pricing measures directly — confirmed a second time by decoding the actual compiled SASS, where `nvcc` hides the byte-stride multiplier inside an `HFMA2.MMA` half-precision bit pattern rather than a plain visible immediate, and where only the struct-array read (not the kernel's separate write to a plain output array) pays the wide-stride cost. A naive convolution kernel reads most of its input several times over — `36` total reads for `16` unique values in this chapter's own `4×4` example, genuinely counted rather than asserted — purely because neighboring output threads' windows overlap; staging the block's entire input footprint into a real `__shared__` array once, using more threads to load than will ever compute an output, cuts that down to exactly `16` global reads, with a real `__syncthreads()` between the loading phase and the computing phase standing between "correct" and a race condition. Padding trades a shrinking output for a fixed-size one by surrounding the input with zeros rather than narrowing the valid window, verified here at both a border position (`11`) and an interior one that exactly reproduces the unpadded chapter's own first answer (`30`), just shifted. Finally, warp-level shuffle instructions — built here on the real `__shfl_down_sync` intrinsic — collapse the last `log2(32)=5` rounds of any tree reduction into register-to-register exchanges with no memory traffic at all, but they require every lane in the warp to still be executing, which makes Section 18.1's own bounds-check pattern, applied carelessly to a kernel that also uses shuffles, the exact mechanism that can break one — genuinely demonstrated here with a broken reduction that produces `5.45e+26` instead of `10.0`, not merely described.

## Self-Check Questions

1. For `size = 5,000` and `THREADS_PER_BLOCK = 512`, compute `num_blocks`, the total number of threads launched, the number wasted, and how many threads in the last block are actually active.
2. A warp-scaled group of `4` threads reads the `duration` field (the seventh of the eight `ZeroCouponBondAoS` fields, float-index `6` within each `8`-float AoS record) for bond records `0` through `3`, using the same `16`-byte/`4`-float chunk convention as Worked Example 18.2.1. How many transactions does the AoS layout need, and what is its bus utilization?
3. Worked Example 18.2.2 decoded `nvcc`'s `HFMA2.MMA`-encoded stride multiplier by dividing the printed float by `2⁻²⁴`. If a third kernel's SASS showed `HFMA2.MMA R9, -RZ, RZ, 0, 4.76837158203125e-07`, what struct or array element size would that decode to, and what does that number tell you about the kernel's layout?
4. A kernel loads its shared-memory tile and then begins its convolution loop immediately, without calling `__syncthreads()` first. Some threads finish their load-and-compute before other threads in the same block have even finished loading. What specific kind of wrong value can this produce, and which threads are affected?
5. Using this chapter's `4×4` input and `3×3` kernel with `padding=1`, compute the padded output value at the top-right corner, `output(0,3)`.
6. A warp-level reduction kernel has `4` of its `32` lanes return early, before the `__shfl_down_sync` calls, because a bounds check found `idx >= size` for those four threads. Why is this a problem, and which earlier section of this same chapter introduced the pattern now causing it?

## Where We Go Next

Chapter 19 (`part5/02-performance-optimization.md`) turns from kernel *design* to kernel *measurement*: vectorized memory access, loop unrolling, compile-time specialization via templates, and the benchmarking discipline used to tell whether any of this chapter's optimizations — ceiling-division launches, SoA layouts, shared-memory tiling, warp shuffles — actually helped, rather than simply looking like they should have.

## Worked Solutions

**1.** `num_blocks = (5,000 + 511) / 512 = 5,511 / 512 = 10`. Total threads launched: `10 × 512 = 5,120`. Wasted: `5,120 - 5,000 = 120`. The last block is block `9`, covering global indices `9 × 512 = 4,608` through `5,119`. Valid indices are `4,608` through `4,999` — `392` active threads — leaving `512 - 392 = 120` idle threads in that one block, exactly matching the wasted count computed from the totals.

**2.** Record `0`'s `duration` sits at float-index `0×8+6=6`; record `1`'s at `8+6=14`; record `2`'s at `16+6=22`; record `3`'s at `24+6=30`. Dividing by `4` to find each `16`-byte chunk: `6→chunk 1`, `14→chunk 3`, `22→chunk 5`, `30→chunk 7` — four distinct chunks, so `4` transactions are needed. Bytes moved: `4×16=64`; bytes used: `4×4=16`; utilization: `16/64=25%` — structurally identical to Worked Example 18.2.1's `risk_free_rate` case, confirming the AoS penalty isn't specific to any one field.

**3.** `4.76837158203125e-07 / 2⁻²⁴ = 8.0`. An element size of `8` bytes most likely means the kernel is indexing an array of `double`s, or a two-`float` struct (or a `float2`) — any type whose `sizeof` is `8`. The number by itself doesn't distinguish between those possibilities; what it confirms is the byte distance the compiler will actually step between consecutive elements when that `IMAD.WIDE` executes, which is the number that determines whether consecutive threads' accesses land in the same memory transaction or different ones — exactly what Worked Example 18.2.1's `16`-byte chunk arithmetic depends on.

**4.** This produces a race condition: a thread that reaches the convolution loop before every thread assigned to load a value it needs has actually written that value reads whatever garbage or stale data happened to already be sitting in that shared-memory slot, not the input value that thread was supposed to load. The affected threads are unpredictable and can vary from run to run — specifically, any thread whose `3×3` window includes a tile cell that a *slower* sibling thread (one still mid-load, or not yet scheduled) hasn't written yet. `__syncthreads()` exists precisely to rule this out, by making every thread in the block wait until all of them have finished the load phase before any of them is allowed to start reading.

**5.** `output(0,3)`'s window needs input columns `2` through `4` and rows `-1` through `1`. Row `-1` is entirely padding and contributes `0` regardless of kernel weights. For row `0`: `input[0,2]=3` × kernel `[1][0]=0` → `0`; `input[0,3]=4` × kernel `[1][1]=1` → `4`; column `4` is padding × kernel `[1][2]=0` → `0`. For row `1`: `input[1,2]=7` × kernel `[2][0]=1` → `7`; `input[1,3]=8` × kernel `[2][1]=0` → `0`; column `4` padding × kernel `[2][2]=1` → `0`. Total: `4 + 7 = 11`.

**6.** `__shfl_down_sync` reads a value from another lane's live register in the same warp — it has no defined result when the lane it's reading from has already exited the kernel. The four early-returning lanes are no longer executing at all by the time the remaining lanes reach `__shfl_down_sync`, so any exchange involving them reads undefined content instead of a meaningful partial sum, silently corrupting the reduction — genuinely demonstrated in Section 18.4 as a reduction that produces `5.45e+26` instead of `10.0`. The pattern responsible is Section 18.1's own bounds-check idiom (`if (idx >= size) return;` or its early-exit variants) — entirely correct and necessary on its own, as Section 18.1 established, but unsafe to combine with a later warp shuffle unless every lane is guaranteed to still reach the shuffle call, typically by substituting the reduction's identity value (`0` for a sum) for out-of-range lanes instead of letting them return early.
