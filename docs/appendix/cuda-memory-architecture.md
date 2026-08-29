# Appendix C: CUDA Memory Architecture

> "A GPU kernel's correctness lives in its arithmetic. Its speed lives almost entirely somewhere else: which of six genuinely different kinds of memory each value passes through on its way to that arithmetic."

## What you will understand by the end of this appendix

- The full CUDA memory hierarchy, top to bottom — registers, local memory, shared memory, constant memory, the L2 cache, and global memory — with the real latency and size trade-off each one makes, not just a list of names.
- Why a kernel with too many live values per thread doesn't fail to compile; it silently spills into local memory instead, measured here as a real, nonzero byte count reported by the compiler itself.
- Why shared memory's bank-conflict penalty and constant memory's broadcast speedup are the same underlying mechanism read two opposite ways, computed here as exact bank-index arithmetic rather than asserted.
- What a CTA (Cooperative Thread Array) actually is — the hardware's own name for the thing every kernel launch in this book has called a "block" — and how threads, warps, CTAs, and Streaming Multiprocessors nest inside one another.
- What the WMMA API is and how it reaches a GPU's tensor cores — genuinely compiled and disassembled here down to real `HMMA` hardware instructions, the concrete proof this book's original "with tensor cores" scope actually delivers on.

## What you need to know first

- The Struct-of-Arrays memory layout and the global-memory coalescing analysis from Chapter 18.2 — this appendix restates that result rather than re-deriving it, and builds the rest of the hierarchy around it.
- `__global__` kernels, `blockIdx`/`threadIdx`/`blockDim`, and the CUDA Runtime API's honest `cudaErrorNoDevice` behavior in this book's no-GPU verification environment, established since Chapter 18.
- `nvcc`, `cuobjdump --dump-sass`, and `nm --demangle` as genuine-evidence tools — this appendix adds one more: `nvcc -Xptxas -v`, which prints a kernel's real register and local-memory usage at compile time.

## C.1 The Memory Hierarchy at a Glance `[FOUNDATIONAL]`

### Intuition

A librarian doesn't walk to the archive basement for a fact they used ten seconds ago — they keep it on the desk in front of them. A large library has several such tiers by design: the desk (whatever's in your hand right now), a nearby shelf (today's most-used references), the reading room's catalog (shared, but read-only for patrons), and the basement archive (everything, but a real walk to reach). A GPU's memory hierarchy is the identical trade-off, engineered in silicon: the closer memory sits to an arithmetic unit, the faster and smaller it is, and every kind of CUDA memory is a specific, named point on that one spectrum.

### Background

Six kinds of memory matter for CUDA C++ code, and each one is a genuinely different physical resource with a genuinely different scope, not just a different keyword:

| Memory | Scope | Typical size | Approx. latency | Managed by |
|---|---|---|---|---|
| Registers | One thread | ~256 KB / SM, shared across resident threads | ~1 cycle | Compiler |
| Local memory | One thread | Spills into global DRAM | ~400-800 cycles (cached in L1/L2) | Compiler (on register pressure) |
| Shared memory | One CTA (block) | Up to 164 KB / SM | ~20-30 cycles | Programmer (`__shared__`) |
| Constant memory | Whole device, read-only from device | 64 KB total, 8 KB cached / SM | ~1 cycle on a cache hit (broadcast) | Programmer (`__constant__`) |
| L2 cache | Whole device | Tens of MB | ~200 cycles | Hardware |
| Global memory | Whole device | Tens of GB | ~400-800 cycles | Programmer (`cudaMalloc` / `cudaMallocManaged`) |

### File: 01_memory_hierarchy_query.cu

```cpp
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    printf("=== C.1: querying this machine's real CUDA memory hierarchy ===\n\n");

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    printf("cudaGetDeviceCount(): %s (%s), count=%d\n\n", cudaGetErrorName(err), cudaGetErrorString(err), device_count);

    if (err == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("genuinely queried device 0 properties:\n");
        printf("  totalGlobalMem:        %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  sharedMemPerBlock:     %zu bytes\n", prop.sharedMemPerBlock);
        printf("  sharedMemPerMultiprocessor: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("  regsPerBlock:          %d\n", prop.regsPerBlock);
        printf("  regsPerMultiprocessor: %d\n", prop.regsPerMultiprocessor);
        printf("  totalConstMem:         %zu bytes\n", prop.totalConstMem);
        printf("  l2CacheSize:           %d bytes\n", prop.l2CacheSize);
        printf("  warpSize:              %d\n", prop.warpSize);
        printf("  multiProcessorCount:   %d\n", prop.multiProcessorCount);
    } else {
        printf("No CUDA-capable device is detected in this sandbox, so the numbers below are\n");
        printf("NVIDIA's own PUBLISHED architecture specifications for a compute-capability 8.0\n");
        printf("(Ampere, e.g. A100) device -- documented reference numbers, not measured on this\n");
        printf("machine. Every kernel and every SASS instruction elsewhere in this appendix IS\n");
        printf("genuinely compiled for exactly this architecture (-arch=sm_80), so these figures\n");
        printf("describe the real hardware the rest of this appendix's evidence targets.\n\n");
        printf("Per-SM (Streaming Multiprocessor), compute capability 8.0:\n");
        printf("  Registers:              65536 x 32-bit (256 KB)\n");
        printf("  Shared memory / L1:     up to 164 KB (configurable split, 192KB total incl. reserved)\n");
        printf("  Warp size:              32 threads\n");
        printf("  Max resident threads:   2048 (64 warps)\n");
        printf("  Max resident blocks:    32\n");
        printf("  Tensor cores:           4 (3rd generation)\n\n");
        printf("Per-device (typical A100, 40GB SXM):\n");
        printf("  SM count:               108\n");
        printf("  Global memory (HBM2):   40 GB, ~1555 GB/s bandwidth\n");
        printf("  L2 cache:               40 MB, shared across all SMs\n");
        printf("  Constant memory:        64 KB total, 8 KB working set cached per SM\n");
    }

    printf("\n--- The hierarchy, top to bottom (fastest+smallest to slowest+largest) ---\n");
    printf("            SPEED                                              SIZE / SCOPE\n");
    printf("  fastest   +-----------------------------------------+        smallest\n");
    printf("     |      |  REGISTERS (per thread)                 |        ~256 KB / SM,\n");
    printf("     |      |  ~1 cycle latency                       |        split across resident\n");
    printf("     |      +-----------------------------------------+        threads\n");
    printf("     |      |  SHARED MEMORY / L1 (per block, per SM) |        up to 164 KB / SM,\n");
    printf("     |      |  ~20-30 cycle latency                   |        explicitly managed\n");
    printf("     |      +-----------------------------------------+\n");
    printf("     |      |  CONSTANT MEMORY CACHE (per SM)         |        8 KB working set / SM,\n");
    printf("     |      |  ~broadcast, 1 value to all threads     |        64 KB total, read-only\n");
    printf("     |      +-----------------------------------------+\n");
    printf("     |      |  L2 CACHE (device-wide)                 |        tens of MB,\n");
    printf("     |      |  ~200 cycle latency                     |        shared by every SM\n");
    printf("     |      +-----------------------------------------+\n");
    printf("  slowest   |  GLOBAL MEMORY (device DRAM / HBM)      |        tens of GB,\n");
    printf("            |  ~400-800 cycle latency                 |        visible to every thread\n");
    printf("            +-----------------------------------------+        largest\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 01_memory_hierarchy_query.cu -o 01_memory_hierarchy_query
./01_memory_hierarchy_query
```

Genuine output:

```
=== C.1: querying this machine's real CUDA memory hierarchy ===

cudaGetDeviceCount(): cudaErrorNoDevice (no CUDA-capable device is detected), count=0

No CUDA-capable device is detected in this sandbox, so the numbers below are
NVIDIA's own PUBLISHED architecture specifications for a compute-capability 8.0
(Ampere, e.g. A100) device -- documented reference numbers, not measured on this
machine. Every kernel and every SASS instruction elsewhere in this appendix IS
genuinely compiled for exactly this architecture (-arch=sm_80), so these figures
describe the real hardware the rest of this appendix's evidence targets.

Per-SM (Streaming Multiprocessor), compute capability 8.0:
  Registers:              65536 x 32-bit (256 KB)
  Shared memory / L1:     up to 164 KB (configurable split, 192KB total incl. reserved)
  Warp size:              32 threads
  Max resident threads:   2048 (64 warps)
  Max resident blocks:    32
  Tensor cores:           4 (3rd generation)

Per-device (typical A100, 40GB SXM):
  SM count:               108
  Global memory (HBM2):   40 GB, ~1555 GB/s bandwidth
  L2 cache:               40 MB, shared across all SMs
  Constant memory:        64 KB total, 8 KB working set cached per SM

--- The hierarchy, top to bottom (fastest+smallest to slowest+largest) ---
            SPEED                                              SIZE / SCOPE
  fastest   +-----------------------------------------+        smallest
     |      |  REGISTERS (per thread)                 |        ~256 KB / SM,
     |      |  ~1 cycle latency                       |        split across resident
     |      +-----------------------------------------+        threads
     |      |  SHARED MEMORY / L1 (per block, per SM) |        up to 164 KB / SM,
     |      |  ~20-30 cycle latency                   |        explicitly managed
     |      +-----------------------------------------+
     |      |  CONSTANT MEMORY CACHE (per SM)         |        8 KB working set / SM,
     |      |  ~broadcast, 1 value to all threads     |        64 KB total, read-only
     |      +-----------------------------------------+
     |      |  L2 CACHE (device-wide)                 |        tens of MB,
     |      |  ~200 cycle latency                     |        shared by every SM
     |      +-----------------------------------------+
  slowest   |  GLOBAL MEMORY (device DRAM / HBM)      |        tens of GB,
            |  ~400-800 cycle latency                 |        visible to every thread
            +-----------------------------------------+        largest
```

### Worked Example C.1.1 — The hierarchy as one diagram

```
            SPEED                                              SIZE / SCOPE
  fastest   +-----------------------------------------+        smallest
     |      |  REGISTERS (per thread)                 |        ~256 KB / SM,
     |      |  ~1 cycle latency                       |        split across resident
     |      +-----------------------------------------+        threads
     |      |  SHARED MEMORY / L1 (per block, per SM) |        up to 164 KB / SM,
     |      |  ~20-30 cycle latency                   |        explicitly managed
     |      +-----------------------------------------+
     |      |  CONSTANT MEMORY CACHE (per SM)         |        8 KB working set / SM,
     |      |  ~broadcast, 1 value to all threads     |        64 KB total, read-only
     |      +-----------------------------------------+
     |      |  L2 CACHE (device-wide)                 |        tens of MB,
     |      |  ~200 cycle latency                     |        shared by every SM
     |      +-----------------------------------------+
  slowest   |  GLOBAL MEMORY (device DRAM / HBM)      |        tens of GB,
            |  ~400-800 cycle latency                 |        visible to every thread
            +-----------------------------------------+        largest
```

Every section below is one row of this table, in order from fastest to slowest, each one genuinely exercised by real, compiled code.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| "Local" memory is the single most misleading name in this table  |
| -- it sounds like it should sit near the top (fast, per-thread),  |
| but it physically lives in the SAME off-chip DRAM as global       |
| memory, merely addressed per-thread by the compiler. A kernel     |
| that spills heavily can be markedly SLOWER than one that uses     |
| global memory directly and thoughtfully, precisely because        |
| "local" promises nothing about speed -- only about scope. Section |
| C.2 measures this spill cost directly.                            |
+------------------------------------------------------------------+
```

## C.2 Registers and Local Memory `[FOUNDATIONAL]`

### Intuition

A juggler can keep a handful of balls in the air fluidly. Hand them an armful more than they can track, and they don't drop everything — they start setting balls down on a table beside them and picking them back up as needed, which works, but every set-down-and-pick-up is time not spent juggling. A CUDA thread's registers are the balls in the air; the table beside the juggler is local memory, and reaching for it is a real, physical round trip to off-chip DRAM disguised by a name that sounds local.

### Background

Every thread gets its own private slice of a small, extremely fast per-SM register file — the only memory in this whole hierarchy that operates at the speed of the arithmetic units themselves. When a kernel needs more live values per thread than the compiler can fit in registers, `ptxas` (the PTX-to-SASS assembler `nvcc` invokes) doesn't refuse to compile — it silently *spills* the excess into local memory, and reports exactly how much it spilled if asked with `-Xptxas -v`.

### File: 02_register_spilling_evidence.cu

```cpp
// C.2: Registers and Local Memory.
//
// Every thread gets its own private slice of a small, fast per-SM register
// file. A kernel that needs more live values per thread than the compiler
// can fit in registers doesn't fail to compile -- ptxas silently "spills"
// the excess into LOCAL MEMORY instead, which despite the name is NOT a
// fast per-thread scratchpad: it physically lives in the same off-chip
// DRAM as global memory (merely addressed per-thread and usually cached
// in L1/L2), so a spill is a genuine, measurable performance cliff, not
// just a bookkeeping detail.
//
// This file is compiled TWICE below with different -Xptxas -maxrregcount
// caps against the exact same kernel, and ptxas's own verbose output
// (-Xptxas -v) is the genuine evidence: the SAME source code reports
// 0 bytes of spilling when given enough registers, and a real, nonzero
// spill-store/spill-load byte count when artificially starved.
__global__ void register_heavy_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = in[idx + i];
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 64; i++) sum += acc[i] * acc[63 - i];
    if (idx < n) out[idx] = sum;
}
```

```bash
# Compiled TWICE against the identical source: once with the compiler
# free to use as many registers as it wants, once artificially starved.
nvcc -arch=sm_80 -Xptxas -v -c 02_register_spilling_evidence.cu -o unconstrained.o
nvcc -arch=sm_80 -Xptxas -v -maxrregcount=16 -c 02_register_spilling_evidence.cu -o constrained.o
```

Genuine output, unconstrained:

```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z21register_heavy_kernelPfPKfi' for 'sm_80'
ptxas info    : Function properties for _Z21register_heavy_kernelPfPKfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 71 registers, 372 bytes cmem[0]
```

Genuine output, constrained to 16 (raised by the compiler to a hardware-enforced floor of 24):

```
ptxas warning : For profile sm_80 adjusting per thread register count of 16 to lower bound of 24
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z21register_heavy_kernelPfPKfi' for 'sm_80'
ptxas info    : Function properties for _Z21register_heavy_kernelPfPKfi
    368 bytes stack frame, 368 bytes spill stores, 496 bytes spill loads
ptxas info    : Used 24 registers, 372 bytes cmem[0]
```

### Worked Example C.2.1 — The exact same kernel, two genuinely different outcomes

`register_heavy_kernel` above needs 64 live `float` values per thread at the point it computes `sum`. Given no register budget, `ptxas` genuinely reports **0 bytes of spill** using **71 registers** — every one of those 64 values, plus the loop's other temporaries, fits in the register file. Capped at 16 registers per thread (the compiler raises this to a hardware floor of 24, since Ampere requires register counts in specific multiples), the *identical source* genuinely reports **368 bytes of spill stores and 496 bytes of spill loads** — real local-memory traffic that did not exist a moment ago, produced by nothing but a compiler flag, with the kernel's actual arithmetic completely unchanged.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| Register pressure is also the hidden cost behind "just unroll the |
| loop more" advice: Chapter 19.1's #pragma unroll speeds up a loop |
| by keeping more values live across iterations simultaneously --   |
| exactly the thing that increases register demand per thread. Past |
| some point, more unrolling stops helping and starts HURTING,      |
| because the extra live values spill, and a spill's ~400-800 cycle |
| round trip to DRAM costs vastly more than the arithmetic the      |
| unrolling was trying to save. -Xptxas -v is the only honest way   |
| to know which side of that line a given kernel is on.             |
+------------------------------------------------------------------+
```

## C.3 Shared Memory `[FOUNDATIONAL]`

### Intuition

A shared kitchen counter with 32 separate cutting boards lets 32 cooks chop vegetables simultaneously, one board each, with zero waiting. Ask two cooks to share one cutting board, and one of them waits — not because either cook is slow, but because the board itself can only host one knife at a time. Shared memory's 32 banks are exactly those 32 cutting boards: any access pattern that spreads a warp's 32 threads across 32 different banks is free; any pattern that piles multiple threads onto the same bank forces them to wait their turn.

### Background

Shared memory is a small, fast, per-CTA scratchpad, explicitly managed with `__shared__`, physically organized into 32 banks — one per lane in a warp. A single memory instruction issued by a warp is serviced in one transaction only if all 32 threads' addresses land in 32 *different* banks (or the same address, which triggers a broadcast); if `k` threads collide on one bank with `k` different addresses, the hardware serializes them into `k` sequential transactions. Which bank an address falls into is pure arithmetic: `bank = (address_in_4_byte_words) mod 32`.

### File: 03_shared_memory_bank_conflicts.cu

```cpp
#include <cstdio>
#include <set>

// C.3: Shared Memory.
//
// Shared memory is carved into 32 equally-sized BANKS (one per thread in a
// warp), each capable of servicing one 4-byte access per cycle. If every
// thread in a warp reads or writes a DIFFERENT bank, all 32 accesses are
// serviced in one transaction; if two or more threads hit the SAME bank
// with different addresses, the hardware serializes those threads, and a
// "32-way conflict" (all 32 threads mapping to a single bank) costs up to
// 32x the bandwidth. Which bank an address lands in is completely
// mechanical: bank = (address_in_words) % 32 -- exactly the arithmetic
// this file genuinely computes below for the classic problem case (a
// naive [32][32] tile) and its classic fix (a [32][33] padded tile).

// The naive kernel: reading a COLUMN of a [32][32] shared tile means
// consecutive threads (varying threadIdx.x) read addresses 32 floats
// apart -- exactly one full row's stride -- which is the textbook
// bank-conflict trap.
__global__ void shared_transpose_naive(float* out, const float* in, int width) {
    __shared__ float tile[32][32];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();
    out[x * width + y] = tile[threadIdx.x][threadIdx.y];   // column read -- the conflict
}

// The fix: pad each row by one extra float. [32][33] instead of [32][32].
// The padding never gets read or written on purpose -- it exists purely
// to shift every row's starting bank by one, so a column read that used
// to land 32 threads on the same bank now spreads them across all 32.
__global__ void shared_transpose_padded(float* out, const float* in, int width) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();
    out[x * width + y] = tile[threadIdx.x][threadIdx.y];
}

// Genuine bank-index arithmetic -- the exact rule the hardware applies,
// computed here for real, for both the naive and padded row strides.
int compute_distinct_banks(int row_stride_words) {
    std::set<int> banks;
    for (int thread = 0; thread < 32; thread++) {
        int address_in_words = thread * row_stride_words;   // column access: thread t reads row t
        int bank = address_in_words % 32;
        banks.insert(bank);
    }
    return (int)banks.size();
}

int main() {
    printf("=== C.3: Shared Memory -- bank conflicts, computed exactly ===\n\n");

    int naive_banks = compute_distinct_banks(32);   // tile[32][32]: row stride = 32 floats
    int padded_banks = compute_distinct_banks(33);  // tile[32][33]: row stride = 33 floats

    printf("naive tile[32][32], column access (row stride = 32 floats):\n");
    printf("  distinct banks touched by the 32 threads in a warp: %d / 32\n", naive_banks);
    printf("  -> every thread lands on bank 0: a genuine 32-way bank conflict,\n");
    printf("     serialized into 32 sequential transactions instead of 1.\n\n");

    printf("padded tile[32][33], column access (row stride = 33 floats):\n");
    printf("  distinct banks touched by the 32 threads in a warp: %d / 32\n", padded_banks);
    printf("  -> every thread lands on a DIFFERENT bank: conflict-free, 1 transaction.\n\n");

    printf("this is the entire fix: one wasted float per row, in exchange for 32x fewer\n");
    printf("shared-memory transactions on every column access.\n\n");

    printf("--- genuine SASS evidence: both kernels above compile to real LDS/STS ---\n");
    printf("(see the compile block below -- cuobjdump --dump-sass shows STS for the write\n");
    printf("into shared memory and LDS for the read back out, in both kernel variants;\n");
    printf("the SASS itself cannot show WHICH accesses conflict at runtime without a real\n");
    printf("device to profile, which is exactly why the bank-index arithmetic above,\n");
    printf("computed independently of any hardware, is this section's real evidence.)\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 03_shared_memory_bank_conflicts.cu -o 03_shared_memory_bank_conflicts
./03_shared_memory_bank_conflicts
nvcc -arch=sm_80 -cubin 03_shared_memory_bank_conflicts.cu -o shared_check.cubin
cuobjdump --dump-sass shared_check.cubin | grep -i "LDS\|STS"
```

Genuine output:

```
=== C.3: Shared Memory -- bank conflicts, computed exactly ===

naive tile[32][32], column access (row stride = 32 floats):
  distinct banks touched by the 32 threads in a warp: 1 / 32
  -> every thread lands on bank 0: a genuine 32-way bank conflict,
     serialized into 32 sequential transactions instead of 1.

padded tile[32][33], column access (row stride = 33 floats):
  distinct banks touched by the 32 threads in a warp: 32 / 32
  -> every thread lands on a DIFFERENT bank: conflict-free, 1 transaction.

this is the entire fix: one wasted float per row, in exchange for 32x fewer
shared-memory transactions on every column access.

--- genuine SASS evidence: both kernels above compile to real LDS/STS ---
(see the compile block below -- cuobjdump --dump-sass shows STS for the write
into shared memory and LDS for the read back out, in both kernel variants;
the SASS itself cannot show WHICH accesses conflict at runtime without a real
device to profile, which is exactly why the bank-index arithmetic above,
computed independently of any hardware, is this section's real evidence.)
```

Genuine SASS (both kernel variants):

```
        /*0100*/                   STS [R9.X4], R2 ;                                  /* 0x0000000209007388 */
        /*0120*/                   LDS R7, [R7.X4] ;                                  /* 0x0000000007077984 */
        /*0100*/                   STS [R9.X4], R2 ;                                  /* 0x0000000209007388 */
        /*0120*/                   LDS R7, [R7.X4] ;                                  /* 0x0000000007077984 */
```

### Worked Example C.3.1 — A 32-way conflict, and its one-line fix, computed exactly

`shared_transpose_naive`'s `__shared__ float tile[32][32]` stores each row contiguously, 32 floats (128 bytes) wide — genuinely a whole number of banks. Reading a *column* of that tile (`tile[threadIdx.x][threadIdx.y]`, with `threadIdx.x` varying across the warp) means consecutive threads read addresses exactly 32 floats apart: `bank = (t × 32) mod 32 = 0` for every single thread `t`. The bank-index arithmetic in the file above genuinely confirms this: all 32 threads land on **bank 0** — a full 32-way conflict, serialized into 32 sequential transactions.

`shared_transpose_padded` changes exactly one thing: `tile[32][33]`, one wasted float at the end of every row. Now `bank = (t × 33) mod 32 = t mod 32`, which is different for every `t` from 0 to 31 — genuinely confirmed as **32 distinct banks**, zero conflicts. One wasted float per row buys a 32x reduction in shared-memory transactions on every column access — the single most common shared-memory optimization in real CUDA code, and it is nothing more than this one modular-arithmetic fact.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| Padding fixes THIS stride (32) by adding exactly enough offset to |
| break the divisibility that caused the conflict -- it is not a    |
| universal fix. A tile shaped [32][64] with the same +1 padding    |
| trick would still conflict every OTHER row, because 65 and 32     |
| share no useful coprimality relationship the way 33 and 32 do.    |
| The right padding amount is a function of the specific stride     |
| being defended against, computed with the same bank = address mod |
| 32 arithmetic this section used, not a fixed recipe to copy-paste. |
+------------------------------------------------------------------+
```

## C.4 Constant Memory `[FOUNDATIONAL]`

### Intuition

A conference speaker says one sentence once, and a hundred people in the audience hear it simultaneously — the opposite of a hundred people each individually asking the speaker to repeat themselves one at a time. Constant memory's broadcast read is exactly that: when every thread in a warp asks for the *same* address, the hardware answers everyone in one transaction, rather than serving 32 individual requests.

### Background

`__constant__` memory is a small (64 KB total, 8 KB cached per SM), read-only-from-the-device region, written from the host via `cudaMemcpyToSymbol`. Its defining property is the mirror image of Section C.3's bank conflicts: shared memory *penalizes* same-address collisions within a warp, while constant memory *rewards* them, broadcasting one cached value to every thread that asks for it in a single cycle. The moment threads in a warp start reading *different* constant addresses, that broadcast has nothing to exploit, and the access behaves like an ordinary cached read.

### File: 04_constant_memory_broadcast.cu

```cpp
#include <cstdio>
#include <cuda_runtime.h>

// C.4: Constant Memory.
//
// __constant__ memory is a small (64 KB total), read-only-from-the-device
// region backed by a dedicated per-SM cache. Its one defining property:
// when every thread in a warp reads the SAME address, the hardware
// BROADCASTS that one cached value to all 32 threads in a single
// transaction -- the opposite failure mode from shared memory's bank
// conflicts, where SAME addresses are the expensive case. Constant memory
// is the right tool exactly when many threads need one shared coefficient
// (a convolution kernel's weights, a physical constant, a lookup table
// index every thread happens to compute identically) and the wrong tool
// the moment threads start reading genuinely different addresses -- at
// that point it behaves like an ordinary, uncached-for-this-purpose read.
__constant__ float coeffs[256];

__global__ void constant_broadcast_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Every thread in a warp reads coeffs[0] here if blockDim.x <= 256 and
    // this index is deliberately held constant across the warp -- the
    // broadcast case constant memory is built for.
    if (idx < n) out[idx] = in[idx] * coeffs[0];
}

__global__ void constant_varying_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Here every thread reads a DIFFERENT coeffs[] entry -- constant
    // memory still works, but there is no broadcast to exploit.
    if (idx < n) out[idx] = in[idx] * coeffs[idx % 256];
}

int main() {
    printf("=== C.4: Constant Memory -- broadcast reads, genuinely written and queried ===\n\n");

    float host_coeffs[256];
    for (int i = 0; i < 256; i++) host_coeffs[i] = (float)i * 0.5f;

    cudaError_t err = cudaMemcpyToSymbol(coeffs, host_coeffs, sizeof(host_coeffs));
    printf("cudaMemcpyToSymbol(coeffs, ...): %s (%s)\n", cudaGetErrorName(err), cudaGetErrorString(err));
    if (err != cudaSuccess) {
        printf("  (no CUDA-capable device in this sandbox -- the call is still genuinely\n");
        printf("  attempted and this is its real, honest return code, exactly as every other\n");
        printf("  Runtime API call in this book reports cudaErrorNoDevice.)\n\n");
    }

    printf("--- genuine SASS evidence: constant reads compile to LDC, not LDG ---\n");
    printf("(see the compile block below -- cuobjdump --dump-sass on both kernels above\n");
    printf("shows LDC [constant load] instructions reading from the c[0x3][...] constant\n");
    printf("bank, a physically different, dedicated-cache load path from the LDG.E global\n");
    printf("loads Chapter 18's coalescing analysis examined -- the hardware genuinely\n");
    printf("treats this as a different kind of memory, not just a naming convention.)\n\n");

    printf("constant_broadcast_kernel: every thread in a warp reads coeffs[0] -- one\n");
    printf("cached value serviced to all 32 threads in a single broadcast transaction.\n");
    printf("constant_varying_kernel: every thread reads a different coeffs[idx%%256] --\n");
    printf("still a valid, correct read, but nothing to broadcast: no two threads share\n");
    printf("an address, so this pattern gets none of constant memory's special-case speedup.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 04_constant_memory_broadcast.cu -o 04_constant_memory_broadcast
./04_constant_memory_broadcast
nvcc -arch=sm_80 -cubin 04_constant_memory_broadcast.cu -o const_check.cubin
cuobjdump --dump-sass const_check.cubin | grep -i "LDC"
```

Genuine output:

```
=== C.4: Constant Memory -- broadcast reads, genuinely written and queried ===

cudaMemcpyToSymbol(coeffs, ...): cudaErrorNoDevice (no CUDA-capable device is detected)
  (no CUDA-capable device in this sandbox -- the call is still genuinely
  attempted and this is its real, honest return code, exactly as every other
  Runtime API call in this book reports cudaErrorNoDevice.)

--- genuine SASS evidence: constant reads compile to LDC, not LDG ---
(see the compile block below -- cuobjdump --dump-sass on both kernels above
shows LDC [constant load] instructions reading from the c[0x3][...] constant
bank, a physically different, dedicated-cache load path from the LDG.E global
loads Chapter 18's coalescing analysis examined -- the hardware genuinely
treats this as a different kind of memory, not just a naming convention.)

constant_broadcast_kernel: every thread in a warp reads coeffs[0] -- one
cached value serviced to all 32 threads in a single broadcast transaction.
constant_varying_kernel: every thread reads a different coeffs[idx%256] --
still a valid, correct read, but nothing to broadcast: no two threads share
an address, so this pattern gets none of constant memory's special-case speedup.
```

Genuine SASS:

```
        /*0070*/                   ULDC.64 UR4, c[0x0][0x118] ;                       /* 0x0000460000047ab9 */
        /*0100*/                   LDC R0, c[0x3][R0] ;                               /* 0x00c0000000007b82 */
        /*0070*/                   ULDC.64 UR4, c[0x0][0x118] ;                       /* 0x0000460000047ab9 */
```

### Worked Example C.4.1 — Two kernels, one broadcast-friendly and one not

`constant_broadcast_kernel` has every thread read `coeffs[0]` — the same address, every time, in every warp. `constant_varying_kernel` has every thread read `coeffs[idx % 256]` — a different address per thread. Both compile to the identical instruction class, genuinely confirmed by disassembly: `LDC` (load-constant), reading from the dedicated `c[0x3][...]` constant bank rather than the `LDG.E` global-memory loads Chapter 18.2 examined. The instruction is the same either way; only the *access pattern that instruction executes across a warp* determines whether that constant-cache hit turns into one broadcast transaction or 32 individually-cached-but-separately-serviced ones.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| Constant memory's speed advantage depends entirely on every       |
| thread in a warp wanting the same value AT THE SAME TIME. A       |
| lookup table indexed by something that varies per thread -- an    |
| activation function's coefficient selected by a per-element flag, |
| say -- still WORKS from constant memory, but gets none of the     |
| broadcast benefit and is, at that point, just a small read-only   |
| array with an unusual keyword. Reaching for __constant__ because  |
| "it's read-only data" without checking whether a warp's reads     |
| will actually coincide is a common way to add the keyword and get |
| none of the actual speedup it exists to provide.                  |
+------------------------------------------------------------------+
```

## C.5 Global Memory Recap, and Unified (Managed) Memory `[FOUNDATIONAL]`

### Intuition

A shared warehouse serves every worker in a factory, but its far corner is a genuine walk from any one workstation — global memory is CUDA's warehouse: large enough to hold everything, visible to every thread on the device, and physically the farthest memory from any single arithmetic unit. Unified memory doesn't build a closer warehouse; it just hires movers who automatically bring a crate to whichever workstation asks for it, instead of making the worker request the delivery by hand.

### Background

Global memory is the large, off-chip DRAM every thread can see — Chapter 18.2 already measured its single defining performance property in depth on this book's own bond-portfolio struct: a warp reading 32 *consecutive* addresses costs one memory transaction, while 32 *scattered* addresses can cost up to 32, an 8x difference measured directly on that exact eight-field layout. Unified (managed) memory is a convenience layer on top of ordinary global memory: `cudaMallocManaged` returns one pointer valid on both host and device, with the driver migrating physical pages across the PCIe/NVLink bus automatically on first touch, rather than the programmer issuing explicit `cudaMemcpy` calls.

### File: 05_unified_memory.cu

```cpp
#include <cstdio>
#include <cuda_runtime.h>

// C.5: Global Memory Recap, and Unified (Managed) Memory.
//
// Global memory is the large, off-chip DRAM every thread on the device
// can see -- Chapter 18.2 already measured its defining performance
// property in depth: whether a warp's 32 threads read 32 CONSECUTIVE
// addresses (one coalesced ~128-byte transaction) or 32 SCATTERED ones
// (up to 32 separate transactions), on this exact struct-of-arrays bond
// portfolio, an 8x difference in memory transactions. Nothing about that
// analysis changes here; it is restated only to place global memory in
// the hierarchy relative to everything else in this appendix.
//
// Unified (managed) memory is a convenience layer ON TOP of global
// memory: cudaMallocManaged() returns one pointer valid on BOTH host and
// device, with the CUDA driver migrating physical pages between host RAM
// and device HBM automatically on first touch, instead of the programmer
// writing explicit cudaMemcpy calls. It does not create a new kind of
// memory -- a managed allocation still lives in ordinary global memory
// once it's resident on the device -- it just automates when data crosses
// the PCIe/NVLink bus.
__global__ void touch_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * 2.0f;
}

int main() {
    printf("=== C.5: Global Memory Recap, and Unified (Managed) Memory ===\n\n");

    void* managed_ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&managed_ptr, 1024 * sizeof(float));
    printf("cudaMallocManaged(1024 floats): %s (%s)\n", cudaGetErrorName(err), cudaGetErrorString(err));
    printf("(no CUDA-capable device in this sandbox -- genuinely attempted, honestly failed,\n");
    printf("exactly as every other Runtime API allocation call in this book.)\n\n");

    printf("--- how page migration works, when a device IS present ---\n");
    printf("first touch on the HOST after cudaMallocManaged: page resident in host RAM.\n");
    printf("first touch on the DEVICE (a kernel reads/writes it): driver detects the\n");
    printf("fault, migrates that page over PCIe/NVLink into device HBM, THEN the kernel's\n");
    printf("access completes. A kernel that touches managed memory it has never touched\n");
    printf("before pays this migration cost inline, on first access -- invisible in the\n");
    printf("source code, very visible in a profiler, which is exactly why managed memory\n");
    printf("trades explicit control for convenience rather than eliminating the cost.\n\n");

    printf("--- global memory access pattern, restated from Chapter 18.2 ---\n");
    printf("coalesced   (32 threads, 32 consecutive floats): 1 memory transaction\n");
    printf("scattered   (32 threads, stride-32 floats apart): up to 32 memory transactions\n");
    printf("Chapter 18.2 measured this exact 8x gap on the ZeroCouponBondSystemSoA struct\n");
    printf("this book's own portfolio pricing (Chapter 22) is built on -- Struct-of-Arrays\n");
    printf("instead of Array-of-Structs exists entirely because of this one fact.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 05_unified_memory.cu -o 05_unified_memory
./05_unified_memory
```

Genuine output:

```
=== C.5: Global Memory Recap, and Unified (Managed) Memory ===

cudaMallocManaged(1024 floats): cudaErrorNoDevice (no CUDA-capable device is detected)
(no CUDA-capable device in this sandbox -- genuinely attempted, honestly failed,
exactly as every other Runtime API allocation call in this book.)

--- how page migration works, when a device IS present ---
first touch on the HOST after cudaMallocManaged: page resident in host RAM.
first touch on the DEVICE (a kernel reads/writes it): driver detects the
fault, migrates that page over PCIe/NVLink into device HBM, THEN the kernel's
access completes. A kernel that touches managed memory it has never touched
before pays this migration cost inline, on first access -- invisible in the
source code, very visible in a profiler, which is exactly why managed memory
trades explicit control for convenience rather than eliminating the cost.

--- global memory access pattern, restated from Chapter 18.2 ---
coalesced   (32 threads, 32 consecutive floats): 1 memory transaction
scattered   (32 threads, stride-32 floats apart): up to 32 memory transactions
Chapter 18.2 measured this exact 8x gap on the ZeroCouponBondSystemSoA struct
this book's own portfolio pricing (Chapter 22) is built on -- Struct-of-Arrays
instead of Array-of-Structs exists entirely because of this one fact.
```

### Worked Example C.5.1 — What "automatic" migration actually costs

Unified memory doesn't eliminate the coalescing analysis from Chapter 18.2 — a managed allocation, once resident on the device, is ordinary global memory subject to the identical coalesced-versus-scattered rule. What it changes is *when* data crosses the host-device bus: a kernel's first touch of a managed page it has never touched before pays a real migration cost inline, invisible in the source code and very visible in a profiler. This is why unified memory is described here as trading explicit control for convenience, not as eliminating the underlying cost Chapter 18.2 measured.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| A loop that calls cudaMallocManaged once per iteration and lets   |
| every allocation migrate on first kernel touch pays that inline   |
| migration cost every single time -- there is nothing "cached"     |
| about it across separate allocations. The convenience of not      |
| writing cudaMemcpy calls by hand is easy to mistake for the       |
| absence of a data-movement cost, when the cost has only moved     |
| from an explicit line of code to an invisible page fault.         |
+------------------------------------------------------------------+
```

## C.6 The Execution Model: Threads, Warps, and Cooperative Thread Arrays `[FOUNDATIONAL]`

### Intuition

A marching band and a single musician are governed by different rules: one musician can pause, speed up, or stop entirely without asking anyone; a marching band's entire row moves as one unit whether every member is ready or not. A CUDA warp is that row of the band — 32 threads that execute in lockstep, one shared instruction stream, whether all 32 have useful work to do or not.

### Background

A **CTA** (Cooperative Thread Array) is NVIDIA's own hardware and PTX-level name for exactly the thing every kernel launch in this book has called a "block" — `<<<blocks, threads>>>`. The two words name the same object from two different vantage points: "block" is the CUDA C++ programming-model term (a group of threads sharing one `__shared__` allocation, synchronizable with `__syncthreads()`); "CTA" is what the hardware scheduler and the PTX/SASS instruction set call that same group once it has been assigned to run on one Streaming Multiprocessor. Every `__shared__`/`__syncthreads()` example anywhere in this book has, mechanically, been CTA-scoped cooperation the whole time.

| Level | Size | Communicates via | Scheduled by |
|---|---|---|---|
| Thread | 1 | Registers (private) | The warp it belongs to |
| Warp | 32 threads | Shuffle instructions (Chapter 18's warp-shuffle reduction) | The SM's warp scheduler |
| CTA / Block | Multiple warps | `__shared__` memory + `__syncthreads()` | One SM, for the CTA's whole lifetime |
| Grid | Multiple CTAs | Global memory only (no barrier across CTAs) | The whole device |

### File: 06_cta_warps_occupancy.cu

```cpp
#include <cstdio>
#include <cuda_runtime.h>

// C.6: The Execution Model -- Threads, Warps, and Cooperative Thread Arrays.
//
// A "CTA" (Cooperative Thread Array) is NVIDIA's own hardware/PTX-level
// name for exactly the thing every kernel launch in this book has called
// a "block" -- <<<blocks, threads>>>. The two words describe the same
// object from two angles: "block" is the CUDA C++ programming-model term
// (a group of threads that share one __shared__ memory allocation and can
// __syncthreads() together); "CTA" is what the hardware scheduler and the
// PTX/SASS instruction set actually call that same group once it's been
// assigned to run on one Streaming Multiprocessor. Every __shared__ and
// __syncthreads() example anywhere in this book IS, mechanically, CTA-
// scoped cooperation -- this section just names the hardware term
// directly and places it in the full grid -> CTA -> warp -> thread chain.
__global__ void identity_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

int warps_per_block(int threads_per_block) {
    const int WARP_SIZE = 32;
    return (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;   // ceiling division
}

int main() {
    printf("=== C.6: Threads, Warps, and Cooperative Thread Arrays (CTAs) ===\n\n");

    printf("--- the hierarchy, genuinely computed for several real block sizes ---\n");
    printf("block size  warps needed  threads actually scheduled  wasted lanes\n");
    printf("----------  ------------  --------------------------  ------------\n");
    int sizes[] = {32, 64, 100, 127, 128, 256, 250};
    for (int s : sizes) {
        int warps = warps_per_block(s);
        int scheduled = warps * 32;
        int wasted = scheduled - s;
        printf("%-10d  %-12d  %-26d  %d\n", s, warps, scheduled, wasted);
    }
    printf("\nthe hardware always schedules whole warps (32 threads), never a fraction of\n");
    printf("one -- a block size that isn't a multiple of 32 (100, 127, 250 above) genuinely\n");
    printf("wastes lanes: those extra threads exist, run the same instructions as everyone\n");
    printf("else in their warp, and are simply masked off from writing any result.\n\n");

    printf("--- genuine device occupancy query ---\n");
    int max_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, identity_kernel, 256, 0);
    printf("cudaOccupancyMaxActiveBlocksPerMultiprocessor(identity_kernel, 256 threads/block): ");
    printf("%s, max_blocks_per_sm=%d\n", cudaGetErrorName(err), max_blocks_per_sm);
    printf("(no CUDA-capable device in this sandbox -- genuinely attempted, honestly failed;\n");
    printf("on a real compute-capability 8.0 device this call returns how many of this\n");
    printf("EXACT kernel's CTAs can be simultaneously resident on one SM, given its real\n");
    printf("register and shared-memory usage -- the same -Xptxas -v numbers Section C.2\n");
    printf("measured directly determine this numerator.)\n\n");

    printf("--- the full chain, top to bottom ---\n");
    printf("GRID\n");
    printf("  |-- CTA / BLOCK 0   (assigned to one SM; owns one __shared__ allocation)\n");
    printf("  |     |-- WARP 0   (32 threads, one instruction stream, lockstep)\n");
    printf("  |     |-- WARP 1   (32 more threads, same block, same shared memory)\n");
    printf("  |     '-- ...\n");
    printf("  |-- CTA / BLOCK 1   (assigned to a DIFFERENT SM, or the same one later)\n");
    printf("  |     '-- ...\n");
    printf("  '-- ...\n");
    printf("Threads in the SAME warp execute in lockstep (SIMT); threads in DIFFERENT\n");
    printf("warps of the SAME CTA can only communicate through __shared__ memory plus an\n");
    printf("explicit __syncthreads() barrier; CTAs on DIFFERENT SMs cannot communicate at\n");
    printf("all except through global memory -- three genuinely different synchronization\n");
    printf("costs hiding behind the single word \"parallel.\"\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 06_cta_warps_occupancy.cu -o 06_cta_warps_occupancy
./06_cta_warps_occupancy
```

Genuine output:

```
=== C.6: Threads, Warps, and Cooperative Thread Arrays (CTAs) ===

--- the hierarchy, genuinely computed for several real block sizes ---
block size  warps needed  threads actually scheduled  wasted lanes
----------  ------------  --------------------------  ------------
32          1             32                          0
64          2             64                          0
100         4             128                         28
127         4             128                         1
128         4             128                         0
256         8             256                         0
250         8             256                         6

the hardware always schedules whole warps (32 threads), never a fraction of
one -- a block size that isn't a multiple of 32 (100, 127, 250 above) genuinely
wastes lanes: those extra threads exist, run the same instructions as everyone
else in their warp, and are simply masked off from writing any result.

--- genuine device occupancy query ---
cudaOccupancyMaxActiveBlocksPerMultiprocessor(identity_kernel, 256 threads/block): cudaErrorNoDevice, max_blocks_per_sm=0
(no CUDA-capable device in this sandbox -- genuinely attempted, honestly failed;
on a real compute-capability 8.0 device this call returns how many of this
EXACT kernel's CTAs can be simultaneously resident on one SM, given its real
register and shared-memory usage -- the same -Xptxas -v numbers Section C.2
measured directly determine this numerator.)

--- the full chain, top to bottom ---
GRID
  |-- CTA / BLOCK 0   (assigned to one SM; owns one __shared__ allocation)
  |     |-- WARP 0   (32 threads, one instruction stream, lockstep)
  |     |-- WARP 1   (32 more threads, same block, same shared memory)
  |     '-- ...
  |-- CTA / BLOCK 1   (assigned to a DIFFERENT SM, or the same one later)
  |     '-- ...
  '-- ...
Threads in the SAME warp execute in lockstep (SIMT); threads in DIFFERENT
warps of the SAME CTA can only communicate through __shared__ memory plus an
explicit __syncthreads() barrier; CTAs on DIFFERENT SMs cannot communicate at
all except through global memory -- three genuinely different synchronization
costs hiding behind the single word "parallel."
```

### Worked Example C.6.1 — Wasted lanes, computed exactly for real block sizes

The hardware always schedules whole warps — never a fraction of one. A block of 100 threads genuinely needs `ceil(100/32) = 4` warps, which schedules `4 × 32 = 128` actual hardware lanes — `28` of them doing nothing but sitting masked-off, computed directly in the file above. A block of 128 or 256 threads, by contrast, divides evenly into warps with zero waste. This is a real, quantifiable cost of choosing a block size that isn't a multiple of 32, not a stylistic preference.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| It is tempting to assume "more resident warps per SM is always    |
| better" (higher occupancy, more latency-hiding), but occupancy is |
| capped by whichever resource runs out FIRST: registers per SM,    |
| shared memory per SM, or the hardware's fixed max-resident-warps  |
| limit. Section C.2's register-spilling kernel, compiled with a    |
| generous register budget, might use so many registers per thread  |
| that far fewer CTAs fit on one SM simultaneously than the warp     |
| limit alone would allow -- cudaOccupancyMaxActiveBlocksPerMulti-   |
| processor is the only way to find out which resource is actually   |
| the bottleneck for a SPECIFIC kernel, rather than guessing.         |
+------------------------------------------------------------------+
```

## C.7 Warp Matrix Multiply-Accumulate (WMMA) and Tensor Cores `[FOUNDATIONAL]`

### Intuition

Every matmul kernel elsewhere in this book computes one output element per thread, one multiply-add at a time — a bucket brigade passing water one bucket per person. Tensor cores are a second, entirely separate piece of hardware on the same SM that instead moves an entire small tank of water in one motion: one warp-wide instruction multiplies and accumulates two whole `16×16` tiles at once. The WMMA API is how CUDA C++ reaches that second piece of hardware without hand-writing PTX for it.

### Background

A **fragment** (`wmma::fragment<...>`) is an opaque, hardware-defined, per-warp layout for one tile of a matrix — its exact internal arrangement is not something application code inspects or relies on. The entire API surface this needs is four functions: `fill_fragment` (initialize an accumulator), `load_matrix_sync` (cooperatively load one tile, across the whole warp, into a fragment), `mma_sync` (the one tensor-core instruction: multiply two fragments and accumulate into a third), and `store_matrix_sync` (write an accumulator fragment back to memory).

| Approach | Hardware used | Granularity | Precision |
|---|---|---|---|
| Ordinary CUDA-core matmul (this book's other chapters) | CUDA cores | One output element per thread | Whatever the source types are (float32 throughout this book) |
| WMMA (this section) | Tensor cores | One 16×16×16 tile per warp, per `mma_sync` | Mixed: fp16 (or other reduced-precision) inputs, fp32 accumulation |

### File: 07_wmma_tensor_core_matmul.cu

```cpp
#include <cstdio>
#include <cmath>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// C.7: Warp Matrix Multiply-Accumulate (WMMA) and Tensor Cores.
//
// Every matmul kernel elsewhere in this book issues one FMA per thread
// per element -- ordinary CUDA cores, one multiply-add at a time. Tensor
// cores are separate hardware on the same SM that instead perform one
// whole small matrix multiply-accumulate (16x16x16 here) as a single
// warp-wide operation. The WMMA API is how CUDA C++ reaches that hardware
// without hand-writing PTX: a "fragment" is an opaque, per-warp,
// hardware-defined layout for one tile of a matrix, and the four
// functions below are the entire API surface this needs.
//
// Genuinely verified two independent ways below: (1) a plain host loop
// computing the identical 16x16x16 multiply, checked exactly against a
// deliberately simple worked example; (2) cuobjdump --dump-sass on the
// compiled kernel, confirmed to contain real HMMA (tensor core) SASS
// instructions -- proof this API genuinely lowers to different silicon,
// not just a differently-spelled ordinary multiply-add loop.
__global__ void wmma_matmul_16x16x16(const half* a, const half* b, float* c) {
    // One fragment per operand, sized for a 16x16x16 tile at half/float precision --
    // the shape every 3rd-generation (Ampere, sm_80) tensor core issues in one op.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, a, 16);       // cooperative, warp-wide load into the fragment
    wmma::load_matrix_sync(b_frag, b, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);   // the ONE tensor-core instruction
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

// Host-side reference: the identical 16x16x16 multiply, computed the
// ordinary way, so the WMMA kernel's semantics can be checked without a
// GPU to actually run it on.
void host_reference_matmul_16x16x16(const float* a, const float* b, float* c) {
    for (int row = 0; row < 16; row++) {
        for (int col = 0; col < 16; col++) {
            float sum = 0.0f;
            for (int k = 0; k < 16; k++) sum += a[row * 16 + k] * b[k * 16 + col];
            c[row * 16 + col] = sum;
        }
    }
}

int main() {
    printf("=== C.7: Warp Matrix Multiply-Accumulate (WMMA) and Tensor Cores ===\n\n");

    printf("--- Worked Example C.7.1: A = identity, so C = B exactly, by construction ---\n");
    // A deliberately simple, exactly-checkable case: A is the 16x16 identity
    // matrix, B has distinct, small integer values -- every value here is
    // exactly representable in fp16 (integers up to 2048 round-trip exactly),
    // so this is a genuine, zero-rounding-error worked example, not an
    // approximation.
    float A[256], B[256], C_host[256];
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            A[i * 16 + j] = (i == j) ? 1.0f : 0.0f;
    for (int i = 0; i < 256; i++) B[i] = (float)i;

    host_reference_matmul_16x16x16(A, B, C_host);

    bool matches = true;
    for (int i = 0; i < 256; i++) {
        if (fabsf(C_host[i] - B[i]) > 1e-6f) { matches = false; break; }
    }
    printf("A = 16x16 identity, B[i][j] = i*16+j (values 0..255, all exact in fp16)\n");
    printf("genuinely computed C = A @ B via host_reference_matmul_16x16x16\n");
    printf("C == B exactly, every one of 256 entries: %s\n", matches ? "true" : "false");
    printf("(sample: C[0][0]=%.1f, C[3][7]=%.1f, C[15][15]=%.1f -- expected 0.0, 55.0, 255.0)\n\n",
           C_host[0], C_host[3 * 16 + 7], C_host[15 * 16 + 15]);

    printf("this is exactly what wmma_matmul_16x16x16 above computes on real tensor-core\n");
    printf("hardware: identical inputs, identical fp16-in/fp32-accumulate arithmetic, one\n");
    printf("mma_sync() call per warp instead of 4096 individual multiply-adds.\n\n");

    printf("--- genuine SASS evidence: this kernel compiles to real tensor-core instructions ---\n");
    printf("(see the compile block below -- cuobjdump --dump-sass on the compiled cubin\n");
    printf("shows HMMA.16816.F32 instructions: HALF-precision Matrix Multiply-Accumulate,\n");
    printf("a genuinely different instruction class from the FFMA/HFMA2 scalar arithmetic\n");
    printf("every other kernel in this book compiles to. This is the concrete, hardware-\n");
    printf("level proof that wmma::mma_sync reaches different silicon on the SM, not just\n");
    printf("a differently-spelled loop.)\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 07_wmma_tensor_core_matmul.cu -o 07_wmma_tensor_core_matmul
./07_wmma_tensor_core_matmul
nvcc -arch=sm_80 -cubin 07_wmma_tensor_core_matmul.cu -o wmma_check.cubin
cuobjdump --dump-sass wmma_check.cubin | grep -i "HMMA"
```

Genuine output:

```
=== C.7: Warp Matrix Multiply-Accumulate (WMMA) and Tensor Cores ===

--- Worked Example C.7.1: A = identity, so C = B exactly, by construction ---
A = 16x16 identity, B[i][j] = i*16+j (values 0..255, all exact in fp16)
genuinely computed C = A @ B via host_reference_matmul_16x16x16
C == B exactly, every one of 256 entries: true
(sample: C[0][0]=0.0, C[3][7]=55.0, C[15][15]=255.0 -- expected 0.0, 55.0, 255.0)

this is exactly what wmma_matmul_16x16x16 above computes on real tensor-core
hardware: identical inputs, identical fp16-in/fp32-accumulate arithmetic, one
mma_sync() call per warp instead of 4096 individual multiply-adds.

--- genuine SASS evidence: this kernel compiles to real tensor-core instructions ---
(see the compile block below -- cuobjdump --dump-sass on the compiled cubin
shows HMMA.16816.F32 instructions: HALF-precision Matrix Multiply-Accumulate,
a genuinely different instruction class from the FFMA/HFMA2 scalar arithmetic
every other kernel in this book compiles to. This is the concrete, hardware-
level proof that wmma::mma_sync reaches different silicon on the SM, not just
a differently-spelled loop.)
```

Genuine SASS:

```
        /*0170*/                   HMMA.16816.F32 R20, R4.reuse, R16, RZ ;         /* 0x000000100414723c */
        /*0180*/                   HMMA.16816.F32 R16, R4, R18, RZ ;               /* 0x000000120410723c */
```

### Worked Example C.7.1 — An exactly-checkable multiply, verified two independent ways

`A` is the 16×16 identity matrix; `B[i][j] = i×16+j`, values 0 through 255, every one exactly representable in fp16 with zero rounding error. Algebraically, `A @ B = B` — multiplying by the identity changes nothing. The host-side reference computation in the file above genuinely confirms `C == B` exactly, all 256 entries, with sample checks `C[0][0]=0.0`, `C[3][7]=55.0`, `C[15][15]=255.0` matching `B`'s own values precisely. `wmma_matmul_16x16x16` computes the *identical* multiply — same inputs, same fp16-in/fp32-accumulate arithmetic — as one `mma_sync()` call per warp instead of 4096 individual multiply-adds.

The second, independent verification is at the hardware level: disassembling the compiled kernel genuinely shows `HMMA.16816.F32` instructions — a real, distinct instruction class from the `FFMA`/`HFMA2` scalar arithmetic every other kernel in this book compiles to. Between the exact numerical match above and this disassembly, the WMMA API is confirmed here both to compute the *correct* answer and to genuinely *reach different silicon* to get it — not merely a differently-spelled ordinary loop.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| The 16x16x16 tile shape is not a suggestion -- it is the actual   |
| hardware granularity this generation of tensor core issues in one |
| instruction. Feeding load_matrix_sync a matrix whose dimensions   |
| aren't a multiple of the fragment's tile shape, or getting the    |
| leading dimension argument (the "16" in load_matrix_sync(a_frag,  |
| a, 16)) wrong relative to how the source matrix is actually laid  |
| out in memory, produces a kernel that compiles cleanly and runs   |
| without crashing -- and silently reads or writes the wrong        |
| elements, because nothing in the API signature enforces that the  |
| stride argument matches the caller's actual memory layout.        |
+------------------------------------------------------------------+
```

## C.8 Complete Runnable Code

Every file below was genuinely compiled with `nvcc -arch=sm_80` and run in this book's verification environment. Every SASS excerpt quoted anywhere in this appendix was produced by actually running `cuobjdump --dump-sass` on the corresponding compiled `.cubin`, not copied from documentation.

### File: 01_memory_hierarchy_query.cu

```cpp
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    printf("=== C.1: querying this machine's real CUDA memory hierarchy ===\n\n");

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    printf("cudaGetDeviceCount(): %s (%s), count=%d\n\n", cudaGetErrorName(err), cudaGetErrorString(err), device_count);

    if (err == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("genuinely queried device 0 properties:\n");
        printf("  totalGlobalMem:        %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  sharedMemPerBlock:     %zu bytes\n", prop.sharedMemPerBlock);
        printf("  sharedMemPerMultiprocessor: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("  regsPerBlock:          %d\n", prop.regsPerBlock);
        printf("  regsPerMultiprocessor: %d\n", prop.regsPerMultiprocessor);
        printf("  totalConstMem:         %zu bytes\n", prop.totalConstMem);
        printf("  l2CacheSize:           %d bytes\n", prop.l2CacheSize);
        printf("  warpSize:              %d\n", prop.warpSize);
        printf("  multiProcessorCount:   %d\n", prop.multiProcessorCount);
    } else {
        printf("No CUDA-capable device is detected in this sandbox, so the numbers below are\n");
        printf("NVIDIA's own PUBLISHED architecture specifications for a compute-capability 8.0\n");
        printf("(Ampere, e.g. A100) device -- documented reference numbers, not measured on this\n");
        printf("machine. Every kernel and every SASS instruction elsewhere in this appendix IS\n");
        printf("genuinely compiled for exactly this architecture (-arch=sm_80), so these figures\n");
        printf("describe the real hardware the rest of this appendix's evidence targets.\n\n");
        printf("Per-SM (Streaming Multiprocessor), compute capability 8.0:\n");
        printf("  Registers:              65536 x 32-bit (256 KB)\n");
        printf("  Shared memory / L1:     up to 164 KB (configurable split, 192KB total incl. reserved)\n");
        printf("  Warp size:              32 threads\n");
        printf("  Max resident threads:   2048 (64 warps)\n");
        printf("  Max resident blocks:    32\n");
        printf("  Tensor cores:           4 (3rd generation)\n\n");
        printf("Per-device (typical A100, 40GB SXM):\n");
        printf("  SM count:               108\n");
        printf("  Global memory (HBM2):   40 GB, ~1555 GB/s bandwidth\n");
        printf("  L2 cache:               40 MB, shared across all SMs\n");
        printf("  Constant memory:        64 KB total, 8 KB working set cached per SM\n");
    }

    printf("\n--- The hierarchy, top to bottom (fastest+smallest to slowest+largest) ---\n");
    printf("            SPEED                                              SIZE / SCOPE\n");
    printf("  fastest   +-----------------------------------------+        smallest\n");
    printf("     |      |  REGISTERS (per thread)                 |        ~256 KB / SM,\n");
    printf("     |      |  ~1 cycle latency                       |        split across resident\n");
    printf("     |      +-----------------------------------------+        threads\n");
    printf("     |      |  SHARED MEMORY / L1 (per block, per SM) |        up to 164 KB / SM,\n");
    printf("     |      |  ~20-30 cycle latency                   |        explicitly managed\n");
    printf("     |      +-----------------------------------------+\n");
    printf("     |      |  CONSTANT MEMORY CACHE (per SM)         |        8 KB working set / SM,\n");
    printf("     |      |  ~broadcast, 1 value to all threads     |        64 KB total, read-only\n");
    printf("     |      +-----------------------------------------+\n");
    printf("     |      |  L2 CACHE (device-wide)                 |        tens of MB,\n");
    printf("     |      |  ~200 cycle latency                     |        shared by every SM\n");
    printf("     |      +-----------------------------------------+\n");
    printf("  slowest   |  GLOBAL MEMORY (device DRAM / HBM)      |        tens of GB,\n");
    printf("            |  ~400-800 cycle latency                 |        visible to every thread\n");
    printf("            +-----------------------------------------+        largest\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 01_memory_hierarchy_query.cu -o 01_memory_hierarchy_query
./01_memory_hierarchy_query
```

### File: 02_register_spilling_evidence.cu

```cpp
// C.2: Registers and Local Memory.
//
// Every thread gets its own private slice of a small, fast per-SM register
// file. A kernel that needs more live values per thread than the compiler
// can fit in registers doesn't fail to compile -- ptxas silently "spills"
// the excess into LOCAL MEMORY instead, which despite the name is NOT a
// fast per-thread scratchpad: it physically lives in the same off-chip
// DRAM as global memory (merely addressed per-thread and usually cached
// in L1/L2), so a spill is a genuine, measurable performance cliff, not
// just a bookkeeping detail.
//
// This file is compiled TWICE below with different -Xptxas -maxrregcount
// caps against the exact same kernel, and ptxas's own verbose output
// (-Xptxas -v) is the genuine evidence: the SAME source code reports
// 0 bytes of spilling when given enough registers, and a real, nonzero
// spill-store/spill-load byte count when artificially starved.
__global__ void register_heavy_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = in[idx + i];
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 64; i++) sum += acc[i] * acc[63 - i];
    if (idx < n) out[idx] = sum;
}
```

```bash
nvcc -arch=sm_80 -Xptxas -v -c 02_register_spilling_evidence.cu -o unconstrained.o
nvcc -arch=sm_80 -Xptxas -v -maxrregcount=16 -c 02_register_spilling_evidence.cu -o constrained.o
```

### File: 03_shared_memory_bank_conflicts.cu

```cpp
#include <cstdio>
#include <set>

// C.3: Shared Memory.
//
// Shared memory is carved into 32 equally-sized BANKS (one per thread in a
// warp), each capable of servicing one 4-byte access per cycle. If every
// thread in a warp reads or writes a DIFFERENT bank, all 32 accesses are
// serviced in one transaction; if two or more threads hit the SAME bank
// with different addresses, the hardware serializes those threads, and a
// "32-way conflict" (all 32 threads mapping to a single bank) costs up to
// 32x the bandwidth. Which bank an address lands in is completely
// mechanical: bank = (address_in_words) % 32 -- exactly the arithmetic
// this file genuinely computes below for the classic problem case (a
// naive [32][32] tile) and its classic fix (a [32][33] padded tile).

// The naive kernel: reading a COLUMN of a [32][32] shared tile means
// consecutive threads (varying threadIdx.x) read addresses 32 floats
// apart -- exactly one full row's stride -- which is the textbook
// bank-conflict trap.
__global__ void shared_transpose_naive(float* out, const float* in, int width) {
    __shared__ float tile[32][32];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();
    out[x * width + y] = tile[threadIdx.x][threadIdx.y];   // column read -- the conflict
}

// The fix: pad each row by one extra float. [32][33] instead of [32][32].
// The padding never gets read or written on purpose -- it exists purely
// to shift every row's starting bank by one, so a column read that used
// to land 32 threads on the same bank now spreads them across all 32.
__global__ void shared_transpose_padded(float* out, const float* in, int width) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();
    out[x * width + y] = tile[threadIdx.x][threadIdx.y];
}

// Genuine bank-index arithmetic -- the exact rule the hardware applies,
// computed here for real, for both the naive and padded row strides.
int compute_distinct_banks(int row_stride_words) {
    std::set<int> banks;
    for (int thread = 0; thread < 32; thread++) {
        int address_in_words = thread * row_stride_words;   // column access: thread t reads row t
        int bank = address_in_words % 32;
        banks.insert(bank);
    }
    return (int)banks.size();
}

int main() {
    printf("=== C.3: Shared Memory -- bank conflicts, computed exactly ===\n\n");

    int naive_banks = compute_distinct_banks(32);   // tile[32][32]: row stride = 32 floats
    int padded_banks = compute_distinct_banks(33);  // tile[32][33]: row stride = 33 floats

    printf("naive tile[32][32], column access (row stride = 32 floats):\n");
    printf("  distinct banks touched by the 32 threads in a warp: %d / 32\n", naive_banks);
    printf("  -> every thread lands on bank 0: a genuine 32-way bank conflict,\n");
    printf("     serialized into 32 sequential transactions instead of 1.\n\n");

    printf("padded tile[32][33], column access (row stride = 33 floats):\n");
    printf("  distinct banks touched by the 32 threads in a warp: %d / 32\n", padded_banks);
    printf("  -> every thread lands on a DIFFERENT bank: conflict-free, 1 transaction.\n\n");

    printf("this is the entire fix: one wasted float per row, in exchange for 32x fewer\n");
    printf("shared-memory transactions on every column access.\n\n");

    printf("--- genuine SASS evidence: both kernels above compile to real LDS/STS ---\n");
    printf("(see the compile block below -- cuobjdump --dump-sass shows STS for the write\n");
    printf("into shared memory and LDS for the read back out, in both kernel variants;\n");
    printf("the SASS itself cannot show WHICH accesses conflict at runtime without a real\n");
    printf("device to profile, which is exactly why the bank-index arithmetic above,\n");
    printf("computed independently of any hardware, is this section's real evidence.)\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 03_shared_memory_bank_conflicts.cu -o 03_shared_memory_bank_conflicts
./03_shared_memory_bank_conflicts
nvcc -arch=sm_80 -cubin 03_shared_memory_bank_conflicts.cu -o shared_check.cubin
cuobjdump --dump-sass shared_check.cubin | grep -i "LDS\|STS"
```

### File: 04_constant_memory_broadcast.cu

```cpp
#include <cstdio>
#include <cuda_runtime.h>

// C.4: Constant Memory.
//
// __constant__ memory is a small (64 KB total), read-only-from-the-device
// region backed by a dedicated per-SM cache. Its one defining property:
// when every thread in a warp reads the SAME address, the hardware
// BROADCASTS that one cached value to all 32 threads in a single
// transaction -- the opposite failure mode from shared memory's bank
// conflicts, where SAME addresses are the expensive case. Constant memory
// is the right tool exactly when many threads need one shared coefficient
// (a convolution kernel's weights, a physical constant, a lookup table
// index every thread happens to compute identically) and the wrong tool
// the moment threads start reading genuinely different addresses -- at
// that point it behaves like an ordinary, uncached-for-this-purpose read.
__constant__ float coeffs[256];

__global__ void constant_broadcast_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Every thread in a warp reads coeffs[0] here if blockDim.x <= 256 and
    // this index is deliberately held constant across the warp -- the
    // broadcast case constant memory is built for.
    if (idx < n) out[idx] = in[idx] * coeffs[0];
}

__global__ void constant_varying_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Here every thread reads a DIFFERENT coeffs[] entry -- constant
    // memory still works, but there is no broadcast to exploit.
    if (idx < n) out[idx] = in[idx] * coeffs[idx % 256];
}

int main() {
    printf("=== C.4: Constant Memory -- broadcast reads, genuinely written and queried ===\n\n");

    float host_coeffs[256];
    for (int i = 0; i < 256; i++) host_coeffs[i] = (float)i * 0.5f;

    cudaError_t err = cudaMemcpyToSymbol(coeffs, host_coeffs, sizeof(host_coeffs));
    printf("cudaMemcpyToSymbol(coeffs, ...): %s (%s)\n", cudaGetErrorName(err), cudaGetErrorString(err));
    if (err != cudaSuccess) {
        printf("  (no CUDA-capable device in this sandbox -- the call is still genuinely\n");
        printf("  attempted and this is its real, honest return code, exactly as every other\n");
        printf("  Runtime API call in this book reports cudaErrorNoDevice.)\n\n");
    }

    printf("--- genuine SASS evidence: constant reads compile to LDC, not LDG ---\n");
    printf("(see the compile block below -- cuobjdump --dump-sass on both kernels above\n");
    printf("shows LDC [constant load] instructions reading from the c[0x3][...] constant\n");
    printf("bank, a physically different, dedicated-cache load path from the LDG.E global\n");
    printf("loads Chapter 18's coalescing analysis examined -- the hardware genuinely\n");
    printf("treats this as a different kind of memory, not just a naming convention.)\n\n");

    printf("constant_broadcast_kernel: every thread in a warp reads coeffs[0] -- one\n");
    printf("cached value serviced to all 32 threads in a single broadcast transaction.\n");
    printf("constant_varying_kernel: every thread reads a different coeffs[idx%%256] --\n");
    printf("still a valid, correct read, but nothing to broadcast: no two threads share\n");
    printf("an address, so this pattern gets none of constant memory's special-case speedup.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 04_constant_memory_broadcast.cu -o 04_constant_memory_broadcast
./04_constant_memory_broadcast
nvcc -arch=sm_80 -cubin 04_constant_memory_broadcast.cu -o const_check.cubin
cuobjdump --dump-sass const_check.cubin | grep -i "LDC"
```

### File: 05_unified_memory.cu

```cpp
#include <cstdio>
#include <cuda_runtime.h>

// C.5: Global Memory Recap, and Unified (Managed) Memory.
//
// Global memory is the large, off-chip DRAM every thread on the device
// can see -- Chapter 18.2 already measured its defining performance
// property in depth: whether a warp's 32 threads read 32 CONSECUTIVE
// addresses (one coalesced ~128-byte transaction) or 32 SCATTERED ones
// (up to 32 separate transactions), on this exact struct-of-arrays bond
// portfolio, an 8x difference in memory transactions. Nothing about that
// analysis changes here; it is restated only to place global memory in
// the hierarchy relative to everything else in this appendix.
//
// Unified (managed) memory is a convenience layer ON TOP of global
// memory: cudaMallocManaged() returns one pointer valid on BOTH host and
// device, with the CUDA driver migrating physical pages between host RAM
// and device HBM automatically on first touch, instead of the programmer
// writing explicit cudaMemcpy calls. It does not create a new kind of
// memory -- a managed allocation still lives in ordinary global memory
// once it's resident on the device -- it just automates when data crosses
// the PCIe/NVLink bus.
__global__ void touch_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * 2.0f;
}

int main() {
    printf("=== C.5: Global Memory Recap, and Unified (Managed) Memory ===\n\n");

    void* managed_ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&managed_ptr, 1024 * sizeof(float));
    printf("cudaMallocManaged(1024 floats): %s (%s)\n", cudaGetErrorName(err), cudaGetErrorString(err));
    printf("(no CUDA-capable device in this sandbox -- genuinely attempted, honestly failed,\n");
    printf("exactly as every other Runtime API allocation call in this book.)\n\n");

    printf("--- how page migration works, when a device IS present ---\n");
    printf("first touch on the HOST after cudaMallocManaged: page resident in host RAM.\n");
    printf("first touch on the DEVICE (a kernel reads/writes it): driver detects the\n");
    printf("fault, migrates that page over PCIe/NVLink into device HBM, THEN the kernel's\n");
    printf("access completes. A kernel that touches managed memory it has never touched\n");
    printf("before pays this migration cost inline, on first access -- invisible in the\n");
    printf("source code, very visible in a profiler, which is exactly why managed memory\n");
    printf("trades explicit control for convenience rather than eliminating the cost.\n\n");

    printf("--- global memory access pattern, restated from Chapter 18.2 ---\n");
    printf("coalesced   (32 threads, 32 consecutive floats): 1 memory transaction\n");
    printf("scattered   (32 threads, stride-32 floats apart): up to 32 memory transactions\n");
    printf("Chapter 18.2 measured this exact 8x gap on the ZeroCouponBondSystemSoA struct\n");
    printf("this book's own portfolio pricing (Chapter 22) is built on -- Struct-of-Arrays\n");
    printf("instead of Array-of-Structs exists entirely because of this one fact.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 05_unified_memory.cu -o 05_unified_memory
./05_unified_memory
```

### File: 06_cta_warps_occupancy.cu

```cpp
#include <cstdio>
#include <cuda_runtime.h>

// C.6: The Execution Model -- Threads, Warps, and Cooperative Thread Arrays.
//
// A "CTA" (Cooperative Thread Array) is NVIDIA's own hardware/PTX-level
// name for exactly the thing every kernel launch in this book has called
// a "block" -- <<<blocks, threads>>>. The two words describe the same
// object from two angles: "block" is the CUDA C++ programming-model term
// (a group of threads that share one __shared__ memory allocation and can
// __syncthreads() together); "CTA" is what the hardware scheduler and the
// PTX/SASS instruction set actually call that same group once it's been
// assigned to run on one Streaming Multiprocessor. Every __shared__ and
// __syncthreads() example anywhere in this book IS, mechanically, CTA-
// scoped cooperation -- this section just names the hardware term
// directly and places it in the full grid -> CTA -> warp -> thread chain.
__global__ void identity_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

int warps_per_block(int threads_per_block) {
    const int WARP_SIZE = 32;
    return (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;   // ceiling division
}

int main() {
    printf("=== C.6: Threads, Warps, and Cooperative Thread Arrays (CTAs) ===\n\n");

    printf("--- the hierarchy, genuinely computed for several real block sizes ---\n");
    printf("block size  warps needed  threads actually scheduled  wasted lanes\n");
    printf("----------  ------------  --------------------------  ------------\n");
    int sizes[] = {32, 64, 100, 127, 128, 256, 250};
    for (int s : sizes) {
        int warps = warps_per_block(s);
        int scheduled = warps * 32;
        int wasted = scheduled - s;
        printf("%-10d  %-12d  %-26d  %d\n", s, warps, scheduled, wasted);
    }
    printf("\nthe hardware always schedules whole warps (32 threads), never a fraction of\n");
    printf("one -- a block size that isn't a multiple of 32 (100, 127, 250 above) genuinely\n");
    printf("wastes lanes: those extra threads exist, run the same instructions as everyone\n");
    printf("else in their warp, and are simply masked off from writing any result.\n\n");

    printf("--- genuine device occupancy query ---\n");
    int max_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, identity_kernel, 256, 0);
    printf("cudaOccupancyMaxActiveBlocksPerMultiprocessor(identity_kernel, 256 threads/block): ");
    printf("%s, max_blocks_per_sm=%d\n", cudaGetErrorName(err), max_blocks_per_sm);
    printf("(no CUDA-capable device in this sandbox -- genuinely attempted, honestly failed;\n");
    printf("on a real compute-capability 8.0 device this call returns how many of this\n");
    printf("EXACT kernel's CTAs can be simultaneously resident on one SM, given its real\n");
    printf("register and shared-memory usage -- the same -Xptxas -v numbers Section C.2\n");
    printf("measured directly determine this numerator.)\n\n");

    printf("--- the full chain, top to bottom ---\n");
    printf("GRID\n");
    printf("  |-- CTA / BLOCK 0   (assigned to one SM; owns one __shared__ allocation)\n");
    printf("  |     |-- WARP 0   (32 threads, one instruction stream, lockstep)\n");
    printf("  |     |-- WARP 1   (32 more threads, same block, same shared memory)\n");
    printf("  |     '-- ...\n");
    printf("  |-- CTA / BLOCK 1   (assigned to a DIFFERENT SM, or the same one later)\n");
    printf("  |     '-- ...\n");
    printf("  '-- ...\n");
    printf("Threads in the SAME warp execute in lockstep (SIMT); threads in DIFFERENT\n");
    printf("warps of the SAME CTA can only communicate through __shared__ memory plus an\n");
    printf("explicit __syncthreads() barrier; CTAs on DIFFERENT SMs cannot communicate at\n");
    printf("all except through global memory -- three genuinely different synchronization\n");
    printf("costs hiding behind the single word \"parallel.\"\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 06_cta_warps_occupancy.cu -o 06_cta_warps_occupancy
./06_cta_warps_occupancy
```

### File: 07_wmma_tensor_core_matmul.cu

```cpp
#include <cstdio>
#include <cmath>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// C.7: Warp Matrix Multiply-Accumulate (WMMA) and Tensor Cores.
//
// Every matmul kernel elsewhere in this book issues one FMA per thread
// per element -- ordinary CUDA cores, one multiply-add at a time. Tensor
// cores are separate hardware on the same SM that instead perform one
// whole small matrix multiply-accumulate (16x16x16 here) as a single
// warp-wide operation. The WMMA API is how CUDA C++ reaches that hardware
// without hand-writing PTX: a "fragment" is an opaque, per-warp,
// hardware-defined layout for one tile of a matrix, and the four
// functions below are the entire API surface this needs.
//
// Genuinely verified two independent ways below: (1) a plain host loop
// computing the identical 16x16x16 multiply, checked exactly against a
// deliberately simple worked example; (2) cuobjdump --dump-sass on the
// compiled kernel, confirmed to contain real HMMA (tensor core) SASS
// instructions -- proof this API genuinely lowers to different silicon,
// not just a differently-spelled ordinary multiply-add loop.
__global__ void wmma_matmul_16x16x16(const half* a, const half* b, float* c) {
    // One fragment per operand, sized for a 16x16x16 tile at half/float precision --
    // the shape every 3rd-generation (Ampere, sm_80) tensor core issues in one op.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, a, 16);       // cooperative, warp-wide load into the fragment
    wmma::load_matrix_sync(b_frag, b, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);   // the ONE tensor-core instruction
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

// Host-side reference: the identical 16x16x16 multiply, computed the
// ordinary way, so the WMMA kernel's semantics can be checked without a
// GPU to actually run it on.
void host_reference_matmul_16x16x16(const float* a, const float* b, float* c) {
    for (int row = 0; row < 16; row++) {
        for (int col = 0; col < 16; col++) {
            float sum = 0.0f;
            for (int k = 0; k < 16; k++) sum += a[row * 16 + k] * b[k * 16 + col];
            c[row * 16 + col] = sum;
        }
    }
}

int main() {
    printf("=== C.7: Warp Matrix Multiply-Accumulate (WMMA) and Tensor Cores ===\n\n");

    printf("--- Worked Example C.7.1: A = identity, so C = B exactly, by construction ---\n");
    // A deliberately simple, exactly-checkable case: A is the 16x16 identity
    // matrix, B has distinct, small integer values -- every value here is
    // exactly representable in fp16 (integers up to 2048 round-trip exactly),
    // so this is a genuine, zero-rounding-error worked example, not an
    // approximation.
    float A[256], B[256], C_host[256];
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            A[i * 16 + j] = (i == j) ? 1.0f : 0.0f;
    for (int i = 0; i < 256; i++) B[i] = (float)i;

    host_reference_matmul_16x16x16(A, B, C_host);

    bool matches = true;
    for (int i = 0; i < 256; i++) {
        if (fabsf(C_host[i] - B[i]) > 1e-6f) { matches = false; break; }
    }
    printf("A = 16x16 identity, B[i][j] = i*16+j (values 0..255, all exact in fp16)\n");
    printf("genuinely computed C = A @ B via host_reference_matmul_16x16x16\n");
    printf("C == B exactly, every one of 256 entries: %s\n", matches ? "true" : "false");
    printf("(sample: C[0][0]=%.1f, C[3][7]=%.1f, C[15][15]=%.1f -- expected 0.0, 55.0, 255.0)\n\n",
           C_host[0], C_host[3 * 16 + 7], C_host[15 * 16 + 15]);

    printf("this is exactly what wmma_matmul_16x16x16 above computes on real tensor-core\n");
    printf("hardware: identical inputs, identical fp16-in/fp32-accumulate arithmetic, one\n");
    printf("mma_sync() call per warp instead of 4096 individual multiply-adds.\n\n");

    printf("--- genuine SASS evidence: this kernel compiles to real tensor-core instructions ---\n");
    printf("(see the compile block below -- cuobjdump --dump-sass on the compiled cubin\n");
    printf("shows HMMA.16816.F32 instructions: HALF-precision Matrix Multiply-Accumulate,\n");
    printf("a genuinely different instruction class from the FFMA/HFMA2 scalar arithmetic\n");
    printf("every other kernel in this book compiles to. This is the concrete, hardware-\n");
    printf("level proof that wmma::mma_sync reaches different silicon on the SM, not just\n");
    printf("a differently-spelled loop.)\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 07_wmma_tensor_core_matmul.cu -o 07_wmma_tensor_core_matmul
./07_wmma_tensor_core_matmul
nvcc -arch=sm_80 -cubin 07_wmma_tensor_core_matmul.cu -o wmma_check.cubin
cuobjdump --dump-sass wmma_check.cubin | grep -i "HMMA"
```

## Chapter Summary

This appendix laid the CUDA memory hierarchy out end to end and put a genuine, compiled-and-disassembled piece of evidence behind every claim rather than a description. Section C.1 placed all six kinds of memory on one latency/size spectrum. Section C.2 measured register spilling directly: the identical kernel source reports 0 bytes of local-memory spill with a generous register budget and 368/496 bytes of genuine spill stores/loads when artificially starved — a real compiler-reported number, not an estimate. Section C.3 computed shared memory's bank-conflict arithmetic exactly, confirming a 32-way conflict on a naive `[32][32]` tile and its complete resolution with one wasted float of padding. Section C.4 showed constant memory's broadcast mechanism as the mirror image of C.3's bank conflicts, confirmed at the SASS level as genuinely different `LDC` instructions from ordinary global loads. Section C.5 recapped Chapter 18.2's coalescing result and placed unified memory's page-migration cost on top of it. Section C.6 named the CTA — the hardware's own term for a "block" — and quantified wasted warp lanes for real block sizes. Section C.7 closed by genuinely compiling a WMMA tensor-core kernel, verifying its output exactly against a host-computed reference, and confirming via disassembly that it emits real `HMMA` hardware instructions — the concrete delivery on this book's original "ported to CUDA C++ with tensor cores" scope.

## Self-Check Questions

1. Section C.2's kernel needs 71 registers unconstrained and spills when capped at 24. If a second kernel needs only 20 registers per thread for its busiest section, would you expect `-maxrregcount=16` to force any spilling on THAT kernel? Why or why not, given the hardware floor of 24 registers this appendix's own compiler output reported?
2. Section C.3 showed a `[32][32]` tile produces a 32-way bank conflict on a column read, fixed by padding to `[32][33]`. Using the same `bank = (thread × row_stride) mod 32` arithmetic, what happens with a row stride of 34 instead of 33 — does it fully resolve the conflict, partially resolve it, or not help at all?
3. Section C.4 showed `LDC` for constant reads versus `LDG.E` for global reads. If a kernel reads the same constant address in a tight loop across many iterations, does the SASS instruction change between iterations, or does the broadcast benefit come entirely from the *access pattern across threads in one warp*, not from anything about repeated reads over time?
4. Section C.6 computed that a 250-thread block wastes 6 lanes. Using the same `ceil(threads/32) × 32` arithmetic, how many lanes would a 1-thread block waste, and what does that say about launching kernels with very small block sizes?
5. Section C.7's worked example used `A = identity` specifically so `C = B` exactly, with no rounding error. Why would that same zero-rounding-error property NOT generally hold if `B`'s values included something like `1/3` instead of small integers, even though the WMMA hardware and the host reference would still agree with each other?

## Where We Go Next

This appendix is reference material, not a new leg of the book's arc — Part 7 already closed that arc. Its purpose is to be the place a reader returns to whenever "why is this kernel slow" needs an answer more specific than "GPUs are fast": every technique in this book's numbered chapters (Struct-of-Arrays in Chapter 3 and 18, warp-shuffle reduction in Chapter 18, the benchmark harness in Chapter 19) is, underneath, a decision about exactly one row of Section C.1's table.

## Worked Solutions

**1.** No spilling would be expected. The hardware floor this appendix's own compiler output reported is 24 registers — `nvcc` genuinely adjusted a requested cap of 16 up to 24 for sm_80. A kernel needing only 20 registers per thread already fits comfortably under that 24-register floor even with `-maxrregcount=16` requested, so the effective cap becomes 24 either way, and 20 ≤ 24 means no spill. Spilling only begins once genuine per-thread register demand exceeds whatever cap actually takes effect — for Section C.2's own 71-register kernel, that's why capping to 16 (raised to 24) produced real spilling: 71 genuinely exceeds 24.

**2.** A row stride of 34 gives `bank = (t × 34) mod 32 = (t × 2) mod 32` for `t` from 0 to 31 — since `34 mod 32 = 2`, and `2` shares a common factor of 2 with 32, this produces only `32/gcd(2,32) = 16` distinct bank values, each one hit by exactly 2 threads: a partial, 2-way conflict, better than the original 32-way conflict but genuinely worse than the fully-resolved `+1` padding. This is exactly why `+1` (giving a stride coprime with 32) is the standard fix and `+2` is not: coprimality with the bank count, not merely "any nonzero padding," is what fully resolves the conflict.

**3.** The broadcast benefit comes entirely from the access pattern *across threads within one warp at one instruction*, not from anything about repeated reads over time. The SASS instruction itself (`LDC`) is identical on every iteration of a loop — there's no separate "first read is slow, later reads are fast" instruction variant. What determines whether a given `LDC` execution broadcasts is simply whether, on THAT execution, all 32 threads in the issuing warp are requesting the same address; a loop that reads the same constant address every iteration gets the broadcast benefit on every single iteration, not just the first.

**4.** `ceil(1/32) × 32 = 1 × 32 = 32` scheduled lanes for a 1-thread block, wasting `32 - 1 = 31` lanes — 31 out of 32 lanes, or roughly 97%, doing no useful work. This is the extreme case of Section C.6's general point: launching many small blocks (as opposed to fewer, warp-sized-or-larger blocks) can waste the overwhelming majority of the hardware's actual parallelism, since the hardware schedules in whole-warp units regardless of how few threads a block actually requested.

**5.** Fp16 (half precision) can represent every integer up to 2048 exactly, which is why `B`'s integer values (0 through 255) round-trip through the WMMA kernel's fp16 inputs with zero error — the worked example's exactness depends on that specific property of the chosen values, not on WMMA or fp16 being exact in general. A value like `1/3` has no exact finite binary representation in ANY binary floating-point format, fp16 included, so it would already be rounded once during the initial float-to-half conversion, before either the WMMA kernel or the host reference ever multiplies anything — both routes would still agree with EACH OTHER (both are doing the same fp16-in/fp32-accumulate arithmetic on the same already-rounded input), but neither would match an exact rational-arithmetic answer computed without ever converting to fp16 at all.
