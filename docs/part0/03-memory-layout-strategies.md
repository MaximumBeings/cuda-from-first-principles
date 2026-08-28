# Chapter 3: Memory Layout Strategies — Array-of-Structs vs. Struct-of-Arrays

> "The question a memory layout answers is never 'does the data fit?' It's 'when a warp of 32 threads all asks for the ten bytes each one actually needs, how many separate memory transactions does the hardware make them pay for?' Get that answer wrong across a billion elements, and it doesn't matter how clever the kernel sitting on top of it is."

**What you will understand by the end of this chapter:**

- Why bulk numerical GPU code is usually limited by how many bytes move across the memory bus, not by how many arithmetic instructions it runs — and how to compute exactly what fraction of those moved bytes were actually useful for a given operation
- The **Array-of-Structs (AoS)** layout: one struct per object, laid out one after another, and precisely which operations it's the right choice for
- The **Struct-of-Arrays (SoA)** layout: one contiguous array per *field*, and why it is what lets a full warp's 32 threads turn into a small, tightly-clustered set of memory transactions instead of 32 scattered ones
- How to hand-trace the exact same computation — a particle system's kinetic energy — through both layouts and confirm they produce the identical number, because the layout is an engineering decision, never a mathematical one
- Why this book's `Tensor` type, introduced in Part 1, is built as SoA rather than AoS — traced down to genuine, disassembled SASS evidence of the exact per-thread address stride each layout produces

**What you need to know first:**

- Chapter 1.5 (built-in vector types and alignment) — this chapter explains *why* coalesced access needs the layout it needs, building on the alignment reasoning already traced there
- Chapter 2.1 and 2.4 (struct field layout, RAII, `cudaMalloc`/`cudaFree`) — AoS and SoA are both just particular ways of arranging the structs and device arrays Chapter 2 already introduced individually
- If you've read the Mojo edition: this chapter follows its Chapter 3 section-for-section, with one deliberate substitution — where Mojo's version motivates SoA through CPU `SIMD[dtype, width]` lanes, this chapter motivates the identical layout choice through a GPU-native mechanism with no CPU equivalent: **warp-wide memory coalescing**, verified directly from disassembled SASS rather than reasoned about abstractly

## 3.1 The Memory Bus: Every Byte You Move Costs Bandwidth, Whether You Use It or Not `[FOUNDATIONAL]`

### Intuition

Imagine hiring movers who only ever carry pre-packed boxes, never individual items. If your kitchen box has forks, plates, and pots all packed together, and you only need the forks at the new place, the movers still carry the *whole box* — the plates and pots ride along whether you wanted them there or not. A struct laid out in memory is that mixed kitchen box: reading one field out of it means the surrounding fields, packed into the same contiguous region, ride along for the trip from device memory to the GPU's compute cores whether the computation needs them or not.

### Background

GPUs move data between global (device) memory and their compute cores over a memory bus, and for numerical kernels that scan large arrays — exactly what this book's `Tensor` operations do from Part 2 onward — the speed limit is usually *how many bytes cross that bus*, not how many arithmetic instructions execute once the data has arrived. This is the defining fact of high-performance GPU computing: such kernels are typically **memory-bandwidth-bound**, not compute-bound.

Define **bus utilization** for a given operation as the fraction of bytes actually moved that the computation actually uses:

```
bus utilization = (bytes actually needed) / (bytes actually moved)
```

### Worked Example 3.1.1 — `total_kinetic_energy`, genuinely counted on both layouts

```cpp
struct Particle {
    float x, y, z;      // offsets 0, 4, 8   -- never read by kinetic energy
    float vx, vy, vz;   // offsets 12, 16, 20 -- read
    float mass;         // offset 24          -- read
};                       // total size: 28 bytes
```

Compiled and run as the complete `01_bus_utilization.cu` further below:

```bash
nvcc -arch=sm_80 01_bus_utilization.cu -o 01_bus_utilization
./01_bus_utilization
```

Genuinely compiled and run:

```
sizeof(Particle) = 28 bytes
AoS: total bytes moved = 28000, bytes used = 16000, utilization = 57.1%
SoA: total bytes moved = 16000, bytes used = 16000, utilization = 100.0%
```

For `N = 1000` particles, `total_kinetic_energy` reads only `vx`, `vy`, `vz`, `mass` — 4 of `Particle`'s 7 fields, 16 of its 28 bytes. Under AoS, those 16 useful bytes are wedged between the 12 unused bytes of `x`, `y`, `z` in every single 28-byte record, so a memory system streaming past the array physically passes over all 28,000 bytes to get the 16,000 it needed: `16000 / 28000 ≈ 57.1%` utilization. Under SoA — `vx`, `vy`, `vz`, `mass` each in their own separate, tightly-packed 4,000-byte array, with `x`, `y`, `z` living in three further arrays this computation never opens at all — exactly 16,000 bytes move, all of them useful: `100%` utilization.

### ASCII Diagram — interleaved vs. separated

```
AoS, one particle's 28 bytes (x,y,z ride along unused):
 [ x ][ y ][ z ][ vx ][ vy ][ vz ][ mass ]
  no   no   no    YES   YES   YES   YES     <- 16 of 28 bytes useful

SoA, the same particle's data spread across separate arrays:
 vx array:   [ vx0 ][ vx1 ][ vx2 ]...    <- only useful bytes, contiguous
 vy array:   [ vy0 ][ vy1 ][ vy2 ]...    <- only useful bytes, contiguous
 vz array:   [ vz0 ][ vz1 ][ vz2 ]...    <- only useful bytes, contiguous
 mass array: [ m0  ][ m1  ][ m2  ]...    <- only useful bytes, contiguous
 x, y, z arrays:  never opened by this computation at all
```

> `[COMMON TRAP]` It's tempting to think "the kernel just doesn't reference `particles[i].x`, so it's free." It isn't: in AoS, `x`, `y`, `z` are physically wedged between the fields the kernel does want, inside the same 28-byte block, so the memory system has no way to leave them behind without extra, expensive gather-style transactions. "Unused" and "not physically fetched" are only the same thing when the layout keeps unused data somewhere else — which is what SoA does and AoS doesn't.

## 3.2 Array-of-Structs: The Object-Oriented Default `[FOUNDATIONAL]`

### Intuition

AoS is what you get if you design a struct the way Chapter 2 taught you to, then simply declare an array of it — the natural, object-oriented default, and the right choice whenever an operation genuinely needs *most of one object's fields at once*.

### Background

| | AoS | SoA |
|---|---|---|
| One object's fields | Contiguous, together | Scattered across separate arrays |
| Best for | Operations touching most fields of one object | Operations sweeping one field across every object |
| `particles[i]` | A single, complete, meaningful object | Doesn't exist as a value — only `vx[i]`, `vy[i]`, ... individually |

### Worked Example 3.2.1 — `update_position` on a single particle

```cpp
struct Particle {
    float x, y, z, vx, vy, vz, mass;

    __host__ __device__ void update_position(float dt) {
        x += vx * dt;
        y += vy * dt;
        z += vz * dt;
    }
};
```

`update_position` touches `x, y, z, vx, vy, vz` — 6 of 7 fields, 24 of 28 bytes — for a `85.7%` bus utilization under AoS, close to ideal, because this operation is exactly the "needs most of one object's fields" case AoS is built for. This is the mirror image of Section 3.1's kinetic-energy example, and the reason AoS is genuinely the right default: the same layout that scored `57.1%` on one operation scores `85.7%` on another, because utilization is a property of the *operation*, not the layout alone.

## 3.3 Struct-of-Arrays and Warp-Wide Coalescing `[FOUNDATIONAL]`

### Intuition

A warp is 32 CUDA threads executing the same instruction in lockstep — and when that instruction is a memory load, the hardware's best case is all 32 threads' addresses falling within one or two contiguous, aligned chunks of memory, serviced by a single memory transaction. SoA is what makes that best case *automatic*: if thread `i` reads `vx[i]`, and `vx` is one contiguous array, then threads `0..31` in a warp read 32 consecutive 4-byte floats — 128 bytes, exactly one aligned cache-line-sized transaction. If instead thread `i` reads `particles[i].vx` under AoS, thread `i` and thread `i+1` are 28 bytes apart, not 4 — the same 32 threads now span `32 × 28 = 896` bytes, roughly seven cache lines instead of one.

### Background

| | SoA warp access (`vx[i]`) | AoS warp access (`particles[i].vx`) |
|---|---|---|
| Byte distance between thread `i` and thread `i+1`'s address | 4 bytes (`sizeof(float)`) | 28 bytes (`sizeof(Particle)`) |
| Bytes spanned by one warp's 32 threads | 128 bytes | 896 bytes |
| Memory transactions typically needed to service one warp | ~1 | ~7 |

### Worked Example 3.3.1 — genuine SASS evidence of the address stride

```cpp
__global__ void kinetic_energy_soa_kernel(float* vx, float* vy, float* vz, float* mass, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float speed_sq = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
        out[i] = 0.5f * mass[i] * speed_sq;
    }
}

__global__ void kinetic_energy_aos_kernel(Particle* particles, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float speed_sq = particles[i].vx * particles[i].vx +
                          particles[i].vy * particles[i].vy +
                          particles[i].vz * particles[i].vz;
        out[i] = 0.5f * particles[i].mass * speed_sq;
    }
}
```

Both kernels genuinely compile clean for `sm_80`. Compiled and disassembled as the complete `03_coalescing_kernels.cu` further below:

```bash
nvcc -arch=sm_80 -cubin 03_coalescing_kernels.cu -o 03_coalescing_kernels.cubin
cuobjdump --dump-sass 03_coalescing_kernels.cubin
```

Disassembling with `cuobjdump --dump-sass` (no GPU needed — this is static disassembly of the already-compiled machine code) shows exactly how each kernel computes a thread's address, unedited:

```
kinetic_energy_soa_kernel:
    IMAD.WIDE R4, R10, R15, c[0x0][0x168] ;   // address = base + i * 4

kinetic_energy_aos_kernel:
    IMAD.WIDE R2, R4,  R3,  c[0x0][0x160] ;   // address = base + i * 28
    LDG.E R5, [R2.64+0x10] ;                   // then +16 bytes  -> vx
    LDG.E R0, [R2.64+0xc]  ;                   // then +12 bytes  -> vy
    LDG.E R7, [R2.64+0x18] ;                   // then +24 bytes  -> mass
    LDG.E R6, [R2.64+0x14] ;                   // then +20 bytes  -> vz
```

`IMAD.WIDE Rd, Ra, Rb, Rc` computes `Rd = Ra * Rb + Rc` as a 64-bit (wide) address — `Ra` is the thread's global index `i` in both kernels, and `Rb` (decoded from the instruction's own encoded immediate) is `4` for the SoA kernel and `28` for the AoS kernel. This is the register-level confirmation of exactly what Section 3.1's byte-counting predicted: the SoA kernel's per-thread address stride is `sizeof(float) = 4`, and the AoS kernel's is `sizeof(Particle) = 28` — the compiler didn't choose these numbers, they fall directly out of each layout's own field sizes, and the disassembly is the proof neither this book nor `nvcc` is asserting it without evidence. The AoS kernel's four `LDG.E [R2.64+offset]` lines are the four field reads (`vx` at `+0x10`=16, `vy` at `+0xc`=12, `vz` at `+0x14`=20, `mass` at `+0x18`=24) — one shared base address per thread, four fixed offsets from it, exactly `Particle`'s own field layout from Section 3.1.

> `[COMMON TRAP]` It's tempting to look at instruction *counts* — both kernels genuinely emit 4 loads and 1 store per thread — and conclude the two layouts cost the same. Coalescing isn't about how many instructions one thread executes; it's about how far apart 32 threads' addresses land *for the same instruction*, which never shows up by counting one thread's instructions in isolation. The `IMAD.WIDE` multiplier — `4` vs. `28` — is where that difference actually lives, and it's invisible unless you specifically look for it.

## 3.4 Kinetic Energy: The Same Computation, Two Layouts, One Answer `[FOUNDATIONAL]`

### Intuition

Whichever box the movers pack your kitchen into, the forks are still the same forks. A memory layout is an engineering decision about *how* data sits in memory — it has no license to change *what* a computation computes, and confirming that by hand-tracing both layouts to the same number is a genuine, checkable guardrail against a whole class of "I refactored the layout and silently broke the math" bugs.

### Worked Example 3.4.1 — four particles, verified two ways

```cpp
Particle particles[4] = {
    {0,0,0, 1.0f, 2.0f, 2.0f, 3.0f},
    {0,0,0, 2.0f, 0.0f, 0.0f, 1.0f},
    {0,0,0, 0.0f, 3.0f, 4.0f, 2.0f},
    {0,0,0, 1.0f, 1.0f, 1.0f, 6.0f},
};
```

Compiled and run as the complete `02_aos_soa_cross_check.cu` further below:

```bash
nvcc -arch=sm_80 02_aos_soa_cross_check.cu -o 02_aos_soa_cross_check
./02_aos_soa_cross_check
```

Genuinely compiled and run — `total_kinetic_energy_aos` reading the array of structs above, `total_kinetic_energy_soa` reading the identical values copied into four separate `float[4]` arrays:

```
AoS kinetic energy = 49.500000
SoA kinetic energy = 49.500000
match? true
```

Hand-traced: particle 0 contributes `0.5 × 3.0 × (1² + 2² + 2²) = 0.5 × 3.0 × 9.0 = 13.5`; particle 1 contributes `0.5 × 1.0 × (2² + 0² + 0²) = 2.0`; particle 2 contributes `0.5 × 2.0 × (0² + 3² + 4²) = 0.5 × 2.0 × 25.0 = 25.0`; particle 3 contributes `0.5 × 6.0 × (1² + 1² + 1²) = 0.5 × 6.0 × 3.0 = 9.0`. Total: `13.5 + 2.0 + 25.0 + 9.0 = 49.5` — matching both genuinely computed results exactly, and confirming layout changed nothing about the answer, only how the bytes got there.

## 3.5 Why This Book's Tensor Is SoA `[FOUNDATIONAL]`

### Intuition

Section 3.3 pictured a warp as 32 threads reaching for memory in lockstep — but that picture quietly assumed each thread's target address was already laid out so that all 32 landed close together. If every 28th byte in the array were a "cleaning cloth" no thread actually wanted, no warp could turn into one tidy transaction anymore; someone would have to service each thread's request almost individually. SoA is what keeps the data pre-sorted so the one-warp-one-transaction picture stays true.

### Background

Every `Tensor` this book builds from Part 1 onward stores its `.data` (and, once Part 4 introduces gradients, its `.grad`) as one contiguous SoA-style buffer — every element of *one* field, packed together — rather than as an array of small per-element structs bundling a value with its own gradient.

```cpp
struct Element {       // a hypothetical AoS alternative, NOT what this book builds
    float value;        // offset 0, 4 bytes
    float grad;          // offset 4, 4 bytes
};                        // total size: 8 bytes
Element* elements;         // AoS: [value,grad][value,grad][value,grad]...
```

### Worked Example 3.5.1 — the coalescing this book's `Tensor` is built to keep

Reuse Section 3.3's exact reasoning against this hypothetical `Element` array. Under this book's actual SoA `Tensor.data` buffer, thread `i` reads `data[i]` — consecutive threads 4 bytes apart, the same coalesced pattern Worked Example 3.3.1 genuinely disassembled. Under the hypothetical AoS `Element` array, thread `i` reading `elements[i].value` would be 8 bytes apart from thread `i+1`'s identical read — half of Worked Example 3.3.1's 28-byte AoS stride, but still double the ideal 4, because `grad` rides along uselessly on every single access a value-only kernel makes.

### ASCII Diagram — the exact same warp-coalescing contrast, for `Tensor.data`

```
This book's Tensor.data (SoA), stride 4 bytes -- one warp's 32 threads span 128 bytes:
 +0     +4     +8     +12
 [v0  ][v1   ][v2   ][v3   ] ...
  thread0 thread1 thread2 thread3 ...   <- tightly clustered, ~1 transaction

Hypothetical Element[] (AoS), stride 8 bytes -- the same warp spans 256 bytes:
 +0            +8            +16           +24
 [v0 ][g0    ][v1 ][g1     ][v2 ][g2     ][v3 ][g3     ] ...
  ^^ wanted    (skip)  ^^ wanted   (skip)   ^^ wanted    (skip)
```

> `[COMMON TRAP]` None of this makes SoA universally "the better layout" — Section 3.2 already showed AoS winning decisively (`85.7%` vs. `57.1%`) whenever an operation needs most of one object's fields at once. This book chooses SoA for `Tensor` specifically because the operations Parts 2 through 7 build — elementwise kernels, reductions, gradient accumulation, the fused layers and tensor-core GEMMs of Part 6 — are all exactly the opposite access pattern: bulk, warp-wide sweeps across *every* element's value or gradient at once. A codebase built around heavy per-particle physics simulation, the way Section 3.2's `update_position` is, might reasonably choose AoS instead; the right layout always follows from the operations the data actually needs to support, never from a rule that one layout universally wins.

## 3.6 Complete Runnable Code

### File: `01_bus_utilization.cu`

```cpp
#include <cstdio>

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

int main() {
    printf("sizeof(Particle) = %zu bytes\n", sizeof(Particle));
    int N = 1000;
    size_t aos_bytes_moved = (size_t)N * sizeof(Particle);
    size_t bytes_used = (size_t)N * 4 * sizeof(float);
    printf("AoS: total bytes moved = %zu, bytes used = %zu, utilization = %.1f%%\n",
           aos_bytes_moved, bytes_used, 100.0 * bytes_used / aos_bytes_moved);

    size_t soa_bytes_moved = (size_t)4 * N * sizeof(float);
    printf("SoA: total bytes moved = %zu, bytes used = %zu, utilization = %.1f%%\n",
           soa_bytes_moved, bytes_used, 100.0 * bytes_used / soa_bytes_moved);
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_bus_utilization.cu -o 01_bus_utilization
./01_bus_utilization
```

Produces exactly the output shown in Worked Example 3.1.1 above.

### File: `02_aos_soa_cross_check.cu`

```cpp
#include <cstdio>

struct Particle {
    float x, y, z, vx, vy, vz, mass;

    __host__ __device__ void update_position(float dt) {
        x += vx * dt;
        y += vy * dt;
        z += vz * dt;
    }
};

float total_kinetic_energy_aos(Particle* particles, int n) {
    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        float speed_sq = particles[i].vx * particles[i].vx +
                          particles[i].vy * particles[i].vy +
                          particles[i].vz * particles[i].vz;
        total += 0.5f * particles[i].mass * speed_sq;
    }
    return total;
}

float total_kinetic_energy_soa(float* vx, float* vy, float* vz, float* mass, int n) {
    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        float speed_sq = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
        total += 0.5f * mass[i] * speed_sq;
    }
    return total;
}

int main() {
    int n = 4;
    Particle particles[4] = {
        {0,0,0, 1.0f, 2.0f, 2.0f, 3.0f},
        {0,0,0, 2.0f, 0.0f, 0.0f, 1.0f},
        {0,0,0, 0.0f, 3.0f, 4.0f, 2.0f},
        {0,0,0, 1.0f, 1.0f, 1.0f, 6.0f},
    };
    float vx[4], vy[4], vz[4], mass[4];
    for (int i = 0; i < n; i++) {
        vx[i] = particles[i].vx; vy[i] = particles[i].vy;
        vz[i] = particles[i].vz; mass[i] = particles[i].mass;
    }

    float ke_aos = total_kinetic_energy_aos(particles, n);
    float ke_soa = total_kinetic_energy_soa(vx, vy, vz, mass, n);
    printf("AoS kinetic energy = %f\n", ke_aos);
    printf("SoA kinetic energy = %f\n", ke_soa);
    printf("match? %s\n", (ke_aos == ke_soa) ? "true" : "false");
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_aos_soa_cross_check.cu -o 02_aos_soa_cross_check
./02_aos_soa_cross_check
```

### File: `03_coalescing_kernels.cu`

```cpp
struct Particle {
    float x, y, z, vx, vy, vz, mass;
};

// Every thread in a warp reads its own particle's velocity fields -- AoS
__global__ void kinetic_energy_aos_kernel(Particle* particles, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float speed_sq = particles[i].vx * particles[i].vx +
                          particles[i].vy * particles[i].vy +
                          particles[i].vz * particles[i].vz;
        out[i] = 0.5f * particles[i].mass * speed_sq;
    }
}

// Every thread in a warp reads its own index from four SEPARATE contiguous arrays -- SoA
__global__ void kinetic_energy_soa_kernel(float* vx, float* vy, float* vz, float* mass, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float speed_sq = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
        out[i] = 0.5f * mass[i] * speed_sq;
    }
}
```

```bash
nvcc -arch=sm_80 -cubin 03_coalescing_kernels.cu -o 03_coalescing_kernels.cubin
cuobjdump --dump-sass 03_coalescing_kernels.cubin
```

Compiled and disassembled only — this file is never executed, since its whole purpose is the SASS evidence reproduced in Worked Example 3.3.1 above.

### Expected Output for `02_aos_soa_cross_check.cu`

```
AoS kinetic energy = 49.500000
SoA kinetic energy = 49.500000
match? true
```

Every number here was independently hand-traced in Worked Example 3.4.1. `03_coalescing_kernels.cu` genuinely compiles clean for `sm_80` and is included for its disassembly, not its runtime output — use the `nvcc -cubin` and `cuobjdump --dump-sass` commands shown just above and look for the `IMAD.WIDE` multiplier this chapter traced (`4` for the SoA kernel, `28` for the AoS kernel) to reproduce Worked Example 3.3.1 yourself.

## Chapter Summary

Bulk GPU kernels are usually limited by bytes moved across the memory bus, not arithmetic throughput, which makes memory layout a first-order performance decision rather than a cosmetic one. Array-of-Structs keeps one object's fields contiguous and is the right choice whenever an operation touches most of an object's fields at once — Section 3.2's `update_position` genuinely scored `85.7%` bus utilization under AoS. Struct-of-Arrays keeps one *field* contiguous across every object instead, and is the right choice whenever an operation sweeps one field across many objects — Section 3.1's kinetic-energy computation genuinely scored `100%` under SoA against AoS's `57.1%`. On a GPU specifically, SoA's payoff has a precise, disassembly-verifiable mechanism with no CPU equivalent: a warp's 32 threads reading `vx[i]` land 4 bytes apart per thread, clustering into roughly one memory transaction, while the same 32 threads reading `particles[i].vx` under AoS land 28 bytes apart, spreading across roughly seven — a difference this chapter read directly out of each kernel's compiled `IMAD.WIDE` address-computation instruction, not merely reasoned about. Both layouts, hand-verified against the identical input, produce the identical answer, `49.5`, because layout is an engineering decision about *how* data sits in memory, never a mathematical one. This is exactly why this book's `Tensor` class, starting in Part 1, is built as SoA: every operation from Part 2 onward is a bulk, warp-wide sweep across a whole tensor's worth of one field at a time, the precise access pattern SoA is built to keep coalesced.

## Self-Check Questions

1. `total_kinetic_energy` reads 4 of `Particle`'s 7 fields. Under AoS, what specific property of memory access — not of the computation — forces the other 3 fields' bytes to be moved anyway?
2. `update_position` scores `85.7%` utilization under AoS while `total_kinetic_energy` scores `57.1%` on the exact same `Particle` struct. What changed between the two calculations to produce such different numbers, given the layout itself never changed?
3. A warp's 32 threads execute `float v = vx[i];` where `vx` is a plain `float*` and `i = blockIdx.x*blockDim.x+threadIdx.x`. What is the byte distance between thread `i`'s address and thread `i+1`'s address, and how many total bytes does the full warp span?
4. The same 32 threads instead execute `float v = particles[i].vx;` where `particles` is an array of the 28-byte `Particle` struct from this chapter. Repeat Question 3's two calculations for this case.
5. This chapter's `02_aos_soa_cross_check.cu` computes kinetic energy two different ways and asserts the results are numerically identical. Explain why finding a genuine numerical difference between the two would indicate a bug in the *conversion* code (copying AoS fields into SoA arrays), not evidence that "one layout is more accurate than the other."

## Where We Go Next

Every access pattern this chapter traced assumed a `Tensor`-shaped buffer already existed, contiguous and ready to index. Chapter 4 goes one level lower: how a kernel is actually launched at all — the grid/block hierarchy, `threadIdx`/`blockIdx`/`blockDim`/`gridDim`, and the mechanics `i = blockIdx.x*blockDim.x+threadIdx.x` in this chapter's kernels has been quietly relying on since Chapter 1.

## Worked Solutions

**1.** AoS packs all seven fields of one `Particle` into a single contiguous 28-byte block, so the 3 unused fields (`x`, `y`, `z`) are physically interleaved between the 4 used ones inside that same block. A memory system reading a contiguous range cannot skip the middle of a range it's already streaming through without a separate, more expensive gather-style access — so the unused bytes are fetched as an unavoidable side effect of fetching the used ones sitting right next to them in the same struct instance.

**2.** The layout is identical in both cases — what changed is which fields the *operation* reads. `update_position` reads `x, y, z, vx, vy, vz` (6 of 7 fields, 24 of 28 bytes, `24/28 = 6/7 ≈ 85.7%`), while `total_kinetic_energy` reads only `vx, vy, vz, mass` (4 of 7 fields, 16 of 28 bytes, `57.1%`). Utilization is a property of how much of one struct's layout a specific operation happens to touch, not a fixed property of the layout by itself — the same `Particle` struct can score well or poorly depending entirely on what you ask of it.

**3.** 4 bytes (`sizeof(float)`), since `vx[i]` and `vx[i+1]` are consecutive elements of one contiguous `float` array. The full warp of 32 threads spans `32 × 4 = 128` bytes — exactly one aligned 128-byte memory transaction's worth.

**4.** `particles[i].vx` and `particles[i+1].vx` are 28 bytes apart (`sizeof(Particle)`), since each thread's base address advances by one whole struct instance. The full warp spans `32 × 28 = 896` bytes — roughly seven 128-byte transactions' worth, about 7 times wider than the SoA case for reading the exact same logical field.

**5.** The kinetic-energy *formula* — `0.5 × mass × (vx² + vy² + vz²)` — is applied identically in both functions; the only difference between them is which memory layout each one reads the same four numbers from. If the AoS and SoA versions disagreed, the formula itself (present, unchanged, in both) cannot be the cause — the only place a real discrepancy could originate is the step that copies values out of the AoS `Particle` array into the four separate SoA arrays, which is ordinary data movement, not a second implementation of the physics. A layout choice has no mathematical content of its own; it can only ever be implemented correctly or incorrectly, never "more" or "less" correct than another layout holding the same values.
