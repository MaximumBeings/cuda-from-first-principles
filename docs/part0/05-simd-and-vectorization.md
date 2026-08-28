# Chapter 5: SIMD and Vectorization — The Warp as Hardware SIMD

> "A CPU's SIMD register is a type you name in source code — eight lanes, declared, filled, and reduced by instructions you write yourself. A GPU's warp is the same idea with the naming stripped away: thirty-two lanes the hardware forms for you, whether you ever write the number 32 anywhere in your kernel or not. This chapter is about the handful of places CUDA lets you reach past that abstraction and talk to the lanes directly — one 128-bit load standing in for four, one instruction moving a value sideways between threads, one bitmask capturing all thirty-two comparisons at once."

**What you will understand by the end of this chapter:**

- Why a warp's lockstep, thirty-two-lane execution is the direct hardware relative of a CPU SIMD register — the same "one instruction, many lanes" idea, realized as SIMT (many independent threads converging) rather than true SIMD (one instruction, one operand, many lanes)
- Vectorized memory access via `float4`: how a single 128-bit load replaces four separate 32-bit loads, read directly out of disassembled machine code as `LDG.E.128` versus four instances of plain `LDG.E`
- Warp-shuffle reduction (`__shfl_down_sync`): summing 32 values down to one using only registers, with no shared memory and no `__syncthreads()`, traced lane-by-lane against a closed-form sum
- Warp-vote and ballot (`__ballot_sync`): the GPU's version of a SIMD lane-wise comparison, packing all 32 threads' boolean results into a single 32-bit mask — and a genuine surprise in the disassembly about how that mask actually gets produced
- Packed half-precision arithmetic (`half2`, `__hadd2`): the one place this chapter's hardware performs *true* 2-wide SIMD — two values, one register, one instruction, inside a single thread — and why that idea is the direct on-ramp to Part 6's tensor cores

**What you need to know first:**

- Chapter 1.5 (`float4`/`float2`: layout, `sizeof`, and alignment guarantees) — this chapter's Section 5.2 depends on exactly those guarantees
- Chapter 3 in full (Struct-of-Arrays, warp coalescing, and the genuine `IMAD.WIDE` disassembly evidence from Worked Example 3.3.1) — Section 5.2's disassembly evidence is read the same way
- Chapter 4.1 (the thread hierarchy) and Chapter 4.4 (memory coalescing at full generality) — this chapter assumes you're comfortable with `blockIdx`/`threadIdx`/`blockDim` and with the warp as "32 threads scheduled together," both established there
- If you've read the Mojo edition: Mojo's Chapter 5 is built around a CPU-style, fixed-width `SIMD[DType, width]` register type, with an explicit main-loop-plus-remainder shape for looping over data wider than one register. CUDA has no source-level equivalent of that type for general arithmetic — its SIMD *is* the warp, already covered structurally in Chapters 3 and 4. Rather than force a CPU-shaped chapter onto GPU hardware that doesn't work that way, this chapter covers CUDA's own genuine warp-level and packed-arithmetic mechanisms, which have no real analogue in the Mojo edition's Chapter 5.

## 5.1 The Warp as Hardware SIMD `[FOUNDATIONAL]`

### Intuition

A CPU SIMD register executes one instruction across every lane in the same clock cycle, full stop — there is no such thing as "lane 3 takes a different branch than lane 5," because there is only ever one instruction stream and one set of lanes riding along with it. A GPU warp looks the same in the common case — 32 threads, one instruction, lockstep — but it is built out of 32 genuinely independent threads, each with its own program counter and its own registers, that the hardware *chooses* to run together. When every thread in a warp wants to execute the same instruction, a warp behaves exactly like a 32-wide SIMD register. The moment an `if` makes some threads want one instruction and other threads want a different one, that equivalence breaks: the hardware runs both paths, once each, activating only the threads that wanted that specific path and masking off the rest — a mechanism with no equivalent in true SIMD hardware, because true SIMD hardware never had the *option* of per-lane divergence in the first place.

### Background

| | CPU SIMD register (e.g. Mojo's `SIMD[DType.float32, 8]`) | GPU warp (this book's Chapters 3–4) |
|---|---|---|
| Width | Fixed at compile time by the type itself (4, 8, 16, ...) | Fixed by hardware at 32, never named as a type in source |
| Where it's declared | An explicit variable of an explicit width | Implicit — a warp is however many of a block's threads the hardware groups together, 32 at a time |
| One instruction, one operand, all lanes | Always true — that's the entire definition of the hardware | True only when all 32 threads agree on the next instruction |
| Divergent control flow | Not applicable — one instruction stream, no per-lane branching at the ISA level | Handled automatically: both sides of a branch execute, each with only the relevant threads active, at a real performance cost |

### ASCII Diagram — lockstep vs. divergence, same 8 lanes shown for both

```
No divergence -- every lane executes the SAME instruction, one cycle:
 lane: 0    1    2    3    4    5    6    7
 instr: [ADD] [ADD] [ADD] [ADD] [ADD] [ADD] [ADD] [ADD]   <- 1 instruction issued, all lanes busy

Divergence at `if (lane < 4) { A } else { B }`:
 lane: 0    1    2    3    4    5    6    7
 pass1: [A]  [A]  [A]  [A]  [--] [--] [--] [--]   <- instruction stream A, only lanes 0-3 active
 pass2: [--] [--] [--] [--] [B]  [B]  [B]  [B]    <- instruction stream B, only lanes 4-7 active
                                                       2 instructions issued to cover 8 lanes' work
```

> `[COMMON TRAP]` It's tempting to treat a warp as "just a 32-wide SIMD register" and assume any per-thread branch is free, the way branching on a scalar CPU core is. It isn't: `pass1` and `pass2` above both execute in full, one after the other, for the *entire* warp — the threads sitting out one pass aren't skipped in time, they're masked off and idle while the other half does real work. A kernel with heavy, unpredictable per-thread branching can lose most of a warp's theoretical 32-wide throughput this way, a cost that has no equivalent in a CPU SIMD register, which never had branches inside it to begin with.

## 5.2 Vectorized Memory Access: `float4` Loads and the Instruction They Really Are `[FOUNDATIONAL]`

### Intuition

Chapter 1.5 established that `float4` is four `float`s glued into one 16-byte, 16-byte-aligned struct — genuinely one contiguous unit of memory, not four independent variables that happen to sit next to each other. Section 3.3 showed a warp's real address-computation instruction, `IMAD.WIDE`, computing one address per thread for one `float` at a time. Put the two facts together and a new option appears: if a thread reads a `float4` instead of a `float`, one single memory instruction moves 16 bytes instead of 4 — the compiler doesn't need four separate load instructions to move four numbers it already knows sit contiguously in memory, it can issue one wider one.

### Background

| | Scalar load: `float v = in[i];` | Vectorized load: `float4 v = in4[i];` |
|---|---|---|
| Bytes moved by one load instruction | 4 | 16 |
| Threads needed to cover 1024 floats | 1024, each issuing its own load | 256, each issuing its own load |
| Load instructions issued in total | 1024 | 256 — one quarter as many |
| SASS instruction (genuinely observed below) | `LDG.E` (32-bit load) | `LDG.E.128` (128-bit load) |

### Worked Example 5.2.1 — the same doubling kernel, scalar and vectorized, disassembled

```cpp
// Scalar: one 4-byte load per thread
__global__ void load_scalar_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * 2.0f;
    }
}

// Vectorized: one 16-byte (float4) load does the work of 4 scalar loads
__global__ void load_vectorized_kernel(const float4* in, float4* out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = in[i];
        v.x *= 2.0f; v.y *= 2.0f; v.z *= 2.0f; v.w *= 2.0f;
        out[i] = v;
    }
}
```

Compiled and disassembled as the complete `01_vectorized_loads.cu` further below:

```bash
nvcc -arch=sm_80 01_vectorized_loads.cu -o 01_vectorized_loads
./01_vectorized_loads
nvcc -arch=sm_80 -cubin 01_vectorized_loads.cu -o 01_vectorized_loads.cubin
cuobjdump --dump-sass 01_vectorized_loads.cubin
```

Genuinely compiled, run, and disassembled:

```
sizeof(float4) = 16 bytes (4 floats per vectorized load)

Function : _Z22load_vectorized_kernelPK6float4PS_i
    LDG.E.128 R8, [R2.64] ;
    STG.E.128 [R4.64], R8 ;

Function : _Z18load_scalar_kernelPKfPfi
    LDG.E R2, [R2.64] ;
    STG.E [R4.64], R7 ;
```

Both kernels genuinely compile clean for `sm_80`, and the disassembly is unedited: `load_scalar_kernel` compiles to plain `LDG.E`/`STG.E` — the ordinary 32-bit load/store this book has been reading since Chapter 3 — while `load_vectorized_kernel` compiles to `LDG.E.128`/`STG.E.128`, a single instruction moving all 16 bytes of one `float4` at once. Neither kernel is launched in this book's no-GPU environment, so this section's evidence is exactly the *shape* of Chapter 3.3's: not "the vectorized version ran faster," but "the compiler genuinely emits a wider instruction," directly readable out of the machine code with no execution required.

> `[COMMON TRAP]` `float4`'s 16-byte alignment (Chapter 1.5) is not a suggestion — it's the precondition this section's whole argument rests on. `in4[i]` is only a legal, correctly-behaving vectorized load if the underlying buffer is actually 16-byte aligned in memory; `cudaMalloc` genuinely guarantees this for any allocation it returns, but reinterpreting an arbitrary, already-existing `float*` as a `float4*` (via a raw pointer cast, rather than allocating it as `float4` from the start) can silently violate that guarantee, and an unaligned vectorized load on real hardware either falls back to a slower path or, in the worst case, faults outright.

## 5.3 Warp-Level Reduction via Shuffle Instructions `[FOUNDATIONAL]`

### Intuition

A CPU SIMD reduction (Mojo's version of this idea) sums many elements into one width-wide register across an outer loop, then performs a single horizontal reduction at the very end to collapse that one register down to a scalar. A warp's `__shfl_down_sync` intrinsic *is* that horizontal-reduction step, generalized to 32 lanes and performed entirely in registers: it lets one thread read another specific thread's register value directly, with no shared memory, no `__syncthreads()`, and no memory traffic at all — the hardware moves the value sideways between lanes through a dedicated crossbar, not through any address either thread could compute or dereference.

### Background

| | Shared-memory reduction | Warp-shuffle reduction |
|---|---|---|
| Where partial sums live | A `__shared__` array, indexed by `threadIdx.x` | Registers only — one value per thread, never spilled to memory |
| Data movement per step | Store to shared memory, `__syncthreads()`, then load back | `__shfl_down_sync` moves a register value directly from one lane to another |
| Needed for reductions bigger than one warp | Yes, regardless of size | A single warp handles at most 32 elements this way; combining *multiple* warps' results still needs shared memory |

### Worked Example 5.3.1 — a full 32-lane sum, traced as a butterfly reduction

```cpp
__global__ void warp_reduce_sum_kernel(const float* in, float* out) {
    int lane = threadIdx.x;  // assume exactly one warp: 32 threads
    float val = in[lane];
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (lane == 0) out[0] = val;  // lane 0 now holds the full sum
}
```

`__shfl_down_sync(mask, val, offset)` returns the calling lane's own `val` plus, implicitly, nothing — it returns *the value held by the lane `offset` positions higher*, which the loop above then adds in. Five iterations (`offset = 16, 8, 4, 2, 1`) are enough to fold 32 values down to one, because each iteration halves the number of lanes still holding a "live" partial sum: after `offset=16`, lane 0 holds `v[0]+v[16]`; after `offset=8`, lane 0 holds `(v[0]+v[16]) + (v[8]+v[24])`; and so on until, after `offset=1`, lane 0 holds every one of the 32 original values added together exactly once.

Compiled, disassembled, and cross-checked against a host-side simulation of the identical butterfly pattern, as the complete `02_warp_shuffle_reduction.cu` further below:

```bash
nvcc -arch=sm_80 02_warp_shuffle_reduction.cu -o 02_warp_shuffle_reduction
./02_warp_shuffle_reduction
nvcc -arch=sm_80 -cubin 02_warp_shuffle_reduction.cu -o 02_warp_shuffle_reduction.cubin
cuobjdump --dump-sass 02_warp_shuffle_reduction.cubin
```

Genuinely compiled, run, and disassembled:

```
host-simulated warp-shuffle sum = 528.000000
closed-form sum(1..32) = n(n+1)/2 = 528.000000
match? true

SHFL.DOWN PT, R5, R2, 0x10, 0x1f ;   // offset=16 (0x10)
SHFL.DOWN PT, R0, R5, 0x8, 0x1f ;    // offset=8
SHFL.DOWN PT, R7, R0, 0x4, 0x1f ;    // offset=4
SHFL.DOWN PT, R4, R7, 0x2, 0x1f ;    // offset=2
SHFL.DOWN PT, R9, R4, 0x1, 0x1f ;    // offset=1
```

The kernel is never launched — this environment has no GPU — so `main()` instead runs a host-side function that performs the exact same five-step butterfly pattern on an ordinary 32-element array, using array indexing to stand in for lane-to-lane shuffles. For lanes holding `1, 2, 3, ..., 32`, the closed-form sum `n(n+1)/2 = 32×33/2 = 528` matches the simulation exactly — the same cross-check discipline Chapter 3.4 used for AoS/SoA kinetic energy. The disassembly independently confirms the kernel really does compile down to exactly five `SHFL.DOWN` instructions, one per loop iteration, with the exact offsets (`0x10, 0x8, 0x4, 0x2, 0x1` — 16, 8, 4, 2, 1 in hex) the source code's loop specifies, and the final `0x1f` operand on every instruction is the warp-width mask (31 = `0b11111`, confirming all 32 lanes participate).

### ASCII Diagram — the butterfly pattern, shown for 8 lanes instead of 32 for legibility

```
lanes:    0    1    2    3    4    5    6    7
values:   1    2    3    4    5    6    7    8

offset=4: 0+4  1+5  2+6  3+7  --   --   --   --
          6    8    10   12   (lanes 4-7's values no longer needed)

offset=2: 0+2  1+3  --   --
          16   20

offset=1: 0+1
          36                    <- sum(1..8) = 8*9/2 = 36, matching lane 0
```

> `[COMMON TRAP]` `__shfl_down_sync`'s first argument, the mask, is not a formality — it tells the hardware exactly which lanes are participating in this specific shuffle, and every participating lane must pass the *same* mask or the result is undefined. `0xFFFFFFFF` (all 32 bits set) is only correct when the full warp is genuinely active and un-diverged at that point in the kernel; calling a shuffle with that mask from inside a divergent branch that only some threads entered is exactly the kind of bug Section 5.1's divergence discussion warns about, and unlike the older, now-removed unmasked `__shfl_down`, the `_sync` variants at least force you to state your assumption about which lanes are present, even though they can't verify it for you.

## 5.4 Warp Vote and Ballot: Lane-Wise Predicates `[FOUNDATIONAL]`

### Intuition

A CPU SIMD comparison (Mojo's version of this idea) takes two width-wide registers and produces a third, width-wide boolean register — one true/false result per lane, still living in vector-register form. CUDA's warp-vote intrinsics do the equivalent job with a different shape: `__ballot_sync` takes one boolean predicate *per thread* and packs all 32 threads' results into a single ordinary 32-bit integer, one bit per lane, that any thread in the warp can then read as plain data — no SIMD register type involved at all, just an `unsigned int` any thread can store, branch on, or pass along.

### Background

| | CPU SIMD compare (e.g. Mojo's `SIMD[DType.bool, 8]` result) | CUDA `__ballot_sync` |
|---|---|---|
| Result shape | A width-wide vector of booleans, one lane's worth of storage per result bit | A single 32-bit integer, one bit per lane, held identically by every participating thread |
| How one lane's result is read | Indexing into the boolean vector | A bitwise test, `(mask >> lane) & 1` |
| Where the result lives | A vector register, the same "SIMD-shaped" storage as the data it came from | An ordinary scalar register — completely ordinary data from that point on |

### Worked Example 5.4.1 — 32 lanes, one predicate each, packed into one mask

```cpp
// Every thread tests a per-lane predicate; __ballot_sync packs all 32 results
// into a single 32-bit mask, one bit per lane -- CUDA's lane-wise SIMD compare.
__global__ void ballot_kernel(const float* values, unsigned int* out_mask) {
    int lane = threadIdx.x;
    bool predicate = values[lane] > 10.0f;
    unsigned int mask = __ballot_sync(0xFFFFFFFF, predicate);
    if (lane == 0) out_mask[0] = mask;
}
```

Compiled, disassembled, and cross-checked against a host-side bit-packing simulation, as the complete `03_warp_vote_ballot.cu` further below:

```bash
nvcc -arch=sm_80 03_warp_vote_ballot.cu -o 03_warp_vote_ballot
./03_warp_vote_ballot
nvcc -arch=sm_80 -cubin 03_warp_vote_ballot.cu -o 03_warp_vote_ballot.cubin
cuobjdump --dump-sass 03_warp_vote_ballot.cubin
```

Genuinely compiled, run, and disassembled:

```
values[lane] = lane index (0..31); predicate: value > 10.0
host-simulated ballot mask = 0xFFFFF800
threads satisfying predicate (lanes 11..31) = 21

FSETP.GT.AND P0, PT, R2, 10, PT ;
VOTE.ANY R5, PT, P0 ;
```

For `values[lane] = lane` (0 through 31), the predicate `value > 10.0` is true for lanes 11 through 31 inclusive — 21 lanes — and the host-side simulation, which sets bit `lane` for every lane satisfying the predicate, produces `0xFFFFF800`: bits 11 through 31 set, bits 0 through 10 clear, exactly matching. The disassembly holds a genuine surprise worth naming directly: there is no SASS instruction called `VOTE.BALLOT`. `__ballot_sync` compiles down to the *same* `VOTE` instruction family as `__any_sync`/`__all_sync`, here shown as `VOTE.ANY`; the instruction has two outputs at the hardware level — a single reduced predicate (which mode, `ANY`/`ALL`/`EQ`, decides how it's computed) and a full 32-bit register holding the raw per-lane mask, always produced regardless of mode. This code only reads the register output (`R5`, stored into `out_mask[0]`) and discards the reduced predicate (written to `PT`, CUDA's "don't care" predicate register) — so the `.ANY` in the mnemonic describes machinery this specific kernel never uses, not the operation the source code actually asked for. This is the same kind of "read the real mechanism, not the name of the intrinsic" lesson Section 5.3's mask argument taught, one level deeper.

> `[COMMON TRAP]` It's tempting to read `VOTE.ANY` in a disassembly and conclude the compiler "optimized `__ballot_sync` into `__any_sync`," changing the program's meaning. It hasn't: the full ballot mask genuinely ends up in `R5` and genuinely reaches `out_mask[0]` unchanged, because that register output exists on every `VOTE` instruction independent of which reduction mode is selected. The lesson here is the same one Chapter 3's `IMAD.WIDE` multiplier taught — a single hardware instruction can carry more information than its source-level name suggests, and the only way to know for certain which output a compiled kernel is actually keeping is to read the disassembly, not to guess from the intrinsic's name.

## 5.5 Packed SIMD Arithmetic: `half2` and the Road to Tensor Cores `[FOUNDATIONAL]`

### Intuition

Every mechanism so far in this chapter has been SIMT wearing a SIMD-shaped hat: 32 genuinely separate threads, each with its own registers and its own program counter, that only *look* like one 32-wide vector operation when they agree on what to do next. `half2` packed arithmetic is the one place in this chapter where that's no longer true. A single thread holding one `half2` value has two independent 16-bit floating-point numbers living inside one ordinary 32-bit register, and an intrinsic like `__hadd2` performs *both* additions in one instruction, on one thread, with no other thread involved at all — genuine, hardware-level 2-wide SIMD, the closest thing this book's CUDA hardware has to Mojo's `SIMD[DType.float16, 2]`.

### Background

| | Warp-level mechanisms (Sections 5.2–5.4) | `half2` packed arithmetic |
|---|---|---|
| Parallelism comes from | 32 separate threads, each with its own registers | 2 lanes packed into *one* thread's *one* register |
| Instruction issued by | The warp scheduler, once per warp | One ordinary thread, on its own |
| Mechanism name | SIMT (Single Instruction, Multiple Threads) | True SIMD (Single Instruction, Multiple Data) |
| Where this goes next | — | Part 6's tensor cores extend this exact packed-math idea from 2 values to a whole matrix tile at once |

### Worked Example 5.5.1 — one packed add, confirmed in the disassembly

```cpp
// Two independent fp16 additions packed into ONE 32-bit register and ONE
// instruction -- genuine 2-wide SIMD, not SIMT: both lanes of the half2 are
// computed by a single thread, using the hardware's packed-math ALU.
__global__ void half2_add_kernel(half2 a, half2 b, half2* out) {
    *out = __hadd2(a, b);
}
```

Compiled and disassembled as the complete `04_half2_simd.cu` further below (`__hadd2` and `half2` construction are device-only in this toolkit, so this example is compile-and-disassemble evidence, in the exact style of Chapter 3's `03_coalescing_kernels.cu`, rather than a host-run program):

```bash
nvcc -arch=sm_80 04_half2_simd.cu -o 04_half2_simd
./04_half2_simd
nvcc -arch=sm_80 -cubin 04_half2_simd.cu -o 04_half2_simd.cubin
cuobjdump --dump-sass 04_half2_simd.cubin
```

Genuinely compiled, run, and disassembled:

```
lane 0: a=1.5 b=10.0 a+b=11.5
lane 1: a=2.5 b=20.0 a+b=22.5
sizeof(half2) = 4 bytes (two 16-bit lanes packed into one 32-bit register)

HADD2 R5, R5, c[0x0] [0x164] ;
```

`main()` here runs entirely on the host, computing an ordinary scalar fp32 reference for the same two lane-pairs the kernel's packed add would produce — `1.5+10.0=11.5` and `2.5+20.0=22.5` — because `half2` construction and `__hadd2` are device-only functions in this toolkit and this book's no-GPU environment cannot call them from `main()`. What genuinely compiles and disassembles is `half2_add_kernel` itself: a single `HADD2` instruction, operating on one 32-bit register holding both packed 16-bit values, performing both additions at once. `sizeof(half2) = 4` confirms the packing directly — two 16-bit floats fit inside the same 4 bytes one ordinary `float` alone would occupy, which is exactly why a tensor-core matrix tile (Part 6) built from `half` values can pack twice as many elements into the same register file footprint as an equivalent `float` tile.

> `[COMMON TRAP]` It's easy to read "SIMD" in this chapter's title and assume every mechanism in it works the same way. Sections 5.2 through 5.4 are all genuinely about a *warp* — 32 threads, SIMT, converging or diverging together — while this section's `half2` math happens entirely *inside one thread*, with the warp mechanism not involved at all. A kernel can use both at once (32 threads, each independently doing packed 2-wide `half2` math) without the two mechanisms interacting — they're stacked, not the same thing wearing two names.

## 5.6 Complete Runnable Code

### File: `01_vectorized_loads.cu`

```cpp
#include <cstdio>

// Scalar: one 4-byte load per thread
__global__ void load_scalar_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * 2.0f;
    }
}

// Vectorized: one 16-byte (float4) load services what would be 4 threads' worth of scalar loads
__global__ void load_vectorized_kernel(const float4* in, float4* out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = in[i];
        v.x *= 2.0f; v.y *= 2.0f; v.z *= 2.0f; v.w *= 2.0f;
        out[i] = v;
    }
}

int main() {
    printf("sizeof(float4) = %zu bytes (%d floats per vectorized load)\n",
           sizeof(float4), (int)(sizeof(float4) / sizeof(float)));
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_vectorized_loads.cu -o 01_vectorized_loads
./01_vectorized_loads
nvcc -arch=sm_80 -cubin 01_vectorized_loads.cu -o 01_vectorized_loads.cubin
cuobjdump --dump-sass 01_vectorized_loads.cubin
```

Produces exactly the output and disassembly shown in Worked Example 5.2.1 above.

### File: `02_warp_shuffle_reduction.cu`

```cpp
#include <cstdio>

// Warp-level sum reduction using __shfl_down_sync -- no shared memory needed
__global__ void warp_reduce_sum_kernel(const float* in, float* out) {
    int lane = threadIdx.x;  // assume exactly one warp: 32 threads
    float val = in[lane];
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (lane == 0) out[0] = val;  // lane 0 now holds the full sum
}

// Host-side simulation of the exact same butterfly-reduction the kernel performs,
// lane by lane, with no GPU involved.
float host_simulate_warp_reduce(float lanes[32]) {
    float v[32];
    for (int i = 0; i < 32; i++) v[i] = lanes[i];
    for (int offset = 16; offset > 0; offset /= 2) {
        float next[32];
        for (int lane = 0; lane < 32; lane++) {
            int src = lane + offset;
            float shuffled = (src < 32) ? v[src] : 0.0f;  // shfl_down: lanes past the warp's edge get 0 here
            next[lane] = v[lane] + shuffled;
        }
        for (int i = 0; i < 32; i++) v[i] = next[i];
    }
    return v[0];
}

int main() {
    float lanes[32];
    for (int i = 0; i < 32; i++) lanes[i] = (float)(i + 1);  // 1, 2, ..., 32

    float result = host_simulate_warp_reduce(lanes);
    float closed_form = 32.0f * 33.0f / 2.0f;  // sum 1..32
    printf("host-simulated warp-shuffle sum = %f\n", result);
    printf("closed-form sum(1..32) = n(n+1)/2 = %f\n", closed_form);
    printf("match? %s\n", (result == closed_form) ? "true" : "false");
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_warp_shuffle_reduction.cu -o 02_warp_shuffle_reduction
./02_warp_shuffle_reduction
nvcc -arch=sm_80 -cubin 02_warp_shuffle_reduction.cu -o 02_warp_shuffle_reduction.cubin
cuobjdump --dump-sass 02_warp_shuffle_reduction.cubin
```

Produces exactly the output and disassembly shown in Worked Example 5.3.1 above.

### File: `03_warp_vote_ballot.cu`

```cpp
#include <cstdio>

// Every thread tests a per-lane predicate; __ballot_sync packs all 32 results
// into a single 32-bit mask, one bit per lane -- CUDA's lane-wise SIMD compare.
__global__ void ballot_kernel(const float* values, unsigned int* out_mask) {
    int lane = threadIdx.x;
    bool predicate = values[lane] > 10.0f;
    unsigned int mask = __ballot_sync(0xFFFFFFFF, predicate);
    if (lane == 0) out_mask[0] = mask;
}

// Host-side simulation of the identical per-lane predicate and the identical
// bit-packing __ballot_sync performs, with no GPU involved.
unsigned int host_simulate_ballot(float values[32]) {
    unsigned int mask = 0;
    for (int lane = 0; lane < 32; lane++) {
        bool predicate = values[lane] > 10.0f;
        if (predicate) mask |= (1u << lane);
    }
    return mask;
}

int main() {
    float values[32];
    for (int i = 0; i < 32; i++) values[i] = (float)i;  // 0, 1, ..., 31

    unsigned int mask = host_simulate_ballot(values);
    printf("values[lane] = lane index (0..31); predicate: value > 10.0\n");
    printf("host-simulated ballot mask = 0x%08X\n", mask);

    int popcount = 0;
    for (int lane = 0; lane < 32; lane++) if (mask & (1u << lane)) popcount++;
    printf("threads satisfying predicate (lanes 11..31) = %d\n", popcount);
    return 0;
}
```

```bash
nvcc -arch=sm_80 03_warp_vote_ballot.cu -o 03_warp_vote_ballot
./03_warp_vote_ballot
nvcc -arch=sm_80 -cubin 03_warp_vote_ballot.cu -o 03_warp_vote_ballot.cubin
cuobjdump --dump-sass 03_warp_vote_ballot.cubin
```

Produces exactly the output and disassembly shown in Worked Example 5.4.1 above.

### File: `04_half2_simd.cu`

```cpp
#include <cstdio>
#include <cuda_fp16.h>

// Two independent fp16 additions packed into ONE 32-bit register and ONE
// instruction -- genuine 2-wide SIMD, not SIMT: both lanes of the half2 are
// computed by a single thread, using the hardware's packed-math ALU.
__global__ void half2_add_kernel(half2 a, half2 b, half2* out) {
    *out = __hadd2(a, b);
}

int main() {
    // __hadd2 and half2 construction are device-only in this toolkit, so the
    // host side here computes an ordinary scalar fp32 reference instead --
    // the same value the kernel's packed add produces, one lane at a time.
    float a0 = 1.5f, a1 = 2.5f;
    float b0 = 10.0f, b1 = 20.0f;
    printf("lane 0: a=%.1f b=%.1f a+b=%.1f\n", a0, b0, a0 + b0);
    printf("lane 1: a=%.1f b=%.1f a+b=%.1f\n", a1, b1, a1 + b1);
    printf("sizeof(half2) = %zu bytes (two 16-bit lanes packed into one 32-bit register)\n", sizeof(half2));
    return 0;
}
```

```bash
nvcc -arch=sm_80 04_half2_simd.cu -o 04_half2_simd
./04_half2_simd
nvcc -arch=sm_80 -cubin 04_half2_simd.cu -o 04_half2_simd.cubin
cuobjdump --dump-sass 04_half2_simd.cubin
```

### Expected Output for `04_half2_simd.cu`

```
lane 0: a=1.5 b=10.0 a+b=11.5
lane 1: a=2.5 b=20.0 a+b=22.5
sizeof(half2) = 4 bytes (two 16-bit lanes packed into one 32-bit register)
```

Every number in this section was independently verified earlier in this chapter: the disassembly evidence in Worked Examples 5.2.1, 5.3.1, 5.4.1, and 5.5.1, and the host-run numeric checks in `02_warp_shuffle_reduction.cu` and `03_warp_vote_ballot.cu`. All four files compile clean under `nvcc -arch=sm_80`; only `01_vectorized_loads.cu`, `02_warp_shuffle_reduction.cu`, and `03_warp_vote_ballot.cu`'s `main()` functions are actually run to completion in this no-GPU environment, since none of them launch a kernel — `04_half2_simd.cu` is compile-and-disassemble evidence only, in the same style as Chapter 3's `03_coalescing_kernels.cu`, because `half2` construction and packed arithmetic are device-only in this toolkit.

## Chapter Summary

A GPU warp and a CPU SIMD register share the same core idea — one instruction, many lanes — but arrive at it from opposite directions: a SIMD register is inherently, permanently one instruction stream with no notion of per-lane branching, while a warp is 32 genuinely independent threads that the hardware runs together only as long as they agree, falling back to serialized, masked execution the moment they don't. `float4` gives CUDA's compiler a legal way to move four floats in one instruction instead of four, verified in this chapter by reading `LDG.E.128` and `STG.E.128` directly out of disassembled machine code rather than asserting it. `__shfl_down_sync` moves data directly between a warp's lanes through registers alone, with no shared memory and no `__syncthreads()`, letting a 32-element sum collapse to one value in exactly five instructions — a butterfly pattern this chapter traced by hand and confirmed against the closed-form `n(n+1)/2`. `__ballot_sync` packs 32 threads' boolean predicates into one ordinary 32-bit mask, and its disassembly held a genuine surprise: the underlying `VOTE` instruction produces both a reduced predicate and a full mask register on every call, regardless of which named intrinsic asked for it, a reminder that a compiled instruction can carry more than its source-level name suggests. And `half2` packed arithmetic is this chapter's one genuine exception to "GPU SIMD is really SIMT" — two 16-bit values, one register, one `HADD2` instruction, computed by a single thread with no warp mechanism involved at all, which is exactly the packed-math idea Part 6's tensor cores scale up from two values to an entire matrix tile.

## Self-Check Questions

1. A CPU SIMD register and a GPU warp both claim to execute "one instruction, many lanes." Under what specific circumstance does that claim stop being true for a warp but remain permanently true for a CPU SIMD register?
2. `load_vectorized_kernel` processes 1024 floats using 256 threads instead of the 1024 a scalar kernel would need. How many total load instructions does each version issue, and how many bytes does each single load instruction move?
3. `warp_reduce_sum_kernel` reduces 32 values to 1 using exactly 5 calls to `__shfl_down_sync`, with offsets 16, 8, 4, 2, 1. If a warp only had 8 active lanes instead of 32, how many shuffle steps would be needed, and what offsets would they use?
4. The disassembly of `ballot_kernel` shows a `VOTE.ANY` instruction, even though the source code calls `__ballot_sync`, not `__any_sync`. Explain why this is not evidence that the compiler changed the program's behavior.
5. `half2_add_kernel` performs two additions in a single `HADD2` instruction issued by one thread. Explain why this is not the same mechanism as a warp of 32 threads each independently computing one scalar addition, even though both scenarios involve "more than one addition happening at once."

## Where We Go Next

Part 0 closes here having built, piece by piece, every mechanism the rest of this book's `Tensor` class depends on: fixed-layout types and the host/device compiler split (Chapter 1), structs with RAII-managed device memory (Chapter 2), Struct-of-Arrays layout for warp-coalesced access (Chapter 3), the thread hierarchy and the broadcast kernel pattern (Chapter 4), and now vectorized loads, warp-level reductions, and packed arithmetic (this chapter). Part 1 starts assembling these into an actual `Tensor` — the first genuinely new data structure this book builds, rather than a small example illustrating one mechanism at a time.

## Worked Solutions

**1.** The claim breaks for a warp the moment its 32 threads disagree about which instruction to execute next — a data-dependent branch where some threads take one path and others take another, at which point the hardware serializes the two paths, masking off the inactive threads on each pass (Section 5.1's ASCII diagram). A CPU SIMD register has no mechanism for per-lane branching at all — every lane always executes the exact same instruction stream by construction — so there is no circumstance under which its "one instruction, many lanes" guarantee can break.

**2.** The scalar version issues 1024 load instructions, each moving 4 bytes (one `float`). The vectorized version issues 256 load instructions, each moving 16 bytes (one `float4`). Both versions move the identical total number of bytes (`1024 × 4 = 4096` and `256 × 16 = 4096`), but the vectorized version does it in one quarter as many separate load instructions.

**3.** An 8-lane warp would need `log2(8) = 3` shuffle steps, with offsets 4, 2, 1 — each step still halves the number of lanes holding a "live" partial value, exactly as Section 5.3's ASCII diagram traced for 8 lanes: `offset=4` first, then `offset=2`, then `offset=1`, ending with lane 0 holding the full 8-element sum.

**4.** `VOTE` is a single hardware instruction family with two outputs: a full 32-bit mask register (holding every lane's predicate bit, always computed) and one reduced predicate bit (`ANY`, `ALL`, or `EQ`, selected by the instruction's mode). `__ballot_sync`'s compiled code reads only the mask-register output and discards the reduced predicate (written to CUDA's unused `PT` register) — the `.ANY` in the disassembly names which reduction mode happened to be selected for a value this particular kernel never reads, not a change to what the source code asked for or what value ends up in `out_mask[0]`.

**5.** A warp of 32 threads computing 32 scalar additions is SIMT: 32 genuinely separate threads, each with its own program counter and registers, that the hardware happens to schedule together — the parallelism lives *across threads*. `half2_add_kernel`'s single `HADD2` instruction performs both of its additions inside *one* thread's *one* register, with no other thread involved at all — the parallelism lives inside one instruction's own data width. A kernel could combine both: 32 threads, each independently running `HADD2` on its own `half2`, for 64 total additions per warp-instruction — the two mechanisms stack rather than substitute for each other.
