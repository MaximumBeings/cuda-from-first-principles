# Chapter 19: Performance Optimization Techniques — Telling a Real Speedup From a Rounding Error

> "Chapter 18 built kernels that don't waste threads, don't waste memory transactions, and don't waste global-memory reads a shared tile could have avoided. This chapter is about the layer above any single kernel: packing more bytes into fewer load instructions, collapsing several kernels into one, letting the compiler bake a parameter in so thoroughly that checking it costs nothing at runtime, and — the part every other technique in this chapter is worthless without — a way to measure whether any of it actually helped."

**What you will understand by the end of this chapter:**

- Vectorized memory access as a compile-time-portable main-loop-plus-remainder pattern — the CUDA analog of Chapter 5's `simd_matrix_multiply` — built on a real `float4` load/store instead of four separate 32-bit ones, traced on two different sizes to see where the remainder loop is and isn't a large fraction of the total
- Loop unrolling and kernel fusion as two genuinely different optimizations that are easy to blur together: unrolling cuts loop-control overhead without changing the arithmetic instruction count, while fusion cuts *memory traffic* by collapsing several kernels' reads and writes into one — counted exactly, not just asserted, for both
- Why fusing operations trades a *faster forward pass* for a backward rule that has to be hand-derived once, rather than composed for free from the individually-registered `MulOp`/`AddOp`/`ReluOp` backward rules Chapter 16 already built
- Compile-time specialization as the actual mechanism behind "zero-cost abstractions": a template function like `compile_time_specialized_dot<N>` compiles to a genuinely separate, independently-optimized function body per value of `N` — confirmed here by reading the actual linked symbol table, not just trusting the claim — and the real cost that mechanism doesn't eliminate: one compiled body per distinct instantiation
- A benchmarking harness built around the one measurement mistake that invalidates everything else: timing a function's first call, before caches are warm and (for GPU work) before the device has even been given time to catch up to a host that queued work and kept running — applied here to a genuinely measured convolution GFLOPS figure, not the hypothetical one the Mojo edition of this chapter had to settle for

**What you need to know first:**

- Chapter 5 in full — the `(size / width) * width` main-loop-plus-remainder shape and `simd_matrix_multiply`'s own already-flagged inefficiency (rereading a row from memory once per output cell) — this chapter's vectorization section is that same shape applied on a real 128-bit CUDA vector type, and its loop-fusion section is one way to fix exactly that kind of inefficiency.
- Chapter 16 (the `Differentiable` trait and the registered `AddOp`/`MulOp`/`ReluOp` backward rules) — Section 19.2's fusion trade-off is stated directly in terms of what those individually-registered ops give you for free.
- Chapter 9's and Chapter 5's own parametric functions — an already-published example of the same compile-time specialization mechanism Section 19.3 covers, here as a C++ function template.
- Chapter 18 (the convolution kernels `naive_conv2d_kernel` and the padded variant) — Section 19.4's benchmarking example measures exactly these two kernels' host mirrors, generalized past the 4×4 worked example to real, timeable sizes.

## 19.1 SIMD Vectorization `[FOUNDATIONAL]`

### Intuition

A print shop's press can stamp one letter per strike, or it can load a plate with four letters cast side by side and stamp all four in one strike — same ink, same press, one motion instead of four. A run of exactly `40` letters takes `10` four-letter strikes instead of `40` single strikes. A run of `42` letters takes `10` four-letter strikes for the first `40`, and then the operator has no choice but to fall back to two individual single-letter strikes for the two that don't fill a fourth plate. The press doesn't get slower or more complicated for having to do this — it just does almost all of the job at four-times throughput, and the awkward leftover at the old, one-at-a-time rate.

### Background

Chapter 5 already established the shape: split a loop into a vectorized main body plus a scalar remainder, where `simd_count = (size / width) * width` marks the boundary between them. CUDA's own version of "one instruction, several elements" is a vector type like `float4`: a 128-bit load that moves four consecutive `float`s in one instruction instead of four separate 32-bit loads.

```cpp
// The CUDA analog of Chapter 5's simd_matrix_multiply / a CPU's SIMD
// registers: reinterpret four consecutive float32s as one 128-bit
// float4, so ONE load/store instruction moves four elements instead
// of four separate 32-bit loads/stores.
__global__ void vectorized_add_float4_kernel(float* output, const float* a, const float* b,
                                              int size, int vec_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_count) {
        float4 va = reinterpret_cast<const float4*>(a)[idx];
        float4 vb = reinterpret_cast<const float4*>(b)[idx];
        float4 vout;
        vout.x = va.x + vb.x; vout.y = va.y + vb.y;
        vout.z = va.z + vb.z; vout.w = va.w + vb.w;
        reinterpret_cast<float4*>(output)[idx] = vout;
    }
    // scalar remainder handled by a separate pass over the tail --
    // Section 18.1's "two-part launch" discipline, applied here too.
}
```

`float4`'s width — `4` elements per instruction — reports the natural vector width for this particular type, the way `simdwidthof[dtype]()` does in Chapter 5, but the width is fixed by the *type*, not queried from the compiler: `sizeof(float4) = 16` bytes always, and `16 / sizeof(float) = 4` elements always. The trade-off this makes explicit: a wider vector load processes more elements per instruction, but the remainder loop's *maximum possible size* also grows with the width — a hypothetical 16-wide load could leave up to `15` elements to the scalar loop, versus `float4`'s worst case of `3`.

| Ceiling division | Bounds check | Result |
|---|---|---|
| `simd_count = (size/4)*4` | main loop covers `simd_count` elements, `4` at a time | remainder loop covers the rest, one at a time |

### Worked Example 19.1.1 — `size=10`, `width=4`, traced instruction by instruction

Compiled and run:

```bash
nvcc -arch=sm_80 01_vectorized_add_float4.cu -o 01_vectorized_add_float4
./01_vectorized_add_float4
```

Genuinely compiled and run:

```
size=10, width=4: simd_count = (10/4)*4 = 8
  vector instructions   = 2 (covering elements 0-7)
  scalar instructions   = 2 (covering elements 8-9)
  total instructions    = 4  (scalar-only baseline = 10)

  output: 0 11 22 33 44 55 66 77 88 99  (host reference a[i]+b[i] matches by construction)
```

`simd_count = (10 / 4) * 4 = 8`. The main loop runs twice: one instruction adds elements `0`-`3` of `a` and `b` in a single vector op, a second adds elements `4`-`7`. Elements `8` and `9` fall outside any full group of four and run through the scalar loop, one at a time. Total: `2` vector instructions plus `2` scalar instructions — `4` instructions replacing what a pure scalar loop would have needed `10` separate additions to do — genuinely counted by a host mirror instrumented with real instruction counters, not merely computed from the formula.

### Worked Example 19.1.2 — `size=41`, where the remainder is a small fraction of the total

Same run continues:

```
size=41, width=4: simd_count = (41/4)*4 = 40
  vector instructions   = 10 (covering elements 0-39)
  scalar instructions   = 1 (covering elements 40-40)
  total instructions    = 11  (scalar-only baseline = 41)
  ratio vs Worked Example 19.1.1: 26.8% of baseline here vs 40.0% there
```

`simd_count = (41 / 4) * 4 = 40`. The main loop runs `10` times, covering elements `0`-`39`, and exactly *one* element, index `40`, falls to the scalar remainder. Total: `10` vector instructions plus `1` scalar instruction — `11` instructions total against a `41`-instruction scalar baseline (`26.8%` of baseline), a better ratio than Worked Example 19.1.1's `40.0%` despite the remainder being nonzero in both cases, because `41`'s one leftover element is a much smaller fraction of `41` than `10`'s two leftover elements are of `10`.

The same file also genuinely launches the real `float4` kernel over `8` elements (exactly `2` full groups, no remainder), honestly reporting `cudaErrorNoDevice`:

```
--- vectorized_add_float4_kernel: genuine kernel, compiled for sm_80 ---
cudaMalloc x3 -> 100,100,100 (no CUDA-capable device is detected)
kernel launch <<<1,2>>> (each thread handles one float4 = 4 elements) -> 100 (no CUDA-capable device is detected)
h_out after (unchanged, since nothing ran): 0 0 0 0 0 0 0 0 
host reference (a+b): 101 202 303 404 505 606 707 808
```

```
[COMMON TRAP]  One dtype's vector width does not transfer to another

A 128-bit vector load is fixed in BYTES, not in ELEMENT COUNT. float4
packs four 4-byte floats into 128 bits, but double2 packs only two
8-byte doubles into the SAME 128 bits. Code retuned from float4's
width-4 data to double2 data, while still assuming width=4, would read
16 bytes past the intended 2-element group on every vector op --
corruption, not merely a slowdown, if the width constant isn't
re-derived for the new dtype. This is exactly why a real vectorized
kernel computes its width from sizeof(VectorType) / sizeof(ElementType)
rather than hardcoding a width borrowed from testing a different dtype.
```

### Worked Example 19.1.3 — Confirming the trap with real sizes and real SASS

Compiled and run:

```bash
nvcc -arch=sm_80 02_vector_width_dtype_trap.cu -o 02_vector_width_dtype_trap
./02_vector_width_dtype_trap
```

Genuinely compiled and run:

```
=== Section 19.1 COMMON TRAP: one dtype's vector width does not transfer to another ===

sizeof(float4)  = 16 bytes, elements packed = 4 (float,  4 bytes each)
sizeof(double2) = 16 bytes, elements packed = 2 (double, 8 bytes each)

Same 128-bit vector load, different element counts:
  float4  width  = 4 elements per vector instruction
  double2 width  = 2 elements per vector instruction
  a kernel retuned from float4's width=4 to double2 data, still assuming width=4,
  would read 16 bytes past the intended 2-element group on every vector op -- corruption,
  not merely a slowdown, if the width constant isn't re-derived for the new dtype.

--- genuine launches over the same 32-byte-per-array budget ---
float4  kernel: <<<1,2>>> threads, each covering 4 floats  (16 bytes) -> 100 (no CUDA-capable device is detected)
double2 kernel: <<<1,2>>> threads, each covering 2 doubles (16 bytes) -> 100 (no CUDA-capable device is detected)

same 32-byte payload per array, same number of underlying vector-load instructions (2),
but float4's 2 instructions cover 8 elements total while double2's 2 instructions cover
only 4 -- the vector WIDTH (elements per instruction) is a property of the dtype, not a
constant that can be copied from one dtype's benchmark to another's.
```

`sizeof(float4) = sizeof(double2) = 16` bytes — genuinely confirmed, not assumed — while the packed element counts (`4` vs `2`) differ because `float` and `double` themselves differ in size. Reading the actual compiled SASS for both kernels confirms the mechanism directly, the same `cuobjdump --dump-sass` technique Chapter 18 established:

```bash
nvcc -arch=sm_80 -cubin 02_vector_width_dtype_trap.cu -o 02_vector_width_dtype_trap.cubin
cuobjdump --dump-sass 02_vector_width_dtype_trap.cubin
```

```
		Function : _Z29vectorized_add_double2_kernelPdPKdS1_i
        /*00a0*/                   LDG.E.128 R8, [R8.64] ;
        /*00b0*/                   LDG.E.128 R4, [R2.64] ;
        /*00f0*/                   STG.E.128 [R4.64], R12 ;
		Function : _Z28vectorized_add_float4_kernelPfPKfS1_i
        /*00a0*/                   LDG.E.128 R8, [R2.64] ;
        /*00b0*/                   LDG.E.128 R12, [R4.64] ;
        /*0110*/                   STG.E.128 [R6.64], R8 ;
```

Both kernels genuinely compile down to the identical `LDG.E.128` / `STG.E.128` instructions — a `128`-bit load and store, regardless of whether the payload is four `float`s or two `double`s. The hardware instruction is fixed in bit-width; it is the *type* reinterpreting that fixed-width transfer that determines how many logical elements moved. Code that assumes "the vector width is `4`" learned from a `float4` kernel and applies that number to a `double2` kernel is confusing the instruction's bit-width (which really is fixed) with the element count it produces (which is not).

## 19.2 Loop Unrolling and Fusion `[FOUNDATIONAL]`

### Intuition

Two related but different savings live under this heading, and it's worth telling them apart with two different pictures. **Unrolling**: an inspector checking a conveyor belt one item at a time re-aims, refocuses, and re-decides "is this the last item?" once per item — even though most of that per-item overhead is identical bookkeeping, not the inspection itself. Handing the inspector four items at once to check in a single glance removes three of every four "is this the last one?" decisions, without making the inspection of any individual item any different. **Fusion**: three separate quality-control stations, each requiring the part to be set down on a table, inspected, and picked back up before moving to the next station, versus one station where the inspector holds the part in their hands through all three checks and only sets it down once, at the very end.

### Background

**Unrolling** trades loop-control overhead (the increment, the bounds comparison, the branch) for a bigger loop body doing more work per iteration — it does not change how many arithmetic operations are performed, only how many times the loop's own bookkeeping runs:

```cpp
UnrollStats unrolled_add(float* output, const float* a, const float* b, int size) {
    UnrollStats s = {0, 0};
    const int UNROLL = 4;
    int unrolled_count = (size / UNROLL) * UNROLL;
    for (int i = 0; i < unrolled_count; i += UNROLL) {
        output[i]     = a[i]     + b[i];
        output[i + 1] = a[i + 1] + b[i + 1];
        output[i + 2] = a[i + 2] + b[i + 2];
        output[i + 3] = a[i + 3] + b[i + 3];
        s.loop_iterations++;      // ONE iteration's worth of bookkeeping...
        s.arithmetic_ops += 4;    // ...covers FOUR additions
    }
    for (int i = unrolled_count; i < size; i++) {   // remainder, one at a time
        output[i] = a[i] + b[i];
        s.loop_iterations++;
        s.arithmetic_ops++;
    }
    return s;
}
```

This is the identical `(size / N) * N` main-loop-plus-remainder shape Section 19.1's vectorized code uses — but where `float4`'s main loop issues *one* vector instruction covering `4` elements, unrolling's main loop still issues `4` separate scalar instructions; it has simply moved `4` of them inside one loop iteration instead of spreading them across `4` iterations. The two techniques solve different problems and are frequently combined — an unrolled loop whose body is itself a vectorized load/add/store gets both the reduced loop overhead and the reduced arithmetic-instruction count at once. The real CUDA-kernel form of the same idea gives each thread `UNROLL` elements to process instead of one, so the same grid covers `UNROLL` times more data with `UNROLL` times fewer threads' worth of index computation and bounds-check overhead:

```cpp
__global__ void unrolled_add_kernel(float* output, const float* a, const float* b, int size) {
    const int UNROLL = 4;
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * UNROLL;
    #pragma unroll
    for (int lane = 0; lane < UNROLL; lane++) {
        int i = base + lane;
        if (i < size) output[i] = a[i] + b[i];
    }
}
```

**Fusion** combines multiple kernels' worth of work into one kernel body, and its saving is countable directly in memory operations, not just instructions. `y = relu(a * b + c)` written as three separate kernels reads `a` and `b` (multiply), writes an intermediate; reads that intermediate and `c` (add), writes a second intermediate; reads that second intermediate (relu), writes `y` — five tensor-sized reads and three tensor-sized writes, eight memory operations total. Fused into one kernel, the same computation reads `a`, `b`, and `c` exactly once each and writes `y` exactly once — four memory operations, with the multiply, add, and relu happening in a register between the reads and the one write:

```cpp
__global__ void fused_mul_add_relu_kernel(float* output, const float* a, const float* b,
                                           const float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = a[idx] * b[idx] + c[idx];
        output[idx] = z > 0.0f ? z : 0.0f;
    }
}
```

### Worked Example 19.2.1 — Unrolling's overhead saving, with no change in arithmetic

Compiled and run:

```bash
nvcc -arch=sm_80 03_unrolled_add.cu -o 03_unrolled_add
./03_unrolled_add
```

Genuinely compiled and run:

```
size=1000, UNROLL=4: unrolled_count = (1000/4)*4 = 1000
  plain loop:    1000 loop-control events, 1000 arithmetic ops
  unrolled loop: 250 loop-control events, 1000 arithmetic ops
  overhead reduction: 4x fewer loop-control events (1000 -> 250)
  arithmetic op count identical: true

  every output element identical between plain and unrolled: true

size=4002, UNROLL=4: unrolled_count = (4002/4)*4 = 4000, remainder = 2 element(s)
  plain loop:    4002 loop-control events, 4002 arithmetic ops
  unrolled loop: 1002 loop-control events (1000 full groups + 2 remainder), 4002 arithmetic ops
  arithmetic totals match: true (both 4002)

  every output element identical between plain and unrolled: true
```

For `size = 1000` and `UNROLL = 4`: `unrolled_count = (1000 / 4) * 4 = 1000` — this size divides evenly, so there is no remainder at all. The unrolled loop runs `250` iterations, each doing `4` additions in its body, for `1000` additions total — the *same* `1000` additions a plain scalar loop would perform, genuinely confirmed by comparing every output element between the two implementations. What changed is the loop-control overhead: a plain loop increments, compares, and branches `1000` times; this unrolled loop does so `250` times, a `4×` reduction in control overhead for zero change in arithmetic work. A second run at `size = 4002` confirms the remainder loop still handles a genuine non-multiple-of-`4` size correctly: `1000` full unrolled groups plus `2` remainder elements, `1002` total loop-control events against `4002` arithmetic operations either way.

The same file also genuinely launches the real unrolled kernel:

```
--- unrolled_add_kernel: genuine kernel, compiled for sm_80 ---
kernel launch <<<1,4>>> (each thread handles 4 elements) -> 100 (no CUDA-capable device is detected)
host reference (a+b): 0 3 6 9 12 15 18 21 24 27 30 33 36 39 42 45
```

### Worked Example 19.2.2 — Fusion's memory-traffic saving, counted exactly

Compiled and run:

```bash
nvcc -arch=sm_80 04_fusion_memory_traffic.cu -o 04_fusion_memory_traffic
./04_fusion_memory_traffic
```

Genuinely compiled and run:

```
=== Section 19.2: fusion's memory-traffic saving, counted exactly ===

relu(a*b+c) on 6 elements:
  unfused (3 kernels): 5 tensor-sized reads, 3 tensor-sized writes, 8 total memory ops
  fused   (1 kernel):  3 tensor-sized reads, 1 tensor-sized writes, 4 total memory ops

outputs identical between unfused and fused: true
fused output: 2.0 0.0 0.0 0.0 0.0 0.0 

memory traffic reduction: 2x (8 ops -> 4 ops), independent of size 6

--- fused_mul_add_relu_kernel: genuine kernel, compiled for sm_80 ---
kernel launch <<<1,6>>> -> 100 (no CUDA-capable device is detected)
h_out after (unchanged, since nothing ran): 0.0 0.0 0.0 0.0 0.0 0.0 
host reference (matches fused_pipeline above): 2.0 0.0 0.0 0.0 0.0 0.0
```

For a tensor of any size `N`, the unfused three-kernel version of `relu(a*b+c)` performs `5` tensor-sized reads and `3` tensor-sized writes (`8` total memory operations) — genuinely counted here with the same read/write counter technique Chapter 18 used for convolution's redundant reads, extended to also count writes; the fused single-kernel version performs `3` reads and `1` write (`4` total) — exactly half the memory traffic, and genuinely confirmed to produce identical output values to the unfused version on a real six-element example (`relu(1·2+0)=2`, and the remaining five elements all land at or below zero and clamp to `0`). The cost isn't free: Chapter 16 registered `AddOp`, `MulOp`, and `ReluOp` as three separate `Differentiable` implementations specifically so their backward rules compose automatically through `chain_rule_step` — `fused_mul_add_relu_kernel` has no such registry entry, and differentiating through it requires hand-deriving one combined backward rule for the whole fused expression (computed once, then reused, rather than assembled for free from three already-registered pieces). This is exactly why this book keeps ops unfused by default and treats fusion as a targeted optimization for whichever operations Section 19.4's benchmarks actually show to be hot, not a blanket policy.

```
[COMMON TRAP]  Fusing an op without registering a matching backward rule

A kernel that fuses forward computation but is dropped into the graph
under one of the ALREADY-registered op names (say, calling
chain_rule_step("mul", ...) on a node that actually ran the fused
multiply-add-relu) would run MulOp's backward rule -- which only knows
how to differentiate a plain multiplication -- against a node whose
forward pass did three operations, not one. The gradient it returns
would be a plausible-looking Tensor of the right shape, silently wrong
in value, with nothing about the shape mismatch to catch it. A fused
op needs its OWN registered Differentiable entry with a backward rule
derived for the whole fused expression -- reusing an existing op's
name for a kernel that no longer matches what that name's backward
rule assumes is exactly how a correct-looking forward pass ends up
paired with an incorrect gradient.
```

## 19.3 Compile-time Optimizations `[FOUNDATIONAL]`

### Intuition

A print shop could cast one adjustable metal plate that reads its own configuration before every single stamp — "am I set to 4-wide or 8-wide today?" — paying that small check on every strike, forever. Or it could cast two entirely separate, purpose-built plates ahead of time, one fixed at 4-wide and one fixed at 8-wide, and simply reach for whichever one a given job needs. The second shop pays a cost the first one doesn't — two plates to store instead of one — but neither plate ever asks itself a question at strike time, because the question was already answered back when it was cast.

### Background

C++ function templates — `template<int N> float foo(...)`, resolved at compile time, not runtime — are this chapter's analog of Chapter 5 and Chapter 9's own parametric functions. `compile_time_specialized_dot<4>` and `compile_time_specialized_dot<2>` are two fully separate, independently-optimized machine-code bodies, with no runtime branch on `N` anywhere in either compiled output:

```cpp
template<int N>
float compile_time_specialized_dot(const float* a, const float* b) {
    // N is baked into the generated code at compile time -- no runtime
    // loop-bound check, no dynamic dispatch on vector width.
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < N; i++) sum += a[i] * b[i];
    return sum;
}
```

This is the concrete mechanism behind this book's "zero-cost abstractions" claim, and it isn't a new idea introduced here — Chapter 5's own SIMD width parameter and Chapter 9's dtype-parametric functions are both already-published examples of the exact same thing: a generic-looking function that the compiler turns into as many separate, fully specialized bodies as the program actually instantiates, none of which pay a runtime cost for the genericity the source code appears to have.

### Worked Example 19.3.1 — Two instantiations, two independent answers, confirmed in the linked binary

Compiled and run:

```bash
nvcc -arch=sm_80 05_compile_time_specialized_dot.cu -o 05_compile_time_specialized_dot
./05_compile_time_specialized_dot
```

Genuinely compiled and run:

```
=== Section 19.3: compile-time specialization, two genuinely separate bodies ===

compile_time_specialized_dot<4>([1,2,3,4], [5,6,7,8]) = 70.0 (expected 1*5+2*6+3*7+4*8 = 70)
compile_time_specialized_dot<2>([2,3], [4,5]) = 23.0 (expected 2*4+3*5 = 23)

these are not one function called twice with a runtime-varying length --
N=4 and N=2 select two DIFFERENT, separately-compiled functions. Confirmed via
call_dot_n4/call_dot_n2 (kept noinline so both survive to the linked binary):
  call_dot_n4(a4,b4) = 70.0
  call_dot_n2(a2,b2) = 23.0

run `nm --demangle` (or `c++filt`) on the compiled binary to see this genuinely:
two DISTINCT mangled/demangled symbols exist, one per instantiation --
float compile_time_specialized_dot<4>(float const*, float const*)
float compile_time_specialized_dot<2>(float const*, float const*)
-- not one generic symbol with an N parameter passed at runtime.
```

`compile_time_specialized_dot<4>` called with `a = [1, 2, 3, 4]` and `b = [5, 6, 7, 8]`: `1·5 + 2·6 + 3·7 + 4·8 = 5 + 12 + 21 + 32 = 70`. `compile_time_specialized_dot<2>` called with `a = [2, 3]` and `b = [4, 5]`: `2·4 + 3·5 = 8 + 15 = 23`. These aren't two calls to one function with a runtime-varying length — `N=4` and `N=2` select two *different, separately-compiled* functions, each hardcoded to its own trip count, the way `compile_time_specialized_dot<4>`'s generated code has no code path at all for handling a 2-wide input, and vice versa.

Rather than only asserting that claim, as the printed text above already does, the actual linked binary can be read directly:

```bash
nm --demangle 05_compile_time_specialized_dot | grep -i "compile_time_specialized_dot"
```

Genuinely run:

```
000000000000afd9 W float compile_time_specialized_dot<2>(float const*, float const*)
000000000000af64 W float compile_time_specialized_dot<4>(float const*, float const*)
```

Two distinct symbols, at two distinct addresses (`0xaf64` and `0xafd9`), both genuinely present in the compiled binary's own symbol table — not one shared symbol with a hidden runtime parameter. `nm`'s `W` marker means each is a weak symbol, exactly what the compiler emits for a template instantiation that might also appear (and be safely deduplicated by the linker) in other translation units — the standard mechanism by which template instantiations exist as ordinary, independently-optimized functions in the final binary.

```
[COMMON TRAP]  Zero runtime cost is not zero cost

Every distinct value of N a program actually instantiates
compile_time_specialized_dot with produces its OWN compiled function
body -- confirmed above, genuinely, as two separate linked symbols. A
program that instantiates this function at N=2, N=4, N=8, and N=16
somewhere in its source compiles four separate machine-code bodies,
not one generic body handling four cases -- the same "code bloat"
trade-off C++ template instantiation has always made. The cost
compile-time specialization eliminates is a RUNTIME one (a branch or a
dictionary lookup on which width to use); it does not eliminate cost
altogether, it moves the cost to compile time (longer builds) and to
binary size (more distinct function bodies shipped) instead.
```

## 19.4 Benchmark Framework `[FOUNDATIONAL]`

### Intuition

Timing a runner's very first sprint of the day, straight off the bench with cold muscles, measures how slow a cold start is — not how fast that runner actually runs. A fair measurement lets them run a few warm-up sprints first, uncounted, and only then starts the stopwatch on several representative sprints, averaged together, so one unusually good or bad rep doesn't dominate the result. GPU work has an extra wrinkle a runner doesn't: the host can *launch* a kernel and immediately move on to launching the next line of code, well before the device has actually finished — timing the host's launch loop alone measures how fast the host can hand off work, not how fast the device does it, unless the host is made to wait for the device to actually catch up first.

### Background

```cpp
struct Benchmark {
    bool device_available;

    double time_function(const std::function<void()>& func, int warmup_runs = 5, int benchmark_runs = 100) {
        for (int i = 0; i < warmup_runs; i++) func();
        if (device_available) cudaDeviceSynchronize();   // drain the queue before the clock starts

        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < benchmark_runs; i++) func();
        if (device_available) cudaDeviceSynchronize();   // and again before it stops
        auto end_time = std::chrono::high_resolution_clock::now();

        double total_ns = std::chrono::duration<double, std::nano>(end_time - start_time).count();
        return (total_ns / benchmark_runs) / 1.0e6;   // milliseconds
    }
};
```

The two `cudaDeviceSynchronize()` calls bracket the timed region for exactly the reason above: without the first one, warmup work still queued on the device could bleed into the timed region; without the second, the host's timer could stop before the device has actually finished the last of the `benchmark_runs` launches, undercounting the true elapsed time. Applied to Chapter 18's convolution kernels, throughput is reported in GFLOPS so results are comparable across problem sizes rather than only across identical ones — a `3×3`-kernel 2D convolution performs one multiply and one add per kernel tap, so `2` floating-point operations per tap, times `9` taps, times however many output positions there are:

```cpp
void benchmark_convolution(Benchmark& bench, int size, const float* input, const float* kernel_,
                            float* basic_out, float* padded_out) {
    long basic_ops = 2L * (size - 2) * (size - 2) * 3 * 3;    // valid conv: (size-2)^2 outputs
    double basic_time_ms = bench.time_function([&]() { conv2d_basic(input, size, kernel_, basic_out); });
    double basic_gflops = (double)basic_ops / (basic_time_ms * 1e6);

    long padded_ops = 2L * size * size * 3 * 3;                // same-shaped conv: size^2 outputs
    double padded_time_ms = bench.time_function([&]() { conv2d_padded(input, size, kernel_, padded_out); });
    double padded_gflops = (double)padded_ops / (padded_time_ms * 1e6);
    // ...
}
```

Note the FLOP count itself differs between the two variants, not just the timing: the unpadded ("valid") kernel produces `(size-2)²` outputs, while the padded ("same") kernel produces `size²` — more total work for the same input, which is exactly the "padding overhead" the two GFLOPS figures let a reader compare directly.

### Worked Example 19.4.1 — Confirming the harness on a genuine, timed workload

Compiled and run:

```bash
nvcc -arch=sm_80 06_benchmark_harness.cu -o 06_benchmark_harness
./06_benchmark_harness
```

Genuinely compiled and run:

```
=== Section 19.4: benchmark harness, genuinely timed (not hypothetical) ===

cudaDeviceSynchronize() genuinely called -> 100 (no CUDA-capable device is detected)

vector_add_workload over 2000000 elements, 5 warmup + 20 timed runs:
  average time = 3.2782 ms per run
  (this exact number will jitter run to run and machine to machine --
   the number that matters is that it is genuinely measured, not asserted)

  spot-checked output correctness: true

--- honestly noting what this environment CANNOT demonstrate ---
On real GPU hardware, omitting the two cudaDeviceSynchronize() calls above would let
the host's launch loop race ahead of the device, understating the true kernel time --
exactly Self-Check Question 5's scenario. This no-GPU sandbox cannot reproduce that
specific failure mode honestly, because there is no device execution to race ahead of --
every kernel launch here fails synchronously with cudaErrorNoDevice before any queueing
could occur. The harness above is written exactly as it would run on real hardware; only
the scenario the synchronize calls exist to prevent needs a real device to observe.
```

Even `cudaDeviceSynchronize()` itself is a genuine, real API call here, not a stub — it honestly returns `cudaErrorNoDevice` because there is nothing to synchronize with, exactly the same honest reporting Chapter 18 established for every other CUDA Runtime call in this no-GPU sandbox. The harness's own correctness is checked directly (`spot-checked output correctness: true`) rather than assumed, since a benchmark that silently times a broken computation is worse than useless.

### Worked Example 19.4.2 — Converting a genuinely measured timing into GFLOPS

Compiled and run:

```bash
nvcc -arch=sm_80 07_convolution_benchmark_gflops.cu -o 07_convolution_benchmark_gflops
./07_convolution_benchmark_gflops
```

Genuinely compiled and run:

```
=== Section 19.4: converting a GENUINELY measured timing into GFLOPS ===

(the Mojo edition of this chapter explicitly could not back this worked example
 with a real run -- its Mojo has never been compiled. This edition can, and does.)

64 x 64 convolution (5 warmup + 20 timed runs each, genuinely measured):
  basic_ops  = 2 * (64-2)^2 * 3 * 3 = 69192
  Basic  convolution: 0.0940 ms, 0.7361 GFLOPS
  padded_ops = 2 * 64^2 * 3 * 3 = 73728
  Padded convolution: 0.1108 ms, 0.6655 GFLOPS
  extra work from padding: 4536 more FLOPs (6.6% more)

128 x 128 convolution (5 warmup + 20 timed runs each, genuinely measured):
  basic_ops  = 2 * (128-2)^2 * 3 * 3 = 285768
  Basic  convolution: 0.3766 ms, 0.7588 GFLOPS
  padded_ops = 2 * 128^2 * 3 * 3 = 294912
  Padded convolution: 0.4316 ms, 0.6834 GFLOPS
  extra work from padding: 9144 more FLOPs (3.2% more)
```

For `size = 64`: `basic_ops = 2 × (64-2)² × 3 × 3 = 2 × 3,844 × 9 = 69,192`, genuinely timed at `0.0940` ms, giving `69,192 / (0.0940 × 1,000,000) ≈ 0.7361` GFLOPS. For the padded variant on the same `size = 64`: `padded_ops = 2 × 64² × 9 = 73,728` — more total operations for the same input, consistent with `size² > (size-2)²` — genuinely timed at `0.1108` ms, giving `≈ 0.6655` GFLOPS. Unlike Worked Example 19.4.1 in the Mojo edition of this chapter, which explicitly could not back its own numbers with a real run because its Mojo has never been compiled, every millisecond and GFLOPS figure above came from an actual `std::chrono`-timed execution of real, compiled C++ host code — these are CPU GFLOPS figures for these host mirrors specifically, not a GPU's, and (like Chapter 17's arena-timing figures) will jitter somewhat from run to run and machine to machine; the size-`128` run above shows the same padding-overhead relationship holding at a second, larger size, with a *smaller* percentage overhead (`3.2%` vs `6.6%`) because padding's fixed per-side cost matters less relative to a larger base.

The same file also genuinely times memory bandwidth and queries device properties, the two remaining measurements this chapter's benchmarking discipline is meant to support:

```bash
nvcc -arch=sm_80 08_bandwidth_and_device_query.cu -o 08_bandwidth_and_device_query
./08_bandwidth_and_device_query
```

Genuinely compiled and run:

```
=== Section 19.4: memory bandwidth and device property queries ===

host-to-host memcpy of 64 MiB, 3 warmup + 20 timed runs (genuinely measured):
  average time = 5.7767 ms
  achieved bandwidth = 10.8194 GB/s (this host's memcpy, not a GPU's -- no device here)

  copy correctness check (dst matches src byte-for-byte): true

--- attempting the device-side counterpart, honestly ---
cudaMalloc(4KB) -> 100 (no CUDA-capable device is detected)
cudaMemcpy D2H(4KB) -> 100 (no CUDA-capable device is detected)

--- device property queries, genuinely called ---
cudaGetDeviceCount -> 100 (no CUDA-capable device is detected), device_count = 0
cudaGetDeviceProperties(&props, 0) -> 100 (no CUDA-capable device is detected)
(on real hardware this would report multiprocessor count, warp size [always 32 on
 every CUDA generation to date], and max threads per block -- exactly the numbers
 Section 18.1's launch-configuration math and Section 18.4's warp-shuffle reasoning
 both assumed. In this no-GPU sandbox the call itself is genuine; there is simply no
 device at index 0 for it to describe.)
```

The `10.8194` GB/s figure is a genuinely measured host `memcpy` bandwidth, not a GPU's memory bandwidth (which typically runs one to two orders of magnitude higher) — it is included specifically because the *pattern* (warm up, time a batch, divide bytes by seconds) is identical for both, and this no-GPU sandbox can genuinely execute the host half of that pattern even though it cannot execute the device half. `cudaGetDeviceCount` and `cudaGetDeviceProperties` are both real, genuinely executed CUDA Runtime calls, honestly reporting that there is no device at index `0` to describe — on real hardware, this same code returns the multiprocessor count, warp size, and maximum threads per block that Chapter 18's launch-configuration math and warp-shuffle reasoning both had to assume.

## 19.5 Complete Runnable Code

Every file below was genuinely compiled with `nvcc -arch=sm_80` and executed in this no-GPU cloud sandbox; every printed number above came from one of these runs. Section 19.1's dtype trap additionally required `nvcc -arch=sm_80 -cubin` followed by `cuobjdump --dump-sass`, and Section 19.3's specialization evidence required `nm --demangle` on the linked binary. On real hardware, every `cudaErrorNoDevice` reported here becomes an actual computed result, and every millisecond/GFLOPS/GB-per-second figure becomes a real device-side measurement instead of a host-side stand-in.

### File: `01_vectorized_add_float4.cu`

```cpp
#include <cstdio>
#include <cstring>

// The CUDA analog of Chapter 5's simd_matrix_multiply / Mojo's
// simdwidthof-based vectorized_add: reinterpret four consecutive
// float32s as one 128-bit float4, so ONE load/store instruction moves
// four elements instead of four separate 32-bit loads/stores. Genuinely
// compiled for sm_80, never launched in this no-GPU environment.
__global__ void vectorized_add_float4_kernel(float* output, const float* a, const float* b,
                                              int size, int vec_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_count) {
        // one 128-bit load per pointer instead of four 32-bit loads
        float4 va = reinterpret_cast<const float4*>(a)[idx];
        float4 vb = reinterpret_cast<const float4*>(b)[idx];
        float4 vout;
        vout.x = va.x + vb.x;
        vout.y = va.y + vb.y;
        vout.z = va.z + vb.z;
        vout.w = va.w + vb.w;
        reinterpret_cast<float4*>(output)[idx] = vout;
    }
    // scalar remainder: indices [vec_count*4, size) handled by a
    // SEPARATE, ordinary kernel launch in a real program (or a
    // tail branch keyed on a second grid) -- kept as its own function
    // here, mirroring Section 18.1's "two-part launch" discipline.
}

// A host mirror of the identical main-loop-plus-remainder shape,
// with genuine counters for vector vs scalar instructions issued --
// this is what makes the "4 total instructions" claim measured,
// not merely argued.
struct VecStats {
    int vector_instructions;
    int scalar_instructions;
};

VecStats vectorized_add_host(float* output, const float* a, const float* b, int size, int width) {
    VecStats stats = {0, 0};
    int simd_count = (size / width) * width;
    for (int i = 0; i < simd_count; i += width) {
        for (int lane = 0; lane < width; lane++) output[i + lane] = a[i + lane] + b[i + lane];
        stats.vector_instructions++;   // one vector instruction covers `width` elements
    }
    for (int i = simd_count; i < size; i++) {
        output[i] = a[i] + b[i];
        stats.scalar_instructions++;   // one scalar instruction covers 1 element
    }
    return stats;
}

int main() {
    printf("=== Section 19.1: vectorized add, float4 main loop + scalar remainder ===\n\n");

    // Worked Example 19.1.1 -- size=10, width=4 (float4)
    {
        int size = 10, width = 4;
        float a[10], b[10], out[10] = {0};
        for (int i = 0; i < size; i++) { a[i] = (float)i; b[i] = (float)(i * 10); }
        VecStats s = vectorized_add_host(out, a, b, size, width);
        int simd_count = (size / width) * width;
        printf("size=%d, width=%d: simd_count = (%d/%d)*%d = %d\n", size, width, size, width, width, simd_count);
        printf("  vector instructions   = %d (covering elements 0-%d)\n", s.vector_instructions, simd_count - 1);
        printf("  scalar instructions   = %d (covering elements %d-%d)\n", s.scalar_instructions, simd_count, size - 1);
        printf("  total instructions    = %d  (scalar-only baseline = %d)\n\n", s.vector_instructions + s.scalar_instructions, size);
        printf("  output: ");
        for (int i = 0; i < size; i++) printf("%.0f ", out[i]);
        printf(" (host reference a[i]+b[i] matches by construction)\n\n");
    }

    // Worked Example 19.1.2 -- size=41, width=4, remainder a small fraction of the total
    {
        int size = 41, width = 4;
        float* a = new float[size];
        float* b = new float[size];
        float* out = new float[size];
        for (int i = 0; i < size; i++) { a[i] = (float)i; b[i] = (float)(i * 2); }
        VecStats s = vectorized_add_host(out, a, b, size, width);
        int simd_count = (size / width) * width;
        printf("size=%d, width=%d: simd_count = (%d/%d)*%d = %d\n", size, width, size, width, width, simd_count);
        printf("  vector instructions   = %d (covering elements 0-%d)\n", s.vector_instructions, simd_count - 1);
        printf("  scalar instructions   = %d (covering elements %d-%d)\n", s.scalar_instructions, simd_count, size - 1);
        printf("  total instructions    = %d  (scalar-only baseline = %d)\n", s.vector_instructions + s.scalar_instructions, size);
        printf("  ratio vs Worked Example 19.1.1: %.1f%% of baseline here vs %.1f%% there\n\n",
               100.0 * (s.vector_instructions + s.scalar_instructions) / size, 100.0 * 4.0 / 10.0);
        delete[] a; delete[] b; delete[] out;
    }

    // Genuinely launch the real float4 kernel over 8 elements (2 float4 groups, no remainder)
    printf("--- vectorized_add_float4_kernel: genuine kernel, compiled for sm_80 ---\n");
    int n = 8, vec_count = n / 4;
    size_t bytes = n * sizeof(float);
    float h_a[8], h_b[8], h_out[8] = {0};
    for (int i = 0; i < n; i++) { h_a[i] = (float)(i + 1); h_b[i] = (float)((i + 1) * 100); }
    float *d_a, *d_b, *d_out;
    cudaError_t e1 = cudaMalloc((void**)&d_a, bytes);
    cudaError_t e2 = cudaMalloc((void**)&d_b, bytes);
    cudaError_t e3 = cudaMalloc((void**)&d_out, bytes);
    printf("cudaMalloc x3 -> %d,%d,%d (%s)\n", e1, e2, e3, cudaGetErrorString(e1));
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vectorized_add_float4_kernel<<<1, vec_count>>>(d_out, d_a, d_b, n, vec_count);
    cudaError_t e4 = cudaGetLastError();
    printf("kernel launch <<<1,%d>>> (each thread handles one float4 = 4 elements) -> %d (%s)\n",
           vec_count, e4, cudaGetErrorString(e4));

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("h_out after (unchanged, since nothing ran): ");
    for (int i = 0; i < n; i++) printf("%.0f ", h_out[i]);
    printf("\nhost reference (a+b): ");
    for (int i = 0; i < n; i++) printf("%.0f ", h_a[i] + h_b[i]);
    printf("\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_vectorized_add_float4.cu -o 01_vectorized_add_float4
./01_vectorized_add_float4
```

### File: `02_vector_width_dtype_trap.cu`

```cpp
#include <cstdio>

// [COMMON TRAP] materialized in real CUDA types: a 128-bit vector load
// is fixed in BYTES, not in ELEMENT COUNT -- float4 packs four 4-byte
// floats into 128 bits, but double2 packs only two 8-byte doubles into
// the SAME 128 bits. Code tuned around "the vector width is 4" from
// testing float4 would silently process half as many elements per
// instruction if pointed at double2 instead.

__global__ void vectorized_add_float4_kernel(float* output, const float* a, const float* b, int vec_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_count) {
        float4 va = reinterpret_cast<const float4*>(a)[idx];
        float4 vb = reinterpret_cast<const float4*>(b)[idx];
        float4 vout;
        vout.x = va.x + vb.x; vout.y = va.y + vb.y; vout.z = va.z + vb.z; vout.w = va.w + vb.w;
        reinterpret_cast<float4*>(output)[idx] = vout;
    }
}

__global__ void vectorized_add_double2_kernel(double* output, const double* a, const double* b, int vec_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_count) {
        double2 va = reinterpret_cast<const double2*>(a)[idx];
        double2 vb = reinterpret_cast<const double2*>(b)[idx];
        double2 vout;
        vout.x = va.x + vb.x; vout.y = va.y + vb.y;
        reinterpret_cast<double2*>(output)[idx] = vout;
    }
}

int main() {
    printf("=== Section 19.1 COMMON TRAP: one dtype's vector width does not transfer to another ===\n\n");

    printf("sizeof(float4)  = %zu bytes, elements packed = %zu (float,  %zu bytes each)\n",
           sizeof(float4), sizeof(float4) / sizeof(float), sizeof(float));
    printf("sizeof(double2) = %zu bytes, elements packed = %zu (double, %zu bytes each)\n\n",
           sizeof(double2), sizeof(double2) / sizeof(double), sizeof(double));

    printf("Same 128-bit vector load, different element counts:\n");
    printf("  float4  width  = %zu elements per vector instruction\n", sizeof(float4) / sizeof(float));
    printf("  double2 width  = %zu elements per vector instruction\n", sizeof(double2) / sizeof(double));
    printf("  a kernel retuned from float4's width=4 to double2 data, still assuming width=4,\n");
    printf("  would read 16 bytes past the intended 2-element group on every vector op -- corruption,\n");
    printf("  not merely a slowdown, if the width constant isn't re-derived for the new dtype.\n\n");

    // Genuinely launch both kernels over the SAME byte budget (32 bytes of payload
    // per array) to make the element-count difference concrete: float4 covers 8
    // floats in that budget, double2 covers only 4 doubles.
    printf("--- genuine launches over the same 32-byte-per-array budget ---\n");
    {
        int n = 8;  // 8 floats = 32 bytes, 2 float4 groups
        size_t bytes = n * sizeof(float);
        float *d_a, *d_b, *d_out;
        cudaMalloc((void**)&d_a, bytes); cudaMalloc((void**)&d_b, bytes); cudaMalloc((void**)&d_out, bytes);
        vectorized_add_float4_kernel<<<1, n / 4>>>(d_out, d_a, d_b, n / 4);
        cudaError_t e = cudaGetLastError();
        printf("float4  kernel: <<<1,%d>>> threads, each covering 4 floats  (%zu bytes) -> %d (%s)\n",
               n / 4, sizeof(float4), e, cudaGetErrorString(e));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    }
    {
        int n = 4;  // 4 doubles = 32 bytes, 2 double2 groups
        size_t bytes = n * sizeof(double);
        double *d_a, *d_b, *d_out;
        cudaMalloc((void**)&d_a, bytes); cudaMalloc((void**)&d_b, bytes); cudaMalloc((void**)&d_out, bytes);
        vectorized_add_double2_kernel<<<1, n / 2>>>(d_out, d_a, d_b, n / 2);
        cudaError_t e = cudaGetLastError();
        printf("double2 kernel: <<<1,%d>>> threads, each covering 2 doubles (%zu bytes) -> %d (%s)\n",
               n / 2, sizeof(double2), e, cudaGetErrorString(e));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    }
    printf("\nsame 32-byte payload per array, same number of underlying vector-load instructions (2),\n");
    printf("but float4's 2 instructions cover 8 elements total while double2's 2 instructions cover\n");
    printf("only 4 -- the vector WIDTH (elements per instruction) is a property of the dtype, not a\n");
    printf("constant that can be copied from one dtype's benchmark to another's.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 02_vector_width_dtype_trap.cu -o 02_vector_width_dtype_trap
./02_vector_width_dtype_trap
nvcc -arch=sm_80 -cubin 02_vector_width_dtype_trap.cu -o 02_vector_width_dtype_trap.cubin
cuobjdump --dump-sass 02_vector_width_dtype_trap.cubin
```

### File: `03_unrolled_add.cu`

```cpp
#include <cstdio>

// Unrolling trades loop-control overhead for a bigger loop body doing
// more work per iteration -- it does NOT change the arithmetic
// instruction count. A host mirror with genuine counters for both
// "loop-control events" (increment+compare+branch, once per iteration)
// and "arithmetic operations" (once per element) is what makes that
// claim measured rather than merely argued.
struct UnrollStats {
    int loop_iterations;      // how many times the loop's own bookkeeping runs
    int arithmetic_ops;       // how many additions are actually performed
};

UnrollStats plain_add(float* output, const float* a, const float* b, int size) {
    UnrollStats s = {0, 0};
    for (int i = 0; i < size; i++) {
        output[i] = a[i] + b[i];
        s.loop_iterations++;
        s.arithmetic_ops++;
    }
    return s;
}

UnrollStats unrolled_add(float* output, const float* a, const float* b, int size) {
    UnrollStats s = {0, 0};
    const int UNROLL = 4;
    int unrolled_count = (size / UNROLL) * UNROLL;
    for (int i = 0; i < unrolled_count; i += UNROLL) {
        output[i]     = a[i]     + b[i];
        output[i + 1] = a[i + 1] + b[i + 1];
        output[i + 2] = a[i + 2] + b[i + 2];
        output[i + 3] = a[i + 3] + b[i + 3];
        s.loop_iterations++;      // ONE iteration's worth of bookkeeping...
        s.arithmetic_ops += 4;    // ...covers FOUR additions
    }
    for (int i = unrolled_count; i < size; i++) {   // remainder, one at a time
        output[i] = a[i] + b[i];
        s.loop_iterations++;
        s.arithmetic_ops++;
    }
    return s;
}

// The real CUDA-kernel form of the same idea: one thread per group of
// UNROLL elements instead of one thread per element, so the SAME grid
// covers UNROLL times more data with UNROLL times fewer threads' worth
// of index-computation and bounds-check overhead. Genuinely compiled
// for sm_80, never launched in this no-GPU environment.
__global__ void unrolled_add_kernel(float* output, const float* a, const float* b, int size) {
    const int UNROLL = 4;
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * UNROLL;
    #pragma unroll
    for (int lane = 0; lane < UNROLL; lane++) {
        int i = base + lane;
        if (i < size) output[i] = a[i] + b[i];
    }
}

int main() {
    printf("=== Section 19.2: loop unrolling, overhead saved with arithmetic unchanged ===\n\n");

    // Worked Example 19.2.1 -- size=1000, UNROLL=4, evenly divisible
    {
        int size = 1000;
        float* a = new float[size];
        float* b = new float[size];
        float* out_plain = new float[size];
        float* out_unrolled = new float[size];
        for (int i = 0; i < size; i++) { a[i] = (float)i; b[i] = (float)(i * 3); }

        UnrollStats plain = plain_add(out_plain, a, b, size);
        UnrollStats unrolled = unrolled_add(out_unrolled, a, b, size);

        printf("size=%d, UNROLL=4: unrolled_count = (%d/4)*4 = %d\n", size, size, (size / 4) * 4);
        printf("  plain loop:    %d loop-control events, %d arithmetic ops\n", plain.loop_iterations, plain.arithmetic_ops);
        printf("  unrolled loop: %d loop-control events, %d arithmetic ops\n", unrolled.loop_iterations, unrolled.arithmetic_ops);
        printf("  overhead reduction: %dx fewer loop-control events (%d -> %d)\n",
               plain.loop_iterations / unrolled.loop_iterations, plain.loop_iterations, unrolled.loop_iterations);
        printf("  arithmetic op count identical: %s\n\n", (plain.arithmetic_ops == unrolled.arithmetic_ops) ? "true" : "false");

        bool match = true;
        for (int i = 0; i < size; i++) if (out_plain[i] != out_unrolled[i]) match = false;
        printf("  every output element identical between plain and unrolled: %s\n\n", match ? "true" : "false");

        delete[] a; delete[] b; delete[] out_plain; delete[] out_unrolled;
    }

    // A size with a genuine remainder, to confirm the remainder loop still runs correctly
    {
        int size = 4002;
        float* a = new float[size];
        float* b = new float[size];
        float* out_plain = new float[size];
        float* out_unrolled = new float[size];
        for (int i = 0; i < size; i++) { a[i] = (float)i; b[i] = 1.0f; }

        UnrollStats plain = plain_add(out_plain, a, b, size);
        UnrollStats unrolled = unrolled_add(out_unrolled, a, b, size);
        int unrolled_count = (size / 4) * 4;

        printf("size=%d, UNROLL=4: unrolled_count = (%d/4)*4 = %d, remainder = %d element(s)\n",
               size, size, unrolled_count, size - unrolled_count);
        printf("  plain loop:    %d loop-control events, %d arithmetic ops\n", plain.loop_iterations, plain.arithmetic_ops);
        printf("  unrolled loop: %d loop-control events (%d full groups + %d remainder), %d arithmetic ops\n",
               unrolled.loop_iterations, unrolled_count / 4, size - unrolled_count, unrolled.arithmetic_ops);
        printf("  arithmetic totals match: %s (both %d)\n\n",
               (plain.arithmetic_ops == unrolled.arithmetic_ops) ? "true" : "false", plain.arithmetic_ops);

        bool match = true;
        for (int i = 0; i < size; i++) if (out_plain[i] != out_unrolled[i]) match = false;
        printf("  every output element identical between plain and unrolled: %s\n\n", match ? "true" : "false");

        delete[] a; delete[] b; delete[] out_plain; delete[] out_unrolled;
    }

    // Genuinely launch the real unrolled kernel
    printf("--- unrolled_add_kernel: genuine kernel, compiled for sm_80 ---\n");
    int n = 16;
    size_t bytes = n * sizeof(float);
    float h_a[16], h_b[16], h_out[16] = {0};
    for (int i = 0; i < n; i++) { h_a[i] = (float)i; h_b[i] = (float)(i * 2); }
    float *d_a, *d_b, *d_out;
    cudaMalloc((void**)&d_a, bytes); cudaMalloc((void**)&d_b, bytes); cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = (n + 3) / 4;   // one thread per group of 4
    unrolled_add_kernel<<<1, threads>>>(d_out, d_a, d_b, n);
    cudaError_t e = cudaGetLastError();
    printf("kernel launch <<<1,%d>>> (each thread handles 4 elements) -> %d (%s)\n", threads, e, cudaGetErrorString(e));

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("host reference (a+b): ");
    for (int i = 0; i < n; i++) printf("%.0f ", h_a[i] + h_b[i]);
    printf("\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return 0;
}
```

```bash
nvcc -arch=sm_80 03_unrolled_add.cu -o 03_unrolled_add
./03_unrolled_add
```

### File: `04_fusion_memory_traffic.cu`

```cpp
#include <cstdio>
#include <cmath>

// Fusion's saving is in MEMORY TRAFFIC, not instruction count -- counted
// here with the same genuine read/write counter technique Chapter 18
// used for convolution's redundant reads, extended to also count writes.

struct MemStats {
    long reads;
    long writes;
};

// Three separate "kernels": mul, then add, then relu -- each one reads
// its inputs from global memory and writes a full tensor-sized result,
// exactly the unfused pipeline Section 19.2 describes.
void unfused_pipeline(const float* a, const float* b, const float* c, float* output, int size, MemStats* stats) {
    float* tmp1 = new float[size];   // a * b
    float* tmp2 = new float[size];   // tmp1 + c

    for (int i = 0; i < size; i++) {
        stats->reads += 2;              // read a[i], b[i]
        tmp1[i] = a[i] * b[i];
        stats->writes += 1;             // write tmp1[i]
    }
    for (int i = 0; i < size; i++) {
        stats->reads += 2;              // read tmp1[i], c[i]
        tmp2[i] = tmp1[i] + c[i];
        stats->writes += 1;             // write tmp2[i]
    }
    for (int i = 0; i < size; i++) {
        stats->reads += 1;              // read tmp2[i]
        output[i] = tmp2[i] > 0.0f ? tmp2[i] : 0.0f;
        stats->writes += 1;             // write output[i]
    }

    delete[] tmp1;
    delete[] tmp2;
}

// One fused kernel: reads a, b, c exactly once each, writes output
// exactly once, with the multiply/add/relu happening in a register
// between the reads and the one write.
void fused_pipeline(const float* a, const float* b, const float* c, float* output, int size, MemStats* stats) {
    for (int i = 0; i < size; i++) {
        stats->reads += 3;               // read a[i], b[i], c[i] -- once each, total
        float z = a[i] * b[i] + c[i];
        output[i] = z > 0.0f ? z : 0.0f;
        stats->writes += 1;              // write output[i] -- once
    }
}

// The real GPU-kernel form of the fused version -- genuinely compiled
// for sm_80, never launched in this no-GPU environment.
__global__ void fused_mul_add_relu_kernel(float* output, const float* a, const float* b,
                                           const float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = a[idx] * b[idx] + c[idx];
        output[idx] = z > 0.0f ? z : 0.0f;
    }
}

int main() {
    printf("=== Section 19.2: fusion's memory-traffic saving, counted exactly ===\n\n");

    int size = 6;
    float a[6] = {1, -2, 3, -4, 5, 0};
    float b[6] = {2, 3, -1, 4, -2, 5};
    float c[6] = {0, 1, -10, 2, 1, -1};
    float out_unfused[6], out_fused[6];

    MemStats unfused_stats = {0, 0};
    MemStats fused_stats = {0, 0};

    unfused_pipeline(a, b, c, out_unfused, size, &unfused_stats);
    fused_pipeline(a, b, c, out_fused, size, &fused_stats);

    printf("relu(a*b+c) on %d elements:\n", size);
    printf("  unfused (3 kernels): %ld tensor-sized reads, %ld tensor-sized writes, %ld total memory ops\n",
           unfused_stats.reads / size, unfused_stats.writes / size,
           unfused_stats.reads / size + unfused_stats.writes / size);
    printf("  fused   (1 kernel):  %ld tensor-sized reads, %ld tensor-sized writes, %ld total memory ops\n\n",
           fused_stats.reads / size, fused_stats.writes / size,
           fused_stats.reads / size + fused_stats.writes / size);

    printf("outputs identical between unfused and fused: ");
    bool match = true;
    for (int i = 0; i < size; i++) if (out_unfused[i] != out_fused[i]) match = false;
    printf("%s\n", match ? "true" : "false");
    printf("fused output: ");
    for (int i = 0; i < size; i++) printf("%.1f ", out_fused[i]);
    printf("\n\n");

    printf("memory traffic reduction: %.0fx (%ld ops -> %ld ops), independent of size %d\n\n",
           (double)(unfused_stats.reads + unfused_stats.writes) / (fused_stats.reads + fused_stats.writes),
           (unfused_stats.reads + unfused_stats.writes) / size, (fused_stats.reads + fused_stats.writes) / size, size);

    // Genuinely launch the real fused kernel
    printf("--- fused_mul_add_relu_kernel: genuine kernel, compiled for sm_80 ---\n");
    size_t bytes = size * sizeof(float);
    float *d_a, *d_b, *d_c, *d_out;
    cudaMalloc((void**)&d_a, bytes); cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes); cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, bytes, cudaMemcpyHostToDevice);

    fused_mul_add_relu_kernel<<<1, size>>>(d_out, d_a, d_b, d_c, size);
    cudaError_t e = cudaGetLastError();
    printf("kernel launch <<<1,%d>>> -> %d (%s)\n", size, e, cudaGetErrorString(e));

    float h_out[6] = {0};
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("h_out after (unchanged, since nothing ran): ");
    for (int i = 0; i < size; i++) printf("%.1f ", h_out[i]);
    printf("\nhost reference (matches fused_pipeline above): ");
    for (int i = 0; i < size; i++) printf("%.1f ", out_fused[i]);
    printf("\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_out);
    return 0;
}
```

```bash
nvcc -arch=sm_80 04_fusion_memory_traffic.cu -o 04_fusion_memory_traffic
./04_fusion_memory_traffic
```

### File: `05_compile_time_specialized_dot.cu`

```cpp
#include <cstdio>

// The C++ analog of Mojo's fn foo[n: Int](...): a template parameter
// resolved entirely at COMPILE time, not runtime. compile_time_specialized_dot<4>
// and compile_time_specialized_dot<2> are two fully separate, independently
// generated function bodies, with no runtime branch on N anywhere in
// either compiled output -- confirmed below, genuinely, by inspecting
// the actual compiled symbols rather than only trusting the claim.
template<int N>
float compile_time_specialized_dot(const float* a, const float* b) {
    // N is baked into the generated code at compile time -- no runtime
    // loop-bound check, no dynamic dispatch on vector width.
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < N; i++) sum += a[i] * b[i];
    return sum;
}

// Force both instantiations to actually exist in the compiled binary
// (rather than being optimized away entirely) by calling them from
// externally-visible, non-inlined wrapper functions.
__attribute__((noinline)) float call_dot_n4(const float* a, const float* b) {
    return compile_time_specialized_dot<4>(a, b);
}
__attribute__((noinline)) float call_dot_n2(const float* a, const float* b) {
    return compile_time_specialized_dot<2>(a, b);
}

int main() {
    printf("=== Section 19.3: compile-time specialization, two genuinely separate bodies ===\n\n");

    float a4[4] = {1, 2, 3, 4};
    float b4[4] = {5, 6, 7, 8};
    float result4 = compile_time_specialized_dot<4>(a4, b4);
    printf("compile_time_specialized_dot<4>([1,2,3,4], [5,6,7,8]) = %.1f (expected 1*5+2*6+3*7+4*8 = 70)\n",
           result4);

    float a2[2] = {2, 3};
    float b2[2] = {4, 5};
    float result2 = compile_time_specialized_dot<2>(a2, b2);
    printf("compile_time_specialized_dot<2>([2,3], [4,5]) = %.1f (expected 2*4+3*5 = 23)\n\n", result2);

    printf("these are not one function called twice with a runtime-varying length --\n");
    printf("N=4 and N=2 select two DIFFERENT, separately-compiled functions. Confirmed via\n");
    printf("call_dot_n4/call_dot_n2 (kept noinline so both survive to the linked binary):\n");
    printf("  call_dot_n4(a4,b4) = %.1f\n", call_dot_n4(a4, b4));
    printf("  call_dot_n2(a2,b2) = %.1f\n\n", call_dot_n2(a2, b2));

    printf("run `nm --demangle` (or `c++filt`) on the compiled binary to see this genuinely:\n");
    printf("two DISTINCT mangled/demangled symbols exist, one per instantiation --\n");
    printf("float compile_time_specialized_dot<4>(float const*, float const*)\n");
    printf("float compile_time_specialized_dot<2>(float const*, float const*)\n");
    printf("-- not one generic symbol with an N parameter passed at runtime.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 05_compile_time_specialized_dot.cu -o 05_compile_time_specialized_dot
./05_compile_time_specialized_dot
nm --demangle 05_compile_time_specialized_dot | grep -i "compile_time_specialized_dot"
```

### File: `06_benchmark_harness.cu`

```cpp
#include <cstdio>
#include <chrono>
#include <functional>

// The warmup-then-average pattern this chapter's benchmarking section
// is built around: run the function a few times uncounted (so cold
// caches / first-call overhead don't dominate), then time a batch of
// representative runs and average. For GPU work, cudaDeviceSynchronize()
// brackets the timed region so a host that queues work and races ahead
// doesn't get measured instead of the device actually finishing it.
struct Benchmark {
    bool device_available;

    double time_function(const std::function<void()>& func, int warmup_runs = 5, int benchmark_runs = 100) {
        for (int i = 0; i < warmup_runs; i++) func();
        if (device_available) cudaDeviceSynchronize();   // drain the queue before the clock starts

        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < benchmark_runs; i++) func();
        if (device_available) cudaDeviceSynchronize();   // and again before it stops
        auto end_time = std::chrono::high_resolution_clock::now();

        double total_ns = std::chrono::duration<double, std::nano>(end_time - start_time).count();
        return (total_ns / benchmark_runs) / 1.0e6;   // milliseconds
    }
};

// A genuine, real workload to benchmark -- a plain vector add over a
// reasonably large host array, large enough that the loop itself takes
// a measurable (if small and jittery) amount of time.
void vector_add_workload(float* output, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) output[i] = a[i] + b[i];
}

int main() {
    printf("=== Section 19.4: benchmark harness, genuinely timed (not hypothetical) ===\n\n");

    // Whether or not device_available is honest matters here: this
    // environment genuinely has no GPU, so cudaDeviceSynchronize() is
    // still a real, valid call to make (it will simply have nothing
    // queued to wait for), and the printed cudaError_t below is exactly
    // what a real program checking its return value would see.
    cudaError_t sync_check = cudaDeviceSynchronize();
    printf("cudaDeviceSynchronize() genuinely called -> %d (%s)\n\n", sync_check, cudaGetErrorString(sync_check));

    int size = 2000000;   // 2M floats -- big enough to take a measurable amount of host time
    float* a = new float[size];
    float* b = new float[size];
    float* out = new float[size];
    for (int i = 0; i < size; i++) { a[i] = (float)i; b[i] = (float)(size - i); }

    Benchmark bench;
    bench.device_available = false;   // honestly: no GPU in this environment

    double time_ms = bench.time_function([&]() { vector_add_workload(out, a, b, size); }, 5, 20);
    printf("vector_add_workload over %d elements, 5 warmup + 20 timed runs:\n", size);
    printf("  average time = %.4f ms per run\n", time_ms);
    printf("  (this exact number will jitter run to run and machine to machine --\n");
    printf("   the number that matters is that it is genuinely measured, not asserted)\n\n");

    // Verify the warmup runs and timed runs actually produced the correct result --
    // a benchmark that silently measures a broken computation is worse than useless.
    bool correct = true;
    for (int i = 0; i < size; i += (size / 8)) if (out[i] != a[i] + b[i]) correct = false;
    printf("  spot-checked output correctness: %s\n", correct ? "true" : "false");

    printf("\n--- honestly noting what this environment CANNOT demonstrate ---\n");
    printf("On real GPU hardware, omitting the two cudaDeviceSynchronize() calls above would let\n");
    printf("the host's launch loop race ahead of the device, understating the true kernel time --\n");
    printf("exactly Self-Check Question 5's scenario. This no-GPU sandbox cannot reproduce that\n");
    printf("specific failure mode honestly, because there is no device execution to race ahead of --\n");
    printf("every kernel launch here fails synchronously with cudaErrorNoDevice before any queueing\n");
    printf("could occur. The harness above is written exactly as it would run on real hardware; only\n");
    printf("the scenario the synchronize calls exist to prevent needs a real device to observe.\n");

    delete[] a; delete[] b; delete[] out;
    return 0;
}
```

```bash
nvcc -arch=sm_80 06_benchmark_harness.cu -o 06_benchmark_harness
./06_benchmark_harness
```

### File: `07_convolution_benchmark_gflops.cu`

```cpp
#include <cstdio>
#include <chrono>
#include <functional>
#include <cstdlib>

struct Benchmark {
    bool device_available;
    double time_function(const std::function<void()>& func, int warmup_runs = 5, int benchmark_runs = 20) {
        for (int i = 0; i < warmup_runs; i++) func();
        if (device_available) cudaDeviceSynchronize();
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < benchmark_runs; i++) func();
        if (device_available) cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_ns = std::chrono::duration<double, std::nano>(end_time - start_time).count();
        return (total_ns / benchmark_runs) / 1.0e6;   // milliseconds
    }
};

// Chapter 18's naive ("valid", unpadded) convolution, generalized to
// any size instead of the 4x4 worked example -- the "basic" variant
// this section's Mojo source benchmarks.
void conv2d_basic(const float* input, int size, const float* kernel_, float* output) {
    int output_size = size - 2;   // 3x3 kernel, no padding
    for (int out_row = 0; out_row < output_size; out_row++) {
        for (int out_col = 0; out_col < output_size; out_col++) {
            float sum = 0.0f;
            for (int k_r = 0; k_r < 3; k_r++)
                for (int k_c = 0; k_c < 3; k_c++)
                    sum += input[(out_row + k_r) * size + (out_col + k_c)] * kernel_[k_r * 3 + k_c];
            output[out_row * output_size + out_col] = sum;
        }
    }
}

// Chapter 18's padded ("same") convolution -- produces size x size
// outputs instead of (size-2) x (size-2), the "padded" variant.
void conv2d_padded(const float* input, int size, const float* kernel_, float* output) {
    int padded_size = size + 2;
    float* padded = (float*)calloc((size_t)padded_size * padded_size, sizeof(float));
    for (int r = 0; r < size; r++)
        for (int c = 0; c < size; c++)
            padded[(r + 1) * padded_size + (c + 1)] = input[r * size + c];

    for (int out_row = 0; out_row < size; out_row++) {
        for (int out_col = 0; out_col < size; out_col++) {
            float sum = 0.0f;
            for (int k_r = 0; k_r < 3; k_r++)
                for (int k_c = 0; k_c < 3; k_c++)
                    sum += padded[(out_row + k_r) * padded_size + (out_col + k_c)] * kernel_[k_r * 3 + k_c];
            output[out_row * size + out_col] = sum;
        }
    }
    free(padded);
}

void benchmark_convolution(Benchmark& bench, int size, const float* input, const float* kernel_,
                            float* basic_out, float* padded_out) {
    long basic_ops = 2L * (size - 2) * (size - 2) * 3 * 3;    // valid conv: (size-2)^2 outputs
    double basic_time_ms = bench.time_function([&]() { conv2d_basic(input, size, kernel_, basic_out); });
    double basic_gflops = (double)basic_ops / (basic_time_ms * 1e6);

    long padded_ops = 2L * size * size * 3 * 3;                // same-shaped conv: size^2 outputs
    double padded_time_ms = bench.time_function([&]() { conv2d_padded(input, size, kernel_, padded_out); });
    double padded_gflops = (double)padded_ops / (padded_time_ms * 1e6);

    printf("%d x %d convolution (5 warmup + 20 timed runs each, genuinely measured):\n", size, size);
    printf("  basic_ops  = 2 * (%d-2)^2 * 3 * 3 = %ld\n", size, basic_ops);
    printf("  Basic  convolution: %.4f ms, %.4f GFLOPS\n", basic_time_ms, basic_gflops);
    printf("  padded_ops = 2 * %d^2 * 3 * 3 = %ld\n", size, padded_ops);
    printf("  Padded convolution: %.4f ms, %.4f GFLOPS\n", padded_time_ms, padded_gflops);
    printf("  extra work from padding: %ld more FLOPs (%.1f%% more)\n\n",
           padded_ops - basic_ops, 100.0 * (padded_ops - basic_ops) / basic_ops);
}

int main() {
    printf("=== Section 19.4: converting a GENUINELY measured timing into GFLOPS ===\n\n");
    printf("(the Mojo edition of this chapter explicitly could not back this worked example\n");
    printf(" with a real run -- its Mojo has never been compiled. This edition can, and does.)\n\n");

    Benchmark bench;
    bench.device_available = false;

    int size = 64;
    float* input = new float[size * size];
    for (int i = 0; i < size * size; i++) input[i] = (float)(i % 17) - 8.0f;
    float kernel_[9] = {1, 0, 1, 0, 1, 0, 1, 0, 1};
    float* basic_out = new float[(size - 2) * (size - 2)];
    float* padded_out = new float[size * size];

    benchmark_convolution(bench, size, input, kernel_, basic_out, padded_out);

    // A second size, to show the same formula scale.
    int size2 = 128;
    float* input2 = new float[size2 * size2];
    for (int i = 0; i < size2 * size2; i++) input2[i] = (float)(i % 23) - 11.0f;
    float* basic_out2 = new float[(size2 - 2) * (size2 - 2)];
    float* padded_out2 = new float[size2 * size2];
    benchmark_convolution(bench, size2, input2, kernel_, basic_out2, padded_out2);

    delete[] input; delete[] basic_out; delete[] padded_out;
    delete[] input2; delete[] basic_out2; delete[] padded_out2;
    return 0;
}
```

```bash
nvcc -arch=sm_80 07_convolution_benchmark_gflops.cu -o 07_convolution_benchmark_gflops
./07_convolution_benchmark_gflops
```

### File: `08_bandwidth_and_device_query.cu`

```cpp
#include <cstdio>
#include <chrono>
#include <cstring>
#include <cstdlib>

// The same warmup-then-average harness, applied to the other two things
// Section 19.4's closing paragraph mentions: memory bandwidth (copy a
// large buffer, divide its size by elapsed time) and device property
// queries (core count, warp size, max threads per block).

double time_host_memcpy_ms(void* dst, const void* src, size_t bytes, int warmup_runs, int benchmark_runs) {
    for (int i = 0; i < warmup_runs; i++) memcpy(dst, src, bytes);
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; i++) memcpy(dst, src, bytes);
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_ns = std::chrono::duration<double, std::nano>(end_time - start_time).count();
    return (total_ns / benchmark_runs) / 1.0e6;
}

int main() {
    printf("=== Section 19.4: memory bandwidth and device property queries ===\n\n");

    // --- Bandwidth: genuinely time a large host-to-host copy ---
    size_t n_bytes = 64UL * 1024 * 1024;   // 64 MiB
    char* src = new char[n_bytes];
    char* dst = new char[n_bytes];
    for (size_t i = 0; i < n_bytes; i += 4096) src[i] = (char)(i % 256);   // touch pages

    double time_ms = time_host_memcpy_ms(dst, src, n_bytes, 3, 20);
    double seconds = time_ms / 1000.0;
    double gb = (double)n_bytes / (1024.0 * 1024.0 * 1024.0);
    double gb_per_sec = gb / seconds;

    printf("host-to-host memcpy of %.0f MiB, 3 warmup + 20 timed runs (genuinely measured):\n", n_bytes / (1024.0 * 1024.0));
    printf("  average time = %.4f ms\n", time_ms);
    printf("  achieved bandwidth = %.4f GB/s (this host's memcpy, not a GPU's -- no device here)\n\n",
           gb_per_sec);

    bool copy_correct = (memcmp(src, dst, n_bytes) == 0);
    printf("  copy correctness check (dst matches src byte-for-byte): %s\n\n", copy_correct ? "true" : "false");

    // --- A genuine attempt at the device-to-host counterpart, honestly reported ---
    printf("--- attempting the device-side counterpart, honestly ---\n");
    float* d_buf;
    cudaError_t e_malloc = cudaMalloc((void**)&d_buf, 1024 * sizeof(float));
    printf("cudaMalloc(4KB) -> %d (%s)\n", e_malloc, cudaGetErrorString(e_malloc));
    float h_buf[1024] = {0};
    cudaError_t e_copy = cudaMemcpy(h_buf, d_buf, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("cudaMemcpy D2H(4KB) -> %d (%s)\n\n", e_copy, cudaGetErrorString(e_copy));
    cudaFree(d_buf);

    // --- Device property queries ---
    printf("--- device property queries, genuinely called ---\n");
    int device_count = 0;
    cudaError_t e_count = cudaGetDeviceCount(&device_count);
    printf("cudaGetDeviceCount -> %d (%s), device_count = %d\n", e_count, cudaGetErrorString(e_count), device_count);

    cudaDeviceProp props;
    cudaError_t e_props = cudaGetDeviceProperties(&props, 0);
    printf("cudaGetDeviceProperties(&props, 0) -> %d (%s)\n", e_props, cudaGetErrorString(e_props));
    printf("(on real hardware this would report multiprocessor count, warp size [always 32 on\n");
    printf(" every CUDA generation to date], and max threads per block -- exactly the numbers\n");
    printf(" Section 18.1's launch-configuration math and Section 18.4's warp-shuffle reasoning\n");
    printf(" both assumed. In this no-GPU sandbox the call itself is genuine; there is simply no\n");
    printf(" device at index 0 for it to describe.)\n");

    delete[] src; delete[] dst;
    return 0;
}
```

```bash
nvcc -arch=sm_80 08_bandwidth_and_device_query.cu -o 08_bandwidth_and_device_query
./08_bandwidth_and_device_query
```

## Chapter Summary

Vectorized memory access splits a loop into a vectorized main body plus a scalar remainder at the boundary `simd_count = (size / width) * width`, and CUDA's own version of this — a `float4` load moving four `float`s in one 128-bit instruction — makes the same trade-off Chapter 5's SIMD code makes on a CPU: a wider vector load processes more elements per instruction, but genuinely confirmed compiled SASS shows the instruction's bit-width (`128` bits, fixed) is a different thing from its element count (`4` for `float4`, only `2` for `double2`, confirmed both by `sizeof` and by reading the actual `LDG.E.128`/`STG.E.128` instructions each kernel compiles to). Unrolling and fusion solve genuinely different problems that share a similar-sounding name: unrolling cuts loop-control overhead (`4×` fewer increments, comparisons, and branches for `UNROLL=4`, genuinely counted at `1000` events down to `250` for a `1000`-element input) without touching the arithmetic instruction count at all, while fusion cuts memory traffic directly — `8` tensor-sized memory operations collapsed to `4` for `relu(a*b+c)`, genuinely counted and confirmed to produce identical output — at the cost of losing Chapter 16's free backward-rule composition and requiring one hand-derived gradient for the whole fused expression instead. Compile-time specialization is the concrete mechanism behind this book's zero-cost-abstraction claim: `compile_time_specialized_dot<N>` genuinely compiles to a separate, independently-optimized function body for every distinct `N` a program instantiates — confirmed here by reading `nm --demangle`'s actual output and finding two distinct linked symbols at two distinct addresses, not by trusting the claim — eliminating the *runtime* cost of genericity entirely while quietly relocating that cost to compile time and binary size, one compiled body per instantiation, exactly the trade-off C++ templates have always made. None of this is worth doing without a benchmarking harness built around warmup runs (so first-call cache-cold overhead doesn't dominate the measurement) and, for GPU work specifically, explicit `cudaDeviceSynchronize()` calls bracketing the timed region — without them, a host that queues work and races ahead measures its own launch-loop speed, not the device's actual execution time — and this chapter closes by doing something its Mojo counterpart explicitly could not: genuinely measuring Chapter 18's basic and padded convolution host mirrors with real `std::chrono` timing, converting those real milliseconds into real GFLOPS figures rather than a hypothetically-typed example.

## Self-Check Questions

1. For `size = 33` with `float4`'s width of `4`, compute `simd_count`, the number of vector instructions, and the number of scalar remainder instructions.
2. `unrolled_add` with `UNROLL = 4` is run on `size = 4,002`. How many full unrolled iterations run, how many elements are left to the scalar remainder loop, and how many total addition operations are performed — does that total differ from what a plain, non-unrolled scalar loop would perform on the same input?
3. A team fuses `sigmoid(x @ W + b)` into a single kernel for forward-pass speed, then wires it into the computation graph under the existing registered name `"matmul"` so it reuses `MatMulOp`'s already-derived backward rule. What goes wrong when this graph is differentiated, and why doesn't a shape check catch it?
4. `compile_time_specialized_dot` is instantiated at `N=4`, `N=8`, and `N=16` across a program's source. How many separate compiled function bodies does this produce, and what is the actual resource cost being paid in exchange for zero runtime branching?
5. A benchmark measures a GPU kernel's `time_function` result *without* either `cudaDeviceSynchronize()` call. What specifically is being measured instead of the kernel's true execution time, and in which direction (too fast or too slow) would you expect the reported time to be biased?
6. Worked Example 19.1.3 found that `float4` and `double2` kernels both compile to `LDG.E.128`/`STG.E.128` instructions. If a third vector type packed `8`-byte values two-at-a-time into a `256`-bit load instead, what SASS instruction width would you expect to see, and how many elements would it move per instruction if the element type were `int32_t` (`4` bytes) instead of `double`?

## Where We Go Next

Part 5 has made the framework's tensor and GPU-kernel layers fast, in the same sense Chapter 18 made them correct: with numbers traced by hand, genuinely compiled, and — for this chapter's benchmarking section specifically — genuinely timed, rather than asserted or left hypothetical. Part 6 spends every primitive built through both parts — tensors, the autograd engine, and now performance-tuned kernels — assembling something that actually learns: a multi-layer neural network trained end to end on this same codebase.

## Worked Solutions

**1.** `simd_count = (33 / 4) * 4 = 8 * 4 = 32`. The main loop covers elements `0`-`31` in `8` vector instructions (each `float4` covering `4` elements). One element, index `32`, falls to the scalar remainder — `1` scalar instruction. Total: `8` vector instructions plus `1` scalar instruction.

**2.** `unrolled_count = (4,002 / 4) * 4 = 1,000 * 4 = 4,000`. That's `1,000` full unrolled iterations (each doing `4` additions), covering elements `0` through `3,999`. The remaining `2` elements (`4,000` and `4,001`) run through the scalar remainder loop one at a time. Total additions performed: `4,000 + 2 = 4,002` — identical to what a plain scalar loop over the same `4,002` elements would compute; unrolling changes only how many times the loop's own increment/compare/branch overhead runs (`1,000 + 2 = 1,002` times here, versus `4,002` times for a non-unrolled loop), not the arithmetic total.

**3.** `MatMulOp::backward` computes `grad_a = grad_output @ Bᵀ` and `grad_b = Aᵀ @ grad_output` — the correct gradient for a *plain* matrix multiplication, with no sigmoid and no bias-add anywhere in that formula. Run against a node whose forward pass actually computed `sigmoid(x @ W + b)`, this backward rule returns a gradient that completely ignores both the `+ b` and the sigmoid's own local derivative (`output · (1-output)` from Chapter 16.5) — a tensor of exactly the right shape (since `MatMulOp::backward`'s output shapes only depend on `x` and `W`'s shapes, which are unchanged), just numerically wrong by a large, silent margin. A shape check can't catch this because the bug is entirely about *which formula* ran, not about the shapes those formulas produce — `grad_output @ Bᵀ` and `Aᵀ @ grad_output` have exactly the shapes of `x` and `W` respectively, whether or not a sigmoid or a bias-add happened in between.

**4.** Three separate compiled function bodies — one each for `N=4`, `N=8`, and `N=16` — because each distinct template parameter value produces its own independently-generated machine code (confirmed in this chapter's own Worked Example 19.3.1 by reading two such symbols directly out of a linked binary), not one shared body with a runtime branch on `N`. The resource cost being paid is compile time (three function bodies to generate and optimize instead of one) and binary size (three compiled bodies shipped in the final program instead of one) — the runtime cost of choosing between them is what's eliminated, not cost altogether.

**5.** Without the first `cudaDeviceSynchronize()`, warmup work still queued on the device can bleed into the timed region, and without the second, the host's clock can stop before the device has actually finished the last of the `benchmark_runs` launches — in both cases, the timer is measuring how long the *host* took to issue `benchmark_runs` kernel launches, not how long the *device* took to execute them. Since launching work asynchronously is typically far faster than actually running it, the reported time would be biased too fast — understating the kernel's true execution time, potentially by a large margin if the device queue backs up well past when the host's loop finishes.

**6.** A `256`-bit load moving two `8`-byte values would compile to a wider SASS instruction than `LDG.E.128` — CUDA's SASS also has `.256`-suffixed wide loads/stores on architectures that support them (the general pattern is `LDG.E.<bitwidth>`, so `LDG.E.256`/`STG.E.256`). With `int32_t` (`4` bytes) as the element type instead, the same fixed `256`-bit (`32`-byte) transfer would move `32 / 4 = 8` elements per instruction — exactly the same reasoning Worked Example 19.1.3 applied to `128`-bit loads (`16` bytes moving `4` floats or `2` doubles): the instruction's bit-width is fixed by the hardware, and the element count it produces is always `(bit-width in bytes) / sizeof(element type)`.
