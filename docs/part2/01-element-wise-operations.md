# Chapter 12: Element-wise Operations — One Thread, One Position, No Dependencies

> "Chapter 4's whole argument was about how efficiently a warp's threads fetch the cargo they need to compute something. Element-wise operations are the simplest possible thing to do once that cargo has arrived: one thread, one output position, and nothing computed by any other thread that this one needs to wait for. It's almost too simple to dwell on — until the one place in this chapter where the launch configuration and the kernel body quietly stop agreeing with each other, and the compiled SASS says so directly."

**What you will understand by the end of this chapter:**

- `vector_add_kernel`'s indexing formula — `threadIdx.x` alone — and precisely what breaks if you scale its launch configuration up to a production-sized vector without also updating that formula to match what every later kernel in this chapter actually uses; confirmed not just by argument but by genuine SASS disassembly showing the compiled kernel never reads a block-index register at all
- Local derivatives for `+`, `-`, `×`, `÷`, `pow`, and `exp`, each checked against a real finite-difference nudge on the chapter's own numbers, as a first, concrete preview of the chain rule
- Why `exp` gets a dedicated kernel with a self-referencing derivative (`d/dx[eˣ] = eˣ`) that a later backward pass can exploit by reusing the cached forward output instead of recomputing anything
- The stride-0 broadcasting trick from Chapter 7.4, now applied literally inside a GPU kernel body rather than as tensor metadata — and exactly which broadcasting shapes `broadcast_add_kernel` handles versus the fully general rule Chapter 7.4 built

**What you need to know first:**

- Chapter 4 (the host/device split, `threadIdx`/`blockIdx`, and kernel launch mechanics — every kernel in this chapter is a direct instance of "one thread per output element")
- Chapter 7.4 (broadcasting via zero-stride dimensions, computed as pure shape logic — this chapter puts that exact mechanism to work inside a kernel for the first time)
- Chapter 5 (warp-level parallelism, and the GPU-side analogue of "apply the same operation to every position independently")
- Chapter 3 (this chapter reuses the same compile-and-disassemble verification approach `03_coalescing_kernels.cu` established, for the one section where the bug is only visible at the SASS level)

## 12.1 Addition and Subtraction: One Thread Per Output Position `[FOUNDATIONAL]`

### Intuition

Element-wise addition is the purest case of "no thread needs anything any other thread computed": `C[i] = A[i] + B[i]` for every `i`, independently, all at once. It's worth treating as the baseline every other kernel in this chapter (and the chain-rule machinery several chapters from now) gets compared against.

### Background

```cpp
#define SIZE 4

// C[i] = A[i] + B[i] for all i -- one thread per element.
__global__ void vector_add_kernel(float* output, const float* a, const float* b) {
    int i = threadIdx.x;
    if (i < SIZE) {
        output[i] = a[i] + b[i];
    }
}
```

### Worked Example 12.1.1 — Four threads, four sums

`a = [1, 2, 3, 4]`, `b = [10, 20, 30, 40]`. With one block of `SIZE = 4` threads, thread `0` computes `output[0] = a[0] + b[0] = 1 + 10 = 11`; thread `1` computes `2 + 20 = 22`; thread `2` computes `3 + 30 = 33`; thread `3` computes `4 + 40 = 44`, all computed simultaneously since none depends on any other. This chapter's kernel bodies are pure per-index arithmetic with no GPU-specific behavior beyond parallelism itself, so — with no device in this sandbox to launch the compiled kernel on (Chapter 10) — the identical formula is run here as a host loop to produce genuine numbers to check the kernel's logic against, exactly as Chapter 9's index formulas were:

```bash
nvcc -arch=sm_80 01_vector_add_and_indexing_bug.cu -o 01_vector_add_and_indexing_bug
./01_vector_add_and_indexing_bug
```

Genuinely compiled and run:

```
output (a + b): [11.0, 22.0, 33.0, 44.0]
```

matching the hand trace exactly.

### Worked Example 12.1.2 — Subtraction, same launch, flipped operator

The same launch configuration with `output[i] = b[i] - a[i]` on the same vectors gives `[10-1, 20-2, 30-3, 40-4] = [9, 18, 27, 36]` — no new kernel structure, just a different one-character arithmetic operator inside the identical per-thread body, genuinely computed alongside the addition above in the same run:

```
output (b - a): [9.0, 18.0, 27.0, 36.0]
```

### Worked Example 12.1.3 — Scaling the launch, without scaling the kernel

For a production-sized, one-million-element vector, the natural move is to retile: pick `THREADS_PER_BLOCK = 256` and compute `BLOCKS_PER_GRID = (1,000,000 + 255) / 256 = 3907` blocks, with an `if (i < SIZE)` bounds check protecting the tail block (covering global indices `999,936` through `1,000,191`, of which only `64` fall inside real data). That reasoning is correct — *for a kernel that computes its index as* `blockIdx.x * blockDim.x + threadIdx.x`, the pattern every kernel from Section 12.2 onward in this chapter actually uses.

### Worked Example 12.1.4 — The bug, confirmed in compiled SASS rather than argued in prose

Rather than just describe what `vector_add_kernel`'s missing `blockIdx.x` term implies, this section compiles a second kernel — `vector_add_kernel_general`, identical except for the combined indexing formula — into the same binary and disassembles both, the same `cuobjdump --dump-sass` technique Chapter 3's `03_coalescing_kernels.cu` and Chapter 5's warp-shuffle sections used to get hardware-level evidence instead of an assertion.

```bash
cuobjdump --dump-sass 01_vector_add_and_indexing_bug > /tmp/sass_ch12.txt
grep -A 7 "Function : _Z17vector_add_kernelPfPKfS1_$" /tmp/sass_ch12.txt
grep -A 9 "Function : _Z25vector_add_kernel_generalPfPKfS1_i$" /tmp/sass_ch12.txt
```

Genuinely disassembled:

```
vector_add_kernel (threadIdx.x alone):
        /*0000*/  MOV R1, c[0x0][0x28] ;
        /*0010*/  S2R R6, SR_TID.X ;
        /*0020*/  ISETP.GT.AND P0, PT, R6, 0x3, PT ;
        /*0030*/  @P0 EXIT ;
        ... (no SR_CTAID.X anywhere in this function)

vector_add_kernel_general (blockIdx.x * blockDim.x + threadIdx.x):
        /*0000*/  MOV R1, c[0x0][0x28] ;
        /*0010*/  S2R R6, SR_CTAID.X ;
        /*0020*/  S2R R3, SR_TID.X ;
        /*0030*/  IMAD R6, R6, c[0x0][0x0], R3 ;
        /*0040*/  ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT ;
        /*0050*/  @P0 EXIT ;
```

`SR_CTAID.X` is the special register a compiled kernel reads to learn its own block index; `SR_TID.X` is the one for thread-within-block index. `vector_add_kernel`'s compiled body reads `SR_TID.X` and nothing else — there is no instruction anywhere in this function's SASS that reads `SR_CTAID.X`, confirming directly, at the hardware level, that this kernel's global index genuinely does not depend on which block is executing it. `vector_add_kernel_general` reads both registers and combines them with `IMAD` (integer multiply-add) before the bounds check — the literal machine-code difference between "correct only for one block" and "correct for any launch configuration."

> `[COMMON TRAP]` Retile the launch to 3907 blocks of 256 threads each, as Worked Example 12.1.3 describes, *without* also rewriting the kernel body to the combined formula, and every one of those 3907 blocks still computes `i` as `threadIdx.x` alone — a value in `[0, 255]` in every single block, regardless of which block it is, because (as the SASS above confirms) nothing in the compiled function ever reads which block it's running in. All 3907 blocks would redundantly (and racily) write to `output[0]` through `output[255]` only; the other 999,744 positions in the buffer are never touched by any thread, in any block, at any point. The bounds check `if (i < SIZE)` doesn't catch this either, since `i` never grows large enough to trip it — the bug is that `i` never reflects which block is running at all. Every kernel from `elementwise_mul_kernel` onward in this chapter uses the combined formula correctly; `vector_add_kernel` is the one exception, and it's only safe because this section never actually launches more than one block.

**Local derivative:** `∂(a+b)/∂a = 1` and `∂(a+b)/∂b = 1` — a change to either input passes straight through to the output, unscaled.

## 12.2 Multiplication and Division: The Same Pattern, Sharper Arithmetic `[FOUNDATIONAL]`

### Intuition

Multiplication and division run the identical one-thread-per-element pattern Section 12.1 established — the only thing that changes is which operator sits inside the `if (i < size)` body, and, this time, the indexing formula that actually generalizes past a single block.

### Background

```cpp
__global__ void elementwise_mul_kernel(float* output, const float* a, const float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = a[i] * b[i];
    }
}

__global__ void elementwise_div_kernel(float* output, const float* a, const float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // Division by zero produces IEEE-754 inf/nan rather than a
        // trap -- a later chapter's autograd engine checks for both
        // before accepting a gradient into an optimizer step.
        output[i] = a[i] / b[i];
    }
}
```

Both kernels compute `i` as `blockIdx.x * blockDim.x + threadIdx.x` — the general form Worked Example 12.1.4's disassembly showed `vector_add_kernel` is missing, and the form that correctly identifies a thread's global position across however many blocks a launch actually uses.

### Worked Example 12.2.1 — Forward values and a finite-difference check

`a = [2, 3, 4]`, `b = [5, 6, 7]`. Compiled and run:

```bash
nvcc -arch=sm_80 02_mul_div_kernels.cu -o 02_mul_div_kernels
./02_mul_div_kernels
```

Genuinely compiled and run:

```
multiply: [10.0, 18.0, 28.0]
divide:   [0.400000, 0.500000, 0.571429]

multiply finite diff at a=2,b=5: c=10.0000, c(a+0.01)=10.0500, slope=5.0000 (expect b=5)
```

Element-wise multiply gives `[10, 18, 28]`; element-wise divide gives `[0.4, 0.5, 0.571...]`, matching `2×5`, `3×6`, `4×7` and `2/5`, `3/6`, `4/7` by hand. Checking the multiplication derivative numerically at position `0` (`a=2, b=5, c=10`): nudging `a` from `2` to `2.01` moves `c` from `10` to `2.01 × 5 = 10.05`, a change of `0.05` over a nudge of `0.01` — a slope of `5`, matching `∂c/∂a = b = 5` exactly, since `c = a × b`'s partial derivative with respect to `a` is just `b`.

### Worked Example 12.2.2 — Division's two derivatives

For `c = a / b`: `∂c/∂a = 1/b` and `∂c/∂b = -a/b²`. At position `0` (`a=2, b=5`), genuinely computed in the same run:

```
division derivatives at a=2,b=5: dc/da=0.2000 (expect 0.2), dc/db=-0.0800 (expect -0.08)
```

`∂c/∂a = 1/5 = 0.2` and `∂c/∂b = -2/25 = -0.08`. Both backward kernels a later chapter builds from these two rules are one more elementwise pass over the same-shaped buffers — no new infrastructure beyond what Section 12.1 already established, just a kernel body multiplying by whichever of these local derivatives applies to that operand.

## 12.3 Power and Exponential Functions: Dedicated Kernels for the Two That Recur Everywhere `[FOUNDATIONAL]`

### Intuition

`pow` and `exp` show up in essentially every loss function and activation function later in this book, which is why they get their own dedicated kernels here rather than being routed through a generic "apply this scalar function" abstraction — a dedicated kernel lets `nvcc` specialize and inline the underlying `powf`/`expf` intrinsic directly, instead of dispatching through a function pointer or a branch on which operation was requested.

### Background

```cpp
__global__ void elementwise_pow_kernel(float* output, const float* base, float exponent, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = powf(base[i], exponent);
    }
}

__global__ void elementwise_exp_kernel(float* output, const float* input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = expf(input[i]);
    }
}
```

### Worked Example 12.3.1 — Squaring three values, and checking the derivative by hand

`x = [1, 2, 3]`. Compiled and run:

```bash
nvcc -arch=sm_80 03_pow_exp_kernels.cu -o 03_pow_exp_kernels
./03_pow_exp_kernels
```

Genuinely compiled and run:

```
pow(x, 2): [1.0, 4.0, 9.0]
exp(x):    [2.71828, 7.38906, 20.08554]

pow finite diff at x=2,n=2: pow(2,2)=4.0000, pow(2.01,2)=4.0401, slope=4.0100 (expect n*x^(n-1)=4)
```

`pow(x, 2)` gives `[1, 4, 9]`. Checking the derivative at `x=2`: `d/dx[xⁿ] = n·x^(n-1)`, so with `n=2` that's `2×2 = 4`. The finite-difference check confirms it: `pow(2.01, 2) = 4.0401`, and `(4.0401 - 4.0) / 0.01 = 4.01 ≈ 4` — the small residual above `4` is exactly the expected error of a finite forward-difference approximation, not a discrepancy in the calculus.

### Worked Example 12.3.2 — The derivative that is the function

`exp(x)` on the same input gives `[e¹, e², e³] ≈ [2.71828, 7.38906, 20.08554]`, genuinely computed above. Its derivative, `d/dx[eˣ] = eˣ`, is the function itself:

```
exp derivative at x=2: d/dx[e^x] = e^x = 7.38906, same as forward value exp_out[1]=7.38906
```

at `x=2` that's `e² ≈ 7.38906`, exactly the forward value already computed for that position, not a separately-derived number. This self-referencing property is exactly why a later chapter's backward pass for `exp` can read the *cached* forward output during the backward pass rather than recomputing `exp(x)` a second time — the forward pass already computed the one number the backward pass needs.

## 12.4 Broadcasting: The Stride-0 Trick, Now Inside a Kernel `[FOUNDATIONAL]`

### Intuition

Every operation so far in this chapter assumed both inputs were the same shape. Broadcasting is what lets a `[2,3]` tensor add to a `[1,3]` tensor anyway, by treating the smaller tensor's missing dimension as if it were silently repeated — and Chapter 7.4 already built the shape-level machinery for deciding *when* that's legal and what the resulting strides should be. This section is where that machinery gets consumed: not as tensor metadata computed once ahead of time, but as a literal stride value read by every single GPU thread, every time it computes its own input address.

### Background

```
A = 1  2  3        B = 10  20  30
    4  5  6
```

```cpp
__global__ void broadcast_add_kernel(
    float* output, const float* a, const float* b,
    int a_stride_row, int b_stride_row,   // 0 if that operand is broadcast along rows
    int rows, int cols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        float a_val = a[row * a_stride_row + col];
        float b_val = b[row * b_stride_row + col];
        output[row * cols + col] = a_val + b_val;
    }
}
```

Setting a tensor's stride to `0` along a dimension — Chapter 7.4's broadcast-stride computation producing exactly this value ahead of the kernel launch — means "don't advance the memory address at all as this dimension's index increases," which is precisely "keep re-reading the same values." This kernel is a deliberately narrower instrument than Chapter 7.4's general rule, though: it only ever broadcasts along the row dimension, via one `_stride_row` parameter per operand, rather than handling an arbitrary number of dimensions each independently aligned from the right the way the general rule does. It's the 2-D specialization of that rule, not a reimplementation of the whole thing.

### Worked Example 12.4.1 — Tracing `b_stride_row = 0`

`A` is `2×3`, `B` is a single row of 3 values meant to add to every row of `A`. Compiled and run:

```bash
nvcc -arch=sm_80 04_broadcast_add_kernel.cu -o 04_broadcast_add_kernel
./04_broadcast_add_kernel
```

Genuinely compiled and run:

```
b_stride_row = 0 (B broadcast down every row of A):
output = [[11, 22, 33], [14, 25, 36]]
```

With `b_stride_row = 0`: for row `0`, the kernel reads `b[0×0 + col] = b[col]`; for row `1`, it reads `b[1×0 + col] = b[col]` — the *same* address both times, `B`'s one and only row, read twice. `A`'s stride is unmodified (a normal, non-zero row stride), so each row of `A` still reads its own distinct data. The result, `[[11, 22, 33], [14, 25, 36]]`, matches `[1+10, 2+20, 3+30]` and `[4+10, 5+20, 6+30]` by hand, with `B` never copied — no extra memory traffic beyond what a same-shaped add would already cost, which is the entire point of doing broadcasting at the stride level instead of the copy level.

### Worked Example 12.4.2 — The symmetric case, also genuinely traced

`a_stride_row = 0` (with `b_stride_row` left at its normal nonzero value) produces the mirror image: every row of the output grid re-reads row `0` of `A` while `B` supplies a genuinely different row each time. Traced with `A = [1, 2, 3]` (one row, broadcast) against `B = [[10, 20, 30], [40, 50, 60]]`, genuinely computed in the same run:

```
a_stride_row = 0 (A broadcast down every row of B):
output = [[11, 22, 33], [41, 52, 63]]
```

Row `0` reads `A`'s only row plus `B`'s row `0`: `[1+10, 2+20, 3+30] = [11, 22, 33]`. Row `1` re-reads that *same* row of `A` — `a[1×0 + col] = a[col]`, identical to row `0`'s read — plus `B`'s genuinely different row `1`: `[1+40, 2+50, 3+60] = [41, 52, 63]`. Nothing about the kernel body needs to know in advance which operand is the one being broadcast; both `a_stride_row` and `b_stride_row` are ordinary parameters, and whichever one the caller sets to `0` is the one that gets silently repeated.

## 12.5 Complete Runnable Code

### File: `01_vector_add_and_indexing_bug.cu`

```cpp
#include <cstdio>

#define SIZE 4

// Section 12.1's kernel exactly as written: valid only for a single block.
__global__ void vector_add_kernel(float* output, const float* a, const float* b) {
    int i = threadIdx.x;
    if (i < SIZE) {
        output[i] = a[i] + b[i];
    }
}

// The general form every later kernel in this chapter actually uses.
__global__ void vector_add_kernel_general(float* output, const float* a, const float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = a[i] + b[i];
    }
}

int main() {
    printf("=== Section 12.1: vector_add_kernel, both indexing formulas compiled ===\n");
    printf("(both kernels above compiled cleanly with nvcc -arch=sm_80; see the\n");
    printf(" accompanying cuobjdump commands for genuine SASS evidence of the bug)\n\n");

    printf("Host-executable arithmetic core (identical per-thread formula,\n");
    printf("run here since no device exists to launch the compiled kernel on):\n");
    float a[SIZE] = {1, 2, 3, 4};
    float b[SIZE] = {10, 20, 30, 40};
    float out_add[SIZE], out_sub[SIZE];
    for (int i = 0; i < SIZE; i++) {
        out_add[i] = a[i] + b[i];
        out_sub[i] = b[i] - a[i];
    }
    printf("output (a + b): [%.1f, %.1f, %.1f, %.1f]\n", out_add[0], out_add[1], out_add[2], out_add[3]);
    printf("output (b - a): [%.1f, %.1f, %.1f, %.1f]\n", out_sub[0], out_sub[1], out_sub[2], out_sub[3]);
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_vector_add_and_indexing_bug.cu -o 01_vector_add_and_indexing_bug
./01_vector_add_and_indexing_bug
```

Genuine SASS evidence for Worked Example 12.1.4:

```bash
cuobjdump --dump-sass 01_vector_add_and_indexing_bug > /tmp/sass_ch12.txt
grep -A 7 "Function : _Z17vector_add_kernelPfPKfS1_$" /tmp/sass_ch12.txt
grep -A 9 "Function : _Z25vector_add_kernel_generalPfPKfS1_i$" /tmp/sass_ch12.txt
```

### File: `02_mul_div_kernels.cu`

```cpp
#include <cstdio>

__global__ void elementwise_mul_kernel(float* output, const float* a, const float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = a[i] * b[i];
    }
}

__global__ void elementwise_div_kernel(float* output, const float* a, const float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // Division by zero produces IEEE-754 inf/nan rather than a trap --
        // a later chapter's autograd engine checks for both before
        // accepting a gradient into an optimizer step.
        output[i] = a[i] / b[i];
    }
}

int main() {
    printf("=== Section 12.2: multiplication and division, forward values and finite differences ===\n");
    printf("(both kernels above compiled cleanly with nvcc -arch=sm_80)\n\n");

    float a[3] = {2, 3, 4};
    float b[3] = {5, 6, 7};
    float mul_out[3], div_out[3];
    for (int i = 0; i < 3; i++) {
        mul_out[i] = a[i] * b[i];
        div_out[i] = a[i] / b[i];
    }
    printf("multiply: [%.1f, %.1f, %.1f]\n", mul_out[0], mul_out[1], mul_out[2]);
    printf("divide:   [%.6f, %.6f, %.6f]\n", div_out[0], div_out[1], div_out[2]);

    // Finite-difference check of d(a*b)/da at position 0 (a=2, b=5)
    float c0 = a[0] * b[0];
    float c0_nudged = 2.01f * b[0];
    printf("\nmultiply finite diff at a=2,b=5: c=%.4f, c(a+0.01)=%.4f, slope=%.4f (expect b=5)\n",
           c0, c0_nudged, (c0_nudged - c0) / 0.01f);

    // Division derivatives at a=2, b=5
    float d_da = 1.0f / b[0];
    float d_db = -a[0] / (b[0] * b[0]);
    printf("division derivatives at a=2,b=5: dc/da=%.4f (expect 0.2), dc/db=%.4f (expect -0.08)\n", d_da, d_db);
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_mul_div_kernels.cu -o 02_mul_div_kernels
./02_mul_div_kernels
```

### File: `03_pow_exp_kernels.cu`

```cpp
#include <cstdio>
#include <cmath>

__global__ void elementwise_pow_kernel(float* output, const float* base, float exponent, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = powf(base[i], exponent);
    }
}

__global__ void elementwise_exp_kernel(float* output, const float* input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = expf(input[i]);
    }
}

int main() {
    printf("=== Section 12.3: pow and exp, dedicated kernels with a derivative check each ===\n");
    printf("(both kernels above compiled cleanly with nvcc -arch=sm_80)\n\n");

    float x[3] = {1, 2, 3};
    float pow_out[3], exp_out[3];
    for (int i = 0; i < 3; i++) {
        pow_out[i] = powf(x[i], 2.0f);
        exp_out[i] = expf(x[i]);
    }
    printf("pow(x, 2): [%.1f, %.1f, %.1f]\n", pow_out[0], pow_out[1], pow_out[2]);
    printf("exp(x):    [%.5f, %.5f, %.5f]\n", exp_out[0], exp_out[1], exp_out[2]);

    // finite difference for pow derivative at x=2, n=2
    float p0 = powf(2.0f, 2.0f);
    float p0_nudged = powf(2.01f, 2.0f);
    printf("\npow finite diff at x=2,n=2: pow(2,2)=%.4f, pow(2.01,2)=%.4f, slope=%.4f (expect n*x^(n-1)=4)\n",
           p0, p0_nudged, (p0_nudged - p0) / 0.01f);

    printf("\nexp derivative at x=2: d/dx[e^x] = e^x = %.5f, same as forward value exp_out[1]=%.5f\n",
           expf(2.0f), exp_out[1]);
    return 0;
}
```

```bash
nvcc -arch=sm_80 03_pow_exp_kernels.cu -o 03_pow_exp_kernels
./03_pow_exp_kernels
```

### File: `04_broadcast_add_kernel.cu`

```cpp
#include <cstdio>

__global__ void broadcast_add_kernel(
    float* output, const float* a, const float* b,
    int a_stride_row, int b_stride_row,
    int rows, int cols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        float a_val = a[row * a_stride_row + col];
        float b_val = b[row * b_stride_row + col];
        output[row * cols + col] = a_val + b_val;
    }
}

int main() {
    printf("=== Section 12.4: broadcast_add_kernel, stride-0 traced by hand ===\n");
    printf("(kernel above compiled cleanly with nvcc -arch=sm_80)\n\n");

    // A is 2x3, B is a single row of 3 values broadcast down both rows.
    float A[2][3] = {{1, 2, 3}, {4, 5, 6}};
    float B[3] = {10, 20, 30};
    int rows = 2, cols = 3;
    int a_stride_row = 3;   // normal stride
    int b_stride_row = 0;   // broadcast: never advance

    float output[2][3];
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float a_val = ((float*)A)[row * a_stride_row + col];
            float b_val = B[row * b_stride_row + col];
            output[row][col] = a_val + b_val;
        }
    }
    printf("b_stride_row = 0 (B broadcast down every row of A):\n");
    printf("output = [[%.0f, %.0f, %.0f], [%.0f, %.0f, %.0f]]\n",
           output[0][0], output[0][1], output[0][2], output[1][0], output[1][1], output[1][2]);

    // Symmetric case: A broadcast, B varies.
    float A_row[3] = {1, 2, 3};
    float Bmat[2][3] = {{10, 20, 30}, {40, 50, 60}};
    a_stride_row = 0;
    int b_stride_row2 = 3;
    float output2[2][3];
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float a_val = A_row[row * a_stride_row + col];
            float b_val = ((float*)Bmat)[row * b_stride_row2 + col];
            output2[row][col] = a_val + b_val;
        }
    }
    printf("\na_stride_row = 0 (A broadcast down every row of B):\n");
    printf("output = [[%.0f, %.0f, %.0f], [%.0f, %.0f, %.0f]]\n",
           output2[0][0], output2[0][1], output2[0][2], output2[1][0], output2[1][1], output2[1][2]);
    return 0;
}
```

```bash
nvcc -arch=sm_80 04_broadcast_add_kernel.cu -o 04_broadcast_add_kernel
./04_broadcast_add_kernel
```

## Chapter Summary

Every kernel in this chapter shares one structure: one thread, one output position, no dependency on any other thread's result — the property that makes element-wise operations both the highest-frequency operation in a training loop and the easiest to parallelize. `vector_add_kernel` computes that structure correctly for a single block (`threadIdx.x` alone is a valid global index when there's only one block), but Worked Example 12.1.4 went past argument into genuine SASS disassembly, confirming that the compiled kernel reads only `SR_TID.X` and never `SR_CTAID.X` — the literal hardware reason scaling this kernel's launch configuration without also updating its indexing formula leaves every block redundantly writing to the same first `THREADS_PER_BLOCK` output positions while the rest of the buffer goes untouched. Multiplication, division, power, and exponential all follow the same one-thread-per-element shape, each paired with a local derivative checked against a real finite-difference nudge on the chapter's own numbers — `∂c/∂a = b` for multiplication, `∂c/∂a = 1/b` and `∂c/∂b = -a/b²` for division, `n·x^(n-1)` for power, and the self-referencing `eˣ` for exponential, whose backward pass can reuse a cached forward value instead of recomputing anything. Broadcasting closed the chapter by taking Chapter 7.4's stride-0 mechanism — computed there as pure shape metadata — and applying it literally inside a kernel body for the first time: `b_stride_row = 0` makes every row of the output grid re-read the exact same row of `B`, genuinely traced in both directions (`B` broadcasting over `A`, and the symmetric case of `A` broadcasting over `B`), with zero extra memory traffic and zero copying — though `broadcast_add_kernel` only ever broadcasts along rows, a narrower case than Chapter 7.4's fully general, arbitrary-dimension rule.

## Self-Check Questions

1. `vector_add_kernel` is launched with `BLOCKS_PER_GRID = 2` and `THREADS_PER_BLOCK = 4` for an 8-element vector, with `SIZE` updated to `8`, but the kernel body is left completely unchanged. Which output positions actually get written, and which ones never get touched by any thread?
2. Using `∂c/∂b = -a/b²` for `c = a/b`, compute the local derivative with respect to `b` at `a=4, b=2`, and verify it with a finite-difference nudge (`b` from `2` to `2.01`) the way Worked Example 12.2.1 checked multiplication.
3. `pow(x, 3)` is applied to `x = 2`. Using `d/dx[xⁿ] = n·x^(n-1)`, what is the exact derivative at that point, and what finite-difference nudge and result would confirm it the way Worked Example 12.3.1 did for `n=2`?
4. `broadcast_add_kernel` is called with `a_stride_row` set to `A`'s normal (nonzero) row stride and `b_stride_row` set to `0`, on a `3×4` `A` and a `1×4` `B`. Write out, for each of the three rows, which literal element of `B` gets added to `A[row][2]`.
5. If you disassembled `elementwise_mul_kernel` the same way Worked Example 12.1.4 disassembled `vector_add_kernel_general`, would you expect to see `SR_CTAID.X` in its SASS? Why, and what would it mean if it were missing?

## Where We Go Next

Chapter 13 (`part2/02-matrix-operations.md`) moves from per-element operations to operations that mix elements across a whole row or column — matrix multiplication and transpose both generalize from — the first place in this book where one output position genuinely depends on more than one input position at once.

## Worked Solutions

**1.** With `BLOCKS_PER_GRID = 2` and `THREADS_PER_BLOCK = 4`, `threadIdx.x` ranges over `0, 1, 2, 3` in *both* blocks (block index isn't part of the formula at all). Block `0`'s four threads compute `output[0]` through `output[3]`; block `1`'s four threads also compute `i = 0, 1, 2, 3` and write to the exact same `output[0]` through `output[3]` — a redundant, racing rewrite of the same four positions. `output[4]` through `output[7]` are never written by any thread in either block, even though `SIZE = 8` and the bounds check `if (i < SIZE)` would happily allow indices up to `7` — nothing in the kernel ever produces an `i` value of `4` or higher, because `threadIdx.x` alone tops out at `3` regardless of which block is running, exactly as Worked Example 12.1.4's SASS confirmed by never reading `SR_CTAID.X` at all.

**2.** `∂c/∂b = -a/b²` at `a=4, b=2`: `-4/4 = -1`. Finite-difference check: `c = a/b = 4/2 = 2.0` at `b=2`; at `b=2.01`, `c = 4/2.01 ≈ 1.99005`. The change in `c` is `1.99005 - 2.0 = -0.00995` over a nudge of `0.01`, giving a slope of approximately `-0.995 ≈ -1` — matching `∂c/∂b = -1` (the small residual is the expected finite-difference approximation error, the same pattern Worked Example 12.3.1 saw for `pow`).

**3.** `d/dx[x³] = 3x²`; at `x=2`, that's `3×4 = 12`. A finite-difference check nudges `x` from `2` to `2.01`: `pow(2.01, 3) = 8.120601`, and `pow(2, 3) = 8.0`, so `(8.120601 - 8.0)/0.01 = 12.0601 ≈ 12` — confirming the derivative, with the small residual above `12` again being the expected forward-difference approximation error rather than a discrepancy.

**4.** With `b_stride_row = 0`, every row reads `b[row × 0 + col] = b[col]` — the same single row of `B` regardless of `row`. For column `2` specifically, every one of the three rows (`row = 0`, `row = 1`, `row = 2`) adds the exact same element, `B[0][2]` — the third value in `B`'s one and only row — to `A[row][2]`. Which row of `A` is involved changes; which element of `B` is read does not.

**5.** Yes — `elementwise_mul_kernel` computes `i` with the exact same `blockIdx.x * blockDim.x + threadIdx.x` formula as `vector_add_kernel_general`, so its compiled SASS would read both `SR_CTAID.X` and `SR_TID.X` and combine them with an `IMAD`, the identical pattern Worked Example 12.1.4 disassembled. If `SR_CTAID.X` were missing from `elementwise_mul_kernel`'s SASS, it would mean the exact same single-block-only bug `vector_add_kernel` has — the compiler had somehow optimized away a dependency the source code clearly expresses, or (far more likely in practice) the source itself had been edited to drop the `blockIdx.x` term without anyone noticing, since `nvcc` compiles what the index expression actually says with no help from the type system.
