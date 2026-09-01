# Appendix H: Tensor Contractions, From First Principles (CPU)

> "Every tensor contraction you will ever write is either a matrix multiply in disguise, or a small collection of matrix multiplies stapled together. The disguise is worth removing once, carefully, so that every later appearance is instantly recognizable."

## What you will understand

By the end of this appendix you will be able to:

- State the general definition of a tensor contraction, and recognize matrix multiplication as its simplest non-trivial case.
- Implement a fully general N-dimensional contraction in C++, from a shape/stride representation up through a working `contract()` function.
- Contract two tensors over more than one shared axis at once, and independently verify the result against `numpy.tensordot`.
- Recognize and guard against the specific silent-corruption trap that an unvalidated axis pair opens up.
- Explain, with a genuine wall-clock measurement rather than an assertion, why loop *order* changes a contraction's running time without changing its answer.

## What you need to know first

This appendix assumes you are comfortable with the struct and memory-layout material from Chapters 2-3, and with the recursive-lambda and generic-function style used throughout Appendix F and Appendix D. Nothing here requires a GPU -- every example in this appendix is ordinary, portable C++ compiled with `g++` and run on the CPU. The companion appendix, Appendix I, takes the exact same mathematics onto CUDA.

## H.1 What Is a Tensor Contraction?

A **tensor** here is nothing more exotic than a struct earlier chapters have already used many times: a flat buffer of numbers, plus a **shape** describing how many axes it has and how large each one is. A vector is a rank-1 tensor (one axis); a matrix is a rank-2 tensor (two axes); a rank-3 tensor is a "stack" of matrices, and so on.

A **contraction** takes two tensors, picks out one or more axes on *each* that have matching sizes, and produces a new tensor by summing the elementwise product of the two inputs over every combination of values those matched axes can take -- while keeping every other axis untouched. The matched axes are called **contracted axes** (or, in the physics convention this appendix borrows a term from, *dummy indices*, because their specific value never appears in the output -- only every value they range over, summed away). The untouched axes are the **free indices**, and they simply concatenate to form the output's shape.

Matrix multiplication is the contraction everyone already knows, they just haven't always seen it named that way. Given `A` with shape `[M, K]` and `B` with shape `[K, N]`:

```
C[i][j] = Σ_k  A[i][k] · B[k][j]
```

`i` and `j` are free indices (they survive into the output, shape `[M, N]`); `k` is the contracted index (it's summed away). Nothing about this definition requires `A` and `B` to be rank-2. If `A` has shape `[2, 3, 4]` and `B` has shape `[3, 4, 5]`, contracting over *both* of `A`'s last two axes against *both* of `B`'s first two axes is exactly as well-defined -- it just sums over a pair of indices instead of one, and the output shape is `[2, 5]` (A's one remaining axis, then B's one remaining axis). Section H.4 builds and verifies exactly this case.

### Formulas and Key Terms

```
General contraction, axis sets S_A (on A) and S_B (on B), |S_A| = |S_B|:

  C[free_A indices, free_B indices]  =  Σ (over all values of the S_A/S_B axes)
                                            A[free_A indices, S_A indices]
                                          · B[S_B indices, free_B indices]

Matrix multiply, the rank-2, single-shared-axis case:

  C[i][j] = Σ_k  A[i][k] · B[k][j]        (A: [M,K], B: [K,N], C: [M,N])
```

- **Rank** — the number of axes a tensor has; a scalar is rank 0, a vector rank 1, a matrix rank 2.
- **Shape** — the size of each axis, e.g. `[2, 3, 4]` for a rank-3 tensor whose axes hold 2, 3, and 4 values respectively.
- **Free index** — an axis that is *not* contracted; it survives into the output tensor's shape unchanged.
- **Contracted index** ("dummy index") — an axis that *is* contracted; it is summed over and does not appear in the output shape at all.
- **Einstein summation convention** — the notational shortcut this appendix's formulas rely on informally: any index letter repeated between two tensors being multiplied together is implicitly summed over. `C[i][j] = A[i][k]·B[k][j]` already means "sum over k" the moment `k` appears on both sides of the product.
- **Output rank** — for a contraction over `p` shared axis pairs, `rank(C) = rank(A) + rank(B) - 2p`. Matrix multiply: `2 + 2 - 2(1) = 2`, a matrix, as expected.

## H.2 Shapes, Strides, and Indexing

Before any contraction logic can be written, there is one piece of bookkeeping every later section depends on: converting between a **multi-index** (one integer per axis) and the single flat integer offset into a tensor's underlying buffer.

This appendix uses **row-major** layout throughout (the last axis varies fastest in memory -- the same convention C and C++ arrays already use, and the one this book has used since Chapter 2). The **stride** of an axis is how many flat positions a single step along that axis is worth; in row-major order, the stride of the *last* axis is always 1, and each stride moving left is the product of everything to its right.

### Worked Example H.2.1 — Round-tripping every index of a small tensor

`01_tensor_indexing.cpp` computes strides for a `[2, 3, 4]` tensor, unravels one specific linear offset by hand-checked arithmetic, and then exhaustively round-trips *every* one of its 24 offsets through unravel-then-ravel to confirm none of them get corrupted:

```
shape = [2, 3, 4]
row-major strides = [12, 4, 1]
total elements = 24

unravel_index(17) = (1, 1, 1)  [hand-computed expected: (1, 1, 1)]
ravel_index(1,1,1) = 17  [must round-trip back to 17]

exhaustive round-trip check over all 24 offsets: 0 failures

All checks passed.
```

Genuinely compiled with `g++ -std=c++17 -Wall -Wextra -fsanitize=address -g` and run; the hand-worked check on linear offset 17 (`17 / 12 = 1` remainder `5`, `5 / 4 = 1` remainder `1`, `1 / 1 = 1` remainder `0`, giving multi-index `(1, 1, 1)`) matches the code's own output exactly, and the exhaustive loop confirms the same holds for every other offset, not just the one checked by hand.

## H.3 A Generic Contraction Function

With indexing settled, `contract()` in `02_generic_contraction.cpp` implements the general definition from Section H.1 directly: given two tensors and a list of axes to contract on each, it works out which axes are free, builds the output shape, and fills it in by walking every combination of free indices, and for each one, summing over every combination of contracted indices.

The walk itself uses a small recursive helper, `for_each_index()`, which visits every multi-index of a given shape in row-major order -- one level of recursion per axis. This is a different (and, for a teaching example, more directly readable) technique than the iterative divisor/modulo unraveling of Section H.2; both compute the same thing, and Appendix I.3 deliberately returns to the *iterative* style, because a GPU kernel cannot recurse over an unknown number of axes the way host C++ can.

### Worked Example H.3.1 — Matrix multiply, recovered as `contract(A, {1}, B, {0})`

```
=== Matrix multiply as a rank-2 contraction over ONE shared axis ===

A [3 x 2]:
     1.00    2.00 
     3.00    4.00 
     5.00    6.00 
B [2 x 4]:
     1.00    0.00    2.00    1.00 
     0.00    1.00    1.00    2.00 
C = contract(A, axis 1, B, axis 0) [3 x 4]:
     1.00    2.00    4.00    5.00 
     3.00    4.00   10.00   11.00 
     5.00    6.00   16.00   17.00 

hand-computed cross-check:
  C[0][0] expected 1*1 + 2*0 = 1.00   -> got 1.00
  C[2][3] expected 5*1 + 6*2 = 17.00  -> got 17.00

All checks passed. (Full cross-check against numpy in the next step.)
```

The two spot-checked entries above were worked out by hand before running the program, not after. A separate, independent check goes further and compares *every* entry of `C` against `numpy`'s own `@` matrix-multiply operator, computed by a completely different implementation in a different language:

```python
>>> import numpy as np
>>> A = np.array([[1,2],[3,4],[5,6]], dtype=float)
>>> B = np.array([[1,0,2,1],[0,1,1,2]], dtype=float)
>>> np.allclose(A @ B, [[1,2,4,5],[3,4,10,11],[5,6,16,17]])
True
```

## H.4 Contracting Over Multiple Axes at Once

The whole point of writing `contract()` generically in Section H.3 -- rather than hard-coding "one shared axis" -- is that it already handles more than one shared axis, with no changes at all. `03_double_contraction.cpp` contracts a `[2, 3, 4]` tensor against a `[3, 4, 5]` tensor over *two* axis pairs simultaneously (A's axes `{1, 2}` against B's axes `{0, 1}`), producing a `[2, 5]` output -- exactly the case introduced at the end of Section H.1.

### Worked Example H.4.1 — A double contraction, cross-checked against `numpy.tensordot`

```
=== A double contraction: TWO shared axes at once ===

output shape: [2, 5]  (expected [2, 5])
C =
     10.00     5.00     0.00    -5.00   -10.00 
      2.00     1.00     0.00    -1.00    -2.00 
```

`numpy.tensordot` is numpy's own general-purpose contraction function, built independently of anything in this appendix, and it takes the identical axis specification (`axes=([1,2],[0,1])`):

```python
>>> import numpy as np
>>> C = np.tensordot(A, B, axes=([1,2],[0,1]))
>>> C
array([[ 10.,   5.,   0.,  -5., -10.],
       [  2.,   1.,   0.,  -1.,  -2.]])
>>> np.allclose(C, cpp_output)
True
```

Two structurally different implementations -- this appendix's recursive, hand-rolled `contract()`, and numpy's own compiled tensordot -- agree exactly on every one of the ten output values.

## H.5 [COMMON TRAP] Validating Axis Pairs

It is easy to pass the wrong axis, or an axis pair whose sizes don't actually match, and have the mistake produce a plausible-looking but *wrong* tensor instead of an error. `contract()` guards against this up front: before doing any arithmetic, it checks that every contracted axis pair has matching sizes, and throws `std::invalid_argument` if not.

`04_shape_mismatch_trap.cpp` genuinely triggers this check, rather than only describing it:

```
=== [COMMON TRAP] contracting a mismatched axis pair ===

A.shape = [3, 2], B.shape = [4, 5]
attempting contract(A, axis 1 [size 2], B, axis 0 [size 4]) ...

caught std::invalid_argument, as expected:
  "contract: mismatched dimension on contracted axis pair 0 (A.shape[1]=2 vs B.shape[0]=4)"
```

What would happen *without* this check is not merely asserted -- it was independently built and run to find out. An unchecked version of `contract()`, given the same mismatched inputs, does not crash and does not fail loudly: it silently sizes the contracted-index loop from `A`'s axis alone (size 2) and reads only the first 2 of `B`'s 4 rows, producing a fully-formed, wrong `3x5` result that is bit-for-bit identical to what you'd get from manually slicing `B` down to its first two rows and multiplying that instead:

```
output shape: [3, 5]     <- looks completely normal
  13.0   16.0   19.0   22.0   25.0     <- matches "A @ B[:2, :]" exactly
  27.0   34.0   41.0   48.0   55.0     <- i.e. HALF of B silently discarded
  41.0   52.0   63.0   74.0   85.0
```

No warning, no crash, no obviously-wrong shape -- just a confidently wrong answer built from half the intended data. This is precisely the failure mode the up-front validation in Section H.3 exists to convert into an immediate, loud exception instead.

## H.6 Performance: Loop Order and Cache Locality

The mathematical definition of a contraction says nothing about the order its loops run in. On real hardware, that order decides whether the innermost loop reads memory contiguously or with a large stride -- and `05_loop_order_performance.cpp` *measures* the difference this makes on matrix multiply, rather than asserting one loop order is faster.

Two loop orders compute the identical sum:

- **`ijk`** (the order `contract()` itself uses): for each output row `i`, for each column `j`, sum over `k`. `B[k][j]` is read with stride `N` (a whole row skipped per step) in the innermost loop.
- **`ikj`**: for each row `i`, for each `k`, sweep `j` across a full row of both `C` and `B`. Every innermost-loop access is now stride-1.

```
=== Loop order and cache locality, measured (not asserted) ===

     N       ijk (ms)       ikj (ms) ikj speedup
    64          0.236          0.194      1.22x   (sum match: rel diff 0.00e+00)
   128          2.271          2.161      1.05x   (sum match: rel diff 0.00e+00)
   256         19.910         11.278      1.77x   (sum match: rel diff 0.00e+00)
   384         79.027         37.822      2.09x   (sum match: rel diff 0.00e+00)

Both loop orders compute the identical sum at every size (confirmed above);
only the WALL-CLOCK cost differs, purely from how memory is walked.
```

**On these specific numbers:** this ran inside a shared cloud container, not dedicated hardware, so the exact millisecond values above are not portable to your own machine and should not be quoted as general facts about "matrix multiply performance." What the measurement *does* establish -- and what is worth re-running on your own hardware to confirm for yourself -- is the trend: the `ikj` speedup grows with `N`, because the working set outgrows cache faster than the stride-`N` access pattern can hide it. The `(sum match: rel diff 0.00e+00)` column on every row is the other half of the point: reordering loops for speed changed nothing about the answer.

### Formulas and Key Terms

- **Cache line** — the unit of memory a CPU actually fetches from main memory on a miss (commonly 64 bytes, or 8 `double`s); a stride-1 access pattern uses every value fetched, a large-stride pattern discards most of each line fetched.
- **Arithmetic intensity** — the ratio of floating-point operations performed to bytes of memory moved; a contraction's *total* FLOP count doesn't change with loop order, but how many bytes must actually be re-fetched from slow memory (versus reused from cache) does, which is exactly what Section H.6's timings are measuring the consequence of.
- **FLOP count** — for a contraction summing over `p` shared axes each of size `K_1..K_p`, producing an output of `T` total elements, the total multiply-add count is `T · Π K_i`; for matrix multiply (`M×K` by `K×N`), that's `M·N·K` multiply-adds, or `2MNK` FLOPs counting the multiply and add separately.

## H.7 Complete Runnable Code

### File: 01_tensor_indexing.cpp

```cpp
#include <cstdio>
#include <vector>
#include <cassert>

std::vector<long long> row_major_strides(const std::vector<int>& shape) {
    std::vector<long long> strides(shape.size());
    long long acc = 1;
    for (int i = (int)shape.size() - 1; i >= 0; i--) {
        strides[i] = acc;
        acc *= shape[i];
    }
    return strides;
}

long long ravel_index(const std::vector<int>& idx, const std::vector<long long>& strides) {
    long long lin = 0;
    for (size_t i = 0; i < idx.size(); i++) lin += (long long)idx[i] * strides[i];
    return lin;
}

std::vector<int> unravel_index(long long lin, const std::vector<int>& shape,
                                const std::vector<long long>& strides) {
    std::vector<int> idx(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        idx[i] = (int)(lin / strides[i]);
        lin %= strides[i];
    }
    return idx;
}

int main() {
    std::vector<int> shape = {2, 3, 4};
    auto strides = row_major_strides(shape);

    printf("shape = [%d, %d, %d]\n", shape[0], shape[1], shape[2]);
    printf("row-major strides = [%lld, %lld, %lld]\n", strides[0], strides[1], strides[2]);

    long long total = 1;
    for (int s : shape) total *= s;
    printf("total elements = %lld\n\n", total);

    auto idx17 = unravel_index(17, shape, strides);
    printf("unravel_index(17) = (%d, %d, %d)  [hand-computed expected: (1, 1, 1)]\n",
           idx17[0], idx17[1], idx17[2]);
    assert(idx17[0] == 1 && idx17[1] == 1 && idx17[2] == 1);

    long long back = ravel_index(idx17, strides);
    printf("ravel_index(1,1,1) = %lld  [must round-trip back to 17]\n\n", back);
    assert(back == 17);

    int failures = 0;
    for (long long lin = 0; lin < total; lin++) {
        auto idx = unravel_index(lin, shape, strides);
        long long round_trip = ravel_index(idx, strides);
        if (round_trip != lin) failures++;
    }
    printf("exhaustive round-trip check over all %lld offsets: %d failures\n",
           total, failures);
    assert(failures == 0);

    printf("\nAll checks passed.\n");
    return 0;
}
```

### File: 02_generic_contraction.cpp

```cpp
#include <cstdio>
#include <vector>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <cmath>

struct Tensor {
    std::vector<int> shape;
    std::vector<double> data;

    explicit Tensor(std::vector<int> shape_) : shape(std::move(shape_)) {
        long long total = 1;
        for (int s : shape) total *= s;
        data.assign((size_t)total, 0.0);
    }
};

std::vector<long long> row_major_strides(const std::vector<int>& shape) {
    std::vector<long long> strides(shape.size());
    long long acc = 1;
    for (int i = (int)shape.size() - 1; i >= 0; i--) {
        strides[i] = acc;
        acc *= shape[i];
    }
    return strides;
}

long long ravel_index(const std::vector<int>& idx, const std::vector<long long>& strides) {
    long long lin = 0;
    for (size_t i = 0; i < idx.size(); i++) lin += (long long)idx[i] * strides[i];
    return lin;
}

template <typename F>
void for_each_index(const std::vector<int>& shape, F&& visit) {
    std::vector<int> idx(shape.size(), 0);
    std::function<void(size_t)> recurse = [&](size_t axis) {
        if (axis == shape.size()) { visit(idx); return; }
        for (int i = 0; i < shape[axis]; i++) {
            idx[axis] = i;
            recurse(axis + 1);
        }
    };
    if (shape.empty()) { visit(idx); return; }
    recurse(0);
}

Tensor contract(const Tensor& A, std::vector<int> axes_a,
                 const Tensor& B, std::vector<int> axes_b) {
    if (axes_a.size() != axes_b.size())
        throw std::invalid_argument("contract: axes_a and axes_b must have the same length");

    for (size_t k = 0; k < axes_a.size(); k++) {
        int da = A.shape.at(axes_a[k]);
        int db = B.shape.at(axes_b[k]);
        if (da != db) {
            throw std::invalid_argument(
                "contract: mismatched dimension on contracted axis pair " + std::to_string(k) +
                " (A.shape[" + std::to_string(axes_a[k]) + "]=" + std::to_string(da) +
                " vs B.shape[" + std::to_string(axes_b[k]) + "]=" + std::to_string(db) + ")");
        }
    }

    std::vector<int> free_a, free_b;
    for (int a = 0; a < (int)A.shape.size(); a++)
        if (std::find(axes_a.begin(), axes_a.end(), a) == axes_a.end()) free_a.push_back(a);
    for (int b = 0; b < (int)B.shape.size(); b++)
        if (std::find(axes_b.begin(), axes_b.end(), b) == axes_b.end()) free_b.push_back(b);

    std::vector<int> out_shape;
    for (int a : free_a) out_shape.push_back(A.shape[a]);
    for (int b : free_b) out_shape.push_back(B.shape[b]);

    std::vector<int> contracted_shape;
    for (int a : axes_a) contracted_shape.push_back(A.shape[a]);

    Tensor C(out_shape);
    auto sA = row_major_strides(A.shape);
    auto sB = row_major_strides(B.shape);
    auto sC = row_major_strides(out_shape);

    for_each_index(out_shape, [&](const std::vector<int>& out_idx) {
        std::vector<int> idx_a(A.shape.size(), 0), idx_b(B.shape.size(), 0);
        for (size_t i = 0; i < free_a.size(); i++) idx_a[free_a[i]] = out_idx[i];
        for (size_t j = 0; j < free_b.size(); j++) idx_b[free_b[j]] = out_idx[free_a.size() + j];

        double acc = 0.0;
        for_each_index(contracted_shape, [&](const std::vector<int>& c_idx) {
            for (size_t k = 0; k < axes_a.size(); k++) {
                idx_a[axes_a[k]] = c_idx[k];
                idx_b[axes_b[k]] = c_idx[k];
            }
            acc += A.data[ravel_index(idx_a, sA)] * B.data[ravel_index(idx_b, sB)];
        });
        C.data[ravel_index(out_idx, sC)] = acc;
    });
    return C;
}

static void print_matrix(const char* label, const Tensor& M) {
    printf("%s [%d x %d]:\n", label, M.shape[0], M.shape[1]);
    for (int i = 0; i < M.shape[0]; i++) {
        printf("  ");
        for (int j = 0; j < M.shape[1]; j++) printf("%7.2f ", M.data[i * M.shape[1] + j]);
        printf("\n");
    }
}

int main() {
    printf("=== Matrix multiply as a rank-2 contraction over ONE shared axis ===\n\n");

    Tensor A({3, 2});
    Tensor B({2, 4});
    double a_vals[6] = {1, 2, 3, 4, 5, 6};
    double b_vals[8] = {1, 0, 2, 1, 0, 1, 1, 2};
    for (int i = 0; i < 6; i++) A.data[i] = a_vals[i];
    for (int i = 0; i < 8; i++) B.data[i] = b_vals[i];

    print_matrix("A", A);
    print_matrix("B", B);

    Tensor C = contract(A, {1}, B, {0});
    print_matrix("C = contract(A, axis 1, B, axis 0)", C);

    printf("\nhand-computed cross-check:\n");
    printf("  C[0][0] expected 1*1 + 2*0 = 1.00   -> got %.2f\n", C.data[0 * 4 + 0]);
    printf("  C[2][3] expected 5*1 + 6*2 = 17.00  -> got %.2f\n", C.data[2 * 4 + 3]);
    assert(std::abs(C.data[0 * 4 + 0] - 1.0) < 1e-9);
    assert(std::abs(C.data[2 * 4 + 3] - 17.0) < 1e-9);

    printf("\nAll checks passed. (Full cross-check against numpy in the next step.)\n");
    return 0;
}
```

### File: 03_double_contraction.cpp

```cpp
#include <cstdio>
#include <vector>
#include <functional>
#include <algorithm>
#include <stdexcept>

struct Tensor {
    std::vector<int> shape;
    std::vector<double> data;
    explicit Tensor(std::vector<int> shape_) : shape(std::move(shape_)) {
        long long total = 1;
        for (int s : shape) total *= s;
        data.assign((size_t)total, 0.0);
    }
};

std::vector<long long> row_major_strides(const std::vector<int>& shape) {
    std::vector<long long> strides(shape.size());
    long long acc = 1;
    for (int i = (int)shape.size() - 1; i >= 0; i--) { strides[i] = acc; acc *= shape[i]; }
    return strides;
}
long long ravel_index(const std::vector<int>& idx, const std::vector<long long>& strides) {
    long long lin = 0;
    for (size_t i = 0; i < idx.size(); i++) lin += (long long)idx[i] * strides[i];
    return lin;
}
template <typename F>
void for_each_index(const std::vector<int>& shape, F&& visit) {
    std::vector<int> idx(shape.size(), 0);
    std::function<void(size_t)> recurse = [&](size_t axis) {
        if (axis == shape.size()) { visit(idx); return; }
        for (int i = 0; i < shape[axis]; i++) { idx[axis] = i; recurse(axis + 1); }
    };
    if (shape.empty()) { visit(idx); return; }
    recurse(0);
}
Tensor contract(const Tensor& A, std::vector<int> axes_a, const Tensor& B, std::vector<int> axes_b) {
    if (axes_a.size() != axes_b.size())
        throw std::invalid_argument("contract: axes_a and axes_b must have the same length");
    for (size_t k = 0; k < axes_a.size(); k++) {
        if (A.shape.at(axes_a[k]) != B.shape.at(axes_b[k]))
            throw std::invalid_argument("contract: mismatched dimension on axis pair " + std::to_string(k));
    }
    std::vector<int> free_a, free_b;
    for (int a = 0; a < (int)A.shape.size(); a++)
        if (std::find(axes_a.begin(), axes_a.end(), a) == axes_a.end()) free_a.push_back(a);
    for (int b = 0; b < (int)B.shape.size(); b++)
        if (std::find(axes_b.begin(), axes_b.end(), b) == axes_b.end()) free_b.push_back(b);
    std::vector<int> out_shape;
    for (int a : free_a) out_shape.push_back(A.shape[a]);
    for (int b : free_b) out_shape.push_back(B.shape[b]);
    std::vector<int> contracted_shape;
    for (int a : axes_a) contracted_shape.push_back(A.shape[a]);

    Tensor C(out_shape);
    auto sA = row_major_strides(A.shape), sB = row_major_strides(B.shape), sC = row_major_strides(out_shape);
    for_each_index(out_shape, [&](const std::vector<int>& out_idx) {
        std::vector<int> idx_a(A.shape.size(), 0), idx_b(B.shape.size(), 0);
        for (size_t i = 0; i < free_a.size(); i++) idx_a[free_a[i]] = out_idx[i];
        for (size_t j = 0; j < free_b.size(); j++) idx_b[free_b[j]] = out_idx[free_a.size() + j];
        double acc = 0.0;
        for_each_index(contracted_shape, [&](const std::vector<int>& c_idx) {
            for (size_t k = 0; k < axes_a.size(); k++) { idx_a[axes_a[k]] = c_idx[k]; idx_b[axes_b[k]] = c_idx[k]; }
            acc += A.data[ravel_index(idx_a, sA)] * B.data[ravel_index(idx_b, sB)];
        });
        C.data[ravel_index(out_idx, sC)] = acc;
    });
    return C;
}

int main() {
    printf("=== A double contraction: TWO shared axes at once ===\n\n");

    Tensor A({2, 3, 4});
    Tensor B({3, 4, 5});
    for (size_t i = 0; i < A.data.size(); i++) A.data[i] = (double)(i % 7) - 3.0;
    for (size_t i = 0; i < B.data.size(); i++) B.data[i] = (double)(i % 5) - 2.0;

    Tensor C = contract(A, {1, 2}, B, {0, 1});
    printf("output shape: [%d, %d]  (expected [2, 5])\n", C.shape[0], C.shape[1]);

    printf("C =\n");
    for (int i = 0; i < C.shape[0]; i++) {
        printf("  ");
        for (int j = 0; j < C.shape[1]; j++) printf("%8.2f ", C.data[i * C.shape[1] + j]);
        printf("\n");
    }
    return 0;
}
```

### File: 04_shape_mismatch_trap.cpp

```cpp
#include <cstdio>
#include <vector>
#include <functional>
#include <algorithm>
#include <stdexcept>

struct Tensor {
    std::vector<int> shape;
    std::vector<double> data;
    explicit Tensor(std::vector<int> shape_) : shape(std::move(shape_)) {
        long long total = 1;
        for (int s : shape) total *= s;
        data.assign((size_t)total, 0.0);
    }
};
std::vector<long long> row_major_strides(const std::vector<int>& shape) {
    std::vector<long long> strides(shape.size());
    long long acc = 1;
    for (int i = (int)shape.size() - 1; i >= 0; i--) { strides[i] = acc; acc *= shape[i]; }
    return strides;
}
long long ravel_index(const std::vector<int>& idx, const std::vector<long long>& strides) {
    long long lin = 0;
    for (size_t i = 0; i < idx.size(); i++) lin += (long long)idx[i] * strides[i];
    return lin;
}
template <typename F>
void for_each_index(const std::vector<int>& shape, F&& visit) {
    std::vector<int> idx(shape.size(), 0);
    std::function<void(size_t)> recurse = [&](size_t axis) {
        if (axis == shape.size()) { visit(idx); return; }
        for (int i = 0; i < shape[axis]; i++) { idx[axis] = i; recurse(axis + 1); }
    };
    if (shape.empty()) { visit(idx); return; }
    recurse(0);
}
Tensor contract(const Tensor& A, std::vector<int> axes_a, const Tensor& B, std::vector<int> axes_b) {
    if (axes_a.size() != axes_b.size())
        throw std::invalid_argument("contract: axes_a and axes_b must have the same length");
    for (size_t k = 0; k < axes_a.size(); k++) {
        int da = A.shape.at(axes_a[k]);
        int db = B.shape.at(axes_b[k]);
        if (da != db) {
            throw std::invalid_argument(
                "contract: mismatched dimension on contracted axis pair " + std::to_string(k) +
                " (A.shape[" + std::to_string(axes_a[k]) + "]=" + std::to_string(da) +
                " vs B.shape[" + std::to_string(axes_b[k]) + "]=" + std::to_string(db) + ")");
        }
    }
    std::vector<int> free_a, free_b;
    for (int a = 0; a < (int)A.shape.size(); a++)
        if (std::find(axes_a.begin(), axes_a.end(), a) == axes_a.end()) free_a.push_back(a);
    for (int b = 0; b < (int)B.shape.size(); b++)
        if (std::find(axes_b.begin(), axes_b.end(), b) == axes_b.end()) free_b.push_back(b);
    std::vector<int> out_shape;
    for (int a : free_a) out_shape.push_back(A.shape[a]);
    for (int b : free_b) out_shape.push_back(B.shape[b]);
    std::vector<int> contracted_shape;
    for (int a : axes_a) contracted_shape.push_back(A.shape[a]);
    Tensor C(out_shape);
    auto sA = row_major_strides(A.shape), sB = row_major_strides(B.shape), sC = row_major_strides(out_shape);
    for_each_index(out_shape, [&](const std::vector<int>& out_idx) {
        std::vector<int> idx_a(A.shape.size(), 0), idx_b(B.shape.size(), 0);
        for (size_t i = 0; i < free_a.size(); i++) idx_a[free_a[i]] = out_idx[i];
        for (size_t j = 0; j < free_b.size(); j++) idx_b[free_b[j]] = out_idx[free_a.size() + j];
        double acc = 0.0;
        for_each_index(contracted_shape, [&](const std::vector<int>& c_idx) {
            for (size_t k = 0; k < axes_a.size(); k++) { idx_a[axes_a[k]] = c_idx[k]; idx_b[axes_b[k]] = c_idx[k]; }
            acc += A.data[ravel_index(idx_a, sA)] * B.data[ravel_index(idx_b, sB)];
        });
        C.data[ravel_index(out_idx, sC)] = acc;
    });
    return C;
}

int main() {
    printf("=== [COMMON TRAP] contracting a mismatched axis pair ===\n\n");

    Tensor A({3, 2});
    Tensor B({4, 5});

    printf("A.shape = [3, 2], B.shape = [4, 5]\n");
    printf("attempting contract(A, axis 1 [size 2], B, axis 0 [size 4]) ...\n\n");

    bool threw = false;
    try {
        Tensor C = contract(A, {1}, B, {0});
        printf("(unreached -- this should have thrown)\n");
        (void)C;
    } catch (const std::invalid_argument& e) {
        threw = true;
        printf("caught std::invalid_argument, as expected:\n  \"%s\"\n", e.what());
    }

    if (!threw) { printf("FAILURE: no exception was thrown\n"); return 1; }

    printf("\nWithout this check, contract() would have read B.data at offsets\n");
    printf("computed from a stride table sized for B's ACTUAL shape [4,5], while\n");
    printf("looping the contracted index only up to A's axis-1 size (2) -- silently\n");
    printf("producing a 3x5 tensor built from only the first 2 of B's 4 rows, with\n");
    printf("no error at all. Validating the axis pair up front turns a silent wrong\n");
    printf("answer into a loud, immediate one.\n");

    return 0;
}
```

### File: 05_loop_order_performance.cpp

```cpp
#include <cstdio>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>

using Clock = std::chrono::steady_clock;

static void matmul_ijk(const std::vector<double>& A, const std::vector<double>& B,
                        std::vector<double>& C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < N; k++) acc += A[i * N + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}

static void matmul_ikj(const std::vector<double>& A, const std::vector<double>& B,
                        std::vector<double>& C, int N) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a_ik = A[i * N + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a_ik * B[k * N + j];
        }
    }
}

int main() {
    printf("=== Loop order and cache locality, measured (not asserted) ===\n\n");
    printf("%6s %14s %14s %10s\n", "N", "ijk (ms)", "ikj (ms)", "ikj speedup");

    std::vector<int> sizes = {64, 128, 256, 384};
    for (int N : sizes) {
        std::vector<double> A((size_t)N * N), B((size_t)N * N), C((size_t)N * N);
        unsigned seed = 12345;
        for (auto& v : A) { seed = seed * 1103515245u + 12345u; v = (double)(seed % 1000) / 100.0; }
        for (auto& v : B) { seed = seed * 1103515245u + 12345u; v = (double)(seed % 1000) / 100.0; }

        auto t0 = Clock::now();
        matmul_ijk(A, B, C, N);
        auto t1 = Clock::now();
        double sum_ijk = 0.0;
        for (double v : C) sum_ijk += v;

        auto t2 = Clock::now();
        matmul_ikj(A, B, C, N);
        auto t3 = Clock::now();
        double sum_ikj = 0.0;
        for (double v : C) sum_ikj += v;

        double ms_ijk = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_ikj = std::chrono::duration<double, std::milli>(t3 - t2).count();

        printf("%6d %14.3f %14.3f %9.2fx", N, ms_ijk, ms_ikj, ms_ijk / ms_ikj);
        double rel_diff = std::abs(sum_ijk - sum_ikj) / std::abs(sum_ijk);
        printf("   (sum match: rel diff %.2e)\n", rel_diff);
        if (rel_diff > 1e-9) { printf("MISMATCH -- loop reorder changed the answer!\n"); return 1; }
    }

    printf("\nBoth loop orders compute the identical sum at every size (confirmed above);\n");
    printf("only the WALL-CLOCK cost differs, purely from how memory is walked.\n");
    return 0;
}
```

## Chapter Summary

A tensor contraction generalizes matrix multiplication by allowing more than one shared axis, and more than two axes total, while keeping the same core operation: sum the elementwise product of two tensors over their matched axes, keep everything else. This appendix built that generality from the ground up -- shapes and strides, a generic `contract()` function, matrix multiply and a genuine double contraction recovered as special cases of it -- and checked every numeric claim against an independent computation: hand-worked arithmetic for the smallest cases, and `numpy`'s own, differently-implemented `@` and `tensordot` for the larger ones. Two further lessons came from *running* code rather than reasoning about it: an unvalidated axis pair doesn't crash, it silently discards data, which is why `contract()` validates axis sizes before touching any memory; and loop order changes a contraction's wall-clock cost by an amount that grows with problem size, without changing a single digit of its answer. Appendix I carries this same mathematics onto the GPU, where the free-index loop that this appendix's `for_each_index()` walks in software becomes the grid of threads a CUDA kernel launches in hardware.

## Self-Check Questions

1. Given tensors of rank 3 and rank 4, contracted over exactly 2 shared axis pairs, what is the rank of the output?
2. Why does row-major layout give the *last* axis a stride of 1, rather than the first?
3. In `contract(A, {1}, B, {0})` for matrix multiply, which index (`i`, `j`, or `k`) is the contracted index, and which two are free?
4. What specifically goes wrong -- not just "it's wrong," but the actual mechanism -- when `contract()`'s axis-size validation is skipped and a mismatched pair is contracted anyway?
5. Section H.6 shows `ikj` beating `ijk` by a larger margin at `N=384` than at `N=64`. Why would the gap widen with `N` rather than stay constant?
6. `for_each_index()` in Section H.3 is recursive, with one level of recursion per axis. Why can't the exact same technique be reused, unmodified, inside a CUDA `__global__` kernel?
7. Two independent implementations agreeing (this appendix's `contract()` and `numpy.tensordot`) is stronger evidence of correctness than either one alone. What kind of bug could still slip past *both* of them agreeing?
8. If a contraction's total FLOP count doesn't change with loop order, what quantity *does* change, and why does that quantity affect wall-clock time on real hardware?

### Worked Solutions

1. `rank(C) = rank(A) + rank(B) - 2p = 3 + 4 - 2(2) = 3`.
2. Because the last axis is defined to be the one that varies fastest in memory in row-major order -- "stride 1" and "varies fastest" are the same statement. Column-major layout (used by, e.g., Fortran and classic BLAS) makes the same choice for the *first* axis instead; row-major and column-major differ only in which axis gets stride 1, not in the underlying idea.
3. `k` is contracted (it appears on both `A[i][k]` and `B[k][j]`, and is summed away); `i` and `j` are free (each appears on exactly one input, and both survive into `C[i][j]`).
4. The contracted-index loop gets its size from only ONE of the two tensors (`A`'s axis, in `contract()`'s implementation) rather than validating that both agree. With `A`'s axis smaller than `B`'s matching axis, the loop simply never reaches most of `B`'s data along that axis -- it isn't read out of bounds, it's just never read at all, silently discarding it. The output has the right *shape* and looks entirely plausible; only its *values* are wrong, computed from a truncated slice of one input. Section H.5 verified this exact mechanism by building and running an unchecked version and comparing its output to the value you'd get from deliberately slicing `B` down to match.
5. Cache capacity is fixed; the matrices' total memory footprint grows as `N²`. At `N=64` even the "bad" stride-`N` access pattern may still mostly hit cache, so there's little difference to exploit. By `N=384`, the working set no longer fits, so every stride-`N` step in the `ijk` order is increasingly likely to be a genuine cache miss, while `ikj`'s stride-1 innermost loop keeps reusing what it just fetched -- the *relative* penalty of the bad access pattern grows as the good pattern's advantage stays available and the bad pattern's cost compounds.
6. Recursion depth in `for_each_index()` equals the tensor's rank, decided at runtime from a `std::vector`'s size -- ordinary function-call recursion, which device code can do in principle, but which needs a rank known at compile time or a call stack CUDA threads don't cheaply have thousands of copies of. Section I.3 solves this the way Section H.2 already solved indexing before `contract()` existed: fixed-size arrays and iterative divmod unraveling, capped at a compile-time `MAX_RANK`, rather than recursion.
7. A bug present in the *mathematical specification itself*, shared by construction -- for example, if this appendix had mis-stated which axes to contract in the double-contraction example, both `contract()` and `numpy.tensordot` would faithfully compute what was actually asked for and agree with each other, while still answering a different question than the one intended. Independent implementations catch implementation bugs; they do not catch a shared misunderstanding of the problem, which is exactly why Section H.4's hand-picked axis pairs were chosen to match a case worth checking by inspection, not just by agreement.
8. The number of bytes actually moved between main memory and the CPU's cache (as opposed to reused from cache) changes with loop order, even though the FLOP count is fixed. Real hardware spends time waiting on memory transfers, not only on arithmetic -- so a loop order that reuses more of what it fetches, before that data is evicted, spends less wall-clock time waiting, for the identical number of multiply-adds.
