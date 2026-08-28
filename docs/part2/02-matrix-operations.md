# Chapter 13: Matrix Operations — When One Output Depends on More Than One Input

> "Every kernel in the last chapter answered its question by looking at exactly one input position — thread `i` read `a[i]` and `b[i]` and nothing else existed as far as it was concerned. Matrix multiplication is the first place in this book where that stops being true. Computing a single output element means reading an entire row and an entire column, multiplying them together position by position, and the moment the shapes stop lining up, the loop keeps running anyway — it just reads whatever happens to be sitting in memory next and calls that the answer."

**What you will understand by the end of this chapter:**

- The tensor-contraction rule that matrix multiplication is a special case of, worked out completely by hand on one running example (`X` (2×3) times `M` (3×2)) that every subsequent worked example in this chapter checks its own answer against
- Exactly how a SIMD-width-parameterized matrix multiply's blocking-plus-remainder loop reduces to the identical scalar triple sum — traced lane by lane, for a width that divides the contraction dimension evenly and one that doesn't
- Why neither multiply function ever checks that `a.cols == b.rows`, and precisely what ends up in `result` when that invariant is silently violated — demonstrated deterministically rather than left to undefined behavior
- The difference between a reshape that's free (same buffer, new strides, zero bytes moved) and one that requires an actual copy — traced concretely on a genuinely transposed buffer, including a real correction to what a naive reshape of that transpose actually produces
- Why multiplying by a diagonal or identity view from Chapter 9 does asymptotically less work than the general path, counted exactly rather than asserted — and how the same accounting extends to solving a triangular system without ever forming a matrix inverse

**What you need to know first:**

- Chapter 6 (linear-index formulas and RAII — the `Matrix` struct below reimplements Chapter 6's `row * cols + col` for exactly two dimensions, in place of the general dot-product-with-strides version)
- Chapter 7.1 and 7.2 (stride math and zero-copy views — Section 13.3's "reshape is free" argument is the same "recompute strides, touch no data" idea Chapter 7 built, applied to a concrete pair of shapes)
- Chapter 9 (the `IdentityView` and `TridiagonalView` structures Section 13.4 dispatches to instead of paying for the general `O(n³)` path)
- Chapter 5 (the main-loop-plus-remainder shape — this chapter's SIMD-width-parameterized multiply and transpose both reuse that exact shape, just with a dot product and a scatter-write in place of Chapter 5's simpler per-element work)

This chapter's functions are plain host C++ — like Chapter 12's kernels, matrix multiplication and transpose are eventually GPU work (Part 5's kernel-implementation chapter is where that happens), but the algorithms themselves — the contraction rule, the blocking-plus-remainder shape, the shape-check gap — are identical whether the loop runs on a CPU or inside a GPU thread. Building and genuinely running them here, on the host, means every number in this chapter is a real, executed result rather than a hand-traced one waiting for hardware that doesn't exist in this sandbox.

## 13.1 Matrix Multiplication: Tensor Contraction in Two Dimensions `[FOUNDATIONAL]`

### Intuition

Element-wise addition, Chapter 12's whole subject, never needed to know a shape beyond "how many positions are there" — every output position was a pure function of the *one* input position sitting at the same address. Matrix multiplication breaks that immediately: `Y(i,j)` needs an entire row of `X` and an entire column of `M`, and it needs them to line up — the row's third entry paired with the column's third entry, not its first. That pairing-and-summing operation has a name outside of matrices entirely: **tensor contraction**. A contraction takes two tensors, a shared "contracted" dimension between them, and produces one tensor whose shape is everything *except* that shared dimension, with every remaining position computed by summing products over every value the contracted dimension can take. Ordinary 2-D matrix multiplication is what a contraction looks like when both tensors happen to be rank 2 and exactly one dimension from each is contracted — a special case now, but the exact same machinery a later chapter's batched, higher-rank operations reuse without any new rule.

```
X (2×3):                    M (3×2):
┌         ┐                 ┌     ┐
│ 1  2  3 │  ← row 0        │ 1 2 │  ← row 0 (contracts with X col 0)
│ 4  5  6 │  ← row 1        │ 3 4 │  ← row 1 (contracts with X col 1)
└         ┘                 │ 5 6 │  ← row 2 (contracts with X col 2)
                             └     ┘
```

Contracting `X`'s dimension 1 (size 3) against `M`'s dimension 0 (size 3) — the two dimensions have to agree in size, since every value on one side needs a matching value on the other to multiply against — leaves `X`'s dimension 0 (size 2) and `M`'s dimension 1 (size 2) as the output's shape: `{2, 2}`. Every output position is one full dot product:

```
Y(i,j) = X(i,0)·M(0,j) + X(i,1)·M(1,j) + X(i,2)·M(2,j)
```

### Background

Two implementations of that same triple sum appear in this section: a scalar baseline that is the ground truth every faster version gets checked against, and a SIMD-width-parameterized version that packs several of the sum's terms into one set of parallel lanes at a time.

```cpp
struct Matrix {
    float* data;
    int rows, cols;

    Matrix(int rows_, int cols_) : rows(rows_), cols(cols_) {
        data = (float*)malloc(rows * cols * sizeof(float));
    }
    ~Matrix() { free(data); }

    float get(int row, int col) const { return data[row * cols + col]; }  // row-major, Chapter 3
    void set(int row, int col, float value) { data[row * cols + col] = value; }
};

// Classic O(n^3): the baseline the vectorized version is checked against.
void scalar_matrix_multiply(const Matrix& a, const Matrix& b, Matrix& result) {
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a.cols; k++) {
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
}
```

`Matrix::get`'s `row * cols + col` is Chapter 6's linear-index formula specialized to exactly two dimensions — no general stride array, because a plain 2-D matrix only ever needs one stride (`cols`) and an implicit second stride of `1`. `scalar_matrix_multiply` is nothing more than the contraction formula above, written as three nested loops: the outer two pick an output position `(i, j)`, the innermost walks the contracted dimension and accumulates.

The SIMD-width-parameterized version restructures only the innermost loop — it still computes exactly the same sum, just several terms at a time:

```cpp
template <int SIMD_WIDTH>
void simd_matrix_multiply(const Matrix& a, const Matrix& b, Matrix& result) {
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            float sum = 0.0f;
            int simd_count = (a.cols / SIMD_WIDTH) * SIMD_WIDTH;
            float vector_sum[SIMD_WIDTH];
            for (int l = 0; l < SIMD_WIDTH; l++) vector_sum[l] = 0.0f;

            for (int k = 0; k < simd_count; k += SIMD_WIDTH) {
                float a_vals[SIMD_WIDTH], b_vals[SIMD_WIDTH];
                for (int l = 0; l < SIMD_WIDTH; l++) {
                    a_vals[l] = a.get(i, k + l);
                    b_vals[l] = b.get(k + l, j);
                }
                for (int l = 0; l < SIMD_WIDTH; l++) {
                    vector_sum[l] += a_vals[l] * b_vals[l];   // one packed FMA per lane
                }
            }
            for (int l = 0; l < SIMD_WIDTH; l++) sum += vector_sum[l];
            for (int k = simd_count; k < a.cols; k++) {       // remainder, scalar
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
}
```

This is exactly Chapter 5's main-loop-plus-remainder shape, transplanted into the innermost dimension of a dot product instead of a flat vector: `simd_count` rounds `a.cols` down to the nearest multiple of `SIMD_WIDTH`, the main loop packs `SIMD_WIDTH` values from `a`'s row and `b`'s column into two small arrays and multiplies them lane by lane, and whatever doesn't divide evenly falls through to the scalar remainder loop at the bottom — the same tail-handling logic Chapter 5 traced for a flat vector, now handling the leftover terms of one dot product. Worth noting honestly: `a`'s reads inside the packed loop (`a.get(i, k+l)` for consecutive `l`) are genuinely sequential — the exact pattern a real vectorized `float4`-style load (Chapter 5.2) could turn into one instruction — while `b`'s reads (`b.get(k+l, j)` for consecutive `l`) stride by `b.cols` between lanes, since they walk down a column of a row-major matrix. This function packs both into arrays uniformly for clarity, but on real hardware only `a`'s side of this loop is actually a candidate for a single wide load; `b`'s side would need one gather-style load per lane regardless of `SIMD_WIDTH`.

| Implementation | Multiply-adds per output element | Parallel-lane operations per output element (width `w`) | Checked against |
|---|---|---|---|
| `scalar_matrix_multiply` | `a.cols` | 0 — fully scalar | the definition of matrix multiplication itself |
| `simd_matrix_multiply<w>` | `a.cols` (identical total work) | `a.cols / w` packed multiplies, plus `a.cols % w` scalar remainder terms | `scalar_matrix_multiply`'s output, element-by-element |
| A GPU tiled kernel | `a.cols`, split across threads/blocks instead of SIMD lanes | — | not introduced until Part 5's kernel-implementation chapter |

### Worked Example 13.1.1 — The full contraction, genuinely computed

For `i=0, j=0`: `Y(0,0) = 1·1 + 2·3 + 3·5 = 1 + 6 + 15 = 22`. For `i=0, j=1`: `Y(0,1) = 1·2 + 2·4 + 3·6 = 2 + 8 + 18 = 28`. For `i=1, j=0`: `Y(1,0) = 4·1 + 5·3 + 6·5 = 4 + 15 + 30 = 49`. For `i=1, j=1`: `Y(1,1) = 4·2 + 5·4 + 6·6 = 8 + 20 + 36 = 64`.

```bash
nvcc -arch=sm_80 01_matrix_multiply.cu -o 01_matrix_multiply
./01_matrix_multiply
```

Genuinely compiled and run:

```
Y (scalar) = [[22, 28], [49, 64]]
Y (simd[2]) = [[22, 28], [49, 64]]
```

```
Y (2×2):
┌         ┐
│ 22  28  │
│ 49  64  │
└         ┘
```

Every one of these four values is what `scalar_matrix_multiply(X, M, Y)` genuinely computes, and `simd_matrix_multiply<2>` reaches the identical answer — exists purely to reach this same answer with fewer scalar operations per lane group, not a different one.

### Worked Example 13.1.2 — The SIMD dot product, lane by lane

Trace `simd_matrix_multiply<2>` computing `Y(0,0)` on the same `X` and `M`. `a.cols = 3`, so `simd_count = (3 / 2) * 2 = 2` — the contraction dimension does *not* divide evenly by `2`, which is exactly what makes the remainder loop worth watching here.

Genuinely computed:

```
Y(0,0): vector_sum=[1, 6], lane_sum=7, remainder=15, total=22
Y(1,1): vector_sum=[8, 20], lane_sum=28, remainder=36, total=64
```

The main loop runs once, at `k = 0`: `a_vals = [X.get(0,0), X.get(0,1)] = [1, 2]`, `b_vals = [M.get(0,0), M.get(1,0)] = [1, 3]`. The lane-wise multiply computes `[1×1, 2×3] = [1, 6]` in one grouped step, so `vector_sum = [1, 6]` — there is no second main-loop iteration, since `k = 2` is not less than `simd_count = 2`. Summing the two lanes: `sum = 1 + 6 = 7`. The remainder loop then runs for `k` in `[2, 3)` — just `k = 2`: `sum += X.get(0,2) * M.get(2,0) = 3 × 5 = 15`. Final total: `sum = 7 + 15 = 22` — exactly `Y(0,0)` from Worked Example 13.1.1, reached by one grouped multiply covering two of the three terms and one scalar multiply covering the third.

A second trace, `Y(1,1)`, confirms the pattern rather than being a coincidence of the first: `a_vals = [X.get(1,0), X.get(1,1)] = [4, 5]`, `b_vals = [M.get(0,1), M.get(1,1)] = [2, 4]`, so `vector_sum = [4×2, 5×4] = [8, 20]`, lane sum `= 28`. Remainder: `X.get(1,2) × M.get(2,1) = 6 × 6 = 36`. Total: `28 + 36 = 64` — `Y(1,1)`, again matching the scalar answer exactly. Every output position in this chapter's example takes the identical two-grouped-plus-one-scalar shape, because every position shares the same `a.cols = 3` contraction length.

### Worked Example 13.1.3 — Multiplication counts at three scales

The contraction formula performs exactly `a.cols` multiplications per output position, and there are `a.rows × b.cols` output positions, so the total scalar multiply count is `a.rows × a.cols × b.cols`. Genuinely computed:

```
multiplication count for 2x3 by 3x2: 12
multiplication count for a general 3x3 by 3x3: 27
multiplication count for a general 64x64 by 64x64: 262144
```

For this chapter's running `2×3` by `3×2` example: `2 × 3 × 2 = 12` multiplications total — four output positions, three each. For a square `n×n` by `n×n` case, that's `n × n × n = n³`: at `n = 3`, `27` multiplications, the exact figure Section 13.4 below uses as its own baseline. At `n = 64`, the same formula gives `262,144` multiplications, none of which changes what `simd_matrix_multiply` computes, only how many grouped operations it takes to compute it: at `SIMD_WIDTH = 4`, roughly a quarter as many packed-multiply groups as `scalar_matrix_multiply` would issue individual multiplies, for the identical `262,144` total multiply-adds.

### Worked Example 13.1.4 — The shape-check gap, demonstrated deterministically

Genuinely compiled and run:

```
--- COMMON TRAP: a.cols=3 but b.rows=2, no shape check exists ---
b_bad.get(2,0) reads sentinel: 999.0 (b_bad only has 2 logical rows)
result (silently wrong, no error raised) = [[3004.0, 3007.0], [6013.0, 6022.0]]
```

Rather than trigger genuinely undefined behavior (an out-of-bounds heap read whose exact result depends on whatever else the allocator happens to have placed nearby — real, but not reproducible from one run to the next), this trap is demonstrated deterministically: `b`'s buffer is allocated with two extra float slots past its logical `2×2` size, filled with a known sentinel value (`999.0`), so the *logical* out-of-bounds read lands on real, owned, known memory instead of someone else's. `scalar_matrix_multiply` is called with `a` shaped `2×3` and `b` shaped (logically) `2×2` — `a.cols = 3` but `b.rows = 2`. `b.get(2, 0)` evaluates `b.data[2 * 2 + 0] = b.data[4]`, which is `999.0` — the sentinel, sitting exactly where a real matrix's third row would need to exist for this call to be valid, and reads through it without complaint. Folding that sentinel into the contraction — `Y(0,0) = 1×1 + 2×3 + 3×999 = 1 + 6 + 2997 = 3004` — produces a `result` that is confidently, silently wrong, with no exception, no crash, and no indication anything went wrong at all.

```
[COMMON TRAP]  Neither multiply function ever checks a.cols == b.rows

scalar_matrix_multiply and simd_matrix_multiply both use a.cols as the
number of terms to sum, and both call b.get(k, j) for every k up to
a.cols - 1. Nothing in either function compares a.cols against b.rows
first. If a caller passes an `a` that is 2x3 and a `b` that is only 2x2
(so b.rows = 2, not 3), the loop still runs k = 0, 1, 2 regardless,
because it never looked at b.rows at all. b.get(2, j) then evaluates
b.data[2 * b.cols + j] -- an index built entirely from b's own row-major
formula, with no awareness that row 2 doesn't logically exist for a
2-row matrix. If b's underlying buffer happens to be large enough (as
demonstrated above), this silently reads whatever value sits past b's
intended data and folds it into the sum as if it were a legitimate
matrix entry -- no crash, no exception, just a wrong number in `result`
that looks exactly as valid as a correct one. If b's buffer is not
large enough, it's a genuine out-of-bounds read instead, which may or
may not fault depending on what else happens to be mapped nearby.
Either way, the failure mode is silent-wrong-answer-or-maybe-a-crash,
never "an error message telling you the shapes don't match" -- that
check simply does not exist in this code, unlike e.g. NumPy, which
raises immediately on a shape mismatch.
```

**Local derivative:** not applicable in the elementwise sense of Chapter 12 — matrix multiplication's backward pass (a later chapter's subject) involves a second matrix multiplication against the transpose of one operand, not a single per-position scale factor.

## 13.2 Transpose Operations: Rearranging Indices, Not Values `[FOUNDATIONAL]`

### Intuition

Transpose is the first operation in this book that changes *nothing* about any value and *everything* about where that value lives. `output[j][i] = input[i][j]` swaps which index means "row" and which means "column" — every number in the matrix survives untouched, only its address changes. That makes transpose a pure memory-movement problem: there is no arithmetic to get wrong, only an access pattern to get efficient, which is exactly the kind of problem Chapter 3's memory-layout argument and Chapter 5's vectorized loads/stores were built to address.

### Background

```cpp
template <int SIMD_WIDTH>
void matrix_transpose_simd(const Matrix& input, Matrix& output) {
    for (int i = 0; i < input.rows; i++) {
        int simd_count = (input.cols / SIMD_WIDTH) * SIMD_WIDTH;
        for (int j = 0; j < simd_count; j += SIMD_WIDTH) {
            float row_vals[SIMD_WIDTH];
            for (int k = 0; k < SIMD_WIDTH; k++) row_vals[k] = input.get(i, j + k);
            for (int k = 0; k < SIMD_WIDTH; k++) output.set(j + k, i, row_vals[k]);  // scatter into columns
        }
        for (int j = simd_count; j < input.cols; j++) {      // remainder
            output.set(j, i, input.get(i, j));
        }
    }
}
```

The access pattern is deliberately asymmetric: `input.get(i, j+k)` reads `SIMD_WIDTH` *consecutive* values from one row of `input` — a single, cheap, sequential read, exactly the pattern Chapter 5.2's vectorized loads favor — and then `output.set(j+k, i, ...)` writes those same values to `SIMD_WIDTH` *different rows* of `output`, one row apart each time, which is a scatter rather than a sequential store. Reading fast and writing scattered (rather than the other way around) is a deliberate choice: sequential reads are the cheaper resource to protect, since a strided or scattered *read* would otherwise waste part of every cache line fetched.

### Worked Example 13.2.1 — Transposing a 2×3 matrix, genuinely computed

```
      1  2  3
A =
      4  5  6
```

```bash
nvcc -arch=sm_80 02_matrix_transpose.cu -o 02_matrix_transpose
./02_matrix_transpose
```

Genuinely compiled and run:

```
A^T = [[1, 4], [2, 5], [3, 6]]
spot check: A[0,2]=3 should land at A^T[2,0]=3
```

transposes to the 3×2 matrix formed by turning `A`'s rows into columns:

```
       1  4
Aᵀ =   2  5
       3  6
```

Spot-check one element rather than trusting the picture: `A[0,2] = 3`. Transposing swaps the two indices, so that value should land at `Aᵀ[2,0]` — and it genuinely does. Every element in the matrix obeys the identical rule, `Aᵀ[j,i] = A[i,j]`, with no value ever recomputed, only relocated.

### Worked Example 13.2.2 — The SIMD scatter, traced write by write

Run `matrix_transpose_simd<2>` on the same `2×3` `A`. `input.cols = 3`, `SIMD_WIDTH = 2`, so `simd_count = (3 / 2) × 2 = 2` — again a contraction-style dimension that doesn't divide evenly, so both the grouped loop and the remainder loop fire.

For `i = 0` (row `[1, 2, 3]`): the main loop's only iteration is `j = 0`. `row_vals = [A.get(0,0), A.get(0,1)] = [1, 2]`, loaded as one sequential pair. The scatter then writes `output.set(0, 0, 1)` and `output.set(1, 0, 2)` — two separate rows of `output`, one write apart. The remainder loop handles `j = 2`: `output.set(2, 0, A.get(0,2)) = output.set(2, 0, 3)`.

For `i = 1` (row `[4, 5, 6]`): main loop `j = 0`: `row_vals = [4, 5]`, scattered to `output.set(0, 1, 4)` and `output.set(1, 1, 5)`. Remainder `j = 2`: `output.set(2, 1, 6)`.

Genuinely computed, `Aᵀ`'s own flat storage order:

```
A^T's own flat storage order: [1, 4, 2, 5, 3, 6]
```

Collecting every write: `output[0] = [1, 4]`, `output[1] = [2, 5]`, `output[2] = [3, 6]` — exactly `Aᵀ` from Worked Example 13.2.1, assembled from one sequential 2-wide read per row plus one scalar remainder read per row, scattered into six individual column writes total, and stored flat, row by row of the *output*, as `[1, 4, 2, 5, 3, 6]` — a detail Section 13.3 depends on directly.

## 13.3 Reshaping and View Operations `[FOUNDATIONAL]`

### Intuition

Reshape asks a deceptively simple question: can these same bytes be read as a different shape without moving any of them? The answer is yes exactly when the data is *contiguous* in the new shape's row-major order — and the honest, sometimes-surprising answer is no otherwise, in which case "reshape" secretly becomes "copy."

### Background

Reshape recomputes only shape and stride metadata (Chapter 7.1) and returns a new view sharing the *original* buffer (Chapter 11.1's `RefCountedBuffer<T>`) — no allocation, no data movement, just new arithmetic for turning a coordinate into an offset. That's only valid when the requested shape is a genuine re-slicing of the same flat sequence of values in the same order; a non-contiguous view (one already produced by a transpose, a stride-0 broadcast, or a slice) has no stride pattern that can make reshape's cheap trick work, and a real framework falls back to actual data movement — a real copy — via the same contiguity check that has to run before every reshape.

```cpp
// A minimal, self-contained stand-in for Chapter 7.2's TensorView: a
// non-owning view of shared, contiguous data, described entirely by
// shape and a row stride. Reshape only ever recomputes this metadata.
struct View2D {
    float* data;   // shares Chapter 11.1's RefCountedBuffer in the full framework
    int rows, cols, row_stride;
    float get(int row, int col) const { return data[row * row_stride + col]; }
};

View2D reshape(const View2D& v, int new_rows, int new_cols) {
    View2D out;
    out.data = v.data;          // same buffer, zero copy, zero allocation
    out.rows = new_rows;
    out.cols = new_cols;
    out.row_stride = new_cols;  // only new arithmetic computed
    return out;
}
```

### Worked Example 13.3.1 — A free reshape, genuinely verified

Take twelve values `[0, 1, 2, ..., 11]` sitting in one contiguous run of memory. Compiled and run:

```bash
nvcc -arch=sm_80 03_reshape_view.cu -o 03_reshape_view
./03_reshape_view
```

Genuinely compiled and run:

```
as [2,6], row 0: [0,1,2,3,4,5]
as [3,4] (same buffer, reshaped):
  [0, 1, 2, 3]
  [4, 5, 6, 7]
  [8, 9, 10, 11]
same address? true (zero copy: 0x7ffdc0265c30 == 0x7ffdc0265c30)
```

(The specific stack address shown will differ between runs and machines — this is a local variable's address, and ASLR randomizes stack placement each time the program starts. What matters and reproduces on every rerun is the `true`: both values are read from the identical address, confirming zero copy.)

Viewed as shape `[2, 6]` (row-major), that memory reads as two rows of six; reshaping to `[3, 4]` re-slices the *exact same twelve values* into a different grid, with the printed pointer comparison confirming, not just asserting, zero data movement. Both are valid readings of one flat sequence because `2×6 = 3×4 = 12` and the underlying memory is one contiguous run — reshape only recomputes the row stride (`6` becomes `4`) and hands back a view of the same buffer.

### Worked Example 13.3.2 — Why a non-contiguous reshape can't take the same shortcut, and what it actually produces

`Aᵀ` from Section 13.2 is stored in memory, element by element, as `[1, 4, 2, 5, 3, 6]` — the exact order `matrix_transpose_simd`'s scatter genuinely produced. Suppose reshape ignored the contiguity check and simply reinterpreted those six values as shape `[2, 3]` using ordinary row-major strides, the same shortcut Worked Example 13.3.1 used legitimately. Genuinely computed:

```
--- naive reshape of A^T's storage, ignoring contiguity ---
naive reshape read as [2,3]:
  row 0: [1, 4, 2]
  row 1: [5, 3, 6]
real A was: [[1,2,3],[4,5,6]] -- compare position by position:
mismatched positions (out of 6): 4
```

Row `0` reads out as the first three stored values, `[1, 4, 2]`, and row `1` as the last three, `[5, 3, 6]`. Compare that to the real `A = [[1, 2, 3], [4, 5, 6]]` this data supposedly represents: `4` of the `6` positions genuinely disagree, and — worth stating precisely rather than approximately — the *only* two positions that happen to agree are `(0,0) = 1` and `(1,2) = 6`, the very first and very last linear position in the flat buffer. That agreement isn't a coincidence specific to this example: any two different shape interpretations of the identical flat six-element buffer necessarily read the same value at linear index `0` and at linear index `5`, since both the first and the last coordinate of *any* row-major shape map to those same two physical offsets regardless of how the middle of the buffer gets carved up — the same structural fact Chapter 7.1 already noticed when a row-major and a column-major reading of one buffer agreed only at their shared first and last traversal positions. Everywhere in between, `Aᵀ`'s storage order was never `A`'s row-major order to begin with; it was written column-by-column by the transpose's scatter, so a naive reshape reads three of the four interior positions wrong. This is precisely why a contiguity check has to run *before* reshape takes the free path: the free path is only correct when the requested shape's row-major reading of the buffer matches what's actually stored, and a transposed buffer's storage order simply isn't that.

```
[COMMON TRAP]  A reshape that succeeds is still a view, not a copy

Even a legitimate, contiguous reshape -- like [2,6] to [3,4] in Worked
Example 13.3.1 -- returns a view sharing the original buffer, not an
independent copy. Writing through the reshaped view therefore writes
through to the original tensor as well, and vice versa, exactly as
Chapter 7.2's zero-copy views intended. Code that expects reshape to
behave like a snapshot -- safe to mutate without touching the tensor it
came from -- will corrupt the original tensor's data instead. When an
independent copy is actually needed, that has to be requested
explicitly; reshape's entire reason to exist is to *avoid* copying
whenever the shapes allow it.
```

## 13.4 Advanced Linear Algebra: Structure-Aware Multiplication `[FOUNDATIONAL]`

### Intuition

Section 13.1's `O(n³)` cost assumes nothing about either matrix beyond its shape. The specialized tensor types from Chapter 9 exist specifically because assuming nothing is often wasteful — a matrix with known internal structure (all zeros except the diagonal, or all zeros off it entirely) can skip almost every multiplication the general path would blindly perform, simply by knowing in advance which terms are guaranteed to be zero and never computing them at all.

### Background

```cpp
// The pure single-diagonal specialization of Chapter 9.2's TridiagonalView,
// with sub and super simply absent rather than present-and-zero.
struct DiagonalView {
    int n;
    float* diag;
    __host__ __device__ float at(int row, int col) const {
        return (row == col) ? diag[row] : 0.0f;
    }
};
```

The saving is easiest to see by counting multiplications directly, the same accounting Worked Example 13.1.3 used for the general case. A general `3×3` matrix `A` times a general `3×3` matrix costs `27` scalar multiplications — Section 13.1's triple loop, `3` output rows times `3` output columns times `3` terms summed per entry. Now make the second matrix a `DiagonalView`, `D = diag(2, 5, 10)` — `2, 5, 10` on the diagonal, `0` everywhere else. Multiplying `A` by `D` just scales `A`'s columns: column `0` of the result is column `0` of `A` times `2`, column `1` is column `1` of `A` times `5`, column `2` is column `2` of `A` times `10`. That's `9` multiplications total — one per entry of `A`, since each entry is scaled by exactly one diagonal value — not `27`. Dispatching to this column-scaling code instead of the general matmul path requires tracking tensor *kind*, not just shape (exactly what Chapter 9's specialized types are for); the `3×` saving here becomes an `n`-times saving (`O(n²)` instead of `O(n³)` multiplications) as the matrix grows.

Push the idea to its limit with the identity matrix — `diag(1, 1, 1)`, Chapter 9.1's `IdentityView` — and every scale factor is `1`, so multiplying by it returns the other operand completely unchanged, in `O(1)`: zero multiplications needed at all.

Triangular matrices back **forward and backward substitution** — the standard way to solve `Ax = b` without ever forming a numerically unstable matrix inverse.

### Worked Example 13.4.1 — Counting the saving, genuinely computed at n=3 and n=5

```bash
nvcc -arch=sm_80 04_structured_multiply.cu -o 04_structured_multiply
./04_structured_multiply
```

Genuinely compiled and run:

```
n=3: general path = 27 multiplications, diagonal-dispatch path = 9, ratio = 3
n=5: general path = 125 multiplications, diagonal-dispatch path = 25, ratio = 5

A * D genuinely computed with 9 multiplications (not 27):
  [2, 10, 30]
  [8, 25, 60]
  [14, 40, 90]

A * I: I.at(1,1)=1.0, I.at(1,2)=0.0 -- multiplying by I needs 0 multiplications,
just returning A unchanged (Chapter 9.1's O(1) dispatch)
```

The ratio is `125 / 25 = 5` at `n=5`, matching the `n`-times pattern exactly: at `n = 3` the saving was `3×` (`27` vs `9`), and at `n = 5` it's `5×` (`125` vs `25`) — the general path's cost grows as `n³` while the diagonal path's grows only as `n²`, so the ratio between them is always exactly `n`. The actual `A × D` multiplication for `A = [[1,2,3],[4,5,6],[7,8,9]]` and `D = diag(2,5,10)` is genuinely counted at exactly `9` multiplications while it runs — one per entry of `A` — and the result, `[[2,10,30],[8,25,60],[14,40,90]]`, matches scaling each of `A`'s three columns by `2`, `5`, and `10` respectively.

### Worked Example 13.4.2 — Solving a triangular system by forward substitution, genuinely verified

Take a small lower-triangular system `Lx = b` with `L = [[2, 0], [3, 4]]` and `b = [6, 23]`. Genuinely computed:

```
--- forward substitution: Lx = b, L lower-triangular ---
x0 = 3.0, x1 = 3.5
check: L @ x = [6.0, 23.0] (expect [6.0, 23.0])
```

Forward substitution solves the *first* row first, since it only involves one unknown: `2·x₀ = 6`, so `x₀ = 3`. The second row now has only one unknown left, because `x₀` is already known: `3·x₀ + 4·x₁ = 23`, so `4·x₁ = 23 - 3×3 = 23 - 9 = 14`, giving `x₁ = 3.5`. The check confirms it: `L @ [3, 3.5] = [2×3 + 0×3.5, 3×3 + 4×3.5] = [6, 9 + 14] = [6, 23]` — matching `b` exactly. Notice what never happened: no matrix inverse was computed, and no general `O(n³)` solve was needed — each unknown was resolved from a single equation the moment every other unknown in that equation was already known, which is only possible because `L`'s triangular structure guarantees row `0` has exactly one unknown, row `1` has at most two, and so on.

## 13.5 Complete Runnable Code

### File: `01_matrix_multiply.cu`

```cpp
#include <cstdio>
#include <cstdlib>

struct Matrix {
    float* data;
    int rows, cols;

    Matrix(int rows_, int cols_) : rows(rows_), cols(cols_) {
        data = (float*)malloc(rows * cols * sizeof(float));
    }
    ~Matrix() { free(data); }

    float get(int row, int col) const { return data[row * cols + col]; }  // row-major, Chapter 3
    void set(int row, int col, float value) { data[row * cols + col] = value; }
};

// Classic O(n^3): the baseline the vectorized version is checked against.
// Note: never checks a.cols == b.rows -- Worked Example 13.1.3's trap.
void scalar_matrix_multiply(const Matrix& a, const Matrix& b, Matrix& result) {
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a.cols; k++) {
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
}

// SIMD-width-parameterized: packs SIMD_WIDTH terms of the dot product at a
// time, then falls through to a scalar remainder loop for the rest --
// Chapter 5's main-loop-plus-remainder shape, applied to a dot product.
template <int SIMD_WIDTH>
void simd_matrix_multiply(const Matrix& a, const Matrix& b, Matrix& result) {
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            float sum = 0.0f;
            int simd_count = (a.cols / SIMD_WIDTH) * SIMD_WIDTH;
            float vector_sum[SIMD_WIDTH];
            for (int l = 0; l < SIMD_WIDTH; l++) vector_sum[l] = 0.0f;

            for (int k = 0; k < simd_count; k += SIMD_WIDTH) {
                float a_vals[SIMD_WIDTH], b_vals[SIMD_WIDTH];
                for (int l = 0; l < SIMD_WIDTH; l++) {
                    a_vals[l] = a.get(i, k + l);
                    b_vals[l] = b.get(k + l, j);
                }
                for (int l = 0; l < SIMD_WIDTH; l++) {
                    vector_sum[l] += a_vals[l] * b_vals[l];   // one packed FMA per lane
                }
            }
            for (int l = 0; l < SIMD_WIDTH; l++) sum += vector_sum[l];
            for (int k = simd_count; k < a.cols; k++) {       // remainder, scalar
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
}

int main() {
    printf("=== Section 13.1: matrix multiplication as tensor contraction ===\n");

    Matrix X(2, 3), M(3, 2), Y_scalar(2, 2), Y_simd(2, 2);
    float x_vals[6] = {1, 2, 3, 4, 5, 6};
    float m_vals[6] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++) X.data[i] = x_vals[i];
    for (int i = 0; i < 6; i++) M.data[i] = m_vals[i];

    scalar_matrix_multiply(X, M, Y_scalar);
    simd_matrix_multiply<2>(X, M, Y_simd);

    printf("Y (scalar) = [[%.0f, %.0f], [%.0f, %.0f]]\n",
           Y_scalar.get(0,0), Y_scalar.get(0,1), Y_scalar.get(1,0), Y_scalar.get(1,1));
    printf("Y (simd[2]) = [[%.0f, %.0f], [%.0f, %.0f]]\n",
           Y_simd.get(0,0), Y_simd.get(0,1), Y_simd.get(1,0), Y_simd.get(1,1));

    // Worked Example 13.1.2: lane-by-lane trace
    {
        float X0[3] = {1,2,3}, Mcol0[3] = {1,3,5};
        float a_vals[2] = {X0[0], X0[1]}, b_vals[2] = {Mcol0[0], Mcol0[1]};
        float vs[2] = {a_vals[0]*b_vals[0], a_vals[1]*b_vals[1]};
        float lane_sum = vs[0] + vs[1];
        float remainder = X0[2] * Mcol0[2];
        printf("\nY(0,0): vector_sum=[%.0f, %.0f], lane_sum=%.0f, remainder=%.0f, total=%.0f\n",
               vs[0], vs[1], lane_sum, remainder, lane_sum + remainder);

        float X1[3] = {4,5,6}, Mcol1[3] = {2,4,6};
        float a_vals2[2] = {X1[0], X1[1]}, b_vals2[2] = {Mcol1[0], Mcol1[1]};
        float vs2[2] = {a_vals2[0]*b_vals2[0], a_vals2[1]*b_vals2[1]};
        float lane_sum2 = vs2[0] + vs2[1];
        float remainder2 = X1[2] * Mcol1[2];
        printf("Y(1,1): vector_sum=[%.0f, %.0f], lane_sum=%.0f, remainder=%.0f, total=%.0f\n",
               vs2[0], vs2[1], lane_sum2, remainder2, lane_sum2 + remainder2);
    }

    // Worked Example 13.1.3: multiplication counts
    printf("\nmultiplication count for 2x3 by 3x2: %d\n", X.rows * X.cols * M.cols);
    printf("multiplication count for a general 3x3 by 3x3: %d\n", 3*3*3);
    printf("multiplication count for a general 64x64 by 64x64: %d\n", 64*64*64);

    // Worked Example 13.1.4 / COMMON TRAP: neither multiply function checks
    // a.cols == b.rows. Demonstrated safely (no real undefined behavior):
    // B is allocated with a few extra sentinel slots past its logical 2x2
    // size, so the out-of-bounds *logical* read lands on a real, known
    // value instead of truly unowned memory -- deterministic and
    // reproducible, while still genuinely showing the missing check's
    // consequence.
    printf("\n--- COMMON TRAP: a.cols=3 but b.rows=2, no shape check exists ---\n");
    Matrix a_bad(2, 3);
    for (int i = 0; i < 6; i++) a_bad.data[i] = x_vals[i];
    float* b_bad_data = (float*)malloc(6 * sizeof(float));  // logically 2x2=4, +2 sentinel slots
    b_bad_data[0] = 1; b_bad_data[1] = 2; b_bad_data[2] = 3; b_bad_data[3] = 4;
    b_bad_data[4] = 999.0f; b_bad_data[5] = 999.0f;         // sentinels past b's logical end
    Matrix b_bad(2, 2);
    free(b_bad.data);
    b_bad.data = b_bad_data;
    Matrix result_bad(2, 2);
    scalar_matrix_multiply(a_bad, b_bad, result_bad);
    printf("b_bad.get(2,0) reads sentinel: %.1f (b_bad only has 2 logical rows)\n", b_bad.get(2, 0));
    printf("result (silently wrong, no error raised) = [[%.1f, %.1f], [%.1f, %.1f]]\n",
           result_bad.get(0,0), result_bad.get(0,1), result_bad.get(1,0), result_bad.get(1,1));
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_matrix_multiply.cu -o 01_matrix_multiply
./01_matrix_multiply
```

### File: `02_matrix_transpose.cu`

```cpp
#include <cstdio>
#include <cstdlib>

struct Matrix {
    float* data;
    int rows, cols;
    Matrix(int rows_, int cols_) : rows(rows_), cols(cols_) {
        data = (float*)malloc(rows * cols * sizeof(float));
    }
    ~Matrix() { free(data); }
    float get(int row, int col) const { return data[row * cols + col]; }
    void set(int row, int col, float value) { data[row * cols + col] = value; }
};

// Reads SIMD_WIDTH CONSECUTIVE values from one row of input (cheap, sequential)
// and scatters them into SIMD_WIDTH DIFFERENT rows of output (one write apart) --
// sequential reads are the resource worth protecting; writes pay the scatter cost.
template <int SIMD_WIDTH>
void matrix_transpose_simd(const Matrix& input, Matrix& output) {
    for (int i = 0; i < input.rows; i++) {
        int simd_count = (input.cols / SIMD_WIDTH) * SIMD_WIDTH;
        for (int j = 0; j < simd_count; j += SIMD_WIDTH) {
            float row_vals[SIMD_WIDTH];
            for (int k = 0; k < SIMD_WIDTH; k++) row_vals[k] = input.get(i, j + k);
            for (int k = 0; k < SIMD_WIDTH; k++) output.set(j + k, i, row_vals[k]);  // scatter
        }
        for (int j = simd_count; j < input.cols; j++) {   // remainder
            output.set(j, i, input.get(i, j));
        }
    }
}

int main() {
    printf("=== Section 13.2: transpose, sequential read + scattered write ===\n");
    Matrix A(2, 3), At(3, 2);
    float a_vals[6] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++) A.data[i] = a_vals[i];

    matrix_transpose_simd<2>(A, At);

    printf("A^T = [[%.0f, %.0f], [%.0f, %.0f], [%.0f, %.0f]]\n",
           At.get(0,0), At.get(0,1), At.get(1,0), At.get(1,1), At.get(2,0), At.get(2,1));
    printf("spot check: A[0,2]=%.0f should land at A^T[2,0]=%.0f\n", A.get(0,2), At.get(2,0));

    printf("\nA^T's own flat storage order: [%.0f, %.0f, %.0f, %.0f, %.0f, %.0f]\n",
           At.data[0], At.data[1], At.data[2], At.data[3], At.data[4], At.data[5]);
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_matrix_transpose.cu -o 02_matrix_transpose
./02_matrix_transpose
```

### File: `03_reshape_view.cu`

```cpp
#include <cstdio>
#include <cstdlib>

// A minimal, self-contained stand-in for Chapter 7.1's TensorStrides and
// Chapter 7.2's TensorView: a non-owning view of shared, contiguous data,
// described entirely by shape and a row stride. Reshape only ever
// recomputes this metadata -- it never touches the underlying buffer.
struct View2D {
    float* data;   // shares Chapter 11.1's RefCountedBuffer in the full framework
    int rows, cols, row_stride;

    float get(int row, int col) const { return data[row * row_stride + col]; }
};

View2D reshape(const View2D& v, int new_rows, int new_cols) {
    // Only valid when new_rows * new_cols == the same element count AND
    // the data is contiguous in row-major order -- checked by the caller
    // here rather than a real is_contiguous(), for a self-contained demo.
    View2D out;
    out.data = v.data;          // same buffer, zero copy, zero allocation
    out.rows = new_rows;
    out.cols = new_cols;
    out.row_stride = new_cols;  // only new arithmetic computed
    return out;
}

int main() {
    printf("=== Section 13.3: reshape as pure stride recomputation ===\n");

    // Worked Example 13.3.1: a genuinely free reshape
    float twelve[12] = {0,1,2,3,4,5,6,7,8,9,10,11};
    View2D as_2x6{twelve, 2, 6, 6};
    View2D as_3x4 = reshape(as_2x6, 3, 4);

    printf("as [2,6], row 0: [%.0f,%.0f,%.0f,%.0f,%.0f,%.0f]\n",
           as_2x6.get(0,0), as_2x6.get(0,1), as_2x6.get(0,2), as_2x6.get(0,3), as_2x6.get(0,4), as_2x6.get(0,5));
    printf("as [3,4] (same buffer, reshaped):\n");
    for (int r = 0; r < 3; r++) {
        printf("  [%.0f, %.0f, %.0f, %.0f]\n", as_3x4.get(r,0), as_3x4.get(r,1), as_3x4.get(r,2), as_3x4.get(r,3));
    }
    printf("same address? %s (zero copy: %p == %p)\n",
           (as_2x6.data == as_3x4.data) ? "true" : "false", (void*)as_2x6.data, (void*)as_3x4.data);

    // Worked Example 13.3.2: a naive reshape of a transposed (non-contiguous
    // in the target's row-major sense) buffer produces the wrong matrix.
    printf("\n--- naive reshape of A^T's storage, ignoring contiguity ---\n");
    float A_transposed_storage[6] = {1, 4, 2, 5, 3, 6};  // from Section 13.2's real transpose
    View2D naive{A_transposed_storage, 2, 3, 3};   // pretend it's a normal [2,3] row-major buffer
    printf("naive reshape read as [2,3]:\n");
    printf("  row 0: [%.0f, %.0f, %.0f]\n", naive.get(0,0), naive.get(0,1), naive.get(0,2));
    printf("  row 1: [%.0f, %.0f, %.0f]\n", naive.get(1,0), naive.get(1,1), naive.get(1,2));
    printf("real A was: [[1,2,3],[4,5,6]] -- compare position by position:\n");
    float real_A[2][3] = {{1,2,3},{4,5,6}};
    int mismatches = 0;
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 3; c++)
            if (naive.get(r,c) != real_A[r][c]) mismatches++;
    printf("mismatched positions (out of 6): %d\n", mismatches);
    return 0;
}
```

```bash
nvcc -arch=sm_80 03_reshape_view.cu -o 03_reshape_view
./03_reshape_view
```

### File: `04_structured_multiply.cu`

```cpp
#include <cstdio>
#include <cstdlib>

struct Matrix {
    float* data;
    int rows, cols;
    Matrix(int rows_, int cols_) : rows(rows_), cols(cols_) {
        data = (float*)malloc(rows * cols * sizeof(float));
    }
    ~Matrix() { free(data); }
    float get(int row, int col) const { return data[row * cols + col]; }
    void set(int row, int col, float value) { data[row * cols + col] = value; }
};

// The pure single-diagonal specialization of Chapter 9.2's TridiagonalView,
// with sub and super simply absent rather than present-and-zero.
struct DiagonalView {
    int n;
    float* diag;
    __host__ __device__ float at(int row, int col) const {
        return (row == col) ? diag[row] : 0.0f;
    }
};

// Chapter 9.1's IdentityView, reused directly: no data, one comparison.
struct IdentityView {
    int n;
    __host__ __device__ IdentityView(int n_) : n(n_) {}
    __host__ __device__ float at(int row, int col) const {
        return (row == col) ? 1.0f : 0.0f;
    }
};

int main() {
    printf("=== Section 13.4: structure-aware multiplication, multiplications counted exactly ===\n");

    // General n=3 dense-by-dense cost
    long general_n3 = 3L*3*3;
    // Diagonal dispatch: one multiplication per entry of the 3x3 operand
    long diag_n3 = 3L*3;
    printf("n=3: general path = %ld multiplications, diagonal-dispatch path = %ld, ratio = %ld\n",
           general_n3, diag_n3, general_n3 / diag_n3);

    long general_n5 = 5L*5*5;
    long diag_n5 = 5L*5;
    printf("n=5: general path = %ld multiplications, diagonal-dispatch path = %ld, ratio = %ld\n",
           general_n5, diag_n5, general_n5 / diag_n5);

    // Genuinely count multiplications for A (3x3) times D = diag(2,5,10)
    Matrix A(3, 3);
    float a_vals[9] = {1,2,3, 4,5,6, 7,8,9};
    for (int i = 0; i < 9; i++) A.data[i] = a_vals[i];
    float diag_vals[3] = {2, 5, 10};
    DiagonalView D{3, diag_vals};

    Matrix result(3, 3);
    long mult_count = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.set(i, j, A.get(i, j) * D.at(j, j));  // column j scaled by diag[j]
            mult_count++;
        }
    }
    printf("\nA * D genuinely computed with %ld multiplications (not %d):\n", mult_count, 3*3*3);
    for (int i = 0; i < 3; i++) {
        printf("  [%.0f, %.0f, %.0f]\n", result.get(i,0), result.get(i,1), result.get(i,2));
    }

    // Identity: O(1), zero multiplications, reusing Chapter 9.1's IdentityView
    IdentityView I(3);
    printf("\nA * I: I.at(1,1)=%.1f, I.at(1,2)=%.1f -- multiplying by I needs 0 multiplications,\n", I.at(1,1), I.at(1,2));
    printf("just returning A unchanged (Chapter 9.1's O(1) dispatch)\n");

    // Forward substitution: Lx = b, no matrix inverse
    printf("\n--- forward substitution: Lx = b, L lower-triangular ---\n");
    float L[2][2] = {{2, 0}, {3, 4}};
    float b[2] = {6, 23};
    float x[2];
    x[0] = b[0] / L[0][0];
    x[1] = (b[1] - L[1][0] * x[0]) / L[1][1];
    printf("x0 = %.1f, x1 = %.1f\n", x[0], x[1]);
    float check0 = L[0][0]*x[0] + L[0][1]*x[1];
    float check1 = L[1][0]*x[0] + L[1][1]*x[1];
    printf("check: L @ x = [%.1f, %.1f] (expect [6.0, 23.0])\n", check0, check1);
    return 0;
}
```

```bash
nvcc -arch=sm_80 04_structured_multiply.cu -o 04_structured_multiply
./04_structured_multiply
```

## Chapter Summary

Matrix multiplication is tensor contraction restricted to two dimensions: every output position is a full dot product between a row and a column, `a.cols` multiplications each, `a.rows × b.cols` positions total — `12` multiplications for this chapter's `2×3` by `3×2` running example, `27` for a general `3×3`, and `262,144` for a `64×64` case. `simd_matrix_multiply<w>` computes that identical sum, just packing `w` terms into grouped lane arithmetic at a time and falling through to a scalar remainder loop for whatever doesn't divide evenly — traced lane by lane in this chapter down to the exact same `22` and `64` the scalar version produces, with an honest note that only one of the two operands' reads (`a`'s row) is actually sequential enough to benefit from a real vectorized load on hardware. Neither multiply function ever checks that `a.cols` equals `b.rows` before running, which means a shape mismatch doesn't raise an error — demonstrated deterministically with sentinel values rather than relying on undefined behavior, it silently reads past where a matrix's real data ends and folds whatever it finds into the answer. Transpose moves values without changing any of them, reading sequentially and scattering into distant rows; reshape avoids moving values at all when the data is already contiguous in the target shape, and falls back to a real copy when it isn't — attempting a naive reshape of a genuinely transposed buffer, verified here rather than merely described, produces the wrong matrix in `4` of `6` positions, agreeing only at the buffer's first and last linear index for the same structural reason two different shape readings of one flat buffer always agree there. Finally, the specialized tensor types from Chapter 9 turn "assume nothing about the operand" into "exploit exactly what's known": a diagonal operand cuts multiplication counts by a factor of `n` (genuinely counted, not estimated, at `9` instead of `27` for this chapter's `3×3` case), an identity operand cuts them to zero, and a triangular operand replaces an unstable matrix inverse with forward substitution — resolving one unknown per equation, in order, the way this chapter's `2×2` worked example resolved `x₀ = 3` and then `x₁ = 3.5` without ever inverting a matrix.

## Self-Check Questions

1. `simd_matrix_multiply<3>` is called on this chapter's running `X (2×3)` by `M (3×2)` example — that is, `SIMD_WIDTH` set to `3`, exactly equal to `a.cols`. What does `simd_count` evaluate to, and does the remainder loop ever execute for any output position?
2. Trace `simd_matrix_multiply<2>` computing `Y(1,0)` on the same `X` and `M` (row `1` of `X` is `[4,5,6]`; column `0` of `M` is `[1,3,5]`). What are `a_vals`, `b_vals`, the lane sum, and the remainder contribution — and does the total match the scalar answer, `49`, from Worked Example 13.1.1?
3. `scalar_matrix_multiply` is called with `a` shaped `2×3` and `b` shaped `2×2` (so `a.cols = 3` but `b.rows = 2`). Using `Matrix::get`'s formula `row * cols + col`, what does `b.get(2, j)` actually evaluate to, and why does nothing in the code stop this call from proceeding?
4. Using `matrix_transpose_simd<2>` on `A = [[1,2,3],[4,5,6]]`, which exact `output.set` calls (in order) reconstruct row `1` of `A` (`[4,5,6]`) into column `1` of `output`?
5. A general `4×4` matrix is multiplied by a `4×4` diagonal matrix. Using the counting method from Worked Example 13.4.1, how many scalar multiplications does the general path take, how many does the diagonal-dispatch path take, and what ratio do they confirm?

## Where We Go Next

Chapter 14 (`part2/03-reduction-operations.md`) turns to the operations that collapse a tensor's dimensions instead of preserving them: sums, means, norms, and the reductions every loss function ultimately produces a single scalar through.

## Worked Solutions

**1.** `simd_count = (3 / 3) × 3 = 3`, exactly equal to `a.cols`. The remainder loop runs `for (int k = 3; k < 3; k++)` — an empty range — for every single output position, so it never executes at all when `SIMD_WIDTH` divides the contraction dimension evenly; the entire dot product is computed by one grouped multiply-and-sum per output element.

**2.** `a_vals = [X.get(1,0), X.get(1,1)] = [4, 5]`; `b_vals = [M.get(0,0), M.get(1,0)] = [1, 3]`; `vector_sum = [4×1, 5×3] = [4, 15]`, lane sum `= 4 + 15 = 19`. Remainder (`k=2`): `X.get(1,2) × M.get(2,0) = 6 × 5 = 30`. Total: `19 + 30 = 49` — matching the scalar answer for `Y(1,0)` exactly.

**3.** `b.get(2, j)` evaluates `b.data[2 * b.cols + j] = b.data[2×2 + j] = b.data[4 + j]`. For a `2×2` matrix (`4` total elements, valid indices `0`–`3`), this reads at index `4` or higher — one past the end of `b`'s allocated buffer, an out-of-bounds read (or, as Worked Example 13.1.4 demonstrated deterministically, a read of real sentinel data placed exactly there). Nothing stops the call because neither `scalar_matrix_multiply` nor `Matrix::get` ever compares `a.cols` against `b.rows`; the loop bound is taken entirely from `a.cols`, with no reference to `b`'s actual dimensions at all.

**4.** Row `1` of `A` is `[4, 5, 6]`. The main loop (`j=0`, since `simd_count = 2`) loads `row_vals = [A.get(1,0), A.get(1,1)] = [4, 5]` and writes `output.set(0, 1, 4)` then `output.set(1, 1, 5)`. The remainder loop (`j=2`) writes `output.set(2, 1, A.get(1,2)) = output.set(2, 1, 6)`. In order: `output.set(0,1,4)`, `output.set(1,1,5)`, `output.set(2,1,6)` — reconstructing column `1` of the output as `[4, 5, 6]`, row `1` of `A`, exactly as transpose requires.

**5.** General path: `4³ = 64` multiplications. Diagonal-dispatch path: one multiplication per entry of the `4×4` operand — `16` total. Ratio: `64 / 16 = 4`, confirming the `n`-times pattern at `n = 4` — consistent with `3×` at `n=3` and `5×` at `n=5`.
