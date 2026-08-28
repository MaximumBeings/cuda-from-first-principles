# Chapter 9: Specialized Tensor Types — Identity, Diagonal, Sparse, and Triangular

> "Chapter 6's `Tensor` stores every element it claims to hold — a fair, general-purpose deal, and a wasteful one the moment most of those elements are predictable, or absent, or mirror images of each other. This chapter is four increasingly aggressive answers to the same question: how much of a shape's data can a struct simply *not store*, computing or skipping it instead — and one genuine, discovered bug that shows exactly what goes wrong when the arithmetic behind 'skip it' is wrong by one dimension's worth of assumption."

**What you will understand by the end of this chapter:**

- `IdentityView`: a struct holding no data at all, whose `sizeof` never grows no matter how large a matrix it represents, because every one of its values is a rule (`row == col`) rather than a stored fact
- `TridiagonalView`: storage proportional to the *number of diagonals actually present* (three, for a tridiagonal matrix) rather than to the matrix's total element count — and the exact sparsity fraction that follows from it
- `SparseCOO`: a coordinate-triplet format where "adding a zero" is a genuine no-op, and where a value that *becomes* zero later can leave a real, wasted, discoverable slot behind until something explicitly compresses it away
- Packed triangular storage, and a genuine index-collision bug this chapter discovers by testing: applying the wrong formula's packing order corrupts real data, silently, in a way a naive test can miss

**What you need to know first:**

- Chapter 6.4 (the offset formula, and its complete absence of bounds checking) — every specialized type in this chapter replaces that formula with its own, more specific rule
- Chapter 3.4's cross-check discipline and Chapter 8.2–8.3's habit of testing code against the specific input that breaks it, not just the input that doesn't — Section 9.4 finds this chapter's bug exactly that way
- Chapter 7.4 (broadcasting via a zero stride) — conceptually the closest thing this book has already built to "many logical positions, one small backing store," which this chapter's `IdentityView` and `SparseCOO` both extend further
- If you've read the Mojo edition: this chapter follows its Chapter 9 section-for-section (identity, diagonal, sparse, triangular), including its practice of finding a genuine bug in the triangular section rather than describing a bug hypothetically — the specific collision differs because it's driven by this chapter's own from-scratch CUDA C++ formula rather than Mojo's.

## 9.1 Identity Tensors: When the Data *Is* the Rule `[FOUNDATIONAL]`

### Intuition

An identity matrix's every value is entirely determined by one comparison — `row == col` — with no information left over that actually needs remembering. `IdentityView` takes this literally: it stores nothing but the dimension `n` itself, and computes `1.0` or `0.0` on demand for any coordinate asked of it, the same non-owning, no-allocation spirit as Chapter 7.2's `TensorView`, pushed to its logical extreme — there's no buffer to point *at* here at all.

### Background

| | Chapter 6's dense `Tensor` | `IdentityView` |
|---|---|---|
| Bytes stored, `n × n` identity | `n² × sizeof(float)` — grows with `n²` | A fixed handful of bytes (just the `int n` field) — never grows |
| How `at(row, col)` is computed | A memory read at a precomputed offset | A single comparison, no memory touched at all |
| `cudaMalloc` calls | One, sized by `numel()` | None — there is nothing to allocate |

### Worked Example 9.1.1 — memory that provably does not grow

```cpp
// No data pointer at all -- an n x n identity's every value is a RULE
// ("1 if row==col, else 0"), not something that needs to be stored.
struct IdentityView {
    int n;
    __host__ __device__ IdentityView(int n_) : n(n_) {}
    __host__ __device__ float at(int row, int col) const {
        return (row == col) ? 1.0f : 0.0f;
    }
};
```

Compiled and run as the complete `01_identity_view.cu` further below:

```bash
nvcc -arch=sm_80 01_identity_view.cu -o 01_identity_view
./01_identity_view
```

Genuinely compiled and run:

```
sizeof(IdentityView) = 4 bytes -- IDENTICAL for a 4x4 or a 1,000,000x1,000,000 identity

id_small.at(2,2) = 1.0, id_small.at(2,3) = 0.0
id_huge.at(999999,999999) = 1.0  (computed instantly, no million-squared buffer touched)

a DENSE 4x4 Tensor would need 64 bytes
a DENSE 1,000,000x1,000,000 Tensor would need 4000000000000 bytes (4000.0 GB) -- IdentityView needs 4
```

`sizeof(IdentityView)` is genuinely `4` bytes — one `int` — whether it represents a `4×4` identity or a `1,000,000 × 1,000,000` one; `id_huge.at(999999, 999999)` returns instantly because it's one comparison, not a lookup into a buffer that (at 4 terabytes for a dense equivalent) could never actually be allocated on any real machine. This is the same zero-allocation argument Chapter 7.2 made for views in general, taken to the point where there's no underlying buffer left to view at all.

### Worked Example 9.1.2 — diagonal length and transpose, traced through the rule itself

An identity matrix's "diagonal length" is trivially `min(rows, cols)` — for a non-square `IdentityView` extended to `m × n`, the rule `row == col` can only ever be satisfied for `min(m, n)` positions, since the larger dimension simply runs out of matching partners past that point. Transposing an `IdentityView` is even simpler than Chapter 7.2's general `transpose()`: swapping `row` and `col` inside `row == col` produces the exact same condition, `col == row` — an identity matrix is its own transpose, with nothing to compute or verify beyond noticing the rule itself doesn't change when its two inputs are swapped.

> `[COMMON TRAP]` `IdentityView` genuinely represents an `n × n` matrix, but nothing in its type prevents someone from constructing one with a negative or absurdly large `n` and never actually using it in an operation that would catch the mistake — because `at()` never touches memory, there is no out-of-bounds access for a bad `n` to trigger the way Chapter 6.4's raw offset arithmetic could at least, in principle, be caught by a debug-mode bounds check on a real buffer. A struct with no data to corrupt also has no data whose corruption would ever tell you something is wrong.

## 9.2 Diagonal Tensors: Storage Proportional to What Actually Varies `[FOUNDATIONAL]`

### Intuition

A tridiagonal matrix — nonzero only on its main diagonal and the one diagonal immediately above and below it — is a natural generalization of Section 9.1's identity: instead of *one* rule (`row == col`) covering every stored value, there are *three* rules, one per diagonal, each backed by its own small array. Everywhere else, the matrix is zero by construction, and Section 9.1's argument applies again: a value that's always zero doesn't need storage, it needs a rule that returns zero.

### Background

| Diagonal | Condition | Backing array | Entries for an `n × n` matrix |
|---|---|---|---|
| Main | `row == col` | `main[n]` | `n` |
| Sub (below main) | `row == col + 1` | `sub[n-1]` | `n - 1` |
| Super (above main) | `col == row + 1` | `super[n-1]` | `n - 1` |
| Everywhere else | — | (nothing) | `0`, always |

### Worked Example 9.2.1 — a `5×5` tridiagonal matrix, built and displayed

```cpp
struct TridiagonalView {
    int n;
    float* sub;
    float* main;
    float* super;

    __host__ __device__ float at(int row, int col) const {
        if (row == col) return main[row];
        if (row == col + 1) return sub[col];     // one below the main diagonal
        if (col == row + 1) return super[row];   // one above the main diagonal
        return 0.0f;                              // everywhere else: zero, never stored
    }
};
```

Compiled and run as the complete `02_diagonal_tensor.cu` further below:

```bash
nvcc -arch=sm_80 02_diagonal_tensor.cu -o 02_diagonal_tensor
./02_diagonal_tensor
```

Genuinely compiled and run:

```
5x5 tridiagonal matrix, built by hand:
  [ 4.0  2.0  0.0  0.0  0.0]
  [ 1.0  4.0  2.0  0.0  0.0]
  [ 0.0  1.0  4.0  2.0  0.0]
  [ 0.0  0.0  1.0  4.0  2.0]
  [ 0.0  0.0  0.0  1.0  4.0]

stored elements = 13, dense total = 25, sparsity fraction stored = 52.0%
```

Every row's nonzero pattern shifts one column right as `row` increases, exactly the diagonal-band structure `at()`'s three conditions describe — row `2`'s nonzero entries land at columns `1`, `2`, `3`, matching `sub[1]=1.0` (condition `row==col+1`, i.e. `2==1+1`), `main[2]=4.0`, and `super[2]=2.0` (condition `col==row+1`, i.e. `3==2+1`). `13` stored floats (`4 + 5 + 4`) against `25` for the dense equivalent is `52%` — genuinely more than half, which is worth noticing directly: a tridiagonal matrix isn't *sparse* in the way Section 9.3 means the term, it's *structured*, and for small `n` specifically, three separate arrays plus their own bookkeeping can cost more overhead than it saves.

> `[COMMON TRAP]` `TridiagonalView::at()` reads `sub[col]` (not `sub[row]`) for the sub-diagonal case, and this asymmetry is easy to get backwards. `sub[i]` is defined as the element at `(row=i+1, col=i)`, so given a `(row, col)` pair satisfying `row == col+1`, the correct array index is `col` (equivalently `row-1`) — reaching for `sub[row]` instead would read one element past where the actual value lives, an off-by-one that a test using only the main diagonal's own symmetric indices (where `row` and `col` are close enough not to expose the mistake) could easily miss.

## 9.3 Sparse Tensors: Coordinates Instead of a Grid `[FOUNDATIONAL]`

### Intuition

Section 9.2's diagonals still assume a fixed, predictable *pattern* of nonzero positions. A genuinely sparse matrix — nonzero values scattered with no such pattern — needs a format that stores positions *and* values together, as a flat list of triplets: **COO** (coordinate) format. The one rule that makes COO work at all is Worked Example 9.3.1's: a value of exactly `0.0` is indistinguishable from "never inserted" (`at()` returns `0.0` for both), so inserting a zero would be pure waste — storage spent recording a fact the format already gives away for free.

### Background

| | Dense `Tensor` (Chapter 6) | `SparseCOO` |
|---|---|---|
| Storage per element | Fixed — every position, whether zero or not | Proportional to nonzero count only |
| Reading an absent position | N/A — every position exists | Returns `0.0` by definition (a linear search finds nothing) |
| Reading cost | `O(1)` — direct offset arithmetic | `O(count)` — a linear scan over every stored triplet |
| A value that becomes zero *after* insertion | Overwritten in place, no size change | Still occupies a triplet until something explicitly removes it |

### Worked Example 9.3.1 — inserting a zero adds nothing

```cpp
struct SparseCOO {
    int rows[16], cols[16];
    float vals[16];
    int count;

    __host__ __device__ void set(int row, int col, float value) {
        if (value == 0.0f) return;  // adding a zero doesn't add anything
        rows[count] = row; cols[count] = col; vals[count] = value;
        count++;
    }

    __host__ __device__ float at(int row, int col) const {
        for (int k = 0; k < count; k++)
            if (rows[k] == row && cols[k] == col) return vals[k];
        return 0.0f;  // not found -- absent means zero
    }
};
```

Compiled and run as part of the complete `03_sparse_coo.cu` further below:

```bash
nvcc -arch=sm_80 03_sparse_coo.cu -o 03_sparse_coo
./03_sparse_coo
```

Genuinely compiled and run:

```
after 4 set() calls (one was a zero, on a never-before-touched position):
  count = 3  (only the 3 genuinely nonzero inserts)
  at(0,0) = 5.0, at(2,3) = 7.0, at(4,4) = -2.0, at(1,1) [never inserted] = 0.0
```

Four calls to `set()`, only three of which actually pass the `value == 0.0f` check — `set(1, 1, 0.0f)` returns immediately, leaving `count` at `3`, not `4`. Reading `at(1, 1)` afterward still correctly returns `0.0`, not because a triplet says so, but because the linear search in `at()` finds nothing at `(1,1)` and falls through to its own `return 0.0f` — "never inserted" and "explicitly zero" are, by design, completely indistinguishable from the outside, which is exactly what makes skipping the insert safe.

### Worked Example 9.3.2 — `compress_storage()`, and the wasted slot it exists to remove

```cpp
__host__ __device__ void overwrite(int row, int col, float value) {
    for (int k = 0; k < count; k++)
        if (rows[k] == row && cols[k] == col) { vals[k] = value; return; }
}

__host__ __device__ int compress_storage() {
    int write = 0;
    for (int read = 0; read < count; read++) {
        if (vals[read] != 0.0f) {
            rows[write] = rows[read]; cols[write] = cols[read]; vals[write] = vals[read];
            write++;
        }
    }
    int removed = count - write;
    count = write;
    return removed;
}
```

Compiled and run as the complete `03_sparse_coo.cu` further below:

```bash
nvcc -arch=sm_80 03_sparse_coo.cu -o 03_sparse_coo
./03_sparse_coo
```

Genuinely compiled and run:

```
after overwrite(2,3, 0.0) -- the entry still EXISTS, just holds 0.0:
  count = 3  (still 3 -- overwrite() never removes a triplet)
  at(2,3) = 0.0  (reads back correctly, but wastes a storage slot)

compress_storage() removed 1 wasted triplet(s); count now = 2
  at(2,3) after compression = 0.0  (still correctly zero -- absence still means zero)
  at(0,0) after compression = 5.0, at(4,4) after compression = -2.0  (untouched entries preserved)
```

`overwrite(2, 3, 0.0f)` is a fundamentally different operation from `set()`: it finds the *existing* triplet at `(2,3)` (inserted earlier with value `7.0`) and changes its value in place, with no check against zero at all — `compress_storage()` exists specifically to clean up exactly this situation, one step later, rather than trying to prevent it up front. `at(2,3)` reads correctly as `0.0` both before and after compression, for two different reasons each time: before compression, because a real triplet happens to hold the value `0.0`; after, because no triplet exists there at all and the search falls through — the caller can't tell the difference, and Section 9.3's whole design depends on that being true.

> `[COMMON TRAP]` `compress_storage()` shifts every kept triplet's array position, which silently invalidates any index into `rows`/`cols`/`vals` that code elsewhere might have cached from before the call — a bug this chapter's own `at()` never falls into (it always searches fresh) but a hypothetical "remember which slot I found `(2,3)` at, to skip the search next time" optimization absolutely would. Any format that periodically reorganizes its own storage needs every caller to treat cached positions as invalidated by that reorganization, not merely by an explicit removal.

## 9.4 Triangular Tensors: Packed Storage, and a Genuine Collision `[FOUNDATIONAL]`

### Intuition

An upper-triangular `n × n` matrix (nonzero only where `col ≥ row`) has exactly `n(n+1)/2` real entries — almost half of `n²` simply doesn't exist. Packing those entries into a flat array of exactly that size needs a formula mapping each valid `(row, col)` pair to a distinct flat index, and getting that formula wrong doesn't necessarily *crash* — it can just as easily hand two different, valid coordinates the *same* index, silently overwriting one with the other. This section derives the correct formula, but only after testing a plausible-looking wrong one and watching it genuinely corrupt data.

### Background

| | Formula | Where it comes from |
|---|---|---|
| **Buggy**: `row×(row+1)/2 + col` | This is the packed-index formula for *lower*-triangular storage, applied here by mistake | Each row `r` in *lower*-triangular storage holds `r+1` entries (columns `0` through `r`); the formula accumulates that growing count |
| **Correct** (upper-triangular): `row×n - row×(row-1)/2 + (col - row)` | Each row `r` in *upper*-triangular storage holds `n-r` entries (columns `r` through `n-1`) — a *shrinking* count, the opposite shape from the buggy formula's assumption | Skips the entries already used by every earlier, longer row, then adds how far into the current row `(row,col)` sits |

### Worked Example 9.4.1 — the buggy formula, and where it collides

```cpp
// BUGGY: this is the LOWER-triangular packed-index formula, mistakenly
// applied to UPPER-triangular storage.
__host__ __device__ int buggy_upper_index(int row, int col, int n) {
    return row * (row + 1) / 2 + col;
}
```

Compiled and run as part of the complete `04_triangular_packed_bug.cu` further below:

```bash
nvcc -arch=sm_80 04_triangular_packed_bug.cu -o 04_triangular_packed_bug
./04_triangular_packed_bug
```

Genuinely compiled and run:

```
writing with buggy_upper_index (row<=col only):
  (0,0) -> index 0, writing 0.0
  (0,1) -> index 1, writing 1.0
  (0,2) -> index 2, writing 2.0
  (0,3) -> index 3, writing 3.0
  (1,1) -> index 2, writing 11.0
  (1,2) -> index 3, writing 12.0
  (1,3) -> index 4, writing 13.0
  (2,2) -> index 5, writing 22.0
  (2,3) -> index 6, writing 23.0
  (3,3) -> index 9, writing 33.0
```

`(0,2)` writes to index `2`; two writes later, `(1,1)` *also* computes index `2` and overwrites it. `(0,3)` writes to index `3`; `(1,2)` collides there next. Neither collision announces itself at write time — both writes succeed, and the array accepts both values, one silently replacing the other.

### Worked Example 9.4.2 — the collision, traced through an actual read-back

Compiled and run as part of the complete `04_triangular_packed_bug.cu` further below:

```bash
nvcc -arch=sm_80 04_triangular_packed_bug.cu -o 04_triangular_packed_bug
./04_triangular_packed_bug
```

Genuinely compiled and run:

```
reading back with the SAME buggy formula:
  (0,0) -> index 0, read 0.0, expected 0.0
  (0,1) -> index 1, read 1.0, expected 1.0
  (0,2) -> index 2, read 11.0, expected 2.0  <- CORRUPTED
  (0,3) -> index 3, read 12.0, expected 3.0  <- CORRUPTED
  (1,1) -> index 2, read 11.0, expected 11.0
  (1,2) -> index 3, read 12.0, expected 12.0
  (1,3) -> index 4, read 13.0, expected 13.0
  (2,2) -> index 5, read 22.0, expected 22.0
  (2,3) -> index 6, read 23.0, expected 23.0
  (3,3) -> index 9, read 33.0, expected 33.0

any corrupted entries with the buggy formula? true

writing with correct_upper_index:
all 10 entries read back correctly with the fixed formula? true
```

`(1,1)` and `(1,2)` read back *correctly* — `11.0` and `12.0` — for a genuinely misleading reason: they were the *last* writers to indices `2` and `3`, so reading through the same buggy formula that wrote them recovers exactly what they wrote, hiding the fact that anything went wrong at all. `(0,2)` and `(0,3)` expose the real damage — their own values, `2.0` and `3.0`, are simply gone, silently replaced by `(1,1)`'s and `(1,2)'s`. A test that only checked "does reading back what I just wrote, in the same order, give the same answer" for a *single* coordinate at a time would never catch this — the bug only shows up when two *different* coordinates are checked against each other, which is exactly why Worked Example 9.4.1's write loop populates every valid pair before Worked Example 9.4.2's read loop checks every one of them again.

> `[COMMON TRAP]` The two formulas differ by exactly one structural assumption — whether each successive row holds *more* entries (lower-triangular, `row+1`, `row+2`, ...) or *fewer* (upper-triangular, `n-row`, `n-row-1`, ...) — and copying a packed-storage formula from a reference that uses the opposite triangular convention is a completely reasonable, easy mistake, not a typo. Both formulas are individually well-formed, valid-looking arithmetic; the only way to catch the difference is to test against multiple coordinates whose *stored order* the two conventions actually disagree on, exactly as this section's worked examples did.

### Worked Example 9.4.3 — storage efficiency, correctly measured

Compiled and run as the complete `04_triangular_packed_bug.cu` further below:

```bash
nvcc -arch=sm_80 04_triangular_packed_bug.cu -o 04_triangular_packed_bug
./04_triangular_packed_bug
```

Genuinely compiled and run:

```
n=4, packed storage size = n(n+1)/2 = 10 (vs. 16 for a dense n x n)
```

`10/16 = 62.5%` of the dense storage — meaningfully less, and this fraction only improves as `n` grows, since `n(n+1)/2` grows roughly half as fast as `n²`: for `n=100`, packed storage is `5050` floats against `10000` for dense, a genuine `50.5%`, approaching the theoretical `50%` limit Section 9.4's structure implies as `n` grows without bound.

## 9.5 Complete Runnable Code

### File: `01_identity_view.cu`

```cpp
#include <cstdio>

// No data pointer at all -- an n x n identity's every value is a RULE
// ("1 if row==col, else 0"), not something that needs to be stored.
struct IdentityView {
    int n;
    __host__ __device__ IdentityView(int n_) : n(n_) {}
    __host__ __device__ float at(int row, int col) const {
        return (row == col) ? 1.0f : 0.0f;
    }
};

int main() {
    IdentityView id_small(4);
    IdentityView id_huge(1000000);  // a million x a million -- no allocation attempted

    printf("sizeof(IdentityView) = %zu bytes -- IDENTICAL for a 4x4 or a 1,000,000x1,000,000 identity\n",
           sizeof(IdentityView));

    printf("\nid_small.at(2,2) = %.1f, id_small.at(2,3) = %.1f\n", id_small.at(2, 2), id_small.at(2, 3));
    printf("id_huge.at(999999,999999) = %.1f  (computed instantly, no million-squared buffer touched)\n",
           id_huge.at(999999, 999999));

    long long dense_bytes_small = (long long)4 * 4 * sizeof(float);
    long long dense_bytes_huge  = (long long)1000000 * 1000000 * sizeof(float);
    printf("\na DENSE 4x4 Tensor would need %lld bytes\n", dense_bytes_small);
    printf("a DENSE 1,000,000x1,000,000 Tensor would need %lld bytes (%.1f GB) -- IdentityView needs %zu\n",
           dense_bytes_huge, dense_bytes_huge / 1e9, sizeof(IdentityView));
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_identity_view.cu -o 01_identity_view
./01_identity_view
```

Produces exactly the output shown in Worked Example 9.1.1 above.

### File: `02_diagonal_tensor.cu`

```cpp
#include <cstdio>

// Stores only the 3 diagonals of an n x n tridiagonal matrix -- 3n floats
// instead of n^2, reading zero for every position not on one of the 3 bands.
struct TridiagonalView {
    int n;
    float* sub;    // sub[i]   = element at (row=i+1, col=i)   -- n-1 entries
    float* main;   // main[i]  = element at (row=i,   col=i)   -- n entries
    float* super;  // super[i] = element at (row=i,   col=i+1) -- n-1 entries

    __host__ __device__ float at(int row, int col) const {
        if (row == col) return main[row];
        if (row == col + 1) return sub[col];     // one below the main diagonal
        if (col == row + 1) return super[row];   // one above the main diagonal
        return 0.0f;                              // everywhere else: zero, never stored
    }
};

int main() {
    const int n = 5;
    float sub[4]   = {1.0f, 1.0f, 1.0f, 1.0f};       // all -1-position entries
    float main[5]  = {4.0f, 4.0f, 4.0f, 4.0f, 4.0f}; // main diagonal
    float super[4] = {2.0f, 2.0f, 2.0f, 2.0f};       // all +1-position entries

    TridiagonalView tri{n, sub, main, super};

    printf("5x5 tridiagonal matrix, built by hand:\n");
    for (int row = 0; row < n; row++) {
        printf("  [");
        for (int col = 0; col < n; col++) {
            printf("%4.1f", tri.at(row, col));
            if (col < n - 1) printf(" ");
        }
        printf("]\n");
    }

    int stored = (n - 1) + n + (n - 1);  // sub + main + super
    int total = n * n;
    printf("\nstored elements = %d, dense total = %d, sparsity fraction stored = %.1f%%\n",
           stored, total, 100.0f * stored / total);
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_diagonal_tensor.cu -o 02_diagonal_tensor
./02_diagonal_tensor
```

Produces exactly the output shown in Worked Example 9.2.1 above.

### File: `03_sparse_coo.cu`

```cpp
#include <cstdio>

// COO (coordinate) format: parallel arrays of (row, col, value) triplets --
// only entries someone has actually inserted exist at all.
struct SparseCOO {
    static const int CAPACITY = 16;
    int rows[CAPACITY];
    int cols[CAPACITY];
    float vals[CAPACITY];
    int count;

    __host__ __device__ SparseCOO() : count(0) {}

    // Setting a value to exactly 0.0 does NOT insert a triplet -- there is
    // nothing to add, since "absent" already means zero for every position.
    __host__ __device__ void set(int row, int col, float value) {
        if (value == 0.0f) return;  // adding a zero doesn't add anything
        rows[count] = row; cols[count] = col; vals[count] = value;
        count++;
    }

    __host__ __device__ float at(int row, int col) const {
        for (int k = 0; k < count; k++) {
            if (rows[k] == row && cols[k] == col) return vals[k];
        }
        return 0.0f;  // not found -- absent means zero
    }

    // Overwrite an existing entry's value in place, WITHOUT removing it even
    // if the new value happens to be zero -- this is what leaves a real,
    // genuinely wasted triplet behind for compress_storage() to find.
    __host__ __device__ void overwrite(int row, int col, float value) {
        for (int k = 0; k < count; k++) {
            if (rows[k] == row && cols[k] == col) { vals[k] = value; return; }
        }
    }

    // Remove every triplet whose value has become exactly zero (e.g. via
    // overwrite() above), compacting the arrays in place.
    __host__ __device__ int compress_storage() {
        int write = 0;
        for (int read = 0; read < count; read++) {
            if (vals[read] != 0.0f) {
                rows[write] = rows[read];
                cols[write] = cols[read];
                vals[write] = vals[read];
                write++;
            }
        }
        int removed = count - write;
        count = write;
        return removed;
    }
};

int main() {
    SparseCOO m;
    m.set(0, 0, 5.0f);
    m.set(2, 3, 7.0f);
    m.set(1, 1, 0.0f);  // "setting" a zero on a position NEVER touched before
    m.set(4, 4, -2.0f);

    printf("after 4 set() calls (one was a zero, on a never-before-touched position):\n");
    printf("  count = %d  (only the 3 genuinely nonzero inserts)\n", m.count);
    printf("  at(0,0) = %.1f, at(2,3) = %.1f, at(4,4) = %.1f, at(1,1) [never inserted] = %.1f\n",
           m.at(0, 0), m.at(2, 3), m.at(4, 4), m.at(1, 1));

    // Now genuinely zero out an EXISTING entry via overwrite(), leaving a
    // real, wasted triplet behind on purpose.
    m.overwrite(2, 3, 0.0f);
    printf("\nafter overwrite(2,3, 0.0) -- the entry still EXISTS, just holds 0.0:\n");
    printf("  count = %d  (still 3 -- overwrite() never removes a triplet)\n", m.count);
    printf("  at(2,3) = %.1f  (reads back correctly, but wastes a storage slot)\n", m.at(2, 3));

    int removed = m.compress_storage();
    printf("\ncompress_storage() removed %d wasted triplet(s); count now = %d\n", removed, m.count);
    printf("  at(2,3) after compression = %.1f  (still correctly zero -- absence still means zero)\n", m.at(2, 3));
    printf("  at(0,0) after compression = %.1f, at(4,4) after compression = %.1f  (untouched entries preserved)\n",
           m.at(0, 0), m.at(4, 4));
    return 0;
}
```

```bash
nvcc -arch=sm_80 03_sparse_coo.cu -o 03_sparse_coo
./03_sparse_coo
```

Produces exactly the output shown in Worked Examples 9.3.1 and 9.3.2 above.

### File: `04_triangular_packed_bug.cu`

```cpp
#include <cstdio>

// BUGGY: this is the LOWER-triangular packed-index formula, mistakenly
// applied to UPPER-triangular storage.
__host__ __device__ int buggy_upper_index(int row, int col, int n) {
    return row * (row + 1) / 2 + col;
}

// CORRECT upper-triangular packed index (valid for row <= col): skip the
// (row * n - row*(row-1)/2) elements already used by every earlier row,
// which shrinks by one each row since row 0 stores n entries, row 1 stores
// n-1, etc. -- then add how far into THIS row (row, col) sits.
__host__ __device__ int correct_upper_index(int row, int col, int n) {
    return row * n - row * (row - 1) / 2 + (col - row);
}

int main() {
    const int n = 4;
    const int packed_size = n * (n + 1) / 2;  // 10 for n=4

    printf("n=%d, packed storage size = n(n+1)/2 = %d (vs. %d for a dense n x n)\n\n", n, packed_size, n*n);

    // Write a distinct, easily recognizable value to every valid (row,col)
    // pair using the BUGGY formula, then try to read every pair back.
    float buggy_storage[packed_size] = {0};
    printf("writing with buggy_upper_index (row<=col only):\n");
    for (int row = 0; row < n; row++) {
        for (int col = row; col < n; col++) {
            float value = row * 10 + col;  // e.g. (1,2) -> 12.0, easy to recognize
            int idx = buggy_upper_index(row, col, n);
            printf("  (%d,%d) -> index %d, writing %.1f\n", row, col, idx, value);
            buggy_storage[idx] = value;
        }
    }
    printf("\nreading back with the SAME buggy formula:\n");
    bool any_wrong = false;
    for (int row = 0; row < n; row++) {
        for (int col = row; col < n; col++) {
            float expected = row * 10 + col;
            int idx = buggy_upper_index(row, col, n);
            float got = buggy_storage[idx];
            bool wrong = (got != expected);
            if (wrong) any_wrong = true;
            printf("  (%d,%d) -> index %d, read %.1f, expected %.1f%s\n",
                   row, col, idx, got, expected, wrong ? "  <- CORRUPTED" : "");
        }
    }
    printf("\nany corrupted entries with the buggy formula? %s\n", any_wrong ? "true" : "false");

    // Now the FIX: same test, correct formula.
    float fixed_storage[packed_size] = {0};
    printf("\nwriting with correct_upper_index:\n");
    for (int row = 0; row < n; row++) {
        for (int col = row; col < n; col++) {
            float value = row * 10 + col;
            int idx = correct_upper_index(row, col, n);
            fixed_storage[idx] = value;
        }
    }
    bool all_correct = true;
    for (int row = 0; row < n; row++) {
        for (int col = row; col < n; col++) {
            float expected = row * 10 + col;
            int idx = correct_upper_index(row, col, n);
            if (fixed_storage[idx] != expected) all_correct = false;
        }
    }
    printf("all %d entries read back correctly with the fixed formula? %s\n", packed_size, all_correct ? "true" : "false");

    return 0;
}
```

```bash
nvcc -arch=sm_80 04_triangular_packed_bug.cu -o 04_triangular_packed_bug
./04_triangular_packed_bug
```

### Expected Output for `04_triangular_packed_bug.cu`

```
n=4, packed storage size = n(n+1)/2 = 10 (vs. 16 for a dense n x n)

writing with buggy_upper_index (row<=col only):
  (0,0) -> index 0, writing 0.0
  (0,1) -> index 1, writing 1.0
  (0,2) -> index 2, writing 2.0
  (0,3) -> index 3, writing 3.0
  (1,1) -> index 2, writing 11.0
  (1,2) -> index 3, writing 12.0
  (1,3) -> index 4, writing 13.0
  (2,2) -> index 5, writing 22.0
  (2,3) -> index 6, writing 23.0
  (3,3) -> index 9, writing 33.0

reading back with the SAME buggy formula:
  (0,0) -> index 0, read 0.0, expected 0.0
  (0,1) -> index 1, read 1.0, expected 1.0
  (0,2) -> index 2, read 11.0, expected 2.0  <- CORRUPTED
  (0,3) -> index 3, read 12.0, expected 3.0  <- CORRUPTED
  (1,1) -> index 2, read 11.0, expected 11.0
  (1,2) -> index 3, read 12.0, expected 12.0
  (1,3) -> index 4, read 13.0, expected 13.0
  (2,2) -> index 5, read 22.0, expected 22.0
  (2,3) -> index 6, read 23.0, expected 23.0
  (3,3) -> index 9, read 33.0, expected 33.0

any corrupted entries with the buggy formula? true

writing with correct_upper_index:
all 10 entries read back correctly with the fixed formula? true
```

Every number here was independently verified earlier in this chapter, in Worked Examples 9.4.1 through 9.4.3. All four files in this section genuinely compile clean under `nvcc -arch=sm_80` and run to completion in this no-GPU environment — none of them call a single CUDA API function, since every specialized type in this chapter is pure struct-and-arithmetic logic, checkable identically on host or device.

## Chapter Summary

Four specialized tensor types, each skipping storage for a different, increasingly general reason. `IdentityView` stores nothing at all, because every value follows from one comparison rather than needing to be remembered — `sizeof` stays constant no matter how large a matrix it represents. `TridiagonalView` stores exactly the diagonals a banded matrix actually has, at `52%` of dense storage for a small `5×5` case — genuinely less than dense, though not dramatically so until `n` grows much larger. `SparseCOO` stores only nonzero values as explicit coordinate triplets, with "adding a zero" a deliberate no-op and a value that *becomes* zero later left behind as a real, discoverable wasted slot until `compress_storage()` removes it. And packed triangular storage's `n(n+1)/2` formula only works when its row-length assumption — growing, for lower-triangular, or shrinking, for upper-triangular — matches the actual layout being packed; this chapter found, by testing multiple coordinates against each other rather than just re-reading what was just written, that the wrong assumption produces genuine, silent index collisions, with two different valid coordinates corrupting each other's stored values.

## Self-Check Questions

1. For a `6×6` identity matrix, what is `sizeof(IdentityView)`, and how does that number change for a `600×600` identity?
2. A pentadiagonal matrix (5 diagonals: two below main, main, two above) of size `n×n` stores how many floats total, as a formula in `n`? At what point does this formula's stored fraction of `n²` cross below 10%?
3. `SparseCOO::set(3, 3, 0.0f)` is called on a matrix that has never touched position `(3,3)` before. What does `count` become, and what does `at(3,3)` return immediately afterward?
4. Using `correct_upper_index(row, col, n) = row×n - row×(row-1)/2 + (col-row)`, compute the packed index for `(row=2, col=3)` in a `5×5` upper-triangular matrix, and verify it doesn't collide with any other valid pair's index by checking `(row=0, col=3)` and `(row=1, col=3)` too.
5. Worked Example 9.4.2 noted that `(1,1)` and `(1,2)` read back *correctly* despite the buggy formula, while `(0,2)` and `(0,3)` did not. Explain what property of the write order (not the coordinates themselves) determined which pairs survived and which didn't.

## Where We Go Next

Every tensor type this chapter built is read-only in spirit — a rule, a small set of bands, a sparse list, or a packed triangle, each answering "what value lives here?" without yet doing any *arithmetic* between two tensors. Part 2 starts exactly there: element-wise operations, the first place two tensors' values genuinely combine into a third.

## Worked Solutions

**1.** `sizeof(IdentityView)` is `4` bytes (one `int` field) for a `6×6` identity, and remains exactly `4` bytes for a `600×600` identity — it never changes, because `IdentityView` stores only the dimension itself, never the matrix's contents.

**2.** Two below main, main, two above main means 5 diagonals: lengths `n-2, n-1, n, n-1, n-2`, summing to `5n - 6`. As a fraction of `n²`: `(5n-6)/n²`. Setting `(5n-6)/n² < 0.10` and solving approximately: `5n - 6 < 0.1n²`, or `0.1n² - 5n + 6 > 0`. Using the quadratic formula, this crosses zero near `n ≈ 48.8`; for `n = 50`, stored `= 5(50)-6 = 244`, dense `= 2500`, fraction `= 9.76%` — just under 10%, confirming the crossing point lands close to `n ≈ 49`.

**3.** `count` stays at whatever it was before the call — `set()`'s `if (value == 0.0f) return;` check exits before touching `count`, `rows`, `cols`, or `vals` at all. `at(3,3)` returns `0.0f` immediately afterward, for the same reason every never-inserted position does: the linear search in `at()` finds no triplet at `(3,3)` and falls through to its own default.

**4.** `correct_upper_index(2, 3, 5) = 2×5 - 2×1/2 + (3-2) = 10 - 1 + 1 = 10`. For `(0,3)`: `0×5 - 0 + (3-0) = 3`. For `(1,3)`: `1×5 - 1×0/2 + (3-1) = 5 - 0 + 2 = 7`. All three indices — `10`, `3`, `7` — are distinct, consistent with the correct formula's design guarantee that every valid `(row,col)` pair for a fixed `n` maps to its own unique slot in the `n(n+1)/2`-sized packed array.

**5.** The surviving pairs, `(1,1)` and `(1,2)`, were each the *last* write to their respective colliding index in Worked Example 9.4.1's write order — `(1,1)` was written after `(0,2)` at index `2`, and `(1,2)` was written after `(0,3)` at index `3`. Reading through the same buggy formula afterward simply recovers whatever the *most recent* write left behind at each index; the pairs that got overwritten, `(0,2)` and `(0,3)`, were the earlier writers at their shared indices, not written to again afterward. Nothing about the coordinates' own values determines survival — only which write happened later in the loop that populated the array.
