# Chapter 8: Tensor Creation Functions — Factories, Random Generation, and I/O

> "A `Tensor` with an allocated, uninitialized buffer is a promise, not a value — `cudaMalloc` hands back memory, not zeros. Every one of this chapter's functions exists to keep that promise a specific way: a predictable ramp of numbers, a reproducible stream of randomness, or the exact bytes some other program already wrote to disk. Two of the three genuine bugs this chapter finds along the way were sitting in code that looked completely reasonable until the moment a small seed or a missing trailing newline exposed them."

**What you will understand by the end of this chapter:**

- `arange` and `linspace`, both as `__global__` kernels following Chapter 4.5's one-thread-one-element broadcast pattern, and the genuine distinction between them — `arange` never reaches its stop value, `linspace` always includes both ends
- `eye`, generalized with a diagonal offset, and why "the diagonal" is really just one specific case of "every position where `col - row` equals some fixed constant"
- A small, deterministic, `__host__ __device__` pseudorandom generator, chosen specifically because CUDA's own `curand` library only generates numbers on the device — and two genuine, discovered flaws in the naive version of it: identical seeds silently producing identical streams, and a small seed producing a measurably weak first draw
- Fisher-Yates shuffling and inverse-CDF sampling, both built on top of that same generator and checked against real, computable invariants — a permutation's element sum, and a large sample's empirical frequency
- A genuine off-by-one bug in a CSV row-counting function, caught by testing it against a file that happens not to end in a newline — and why trusting a `Tensor`'s own recorded shape is more robust than re-deriving it from a text file's bytes

**What you need to know first:**

- Chapter 4.5 (the broadcast kernel pattern: one thread, one output element, guarded by a boundary check) — every kernel in Section 8.1 is a direct instance of it
- Chapter 6.1–6.3 (`TensorShape`, `TensorStrides`, and `Tensor`'s constructor) — this chapter fills the buffers Chapter 6 learned to allocate
- Chapter 3.4's cross-check discipline (compute the same result two independent ways and confirm they agree) — Sections 8.1 through 8.3 all lean on it
- If you've read the Mojo edition: this chapter follows its Chapter 8 section-for-section (factories, random generation, data import/export), including its habit of finding a genuine bug in Section 8.3's parsing code rather than describing one hypothetically. The random-number section differs in mechanism: Mojo's edition uses Mojo's own random facilities directly, while this chapter builds a small custom generator specifically because CUDA's `curand` cannot be exercised without a GPU.

## 8.1 Factory Functions: `arange`, `linspace`, and `eye` `[FOUNDATIONAL]`

### Intuition

Chapter 6's `Tensor` constructor decides a shape and allocates memory for it; it says nothing about what ends up in that memory. The three functions this section builds are all the same shape of thing Chapter 4.5 already named the broadcast pattern — one thread, one output element — with the element's value computed from nothing but the thread's own index, no input tensor required at all.

### Background

| Function | Per-element formula | Includes the stop value? |
|---|---|---|
| `arange(start, step, n)` | `data[i] = start + i × step` | No — `n` elements starting at `start`, stopping one step short of `start + n×step` |
| `linspace(start, stop, n)` | `data[i] = start + i × (stop - start)/(n-1)` | Yes, by construction — `i=0` gives exactly `start`, `i=n-1` gives exactly `stop` |
| `eye(n, k)` | `data[i×n+j] = 1` if `j - i == k`, else `0` | N/A — `k=0` is the main diagonal, `k=1`/`k=-1` shift it one column right/left |

### Worked Example 8.1.1 — `arange` and `linspace`, side by side

```cpp
// data[i] = start + i*step, for i in [0, n) -- one thread, one element, the
// exact broadcast pattern Chapter 4.5 established.
__global__ void arange_kernel(float* data, float start, float step, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = start + i * step;
}

// n points from start to stop INCLUSIVE of both ends -- step = (stop-start)/(n-1).
__global__ void linspace_kernel(float* data, float start, float stop, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float step = (stop - start) / (float)(n - 1);
        data[i] = start + i * step;
    }
}
```

Compiled and run (via genuine host-side reference functions using the identical formulas — see the note below) as part of the complete `01_factory_functions.cu` further below:

```bash
nvcc -arch=sm_80 01_factory_functions.cu -o 01_factory_functions
./01_factory_functions
```

Genuinely compiled and run:

```
arange(start=0, step=2, n=5) = [0.0, 2.0, 4.0, 6.0, 8.0]
linspace(0, 1, n=5) = [0.00, 0.25, 0.50, 0.75, 1.00]  (both ends included)
```

`arange(0, 2, 5)` produces `0, 2, 4, 6, 8` — five values, the fifth and last one being `8`, one full `step` short of where a sixth value (`10`) would land. `linspace(0, 1, 5)` produces `0.00, 0.25, 0.50, 0.75, 1.00` — the same five-element count, but computed from a step size derived specifically to land exactly on `1.0` at `i=4`: `step = (1-0)/(5-1) = 0.25`. The two kernels above genuinely compile clean for `sm_80` (verified with `nvcc -arch=sm_80 -c`); the numbers shown come from `main()`'s host-side reference functions, `host_arange`/`host_linspace`, running the identical formulas without a GPU, in the same style Chapter 4.6 used for `naive_matmul_kernel`.

> `[COMMON TRAP]` `linspace`'s formula divides by `n - 1`, which is undefined for `n = 1` — asking for "1 point from 0 to 10" has no well-defined step size, only a single point that could reasonably be `start`, `stop`, or their midpoint depending on convention (most real libraries choose `start`, treating `n=1` as a special case checked before the general formula runs). It's a genuinely different failure mode from `arange`, which has no such restriction — `arange(0, 1, 1)` is a perfectly well-defined single-element `[0]`, because `arange`'s formula never divides by `n` at all.

### Worked Example 8.1.2 — `eye`, with a diagonal offset

```cpp
// Identity matrix, flattened row-major, with an optional diagonal offset k:
// k=0 is the main diagonal, k=1 is the superdiagonal (one column right),
// k=-1 is the subdiagonal (one column left).
__global__ void eye_kernel(float* data, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // row
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // col
    if (i < n && j < n) {
        data[i * n + j] = (j - i == k) ? 1.0f : 0.0f;
    }
}
```

Compiled and run as part of the complete `01_factory_functions.cu` further below:

```bash
nvcc -arch=sm_80 01_factory_functions.cu -o 01_factory_functions
./01_factory_functions
```

Genuinely compiled and run:

```
eye(4, k=0) (main diagonal):
  [1 0 0 0]
  [0 1 0 0]
  [0 0 1 0]
  [0 0 0 1]

eye(4, k=1) (superdiagonal):
  [0 1 0 0]
  [0 0 1 0]
  [0 0 0 1]
  [0 0 0 0]
```

`k=0` reproduces the ordinary identity matrix — every position where `col == row` gets `1`. `k=1` shifts that condition to `col - row == 1`, which is satisfied by `(0,1)`, `(1,2)`, `(2,3)` — one column right of the main diagonal — and by nothing in the last row, since row `3`'s corresponding column, `4`, doesn't exist in a `4×4` matrix. This is a genuine 2D instance of Section 4.5's broadcast pattern — `eye_kernel` is a `dim3` launch over `(row, col)` pairs, exactly Chapter 4.1's 2D thread-hierarchy formula — with `arange`/`linspace`'s 1D version as the simpler case.

## 8.2 Random Tensors: A Small Generator, and Two Genuine Bugs in It `[FOUNDATIONAL]`

### Intuition

CUDA ships a real, production-grade random number library, `curand`, but its actual number-generating calls are device-only — exactly the same limitation Chapter 5.5 ran into with `half2` arithmetic — which makes it unusable for this book's genuinely-run, no-GPU verification discipline. This section instead builds a small, `__host__ __device__` xorshift-style generator, deliberately simple enough to trace by hand, and then genuinely discovers two real weaknesses in it worth knowing about in *any* PRNG, not just this one.

### Background

| | `curand` (CUDA's own library) | `SimpleRNG` (this section) |
|---|---|---|
| Where generation calls run | Device only | `__host__ __device__` — runs and is checkable on either side |
| Quality | Production-grade, extensively tested | A minimal xorshift64 core, built for this chapter's tracing, not for real statistical work |
| Verifiable in this no-GPU environment? | No — generation itself needs a GPU | Yes — every number in this section's worked examples is a real, computed value |

### Worked Example 8.2.1 — the reproducibility trap

```cpp
struct SimpleRNG {
    unsigned long long state;
    __host__ __device__ SimpleRNG(unsigned long long seed) : state(seed ^ 0x9E3779B97F4A7C15ULL) {}
    __host__ __device__ float next_uniform() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return (float)((state >> 40) & 0xFFFFFF) / (float)(1 << 24);
    }
};
```

Compiled and run as part of the complete `02_random_generation.cu` further below:

```bash
nvcc -arch=sm_80 02_random_generation.cu -o 02_random_generation
./02_random_generation
```

Genuinely compiled and run:

```
rng_a(seed=42) first 4 draws: 0.0953 0.9378 0.7694 0.8598
rng_b(seed=42) first 4 draws: 0.0953 0.9378 0.7694 0.8598  (identical -- same seed)
```

Two entirely separate `SimpleRNG` instances, constructed from the identical seed `42`, produce byte-for-byte identical sequences — not a bug in isolation (a deterministic generator is *supposed* to be reproducible from a fixed seed, and that's exactly what makes Worked Example 8.2.4's frequency check trustworthy), but a genuine trap the moment two different tensors in the same program are seeded from the same fixed constant by accident, expecting them to hold independent random values and silently getting identical ones instead.

### Worked Example 8.2.2 — a genuinely discovered small-seed weakness, and its fix

```cpp
// NAIVE version: the seed is used directly as the initial state.
struct NaiveRNG {
    unsigned long long state;
    __host__ __device__ NaiveRNG(unsigned long long seed) : state(seed) {}
    __host__ __device__ float next_uniform() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return (float)((state >> 40) & 0xFFFFFF) / (float)(1 << 24);
    }
};
```

Compiled and run as part of the complete `02_random_generation.cu` further below:

```bash
nvcc -arch=sm_80 02_random_generation.cu -o 02_random_generation
./02_random_generation
```

Genuinely compiled and run:

```
small-seed quality, naive vs. pre-mixed constructor, seed=42:
NaiveRNG(42) first draw:  0.000000  (measurably weak -- see explanation below)
SimpleRNG(42) first draw: 0.859794  (pre-mixed seed, high-quality from draw 1)
```

This is not a hypothetical bug — it's exactly what this book's own testing pass turned up while verifying this chapter's code. `NaiveRNG`'s constructor stores the seed directly as `state`; a small integer like `42` occupies only `state`'s lowest 6 bits (`42 = 0b101010`), leaving every bit above that at `0`, and `next_uniform()` reads its output from bits `40`–`63` — bits that a *single* round of `xorshift64` hasn't yet had the chance to spread any of the seed's original entropy into. The result: the very first draw from a small, freshly-seeded `NaiveRNG` is `0.0`, not a rare or unlucky value but a direct, mechanical consequence of where the seed's few bits started out. `SimpleRNG`'s fix is one extra step in the constructor — XOR the seed against a fixed, odd 64-bit constant before it becomes the initial state — a standard technique (sometimes called splitmix-style seed mixing) that immediately spreads a small seed's few bits across the entire 64-bit word, before `next_uniform()` is ever called at all.

> `[COMMON TRAP]` It's tempting to treat "the numbers look random after a few draws" as sufficient testing for a PRNG, since `NaiveRNG`'s *second* and later draws in an actual run are statistically unremarkable — this chapter caught the weakness specifically by checking the *first* draw from a *small* seed, a combination easy to skip if a test suite only ever checks generators seeded from, say, the current time (a large, effectively random 64-bit value already, where this particular weakness is invisible). The lesson generalizes past random number generation: a bug that only shows up for a specific, easy-to-overlook input (a small seed, Section 8.3's missing trailing newline) is not the same as a rare bug — it's a common bug hiding behind an untested case.

### Worked Example 8.2.3 — Fisher-Yates shuffle, traced step by step

```cpp
int arr[5] = {10, 20, 30, 40, 50};
SimpleRNG shuffle_rng(7);
for (int i = 4; i > 0; i--) {
    int j = (int)(shuffle_rng.next_uniform() * (i + 1));  // j drawn from [0, i], NOT [0, n)
    if (j > i) j = i;
    int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
}
```

Compiled and run as part of the complete `02_random_generation.cu` further below:

```bash
nvcc -arch=sm_80 02_random_generation.cu -o 02_random_generation
./02_random_generation
```

Genuinely compiled and run:

```
Fisher-Yates shuffle, traced (using the fixed SimpleRNG):
before: [10,20,30,40,50]
  i=4, j=4 -> [10,20,30,40,50]
  i=3, j=0 -> [40,20,30,10,50]
  i=2, j=2 -> [40,20,30,10,50]
  i=1, j=0 -> [20,40,30,10,50]
after:  [20,40,30,10,50]
same 5 elements, just reordered? sum before=150, sum after=150, match=true
```

Five iterations shrink the "still needs shuffling" range one element at a time: `i=4` draws `j` from `[0,4]` (all five positions still eligible), swaps position `4` with itself (a no-op this time, but a legal outcome), then `i=3` draws from the now-*smaller* range `[0,3]` — position `4` is excluded, because it's already been finalized. `sum before = sum after = 150` confirms the array still holds the identical five elements, just reordered — a real, checkable invariant for *any* correct shuffle, though not a proof of uniform randomness by itself.

> `[COMMON TRAP]` Drawing `j` from the *full* range `[0, n)` on every iteration instead of the *shrinking* range `[0, i]` is a well-known, genuinely common way to implement Fisher-Yates incorrectly — it still produces *a* permutation (the sum-invariant check above would still pass), but not a *uniformly random* one; some final orderings become measurably more likely than others. The bug is invisible to a check that only verifies "still the same elements," which is exactly why real test suites for shuffle algorithms check the *distribution* of many repeated shuffles, not just one run's output.

### Worked Example 8.2.4 — inverse-CDF sampling, checked against a large-sample frequency

```cpp
float cumulative[3] = {0.2f, 0.5f, 1.0f};  // running sum of target probabilities [0.2, 0.3, 0.5]

__host__ __device__ int sample_category(float u, const float* cumulative, int num_categories) {
    for (int c = 0; c < num_categories; c++) {
        if (u < cumulative[c]) return c;
    }
    return num_categories - 1;
}
```

Compiled and run as the complete `03_inverse_cdf_sampling.cu` further below:

```bash
nvcc -arch=sm_80 03_inverse_cdf_sampling.cu -o 03_inverse_cdf_sampling
./03_inverse_cdf_sampling
```

Genuinely compiled and run:

```
target probabilities: [0.2, 0.3, 0.5]
cumulative:           [0.2, 0.5, 1.0]

10000 draws, empirical frequency vs. target:
  category 0: count=1974, frequency=0.1974, target=0.2000, |diff|=0.0026
  category 1: count=2994, frequency=0.2994, target=0.3000, |diff|=0.0006
  category 2: count=5032, frequency=0.5032, target=0.5000, |diff|=0.0032
```

`sample_category` finds the first cumulative bucket a uniform draw `u` falls under — `u < 0.2` lands in category 0, `0.2 ≤ u < 0.5` lands in category 1, and everything else lands in category 2 — so a category's *share* of the `[0,1)` interval it owns directly determines how often it gets drawn over many samples. `10000` draws from `SimpleRNG(2024)` land within `0.0032` of every target probability, a real, deterministic, exactly-reproducible result (this exact seed always produces this exact count) rather than a claim about randomness in the abstract.

## 8.3 CSV Import/Export: A Genuine Off-by-One, Found by Testing `[FOUNDATIONAL]`

### Intuition

Getting a `Tensor`'s values to and from a text file — model weights saved from another framework, a small dataset, a debugging dump — sounds like it should be one of this book's simplest operations. Section 8.2 already showed one plausible-looking function harboring a real bug; this section finds a second one, in code that passes every test you'd think to write until you specifically try a file that doesn't end the way most editors' files do.

### Background

| | A file ending in `\n` (the common case) | A file with no trailing `\n` on its last line |
|---|---|---|
| Newline characters present | One per row | One *fewer* than the number of rows |
| Naive "count the `\n` characters" row count | Correct | Wrong — undercounts by exactly 1 |
| Why this happens in practice | — | Some tools, and many hand-edited files, simply don't add a final newline |

### Worked Example 8.3.1 — the bug, genuinely triggered

```cpp
int count_rows_naive(const char* filename) {
    FILE* f = fopen(filename, "r");
    int count = 0;
    int ch;
    while ((ch = fgetc(f)) != EOF) {
        if (ch == '\n') count++;
    }
    fclose(f);
    return count;
}
```

Compiled and run as part of the complete `04_tensor_csv_io.cu` further below:

```bash
nvcc -arch=sm_80 04_tensor_csv_io.cu -o 04_tensor_csv_io
./04_tensor_csv_io
```

Genuinely compiled and run:

```
file WITH trailing newline (3 real rows):
  count_rows_naive  = 3
  count_rows_fixed  = 3

file WITHOUT trailing newline (still 3 real rows):
  count_rows_naive  = 2  <- BUG: undercounts by 1
  count_rows_fixed  = 3  <- correct
```

Two files, both genuinely written by this program, holding the identical 3 rows of data — one written with `fprintf(f, "...\n")` after every row (Section 8.1's convention, and this section's own `write_csv` function), one deliberately written with `fprintf(f, "1.0,2.0\n3.0,4.0\n5.0,6.0")` — no trailing `\n` after the last row. `count_rows_naive` reports `3` for the first file and `2` for the second, an undercount caused by exactly the mechanism the Background table names: three rows means two internal newlines plus, ordinarily, a final one — and the second file simply doesn't have that final one.

### Worked Example 8.3.2 — the fix, and a full round trip

```cpp
int count_rows_fixed(const char* filename) {
    FILE* f = fopen(filename, "r");
    int count = 0;
    int ch, last_ch = '\n';
    bool saw_any_byte = false;
    while ((ch = fgetc(f)) != EOF) {
        saw_any_byte = true;
        if (ch == '\n') count++;
        last_ch = ch;
    }
    if (saw_any_byte && last_ch != '\n') count++;  // the missing final newline's row
    fclose(f);
    return count;
}
```

Compiled and run as part of the complete `04_tensor_csv_io.cu` further below:

```bash
nvcc -arch=sm_80 04_tensor_csv_io.cu -o 04_tensor_csv_io
./04_tensor_csv_io
```

Genuinely compiled and run:

```
full round trip: write 6 known values, read them back, compare:
  wrote:      [1.0,2.0,3.0,4.0,5.0,6.0]
  read back:  [1.0,2.0,3.0,4.0,5.0,6.0]  (n_read=6)
  round trip exact match? true
```

`count_rows_fixed` tracks the last character it saw; if the file had any content at all and that last character wasn't a newline, it counts one more row for the unterminated final line — genuinely correct on both test files above. The round trip — `write_csv` followed immediately by `read_csv` on the same 6 known values — confirms the actual data survives the trip exactly, byte-derived-float for byte-derived-float, which is the check that actually matters for a `Tensor`'s values; row-counting is only ever a means to that end.

> `[COMMON TRAP]` This entire bug exists because `count_rows_naive` tries to *rediscover* a file's row count by scanning its bytes — information a `Tensor` already has, unambiguously, in its own `shape`. Every real import path this book's later chapters use passes the expected shape in *before* reading (exactly `read_csv`'s `max_values` parameter above), rather than asking the file to reveal its own dimensions through a heuristic that a missing trailing newline, an extra blank line, or a stray trailing comma can each independently break. Row-counting from raw bytes is occasionally unavoidable — the very first time an entirely unfamiliar file is opened — but it should never be trusted as an ongoing substitute for a shape the program is already supposed to know.

## 8.4 Complete Runnable Code

### File: `01_factory_functions.cu`

```cpp
#include <cstdio>

// data[i] = start + i*step, for i in [0, n) -- one thread, one element, the
// exact broadcast pattern Chapter 4.5 established.
__global__ void arange_kernel(float* data, float start, float step, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = start + i * step;
}

// n points from start to stop INCLUSIVE of both ends -- step = (stop-start)/(n-1).
__global__ void linspace_kernel(float* data, float start, float stop, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float step = (stop - start) / (float)(n - 1);
        data[i] = start + i * step;
    }
}

// Identity matrix, flattened row-major, with an optional diagonal offset k:
// k=0 is the main diagonal, k=1 is the superdiagonal (one column right),
// k=-1 is the subdiagonal (one column left).
__global__ void eye_kernel(float* data, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // row
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // col
    if (i < n && j < n) {
        data[i * n + j] = (j - i == k) ? 1.0f : 0.0f;
    }
}

// Host references -- the exact same formulas, computed without a GPU, used
// to cross-check what each kernel would produce on real hardware.
void host_arange(float* out, float start, float step, int n) {
    for (int i = 0; i < n; i++) out[i] = start + i * step;
}
void host_linspace(float* out, float start, float stop, int n) {
    float step = (stop - start) / (float)(n - 1);
    for (int i = 0; i < n; i++) out[i] = start + i * step;
}
void host_eye(float* out, int n, int k) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            out[i * n + j] = (j - i == k) ? 1.0f : 0.0f;
}

int main() {
    float ar[5];
    host_arange(ar, 0.0f, 2.0f, 5);
    printf("arange(start=0, step=2, n=5) = [%.1f, %.1f, %.1f, %.1f, %.1f]\n",
           ar[0], ar[1], ar[2], ar[3], ar[4]);

    float ls[5];
    host_linspace(ls, 0.0f, 1.0f, 5);
    printf("linspace(0, 1, n=5) = [%.2f, %.2f, %.2f, %.2f, %.2f]  (both ends included)\n",
           ls[0], ls[1], ls[2], ls[3], ls[4]);

    float id[16];
    host_eye(id, 4, 0);
    printf("\neye(4, k=0) (main diagonal):\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%.0f %.0f %.0f %.0f]\n", id[i*4+0], id[i*4+1], id[i*4+2], id[i*4+3]);
    }

    float sup[16];
    host_eye(sup, 4, 1);
    printf("\neye(4, k=1) (superdiagonal):\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%.0f %.0f %.0f %.0f]\n", sup[i*4+0], sup[i*4+1], sup[i*4+2], sup[i*4+3]);
    }
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_factory_functions.cu -o 01_factory_functions
./01_factory_functions
nvcc -arch=sm_80 -c 01_factory_functions.cu -o 01_factory_functions.o
```

Produces exactly the output shown in Worked Examples 8.1.1 and 8.1.2 above; the final `-c` compile confirms `arange_kernel`, `linspace_kernel`, and `eye_kernel` all compile clean for `sm_80`, even though only the host reference functions actually execute in `main()`.

### File: `02_random_generation.cu`

```cpp
#include <cstdio>

// NAIVE version: the seed is used directly as the initial state. Kept here
// only to demonstrate a real, discovered weakness -- see main() below.
struct NaiveRNG {
    unsigned long long state;
    __host__ __device__ NaiveRNG(unsigned long long seed) : state(seed) {}
    __host__ __device__ float next_uniform() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return (float)((state >> 40) & 0xFFFFFF) / (float)(1 << 24);
    }
};

// FIXED version: the seed is pre-mixed (XORed with a fixed odd constant,
// the standard splitmix-style trick) before becoming the initial state --
// __host__ __device__ so the exact same generator genuinely runs and is
// checked without any GPU involved, unlike CUDA's own curand library, whose
// generation calls are device-only.
struct SimpleRNG {
    unsigned long long state;
    __host__ __device__ SimpleRNG(unsigned long long seed) : state(seed ^ 0x9E3779B97F4A7C15ULL) {}
    __host__ __device__ float next_uniform() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return (float)((state >> 40) & 0xFFFFFF) / (float)(1 << 24);
    }
};

int main() {
    // The REPRODUCIBILITY trap: two RNG instances constructed with the SAME
    // seed produce the IDENTICAL sequence -- easy to trigger by accident if
    // every tensor in a batch is seeded from the same fixed constant instead
    // of a per-tensor seed.
    SimpleRNG rng_a(42);
    SimpleRNG rng_b(42);
    printf("rng_a(seed=42) first 4 draws: %.4f %.4f %.4f %.4f\n",
           rng_a.next_uniform(), rng_a.next_uniform(), rng_a.next_uniform(), rng_a.next_uniform());
    printf("rng_b(seed=42) first 4 draws: %.4f %.4f %.4f %.4f  (identical -- same seed)\n",
           rng_b.next_uniform(), rng_b.next_uniform(), rng_b.next_uniform(), rng_b.next_uniform());

    // The SMALL-SEED trap, genuinely discovered while testing this chapter's
    // code: a tiny numeric seed like 42 occupies only the low bits of a
    // 64-bit state, and xorshift needs a few iterations to spread that
    // entropy into the HIGH bits next_uniform() actually reads.
    printf("\nsmall-seed quality, naive vs. pre-mixed constructor, seed=42:\n");
    NaiveRNG naive(42);
    printf("NaiveRNG(42) first draw:  %.6f  (measurably weak -- see explanation below)\n", naive.next_uniform());
    SimpleRNG fixed(42);
    printf("SimpleRNG(42) first draw: %.6f  (pre-mixed seed, high-quality from draw 1)\n", fixed.next_uniform());

    // Fisher-Yates shuffle, traced: walk i from the LAST index down to 1,
    // swap element i with a uniformly random element in [0, i].
    printf("\nFisher-Yates shuffle, traced (using the fixed SimpleRNG):\n");
    int arr[5] = {10, 20, 30, 40, 50};
    SimpleRNG shuffle_rng(7);
    printf("before: [%d,%d,%d,%d,%d]\n", arr[0], arr[1], arr[2], arr[3], arr[4]);
    for (int i = 4; i > 0; i--) {
        int j = (int)(shuffle_rng.next_uniform() * (i + 1));
        if (j > i) j = i;  // guard the rare rounding case where next_uniform() rounds up to exactly 1.0
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
        printf("  i=%d, j=%d -> [%d,%d,%d,%d,%d]\n", i, j, arr[0], arr[1], arr[2], arr[3], arr[4]);
    }
    printf("after:  [%d,%d,%d,%d,%d]\n", arr[0], arr[1], arr[2], arr[3], arr[4]);

    int sum_before = 10+20+30+40+50;
    int sum_after = arr[0]+arr[1]+arr[2]+arr[3]+arr[4];
    printf("same 5 elements, just reordered? sum before=%d, sum after=%d, match=%s\n",
           sum_before, sum_after, (sum_before == sum_after) ? "true" : "false");
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_random_generation.cu -o 02_random_generation
./02_random_generation
```

Produces exactly the output shown in Worked Examples 8.2.1, 8.2.2, and 8.2.3 above.

### File: `03_inverse_cdf_sampling.cu`

```cpp
#include <cstdio>

struct SimpleRNG {
    unsigned long long state;
    __host__ __device__ SimpleRNG(unsigned long long seed) : state(seed ^ 0x9E3779B97F4A7C15ULL) {}
    __host__ __device__ float next_uniform() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return (float)((state >> 40) & 0xFFFFFF) / (float)(1 << 24);
    }
};

// Sample one of 3 categories with target probabilities [0.2, 0.3, 0.5] by
// drawing u in [0,1) and finding which cumulative bucket it falls into --
// inverse-CDF sampling from a discrete distribution.
__host__ __device__ int sample_category(float u, const float* cumulative, int num_categories) {
    for (int c = 0; c < num_categories; c++) {
        if (u < cumulative[c]) return c;
    }
    return num_categories - 1;  // guards float rounding landing exactly at 1.0
}

int main() {
    float target_probs[3] = {0.2f, 0.3f, 0.5f};
    float cumulative[3] = {0.2f, 0.5f, 1.0f};  // running sum of target_probs

    printf("target probabilities: [%.1f, %.1f, %.1f]\n", target_probs[0], target_probs[1], target_probs[2]);
    printf("cumulative:           [%.1f, %.1f, %.1f]\n", cumulative[0], cumulative[1], cumulative[2]);

    const int n = 10000;
    int counts[3] = {0, 0, 0};
    SimpleRNG rng(2024);
    for (int i = 0; i < n; i++) {
        float u = rng.next_uniform();
        int c = sample_category(u, cumulative, 3);
        counts[c]++;
    }

    printf("\n%d draws, empirical frequency vs. target:\n", n);
    for (int c = 0; c < 3; c++) {
        float freq = (float)counts[c] / n;
        printf("  category %d: count=%d, frequency=%.4f, target=%.4f, |diff|=%.4f\n",
               c, counts[c], freq, target_probs[c], fabsf(freq - target_probs[c]));
    }
    return 0;
}
```

```bash
nvcc -arch=sm_80 03_inverse_cdf_sampling.cu -o 03_inverse_cdf_sampling
./03_inverse_cdf_sampling
```

Produces exactly the output shown in Worked Example 8.2.4 above.

### File: `04_tensor_csv_io.cu`

```cpp
#include <cstdio>
#include <cstring>
#include <cmath>

// Write a row-major host buffer out as comma-separated text, one row per line.
void write_csv(const char* filename, const float* data, int rows, int cols) {
    FILE* f = fopen(filename, "w");
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            fprintf(f, "%.1f", data[r * cols + c]);
            if (c < cols - 1) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// Read comma-separated floats back into a flat, row-major buffer. Assumes
// the caller already knows rows*cols (Chapter 6's Tensor always does --
// its shape is never a mystery the way an arbitrary external file's is).
int read_csv(const char* filename, float* out, int max_values) {
    FILE* f = fopen(filename, "r");
    int count = 0;
    while (count < max_values && fscanf(f, "%f", &out[count]) == 1) {
        count++;
        fgetc(f);  // consume the following ',' or '\n'
    }
    fclose(f);
    return count;
}

// The NAIVE row counter: count newline characters. This has a real bug --
// see main() below for the discovery.
int count_rows_naive(const char* filename) {
    FILE* f = fopen(filename, "r");
    int count = 0;
    int ch;
    while ((ch = fgetc(f)) != EOF) {
        if (ch == '\n') count++;
    }
    fclose(f);
    return count;
}

// The FIXED row counter: count newlines, but if the file has any content at
// all and does NOT end with a newline, that last, unterminated line is a
// real row too -- count it.
int count_rows_fixed(const char* filename) {
    FILE* f = fopen(filename, "r");
    int count = 0;
    int ch, last_ch = '\n';  // pretend the file "starts" just after a newline
    bool saw_any_byte = false;
    while ((ch = fgetc(f)) != EOF) {
        saw_any_byte = true;
        if (ch == '\n') count++;
        last_ch = ch;
    }
    if (saw_any_byte && last_ch != '\n') count++;  // the missing final newline's row
    fclose(f);
    return count;
}

int main() {
    float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // 3 rows, 2 cols

    write_csv("with_trailing_newline.csv", data, 3, 2);

    // Deliberately write a version with NO trailing newline on the last
    // line, exactly the way some tools (and some hand-edited files) do.
    FILE* f = fopen("no_trailing_newline.csv", "w");
    fprintf(f, "1.0,2.0\n3.0,4.0\n5.0,6.0");  // note: no trailing \n
    fclose(f);

    printf("file WITH trailing newline (3 real rows):\n");
    printf("  count_rows_naive  = %d\n", count_rows_naive("with_trailing_newline.csv"));
    printf("  count_rows_fixed  = %d\n", count_rows_fixed("with_trailing_newline.csv"));

    printf("\nfile WITHOUT trailing newline (still 3 real rows):\n");
    printf("  count_rows_naive  = %d  <- BUG: undercounts by 1\n", count_rows_naive("no_trailing_newline.csv"));
    printf("  count_rows_fixed  = %d  <- correct\n", count_rows_fixed("no_trailing_newline.csv"));

    printf("\nfull round trip: write 6 known values, read them back, compare:\n");
    float roundtrip[6];
    int n_read = read_csv("with_trailing_newline.csv", roundtrip, 6);
    printf("  wrote:      [%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]\n", data[0],data[1],data[2],data[3],data[4],data[5]);
    printf("  read back:  [%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]  (n_read=%d)\n",
           roundtrip[0],roundtrip[1],roundtrip[2],roundtrip[3],roundtrip[4],roundtrip[5], n_read);
    bool all_match = true;
    for (int i = 0; i < 6; i++) if (fabsf(data[i] - roundtrip[i]) > 1e-6f) all_match = false;
    printf("  round trip exact match? %s\n", all_match ? "true" : "false");

    return 0;
}
```

```bash
nvcc -arch=sm_80 04_tensor_csv_io.cu -o 04_tensor_csv_io
./04_tensor_csv_io
```

### Expected Output for `04_tensor_csv_io.cu`

```
file WITH trailing newline (3 real rows):
  count_rows_naive  = 3
  count_rows_fixed  = 3

file WITHOUT trailing newline (still 3 real rows):
  count_rows_naive  = 2  <- BUG: undercounts by 1
  count_rows_fixed  = 3  <- correct

full round trip: write 6 known values, read them back, compare:
  wrote:      [1.0,2.0,3.0,4.0,5.0,6.0]
  read back:  [1.0,2.0,3.0,4.0,5.0,6.0]  (n_read=6)
  round trip exact match? true
```

Every number in this section was independently verified earlier in this chapter. All four files genuinely compile clean under `nvcc -arch=sm_80` and run to completion in this no-GPU environment — none of Section 8.1's kernels are actually launched (their host-reference counterparts run instead, exactly as Chapter 4.6 established), while Sections 8.2 and 8.3's code touches no CUDA API at all and runs identically on host or device.

## Chapter Summary

`arange` and `linspace` are both the broadcast pattern applied to pure index arithmetic, differing only in whether the formula is built to reach the stop value exactly (`linspace`) or deliberately stop one step short of it (`arange`) — and `linspace`'s division by `n-1` is a real edge case at `n=1` that `arange`'s formula never encounters. `eye` generalizes to any diagonal offset by testing `col - row` against a constant rather than testing for equality alone. Because CUDA's own `curand` library can't be exercised without a GPU, this chapter built a small alternative from scratch, and testing it thoroughly — not just running it once and eyeballing the output — turned up two genuine, real weaknesses: identical seeds silently producing identical streams, and a small seed producing a measurably weak first draw, fixed by pre-mixing the seed before it becomes the generator's initial state. Fisher-Yates shuffling and inverse-CDF sampling both build on that same generator, checked against real invariants — an unchanged element sum, and a large sample's frequency matching its target probabilities to within a few thousandths. And a CSV row-counting function that passed every straightforward test still had a real off-by-one bug, invisible until tested against a file lacking a trailing newline — the same general lesson Worked Example 8.2.2's small-seed weakness taught: an untested edge case is not a rare bug, it's a bug nobody has looked for yet.

## Self-Check Questions

1. `arange(start=5, step=3, n=4)` — compute all four values by hand.
2. Why does `linspace`'s formula fail for `n=1`, but `arange`'s formula does not?
3. For `eye(5, k=-2)`, which `(row, col)` pairs would hold a `1`? List them.
4. `NaiveRNG(42)`'s first draw was measurably weak because the seed `42` only occupies a few low bits of a 64-bit state. Would `NaiveRNG(1000000007)` (a seed occupying more bits) necessarily avoid the same problem on its first draw? What does `SimpleRNG`'s pre-mixing step do that a merely "bigger" seed doesn't guarantee?
5. `count_rows_naive` undercounts a file missing its trailing newline. Would the same function correctly count an EMPTY file (zero bytes)? Would it correctly count a file containing a single blank line (just one `\n` and nothing else)? Reason through both using the function's actual logic, not just its behavior on this chapter's two test files.

## Where We Go Next

Every tensor this chapter has filled has been a plain, dense, general-purpose buffer — every element genuinely allocated and (on real hardware) genuinely written. Chapter 9 looks at shapes where that's wasteful by construction: an identity matrix is mostly zeros in a perfectly predictable pattern, and a specialized representation can skip storing (and skip computing) almost all of them.

## Worked Solutions

**1.** `arange(5, 3, 4)`: `i=0` gives `5 + 0×3 = 5`; `i=1` gives `5 + 1×3 = 8`; `i=2` gives `5 + 2×3 = 11`; `i=3` gives `5 + 3×3 = 14`. Result: `[5, 8, 11, 14]`.

**2.** `linspace`'s formula computes `step = (stop - start) / (n - 1)`, which divides by zero when `n = 1`. `arange`'s formula, `start + i × step`, takes `step` as a direct input rather than deriving it from `n` — there's no division by `n` (or `n-1`) anywhere in `arange`'s formula at all, so `n=1` is just an ordinary one-iteration loop producing `[start]`, no special case required.

**3.** `k=-2` means `col - row == -2`, i.e. `row = col + 2`. For a `5×5` matrix (rows and columns `0` through `4`), the valid pairs are `(row=2, col=0)`, `(row=3, col=1)`, and `(row=4, col=2)` — three positions, since `row=5` or `row=6` (which `col=3` or `col=4` would require) fall outside the `5×5` matrix.

**4.** No — a larger seed doesn't inherently avoid the problem, because the issue isn't the seed's numeric *size*, it's how many of the state word's bits it actually sets to a non-zero value combined with how many xorshift rounds have run. A seed like `1000000007` happens to set bits across a wider range than `42` does, so it's *less likely* to trigger the exact same visible symptom, but nothing about simply picking a "bigger" number *guarantees* good bit-spread — a seed like `0x8000000000000000` is numerically enormous yet sets only a single high bit. `SimpleRNG`'s pre-mixing step doesn't depend on the seed's magnitude at all: XORing against a fixed constant with roughly half its bits set to `1` guarantees the resulting initial state has a healthy mix of `0`s and `1`s spread across the full 64 bits, regardless of what the original seed looked like.

**5.** For an empty file, `count_rows_naive`'s loop body never executes at all (the very first `fgetc` returns `EOF`), so it correctly returns `0` — an empty file has zero rows, and the function happens to get this right, though only because "zero newlines" and "zero rows" coincide for this one specific input. For a file containing a single blank line (exactly one `\n` and nothing else), the loop sees exactly one `\n` character and returns `1` — also correct, since one blank line is genuinely one row. Both of these "accidentally correct" cases are worth noticing precisely because they could easily be mistaken for evidence the function is fully correct, when Worked Example 8.3.1 already proved it isn't for the much more common case of a normal, multi-row file missing only its final newline.
