# Chapter 7: Memory Layout Design — Layouts, Views, Alignment, and Broadcasting

> "Chapter 6 wrote `offset = Σ coord[d] × stride[d]` as if it had one right answer per shape. It doesn't — the formula never changes, but the strides feeding it can encode a dozen different physical arrangements of the identical logical tensor, and this chapter is about four of the ones this book's `Tensor` actually needs: a second, genuinely different way to compute strides; a struct that reads an existing buffer without owning a byte of it; the padding a compiler quietly inserts when nobody asks it not to; and a trick that lets one small buffer answer for a much larger shape, one stride set to zero at a time."

**What you will understand by the end of this chapter:**

- Column-major strides, computed by the same right-to-left process Chapter 6.2 used but starting from the opposite end — and why this matters concretely the moment this book's `Tensor` ever needs to talk to cuBLAS
- `TensorView`: a non-owning struct that reads (or writes) an existing buffer through its own shape and strides, with no `cudaMalloc`, no destructor work, and no bytes moved — the mechanism behind transpose, slicing, and reshape
- Why struct field order changes `sizeof` even when the fields themselves are unchanged, traced to the compiler's alignment padding, and why `float4`'s 16-byte alignment (Chapter 5.2) is a special case of the same rule
- Broadcasting: combining a small shape with a larger one by setting a dimension's stride to `0`, so many logical coordinates legitimately alias the same physical element — and the real hazard that appears the moment code tries to *write* through a broadcast view instead of just reading it

**What you need to know first:**

- Chapter 6 in full — this chapter's `TensorStrides::col_major`, `TensorView`, and broadcast strides are all direct extensions of Chapter 6.1–6.4's `TensorShape`, `TensorStrides`, and offset formula
- Chapter 2.4 (RAII over `cudaMalloc`/`cudaFree`) — Section 7.2 depends on contrasting `TensorView`'s *absence* of RAII against `Tensor`'s presence of it
- Chapter 5.2 (`float4`'s 16-byte alignment and the vectorized `LDG.E.128` load it enables)
- If you've read the Mojo edition: this chapter follows its Chapter 7 section-for-section — strides revisited, views, alignment, broadcasting — adapted to CUDA C++'s struct-based views and its own alignment/`cudaMalloc` mechanics in place of Mojo's slice and SIMD-padding syntax.

## 7.1 Strides Revisited: One Formula, Many Layouts `[FOUNDATIONAL]`

### Intuition

Chapter 6.2 computed strides right-to-left, making the *last* dimension contiguous — **row-major**, C's native convention and the only layout this book has used so far. Nothing about the offset formula itself, `offset = Σ coord[d] × stride[d]`, cares which dimension is contiguous; that fact lives entirely in how the strides were computed. **Column-major** — the *first* dimension contiguous instead — is the same formula fed different strides, and it isn't a purely academic alternative: cuBLAS, the CUDA math library every serious GEMM in this book eventually calls into, is column-major by convention, a fact it inherited from Fortran BLAS decades before CUDA existed.

### Background

| | Row-major (this book's default) | Column-major (cuBLAS's convention) |
|---|---|---|
| Contiguous dimension | Last | First |
| Stride computation direction | Right to left: `strides[i] = strides[i+1] × dims[i+1]` | Left to right: `strides[i] = strides[i-1] × dims[i-1]` |
| `strides[0]` for shape `[2,3,4]` | `12` (skip a whole `[3,4]` block) | `1` (the first dimension is packed tightest) |
| Who uses it | This book's own `Tensor`, and C/C++ arrays generally | cuBLAS, and BLAS libraries generally |

### Worked Example 7.1.1 — the same `[2,3,4]` shape, both ways

```cpp
__host__ __device__ static TensorStrides row_major(const TensorShape& shape) {
    TensorStrides s;
    int running = 1;
    for (int i = shape.ndim - 1; i >= 0; i--) {
        s.strides[i] = running;
        running *= shape.dims[i];
    }
    return s;
}

// Column-major: the FIRST dimension is contiguous instead of the last --
// the convention cuBLAS inherits from Fortran BLAS.
__host__ __device__ static TensorStrides col_major(const TensorShape& shape) {
    TensorStrides s;
    int running = 1;
    for (int i = 0; i < shape.ndim; i++) {
        s.strides[i] = running;
        running *= shape.dims[i];
    }
    return s;
}
```

Compiled and run as part of the complete `01_layout_row_col_major.cu` further below:

```bash
nvcc -arch=sm_80 01_layout_row_col_major.cu -o 01_layout_row_col_major
./01_layout_row_col_major
```

Genuinely compiled and run:

```
shape = [2,3,4]
row-major strides = [12, 4, 1]
col-major strides = [1, 2, 6]
```

`col_major` runs the identical accumulation loop as `row_major`, just left-to-right instead of right-to-left: `i=0` sets `strides[0] = 1` (the running product starts at 1), then `running = 1 × dims[0] = 1 × 2 = 2`; `i=1` sets `strides[1] = 2`, then `running = 2 × dims[1] = 2 × 3 = 6`; `i=2` sets `strides[2] = 6`. Same shape, same formula shape, opposite direction, genuinely different strides.

### Worked Example 7.1.2 — index↔offset round trip, and where the two layouts disagree

Compiled and run as part of the complete `01_layout_row_col_major.cu` further below:

```bash
nvcc -arch=sm_80 01_layout_row_col_major.cu -o 01_layout_row_col_major
./01_layout_row_col_major
```

Genuinely compiled and run:

```
index<->offset round trip for coordinate (1, 1, 2):
row-major offset(1,1,2) = 18
col-major offset(1,1,2) = 15

same coordinate, opposite contiguity:
row-major offset(0,0,1) = 1  (innermost dim is contiguous)
col-major offset(0,0,1) = 6  (innermost dim is NOT contiguous here)
row-major offset(1,0,0) = 12  (outermost dim is NOT contiguous here)
col-major offset(1,0,0) = 1  (outermost dim is contiguous)

both layouts agree at the very last coordinate (a permutation always agrees at the boundary):
row-major offset(1,2,3) = 23, col-major offset(1,2,3) = 23, numel-1 = 23
```

`(1,1,2)` genuinely lands at two different flat offsets under the two layouts — `18` under row-major, `15` under column-major — confirming the same logical coordinate refers to physically different memory depending on which strides accompany it. The `(0,0,1)`/`(1,0,0)` pair makes the contiguity claim concrete: stepping by 1 along the innermost coordinate moves 1 flat slot under row-major but 6 under column-major, and vice versa for the outermost coordinate. The last line is a genuine, if slightly surprising, fact rather than a bug: *any* permutation-based relabeling of a full traversal agrees at its very first and very last element, since both layouts visit every one of the same 24 slots exactly once — it's only the coordinates in between that land differently.

### ASCII Diagram — a `[3,4]` matrix, two ways

```
Logical matrix (3 rows, 4 cols):        Row-major flat buffer (row 0, row 1, row 2):
 [ a b c d ]                             [ a b c d | e f g h | i j k l ]
 [ e f g h ]                              strides = [4, 1] -- row stride 4, col stride 1
 [ i j k l ]
                                         Column-major flat buffer (col 0, col 1, col 2, col 3):
                                          [ a e i | b f j | c g k | d h l ]
                                          strides = [1, 3] -- row stride 1, col stride 3
```

> `[COMMON TRAP]` Handing a row-major `Tensor`'s raw `data` pointer to a cuBLAS call without accounting for the layout mismatch doesn't fail loudly — cuBLAS has no way to know which convention the caller meant, so it reads the identical bytes as if they were column-major and silently computes a real, cleanly-executing, *wrong* matrix product. The standard fix, used throughout real CUDA codebases, isn't to physically transpose the data — it's to tell cuBLAS to treat the row-major buffer as an already-transposed column-major one (row-major `A` of shape `[m,n]` is bit-for-bit identical to column-major `Aᵀ` of shape `[n,m]`), a genuinely free reinterpretation that Section 7.2's `TensorView::transpose()` is the exact mechanism for.

## 7.2 `TensorView`: Reading a Tensor Without Copying It `[FOUNDATIONAL]`

### Intuition

Chapter 6's `Tensor` conflates two separate ideas that Section 7.1's cuBLAS trap already needs pulled apart: *owning* memory (RAII, `cudaMalloc`/`cudaFree`, exactly one legitimate owner) and *describing how to read* memory (a shape and a set of strides, which can be reinterpreted freely without touching a single byte). `TensorView` is a struct built for the second idea alone — a raw pointer plus a shape and strides, with no constructor that allocates and no destructor that frees, because it never owned anything to begin with.

### Background

| | `Tensor` (Chapter 6) | `TensorView` (this chapter) |
|---|---|---|
| Owns its memory? | Yes — `cudaMalloc`'d in its constructor | No — points into memory someone else owns |
| Destructor | `cudaFree`s `data` and `grad` | Trivial — there is nothing to free |
| Copying it | Deleted (Chapter 6.5) — copying would double-free | Perfectly safe to copy by value — copying a pointer and two small structs duplicates nothing that needs freeing |
| What transpose/slice/reshape cost | N/A — `Tensor` itself is never reshaped in place | Zero allocation, zero data movement — only the shape/strides/pointer triple changes |

### Worked Example 7.2.1 — transpose, verified against the original

```cpp
struct TensorView {
    float* data;
    Shape2D shape;
    Strides2D strides;

    __host__ __device__ int offset(int i0, int i1) const {
        return i0 * strides.strides[0] + i1 * strides.strides[1];
    }

    // Transpose: SWAP shape and strides. Zero bytes move; only the map changes.
    __host__ __device__ TensorView transpose() const {
        Shape2D new_shape(shape.dims[1], shape.dims[0]);
        Strides2D new_strides(strides.strides[1], strides.strides[0]);
        return TensorView{data, new_shape, new_strides};
    }
};
```

Compiled and run as part of the complete `02_tensor_view_slicing_transpose.cu` further below:

```bash
nvcc -arch=sm_80 02_tensor_view_slicing_transpose.cu -o 02_tensor_view_slicing_transpose
./02_tensor_view_slicing_transpose
```

Genuinely compiled and run:

```
original view: shape=[3,4], strides=[4,1]
view(1,2) = buf[6] = 6.0

transposed view: shape=[4,3], strides=[1,4]  (same buffer, same pointer: true)
t(2,1) = buf[6] = 6.0  (should equal view(1,2) -- transpose just swaps the coordinate order)
```

`transpose()` swaps `shape.dims[0]`↔`shape.dims[1]` and `strides[0]`↔`strides[1]` together, and nothing else — `t.data == view.data` genuinely holds, confirmed directly. Reading `view(1,2)` and `t(2,1)` land at the identical flat offset, `6`, because swapping both the shape and the strides in lockstep is exactly what makes "the element at row 1, column 2" and "the element at row 2, column 1 of the transposed view" refer to the same underlying value — transposition is a relabeling of coordinates, not a rearrangement of memory.

### Worked Example 7.2.2 — a row slice, reached through an advanced pointer

```cpp
// A row slice: same strides, a shrunk shape, and a pointer advanced by
// exactly this row's own starting offset -- again, zero bytes move.
__host__ __device__ TensorView row_slice(int row) const {
    Shape2D new_shape(1, shape.dims[1]);
    int start = offset(row, 0);
    return TensorView{data + start, new_shape, strides};
}
```

Compiled and run as part of the complete `02_tensor_view_slicing_transpose.cu` further below:

```bash
nvcc -arch=sm_80 02_tensor_view_slicing_transpose.cu -o 02_tensor_view_slicing_transpose
./02_tensor_view_slicing_transpose
```

Genuinely compiled and run:

```
row_slice(1): shape=[1,4], data now points at buf+4
r(0,2) = 6.0  (should equal view(1,2) -- same element, reached through the slice)
```

`row_slice(1)` computes `offset(1, 0) = 1×4 + 0×1 = 4`, advances the pointer to `buf + 4`, and keeps the original strides unchanged — reading `r(0,2)` through the *sliced* view's own, now-relative coordinate system reaches `buf[4 + 0×4 + 2×1] = buf[6]`, the identical element `view(1,2)` and Worked Example 7.2.1's transpose both already confirmed.

### ASCII Diagram — one buffer, three views, zero copies

```
Underlying buffer (12 floats, never moved):
 [0  1  2  3 | 4  5  6  7 | 8  9  10 11]

view            : shape=[3,4] strides=[4,1] pointer=buf+0   -- the whole thing
view.transpose(): shape=[4,3] strides=[1,4] pointer=buf+0   -- same bytes, swapped map
view.row_slice(1): shape=[1,4] strides=[4,1] pointer=buf+4  -- same bytes, narrowed map, shifted start
```

> `[COMMON TRAP]` Reshaping a view assumes the view's current strides are *contiguous* in the target shape's own row-major sense — an assumption `row_slice`'s output still satisfies, but `transpose`'s output does not. Calling something like `.reshape(12)` on `view.transpose()` and simply relabeling the existing `[4,3]`/`[1,4]` data as a flat 12-element row-major sequence would silently read the wrong elements in the wrong order, because the transposed view's actual memory order (column-major-ish, following its swapped strides) no longer matches what a fresh row-major reshape assumes. A correct reshape-after-transpose has to either verify contiguity first and fall back to a genuine copy when it doesn't hold, or refuse the reshape outright — silently trusting the shape numbers alone, without checking the strides behind them, is exactly how this class of bug reaches production.

## 7.3 Alignment and Padding: What the Compiler Inserts When You Don't Ask `[FOUNDATIONAL]`

### Intuition

Chapter 5.2's Common Trap named `float4`'s 16-byte alignment as a precondition for its vectorized `LDG.E.128` load; this section generalizes that single fact into the rule behind it. Every type has a required alignment — the address it must start at, a multiple of some power of two — and a struct's overall alignment is simply the largest alignment any of its fields demands. The compiler enforces this by inserting invisible padding bytes between fields whenever a field's natural position would violate its own alignment requirement, and — this is the part that surprises people — *changing the order fields are declared in can change how much padding gets inserted*, even though the fields themselves haven't changed at all.

### Background

| | `InterleavedHeader`: `char, int, char, int` | `GroupedHeader`: `int, int, char, char` |
|---|---|---|
| Padding after each `char` | 3 bytes, twice (to keep each following `int` 4-byte aligned) | Once, at the very end (to keep the whole struct's size a multiple of its alignment) |
| Total padding | 6 bytes | 2 bytes |
| `sizeof` (4 real bytes of `char` + 8 real bytes of `int`, in either order) | Larger | Smaller |

### Worked Example 7.3.1 — identical fields, different order, different `sizeof`

```cpp
// Fields interleaved by size -- the compiler pads after each `char` to keep
// the following `int` aligned to a 4-byte boundary.
struct InterleavedHeader {
    char a;   // 1 byte, then 3 bytes of padding
    int  b;   // 4 bytes
    char c;   // 1 byte, then 3 bytes of padding
    int  d;   // 4 bytes
};

// The SAME four fields, grouped largest-to-smallest -- both `int`s land on
// aligned boundaries back to back, and the two `char`s share one padding tail.
struct GroupedHeader {
    int  b;   // 4 bytes
    int  d;   // 4 bytes
    char a;   // 1 byte
    char c;   // 1 byte, then 2 bytes of padding to reach a 4-byte multiple
};
```

Compiled and run as part of the complete `03_alignment_padding.cu` further below:

```bash
nvcc -arch=sm_80 03_alignment_padding.cu -o 03_alignment_padding
./03_alignment_padding
```

Genuinely compiled and run:

```
InterleavedHeader (char,int,char,int): sizeof = 16, alignof = 4
GroupedHeader     (int,int,char,char): sizeof = 12, alignof = 4
identical fields, 4 fewer bytes just from reordering them
```

`InterleavedHeader` pays for padding twice — after `a` (3 bytes, so `b` starts 4-byte aligned) and after `c` (3 bytes, so `d` starts 4-byte aligned) — for 6 padding bytes total alongside 10 real bytes of data, rounding up to 16. `GroupedHeader` places both 4-byte `int`s back to back with no padding needed between them, then both 1-byte `char`s back to back after, paying only 2 padding bytes at the very end to bring the total to a multiple of the struct's own 4-byte alignment — 10 real bytes plus 2 padding bytes, `sizeof = 12`. Both structs hold the identical fields; only their declared order differs, and that alone accounts for the full 4-byte gap.

### Worked Example 7.3.2 — `float4`'s alignment, and what `cudaMalloc` guarantees around it

Compiled and run as part of the complete `03_alignment_padding.cu` further below:

```bash
nvcc -arch=sm_80 03_alignment_padding.cu -o 03_alignment_padding
./03_alignment_padding
```

Genuinely compiled and run:

```
float4's alignment, which Chapter 5.2's vectorized LDG.E.128 load depends on:
alignof(float4) = 16, sizeof(float4) = 16

cudaMalloc's own alignment guarantee, genuinely queried:
cudaMalloc err=100 (no CUDA-capable device is detected); ptr=(nil)
```

`alignof(float4) == 16 == sizeof(float4)` — the type's size and its alignment requirement coincide exactly, which is precisely why a `float4*` pointer must itself be a multiple of 16 to dereference legally (Chapter 5.2's Common Trap). `cudaMalloc` genuinely reports `cudaErrorNoDevice` here, the same honest disclosure this book has shown since Chapter 1.3 — but on real hardware, `cudaMalloc`'s returned pointer is documented to be aligned far past what `float4` requires (current CUDA implementations guarantee at least 256-byte alignment for any allocation), which is exactly why every `Tensor` in this book can reinterpret its own `cudaMalloc`'d `data` buffer as `float4*` for Chapter 5.2's vectorized loads without a second thought — the allocator's guarantee already covers it.

> `[COMMON TRAP]` Chapter 6's `Tensor` struct itself is subject to this exact same padding rule — `TensorShape` (16 bytes: 4 `int`s worth of `dims` plus `ndim`, actually 20 bytes before its own padding), `TensorStrides` (16 bytes), then two `float*` pointers (8 bytes each on a 64-bit target) — and reordering those four fields could, in principle, change `Tensor`'s own `sizeof` the same way this section's toy structs did. This book's `Tensor` happens to declare its fields in a reasonable order already (largest alignment requirements first), but it's worth genuinely checking with `sizeof`, not assuming, any time a struct's field order changes — the compiler never warns you when reordering fields silently grows or shrinks a type.

## 7.4 Broadcasting: One Small Buffer, Many Logical Coordinates `[FOUNDATIONAL]`

### Intuition

Adding a per-column bias to every row of a matrix shouldn't require physically duplicating that bias once per row — the bias vector is the same three numbers no matter which row is asking for them. **Broadcasting** makes this concrete at the stride level: give the "extra," duplicated dimension a stride of exactly `0`. Section 6.4's offset formula doesn't need to know anything special about broadcasting — `offset = Σ coord[d] × stride[d]` already produces the right answer on its own the moment one of those strides is `0`, because multiplying any coordinate by `0` always contributes nothing to the sum.

### Background

| | An ordinary dimension | A broadcast dimension |
|---|---|---|
| Stride | Some positive value, reflecting real separation in memory | `0` |
| What changing that coordinate does to the offset | Moves to a different, real element | Nothing — the offset is unchanged regardless of the coordinate's value |
| Backing storage | One element's worth of memory per position along this dimension | A single element's worth of memory, reused for every position |

### Worked Example 7.4.1 — a `[3]` vector, broadcast against a `[4, 3]` shape

```cpp
// A real [3] vector, stored contiguously.
float vec[3] = {10.0f, 20.0f, 30.0f};

// Broadcasting vec (shape [3]) against a target shape [4, 3]: the leading
// dimension (size 4) has NO backing storage of its own, so its stride is
// set to 0 -- every row reads the SAME 3 underlying floats. Zero bytes
// are duplicated; only the map claims a shape 4x its real size.
Shape2D broadcast_shape(4, 3);
Strides2D broadcast_strides(0, 1);   // stride 0 along the broadcast dimension
```

Compiled and run as the complete `04_broadcast_strides.cu` further below:

```bash
nvcc -arch=sm_80 04_broadcast_strides.cu -o 04_broadcast_strides
./04_broadcast_strides
```

Genuinely compiled and run:

```
real buffer: vec = [10.0, 20.0, 30.0]  (3 floats, actually stored)
broadcast shape = [4, 3], broadcast strides = [0, 1]

every 'row' of the broadcast view maps to the SAME 3 offsets:
  row 0 -> offsets (0,1,2) -> values (10.0,20.0,30.0)
  row 1 -> offsets (0,1,2) -> values (10.0,20.0,30.0)
  row 2 -> offsets (0,1,2) -> values (10.0,20.0,30.0)
  row 3 -> offsets (0,1,2) -> values (10.0,20.0,30.0)

all 4 rows alias the identical 3 offsets? true
```

Four different `row` values — `0`, `1`, `2`, `3` — each multiplied by the broadcast dimension's `0` stride, all contribute exactly `0` to the offset formula, so `offset(row, col)` collapses to `offset(0, col) = col` for every row: the loop's genuinely printed offsets, `(0,1,2)`, are identical across all four iterations, and the values read back — `10.0, 20.0, 30.0` — are the same three real floats every single time. A `[4,3]`-shaped view exists entirely on top of a 3-float buffer, with zero duplication.

### ASCII Diagram — four logical rows, one physical row

```
Real storage (3 floats):           Broadcast view, shape [4,3], strides [0,1]:
 [ 10.0  20.0  30.0 ]                logical row 0 -----\
                                     logical row 1 ------ all four point at the SAME
                                     logical row 2 ------ 3 physical floats above
                                     logical row 3 -----/
```

> `[COMMON TRAP]` Broadcasting is only safe as a *read* pattern. If four CUDA threads each believed they owned one "row" of this section's `[4,3]` broadcast view and each tried to *write* their own result back into it, all four threads would genuinely be writing to the same three physical addresses — a real data race, with the final value determined by whichever write happens to land last, not by any of the four threads' actual intended output. This is exactly the situation Part 4's autodiff engine runs into for real: when a broadcast input's gradient is computed, the incoming gradients from every logical row have to be *summed* into that one small buffer, using an atomic add or an explicit reduction, rather than written directly — a genuinely different and more careful operation than the elementwise writes Part 2's kernels perform on ordinary, non-broadcast tensors.

## 7.5 Complete Runnable Code

### File: `01_layout_row_col_major.cu`

```cpp
#include <cstdio>

struct TensorShape {
    static const int MAX_DIMS = 4;
    int dims[MAX_DIMS];
    int ndim;
    __host__ __device__ TensorShape(int d0, int d1, int d2) : dims{d0,d1,d2,0}, ndim(3) {}
    __host__ __device__ int numel() const { int t=1; for (int i=0;i<ndim;i++) t*=dims[i]; return t; }
};

struct TensorStrides {
    int strides[TensorShape::MAX_DIMS];

    __host__ __device__ static TensorStrides row_major(const TensorShape& shape) {
        TensorStrides s;
        int running = 1;
        for (int i = shape.ndim - 1; i >= 0; i--) {
            s.strides[i] = running;
            running *= shape.dims[i];
        }
        return s;
    }

    // Column-major: the FIRST dimension is contiguous instead of the last --
    // the convention cuBLAS inherits from Fortran BLAS.
    __host__ __device__ static TensorStrides col_major(const TensorShape& shape) {
        TensorStrides s;
        int running = 1;
        for (int i = 0; i < shape.ndim; i++) {
            s.strides[i] = running;
            running *= shape.dims[i];
        }
        return s;
    }

    __host__ __device__ int offset3(int i0, int i1, int i2) const {
        return i0 * strides[0] + i1 * strides[1] + i2 * strides[2];
    }
};

int main() {
    TensorShape shape(2, 3, 4);
    TensorStrides row = TensorStrides::row_major(shape);
    TensorStrides col = TensorStrides::col_major(shape);

    printf("shape = [2,3,4]\n");
    printf("row-major strides = [%d, %d, %d]\n", row.strides[0], row.strides[1], row.strides[2]);
    printf("col-major strides = [%d, %d, %d]\n", col.strides[0], col.strides[1], col.strides[2]);

    printf("\nindex<->offset round trip for coordinate (1, 1, 2):\n");
    printf("row-major offset(1,1,2) = %d\n", row.offset3(1, 1, 2));
    printf("col-major offset(1,1,2) = %d\n", col.offset3(1, 1, 2));

    printf("\nsame coordinate, opposite contiguity:\n");
    printf("row-major offset(0,0,1) = %d  (innermost dim is contiguous)\n", row.offset3(0, 0, 1));
    printf("col-major offset(0,0,1) = %d  (innermost dim is NOT contiguous here)\n", col.offset3(0, 0, 1));
    printf("row-major offset(1,0,0) = %d  (outermost dim is NOT contiguous here)\n", row.offset3(1, 0, 0));
    printf("col-major offset(1,0,0) = %d  (outermost dim is contiguous)\n", col.offset3(1, 0, 0));

    printf("\nboth layouts agree at the very last coordinate (a permutation always agrees at the boundary):\n");
    printf("row-major offset(1,2,3) = %d, col-major offset(1,2,3) = %d, numel-1 = %d\n",
           row.offset3(1, 2, 3), col.offset3(1, 2, 3), shape.numel() - 1);
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_layout_row_col_major.cu -o 01_layout_row_col_major
./01_layout_row_col_major
```

Produces exactly the output shown in Worked Examples 7.1.1 and 7.1.2 above.

### File: `02_tensor_view_slicing_transpose.cu`

```cpp
#include <cstdio>

struct Shape2D {
    int dims[2];
    __host__ __device__ Shape2D(int d0, int d1) : dims{d0, d1} {}
    __host__ __device__ int numel() const { return dims[0] * dims[1]; }
};

struct Strides2D {
    int strides[2];
    __host__ __device__ Strides2D(int s0, int s1) : strides{s0, s1} {}
    __host__ __device__ static Strides2D row_major(const Shape2D& shape) {
        return Strides2D(shape.dims[1], 1);
    }
};

// TensorView owns NOTHING -- just a raw pointer plus a shape/strides pair
// describing how to read whatever buffer that pointer already points into.
// No constructor allocates; no destructor frees. Chapter 6's Tensor is the
// only thing in this book that ever calls cudaMalloc/cudaFree.
struct TensorView {
    float* data;
    Shape2D shape;
    Strides2D strides;

    __host__ __device__ int offset(int i0, int i1) const {
        return i0 * strides.strides[0] + i1 * strides.strides[1];
    }

    // Transpose: SWAP shape and strides. Zero bytes move; only the map changes.
    __host__ __device__ TensorView transpose() const {
        Shape2D new_shape(shape.dims[1], shape.dims[0]);
        Strides2D new_strides(strides.strides[1], strides.strides[0]);
        return TensorView{data, new_shape, new_strides};
    }

    // A row slice: same strides, a shrunk shape, and a pointer advanced by
    // exactly this row's own starting offset -- again, zero bytes move.
    __host__ __device__ TensorView row_slice(int row) const {
        Shape2D new_shape(1, shape.dims[1]);
        int start = offset(row, 0);
        return TensorView{data + start, new_shape, strides};
    }
};

int main() {
    // A conceptual 3x4 buffer's logical values, laid out row-major: buf[i*4+j]
    // stands in for what Chapter 6's Tensor.data would actually hold on real
    // hardware. This array lives entirely on the host -- only the OFFSET
    // arithmetic below is what this section is actually testing.
    float buf[12];
    for (int i = 0; i < 12; i++) buf[i] = (float)i;

    Shape2D shape(3, 4);
    Strides2D strides = Strides2D::row_major(shape);
    TensorView view{buf, shape, strides};

    printf("original view: shape=[%d,%d], strides=[%d,%d]\n",
           view.shape.dims[0], view.shape.dims[1], view.strides.strides[0], view.strides.strides[1]);
    printf("view(1,2) = buf[%d] = %.1f\n", view.offset(1, 2), buf[view.offset(1, 2)]);

    TensorView t = view.transpose();
    printf("\ntransposed view: shape=[%d,%d], strides=[%d,%d]  (same buffer, same pointer: %s)\n",
           t.shape.dims[0], t.shape.dims[1], t.strides.strides[0], t.strides.strides[1],
           (t.data == view.data) ? "true" : "false");
    printf("t(2,1) = buf[%d] = %.1f  (should equal view(1,2) -- transpose just swaps the coordinate order)\n",
           t.offset(2, 1), buf[t.offset(2, 1)]);

    TensorView r = view.row_slice(1);
    printf("\nrow_slice(1): shape=[%d,%d], data now points at buf+%ld\n",
           r.shape.dims[0], r.shape.dims[1], (long)(r.data - buf));
    printf("r(0,2) = %.1f  (should equal view(1,2) -- same element, reached through the slice)\n", r.data[r.offset(0, 2)]);

    return 0;
}
```

```bash
nvcc -arch=sm_80 02_tensor_view_slicing_transpose.cu -o 02_tensor_view_slicing_transpose
./02_tensor_view_slicing_transpose
```

Produces exactly the output shown in Worked Examples 7.2.1 and 7.2.2 above.

### File: `03_alignment_padding.cu`

```cpp
#include <cstdio>

// Fields interleaved by size -- the compiler pads after each `char` to keep
// the following `int` aligned to a 4-byte boundary.
struct InterleavedHeader {
    char a;   // 1 byte, then 3 bytes of padding
    int  b;   // 4 bytes
    char c;   // 1 byte, then 3 bytes of padding
    int  d;   // 4 bytes
};

// The SAME four fields, grouped largest-to-smallest -- both `int`s land on
// aligned boundaries back to back, and the two `char`s share one padding tail.
struct GroupedHeader {
    int  b;   // 4 bytes
    int  d;   // 4 bytes
    char a;   // 1 byte
    char c;   // 1 byte, then 2 bytes of padding to reach a 4-byte multiple
};

int main() {
    printf("InterleavedHeader (char,int,char,int): sizeof = %zu, alignof = %zu\n",
           sizeof(InterleavedHeader), alignof(InterleavedHeader));
    printf("GroupedHeader     (int,int,char,char): sizeof = %zu, alignof = %zu\n",
           sizeof(GroupedHeader), alignof(GroupedHeader));
    printf("identical fields, %ld fewer bytes just from reordering them\n",
           (long)sizeof(InterleavedHeader) - (long)sizeof(GroupedHeader));

    printf("\nfloat4's alignment, which Chapter 5.2's vectorized LDG.E.128 load depends on:\n");
    printf("alignof(float4) = %zu, sizeof(float4) = %zu\n", alignof(float4), sizeof(float4));

    printf("\ncudaMalloc's own alignment guarantee, genuinely queried:\n");
    float* ptr = nullptr;
    cudaError_t err = cudaMalloc((void**)&ptr, 1024 * sizeof(float));
    printf("cudaMalloc err=%d (%s); ptr=%p\n", err, cudaGetErrorString(err), (void*)ptr);
    return 0;
}
```

```bash
nvcc -arch=sm_80 03_alignment_padding.cu -o 03_alignment_padding
./03_alignment_padding
```

Produces exactly the output shown in Worked Examples 7.3.1 and 7.3.2 above.

### File: `04_broadcast_strides.cu`

```cpp
#include <cstdio>

struct Shape2D {
    int dims[2];
    __host__ __device__ Shape2D(int d0, int d1) : dims{d0, d1} {}
};

struct Strides2D {
    int strides[2];
    __host__ __device__ Strides2D(int s0, int s1) : strides{s0, s1} {}
    __host__ __device__ int offset(int i0, int i1) const {
        return i0 * strides[0] + i1 * strides[1];
    }
};

int main() {
    // A real [3] vector, stored contiguously.
    float vec[3] = {10.0f, 20.0f, 30.0f};

    // Broadcasting vec (shape [3]) against a target shape [4, 3]: the leading
    // dimension (size 4) has NO backing storage of its own, so its stride is
    // set to 0 -- every row reads the SAME 3 underlying floats. Zero bytes
    // are duplicated; only the map claims a shape 4x its real size.
    Shape2D broadcast_shape(4, 3);
    Strides2D broadcast_strides(0, 1);   // stride 0 along the broadcast dimension

    printf("real buffer: vec = [%.1f, %.1f, %.1f]  (3 floats, actually stored)\n",
           vec[0], vec[1], vec[2]);
    printf("broadcast shape = [4, 3], broadcast strides = [%d, %d]\n",
           broadcast_strides.strides[0], broadcast_strides.strides[1]);

    printf("\nevery 'row' of the broadcast view maps to the SAME 3 offsets:\n");
    for (int row = 0; row < broadcast_shape.dims[0]; row++) {
        int o0 = broadcast_strides.offset(row, 0);
        int o1 = broadcast_strides.offset(row, 1);
        int o2 = broadcast_strides.offset(row, 2);
        printf("  row %d -> offsets (%d,%d,%d) -> values (%.1f,%.1f,%.1f)\n",
               row, o0, o1, o2, vec[o0], vec[o1], vec[o2]);
    }

    bool all_rows_match = true;
    int ref0 = broadcast_strides.offset(0, 0), ref1 = broadcast_strides.offset(0, 1), ref2 = broadcast_strides.offset(0, 2);
    for (int row = 1; row < 4; row++) {
        if (broadcast_strides.offset(row, 0) != ref0 || broadcast_strides.offset(row, 1) != ref1 || broadcast_strides.offset(row, 2) != ref2) {
            all_rows_match = false;
        }
    }
    printf("\nall 4 rows alias the identical 3 offsets? %s\n", all_rows_match ? "true" : "false");
    return 0;
}
```

```bash
nvcc -arch=sm_80 04_broadcast_strides.cu -o 04_broadcast_strides
./04_broadcast_strides
```

### Expected Output for `04_broadcast_strides.cu`

```
real buffer: vec = [10.0, 20.0, 30.0]  (3 floats, actually stored)
broadcast shape = [4, 3], broadcast strides = [0, 1]

every 'row' of the broadcast view maps to the SAME 3 offsets:
  row 0 -> offsets (0,1,2) -> values (10.0,20.0,30.0)
  row 1 -> offsets (0,1,2) -> values (10.0,20.0,30.0)
  row 2 -> offsets (0,1,2) -> values (10.0,20.0,30.0)
  row 3 -> offsets (0,1,2) -> values (10.0,20.0,30.0)

all 4 rows alias the identical 3 offsets? true
```

Every number here was independently verified earlier in this chapter, in Worked Example 7.4.1. All four files in this section genuinely compile clean under `nvcc -arch=sm_80` and run to completion in this no-GPU environment — none of them launch a kernel, so none of them touch `cudaErrorNoDevice` except `03_alignment_padding.cu`'s single `cudaMalloc` query, which honestly reports it exactly as every device-touching call in this book has since Chapter 1.3.

## Chapter Summary

The offset formula Chapter 6.4 wrote never changes across this entire chapter — only the strides feeding it do. Column-major strides, computed left-to-right instead of Chapter 6.2's right-to-left, produce a genuinely different memory layout for the identical shape, and matter concretely because cuBLAS expects it. `TensorView` separates *owning* memory from *describing how to read* it: a raw pointer plus a shape and strides, transposable and sliceable at zero cost because swapping or narrowing that description touches no actual bytes — only reshape needs the extra care of checking contiguity first. Struct field order changes `sizeof` through alignment padding alone, with `float4`'s 16-byte alignment (Chapter 5.2) as one specific, already-familiar instance of a general rule that applies to every struct this book writes, `Tensor` included. And broadcasting — setting a dimension's stride to `0` — lets a small buffer legitimately answer for a much larger logical shape, safe to read from freely but genuinely hazardous to write through directly, a distinction Part 4's autodiff engine will need to handle correctly once broadcast inputs need their gradients summed rather than overwritten.

## Self-Check Questions

1. For shape `[5, 2]`, compute both the row-major and column-major strides by hand, following Worked Example 7.1.1's process in both directions.
2. `TensorView::row_slice(row)` advances the pointer by `offset(row, 0)` but keeps the *original* strides unchanged, rather than computing new ones. Explain why the original strides are still correct for indexing within the sliced row.
3. A struct holds one `double` (8 bytes), one `char` (1 byte), and one `double` again (8 bytes), declared in that order. Would reordering it to `double, double, char` change its `sizeof`? Why or why not, in terms of this chapter's padding rule?
4. A broadcast view has shape `[6, 4]` with strides `[0, 1]`, backed by a real 4-element buffer. What does `offset(3, 2)` evaluate to, and what does `offset(5, 2)` evaluate to? Explain why both coordinates are legal even though the view claims 6 rows and only 1 row's worth of memory actually exists.
5. Why is writing through a broadcast view (Section 7.4's Common Trap) a fundamentally different kind of bug than the out-of-bounds indexing Chapter 6.4's Common Trap described, even though both involve a `Tensor`-like struct's raw pointer arithmetic going somewhere unintended?

## Where We Go Next

`Tensor` can now be laid out two different ways, viewed without copying, measured for its own padding, and stretched across a broadcast shape — but every tensor built so far has had its values decided one `printf` at a time. Chapter 8 builds the factory functions — `zeros`, `ones`, `arange`, `full`, and friends — that construct a `Tensor` already filled with a specific pattern of values, the ordinary starting point for essentially every tensor this book's later chapters actually use.

## Worked Solutions

**1.** Row-major, right-to-left: `i=1` sets `strides[1] = 1` (running starts at 1), then `running = 1 × dims[1] = 1 × 2 = 2`; `i=0` sets `strides[0] = 2`. Row-major strides: `[2, 1]`. Column-major, left-to-right: `i=0` sets `strides[0] = 1` (running starts at 1), then `running = 1 × dims[0] = 1 × 5 = 5`; `i=1` sets `strides[1] = 5`. Column-major strides: `[1, 5]`.

**2.** `row_slice(row)`'s new shape is `[1, shape.dims[1]]` — a single row, with the *same* number of columns and the *same* physical spacing between consecutive columns as the original view had. Since the original strides already say "moving one column right skips `strides[1]` flat elements," and a single row's columns are laid out exactly the same way inside the sliced view as they were inside the original, those strides remain correct; only the *starting point* (the pointer) and the *shape* (now just one row) needed to change, not the spacing rule connecting one column to the next.

**3.** No — `sizeof` would be unchanged. `double, char, double` needs padding after the middle `char` (7 bytes, to bring the second `double` back to 8-byte alignment), for `8 + 1 + 7 + 8 = 24` total. `double, double, char` places both 8-byte `double`s back to back with no padding needed between them, then the 1-byte `char`, then padding to bring the total to a multiple of 8: `8 + 8 + 1 + 7 = 24`. Both orderings need exactly 7 bytes of padding somewhere, because there is exactly one 1-byte field needing to be padded up to the struct's 8-byte alignment regardless of where it sits — Worked Example 7.3.1's savings came from having *two* small fields that could share a single padding tail when grouped together, which a single lone `char` here can't do differently no matter where it's placed.

**4.** `offset(3, 2) = 3×0 + 2×1 = 2`. `offset(5, 2) = 5×0 + 2×1 = 2` — identical. Both are legal because the `0` stride on the first dimension means *no* coordinate value in that dimension ever contributes anything to the offset; the view's declared shape, `[6, 4]`, describes how many *logical* coordinates are valid to ask for, not how much *physical* memory backs them, and broadcasting's entire purpose is to let that gap exist deliberately.

**5.** Chapter 6.4's out-of-bounds trap produces an offset that points *outside* the buffer entirely — genuinely undefined behavior, reading or writing memory the `Tensor` never owned in the first place, with a result that depends on whatever happens to occupy that unrelated memory. Writing through a broadcast view produces an offset that points *validly inside* the buffer, every time — the bug isn't that the memory is wrong to touch, it's that *multiple different logical coordinates all resolve to the very same valid address*, so concurrent writes from different threads (or a later write silently overwriting an earlier one) corrupt a real, intended value rather than touching memory nobody meant to touch. One is memory safety; the other is a correctness race on memory the program has every right to be writing to.
