# Chapter 6: The Tensor — Shape, Strides, and Ownership

> "Every worked example so far in this book has used a struct built for exactly one demonstration — a `Point2D`, a `Particle`, a `DeviceBuffer` that exists only to prove a destructor fires in the right order. This chapter builds the struct the rest of this book actually runs on: one type, `Tensor`, that carries its own shape, knows how to turn a coordinate into a flat offset, owns its device memory the way Chapter 2.4 taught, and refuses — at compile time — to let two variables quietly share a buffer neither one thinks it's sharing."

**What you will understand by the end of this chapter:**

- `TensorShape`: a small, fixed-size array of dimension sizes and the one arithmetic fact — `numel = product of dims` — every later chapter's allocation size comes from
- `TensorStrides`: how row-major strides are computed from a shape alone, and why they're the one piece of information that turns a multi-dimensional coordinate into a single flat array offset
- How `Tensor` composes `TensorShape`, `TensorStrides`, and two separate device buffers (`data` and `grad`) into one RAII-managed struct, directly building on Chapter 2.4's `DeviceBuffer` and Chapter 3.5's reason for keeping value and gradient in separate SoA-style arrays
- Indexing: the exact formula, `offset = Σ coord[d] × stride[d]`, that every later chapter's element access compiles down to, and why CUDA never checks it for you
- Why `Tensor`'s copy constructor is deleted rather than compiler-generated — a real compile error this chapter genuinely captures — and how `clone()` and a move constructor together give `Tensor` correct, explicit ownership transfer instead

**What you need to know first:**

- Chapter 2.2 (constructor overloading and value semantics) and Chapter 2.4 (RAII over `cudaMalloc`/`cudaFree`, and its Common Trap foreshadowing a deleted copy constructor plus a move constructor as the real-world fix) — this chapter is where that foreshadowing gets paid off
- Chapter 3.5 (why this book's `Tensor` stores `data` and `grad` as two separate contiguous buffers rather than one array of `{value, grad}` pairs)
- Chapter 1.4 (the genuine, compiler-enforced errors this book captures rather than describes)
- If you've read the Mojo edition: Mojo's Chapter 6 builds `TensorShape`, `TensorStrides`, and `Tensor[dtype]` in the same order this chapter does, then continues in the same file into factory functions, a `DeviceTensor` location tag, and a separate `ValidatedTensor` type with checked indexing. This book's nav gives each of those its own later chapter — Chapter 8 (Tensor Creation Functions), Chapter 10 (Device Abstraction Layer), Chapter 11 (Memory Management System) — so this chapter stays scoped to `TensorShape`, `TensorStrides`, and the core `Tensor` struct's construction, indexing, and ownership, exactly the ground Mojo's Sections 6.1–6.3 cover.

## 6.1 `TensorShape`: A Fixed-Size List of Dimension Sizes `[FOUNDATIONAL]`

### Intuition

A tensor's shape is nothing more than a short list of integers, one per dimension — `[2, 3, 4]` means "2 of the next thing, each containing 3 of the next thing, each containing 4 numbers." The one arithmetic fact this section builds everything else on: the total number of elements a shape describes, `numel`, is simply the product of every dimension size, because each dimension's count multiplies the number of things the dimension "inside" it repeats.

### Background

| | This chapter's `TensorShape` | Chapter 1's primitive types |
|---|---|---|
| Size known at | Compile time — a fixed-capacity array, `MAX_DIMS = 4` | Compile time (Chapter 1.1) |
| Storage | `int dims[4]` plus an `int ndim` recording how many are actually used | A handful of named fields |
| Computed property | `numel()` — the product of the first `ndim` entries of `dims` | N/A |

### Worked Example 6.1.1 — `[2, 3, 4]`, and its element count

```cpp
struct TensorShape {
    static const int MAX_DIMS = 4;
    int dims[MAX_DIMS];
    int ndim;

    __host__ __device__ TensorShape(int d0, int d1, int d2) : dims{d0, d1, d2, 0}, ndim(3) {}

    __host__ __device__ int numel() const {
        int total = 1;
        for (int i = 0; i < ndim; i++) total *= dims[i];
        return total;
    }
};
```

Compiled and run as part of the complete `01_tensor_shape_strides.cu` further below:

```bash
nvcc -arch=sm_80 01_tensor_shape_strides.cu -o 01_tensor_shape_strides
./01_tensor_shape_strides
```

Genuinely compiled and run:

```
shape = [2, 3, 4], ndim = 3
numel() = 24
```

`numel() = 2 × 3 × 4 = 24`, computed by the loop multiplying `dims[0]`, `dims[1]`, and `dims[2]` together — no shortcut, no hardcoded formula, just the general product-of-dims loop that works identically whether `ndim` is 1, 2, or the full `MAX_DIMS = 4`. This is the exact number Section 6.3's constructor uses to decide how many bytes to `cudaMalloc`.

> `[COMMON TRAP]` `MAX_DIMS = 4` is a real, compile-time capacity limit, not a suggestion — `TensorShape` has no dynamic array underneath it to grow, by design, so that a `Tensor`'s shape lives entirely on the stack (or, once a `Tensor` sits inside a kernel argument list, in constant memory) with no heap allocation of its own. A 5-dimensional tensor genuinely cannot be represented by this exact struct; a real framework would either raise `MAX_DIMS` or fall back to a heap-allocated shape for the rare higher-rank case, a tradeoff this book does not need to make for the tensors Parts 2 through 7 actually build.

## 6.2 `TensorStrides`: From a Coordinate to a Flat Offset `[FOUNDATIONAL]`

### Intuition

Underneath its nested, multi-dimensional shape, a tensor is still just one flat, contiguous buffer — Chapter 3 never stopped being true. **Strides** are the missing piece connecting the two pictures: `strides[d]` is exactly how many flat array slots you skip when you move one step along dimension `d`, holding every other dimension fixed. For row-major layout — the only layout this book's `Tensor` uses — the last dimension is the most tightly packed (`stride = 1`), and each dimension moving outward multiplies the previous dimension's stride by that previous dimension's own size.

### Background

| | Computing `strides[ndim-1]` | Computing any earlier `strides[d]` |
|---|---|---|
| Row-major rule | Always `1` — the last dimension is contiguous | `strides[d] = strides[d+1] × dims[d+1]` |
| Direction computed | N/A (base case) | Right to left — each stride depends on the one after it |

### Worked Example 6.2.1 — strides for the same `[2, 3, 4]` shape

```cpp
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
};
```

Compiled and run as part of the complete `01_tensor_shape_strides.cu` further below:

```bash
nvcc -arch=sm_80 01_tensor_shape_strides.cu -o 01_tensor_shape_strides
./01_tensor_shape_strides
```

Genuinely compiled and run:

```
strides = [12, 4, 1]
```

Traced right to left: `i=2` (the last dimension) sets `strides[2] = 1` (the running product starts at 1), then updates `running = 1 × dims[2] = 1 × 4 = 4`. `i=1` sets `strides[1] = running = 4`, then updates `running = 4 × dims[1] = 4 × 3 = 12`. `i=0` sets `strides[0] = running = 12`. Result: `[12, 4, 1]` — moving one step along dimension 0 (the outermost) skips 12 flat elements, exactly the size of one whole `[3,4]` sub-tensor; moving one step along dimension 2 (the innermost) skips exactly 1, confirming the last dimension really is the contiguous one.

### ASCII Diagram — nested shape, one flat buffer underneath

```
Logical shape [2, 3, 4]:                Flat buffer, 24 elements, strides=[12,4,1]:
 [ [ [.,.,.,.], [.,.,.,.], [.,.,.,.] ],   [ 0  1  2  3 | 4  5  6  7 | 8  9  10 11 |
   [ [.,.,.,.], [.,.,.,.], [.,.,.,.] ] ]   12 13 14 15 | 16 17 18 19 | 20 21 22 23 ]
                                            \___________________________________/
 dim 0 (size 2): jump by 12 to move to    \___________/
 the second [3,4] block                  dim 1 (size 3): jump by 4 to move
                                          to the next [4]-row within a block
```

> `[COMMON TRAP]` Reading a multi-dimensional coordinate's flat offset is a *sum* across every dimension, not a single stride lookup: `offset = i0×strides[0] + i1×strides[1] + i2×strides[2]`, all three terms added together. It's a common mistake to reach for just the innermost dimension's stride (`strides[2] = 1`) and treat the coordinate's last index as "the" offset, forgetting that `i0` and `i1` each contribute their own, much larger jump — Section 6.4 makes this formula concrete and checks it against hand-computed values.

## 6.3 Composing `Tensor`: Shape, Strides, and Two Owned Buffers `[FOUNDATIONAL]`

### Intuition

Chapter 2.4's `DeviceBuffer` owned one `cudaMalloc`'d pointer and freed it automatically at scope exit. This book's actual `Tensor` needs exactly that same discipline, twice over: once for the tensor's own values (`data`), and once for the gradient Part 4's autodiff engine will eventually accumulate into (`grad`) — kept as a second, separate buffer rather than interleaved with `data`, for precisely the warp-coalescing reason Chapter 3.5 already argued in full. `Tensor` is nothing conceptually new at this point; it's `TensorShape` plus `TensorStrides` plus two `DeviceBuffer`-style owned allocations, composed into one struct.

### Background

| Field | Type | Role |
|---|---|---|
| `shape` | `TensorShape` | How many elements, along how many dimensions (Section 6.1) |
| `strides` | `TensorStrides` | How to turn a coordinate into a flat offset (Section 6.2) |
| `data` | `float*`, device-owned | The tensor's own values — `shape.numel()` floats, `cudaMalloc`'d in the constructor |
| `grad` | `float*`, device-owned | Space for a gradient of the same shape — allocated alongside `data`, unused until Part 4 |

### Worked Example 6.3.1 — one constructor call, two allocations, the familiar honest disclosure

```cpp
struct Tensor {
    TensorShape shape;
    TensorStrides strides;
    float* data;
    float* grad;

    Tensor(TensorShape shape_) : shape(shape_), strides(TensorStrides::row_major(shape_)), data(nullptr), grad(nullptr) {
        int n = shape.numel();
        cudaError_t e1 = cudaMalloc((void**)&data, n * sizeof(float));
        cudaError_t e2 = cudaMalloc((void**)&grad, n * sizeof(float));
        printf("Tensor(numel=%d) constructed -> data err=%d, grad err=%d\n", n, e1, e2);
    }

    ~Tensor() {
        cudaFree(data);
        cudaFree(grad);
    }
};
```

Compiled and run as the complete `02_tensor_struct.cu` further below:

```bash
nvcc -arch=sm_80 02_tensor_struct.cu -o 02_tensor_struct
./02_tensor_struct
```

Genuinely compiled and run:

```
constructing t
Tensor(numel=24) constructed -> cudaMalloc(data) err=100, cudaMalloc(grad) err=100 (no CUDA-capable device is detected)
t.shape.numel() = 24
t.strides = [12, 4, 1]
leaving main, destructor about to fire
~Tensor(numel=24) destructor firing, freeing data=(nil), grad=(nil)
```

One constructor call, `Tensor t(TensorShape(2, 3, 4))`, genuinely triggers two independent `cudaMalloc` calls — `numel() = 24` computed once (Section 6.1's formula) and reused for both allocation sizes — and both honestly report `cudaErrorNoDevice` in this no-GPU environment, exactly Chapter 2.4's `DeviceBuffer` pattern, just doubled. The destructor frees both pointers; since both stayed `nullptr` here, both `cudaFree` calls are the same safe no-op Chapter 2.4 already established.

> `[COMMON TRAP]` `Tensor` now owns *two* device pointers instead of `DeviceBuffer`'s one — which means the compiler-generated copy constructor Chapter 2.2 explained (copy every field verbatim) would duplicate *both* addresses, setting up not one but two independent double-frees the moment two `Tensor` variables' destructors both run against the same underlying memory. This is exactly why Section 6.5 deletes `Tensor`'s copy constructor outright, rather than hoping nobody ever writes `Tensor t2 = t1;`.

## 6.4 Indexing: The Offset Formula, Made Concrete `[FOUNDATIONAL]`

### Intuition

Section 6.2's Common Trap named the formula; this section runs it. `Tensor::offset(i0, i1, i2)` takes a three-dimensional coordinate and returns the single flat index Section 6.3's `data` array should be read or written at — pure arithmetic on `int`s, touching no memory at all, which means it's one of the few `Tensor` operations this book can genuinely run to completion and check by hand without needing a GPU, exactly like Chapter 1.1's stack-only worked examples.

### Background

| Coordinate | Formula | For shape `[2,3,4]`, strides `[12,4,1]` |
|---|---|---|
| `(0, 0, 0)` | `0×12 + 0×4 + 0×1` | `0` — the very first flat element |
| `(0, 0, 1)` | `0×12 + 0×4 + 1×1` | `1` — one step along the innermost, contiguous dimension |
| `(1, 2, 3)` | `1×12 + 2×4 + 3×1` | `12+8+3 = 23` — the *last* valid coordinate for this shape |

### Worked Example 6.4.1 — five coordinates, hand-traced and confirmed

```cpp
struct Tensor3D {
    TensorShape shape;
    TensorStrides strides;

    Tensor3D(TensorShape shape_) : shape(shape_), strides(TensorStrides::row_major(shape_)) {}

    __host__ __device__ int offset(int i0, int i1, int i2) const {
        return i0 * strides.strides[0] + i1 * strides.strides[1] + i2 * strides.strides[2];
    }
};
```

Compiled and run as the complete `03_tensor_indexing.cu` further below:

```bash
nvcc -arch=sm_80 03_tensor_indexing.cu -o 03_tensor_indexing
./03_tensor_indexing
```

Genuinely compiled and run:

```
shape = [2,3,4], strides = [12,4,1], numel = 24
offset(0,0,0) = 0
offset(0,0,1) = 1
offset(0,1,0) = 4
offset(1,0,0) = 12
offset(1,2,3) = 23  (the last valid coordinate; numel-1 = 23)
```

Every value matches the Background table's hand trace exactly, and `offset(1,2,3) = 23 = numel() - 1` is not a coincidence: `(1,2,3)` is the largest legal coordinate for a `[2,3,4]` shape (indices `0` and `1` for dimension 0, `0`–`2` for dimension 1, `0`–`3` for dimension 2), and a correct row-major layout guarantees the largest coordinate always maps to the very last flat slot.

> `[COMMON TRAP]` `offset()` performs no bounds checking whatsoever — `t.offset(5, 0, 0)` for this same `[2,3,4]` shape compiles cleanly and returns `5×12 = 60`, a flat index 36 slots past the end of a 24-element buffer, with nothing in the type system or at runtime to stop it. This is a genuinely different tradeoff from a language with checked indexing by default: CUDA C++ trusts the caller completely, in exchange for `offset()` compiling down to the handful of integer multiply-adds Chapter 1's zero-cost-abstraction argument promised, with no hidden branch on every single access. A production `Tensor` would typically add an opt-in, debug-only bounds check (`assert(i0 < shape.dims[0])`, compiled out entirely in a release build) rather than pay for one on every access unconditionally — a decision this book leaves as-is rather than building a second, checked `Tensor` variant, since Parts 2 through 7 write their own indexing carefully enough that the checked variant Mojo's edition builds for classroom emphasis isn't worth this book's added complexity.

## 6.5 Copy, Clone, and Move: Correct Ownership for Two Buffers `[FOUNDATIONAL]`

### Intuition

Section 6.3's Common Trap named the danger; this section closes it the way Chapter 2.4 already promised it would: `Tensor`'s copy constructor and copy-assignment operator are both explicitly `= delete`d, turning "someone copies a `Tensor` by accident" from a silent runtime double-free into a compile error caught before the program ever runs — the same category of guarantee Chapter 1.4's `__device__`/`__host__` boundary check gave you, now protecting ownership instead of execution space. In place of implicit copying, `Tensor` offers two explicit operations with two different meanings: `clone()`, which allocates genuinely new buffers and copies values into them, and a move constructor, which transfers ownership from a temporary without allocating anything at all.

### Background

| | Deleted copy constructor | `clone()` | Move constructor |
|---|---|---|---|
| What happens to `data`/`grad` | Nothing — this path is a compile error | Two new `cudaMalloc`s, then `cudaMemcpy(..., cudaMemcpyDeviceToDevice)` | Pointers copied directly, then zeroed on the source — no allocation, no copy of contents |
| When it runs | Never — caught at compile time | Whenever code explicitly calls `.clone()` | Whenever a `Tensor` is returned by value from a function, as `clone()` itself does |
| Result | N/A | A fully independent `Tensor`, same shape, same values, separate memory | The exact same buffers, now owned by a different variable |

### Worked Example 6.5.1 — the deleted copy constructor, genuinely caught

```cpp
Tensor t1(TensorShape(2, 3, 4));
Tensor t2 = t1;  // ERROR: copy constructor is deleted
```

```bash
nvcc -arch=sm_80 -c err_check_tensor_copy.cu -o err_check_tensor_copy.o
```

Genuinely compiled with `nvcc -arch=sm_80 -c err_check_tensor_copy.cu`:

```
err_check_tensor_copy.cu(26): error: function "Tensor::Tensor(const Tensor &)"
(18): here cannot be referenced -- it is a deleted function

1 error detected in the compilation of "err_check_tensor_copy.cu".
```

This is the exact same class of compile-time enforcement Chapter 1.4.1 captured for a `__device__`-only function called from the host — a mistake that never reaches a running program at all, caught during the same compilation pass that resolves ordinary overloads.

### Worked Example 6.5.2 — `clone()`, the move constructor it depends on, and genuine value independence

```cpp
Tensor(Tensor&& other) noexcept : shape(other.shape), data(other.data), grad(other.grad) {
    other.data = nullptr;
    other.grad = nullptr;
}

Tensor clone() const {
    Tensor copy(shape);
    int bytes = shape.numel() * sizeof(float);
    cudaMemcpy(copy.data, data, bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(copy.grad, grad, bytes, cudaMemcpyDeviceToDevice);
    return copy;  // moved out, not copied -- the move constructor above runs
}
```

Compiled and run as the complete `04_tensor_copy_semantics.cu` further below:

```bash
nvcc -arch=sm_80 04_tensor_copy_semantics.cu -o 04_tensor_copy_semantics
./04_tensor_copy_semantics
```

Genuinely compiled and run:

```
constructing t1
  Tensor(numel=24) constructed -> data err=100, grad err=100
cloning t1 into t2
  clone() called -- allocating two NEW buffers, then copying contents:
  Tensor(numel=24) constructed -> data err=100, grad err=100
  clone() cudaMemcpy(data) err=100, cudaMemcpy(grad) err=100
t1.shape.numel()=24, t2.shape.numel()=24 (independent copies, same shape)

host-side stand-in, proving the VALUE independence clone() is built for:
before mutation: original = [1.0,2.0,3.0,4.0], copy = [1.0,2.0,3.0,4.0]
after copy.values[0]=99: original = [1.0,2.0,3.0,4.0], copy = [99.0,2.0,3.0,4.0]

leaving main -- t2 then t1 destructors fire, LIFO:
  ~Tensor(numel=24) destructor firing, freeing data=(nil), grad=(nil)
  ~Tensor(numel=24) destructor firing, freeing data=(nil), grad=(nil)
```

`t1.clone()` genuinely triggers a *second* full `Tensor` construction (a second pair of `cudaMalloc` calls, visibly distinct from `t1`'s own) followed by two `cudaMemcpyDeviceToDevice` calls — the real mechanism, honestly reporting `cudaErrorNoDevice` at every device-touching step in this no-GPU environment, exactly as Chapters 2.4 and 4.2 established. Because this environment can't read back real device bytes to prove numeric independence, the file's `main()` also runs a plain host-side stand-in — `HostTensorStandIn`, an ordinary struct holding a real, host-resident `float[4]` — through the identical clone-then-mutate discipline: `copy.values[0] = 99.0f` changes only `copy`, leaving `original` at its starting values, confirming in concrete numbers the exact independence `clone()`'s device-side `cudaMemcpy` is built to provide once real hardware runs it. `t1` and `t2`'s destructors fire in the reverse of their construction order — `t2` first, then `t1` — the same LIFO discipline Chapter 2.4's `DeviceBuffer` first demonstrated.

> `[COMMON TRAP]` A move constructor being available doesn't mean copying got any cheaper to *write* — it means the one legal way left to duplicate a `Tensor`'s contents, `clone()`, is now explicit and searchable in a codebase, rather than something that could happen silently through an innocent-looking `Tensor t2 = t1;`. Passing a `Tensor` *by value* into an ordinary function is still exactly as much a compile error as Worked Example 6.5.1's assignment, for the same reason — which is precisely why every function this book writes from here forward takes a `Tensor` by reference (`const Tensor&` to read it, `Tensor&` to mutate it) unless it genuinely intends to take ownership.

## 6.6 Complete Runnable Code

### File: `01_tensor_shape_strides.cu`

```cpp
#include <cstdio>

struct TensorShape {
    static const int MAX_DIMS = 4;
    int dims[MAX_DIMS];
    int ndim;

    __host__ __device__ TensorShape() : dims{0,0,0,0}, ndim(0) {}
    __host__ __device__ TensorShape(int d0, int d1, int d2) : dims{d0,d1,d2,0}, ndim(3) {}

    __host__ __device__ int numel() const {
        int total = 1;
        for (int i = 0; i < ndim; i++) total *= dims[i];
        return total;
    }
};

struct TensorStrides {
    int strides[TensorShape::MAX_DIMS];

    __host__ __device__ TensorStrides() : strides{0,0,0,0} {}

    __host__ __device__ static TensorStrides row_major(const TensorShape& shape) {
        TensorStrides s;
        int running = 1;
        for (int i = shape.ndim - 1; i >= 0; i--) {
            s.strides[i] = running;
            running *= shape.dims[i];
        }
        return s;
    }
};

int main() {
    TensorShape shape(2, 3, 4);
    printf("shape = [%d, %d, %d], ndim = %d\n", shape.dims[0], shape.dims[1], shape.dims[2], shape.ndim);
    printf("numel() = %d\n", shape.numel());

    TensorStrides strides = TensorStrides::row_major(shape);
    printf("strides = [%d, %d, %d]\n", strides.strides[0], strides.strides[1], strides.strides[2]);
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_tensor_shape_strides.cu -o 01_tensor_shape_strides
./01_tensor_shape_strides
```

Produces exactly the output shown in Worked Examples 6.1.1 and 6.2.1 above.

### File: `02_tensor_struct.cu`

```cpp
#include <cstdio>

struct TensorShape {
    static const int MAX_DIMS = 4;
    int dims[MAX_DIMS];
    int ndim;

    __host__ __device__ TensorShape() : dims{0,0,0,0}, ndim(0) {}
    __host__ __device__ TensorShape(int d0, int d1, int d2) : dims{d0,d1,d2,0}, ndim(3) {}

    __host__ __device__ int numel() const {
        int total = 1;
        for (int i = 0; i < ndim; i++) total *= dims[i];
        return total;
    }
};

struct TensorStrides {
    int strides[TensorShape::MAX_DIMS];

    __host__ __device__ TensorStrides() : strides{0,0,0,0} {}

    __host__ __device__ static TensorStrides row_major(const TensorShape& shape) {
        TensorStrides s;
        int running = 1;
        for (int i = shape.ndim - 1; i >= 0; i--) {
            s.strides[i] = running;
            running *= shape.dims[i];
        }
        return s;
    }
};

struct Tensor {
    TensorShape shape;
    TensorStrides strides;
    float* data;
    float* grad;

    Tensor(TensorShape shape_) : shape(shape_), strides(TensorStrides::row_major(shape_)), data(nullptr), grad(nullptr) {
        int n = shape.numel();
        cudaError_t e1 = cudaMalloc((void**)&data, n * sizeof(float));
        cudaError_t e2 = cudaMalloc((void**)&grad, n * sizeof(float));
        printf("Tensor(numel=%d) constructed -> cudaMalloc(data) err=%d, cudaMalloc(grad) err=%d (%s)\n",
               n, e1, e2, cudaGetErrorString(e1));
    }

    ~Tensor() {
        printf("~Tensor(numel=%d) destructor firing, freeing data=%p, grad=%p\n", shape.numel(), (void*)data, (void*)grad);
        cudaFree(data);
        cudaFree(grad);
    }
};

int main() {
    printf("constructing t\n");
    Tensor t(TensorShape(2, 3, 4));
    printf("t.shape.numel() = %d\n", t.shape.numel());
    printf("t.strides = [%d, %d, %d]\n", t.strides.strides[0], t.strides.strides[1], t.strides.strides[2]);
    printf("leaving main, destructor about to fire\n");
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_tensor_struct.cu -o 02_tensor_struct
./02_tensor_struct
```

Produces exactly the output shown in Worked Example 6.3.1 above.

### File: `03_tensor_indexing.cu`

```cpp
#include <cstdio>

struct TensorShape {
    static const int MAX_DIMS = 4;
    int dims[MAX_DIMS];
    int ndim;

    __host__ __device__ TensorShape() : dims{0,0,0,0}, ndim(0) {}
    __host__ __device__ TensorShape(int d0, int d1, int d2) : dims{d0,d1,d2,0}, ndim(3) {}

    __host__ __device__ int numel() const {
        int total = 1;
        for (int i = 0; i < ndim; i++) total *= dims[i];
        return total;
    }
};

struct TensorStrides {
    int strides[TensorShape::MAX_DIMS];

    __host__ __device__ TensorStrides() : strides{0,0,0,0} {}

    __host__ __device__ static TensorStrides row_major(const TensorShape& shape) {
        TensorStrides s;
        int running = 1;
        for (int i = shape.ndim - 1; i >= 0; i--) {
            s.strides[i] = running;
            running *= shape.dims[i];
        }
        return s;
    }
};

struct Tensor3D {
    TensorShape shape;
    TensorStrides strides;

    Tensor3D(TensorShape shape_) : shape(shape_), strides(TensorStrides::row_major(shape_)) {}

    // Pure arithmetic -- no data pointer touched, so this genuinely runs
    // to completion with no GPU involved.
    __host__ __device__ int offset(int i0, int i1, int i2) const {
        return i0 * strides.strides[0] + i1 * strides.strides[1] + i2 * strides.strides[2];
    }
};

int main() {
    Tensor3D t(TensorShape(2, 3, 4));
    printf("shape = [2,3,4], strides = [%d,%d,%d], numel = %d\n",
           t.strides.strides[0], t.strides.strides[1], t.strides.strides[2], t.shape.numel());

    printf("offset(0,0,0) = %d\n", t.offset(0, 0, 0));
    printf("offset(0,0,1) = %d\n", t.offset(0, 0, 1));
    printf("offset(0,1,0) = %d\n", t.offset(0, 1, 0));
    printf("offset(1,0,0) = %d\n", t.offset(1, 0, 0));
    printf("offset(1,2,3) = %d  (the last valid coordinate; numel-1 = %d)\n", t.offset(1, 2, 3), t.shape.numel() - 1);
    return 0;
}
```

```bash
nvcc -arch=sm_80 03_tensor_indexing.cu -o 03_tensor_indexing
./03_tensor_indexing
```

Produces exactly the output shown in Worked Example 6.4.1 above.

### File: `04_tensor_copy_semantics.cu`

```cpp
#include <cstdio>

struct TensorShape {
    int dims[4];
    int ndim;
    __host__ __device__ TensorShape(int d0, int d1, int d2) : dims{d0,d1,d2,0}, ndim(3) {}
    __host__ __device__ int numel() const { int t = 1; for (int i = 0; i < ndim; i++) t *= dims[i]; return t; }
};

struct Tensor {
    TensorShape shape;
    float* data;
    float* grad;

    Tensor(TensorShape shape_) : shape(shape_), data(nullptr), grad(nullptr) {
        int n = shape.numel();
        cudaError_t e1 = cudaMalloc((void**)&data, n * sizeof(float));
        cudaError_t e2 = cudaMalloc((void**)&grad, n * sizeof(float));
        printf("  Tensor(numel=%d) constructed -> data err=%d, grad err=%d\n", n, e1, e2);
    }

    // Compiler-generated copy would duplicate `data`/`grad` (the addresses),
    // not the memory they point to -- Chapter 2.4's DeviceBuffer trap, now with
    // two pointers to double-free instead of one. Deleting both forces every
    // copy through the explicit, correct clone() below instead.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move construction transfers ownership instead of duplicating it --
    // exactly the fix Chapter 2.4's Common Trap named as "the far more common
    // real-world choice." clone()'s return value relies on this.
    Tensor(Tensor&& other) noexcept : shape(other.shape), data(other.data), grad(other.grad) {
        other.data = nullptr;
        other.grad = nullptr;
    }

    Tensor clone() const {
        printf("  clone() called -- allocating two NEW buffers, then copying contents:\n");
        Tensor copy(shape);
        int bytes = shape.numel() * sizeof(float);
        cudaError_t e1 = cudaMemcpy(copy.data, data, bytes, cudaMemcpyDeviceToDevice);
        cudaError_t e2 = cudaMemcpy(copy.grad, grad, bytes, cudaMemcpyDeviceToDevice);
        printf("  clone() cudaMemcpy(data) err=%d, cudaMemcpy(grad) err=%d\n", e1, e2);
        return copy;  // moved out, not copied -- the move constructor above runs
    }

    ~Tensor() {
        printf("  ~Tensor(numel=%d) destructor firing, freeing data=%p, grad=%p\n",
               shape.numel(), (void*)data, (void*)grad);
        cudaFree(data);
        cudaFree(grad);
    }
};

// A plain host-side stand-in for the exact same deep-copy discipline Tensor's
// clone() performs on device memory -- used here only because this no-GPU
// environment can't read back real device bytes to prove independence
// numerically. The mechanism (allocate new storage, copy values, never share
// a pointer) is identical; only the memory space differs.
struct HostTensorStandIn {
    float values[4];
    HostTensorStandIn(float a, float b, float c, float d) : values{a, b, c, d} {}
    HostTensorStandIn clone() const {
        return HostTensorStandIn(values[0], values[1], values[2], values[3]);
    }
};

int main() {
    printf("constructing t1\n");
    Tensor t1(TensorShape(2, 3, 4));
    printf("cloning t1 into t2\n");
    Tensor t2 = t1.clone();
    printf("t1.shape.numel()=%d, t2.shape.numel()=%d (independent copies, same shape)\n",
           t1.shape.numel(), t2.shape.numel());

    printf("\nhost-side stand-in, proving the VALUE independence clone() is built for:\n");
    HostTensorStandIn original(1.0f, 2.0f, 3.0f, 4.0f);
    HostTensorStandIn copy = original.clone();
    printf("before mutation: original = [%.1f,%.1f,%.1f,%.1f], copy = [%.1f,%.1f,%.1f,%.1f]\n",
           original.values[0], original.values[1], original.values[2], original.values[3],
           copy.values[0], copy.values[1], copy.values[2], copy.values[3]);
    copy.values[0] = 99.0f;
    printf("after copy.values[0]=99: original = [%.1f,%.1f,%.1f,%.1f], copy = [%.1f,%.1f,%.1f,%.1f]\n",
           original.values[0], original.values[1], original.values[2], original.values[3],
           copy.values[0], copy.values[1], copy.values[2], copy.values[3]);

    printf("\nleaving main -- t2 then t1 destructors fire, LIFO:\n");
    return 0;
}
```

```bash
nvcc -arch=sm_80 04_tensor_copy_semantics.cu -o 04_tensor_copy_semantics
./04_tensor_copy_semantics
```

### Expected Output for `04_tensor_copy_semantics.cu`

```
constructing t1
  Tensor(numel=24) constructed -> data err=100, grad err=100
cloning t1 into t2
  clone() called -- allocating two NEW buffers, then copying contents:
  Tensor(numel=24) constructed -> data err=100, grad err=100
  clone() cudaMemcpy(data) err=100, cudaMemcpy(grad) err=100
t1.shape.numel()=24, t2.shape.numel()=24 (independent copies, same shape)

host-side stand-in, proving the VALUE independence clone() is built for:
before mutation: original = [1.0,2.0,3.0,4.0], copy = [1.0,2.0,3.0,4.0]
after copy.values[0]=99: original = [1.0,2.0,3.0,4.0], copy = [99.0,2.0,3.0,4.0]

leaving main -- t2 then t1 destructors fire, LIFO:
  ~Tensor(numel=24) destructor firing, freeing data=(nil), grad=(nil)
  ~Tensor(numel=24) destructor firing, freeing data=(nil), grad=(nil)
```

Every number here was independently verified earlier in this chapter, and Worked Example 6.5.1's genuine compile error (`err_check_tensor_copy.cu`, a small file containing only the illegal `Tensor t2 = t1;`, deliberately not included as a fifth runnable file since it exists specifically to fail `nvcc -c`) confirms `Tensor`'s deleted copy constructor is caught before any of these four files' `main()` functions ever run. All four files in this section genuinely compile clean under `nvcc -arch=sm_80` and run to completion in this no-GPU environment, honestly reporting `cudaErrorNoDevice` at every device-touching call.

## Chapter Summary

`TensorShape` is a fixed-capacity list of dimension sizes with one derived property, `numel`, computed as the product of every dimension — the number this chapter's allocations, and every later chapter's kernel launch, size themselves against. `TensorStrides` converts that nested shape into the single fact needed to navigate a flat buffer: how many elements to skip per dimension, computed right-to-left so the last dimension is always contiguous, exactly the property Chapter 3's coalescing argument depends on. `Tensor` composes both alongside two independently owned device buffers, `data` and `grad`, each `cudaMalloc`'d and `cudaFree`'d through the same RAII discipline Chapter 2.4 established — genuinely exercised and honestly reporting `cudaErrorNoDevice` in this no-GPU environment, twice per construction instead of once. Indexing is pure, uncontained arithmetic — `offset = Σ coord[d] × stride[d]` — fast because nothing checks it, and dangerous for exactly the same reason. And because `Tensor` now owns two pointers instead of one, this chapter finally deletes the copy constructor Chapter 2.4 warned about, replacing it with an explicit `clone()` for genuine deep copies and a move constructor for cheap ownership transfer — turning "someone copied a `Tensor` by accident" from a silent double-free into a compile error this chapter captured directly.

## Self-Check Questions

1. `TensorShape`'s `numel()` for shape `[5, 1, 3]` — walk through the loop and compute the result.
2. For a shape `[10, 20]` (2-dimensional), compute `TensorStrides::row_major`'s two stride values by hand, following the same right-to-left process Worked Example 6.2.1 traced.
3. Using the strides you computed in Question 2, what is `offset(3, 7)` for that same `[10, 20]` shape? What is the largest legal coordinate, and what offset does it produce?
4. `Tensor`'s constructor calls `cudaMalloc` twice — once for `data`, once for `grad` — every single time a `Tensor` is constructed, even in Part 2's element-wise operations where no gradient is ever computed. Why does this chapter's design still allocate `grad` unconditionally rather than only when it's needed?
5. Explain, in terms of what `Tensor`'s copy constructor would have to do if it weren't deleted, why the compiler cannot simply generate a correct one automatically the way it did for `Point2D` in Chapter 2.2.

## Where We Go Next

`Tensor` can now be constructed, indexed by hand, and safely copied only through `clone()` or a move — but everything constructed so far has used one hardcoded three-dimensional shape and one hardcoded row-major layout. Chapter 7 goes deeper into memory layout itself: strides for arbitrary layouts (including non-default ones), views that reinterpret the same buffer without copying it, and the alignment guarantees a `Tensor`'s buffers need to support Chapter 5's vectorized loads.

## Worked Solutions

**1.** `numel()` initializes `total = 1`, then loops over `ndim = 3` dimensions: `total = 1 × 5 = 5`, then `total = 5 × 1 = 5`, then `total = 5 × 3 = 15`. Result: `15`.

**2.** Following Worked Example 6.2.1's right-to-left process for shape `[10, 20]` (`ndim = 2`): `i=1` (last dimension) sets `strides[1] = 1` (running starts at 1), then updates `running = 1 × dims[1] = 1 × 20 = 20`. `i=0` sets `strides[0] = running = 20`. Result: `strides = [20, 1]`.

**3.** `offset(3, 7) = 3×20 + 7×1 = 60 + 7 = 67`. The largest legal coordinate for shape `[10, 20]` is `(9, 19)` (indices `0`–`9` for dimension 0, `0`–`19` for dimension 1): `offset(9, 19) = 9×20 + 19×1 = 180 + 19 = 199 = numel() - 1 = (10×20) - 1 = 199`, confirming the same last-coordinate-maps-to-last-slot property Worked Example 6.4.1 established for `[2,3,4]`.

**4.** Allocating `grad` unconditionally keeps `Tensor` a single, uniform type with one constructor and one destructor path, which is exactly what lets every later chapter's function accept a plain `Tensor&` without needing to ask "does this one have a gradient buffer or not?" first. The cost is one extra `cudaMalloc` per tensor that never needs backpropagation — real, but small relative to the complexity of maintaining two different `Tensor`-shaped types (one with `grad`, one without) throughout the rest of this book; Part 4's autodiff engine is the first place this chapter's unconditional allocation actually gets used.

**5.** A compiler-generated copy constructor for `Point2D` (Chapter 2.2) copies two `float` fields verbatim, and a `float` genuinely can be duplicated correctly by copying its bits — there's no shared ownership question, because a `float` doesn't *point at* anything. `Tensor`'s `data` and `grad` fields are pointers to heap-allocated device memory; a compiler-generated copy would copy the pointer *values* (the addresses), leaving two `Tensor` instances that both believe they exclusively own the same two allocations. The compiler has no way to know, from the type `float*` alone, whether "copy this field verbatim" or "allocate new memory and copy the pointed-to contents" is the correct behavior — that decision depends on the field's *meaning* (an owned resource, not just data), which is exactly the kind of judgment call C++ leaves to the programmer, enforced here by deleting the ambiguous default outright.
