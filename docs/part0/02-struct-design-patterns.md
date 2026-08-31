# Chapter 2: Struct Design Patterns

> "Every type Chapter 1 didn't have to invent — `int`, `float`, `float4` — was built by someone else out of the same tool this chapter hands you: a fixed, named layout of memory, plus the functions that know what to do with it. A `struct` is not a container for data. It is a promise about a layout, exactly the way `float4` was a promise about sixteen bytes — except now you write the promise yourself, and Chapter 1's host/device split applies to every method you attach to it."

**What you will understand by the end of this chapter:**

- What a `struct` actually is in memory — a fixed, contiguous, compile-time-known layout of named fields — and why attaching `__host__ __device__` to its methods is the direct, necessary extension of Chapter 1.4's qualifier rules to user-defined types
- How constructor overloading lets one struct offer several ways to be built, resolved entirely at compile time, with real value-semantics copying behavior
- How a C++ **template** (`template<typename T, int N> struct Vector`) gets compiled into a separate, fully specialized version of itself for every distinct combination of template arguments it's used with — genuinely verifiable as two different machine-code symbols in one compiled binary
- How constructor/destructor pairs give a struct **RAII** — automatic, scope-tied cleanup of `cudaMalloc`-allocated device memory, closing the exact gap Chapter 1 left open by never calling `cudaFree`
- Why virtual functions and vtables, while they compile cleanly for device code, hide a genuine, well-documented hazard specific to CUDA — a vtable pointer built by the host constructor is not something device code can safely dereference — and why the **CRTP** (Curiously Recurring Template Pattern) gives you an interface with the same compile-time-checked guarantee as Mojo's `trait`, with no vtable and no host/device hazard at all

**What you need to know first:**

- Chapter 1 in full — this chapter reuses its stack-layout reasoning (Section 1.1), its `auto`/type-inference distinction (Section 1.2), its host/device compiler-split model (Section 1.3), and its `__host__`/`__device__`/`__global__` qualifier rules (Section 1.4) without re-explaining them
- If you've read the Mojo edition: this chapter follows its Chapter 2 section-for-section (struct layout, multiple constructors, parametric structs, RAII, trait-equivalent interfaces) — the substitutions are C++ templates for Mojo's parametric structs and CRTP for Mojo's `trait`, since C++ has no first-class trait keyword of its own

## 2.1 What a Struct Actually Is `[FOUNDATIONAL]`

### Intuition

An architectural blueprint doesn't describe *a* house — it describes the layout every house built from it will share: the front door on the north wall, the kitchen four meters from it. A `struct` is that blueprint applied to memory instead of a plot of land. `struct Point2D { float x; float y; };` fixes that field `x` sits at byte offset 0 and field `y` sits at byte offset 4, for every `Point2D` that will ever exist in the program. CUDA C++ adds one thing a plain-C++ blueprint doesn't need to worry about: Chapter 1.4 already established that a *function* is compiled by one pipeline, the other, or both, depending on its qualifiers — and a struct's methods are ordinary functions, so the same rule applies to every one of them, individually.

### Background

| | Plain C `struct` | CUDA C++ `struct` |
|---|---|---|
| Field layout | Fixed offsets, known at compile time | Fixed offsets, known at compile time — identical |
| Methods | Not supported (free functions instead) | Attached directly to the type, each independently `__host__`, `__device__`, or both |
| Usable from a kernel? | N/A | Only if every method the kernel calls is `__device__` or `__host__ __device__` |
| Usable from host setup code? | Yes | Only if every method the host calls is `__host__` or `__host__ __device__` |

A struct with even one method marked `__device__`-only cannot have that method called from `main()` — Chapter 1.4's Worked Example 1.4.1 already showed this exact compile error for a free function, and it applies identically to a member function.

### Worked Example 2.1.1 — the exact byte layout, and a method usable from both sides

```cpp
struct Point2D {
    float x;
    float y;

    __host__ __device__ Point2D(float x_, float y_) : x(x_), y(y_) {}

    __host__ __device__ float distance_from_origin() const {
        return sqrtf(x * x + y * y);
    }

    __host__ __device__ float distance_to(const Point2D& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        return sqrtf(dx * dx + dy * dy);
    }
};
```

Compiled and run as the complete `01_basic_structs.cu` further below, which wraps this struct in a `main()` constructing `point1`/`point2`:

```bash
nvcc -arch=sm_80 01_basic_structs.cu -o 01_basic_structs
./01_basic_structs
```

Genuinely compiled and run:

```
sizeof(Point2D) = 8
point1.distance_from_origin() = 5.000000
point1.distance_to(point2) = 3.605551
```

`sizeof(Point2D) == 8` because two `float` fields at 4 bytes each pack with no padding (both are 4-byte-aligned already). `point1 = Point2D(3.0f, 4.0f)`: `distance_from_origin()` computes `sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5.0` — the classic 3-4-5 triangle. `point2 = Point2D(1.0f, 1.0f)`: `distance_to` computes `dx = 3-1 = 2`, `dy = 4-1 = 3`, `sqrt(4 + 9) = sqrt(13) ≈ 3.6056`. Every method here is `__host__ __device__`, which means — per Chapter 1.4's Worked Example 1.4.2 — `nvcc` genuinely compiles each one *twice*: once as ordinary host machine code (what actually executed above) and once as device machine code, ready for a kernel that constructs and uses `Point2D` instances directly on the GPU, with no source-level difference at all.

### ASCII Diagram — one struct, two instances, identical layout

```
point1 (Point2D):              point2 (Point2D):
 +--------------------+         +--------------------+
 | x (offset+0): 3.0  |         | x (offset+0): 1.0  |
 | y (offset+4): 4.0  |         | y (offset+4): 1.0  |
 +--------------------+         +--------------------+
       8 bytes total                  8 bytes total

distance_to reads other.x at (other's base address + 0) and other.y at
(other's base address + 4) -- the SAME relative offsets point1 uses for
its own fields, because every Point2D shares one compile-time layout.
```

> `[COMMON TRAP]` It's tempting to assume a struct method is "host code by default, like a plain function." A struct method with no qualifier at all defaults to `__host__` exactly like a free function does — but forgetting `__host__ __device__` on a method a kernel needs is a compile error at the *call site inside the kernel*, not at the struct's own definition, which can make the error look unrelated to the struct until you trace it back.

### Worked Example 2.1.2 — the same method, called from a `__global__` kernel via a launch configuration

Worked Example 2.1.1's claim — that `nvcc` compiles `distance_from_origin()` a second time, "ready for a kernel that constructs and uses `Point2D` instances directly on the GPU" — is worth cashing in rather than leaving as an assertion. A `__global__` kernel constructs one `Point2D` per thread and calls the identical `__host__ __device__` method main() already called on the host, launched with an explicit **launch configuration**, `<<<1, N>>>` — one block of `N` threads, so `threadIdx.x` alone indexes every point:

```cpp
__global__ void distance_kernel(float* out, const float* xs, const float* ys, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        Point2D p(xs[idx], ys[idx]);          // constructed ON THE DEVICE
        out[idx] = p.distance_from_origin();   // the SAME method main() calls on the host
    }
}
```

Compiled and run as part of the same `01_basic_structs.cu` further below:

```
host-side reference (same method, called on the host):
  Point2D(3.0, 4.0).distance_from_origin() = 5.000000
  Point2D(0.0, 0.0).distance_from_origin() = 0.000000
  Point2D(1.0, 1.0).distance_from_origin() = 1.414214
  Point2D(5.0, 12.0).distance_from_origin() = 13.000000

cudaMalloc(d_xs): cudaErrorNoDevice, cudaMalloc(d_ys): cudaErrorNoDevice, cudaMalloc(d_out): cudaErrorNoDevice
cudaMemcpy(xs H2D): cudaErrorNoDevice, cudaMemcpy(ys H2D): cudaErrorNoDevice

launching distance_kernel<<<1, 4>>> (one block, 4 threads, one Point2D per thread)
kernel launch: cudaErrorNoDevice
cudaMemcpy(out D2H): cudaErrorNoDevice
device-computed results (all zero: the copy back never received real device output,
because no CUDA-capable device exists in this sandbox -- genuinely attempted, honestly
failed, exactly like every device-touching call in this book from Chapter 1 onward):
  h_out[0] = 0.000000
  h_out[1] = 0.000000
  h_out[2] = 0.000000
  h_out[3] = 0.000000
```

The host-side reference loop reuses `distance_from_origin()` on the same four points the kernel is about to process, confirming the 3-4-5 and 5-12-13 triangles by hand (`5.0` and `13.0`) before the kernel is even launched — the point of comparison the kernel's own results would need to match on real hardware. `distance_kernel<<<1, N>>>(...)` is the launch configuration itself: the first argument is the number of blocks, the second the number of threads per block, and `nvcc` expands this into the same `__cudaLaunchPrologue`/`__cudaLaunch` pair Chapter 1.3 already showed for `hello_kernel<<<2, 4>>>()`. Every device-touching call here — `cudaMalloc`, `cudaMemcpy`, the kernel launch itself, and the copy back — genuinely executes and honestly reports `cudaErrorNoDevice`, so `h_out` never receives real values; what's genuinely verified is that `Point2D`'s constructor and `distance_from_origin()` compile cleanly *inside* `distance_kernel`, and that the launch configuration syntax itself is well-formed and accepted, not that a GPU actually ran it.

> `[COMMON TRAP]` `idx = threadIdx.x` alone (rather than `blockIdx.x * blockDim.x + threadIdx.x`, the pattern Appendix G's kernel uses) is only correct because this launch uses a single block, `<<<1, N>>>`. The moment more than one block is launched, `threadIdx.x` resets to `0` at the start of every block, and omitting `blockIdx.x * blockDim.x` would silently make every block's threads overwrite the same low indices — a bug that a single-block launch like this one can't expose, which is exactly why the general form is worth defaulting to even when, as here, one block happens to be enough.

## 2.2 Multiple Constructors and Value Semantics `[FOUNDATIONAL]`

### Intuition

A furniture kit can ship two ways: fully pre-assembled, or flat-packed with a set of standard parts you assemble yourself. Both end in the same finished chair, through different starting inputs. **Constructor overloading** is exactly this: one `struct` can offer several different `Point2D(...)` signatures — one taking two `float`s, one copying an existing `Point2D`, one defaulting both fields to zero — and the compiler picks the matching one at compile time based purely on the arguments at the call site, the same overload-resolution machinery that picks between any two overloaded functions.

### Background

| | What it does | When it runs |
|---|---|---|
| `Point2D(float, float)` | Builds from two explicit coordinates | Whenever two `float`-convertible arguments are given |
| `Point2D()` | Default-constructs, typically zero-initialized | Whenever no arguments are given |
| `Point2D(const Point2D&)` (the copy constructor) | Copies every field, byte-for-byte, from an existing instance | Whenever a `Point2D` is initialized *from* another `Point2D` |

Unless you write your own copy constructor, C++ generates one automatically that copies every field verbatim — for `Point2D`'s two `float`s, this is a genuine, correct, complete copy. This stops being automatically correct the moment a struct owns a pointer (Section 2.4's subject exactly): the compiler-generated copy constructor copies the *pointer value*, not the memory it points to, leaving two structs that both believe they own the same allocation.

### Worked Example 2.2.1 — two constructors, traced side by side

```cpp
Point2D point1(3.0f, 4.0f);   // two-argument constructor
Point2D point2 = point1;      // copy constructor: point2.x=3.0, point2.y=4.0, independently
point2.x = 99.0f;             // mutating point2 does not touch point1
```

Compiled and run as the same `01_basic_structs.cu` further below, whose `main()` continues past Worked Example 2.1.1's lines into exactly this copy-constructor trace:

```bash
nvcc -arch=sm_80 01_basic_structs.cu -o 01_basic_structs
./01_basic_structs
```

Genuinely compiled and run:

```
point1 = (3.000000, 4.000000)
point2 (copy) = (3.000000, 4.000000)
after point2.x = 99.0: point1 = (3.000000, 4.000000), point2 = (99.000000, 4.000000)
```

`point2`'s fields start identical to `point1`'s because the compiler-generated copy constructor copied both `float`s verbatim — genuinely independent 4-byte values at two different stack addresses. Mutating `point2.x` afterward leaves `point1.x` at its original `3.0`, confirming the copy was real and not a reference: this is **value semantics**, the default for a plain-data C++ struct, in contrast to a pointer or reference, which would let a mutation through one name be visible through the other.

## 2.3 Templates and Compile-Time Specialization `[FOUNDATIONAL]`

### Intuition

A cookie cutter shaped like a star doesn't produce *one* cookie — it produces a whole batch, each one from different dough, but every one exactly star-shaped. A C++ **template** is a cookie cutter for *code*: `template<typename T, int N> struct Vector { T data[N]; ... };` isn't compilable code by itself — it's a pattern the compiler stamps out into genuine, independent, fully compiled code the moment you actually use it with concrete arguments, once per distinct combination of arguments, exactly the way Mojo's parametric structs generate one specialized copy of themselves per set of parameters.

### Background

| | Un-templated struct | Templated struct |
|---|---|---|
| Compiled how many times? | Once | Once *per distinct instantiation* (`Vector<float,4>` and `Vector<int,3>` are two separate compiled types) |
| Where are `T` and `N` resolved? | N/A | At compile time, from the angle-bracket arguments at each use site |
| Runtime cost of genericity | N/A | Zero — by the time the program runs, every instantiation is already ordinary, concrete, fully-typed machine code |

### Worked Example 2.3.1 — two specializations, genuinely proven distinct

```cpp
template <typename T, int N>
struct Vector {
    T data[N];

    __host__ __device__ T sum() const {
        T total = T(0);
        for (int i = 0; i < N; i++) total += data[i];
        return total;
    }
};

Vector<float, 4> vf;   // vf.data[i] = {1.0f, 2.0f, 3.0f, 4.0f}
Vector<int, 3> vi;     // vi.data[i] = {10, 20, 30}
```

Compiled and run as the complete `02_templates.cu` further below:

```bash
nvcc -arch=sm_80 02_templates.cu -o 02_templates
./02_templates
```

Genuinely compiled and run:

```
sizeof(Vector<float,4>) = 16, sum = 10.000000
sizeof(Vector<int,3>)   = 12, sum = 60
```

`Vector<float,4>` is 16 bytes (four 4-byte `float`s) and its sum is `1+2+3+4 = 10.0`. `Vector<int,3>` is 12 bytes (three 4-byte `int`s) and its sum is `10+20+30 = 60`. These aren't two runs of the same code with different data — `nm` (a real symbol-table dump of the compiled binary) genuinely shows two entirely separate compiled functions, at two different addresses, one per instantiation:

```bash
nm -C 02_templates | grep "Vector<"
```

```
000000000000ad92 W Vector<float, 4>::sum() const
000000000000adde W Vector<int, 3>::sum() const
```

Nothing about `Vector<int,3>::sum()`'s machine code contains a runtime branch on "am I holding floats or ints" — that question was answered once, at compile time, for each instantiation separately, the same zero-runtime-cost guarantee Chapter 1.2 established for `auto`.

## 2.4 RAII: Constructors, Destructors, and Device Memory `[FOUNDATIONAL]`

### Intuition

A hotel room key that locks the door automatically the moment you drop it in the checkout slot is a form of automation that doesn't rely on you remembering to lock up — the *scope* of your stay is tied directly to the resource's lifetime. **RAII** ("Resource Acquisition Is Initialization") is this pattern applied to memory: a struct's constructor acquires a resource (here, device memory via `cudaMalloc`), and its destructor releases it (`cudaFree`) automatically, the instant the struct's variable goes out of scope — no matter how that scope is exited. Chapter 1 never called `cudaFree` on anything; this section closes that gap.

### Background

| | Manual `cudaMalloc`/`cudaFree` | RAII-wrapped in a struct |
|---|---|---|
| Who calls `cudaFree`? | You, explicitly, at every exit path | The destructor, automatically, at scope exit |
| Forgetting to free | A silent leak — no warning, no crash | Not possible without also forgetting to declare the variable at all |
| Order of cleanup for multiple objects | Whatever order you wrote it in | Strict reverse-of-construction (LIFO) order, enforced by the language |

### Worked Example 2.4.1 — tracing construction and destruction order by hand

```cpp
struct DeviceBuffer {
    float* ptr;
    int size;

    DeviceBuffer(int n) : ptr(nullptr), size(n) {
        cudaError_t err = cudaMalloc((void**)&ptr, n * sizeof(float));
        printf("  DeviceBuffer(%d) constructed -> cudaMalloc err=%d (%s)\n", n, err, cudaGetErrorString(err));
    }

    ~DeviceBuffer() {
        printf("  ~DeviceBuffer(size=%d) destructor firing, freeing ptr=%p\n", size, (void*)ptr);
        cudaFree(ptr);  // safe even if ptr is nullptr
    }
};

void scoped_demo() {
    printf("entering scoped_demo\n");
    DeviceBuffer a(100);
    DeviceBuffer b(200);
    printf("both buffers constructed, about to leave scope\n");
}
```

Compiled and run as the complete `03_raii_device_memory.cu` further below:

```bash
nvcc -arch=sm_80 03_raii_device_memory.cu -o 03_raii_device_memory
./03_raii_device_memory
```

Genuinely compiled and run:

```
entering scoped_demo
  DeviceBuffer(100) constructed -> cudaMalloc err=100 (no CUDA-capable device is detected)
  DeviceBuffer(200) constructed -> cudaMalloc err=100 (no CUDA-capable device is detected)
both buffers constructed, about to leave scope
  ~DeviceBuffer(size=200) destructor firing, freeing ptr=(nil)
  ~DeviceBuffer(size=100) destructor firing, freeing ptr=(nil)
back in main, both destructors already ran
```

Two things here are genuine and worth separating. First, the RAII *mechanism* itself — construction order (`a` then `b`) and destruction order (`b` then `a`, exactly reversed, LIFO) — is ordinary C++ scope semantics, fully exercised and captured in this no-GPU environment, because it never depends on the GPU being present. Second, the `cudaMalloc` *result* — `err=100`, `cudaErrorNoDevice` — is this book's now-familiar honest disclosure from Chapter 1.3: `ptr` stays `nullptr` because the allocation genuinely failed, and `cudaFree(nullptr)` is documented as a safe no-op, which is exactly why the destructor's `cudaFree(ptr)` call doesn't itself produce a second error here. On real hardware, the mechanism is identical and `ptr` would hold a genuine device address instead of staying null — that specific substitution is exactly what this book's real-hardware pass will confirm.

> `[COMMON TRAP]` `DeviceBuffer`'s compiler-generated copy constructor (Section 2.2) would copy the `ptr` field *value* — the address — not the memory it points to. `DeviceBuffer b_copy = b;` followed by both `b` and `b_copy` going out of scope calls `cudaFree` on the *same* address twice: a double-free, undefined behavior on real hardware. A correct `DeviceBuffer` needs either a hand-written copy constructor that performs a real device-to-device copy, or — the far more common real-world choice, and the one this book's own `Tensor` class uses starting in Part 1 — deleting the copy constructor outright (`DeviceBuffer(const DeviceBuffer&) = delete;`) and providing a move constructor instead, so ownership transfers rather than duplicates.

## 2.5 Compile-Time Interfaces: CRTP vs. Virtual Dispatch

### Intuition

Two ways to guarantee "every shape here has an `area()` method." The first: a shared base class declares `area()` as a promise every subclass must keep, checked and dispatched *at runtime* through a hidden lookup table (a **vtable**) — flexible, but that table costs memory and an indirect call. The second: a compile-time trick where each shape tells the compiler its own concrete type up front, so the compiler resolves `area()` directly, at compile time, with nothing left to look up at runtime at all. Mojo's `trait` is the second kind — checked at compile time, zero runtime cost, no inheritance hierarchy required — and C++'s equivalent is a pattern called **CRTP**, the Curiously Recurring Template Pattern.

### Background

| | Virtual dispatch (inheritance) | CRTP (compile-time) |
|---|---|---|
| How the right `area()` is found | A vtable pointer, stored in every instance, followed at runtime | Resolved directly at compile time via `static_cast<Derived*>(this)` |
| Extra memory per instance | One vtable pointer (8 bytes on a 64-bit target) | None |
| Runtime dispatch cost | One indirect call through the vtable | None — an ordinary direct call, exactly like Section 2.3's templates |
| Usable across the host/device boundary? | Compiles cleanly for `__device__` code, but a vtable built by a *host*-side constructor is not guaranteed valid for a *device*-side call on the same object — a genuine, well-documented CUDA hazard | No vtable exists to be invalid in the first place |

### Worked Example 2.5.1 — virtual dispatch, genuinely compiled and measured

```cpp
struct Shape {
    __host__ __device__ virtual float area() const = 0;
    __host__ __device__ virtual ~Shape() {}
};

struct Circle : Shape {
    float radius;
    __host__ __device__ Circle(float r) : radius(r) {}
    __host__ __device__ float area() const override { return 3.14159f * radius * radius; }
};

__global__ void compute_area_kernel(Shape* s, float* out) {
    out[0] = s->area();  // virtual dispatch from device code -- compiles cleanly
}
```

This genuinely compiles clean under `nvcc -arch=sm_80` — device code is fully allowed to call a virtual function:

```bash
nvcc -arch=sm_80 virtual_dispatch.cu -o virtual_dispatch
```

The host-side call, genuinely run (`compute_area_kernel` above is compiled into the binary but, for the reason explained next, is deliberately never launched):

```bash
./virtual_dispatch
```

```
host-side virtual call: c.area() = 12.566360
```

`3.14159 × 2.0² = 3.14159 × 4.0 = 12.56636`, matching `πr²` for `r=2`. But `compute_area_kernel` above is never actually launched in this book — precisely because of the hazard in the Background table: `Circle c(2.0f)` constructed on the host writes a vtable pointer into `c` that refers to the *host's* compiled vtable, and passing `&c` to a kernel and calling `s->area()` from device code is only well-defined if the object was constructed *on the device* in the first place (or the vtable pointer is otherwise made valid for device code, which real CUDA codebases handle with dedicated "placement new on device" patterns beyond this chapter's scope). Getting this wrong doesn't reliably fail loudly — it's undefined behavior, which is a strictly worse failure mode than Chapter 1.3's clean, checkable `cudaErrorNoDevice`.

### Worked Example 2.5.2 — CRTP, the same interface with no vtable

```cpp
template <typename Derived>
struct ShapeCRTP {
    __host__ __device__ float area() const {
        return static_cast<const Derived*>(this)->area_impl();
    }
};

struct CircleCRTP : ShapeCRTP<CircleCRTP> {
    float radius;
    __host__ __device__ CircleCRTP(float r) : radius(r) {}
    __host__ __device__ float area_impl() const { return 3.14159f * radius * radius; }
};

struct SquareCRTP : ShapeCRTP<SquareCRTP> {
    float side;
    __host__ __device__ SquareCRTP(float s) : side(s) {}
    __host__ __device__ float area_impl() const { return side * side; }
};

template <typename ShapeT>
__host__ __device__ float print_area_generic(const ShapeT& s) {
    return s.area();
}
```

Compiled and run as the complete `04_crtp_interface.cu` further below:

```bash
nvcc -arch=sm_80 04_crtp_interface.cu -o 04_crtp_interface
./04_crtp_interface
```

Genuinely compiled and run:

```
sizeof(CircleCRTP) = 4 (no vtable pointer)
circle area (via CRTP, resolved at compile time) = 12.566360
square area (via CRTP, resolved at compile time) = 9.000000
```

`sizeof(CircleCRTP) == 4` — just the one `float radius` field, nothing else — genuinely measured and directly comparable against `sizeof(Circle)` (the virtual version) which is `16` bytes: an 8-byte vtable pointer plus the 4-byte `radius`, padded to 16 for 8-byte alignment. `print_area_generic<CircleCRTP>` and `print_area_generic<SquareCRTP>` are two separate template instantiations (Section 2.3's mechanism again), each one calling a fully resolved, concrete `area_impl()` with no indirection — the circle area matches Worked Example 2.5.1's virtual version exactly (`12.566360`, the same `πr²` for `r=2`), and the square area is `3.0² = 9.0`. Same interface guarantee — "anything passed to `print_area_generic` has an `.area()`" — checked entirely at compile time, with none of the vtable-validity hazard the virtual version carries across the host/device boundary. This is the pattern this book's own `Tensor`-adjacent code reaches for whenever it needs Mojo's trait-like guarantee.

## 2.6 Complete Runnable Code

### File: `01_basic_structs.cu`

```cpp
#include <cstdio>
#include <cmath>

struct Point2D {
    float x;
    float y;

    __host__ __device__ Point2D(float x_, float y_) : x(x_), y(y_) {}

    __host__ __device__ float distance_from_origin() const {
        return sqrtf(x * x + y * y);
    }

    __host__ __device__ float distance_to(const Point2D& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        return sqrtf(dx * dx + dy * dy);
    }
};

// Section 2.1.2: the SAME method, called from a __global__ kernel via an
// explicit launch configuration, <<<1, N>>> -- one block of N threads,
// threadIdx.x alone indexing every point.
__global__ void distance_kernel(float* out, const float* xs, const float* ys, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        Point2D p(xs[idx], ys[idx]);
        out[idx] = p.distance_from_origin();
    }
}

int main() {
    Point2D point1(3.0f, 4.0f);
    Point2D point2(1.0f, 1.0f);
    printf("sizeof(Point2D) = %zu\n", sizeof(Point2D));
    printf("point1.distance_from_origin() = %f\n", point1.distance_from_origin());
    printf("point1.distance_to(point2) = %f\n", point1.distance_to(point2));

    printf("point1 = (%f, %f)\n", point1.x, point1.y);
    Point2D point2b = point1;
    printf("point2 (copy) = (%f, %f)\n", point2b.x, point2b.y);
    point2b.x = 99.0f;
    printf("after point2.x = 99.0: point1 = (%f, %f), point2 = (%f, %f)\n",
           point1.x, point1.y, point2b.x, point2b.y);

    // --- The SAME distance_from_origin(), now called FROM a kernel ---
    const int N = 4;
    float xs[N] = {3.0f, 0.0f, 1.0f, 5.0f};
    float ys[N] = {4.0f, 0.0f, 1.0f, 12.0f};

    printf("\nhost-side reference (same method, called on the host):\n");
    for (int i = 0; i < N; i++) {
        Point2D p(xs[i], ys[i]);
        printf("  Point2D(%.1f, %.1f).distance_from_origin() = %f\n", xs[i], ys[i], p.distance_from_origin());
    }

    float *d_xs = nullptr, *d_ys = nullptr, *d_out = nullptr;
    cudaError_t err_xs = cudaMalloc(&d_xs, N * sizeof(float));
    cudaError_t err_ys = cudaMalloc(&d_ys, N * sizeof(float));
    cudaError_t err_out = cudaMalloc(&d_out, N * sizeof(float));
    printf("\ncudaMalloc(d_xs): %s, cudaMalloc(d_ys): %s, cudaMalloc(d_out): %s\n",
           cudaGetErrorName(err_xs), cudaGetErrorName(err_ys), cudaGetErrorName(err_out));

    cudaError_t err_cpy_xs = cudaMemcpy(d_xs, xs, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t err_cpy_ys = cudaMemcpy(d_ys, ys, N * sizeof(float), cudaMemcpyHostToDevice);
    printf("cudaMemcpy(xs H2D): %s, cudaMemcpy(ys H2D): %s\n",
           cudaGetErrorName(err_cpy_xs), cudaGetErrorName(err_cpy_ys));

    printf("\nlaunching distance_kernel<<<1, %d>>> (one block, %d threads, one Point2D per thread)\n", N, N);
    distance_kernel<<<1, N>>>(d_out, d_xs, d_ys, N);
    printf("kernel launch: %s\n", cudaGetErrorName(cudaGetLastError()));

    float h_out[N] = {0};
    cudaError_t err_cpy_out = cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("cudaMemcpy(out D2H): %s\n", cudaGetErrorName(err_cpy_out));
    printf("device-computed results (all zero: the copy back never received real device output,\n");
    printf("because no CUDA-capable device exists in this sandbox -- genuinely attempted, honestly\n");
    printf("failed, exactly like every device-touching call in this book from Chapter 1 onward):\n");
    for (int i = 0; i < N; i++) {
        printf("  h_out[%d] = %f\n", i, h_out[i]);
    }

    return 0;
}
```

```bash
nvcc -arch=sm_80 01_basic_structs.cu -o 01_basic_structs
./01_basic_structs
```

Produces the two output blocks shown in Worked Examples 2.1.1 and 2.2.1 above, followed by Worked Example 2.1.2's kernel-launch block, concatenated in the order `main()` prints them.

### File: `02_templates.cu`

```cpp
#include <cstdio>

template <typename T, int N>
struct Vector {
    T data[N];

    __host__ __device__ T sum() const {
        T total = T(0);
        for (int i = 0; i < N; i++) total += data[i];
        return total;
    }
};

int main() {
    Vector<float, 4> vf;
    vf.data[0] = 1.0f; vf.data[1] = 2.0f; vf.data[2] = 3.0f; vf.data[3] = 4.0f;

    Vector<int, 3> vi;
    vi.data[0] = 10; vi.data[1] = 20; vi.data[2] = 30;

    printf("sizeof(Vector<float,4>) = %zu, sum = %f\n", sizeof(vf), vf.sum());
    printf("sizeof(Vector<int,3>)   = %zu, sum = %d\n", sizeof(vi), vi.sum());
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_templates.cu -o 02_templates
./02_templates
nm -C 02_templates | grep "Vector<"
```

Produces the output shown in Worked Example 2.3.1 above, including the two distinct `nm`-reported symbol addresses.

### File: `03_raii_device_memory.cu`

```cpp
#include <cstdio>

struct DeviceBuffer {
    float* ptr;
    int size;

    DeviceBuffer(int n) : ptr(nullptr), size(n) {
        cudaError_t err = cudaMalloc((void**)&ptr, n * sizeof(float));
        printf("  DeviceBuffer(%d) constructed -> cudaMalloc err=%d (%s)\n", n, err, cudaGetErrorString(err));
    }

    ~DeviceBuffer() {
        printf("  ~DeviceBuffer(size=%d) destructor firing, freeing ptr=%p\n", size, (void*)ptr);
        cudaFree(ptr);
    }
};

void scoped_demo() {
    printf("entering scoped_demo\n");
    DeviceBuffer a(100);
    DeviceBuffer b(200);
    printf("both buffers constructed, about to leave scope\n");
}

int main() {
    scoped_demo();
    printf("back in main, both destructors already ran\n");
    return 0;
}
```

```bash
nvcc -arch=sm_80 03_raii_device_memory.cu -o 03_raii_device_memory
./03_raii_device_memory
```

Produces exactly the output shown in Worked Example 2.4.1 above.

### File: `04_crtp_interface.cu`

```cpp
#include <cstdio>

template <typename Derived>
struct ShapeCRTP {
    __host__ __device__ float area() const {
        return static_cast<const Derived*>(this)->area_impl();
    }
};

struct CircleCRTP : ShapeCRTP<CircleCRTP> {
    float radius;
    __host__ __device__ CircleCRTP(float r) : radius(r) {}
    __host__ __device__ float area_impl() const { return 3.14159f * radius * radius; }
};

struct SquareCRTP : ShapeCRTP<SquareCRTP> {
    float side;
    __host__ __device__ SquareCRTP(float s) : side(s) {}
    __host__ __device__ float area_impl() const { return side * side; }
};

template <typename ShapeT>
__host__ __device__ float print_area_generic(const ShapeT& s) {
    return s.area();
}

int main() {
    CircleCRTP c(2.0f);
    SquareCRTP sq(3.0f);
    printf("sizeof(CircleCRTP) = %zu (no vtable pointer)\n", sizeof(CircleCRTP));
    printf("circle area (via CRTP, resolved at compile time) = %f\n", print_area_generic(c));
    printf("square area (via CRTP, resolved at compile time) = %f\n", print_area_generic(sq));
    return 0;
}
```

```bash
nvcc -arch=sm_80 04_crtp_interface.cu -o 04_crtp_interface
./04_crtp_interface
```

### Expected Output for `04_crtp_interface.cu`

```
sizeof(CircleCRTP) = 4 (no vtable pointer)
circle area (via CRTP, resolved at compile time) = 12.566360
square area (via CRTP, resolved at compile time) = 9.000000
```

Every number here was independently traced earlier in this chapter: the `πr²` area computation in Worked Example 2.5.1, and the `4`-byte, no-vtable size in Worked Example 2.5.2. All four files in this section were genuinely compiled with `nvcc -arch=sm_80` and, since none of them launch a kernel, genuinely run to completion in this no-GPU environment — `03_raii_device_memory.cu` is the one exception worth restating: its `cudaMalloc` calls genuinely execute and genuinely return `cudaErrorNoDevice`, exactly as documented in Section 2.4.

## Chapter Summary

A `struct` in CUDA C++ is exactly the fixed, contiguous, compile-time-known memory layout Chapter 1 already established for primitive types, extended to fields you name yourself — and every method attached to it inherits Chapter 1.4's host/device qualifier rules individually, so a struct usable from both a kernel and host setup code needs `__host__ __device__` on every method either side calls. Constructor overloading resolves which constructor runs entirely at compile time, and the compiler-generated copy constructor gives plain-data structs correct, independent value semantics for free — a guarantee that stops holding the moment a struct owns a pointer, which Section 2.4's `DeviceBuffer` made concrete: a naive copy duplicates the address, not the allocation, setting up a double-free that RAII's constructor/destructor pairing is designed to prevent by tying `cudaFree` to scope exit instead of to a call you might forget. Templates compile a struct into one genuinely separate, fully-specialized piece of machine code per distinct combination of type/value arguments — proven directly in this chapter as two different symbols at two different addresses in one compiled binary — with zero runtime cost for the genericity, the same zero-cost guarantee `auto` carries from Chapter 1.2. And where Mojo reaches for a `trait` to declare a compile-time-checked interface with no inheritance hierarchy, C++ has no equivalent keyword, but CRTP reproduces the identical guarantee: no vtable, no runtime dispatch, and — uniquely important on a GPU — none of virtual dispatch's genuine, well-documented hazard of a host-built vtable pointer being dereferenced by device code.

## Self-Check Questions

1. `struct Pair { __host__ float host_only(); __device__ float device_only(); };` — which of its two methods, if either, could a `__global__` kernel call directly, and which, if either, could `main()` call directly?
2. A struct's compiler-generated copy constructor copies every field's *value*. Explain precisely why this is completely correct for `Point2D` (two `float`s) but incorrect for `DeviceBuffer` (a `float*` and an `int`).
3. `Vector<float, 4>` and `Vector<double, 4>` are both instantiated in the same program. Are they the same compiled type with different data, or two entirely separate compiled types? What compiled-binary evidence from this chapter would settle the question either way?
4. `DeviceBuffer a(100); DeviceBuffer b(200);` are declared in that order inside one function, with no early `return`. In what order do their destructors fire, and what specific C++ scoping rule guarantees that order?
5. `sizeof(Circle)` (virtual dispatch) is `16`; `sizeof(CircleCRTP)` is `4`. Account for the entire 12-byte difference, and explain the specific CUDA-related hazard that difference in mechanism creates for the virtual version that the CRTP version simply doesn't have.

## Where We Go Next

Every struct this chapter built lived entirely on the stack or held, at most, a single raw `float*` into device memory — the ownership and lifetime rules stayed simple because there was only ever one field to manage. Chapter 3 tackles what happens once a struct needs to describe *multi-dimensional* data laid out in memory — rows, columns, strides between them — the memory-layout groundwork this book's actual `Tensor` class (Part 1) is built directly on top of.

## Worked Solutions

**1.** `device_only()` could be called by the `__global__` kernel (device code calling a `__device__` method is allowed) but not by `main()` (host code calling a `__device__`-only method is the exact compile error Chapter 1.4's Worked Example 1.4.1 demonstrated for free functions, and it applies identically to methods). `host_only()` is the mirror image: callable from `main()`, not from the kernel. Neither method is callable from both sides — only a `__host__ __device__` method would be.

**2.** For `Point2D`, both fields are plain `float` values with no indirection — copying the bytes *is* copying the complete, independent state, so two `Point2D` instances with the same field values are genuinely independent and mutating one can never affect the other. For `DeviceBuffer`, the `ptr` field is an address, not the data it points to — a byte-for-byte copy produces two structs whose `ptr` fields hold the *identical* address, meaning both believe they own the same device allocation, and both of their destructors will eventually call `cudaFree` on it — a double-free.

**3.** They are two entirely separate compiled types, generated from the same template pattern but instantiated with different template arguments (`float` vs. `double`). This chapter's own `nm` output on `Vector<float,4>::sum()` and `Vector<int,3>::sum()` is exactly this kind of evidence: two distinct symbols at two distinct addresses in one compiled binary, proving the compiler genuinely emitted separate machine code for each instantiation rather than one generic function branching on a type tag at runtime.

**4.** Destructors fire in the exact reverse of construction order: `b` (constructed second) is destroyed first, then `a` — genuinely captured in this chapter as `~DeviceBuffer(size=200)` printing before `~DeviceBuffer(size=100)`. The rule is C++'s scoped-object lifetime guarantee: local objects with automatic storage duration are destroyed in the reverse order of their declaration within the same scope, the same LIFO discipline a call stack itself uses for stack frames.

**5.** `Circle` (virtual) carries an 8-byte vtable pointer (needed to look up `area()` at runtime) plus the 4-byte `radius` field, which the compiler pads to 16 bytes total to keep the whole struct 8-byte aligned (matching the vtable pointer's own alignment requirement) — accounting for the full 12-byte difference against `CircleCRTP`'s 4 bytes (just `radius`, no padding needed since a lone 4-byte `float` is already 4-byte aligned). The CUDA-specific hazard unique to the virtual version: that 8-byte vtable pointer is written by whichever constructor ran, host or device, and points at a vtable compiled for *that* side — a `Circle` constructed on the host and later dereferenced through `->area()` from device code is relying on a vtable pointer that was never guaranteed valid in device address space, a genuine source of undefined behavior with no equivalent risk in the CRTP version, which has no vtable pointer to be wrong about in the first place.
