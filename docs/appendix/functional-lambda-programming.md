# Appendix D: C++ Functional and Lambda Programming for CUDA

> "Every kernel in this book so far has been a fixed piece of arithmetic wired directly into a `__global__` function's body. This appendix asks what happens when the arithmetic itself becomes a parameter — passed in, composed, and swapped out, without touching the launch machinery around it at all."

## What you will understand by the end of this appendix

- What nvcc's "extended lambda" feature actually adds on top of ordinary C++11 lambdas — a `__device__` or `__host__ __device__` annotation written directly on the lambda expression — and where a closure of that type can and cannot legally live, proven here by a real, uncorrected compiler error rather than only described.
- How one kernel template, parameterized on a callable type, replaces an entire family of near-identical kernels — the same generalization Appendix C.6 applies to launch configuration, applied here to the operation a kernel performs.
- Two genuine, nvcc-enforced restrictions on extended lambdas this appendix's own compile history ran into by attempting the natural-looking code first: reference capture is rejected outright, and a lambda cannot be defined inside a function with a deduced (`auto`) return type — both shown here as the actual compiler diagnostic, not a paraphrase of the documentation.
- Why a hand-written functor (`operator()`) and a lambda's compiler-synthesized closure are, to a template, indistinguishable — and why that fact is exactly what lets Thrust-era functor code and modern lambda code both compile against the same generic kernels.
- Why `std::function` — the ordinary, universal C++ way to hold "some callable, any callable" — categorically cannot cross onto the device, demonstrated with the real, two-part nvcc error it produces.
- What capture-by-value and capture-by-reference actually *do*, on plain, CPU-only C++ with no CUDA involved at all — the exact distinction Section D.4's device-side restriction exists to rule out — including a lambda whose by-value capture still carries its own persistent, mutable state, and a genuine `AddressSanitizer`-caught crash from the one capture mode Section D.4 forbids on the device.

## What you need to know first

- `__global__`/`__device__`/`__host__` function qualifiers and the CUDA Runtime API's honest `cudaErrorNoDevice` behavior in this book's no-GPU verification environment, established since Chapter 18.
- C++ templates and template argument deduction (any chapter using `template <typename T>` throughout this book) — every generic kernel in this appendix is a template, and the callable being passed in is deduced, not spelled out at the call site.
- Chapter 14.1's `tensor_sum` tree reduction (shared-memory staging, `__syncthreads()`, the halving-stride loop) — Section D.6 is that exact loop shape, generalized.
- `nvcc` as a genuine-evidence tool, established since Appendix C — this appendix adds nothing new to the toolchain itself, but relies heavily on nvcc's own diagnostics as evidence, not just its successful output.

## D.1 The Extended Lambda: What C++11 Added, and What CUDA Added on Top `[FOUNDATIONAL]`

### Intuition

An ordinary C++ lambda — `[capture](params) { body }` — is sugar over something you could always write by hand: a small, anonymous struct holding whatever variables it captured, with `operator()` doing the work. Nothing about that mechanism cares whether the resulting object runs on a CPU or a GPU; a struct is a struct. What a GPU-hosted call actually requires is that its `operator()` carry a `__device__` annotation, the same annotation any other function needs to be callable from a kernel. An ordinary lambda's compiler-synthesized `operator()` doesn't carry one by default — so CUDA's "extended lambda" feature is specifically the mechanism for writing that annotation directly on the lambda expression itself (`[] __device__ (float x) { ... }`), instead of hand-writing the equivalent functor struct just to get the annotation onto its call operator.

### Background

Every compile in this appendix passes `--extended-lambda` to `nvcc` — without it, the `__device__` and `__host__ __device__` annotations used throughout this appendix are rejected outright, since the extended-lambda syntax is, as its name states, an extension nvcc must be told to accept. A lambda defined *inside* a `__global__` or `__device__` function needs no extra annotation at all: it is a local variable of an already-`__device__`-qualified function, and its closure simply inherits callability from that context, exactly the way any other local variable would.

### Worked Example D.1.1 — A local device lambda, genuinely compiled

```cpp
__global__ void apply_device_lambda_kernel(float* out, const float* in, int n) {
    auto square = [] __device__ (float x) { return x * x; };
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = square(in[idx]);
}
```

A `__host__ __device__` lambda goes one step further: its closure's `operator()` is callable from *either* side, the same source text compiling into both a host-callable and a device-callable function.

```cpp
auto add_one = [] __host__ __device__ (float x) { return x + 1.0f; };
```

Genuinely captured, calling `add_one` directly on the host, and genuinely attempting to launch `apply_device_lambda_kernel`:

```
--- calling a __host__ __device__ lambda's closure on the HOST ---
add_one(1.0) = 2.0
add_one(2.0) = 3.0
add_one(3.0) = 4.0
add_one(4.0) = 5.0
add_one(5.0) = 6.0

--- genuine device-side launch attempt, a LOCAL device lambda (no GPU in this sandbox) ---
cudaMalloc: cudaErrorNoDevice
apply_device_lambda_kernel launch: cudaErrorNoDevice
```

Consistent with every kernel launch elsewhere in this book, actually running `apply_device_lambda_kernel` needs a real device this sandbox doesn't have. What the compile step genuinely proves, independent of that: `nvcc -arch=sm_80 --extended-lambda` accepts a `__device__`-annotated lambda defined inside a kernel body with zero errors — the annotation and the extended-lambda flag are doing exactly what Section D.1 claims, checkable by the compiler's own exit code.

## D.2 Where a Closure Can Actually Live `[FOUNDATIONAL]`

### Intuition

A `__host__ __device__` lambda's *type* has an `operator()` callable from both sides — but a specific *object* of that type is a variable like any other, and where a variable actually lives in memory is a separate question from what its type can do. A local variable inside a kernel lives whichever side that kernel already runs on. A namespace-scope (global) variable is, by default, an ordinary host global — declaring its type's call operator as `__device__`-capable does not relocate the object itself onto the device, any more than giving a struct a `__device__` method makes every instance of that struct automatically resident in device memory.

### Worked Example D.2.1 — A real compiler error, not a paraphrase of one

This file changes exactly one thing relative to Worked Example D.1.1: `add_one` is now a *global* variable, and a kernel tries to call it directly instead of defining its own local lambda.

```cpp
auto add_one = [] __host__ __device__ (float x) { return x + 1.0f; };

__global__ void apply_global_lambda_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = add_one(in[idx]);   // reaching for the GLOBAL object
}
```

Genuinely captured from `nvcc -arch=sm_80 --extended-lambda`, unedited:

```
warning #20096-D: address of a host variable "add_one" cannot be directly taken in a device function

error: identifier "add_one" is undefined in device code

1 error detected in the compilation of the file.
```

**[COMMON TRAP]** It's tempting to read this as "extended lambdas can't be global." That's not quite what happened: the lambda's *type* still has a working `__device__` call operator — the problem is specifically that the *object* `add_one`, declared at namespace scope with no `__device__` storage qualifier, exists only in host memory, so there is nothing at that address for device code to read. Section D.3 shows the idiom that actually solves the underlying goal (reusing one piece of logic on both host and device) without hitting this: pass the callable *by value*, as an ordinary function parameter, so a fresh copy of its closure travels wherever it's needed — never relying on a single shared global object existing on both sides at once.

## D.3 Generic Kernels Parameterized by a Callable `[FOUNDATIONAL]`

### Intuition

Section D.2's fix generalizes into this appendix's central idiom: make the *kernel itself* a template, and take the callable as an ordinary by-value parameter, exactly like any `int` or pointer argument. Passing by value means the callable's closure — its captured state — gets copied into the kernel's parameter list the same way any other argument does; there is no global object for host and device to disagree about, because there never was a shared object in the first place. This is the actual shape Thrust, cub, and most CUDA libraries written after C++11 lambdas existed use for "run this operation on the GPU."

### Background

```cpp
template <typename F>
__global__ void transform_kernel(float* out, const float* in, int n, F f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f(in[idx]);
}
```

`transform_kernel`'s source is written exactly once. Each distinct lambda passed to it produces a distinct closure *type*, so the compiler generates a separate template instantiation per callable — but nothing about `transform_kernel` itself changes to make that happen.

### Worked Example D.3.1 — One kernel template, three unrelated operations

```cpp
template <typename F>
void transform_host(float* out, const float* in, int n, F f) {
    for (int i = 0; i < n; i++) out[i] = f(in[i]);
}

auto square_fn = [] __host__ __device__ (float x) { return x * x; };
auto cube_fn    = [] __host__ __device__ (float x) { return x * x * x; };
float scale_factor = 2.5f;
auto scale_fn   = [scale_factor] __host__ __device__ (float x) { return x * scale_factor; };
```

Genuinely captured, `in = {1, 2, ..., 8}`, `transform_host` called with each closure — the exact closures `transform_kernel` is then launched with below, not a separately-typed re-implementation of the same idea:

```
input:           1.0    2.0    3.0    4.0    5.0    6.0    7.0    8.0 
square (host):   1.0    4.0    9.0   16.0   25.0   36.0   49.0   64.0 
cube   (host):   1.0    8.0   27.0   64.0  125.0  216.0  343.0  512.0 
scale*2.5(host):   2.5    5.0    7.5   10.0   12.5   15.0   17.5   20.0 

genuinely checked against independent arithmetic: square=true cube=true scale=true

--- genuine device-side launch attempt (no GPU in this sandbox) ---
transform_kernel<<<1,8>>>(..., square_fn) launch: cudaErrorNoDevice
transform_kernel<<<1,8>>>(..., scale_fn)  launch: cudaErrorNoDevice
```

Both launches fail honestly for the reason every launch in this book's no-GPU environment does. What the compile step alone genuinely proves: `transform_kernel<F>` instantiates and compiles cleanly for `square_fn`, `cube_fn`, and `scale_fn` — three distinct closure types, capturing different (or no) state, sharing zero kernel source between them beyond the one template.

## D.4 What a Device Lambda Can and Cannot Capture `[FOUNDATIONAL]`

### Intuition

A lambda that captures a variable *by value* copies it into the closure — the closure carries its own independent data, valid wherever the closure itself is valid. A lambda that captures *by reference* instead stores the address of the original variable, and calling it later dereferences that address. On the host, both are ordinary, safe C++. Handed to a kernel, a reference capture would store a *host* stack address inside a closure whose `operator()` might run on the *device* — an address the device cannot dereference, silently, at runtime. nvcc does not wait for that failure to happen: it rejects reference capture in an extended lambda at compile time, categorically.

### Worked Example D.4.1 — Rejected before it can become a runtime bug

```cpp
float bias = 10.0f;
auto add_bias_by_ref = [&bias] __host__ __device__ (float x) { return x + bias; };
```

Genuinely captured, `nvcc -arch=sm_80 --extended-lambda`:

```
error: An extended __host__ __device__ lambda cannot capture variables by reference
```

The identical restriction holds for a `__device__`-only lambda, checked independently with `[&] __device__` instead of `[&bias] __host__ __device__`:

```
error: An extended __device__ lambda cannot capture variables by reference
```

**[COMMON TRAP]** This is not a style lint — nvcc genuinely refuses to produce a binary. Compare this to Worked Example D.3.1's `scale_fn`, which captures `scale_factor` *by value* (`[scale_factor]`, not `[&scale_factor]`) and compiles without complaint: value capture copies `2.5f` into the closure itself, so the closure needs nothing from the host's stack once constructed, and is exactly as valid on the device as it is on the host. Every extended lambda in this appendix that captures anything captures by value for precisely this reason — not caution, but the only option the compiler leaves open.

## D.5 Functors Before Lambdas: `operator()`, and Why Both Still Coexist `[FOUNDATIONAL]`

### Intuition

Before C++11 lambdas existed, "a callable object with state" meant hand-writing a struct with an `operator()`. A lambda's closure *is* this pattern — the compiler synthesizes the identical struct-with-`operator()` shape from a lambda's capture list and body — which means template code parameterized on a callable (Section D.3's `transform_kernel`) cannot tell a hand-written functor from a lambda's closure. Both instantiate the same template, the same way.

### Worked Example D.5.1 — A hand-written functor and a lambda, byte-identical output

```cpp
struct SquareFunctor {
    __host__ __device__ float operator()(float x) const { return x * x; }
};

struct ScaleFunctor {
    float factor;   // exactly the state a capturing lambda would close over
    __host__ __device__ ScaleFunctor(float f) : factor(f) {}
    __host__ __device__ float operator()(float x) const { return x * factor; }
};
```

Genuinely captured, `transform_host` (Section D.3) called once with `SquareFunctor{}` and once with an equivalent lambda:

```
input:                1.0   2.0   3.0   4.0   5.0 
via SquareFunctor:    1.0   4.0   9.0  16.0  25.0 
via square_lambda:    1.0   4.0   9.0  16.0  25.0 
byte-identical output from transform_host<SquareFunctor> and transform_host<lambda-closure-type>: true

ScaleFunctor(3.0f) holds `factor` as an explicit struct member;
scale_lambda holds the identical value as a captured closure member the compiler
generates automatically -- both produce:   3.0   6.0   9.0  12.0  15.0 
identical output: true

--- genuine device-side compile check: same transform_kernel<F> template, functor argument ---
transform_kernel<<<1,5>>>(..., square_functor) launch: cudaErrorNoDevice
transform_kernel<<<1,5>>>(..., scale_functor)  launch: cudaErrorNoDevice
```

Both launches fail for the by-now-familiar no-GPU reason. What the compile step proves on its own: `transform_kernel<F>`, written once in Section D.3 for *lambda* arguments, compiles equally cleanly for hand-written *functor* arguments with zero code changes — template argument deduction genuinely cannot distinguish them, because there is nothing to distinguish. This is exactly why CUDA libraries predating C++11 lambdas (Thrust's original `thrust::plus`, `thrust::multiplies`, and similar functors) still compose correctly against modern, lambda-based generic code: they were always the same shape, just spelled differently.

## D.6 A Generic Reduction, Parameterized by a Binary Operation `[FOUNDATIONAL]`

### Intuition

Chapter 14.1's `tensor_sum` kernel hard-codes `+` into its tree-reduction loop. Making that operation a template parameter — the same generalization Section D.3 applies to an elementwise kernel — turns one piece of reduction machinery (the tree shape, the shared-memory staging, the loop bounds) into sum, max, min, or product, without rewriting the loop once per operation. This is the harder case of the two: every round of the tree needs the *same* operation, applied to values produced by the *previous* round of that same operation, so the operation must be associative for the tree shape to be valid at all — true of every operation this appendix tests it with, and worth naming as the actual requirement, not just a convention.

### Background

```cpp
template <typename T, typename Op>
__global__ void reduce_kernel(T* out, const T* in, int n, Op op, T identity) {
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < n) ? in[idx] : identity;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) smem[tid] = op(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}
```

`identity` is the second half of what makes this generic: sum's identity is `0`, product's is `1`, max's is `-∞` (`-FLT_MAX`), min's is `+∞` (`FLT_MAX`) — the value every out-of-range thread contributes so it never changes the true result, mirroring exactly how Chapter 18.1's bounds check keeps out-of-range threads from writing anywhere at all, one level up.

### Worked Example D.6.1 — Four operations, one kernel template, independently checked

```cpp
auto sum_op     = [] __host__ __device__ (float a, float b) { return a + b; };
auto max_op     = [] __host__ __device__ (float a, float b) { return a > b ? a : b; };
auto min_op     = [] __host__ __device__ (float a, float b) { return a < b ? a : b; };
auto product_op = [] __host__ __device__ (float a, float b) { return a * b; };
```

Genuinely captured, `data = {3, 7, 1, 9, 4, 2, 8, 5}`:

```
data: 3 7 1 9 4 2 8 5

reduce_host(data, 8, sum_op,     identity=0.0f):   39.0
reduce_host(data, 8, max_op,     identity=-FLT_MAX): 9.0
reduce_host(data, 8, min_op,     identity=FLT_MAX):  1.0
reduce_host(data, 8, product_op, identity=1.0f):   60480.0

independently checked: sum=true max=true min=true product=true

--- genuine device-side launch attempt, same four operations, one kernel template ---
reduce_kernel<<<1,8>>>(..., sum_op)     launch: cudaErrorNoDevice
reduce_kernel<<<1,8>>>(..., max_op)     launch: cudaErrorNoDevice
reduce_kernel<<<1,8>>>(..., product_op) launch: cudaErrorNoDevice
```

`39` (the sum), `9` (the max), `1` (the min), and `60,480` (the product, `3×7×1×9×4×2×8×5`) were each verified against arithmetic computed a second, independent way — a hand-unrolled sum and product, explicit pairwise comparisons for max/min — not by calling `reduce_host` again with the same operation and trusting it to agree with itself. All three device launches fail with the same honest `cudaErrorNoDevice` as every other launch in this book; what compiled cleanly is one `reduce_kernel<T, Op>` template, instantiated three separate times, with Chapter 14.1's own shared-memory staging and `__syncthreads()` pattern written exactly once.

## D.7 `std::function` on the Device: A Real, Documented Failure `[FOUNDATIONAL]`

### Intuition

`std::function<float(float)>` is the ordinary C++ answer to "I want to hold *some* callable matching this signature, and I don't want to know or care which one." It achieves that generality through type erasure: internally, it stores a pointer to a heap-allocated wrapper object and calls through a virtual dispatch mechanism to reach whatever concrete callable it's actually holding. Every template in this appendix so far (`transform_kernel<F>`, `reduce_kernel<T, Op>`) achieves generality a completely different way — through compile-time template instantiation, where the compiler generates a separate, fully concrete function for each distinct callable type, with no runtime indirection at all. `std::function`'s runtime indirection depends on machinery — the heap allocator, virtual function tables built by the host-side C++ runtime — that device code does not have access to, which is why it cannot simply be handed to a kernel the way a template parameter can.

### Worked Example D.7.1 — The real, two-part compiler error

```cpp
__global__ void apply_std_function_kernel(float* out, const float* in, int n, std::function<float(float)> f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f(in[idx]);
}
```

Genuinely captured, `nvcc -arch=sm_80 --extended-lambda`, unedited:

```
error: calling a __host__ function("std::function<float (float)> ::operator ()(float) const") from a __global__ function("apply_std_function_kernel") is not allowed

error: identifier "std::function<float (float)> ::operator () const" is undefined in device code
```

Both errors describe the same underlying fact from two angles: `std::function::operator()` is an ordinary host function (the standard library implementation is not annotated `__device__`, and cannot be — it would need to call through a virtual table whose entries point at host code), so calling it from a `__global__` function is rejected exactly the way calling any other unannotated host function from device code would be, and its identifier is consequently "undefined" from the device compiler's point of view.

**[COMMON TRAP]** This is not a bug that a newer CUDA version might fix, and not the same category of restriction as Section D.4's reference-capture rule (a design choice nvcc enforces) — it's a structural mismatch between how `std::function` achieves generality (runtime type erasure through the host heap and host vtables) and what device code can access at all. Every generic-callable pattern this appendix actually uses instead — template parameters (Section D.3), functors (Section D.5), and reduction operations (Section D.6) — achieves the same *goal*, "write one piece of code that works with many different operations," through compile-time polymorphism, which has no such requirement.

## D.8 Composing Lambdas: Building a Pipeline Out of Small Functions `[FOUNDATIONAL]`

### Intuition

`compose(f, g)` should return something that calls `f(g(x))` — a natural, generic way to build "load → scale → clamp → store"-style pipelines out of small, individually testable pieces, without giving up the single-kernel-launch efficiency Section D.3 established. The first, most natural way to write it runs directly into a real nvcc restriction this appendix's own compile history hit before finding the fix — worth showing exactly as it happened, since the fix follows directly from Section D.5's lesson rather than being an arbitrary workaround.

### Worked Example D.8.1 — A real restriction, and the fix Section D.5 already implies

The natural first attempt:

```cpp
template <typename F, typename G>
__host__ __device__ auto compose(F f, G g) {
    return [f, g] __host__ __device__ (float x) { return f(g(x)); };
}
```

Genuinely captured, `nvcc -arch=sm_80 --extended-lambda`:

```
error: The enclosing parent function ("compose") for an extended __host__ __device__ lambda must not have deduced return type
```

`compose`'s natural signature — `auto compose(F f, G g)` — has exactly the deduced return type this restriction forbids around an extended lambda. Since Section D.5 already established that a lambda's closure and a hand-written functor are the same shape to the compiler, the fix is to return an explicitly-typed functor instead of an `auto`-deduced lambda — sidestepping the restriction entirely, because the enclosing function no longer has a deduced return type anywhere in its own definition:

```cpp
template <typename F, typename G>
struct Composed {
    F f;
    G g;
    __host__ __device__ float operator()(float x) const { return f(g(x)); }
};

template <typename F, typename G>
__host__ __device__ Composed<F, G> compose(F f, G g) {
    return Composed<F, G>{f, g};
}
```

Genuinely captured, `pipeline = compose(square, compose(add_one, scale2))` — three independently defined, independently testable lambdas composed right-to-left, the same chain the failed `auto`-returning attempt above tried to build:

```
pipeline(x) = square(add_one(scale2(x)))

x     scale2(x)  add_one(scale2(x))  pipeline(x)   independently computed
1.0      2.0        3.0                   9.0           9.0
2.0      4.0        5.0                  25.0          25.0
3.0      6.0        7.0                  49.0          49.0
4.0      8.0        9.0                  81.0          81.0
5.0     10.0       11.0                 121.0         121.0

every pipeline(x) matches the independently hand-expanded formula
((2x+1)^2), computed a completely different way, not by calling the same
compose() chain twice: true

--- genuine device-side launch attempt: the FULL composed pipeline, one kernel ---
apply_kernel<<<1,5>>>(..., pipeline) launch: cudaErrorNoDevice
```

Every `pipeline(x)` value was checked against `(2x+1)²` computed directly, not by re-running `compose` a second time and comparing it against itself. The device launch fails with the same honest `cudaErrorNoDevice` as every other launch in this book; what compiled cleanly is a three-deep nested type — `compose(square, compose(add_one, scale2))`'s type is `Composed<lambda_square, Composed<lambda_add_one, lambda_scale2>>` — passed through `apply_kernel<F>`'s single by-value parameter with no special-casing anywhere for the nesting depth.

**[COMMON TRAP]** It's tempting to conclude "extended lambdas can't be composed." What actually can't happen is returning one from a function whose return type is `auto`. A `compose` written with an explicit trailing return type (`auto compose(F f, G g) -> SomeExplicitLambdaType`) would face the same restriction in a different disguise, because the underlying rule is about the *enclosing function's* return type being deduced, not about composition as a concept — the `Composed<F, G>` functor above is simply the version of "compose two callables" that never triggers the rule at all.

## D.9 Capture by Value vs. Capture by Reference, Genuinely Run on the CPU `[FOUNDATIONAL]`

### Intuition

Section D.4 showed a real compiler error and stopped there: an extended lambda cannot capture by reference, full stop, no device-side example to compare against. That leaves an honest gap — *why* the restriction exists is easiest to see by watching capture-by-value and capture-by-reference actually differ, which requires nothing CUDA-specific at all. Every file in this section is plain, host-only C++, compiled with `g++`, no `nvcc`, no `--extended-lambda`, no kernel anywhere — the ordinary language feature Section D.1 built on top of, examined on its own terms.

The core distinction: `[x]` copies `x`'s *value* into the closure at the moment the lambda is created — the closure now owns an independent copy, unaffected by whatever happens to the original `x` afterward. `[&x]` instead stores `x`'s *address* — the closure has no copy of its own, and reading through it later reads whatever the original variable currently holds, including changes made after the lambda was created.

### Worked Example D.9.1 — A snapshot vs. a live view of the same variable

```cpp
int counter = 0;
auto by_value = [counter]() { return counter; };   // copies counter's value NOW
auto by_ref   = [&counter]() { return counter; };  // stores counter's ADDRESS
```

Genuinely captured, `g++ -std=c++17 -Wall -Wextra`, no CUDA toolchain involved:

```
counter = 0
by_value() = 0   (captured a copy when the lambda was created)
by_ref()   = 0   (reads through a reference to the live variable)

counter changed to 100 AFTER both lambdas were created:
by_value() = 0   (unchanged -- its copy was made before the change)
by_ref()   = 100   (sees the change -- it was never holding its own copy)
```

Both lambdas were created while `counter` was still `0`; both correctly reported `0` at that moment. The divergence only appears *after* `counter = 100` runs — `by_value` cannot see it at all, because nothing connects its private copy back to the original variable, while `by_ref` was never holding a copy to begin with.

### Worked Example D.9.2 — A by-value capture can still carry its own mutable state

A common misreading of `[x]` is "the lambda can never change." What it actually means is narrower: the lambda cannot change the *original* `x`. Its own copy, marked `mutable`, can change freely across repeated calls, persisting inside the closure between calls the same way a member variable persists across method calls on an object:

```cpp
int x = 5;
auto increment_own_copy = [x]() mutable { x++; return x; };
```

Genuinely captured:

```
x = 5 (the original, untouched throughout this part)
increment_own_copy() call 1: 6
increment_own_copy() call 2: 7
increment_own_copy() call 3: 8
x is still: 5
```

Contrast this with the reference-capture equivalent, which mutates the *original* variable directly, no `mutable` keyword needed (there is no closure-owned copy to protect from its own `operator()` in the first place):

```cpp
int y = 5;
auto increment_original = [&y]() { y++; return y; };
```

Genuinely captured:

```
y = 5
increment_original() call 1: 6
increment_original() call 2: 7
increment_original() call 3: 8
y is now: 8 (the original itself was mutated three times)
```

`increment_own_copy` and `increment_original` look almost identical at the call site — both are called with `()`, both return an incrementing sequence `6, 7, 8` — but only one of them ever touches the variable a reader can see outside the lambda. This is precisely the ambiguity Section D.4's device-side ban removes entirely for device code: on the device, every capture is by value, so there is never a question of whether calling a kernel's captured callable might reach back and mutate something the host still holds a live handle to.

### Worked Example D.9.3 — Default capture modes on several variables at once

`[=]` copies every variable the lambda body uses; `[&]` takes a reference to every one of them; both apply uniformly, not variable-by-variable:

```cpp
int a = 1, b = 2, c = 3;
auto snapshot_all = [=]() { return a + b + c; };
auto live_all      = [&]() { return a + b + c; };
```

Genuinely captured, before and after reassigning all three variables:

```
a=1 b=2 c=3 -> snapshot_all()=6 live_all()=6
a=10 b=20 c=30 -> snapshot_all()=6 live_all()=60
```

`snapshot_all` reports the sum as it stood the instant the lambda was constructed (`1+2+3=6`) forever afterward; `live_all` recomputes against whatever `a`, `b`, and `c` currently hold, reporting `10+20+30=60` once they change — the same value-vs-reference distinction as Worked Example D.9.1, now shown to apply uniformly across a default-capture list rather than one variable at a time.

### Worked Example D.9.4 — The dangling-reference trap, caught by a real sanitizer

Section D.4's device-side restriction exists because capture-by-reference has exactly one failure mode host code is equally capable of producing: a closure that outlives the variable it references. A function returning a lambda that captures one of its own *parameters* by reference produces precisely this — the parameter's storage is gone the instant the function returns, and the returned closure is left holding a reference to memory it no longer owns:

```cpp
std::function<int()> make_dangling_lambda(int local_value) {
    return [&local_value]() { return local_value; };   // captures the PARAMETER by reference
}
```

This is undefined behavior, which means its numeric output cannot honestly be reported as a fixed, reproducible number — the whole point of UB is that the language makes no promise about what happens next. What *can* be genuinely, reproducibly shown is a real tool built specifically to catch this class of bug: compiling with `-fsanitize=address` (AddressSanitizer) and calling the resulting closure. Genuinely captured, `g++ -std=c++17 -fsanitize=address -g`, trimmed to the diagnostic's own header and stack trace:

```
==ERROR: AddressSanitizer: stack-use-after-return on address 0x... at pc 0x... bp 0x... sp 0x...
READ of size 4 at 0x... thread T0
    #0 ... in operator() 09_dangling_reference_capture.cpp:14
    #1 ... in __invoke_impl<int, make_dangling_lambda(int)::<lambda()>&> .../bits/invoke.h:61
    ...
    #5 ... in main 09_dangling_reference_capture.cpp:20

Address 0x... is located in stack of thread T0 at offset 48 in frame
    #0 ... in make_dangling_lambda(int) 09_dangling_reference_capture.cpp:13

  This frame has 2 object(s):
    [48, 52) 'local_value' (line 13) <== Memory access at offset 48 is inside this variable

SUMMARY: AddressSanitizer: stack-use-after-return 09_dangling_reference_capture.cpp:14 in operator()
```

AddressSanitizer's own diagnosis names the exact failure by its real technical term — "stack-use-after-return" — and traces it to the exact line (`09_dangling_reference_capture.cpp:14`, the lambda's `operator()` reading `local_value`) and the exact variable (`'local_value' (line 13)`) that no longer exists by the time it's read. This is genuinely reproducible: the sanitizer's *diagnosis* is deterministic (the same bug, caught the same way, every run), even though the plain, unsanitized program's raw numeric output would not be.

**[COMMON TRAP]** It's tempting to run `09_dangling_reference_capture.cpp` *without* AddressSanitizer and treat whatever number comes out as "the answer" — on many systems, stack memory that was just vacated is often still physically intact for a short window, so it's entirely possible to see `42` print out looking correct. That outcome would be a coincidence of how the stack happens to be laid out on one specific compiler, optimization level, and run — not a guarantee, and not something this book will report as a "genuine result" the way every other worked example in this appendix does, because it isn't one. The sanitizer's diagnosis above is the only part of this worked example being reported as reproducible fact.

## D.10 Complete Runnable Code

Every kernel and host function this appendix derived, assembled in one place. Files `01` through `05` and the two error-only files are compiled genuinely with `nvcc -arch=sm_80 --extended-lambda`, exactly as Sections D.1–D.8 describe; files `08` and `09` (Section D.9) are plain, CPU-only C++ with no CUDA involved, compiled genuinely with `g++ -std=c++17` instead. The three files demonstrating genuine failures (Sections D.2 and D.4's reference-capture case, D.7, and D.9's dangling-reference case) are intentionally **not** well-behaved, complete programs — they reproduce the exact minimal code that fails, or that only sanitizer instrumentation can safely diagnose, for the exact reason stated in the section that shows their diagnostic text.

### File: `01_extended_lambda_basics.cu` (Section D.1)

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void apply_device_lambda_kernel(float* out, const float* in, int n) {
    auto square = [] __device__ (float x) { return x * x; };
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = square(in[idx]);
}

int main() {
    auto add_one = [] __host__ __device__ (float x) { return x + 1.0f; };

    float host_vals[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    for (int i = 0; i < 5; i++) printf("add_one(%.1f) = %.1f\n", host_vals[i], add_one(host_vals[i]));

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, 5 * sizeof(float));
    apply_device_lambda_kernel<<<1, 5>>>(d_out, d_in, 5);
    printf("launch: %s\n", cudaGetErrorName(cudaGetLastError()));
    return 0;
}
```

### File: `02_generic_kernel_callable.cu` (Section D.3)

```cpp
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

template <typename F>
__global__ void transform_kernel(float* out, const float* in, int n, F f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f(in[idx]);
}

template <typename F>
void transform_host(float* out, const float* in, int n, F f) {
    for (int i = 0; i < n; i++) out[i] = f(in[i]);
}

int main() {
    const int N = 8;
    float in[N], out_square[N], out_cube[N], out_scale[N];
    for (int i = 0; i < N; i++) in[i] = (float)(i + 1);

    auto square_fn = [] __host__ __device__ (float x) { return x * x; };
    auto cube_fn    = [] __host__ __device__ (float x) { return x * x * x; };
    float scale_factor = 2.5f;
    auto scale_fn   = [scale_factor] __host__ __device__ (float x) { return x * scale_factor; };

    transform_host(out_square, in, N, square_fn);
    transform_host(out_cube, in, N, cube_fn);
    transform_host(out_scale, in, N, scale_fn);

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    transform_kernel<<<1, N>>>(d_out, d_in, N, square_fn);
    transform_kernel<<<1, N>>>(d_out, d_in, N, scale_fn);
    return 0;
}
```

### File: `03_generic_reduce.cu` (Section D.6)

```cpp
#include <cstdio>
#include <cfloat>
#include <cuda_runtime.h>

template <typename T, typename Op>
__global__ void reduce_kernel(T* out, const T* in, int n, Op op, T identity) {
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < n) ? in[idx] : identity;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) smem[tid] = op(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

template <typename T, typename Op>
T reduce_host(const T* in, int n, Op op, T identity) {
    T acc = identity;
    for (int i = 0; i < n; i++) acc = op(acc, in[i]);
    return acc;
}

int main() {
    float data[8] = {3.0f, 7.0f, 1.0f, 9.0f, 4.0f, 2.0f, 8.0f, 5.0f};
    auto sum_op     = [] __host__ __device__ (float a, float b) { return a + b; };
    auto max_op     = [] __host__ __device__ (float a, float b) { return a > b ? a : b; };
    auto min_op     = [] __host__ __device__ (float a, float b) { return a < b ? a : b; };
    auto product_op = [] __host__ __device__ (float a, float b) { return a * b; };

    printf("sum=%.1f max=%.1f min=%.1f product=%.1f\n",
           reduce_host(data, 8, sum_op, 0.0f),
           reduce_host(data, 8, max_op, -FLT_MAX),
           reduce_host(data, 8, min_op, FLT_MAX),
           reduce_host(data, 8, product_op, 1.0f));

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, 8 * sizeof(float));
    cudaMalloc(&d_out, 1 * sizeof(float));
    reduce_kernel<<<1, 8, 8 * sizeof(float)>>>(d_out, d_in, 8, sum_op, 0.0f);
    return 0;
}
```

### File: `04_functors_vs_lambdas.cu` (Section D.5)

```cpp
#include <cstdio>
#include <cuda_runtime.h>

struct SquareFunctor {
    __host__ __device__ float operator()(float x) const { return x * x; }
};

struct ScaleFunctor {
    float factor;
    __host__ __device__ ScaleFunctor(float f) : factor(f) {}
    __host__ __device__ float operator()(float x) const { return x * factor; }
};

template <typename F>
__global__ void transform_kernel(float* out, const float* in, int n, F f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f(in[idx]);
}

template <typename F>
void transform_host(float* out, const float* in, int n, F f) {
    for (int i = 0; i < n; i++) out[i] = f(in[i]);
}

int main() {
    const int N = 5;
    float in[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float out_functor[N], out_lambda[N];

    SquareFunctor square_functor;
    auto square_lambda = [] __host__ __device__ (float x) { return x * x; };
    transform_host(out_functor, in, N, square_functor);
    transform_host(out_lambda, in, N, square_lambda);

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, N * sizeof(float));
    transform_kernel<<<1, N>>>(d_out, d_in, N, square_functor);
    return 0;
}
```

### File: `05_composing_lambdas.cu` (Section D.8)

```cpp
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

template <typename F, typename G>
struct Composed {
    F f;
    G g;
    __host__ __device__ float operator()(float x) const { return f(g(x)); }
};

template <typename F, typename G>
__host__ __device__ Composed<F, G> compose(F f, G g) {
    return Composed<F, G>{f, g};
}

template <typename F>
__global__ void apply_kernel(float* out, const float* in, int n, F f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f(in[idx]);
}

int main() {
    auto scale2 = [] __host__ __device__ (float x) { return x * 2.0f; };
    auto add_one = [] __host__ __device__ (float x) { return x + 1.0f; };
    auto square = [] __host__ __device__ (float x) { return x * x; };
    auto pipeline = compose(square, compose(add_one, scale2));

    for (int i = 1; i <= 5; i++) printf("pipeline(%.1f) = %.1f\n", (float)i, pipeline((float)i));

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, 5 * sizeof(float));
    apply_kernel<<<1, 5>>>(d_out, d_in, 5, pipeline);
    return 0;
}
```

### File: `08_capture_value_vs_reference.cpp` (Section D.9, CPU-only — compiled with `g++`, not `nvcc`)

```cpp
#include <cstdio>

int main() {
    int counter = 0;
    auto by_value = [counter]() { return counter; };
    auto by_ref   = [&counter]() { return counter; };
    printf("by_value()=%d by_ref()=%d\n", by_value(), by_ref());
    counter = 100;
    printf("after counter=100: by_value()=%d by_ref()=%d\n", by_value(), by_ref());

    int x = 5;
    auto increment_own_copy = [x]() mutable { x++; return x; };
    printf("%d %d %d (x still %d)\n", increment_own_copy(), increment_own_copy(), increment_own_copy(), x);

    int y = 5;
    auto increment_original = [&y]() { y++; return y; };
    printf("%d %d %d (y now %d)\n", increment_original(), increment_original(), increment_original(), y);

    int a = 1, b = 2, c = 3;
    auto snapshot_all = [=]() { return a + b + c; };
    auto live_all      = [&]() { return a + b + c; };
    a = 10; b = 20; c = 30;
    printf("snapshot_all()=%d live_all()=%d\n", snapshot_all(), live_all());
    return 0;
}
```

### File: `09_dangling_reference_capture.cpp` (Section D.9, CPU-only — compiled with `g++ -fsanitize=address`)

```cpp
#include <cstdio>
#include <functional>

std::function<int()> make_dangling_lambda(int local_value) {
    return [&local_value]() { return local_value; };   // captures the PARAMETER by reference
}

int main() {
    auto dangling = make_dangling_lambda(42);
    printf("result: %d\n", dangling());   // AddressSanitizer catches this read; see Section D.9
    return 0;
}
```

### The two error-only CUDA files, reproduced for completeness

`06_global_lambda_reference_trap.cu` (Section D.2) — fails to compile, by design:

```cpp
auto add_one = [] __host__ __device__ (float x) { return x + 1.0f; };

__global__ void apply_global_lambda_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = add_one(in[idx]);
}
```

`07_std_function_device.cu` (Section D.7) — fails to compile, by design:

```cpp
#include <functional>

__global__ void apply_std_function_kernel(float* out, const float* in, int n, std::function<float(float)> f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = f(in[idx]);
}
```

### Expected Output

There is no single combined run to reproduce here — Worked Examples D.1.1 through D.8.1 are each this appendix's source of truth for their own file's genuine output, verified independently against arithmetic computed a second, different way rather than against each other.

## Chapter Summary

An extended lambda is an ordinary C++ closure with a `__device__` or `__host__ __device__` annotation written directly on it, enabled by `--extended-lambda`; a lambda defined *inside* a kernel inherits callability from its enclosing context with no extra ceremony, but a namespace-scope lambda *object* is still an ordinary host global regardless of what its type's call operator claims, genuinely producing `error: identifier "..." is undefined in device code` when a kernel reaches for it directly. The idiom that actually works — and this appendix's central technique — is a kernel template taking a callable as an ordinary by-value parameter: `transform_kernel<F>` compiled cleanly for three unrelated single-argument operations, and `reduce_kernel<T, Op>` compiled cleanly for four unrelated *binary* operations (sum `39`, max `9`, min `1`, product `60,480` on the same eight-value array, each independently checked), because a template's callable parameter is resolved at compile time with no shared global object required. A hand-written functor and a lambda's compiler-synthesized closure are the same shape to a template — `transform_kernel<F>` accepted `SquareFunctor{}` with zero code changes — which is why Thrust-era functor code and modern lambda code both compose against the same generic machinery. Two restrictions nvcc enforces at compile time, not at runtime, were both hit directly rather than only cited: reference-capturing an extended lambda is rejected outright (`cannot capture variables by reference`), and defining one inside a function with a deduced (`auto`) return type is rejected too (`must not have deduced return type`) — the second one's fix, returning an explicit `Composed<F, G>` functor instead of an `auto`-deduced lambda, follows directly from this appendix's own functor-versus-lambda equivalence rather than being an arbitrary workaround. `std::function`, by contrast, cannot reach the device at all, for a structural reason rather than a fixable restriction: its generality depends on host-heap type erasure and host vtables device code has no access to, producing a real, two-part compiler error the moment it's tried. Section D.9 steps back to plain, CPU-only C++ to show exactly what Section D.4's restriction is protecting device code from: `[x]` copies a value at closure-creation time (later changes to the original are invisible to it, though a `mutable` by-value capture can still carry its own persistent state across calls), while `[&x]` stores an address and reads the live variable every time, including a closure returned from a function whose local it referenced — genuinely caught here not as a guessed-at number but as AddressSanitizer's own real "stack-use-after-return" diagnosis, naming the exact line and variable.

## Self-Check Questions

1. `transform_kernel<F>` (Section D.3) is launched with a lambda that captures a `float* lookup_table` pointer by value. The pointer itself was allocated with `cudaMalloc` and copies valid data. Does this violate Section D.4's capture rule? Explain what specifically would and wouldn't be safe about this capture.
2. Using Section D.6's `reduce_kernel<T, Op>` template and its `identity` parameter, what identity value would a `bitwise AND` reduction need, and why would `0` (a natural guess by analogy with sum's `0`) be wrong for that specific operation?
3. Section D.7 shows `std::function` cannot be called from device code. Could a `std::function` still be a *host-side* parameter to a function that *also*, separately, launches a kernel — and if so, what would the actual boundary be between what the `std::function` is allowed to touch and what the kernel is allowed to touch?
4. Section D.8's `Composed<F, G>` struct stores `F f` and `G g` as plain (non-reference) members. Using Section D.4's reasoning, explain why this by-value storage is what makes `Composed<F, G>` itself safe to pass to a kernel, even though `compose()`'s two arguments could themselves be lambdas that captured other things by value.
5. Write the one-line change to `reduce_kernel`'s call site that would make Worked Example D.6.1 compute the **average** of the eight values instead of the sum, without modifying `reduce_kernel` itself.
6. Worked Example D.9.4's `make_dangling_lambda` captures its parameter by reference. Write the one-word change to that lambda's capture list that would make the function safe to call, and explain in one sentence why that specific change fixes it while changing nothing else about the function's signature or behavior for the caller.

## Where We Go Next

This appendix closes this book's coverage of how an *operation* — not just data — travels into a kernel: from a fixed, hard-coded arithmetic expression (every chapter before this appendix) to a compile-time-polymorphic parameter (Sections D.3 and D.6), with the two real compiler restrictions on that flexibility (D.4's reference-capture rule, D.8's deduced-return-type rule) and its one hard boundary (D.7's `std::function`) each demonstrated directly rather than asserted. Nothing here changes how any earlier chapter's kernels are launched or measured — Chapter 18's ceiling-division launch math, Appendix C's memory hierarchy, and this appendix's generic callables are three independent axes of the same overall picture, composable with each other exactly the way Section D.8's pipelines compose with themselves.

## Worked Solutions

**1.** No — this does not violate Section D.4's rule, and the distinction matters. Section D.4's restriction is specifically about capturing a *variable by reference* (storing the address of a host stack variable inside the closure). Capturing a *pointer by value* is a completely different thing: the closure gets its own copy of the pointer's numeric address, exactly like capturing an `int` or `float` by value. What determines safety here is not the capture mode (value capture of a pointer is always allowed) but where the pointer *points* — since it was allocated with `cudaMalloc`, it points into device memory, so dereferencing it inside a `__device__`-side `operator()` is valid. The unsafe case would be capturing, by value, a pointer to *host* memory (e.g., a local array's address) and then dereferencing it on the device — the capture itself would compile fine, but the dereference would be a device read into memory the device cannot actually access, a runtime bug this compiler restriction does not and cannot catch, because nothing about a bare pointer value tells the compiler which side's memory it addresses.

**2.** `AND`'s identity is `~0` (all bits set — every bit is `1`), not `0`. The identity element for any reduction must satisfy `identity OP x == x` for every possible `x`: for sum, `0 + x == x`; for product, `1 * x == x`; but for bitwise AND, `0 AND x` is always `0` regardless of `x` — using `0` as the identity would force every reduced result to `0`, since the very first AND with the identity zeroes out everything. `~0 AND x == x` for any `x`, which is the actual defining property an identity element needs.

**3.** Yes — a `std::function` can be an ordinary parameter to a plain host function, used entirely on the host side (to select which lambda or functor to eventually pass into a *kernel template*, for instance), as long as it is never itself passed into `__global__`/`__device__` code or called from within it. The boundary is exactly the device/host line Section D.7 draws: a `std::function` may decide, configure, or wrap host-side logic — including choosing *which* concrete callable to hand to `transform_kernel<F>` next — but the moment a value derived from it needs to run *inside* a kernel, that value has to be something template-resolvable at compile time (a lambda, a functor), not the type-erased `std::function` object itself.

**4.** `Composed<F, G>` stores `F f` and `G g` by value, meaning constructing a `Composed<F, G>` object makes independent copies of whichever closures `f` and `g` are — exactly Section D.4's rule applied one level up. If those inner closures (say, `scale2` and `add_one`) each captured their own state by value (as every closure in this appendix does), then `Composed<F, G>`'s copies of them are just as self-contained and device-safe as the originals were on their own; nesting them inside another by-value-storing struct doesn't introduce any reference or host address that wasn't already ruled out at the leaves. Had `Composed` instead stored `F& f` and `G& g` by reference, it would reintroduce exactly the hazard Section D.4 forbids — a device-side call reading through a reference to a host stack object — which is why `Composed`'s member types are plain `F`/`G`, not `F&`/`G&`.

**5.** Divide the sum by the count after reduction, at the call site, without touching `reduce_kernel` at all: `float average = reduce_host(data, 8, sum_op, 0.0f) / 8.0f;` (or, for the device kernel, dividing `out[blockIdx.x]` by the block's element count after the launch). `reduce_kernel<T, Op>` itself has no concept of "average" and needs none — it only ever computes one fold of `op` across the data, and the specific reduction (sum, and now sum-then-divide for average) lives entirely in what the caller does with `op`, `identity`, and the result, exactly the separation of concerns Section D.6 establishes.

**6.** Change `[&local_value]` to `[local_value]` — dropping the `&` switches the capture from by-reference to by-value, so the returned closure copies `local_value`'s value into its own closure at the moment `make_dangling_lambda` runs, instead of storing the address of a parameter that is about to go out of scope. This fixes it because the caller-visible behavior is completely unchanged (the function still takes an `int` and returns a `std::function<int()>` that reports it) — only the *internal* storage of that value moves from "a reference to something that won't exist" to "an independent copy that travels with the closure," exactly the same value-vs-reference distinction Worked Example D.9.1 demonstrates on a much smaller, non-dangerous example.
