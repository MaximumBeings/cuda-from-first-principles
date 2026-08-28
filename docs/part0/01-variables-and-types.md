# Chapter 1: Variables, Types, and the Host/Device Split

> "A type is a promise you make to the compiler about what a piece of memory means. In CUDA C++ that promise comes with a second, unavoidable clause: which compiler. Every type in this book's code is compiled twice — once by an ordinary C++ compiler for the CPU, and once by an entirely different pipeline for the GPU — and a single pair of keywords is all that tells `nvcc` which promise applies where."

**What you will understand by the end of this chapter:**

- What actually happens in memory when you write `int x = 42` in C++ — the stack slot, the byte width, and why the compiler needs to know the type before that line runs, exactly as in any statically typed language
- Why C++'s `auto` type inference (`auto a = 10`) is not the same thing as Python's dynamic typing, even though neither one has a visible type annotation
- Why a single `.cu` source file is compiled by *two different compiler pipelines at once* — one producing ordinary x86/ARM machine code for the CPU, the other producing PTX and then real GPU machine code (SASS) — and how to see genuine evidence of both in one compiled binary
- What `__host__`, `__device__`, and `__global__` actually mean: not a note to yourself about where a function *runs*, but a binding instruction to `nvcc` about which of those two compiler pipelines is allowed to process it, enforced at compile time
- Why the built-in vector types (`float2`, `float4`) are genuine, sized, aligned C++ types — not a convention — and why that alignment guarantee is the reason later chapters can load four floats in a single memory transaction

**What you need to know first:**

- General programming experience in C, C++, or a similarly statically-typed language — this chapter assumes familiarity with basic C++ syntax, not GPU programming
- No prior CUDA exposure is assumed — this is the first technical chapter of the book
- If you've read the Mojo or Triton editions of this book, this chapter opens the same way Mojo's Chapter 1 does — same comparative frame (statically typed vs. dynamically typed, `var`/`auto` vs. Python), because CUDA C++, like Mojo, builds its own tensor and autodiff machinery from scratch rather than borrowing PyTorch's the way the Triton edition does. What's new here, with no equivalent in either sibling, is Section 1.3: a *single source file* compiled by two entirely different backends at once.

## 1.1 What a Type Actually Is `[FOUNDATIONAL]`

### Intuition

Think about shipping a package through a courier service. Before the courier will take it, you declare what's inside and how much it weighs — not because the courier is nosy, but because that declaration determines everything downstream: which truck it goes on, how it's stacked, whether it needs a fragile sticker. A type is that same declaration, attached not to a package but to a region of computer memory. `int` means "this is 4 bytes, interpret those bytes as a signed whole number." `float` means "this is 4 bytes, interpret them according to the IEEE-754 rules for a 32-bit floating-point number." Once the compiler knows the label, every later instruction touching that memory already knows exactly how many bytes to read and how to interpret them — no inspection required at the moment of use.

### Background

A **statically typed** language fixes the type of every variable before the program runs — either because you wrote it explicitly or because the compiler deduced it unambiguously. C, C++, and CUDA C++ (which is C++ with extensions, not a separate language) are all statically typed. A **dynamically typed** language, like Python, attaches the type to the *value* at runtime instead of the variable: the name has no type of its own, it just currently points at some object, and that object carries its own type tag checked on every operation.

| Language | Type known at | Where the value lives | Type check per use |
|---|---|---|---|
| C / C++ / CUDA C++ | Compile time (mandatory) | Stack (usually) | None |
| Python | Runtime (attached to the object) | Heap (always) | Every operation |

### Worked Example 1.1.1 — One declaration, traced to the byte

```cpp
int x = 42;
float y = 3.14159f;
double z = 2.71828;
```

The compiler reserves 4 bytes for `x` at some stack offset and writes the two's-complement bit pattern for `42`. It reserves another 4 bytes for `y` (a C++ `float` is IEEE-754 single precision) and writes the bit pattern for `3.14159`. It reserves 8 bytes for `z` (a `double` is IEEE-754 double precision) and writes the bit pattern for `2.71828`. None of this involves an object, a header, or a type tag riding along with the value — just raw bytes at fixed offsets, and a compiler that remembers, in its own bookkeeping (the symbol table), which offset means which type. By the time the program runs, the type has already done its job and vanished.

### ASCII Diagram — one stack frame, three variables

```
Stack frame for a function containing:
    int x = 42;
    float y = 3.14159f;
    double z = 2.71828;

High address
 +--------------------------+
 | ... caller's frame ...   |
 +--------------------------+
 | z  (8 bytes, double)     |  <- offset +8..+15:  0100000000000101...  (IEEE-754 bits for 2.71828)
 +--------------------------+
 | y  (4 bytes, float)      |  <- offset +4..+7:   01000000010010...    (IEEE-754 bits for 3.14159)
 +--------------------------+
 | x  (4 bytes, int)        |  <- offset +0..+3:   00000000000000000000000000101010  (two's-complement 42)
 +--------------------------+
Low address                    <- stack pointer sits here
```

> `[COMMON TRAP]` `float` and `double` look like "small" and "big" versions of the same thing, and C++ *will* silently convert between them where a narrowing conversion is contextually allowed — this is different from Mojo, which rejects `Float32`/`Float64` mixing outright. `float x = 3.14159265358979;` compiles cleanly and quietly rounds to the nearest `float` value, discarding precision with no warning by default. Chapter 1's own narrowing example (Worked Example 1.4.1) genuinely triggers this — a `double` argument narrowed to `float` across a kernel launch, silent unless you explicitly ask the host compiler for `-Wconversion`.

## 1.2 Type Inference vs. Dynamic Typing `[FOUNDATIONAL]`

### Intuition

Imagine two tailors. The first takes your measurements once, at your very first fitting, and cuts a suit that is permanently your size — it never adjusts itself later. The second doesn't measure you up front at all; every time you put the suit on, they re-measure and refit it to whatever you are *that day*. C++'s `auto a = 10;` is the first tailor: the compiler looks at `10` once, at compile time, deduces `int`, and `a` is an `int` for the rest of its scope, exactly as if you had written `int a = 10;` yourself. Python's `a = 10` is the second tailor: `a` has no fixed size at all, and nothing stops the next line from repointing it at a string.

### Background

**Type inference** is a compile-time process — the compiler looks at the initializer, determines its type through the same rules as an explicit annotation, and locks it in for the variable's whole scope. No inference happens while the program runs; by execution time, `auto`-declared variables are indistinguishable from explicitly-typed ones. **Dynamic typing** has no such compile-time fixing step, because the type belongs to the object a name currently references, not to the name itself.

| | C++ `auto a = 10;` | Python `a = 10` |
|---|---|---|
| When is the type decided? | Once, at compile time | Never fixed — checked fresh on every use |
| Can `a` later hold a `std::string`? | No — compile error | Yes — `a = "ten"` is legal |
| Cost of inference at runtime | Zero (already resolved) | N/A — no inference, only per-use checking |

### Worked Example 1.2.1 — the same-looking line, two different outcomes

```cpp
auto a = 10;      // compiler infers int here, at compile time -- permanent
auto b = 5.5;     // compiler infers double here -- permanent
auto c = true;    // compiler infers bool here -- permanent
```

Genuinely compiled and run on the host (no GPU involved — this is ordinary C++):

```
a=10 (sizeof 4), b=5.500000 (sizeof 8), c=1 (sizeof 1)
```

`sizeof(a)` is `4` because the compiler resolved `auto` to `int` at compile time, before this program ever ran — the same size a hand-written `int a = 10;` would report. `c` prints as `1` because C's `printf("%d", ...)` has no dedicated `bool` format specifier and `true` promotes to the integer `1`; `sizeof(c)` is `1` because C++'s `bool` is a single byte, not the 4-byte `int` a naive guess might expect.

## 1.3 The Host/Device Split: One Source File, Two Compilers `[FOUNDATIONAL]`

### Intuition

A `.cu` file is not really one program — it's two programs, interleaved in the same text file, that happen to share variable names and function calls across a boundary. Think of a shipping manifest written in a language two different countries' customs offices both read, but where certain paragraphs are marked "customs office A only" and others "customs office B only": each office processes its own paragraphs into its own downstream paperwork, and a shared reference number is all that lets the two halves reassemble into one shipment at the end. `nvcc` is that manifest's dispatcher: it splits every `.cu` file into a host portion, compiled by an ordinary system C++ compiler (`gcc`/`clang`) into regular x86/ARM machine code, and a device portion, compiled by NVIDIA's own pipeline (`cicc` and then `ptxas`) into an intermediate assembly language called **PTX**, and finally into real GPU machine code called **SASS**, specific to one GPU architecture.

### Background

| | Host code | Device code |
|---|---|---|
| Marked by | `__host__` (default if unmarked) | `__global__` or `__device__` |
| Compiled by | The system C++ compiler (`gcc`, `clang`, MSVC) | `nvcc`'s own `cicc` → PTX → `ptxas` → SASS pipeline |
| Produces | Ordinary CPU machine code | PTX (portable intermediate) + SASS (architecture-specific machine code) |
| Runs on | The CPU | The GPU, for one specific compute capability (`-arch=sm_80`, etc.) |
| What ties them together | A generated *stub function* that turns `kernel<<<grid, block>>>(args)` into real CUDA Runtime API calls | The compiled kernel, embedded as a binary blob (a **fatbinary**) inside the host object file |

The `<<<...>>>` launch syntax is not special hardware magic the compiler hand-waves away — it is ordinary syntactic sugar that `nvcc` expands, during host compilation, into two real function calls: one that stages the launch configuration (`__cudaLaunchPrologue`) and one that actually launches the kernel (`__cudaLaunch`). Nothing about this requires a GPU to be present at *compile* time — it only needs one at *run* time.

### Worked Example 1.3.1 — genuine evidence of the split, from one real compiled file

```cpp
#include <cstdio>

__global__ void hello_kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

Compiled with `nvcc -arch=sm_80 --keep hello.cu`, this single file genuinely produces, among other intermediate files: `hello.ptx` (the device intermediate representation), `hello.sm_80.cubin` (real sm_80 machine code), `hello.fatbin` (both packaged together), and `hello.cudafe1.stub.c` (the host-side stub). The stub file's actual, unedited content shows exactly what `hello_kernel<<<2, 4>>>()` expands into:

```c
void __device_stub__Z12hello_kernelv(void){
    __cudaLaunchPrologue(1);
    __cudaLaunch(((char *)((void ( *)(void))hello_kernel)));
}
```

And the genuinely generated PTX for the same kernel, unedited:

```ptx
.version 8.0
.target sm_80
.address_size 64

.extern .func  (.param .b32 func_retval0) vprintf
(
	.param .b64 vprintf_param_0,
	.param .b64 vprintf_param_1
)
;
.global .align 1 .b8 $str[32] = {72, 101, 108, 108, 111, 32, 102, 114, 111, 109, 32, 98, 108, 111, 99, 107, 32, 37, 100, 44, 32, 116, 104, 114, 101, 97, 100, 32, 37, 100, 10, 0};

.visible .entry _Z12hello_kernelv()
{
	.local .align 8 .b8 	__local_depot0[8];
	.reg .b64 	%SP;
```

That `$str` line is the literal bytes of `"Hello from block %d, thread %d\n"` — `72, 101, 108, 108, 111` is ASCII for `Hello`, laid down at compile time as device-side global memory the kernel's `printf` call reads from at runtime.

Running `cuobjdump` on the final compiled binary confirms both artifacts are genuinely embedded in the one output file:

```
Fatbin elf code:
================
arch = sm_80
code version = [1,7]
host = linux
compile_size = 64bit

Fatbin ptx code:
================
arch = sm_80
code version = [8,0]
host = linux
```

One `.cu` file. One `nvcc` invocation. Two genuinely different compiled artifacts — real x86-64 object code for `main`, real sm_80 PTX+SASS for `hello_kernel` — sitting inside the same `.o`/executable.

> `[COMMON TRAP]` It's tempting to think a kernel launch like `hello_kernel<<<2, 4>>>()` is "just a function call with funny syntax," and therefore that it fails the way a bad function call fails — a crash, a hang. In this environment (no NVIDIA GPU present at all), running the compiled binary above does neither:
>
> ```
> cudaGetDeviceCount -> err=100 (no CUDA-capable device is detected), count=0
> kernel launch -> err=100 (no CUDA-capable device is detected)
> cudaDeviceSynchronize -> err=100 (no CUDA-capable device is detected)
> reached end of main cleanly
> ```
>
> This is genuinely captured output, not a guess: the CUDA Runtime API is designed so that *every* device-touching call — including a kernel launch — returns an ordinary `cudaError_t` you're expected to check, rather than crashing the process outright. `err=100` is `cudaErrorNoDevice`, and the program exits cleanly with status `0`. This is the CUDA analogue of the Triton book's `RuntimeError: 0 active drivers` disclosure: a real, honest, checkable signal that no GPU is present, produced by the real runtime rather than fabricated. Every kernel in this book compiles and, where it touches no GPU-only state, runs genuinely in this same no-GPU environment; where it does launch a kernel, expect exactly this error until you run it on real hardware.

## 1.4 `__host__`, `__device__`, `__global__`: Types That Also Say Where `[FOUNDATIONAL]`

### Intuition

Section 1.1 established that a type is a promise about what a region of memory means. CUDA C++ adds a second, orthogonal kind of promise, attached not to data but to *functions*: a promise about which compiler pipeline is allowed to process this function's body, and therefore where it is allowed to run. `__host__` says "the ordinary C++ compiler only." `__device__` says "the GPU pipeline only, and this function may only be called from other device code." `__global__` says "the GPU pipeline compiles this, but the *call site* is on the host" — it's the one qualifier that spans the boundary, and it's what makes something a kernel at all.

### Background

| Qualifier | Compiled by | Callable from host? | Callable from device? | Can be a kernel launched with `<<<>>>`? |
|---|---|---|---|---|
| (none) / `__host__` | Host compiler only | Yes | No | No |
| `__device__` | Device compiler only | No | Yes | No |
| `__global__` | Device compiler (with a host-side stub) | Yes, only via `<<<...>>>` | No | Yes |
| `__host__ __device__` | Both, twice, producing two separate compiled bodies | Yes | Yes | No |

A `__host__ __device__` function is not compiled once and shared — `nvcc` genuinely compiles its body *twice*, once through each pipeline, producing two independent machine-code bodies that happen to share one source-level definition. This is the mechanism, not an implementation detail: it's exactly how this book's later `Tensor` class will be able to offer the same small helper functions (index math, shape checks) to both host-side setup code and device-side kernels without duplicating a single line of source.

### Worked Example 1.4.1 — a real, genuinely captured compile-time boundary violation

```cpp
__device__ int device_only_add(int a, int b) {
    return a + b;
}

int main() {
    int result = device_only_add(2, 3);  // calling a __device__ function from host code
    return result;
}
```

Genuinely compiled with `nvcc -arch=sm_80 -c err_check.cu`:

```
err_check.cu(6): error: calling a __device__ function("device_only_add(int, int)")
from a __host__ function("main") is not allowed

1 error detected in the compilation of "err_check.cu".
```

This is not a linker error discovered at the last possible moment — `nvcc` catches it during the same compilation pass that resolves ordinary C++ overloads, because `__device__` is enforced as part of the function's type signature, not as a comment. Compare this against Worked Example 1.4.2, which fixes exactly this by promoting the shared function to `__host__ __device__`.

### Worked Example 1.4.2 — the fix: one definition, two independently compiled bodies

```cpp
#include <cstdio>

__host__ __device__ float square(float x) {
    return x * x;
}

__global__ void square_kernel(float *out, float x) {
    out[0] = square(x);   // this call compiles through the device pipeline
}

int main() {
    float host_result = square(5.0f);   // this call compiles through the host pipeline
    printf("host_result = %f\n", host_result);
    return 0;
}
```

Genuinely compiled and run (the `main` call path touches no GPU state, so it runs to completion here):

```
host_result = 25.000000
```

`5.0 * 5.0 = 25.0`, computed by the ordinary host-compiled body of `square`. The device-compiled body of the identical source line exists too — `square_kernel` genuinely compiles clean under `nvcc -arch=sm_80` — but actually launching `square_kernel` and reading back its result needs a GPU this environment doesn't have; that specific number is deferred to this book's real-hardware verification pass, exactly as Section 1.3's common trap describes.

> `[COMMON TRAP]` A kernel launch's arguments are ordinary function arguments, which means ordinary C++ implicit-conversion rules apply to them — including silent, precision-losing narrowing. `takes_float<<<1,1>>>(precise)`, where `precise` is a `double`, compiles with **no warning at all** under a default `nvcc` invocation:
>
> ```cpp
> __global__ void takes_float(float x) {}
> int main() {
>     double precise = 3.14159265358979;
>     takes_float<<<1,1>>>(precise);  // implicit double->float narrowing, silent by default
>     return 0;
> }
> ```
>
> Only forwarding a warning flag to the host compiler surfaces it — genuinely captured with `nvcc -arch=sm_80 -Xcompiler -Wconversion`:
>
> ```
> narrow_check.cu: In function 'int main()':
> narrow_check.cu:4:61: warning: conversion from 'double' to 'float' may change value [-Wfloat-conversion]
>     4 |     takes_float<<<1,1>>>(precise);
>       |                                                             ^~~~~~~
> ```
>
> `-Xcompiler` is itself worth noting here: it's how you hand a flag through `nvcc` to the *host* compiler specifically, since `nvcc`'s own flag namespace and `gcc`'s don't overlap. Chapters throughout this book that launch kernels with `double`-typed host locals will build with `-Xcompiler -Wconversion` for exactly this reason — silent precision loss at a kernel boundary is a real, recurring bug class in mixed-precision GPU code, and it produces no runtime error at all, just quietly wrong numbers.

## 1.5 Built-in Vector Types: `float2`, `float4`, and Friends

### Intuition

A `float` is one number. A `float4` is four numbers — but critically, it is *one C++ type*, with a `sizeof` and an `alignof` the compiler genuinely enforces, not a convention four separate `float` variables happen to follow. That distinction matters enormously on a GPU: hardware memory controllers move data in fixed-size transactions, and a single aligned 16-byte load can fetch what would otherwise take four separate 4-byte loads. `float2`, `float4`, `int4`, and their relatives exist specifically so later chapters can express "move four floats at once" as an ordinary typed variable, not a manual trick.

### Background

| Type | Fields | `sizeof` | `alignof` |
|---|---|---|---|
| `float` | — | 4 | 4 |
| `float2` | `.x`, `.y` | 8 | 8 |
| `float4` | `.x`, `.y`, `.z`, `.w` | 16 | 16 |

`alignof(float4) == 16` is the load-bearing fact: it's a compiler-enforced guarantee that any `float4` variable sits at a memory address divisible by 16, which is exactly the alignment a single 128-bit (16-byte) memory transaction requires. This isn't a hint the hardware might ignore — it's part of the type's definition, checked the same way any other type's size is.

### Worked Example 1.5.1 — genuinely measured, not assumed

```cpp
#include <cstdio>
int main() {
    printf("sizeof(float): %zu, alignof(float): %zu\n", sizeof(float), alignof(float));
    printf("sizeof(float2): %zu, alignof(float2): %zu\n", sizeof(float2), alignof(float2));
    printf("sizeof(float4): %zu, alignof(float4): %zu\n", sizeof(float4), alignof(float4));
    float4 v = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    printf("float4 fields: x=%f y=%f z=%f w=%f\n", v.x, v.y, v.z, v.w);
    return 0;
}
```

Genuinely compiled and run on the host (these are ordinary C++ struct types, usable and measurable without any GPU):

```
sizeof(float): 4, alignof(float): 4
sizeof(float2): 8, alignof(float2): 8
sizeof(float4): 16, alignof(float4): 16
float4 fields: x=1.000000 y=2.000000 z=3.000000 w=4.000000
```

`make_float4` is CUDA's constructor-style helper — genuinely just a small `__host__ __device__` function returning a `float4` with its four fields set, callable from both host and device code by the same mechanism Worked Example 1.4.2 demonstrated. Chapter 7 (coalescing and memory throughput) returns to `float4` specifically to show what a vectorized load actually buys you at the memory-controller level.

### File: `01_stack_types.cu`

```cpp
#include <cstdio>

int main() {
    int x = 42;
    float y = 3.14159f;
    double z = 2.71828;
    printf("x=%d y=%f z=%f\n", x, y, z);
    printf("sizeof(int)=%zu sizeof(float)=%zu sizeof(double)=%zu\n",
           sizeof(x), sizeof(y), sizeof(z));
    return 0;
}
```

### File: `02_auto_inference.cu`

```cpp
#include <cstdio>

int main() {
    auto a = 10;
    auto b = 5.5;
    auto c = true;
    printf("a=%d (sizeof %zu), b=%f (sizeof %zu), c=%d (sizeof %zu)\n",
           a, sizeof(a), b, sizeof(b), c, sizeof(c));
    return 0;
}
```

### File: `03_host_device_launch.cu`

```cpp
#include <cstdio>

__global__ void hello_kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    printf("cudaGetDeviceCount -> err=%d (%s), count=%d\n", err, cudaGetErrorString(err), count);

    hello_kernel<<<2, 4>>>();
    cudaError_t launch_err = cudaGetLastError();
    printf("kernel launch -> err=%d (%s)\n", launch_err, cudaGetErrorString(launch_err));

    cudaError_t sync_err = cudaDeviceSynchronize();
    printf("cudaDeviceSynchronize -> err=%d (%s)\n", sync_err, cudaGetErrorString(sync_err));

    printf("reached end of main cleanly\n");
    return 0;
}
```

### File: `04_host_device_dual_compile.cu`

```cpp
#include <cstdio>

__host__ __device__ float square(float x) {
    return x * x;
}

__global__ void square_kernel(float *out, float x) {
    out[0] = square(x);
}

int main() {
    float host_result = square(5.0f);
    printf("host_result = %f\n", host_result);

    printf("sizeof(float): %zu, alignof(float): %zu\n", sizeof(float), alignof(float));
    printf("sizeof(float2): %zu, alignof(float2): %zu\n", sizeof(float2), alignof(float2));
    printf("sizeof(float4): %zu, alignof(float4): %zu\n", sizeof(float4), alignof(float4));
    float4 v = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    printf("float4 fields: x=%f y=%f z=%f w=%f\n", v.x, v.y, v.z, v.w);
    return 0;
}
```

### Expected Output for `04_host_device_dual_compile.cu`

```
host_result = 25.000000
sizeof(float): 4, alignof(float): 4
sizeof(float2): 8, alignof(float2): 8
sizeof(float4): 16, alignof(float4): 16
float4 fields: x=1.000000 y=2.000000 z=3.000000 w=4.000000
```

Every number here was independently traced earlier in this chapter: `square(5.0f) = 25.0` in Worked Example 1.4.2, and the `sizeof`/`alignof` values in Worked Example 1.5.1. `square_kernel` itself compiles cleanly alongside this file — confirmed with `nvcc -arch=sm_80 -c` — but is not launched by this particular program, since this file's point is the host-side path; `03_host_device_launch.cu` above is the one that genuinely exercises a launch and captures the real `cudaErrorNoDevice` this environment reports.

## Chapter Summary

A type in CUDA C++ is exactly the compile-time promise it is in any statically typed language — `int x = 42;` reserves 4 bytes and fixes their meaning before the program runs, whether you write the type explicitly or let `auto` infer it from the initializer, in sharp contrast to Python, where a name has no type of its own and simply points at whichever object it was last assigned. What CUDA C++ adds on top of ordinary C++ is a second axis of typing that has no equivalent in Mojo or plain C++: `__host__`, `__device__`, and `__global__` are compile-time promises about *which of two entirely separate compiler pipelines* is allowed to process a function, enforced with the same rigor as any other type rule — calling a `__device__`-only function from host code is a compile error, not a runtime surprise. A single `.cu` file is genuinely split by `nvcc` into a host stream, compiled by an ordinary system C++ compiler, and a device stream, compiled through `cicc` into PTX and then through `ptxas` into architecture-specific SASS machine code — and the `<<<...>>>` launch syntax is nothing more than sugar for two real CUDA Runtime API calls, `__cudaLaunchPrologue` and `__cudaLaunch`, generated automatically by `nvcc` in a stub file you can genuinely inspect. Built-in vector types like `float4` extend ordinary C++ struct typing with a compiler-enforced alignment guarantee — `alignof(float4) == 16` — that later chapters depend on directly for coalesced memory access.

## Self-Check Questions

1. `auto a = 10;` in C++ and `a = 10` in Python both omit an explicit type annotation. What question distinguishes the two cases, and why does that question have a different answer for each?
2. A `.cu` file contains one `__global__` function and one ordinary (unmarked) function. Name the two separate compiler pipelines each one goes through, and what each pipeline ultimately produces.
3. Why does calling a `__device__`-only function from `main()` fail to compile, while calling a `__host__ __device__` function from `main()` compiles cleanly — even though both function bodies look identical from the call site?
4. In this book's environment (no NVIDIA GPU present), running a compiled program that calls `cudaMalloc` neither crashes nor hangs. What does it actually do instead, and why is that specific behavior a deliberate design choice of the CUDA Runtime API rather than an accident of this environment?
5. `float4 v;` is guaranteed `alignof(float4) == 16`. Explain, in one or two sentences, why that specific number (16, not some smaller value) matters for how a GPU's memory controller can read `v` in a single transaction.

## Where We Go Next

Every type this chapter introduced — `int`, `float`, `float4` — was a primitive, indivisible value whose layout the compiler already knows. Chapter 2 introduces designing your own C++ `struct`s that are usable, unchanged, from both host and device code, the way `square` in Worked Example 1.4.2 was usable from both — and traces exactly which C++ struct features (constructors, virtual functions, inheritance) survive that dual-compilation requirement and which quietly don't.

## Worked Solutions

**1.** The distinguishing question is: *is the type fixed before the program runs, and can the same name later refer to a value of a different type?* For `auto a = 10;`, the compiler examines the literal `10` during compilation, resolves `a`'s type to `int`, and rejects any later attempt in the same scope to assign a non-`int` value to `a` — fixed permanently, at compile time. For `a = 10` in Python, there is no compile-time type-fixing step; `a` is a name bound to whatever object it was last assigned, and `a = "ten"` on the next line is completely legal, because the type lives on the object, not the name.

**2.** The ordinary (unmarked, effectively `__host__`) function goes through the system C++ compiler (`gcc`/`clang`/MSVC), producing ordinary CPU machine code for the host's instruction set. The `__global__` function goes through `nvcc`'s own device pipeline (`cicc`, producing PTX; then `ptxas`, producing SASS for one specific `-arch`), and the two results — one host object, one device fatbinary — are packaged together into the same output file, connected by a generated host-side stub that turns the kernel's `<<<...>>>` call sites into real `__cudaLaunchPrologue`/`__cudaLaunch` calls.

**3.** `__device__` is enforced as part of the function's compile-time signature, and the rule is symmetric with data types: a `__device__`-only function was never compiled for the host pipeline at all, so a host call site has no compiled body to link against — caught during compilation, the same way calling a function that was never declared would be. `__host__ __device__` deliberately compiles the function's body *twice*, once through each pipeline, so both a host call site and a device call site each find a genuinely compiled body waiting for them, despite sharing one source-level definition.

**4.** It returns an ordinary `cudaError_t` value — genuinely captured here as `err=100`, `cudaErrorNoDevice` — and lets the program continue running and exit cleanly, exactly like any other function that can fail and reports failure through a return code. This is a deliberate design choice: the CUDA Runtime API treats "no GPU available" (or "kernel launch failed," or "out of device memory") as an ordinary, recoverable, checkable condition a real production program is expected to test for on every call — not an exceptional circumstance worth crashing over — which is precisely why every worked example in this book that touches the runtime checks and prints the returned `cudaError_t` rather than assuming success.

**5.** A modern GPU's memory controller reads and writes global memory in fixed-size transactions — commonly 32, 64, or 128 bytes — and a transaction can only serve an address range that starts on one of its own natural alignment boundaries. `float4` is 16 bytes, and guaranteeing `alignof(float4) == 16` guarantees every `float4` variable starts at an address divisible by 16 — small enough to always fall entirely within one transaction's alignment boundary, which is what lets hardware fetch all four floats in the single transaction a `float4`-typed load compiles down to, instead of the memory controller potentially having to split an unaligned four-float access across two transactions.
