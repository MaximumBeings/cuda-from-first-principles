# Appendix G: The Rule of Five, and a Risk Engine to Exercise It

> "Section G.1 starts from `Point2D` — the same struct, called from a kernel instead of only from host code, with nothing about its definition changed. The reason nothing needs to change is the whole subject of this appendix: `Point2D` owns no resource, only two `float`s, so the compiler's default copy is exactly correct. The rest of this appendix is about what happens the moment a struct owns something — heap memory, device memory — and the compiler's default copy silently stops being correct without ever telling you."

## What you will understand by the end of this appendix

- Why `Point2D`'s compiler-generated copy constructor is completely safe, and exactly which property of `Point2D` makes that true — demonstrated first by constructing `Point2D` objects directly inside a `__global__` kernel, using the identical `__host__ __device__` methods host code already uses.
- The specific bug that appears the moment a struct owns a heap resource and leaves its copy constructor and copy assignment operator to the compiler: a double free, caught here not by describing it but by genuinely triggering it under AddressSanitizer and reading its real, unedited `heap-use-after-free` report.
- All five special member functions of the C++ Rule of Five — destructor, copy constructor, copy assignment, move constructor, move assignment — implemented correctly for the same struct, with every one of the five genuinely exercised and its effect independently checked (pointer identity, nulled sources, freed old resources), not merely asserted.
- Why the identical Rule of Five pattern applies to a `cudaMalloc`'d device buffer, and why — unlike `Point2D` — a struct wrapping device memory is necessarily host-only, a direct consequence of `cudaMalloc`/`cudaFree` being host-side Runtime API entry points rather than a stylistic choice.
- How to verify Rule-of-Five behavior for device memory honestly in a sandbox with no GPU, where every `cudaMalloc` call fails and every pointer is null — by switching from comparing addresses (which would trivially and misleadingly show everything "equal") to counting API calls (evidence that survives every allocation failing).
- Why the self-assignment and self-move guards in every copy/move assignment operator above actually matter — tested, not just written, by removing each guard in turn and observing two genuinely different failure modes (silent data corruption from copying uninitialized memory onto itself, and silent data loss from one assignment's own statements overwriting each other), neither of which AddressSanitizer's heap checks catch.
- The Rule of Zero — wrapping an owned resource in something that already manages itself correctly (`std::vector` in place of a raw `new[]`'d array) so the compiler's default special members become safe again, with zero hand-written members, exactly as they already were for `Point2D`.
- Copy-and-swap — the technique that makes self-assignment safety and exception safety fall out of one mechanism, verified against Section G.3's own free-then-allocate ordering by genuinely injecting an allocation failure into both: the naive version is left holding a pointer to already-freed memory (an ASan-confirmed heap-use-after-free just from reading it back), while the copy-and-swap version comes back completely unchanged.
- Why every move constructor and move assignment operator in this appendix is marked `noexcept` — demonstrated by putting two otherwise-identical structs in a `std::vector` and triggering reallocation: the non-`noexcept` one is silently relocated by copying, the `noexcept` one by moving, with no compiler diagnostic marking the difference either way.
- Value-at-Risk and its variants — simulated/historical, parametric (variance-covariance), and Conditional VaR/Expected Shortfall — computed on the same GBM machinery Chapter 22.4 already established, with the CVaR ≥ VaR invariant checked genuinely rather than assumed.
- XVA and its variants — CVA, DVA, and FVA — built from a genuine exposure profile (`EE(t)`/`ENE(t)`) extracted from checkpointed Monte Carlo paths of a forward contract, cross-checked against the closed-form GBM mean, plus honest scope notes on MVA/KVA and on wrong-way risk.

## What you need to know first

- Ordinary C++ constructors, destructors, and the difference between a value and a reference — this appendix assumes that baseline and builds the Rule of Five on top of it.
- Appendix D.9's dangling-reference / stack-use-after-return trap, and this appendix's Section G.2's double-free — both use AddressSanitizer as a tool for demonstrating undefined behavior honestly, and the two bugs are close relatives: one from a reference outliving what it refers to, one from a pointer being freed twice.
- Chapter 22.4's GBM path simulation (`box_muller`, the xorshift PRNG, the GBM update step) — Sections G.5 and G.6 reuse that exact machinery rather than introducing a new one, so familiarity with where those numbers come from helps but isn't required to follow the new sections' own worked checks.
- Basic familiarity with what a forward contract and a discount factor are helps for Section G.6, but every formula used there is derived and checked inline.

## G.1 Calling `Point2D` from a Kernel, and Why Its Default Copy Is Safe `[FOUNDATIONAL]`

### Intuition

`Point2D` is nothing but two `float`s and three `__host__ __device__` member functions that read them. There is exactly one way to construct a `Point2D`, and copying one is exactly as simple as copying a pair of numbers — there is no hidden resource anywhere in the struct that a naive copy could get wrong. That triviality is precisely why the compiler's default copy constructor, the one nobody wrote by hand, is completely correct for `Point2D`. This section demonstrates that safety directly on the host, then shows the identical struct and its identical methods being constructed and used *inside* a `__global__` kernel — not because anything about `Point2D` needs to change for device code, but because nothing does, and seeing why is the setup for every section that follows.

### Background

`__host__ __device__` on a constructor or method means the compiler generates two versions of it — one that runs on the CPU, one that runs on the GPU — from the same source text. Nothing else is required to call `Point2D`'s constructor or its methods from inside a kernel: ordinary C++ object construction, `Point2D p(xs[idx], ys[idx]);`, works identically to constructing it in `main()`, because both target's compiled code came from the same lines.

### Worked Example G.1.1 — Host copy safety, then the same struct from a kernel

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

__global__ void compute_distances_kernel(float* out_from_origin, float* out_between,
                                          const float* xs, const float* ys, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Point2D p(xs[idx], ys[idx]);                 // constructed ON THE DEVICE
        Point2D origin_reference(0.0f, 0.0f);          // a second device-constructed object
        out_from_origin[idx] = p.distance_from_origin();
        out_between[idx] = p.distance_to(origin_reference);
    }
}
```

Genuinely compiled with `nvcc -arch=sm_80` and run:

```
sizeof(Point2D) = 8
point1.distance_from_origin() = 5.000000
point1.distance_to(point2) = 3.605551
point1 = (3.000000, 4.000000)
point2 (copy) = (3.000000, 4.000000)
after point2.x = 99.0: point1 = (3.000000, 4.000000), point2 = (99.000000, 4.000000)

(point1 was NOT affected by mutating point2b -- the compiler-generated
copy constructor copied x and y by VALUE, and neither member is a pointer
to anything shared, so this default copy is completely safe. Section G.2
shows exactly what changes the moment a struct owns a pointer instead.)

--- calling the SAME Point2D methods FROM a kernel ---
host-side reference, using the exact same Point2D methods:
  Point2D(3.0, 4.0).distance_from_origin() = 5.000000
  Point2D(0.0, 0.0).distance_from_origin() = 0.000000
  Point2D(1.0, 1.0).distance_from_origin() = 1.414214
  Point2D(-2.0, 2.0).distance_from_origin() = 2.828427
  Point2D(5.0, 12.0).distance_from_origin() = 13.000000

cudaMalloc: cudaErrorNoDevice
compute_distances_kernel launch (constructs Point2D ON THE DEVICE): cudaErrorNoDevice
(no CUDA-capable device in this sandbox -- genuinely attempted, honestly failed.
What compiled cleanly: Point2D's constructor and both methods running INSIDE a
__global__ function, constructing objects with `Point2D p(xs[idx], ys[idx]);`
directly in device code -- ordinary C++ object construction, on the device.)
```

The 3-4-5 and 5-12-13 triangles cross-check the host-side `distance_from_origin` values by hand: `sqrt(3² + 4²) = 5`, `sqrt(5² + 12²) = 13`, both matching the printed output exactly. The kernel launch itself honestly reports `cudaErrorNoDevice`, the same sandbox limitation every CUDA appendix and chapter in this book from Chapter 18 onward reports rather than hiding — what's genuinely verified here is that `Point2D`'s constructor and methods compile and are legal to call *from device code*, not that a real device executed them.

**[COMMON TRAP]** It is tempting to think `__host__ __device__` is required on every struct that might ever appear in a kernel. It is required only on the specific constructors and methods actually called from device code — and, as Section G.4 shows, there are structs for which adding it would be a compile error, because what they own can only be manipulated from the host.

## G.2 The Bug the Rule of Five Prevents: A Double Free `[FOUNDATIONAL]`

### Intuition

`Point2D`'s default copy was safe because copying two `float`s twice produces two independent objects. The moment a struct's constructor allocates something — `new[]`, `cudaMalloc`, a file handle, anything acquired and requiring explicit release — a memberwise copy stops copying the *resource* and starts copying the *pointer to* the resource. Two objects then believe they each own the same memory, and whichever one is destroyed second frees memory that's already been freed. This section triggers that bug on purpose, on a struct shaped exactly like the ones this book's later chapters actually use — one wrapping a heap array that stands in for the `cudaMalloc`'d device buffer Section G.4 builds next.

### Worked Example G.2.1 — A genuine, ASan-caught heap-use-after-free

```cpp
struct RiskPathBuffer {
    float* paths;
    int count;

    RiskPathBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = (float)i;
    }
    ~RiskPathBuffer() {
        delete[] paths;
    }
    // No copy constructor, no copy assignment operator written here --
    // the compiler generates memberwise-copy versions of both, exactly
    // as it did for Point2D.
};

int main() {
    RiskPathBuffer a(10);
    printf("a.paths[0] = %f\n", a.paths[0]);
    {
        RiskPathBuffer b = a;   // compiler-generated copy: copies the POINTER, not the array
        printf("b.paths[0] = %f (b.paths and a.paths are the SAME address: %s)\n",
               b.paths[0], (b.paths == a.paths) ? "true" : "false");
    }   // b's destructor runs HERE, freeing the SAME memory a.paths still points to
    printf("a.paths[0] = %f (reading through a AFTER b's destructor already freed this memory)\n", a.paths[0]);
    return 0;
}   // a's destructor now runs AGAIN on the SAME pointer -- a double free
```

Genuinely compiled with `g++ -fsanitize=address -g` and run — standard output before the crash, then AddressSanitizer's real, unedited report on standard error:

```
a.paths[0] = 0.000000
b.paths[0] = 0.000000 (b.paths and a.paths are the SAME address: true)
```

```
=================================================================
==1900==ERROR: AddressSanitizer: heap-use-after-free on address 0x504000000010 at pc 0x55d06fc3d6dd bp 0x7ffe1760fcf0 sp 0x7ffe1760fce0
READ of size 4 at 0x504000000010 thread T0
    #0 0x55d06fc3d6dc in main /tmp/cuda_appendix_g/02_double_free_trap.cpp:35

0x504000000010 is located 0 bytes inside of 40-byte region [0x504000000010,0x504000000038)
freed by thread T0 here:
    #0 ... operator delete[](void*) ...
    #1 ... RiskPathBuffer::~RiskPathBuffer() /tmp/cuda_appendix_g/02_double_free_trap.cpp:18
    #2 ... main /tmp/cuda_appendix_g/02_double_free_trap.cpp:32

previously allocated by thread T0 here:
    #0 ... operator new[](unsigned long) ...
    #1 ... RiskPathBuffer::RiskPathBuffer(int) /tmp/cuda_appendix_g/02_double_free_trap.cpp:14
    #2 ... main /tmp/cuda_appendix_g/02_double_free_trap.cpp:26
SUMMARY: AddressSanitizer: heap-use-after-free ... in main
```

The two `printf` lines confirm, before anything crashes, exactly what was suspected: `b.paths == a.paths` is `true` — the compiler-generated copy constructor copied the pointer value, not the sixteen bytes it points to. ASan's report then confirms the consequence precisely where it happens: the read at line 35 (`a.paths[0]` after `b`'s scope ends) touches memory ASan's own bookkeeping shows was already freed by `RiskPathBuffer::~RiskPathBuffer()` at line 18, called from `main` at line 32 — exactly the moment `b` went out of scope. Note that only the deterministic, ASan-diagnosed failure is reported here, not raw undefined-behavior output: without ASan, this same program might print a stale-but-plausible value, or corrupt unrelated memory silently, or crash somewhere else entirely — genuinely undefined, and not something this book will report as if it were a fixed, reproducible number.

**[COMMON TRAP]** The bug here isn't `b`'s destructor running — that's correct and expected. The bug is that `a`'s destructor, when `main` returns, will *also* try to free the same pointer, because nothing about `a`'s copy of `paths` was ever invalidated by `b`'s destruction. A double free's second `delete[]`/`free` call is its own separate piece of undefined behavior, on top of the use-after-free already shown above.

## G.3 The Rule of Five, Correctly Implemented `[FOUNDATIONAL]`

### Intuition

The Rule of Five says: if a class needs to define any one of {destructor, copy constructor, copy assignment operator, move constructor, move assignment operator} because it manages a resource, it almost always needs to define all five. `RiskPathBuffer` only had a destructor in Section G.2 — exactly the situation the Rule of Five warns about, because leaving copy construction and copy assignment to the compiler's default left them copying the pointer rather than the resource. This section implements the same struct with all five special members written explicitly and correctly, and genuinely exercises each one.

### Background

The five members split into two clearly different jobs. Destructor, copy constructor, and copy assignment are about **ownership without transfer**: an object is destroyed and its resource released; an object is copied and a *new*, independent resource must be created for the copy (a "deep copy"), with copy assignment additionally responsible for releasing whatever resource the target already owned *before* taking on the new one. Move constructor and move assignment are about **ownership transfer**: the source object's resource is *stolen* — its pointer is copied into the destination, then the source is nulled out so its own destructor, which will still run, frees nothing. A moved-from object must be left in a state its destructor can safely run against; setting the stolen pointer to `nullptr` is what makes `delete[]`/`cudaFree` on it a safe no-op rather than a second bug.

### Worked Example G.3.1 — All five members, each one genuinely exercised

```cpp
struct RiskPathBuffer {
    float* paths;
    int count;

    RiskPathBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = (float)i;
    }

    ~RiskPathBuffer() {
        delete[] paths;
    }

    // Copy constructor -- a DEEP copy: a brand-new array, contents duplicated.
    RiskPathBuffer(const RiskPathBuffer& other) : count(other.count) {
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
    }

    // Copy assignment -- deep copy, self-assignment check, and the OLD
    // resource must be freed before the new one is taken on, or this
    // leaks exactly the memory the old `paths` pointed to.
    RiskPathBuffer& operator=(const RiskPathBuffer& other) {
        if (this == &other) return *this;         // self-assignment guard
        delete[] paths;                            // free what THIS object already owned
        count = other.count;
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
        return *this;
    }

    // Move constructor -- STEAL the source's pointer, then null the
    // source out so its destructor frees nothing.
    RiskPathBuffer(RiskPathBuffer&& other) noexcept : paths(other.paths), count(other.count) {
        other.paths = nullptr;
        other.count = 0;
    }

    // Move assignment -- free what THIS object owned, steal the source's
    // pointer, null the source, with a self-move guard.
    RiskPathBuffer& operator=(RiskPathBuffer&& other) noexcept {
        if (this == &other) return *this;
        delete[] paths;
        paths = other.paths;
        count = other.count;
        other.paths = nullptr;
        other.count = 0;
        return *this;
    }
};
```

Genuinely compiled with `g++ -Wall -Wextra -fsanitize=address -g` (zero warnings) and run — ASan reports no issues at all:

```
--- copy construction: independent, deep-copied buffers ---
a.paths == b.paths? false (deep copy: must be false)
after b.paths[0]=999: a.paths[0]=0.0 (untouched), b.paths[0]=999.0

--- move construction: ownership transfers, source is nulled ---
d.paths == c's ORIGINAL pointer? true
c.paths is now: (nil) (nullptr, so c's destructor will free nothing)

--- copy assignment: old resource freed, new one deep-copied ---
f.count is now: 2 (was 7, now matches e's count)
f.paths == e.paths? false (deep copy: must be false)

--- move assignment: target's OLD resource freed, then source's pointer stolen ---
g.paths == h's ORIGINAL pointer? true
h.paths is now: (nil) (nullptr, so h's destructor will free nothing)
```

Every one of the five members is exercised, not merely written: copy construction is checked against Section G.2's own failure mode (`a.paths == b.paths` is now `false`, and mutating `b` genuinely leaves `a` untouched); move construction is checked by comparing `d.paths` against a pointer value captured *before* the move (`d` ends up with exactly the address `c` originally held, and `c` is left null); copy assignment is checked by confirming the target's `count` and `paths` genuinely change to match the source, independently; move assignment is checked the same way move construction was, against a captured pre-move pointer. ASan's silence — a clean run with no reports — is itself evidence: this exact resource-ownership shape, exercised through all five paths, produces no double free, no leak, and no use-after-free.

### Worked Example G.3.2 — Self-assignment and self-move: the guards, actually tested

Both `operator=` overloads above open with `if (this == &other) return *this;`, but nothing in Worked Example G.3.1 ever writes `a = a` or `b = std::move(b)` — the two cases that guard exists for. Claiming the guard matters without ever triggering the case it guards against is exactly the kind of unverified claim this book avoids elsewhere; this section closes that gap, on the correct struct and on two deliberately un-guarded variants of it.

```cpp
RiskPathBuffer a(5);
float* a_ptr_before = a.paths;
a = a;   // self copy-assignment
// ... a.paths and a.count come back unchanged

RiskPathBuffer b(7);
float* b_ptr_before = b.paths;
b = std::move(b);   // self move-assignment
// ... b.paths and b.count come back unchanged
```

Genuinely compiled and run:

```
--- Guarded RiskPathBuffer: a = a (self copy-assignment) ---
a.paths unchanged: true (0x503000000040 -> 0x503000000040)
a.count unchanged: true (5)

--- Guarded RiskPathBuffer: b = std::move(b) (self move-assignment) ---
b.paths unchanged: true (0x503000000070 -> 0x503000000070)
b.count unchanged: true (7)
```

Both guards work exactly as claimed. `g++` itself flags the literal self-move at compile time — `warning: moving 'b' of type 'RiskPathBuffer' to itself [-Wself-move]` — a genuine, unedited compiler warning worth noting rather than suppressing: the compiler is telling you this exact call pattern is suspicious enough to warn about even though, here, the guard makes it safe.

Now the same two operations, on structs whose copy and move assignment are identical to `RiskPathBuffer`'s *except* for the missing guard:

```cpp
struct UnguardedCopyBuffer {
    float* paths;
    int count;
    // ... constructor, destructor identical to RiskPathBuffer ...
    UnguardedCopyBuffer& operator=(const UnguardedCopyBuffer& other) {
        // NO "if (this == &other) return *this;" guard here.
        delete[] paths;
        count = other.count;
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
        return *this;
    }
};
```

The expectation going in was a heap-use-after-free, the same failure Section G.2 caught — `delete[] paths` frees the resource, and since `other` and `*this` are the same object for `x = x`, `other.paths` should be reading through the same, now-dangling pointer. That is **not** what genuinely happened:

```
--- UNGUARDED copy assignment: x = x (no self-assignment guard) ---
x.paths BEFORE self-assignment: 0.0 1.0 2.0 3.0
x.paths AFTER  self-assignment: -0.4 -0.4 -0.4 -0.4
(NOT a crash, and NOT what was expected going in -- see the explanation below)
```

No ASan report, no crash — and the reason is worth tracing exactly, since assuming the "obvious" bug without checking would have been a genuine fabrication. `other` and `*this` are the same object, so `other.paths` is just another *name* for `x.paths`. By the time `paths = new float[count];` executes, that line has already overwritten `other.paths` too — they're the same variable. So the following `memcpy(paths, other.paths, ...)` copies the brand-new, **uninitialized** allocation onto itself, not the freed original — a real bug (the original values `0,1,2,3` are genuinely destroyed, replaced with allocator garbage), but a silent-corruption bug, not a use-after-free. AddressSanitizer's heap checks have nothing invalid to catch here — the pointer being read is perfectly valid, just uninitialized; a tool for uninitialized-read detection (e.g. MemorySanitizer) is the one that would flag this specific case.

The unguarded *move* case, run as its own program (the copy case above doesn't crash, but keeping them separate mirrors how Section G.2's crash forced a process boundary):

```cpp
UnguardedMoveBuffer& operator=(UnguardedMoveBuffer&& other) noexcept {
    // NO "if (this == &other) return *this;" guard here.
    delete[] paths;
    paths = other.paths;
    count = other.count;
    other.paths = nullptr;
    other.count = 0;
    return *this;
}
```

```
=== UNGUARDED move assignment: y = std::move(y) (no self-move guard) ===

before self-move: y.paths=0x503000000040, y.count=6
after  self-move: y.paths=(nil), y.count=0
```

Again, no crash and no ASan report — a *third* distinct failure mode. `delete[] paths` frees the buffer; `paths = other.paths` reads the (already-freed, but not-yet-nulled) pointer value back into itself, harmless in isolation; but the next line, `other.paths = nullptr`, operates on the *same* object, so it immediately overwrites the assignment that just happened. The object silently ends up empty — `count=0`, `paths=nullptr` — instead of unchanged. Arguably the most dangerous of the three failure modes shown across this appendix (Section G.2's double free, this section's uninitialized-data corruption, and this silent emptying): nothing here ever touches invalid memory, so no sanitizer this book has used would ever catch it — only a value quietly discarded.

> `[COMMON TRAP]` It's tempting to assume every self-assignment bug looks like Section G.2's double free — a crash a sanitizer catches. This section's two unguarded variants show two *different* failure modes, neither of which AddressSanitizer flags at all: silent corruption from copying uninitialized memory onto itself, and silent data loss from one assignment's own two statements stepping on each other. The guard's job isn't "prevent a crash" specifically — it's "prevent `other` and `*this` from ever being treated as two different objects when they're actually the same one," and the specific way that assumption breaks depends entirely on the exact statements in the assignment operator's body.

### Worked Example G.3.3 — The Rule of Zero: making the default safe again

Section G.1 established that `Point2D`'s compiler-generated copy is safe because it owns nothing beyond two plain `float`s. The **Rule of Zero** is the observation that this situation can be *engineered back into existence* for a resource-owning struct too: wrap the resource in something that already manages itself correctly, and the compiler's defaults become correct again, exactly as they were for `Point2D` — with zero hand-written special members, not five carefully-written ones.

```cpp
struct RiskPathBufferV2 {
    std::vector<float> paths;   // NOT a raw pointer

    RiskPathBufferV2(int n) : paths(n) {
        for (int i = 0; i < n; i++) paths[i] = (float)i;
    }
    // No destructor, no copy constructor, no copy assignment, no move
    // constructor, no move assignment written ANYWHERE in this struct.
};
```

Genuinely compiled and run:

```
--- copy construction: still deep, still independent -- for free ---
a.paths.data() == b.paths.data()? false (deep copy: must be false)
after b.paths[0]=999: a.paths[0]=0.0 (untouched), b.paths[0]=999.0

--- move construction: still steals the buffer, still nulls the source -- for free ---
d.paths.data() == c's ORIGINAL pointer? true
c.paths.size() after move: 0 (empty -- std::vector's own move leaves the source valid but empty)

--- self-assignment and self-move: SAFE, with no guard written anywhere ---
e.paths.data() unchanged after e=e: true
e.paths[0..3] after e=e: 0.0 1.0 2.0 3.0  (still 0,1,2,3 -- NOT corrupted,
unlike Worked Example G.3.2's UnguardedCopyBuffer, because std::vector's own
assignment operator was written by people who already solved this problem)
```

Every property Section G.3.1 spent five hand-written functions establishing — deep-copy independence, correct pointer transfer on move, a nulled-but-valid moved-from source — comes back identically here, all five inherited for free from `std::vector<float>`'s own, already-correct Rule of Five, itself written once by the standard library instead of once per struct that needs it. Section G.3.2's self-assignment corruption doesn't reappear either: `std::vector`'s own `operator=` already handles `e = e` correctly internally, which is exactly why `e.paths[0..3]` reads back `0,1,2,3` unchanged rather than the garbage Worked Example G.3.2's `UnguardedCopyBuffer` produced.

This does not make Section G.3's hand-written version pointless — Section G.4's `GPUPathBuffer` genuinely cannot use this trick, because there is no `std::vector`-equivalent standard container for `cudaMalloc`'d device memory to delegate to (a hand-rolled thin RAII wrapper around `cudaMalloc`/`cudaFree` would itself need exactly Section G.4's five members, just written once and reused). The Rule of Zero and the Rule of Five are not competitors — the Rule of Zero is what the Rule of Five is *for*: writing the five correctly, once, inside whatever wrapper type a codebase reuses, so that everything built on top of it gets to follow the Rule of Zero instead.

> `[COMMON TRAP]` It's tempting to read the Rule of Zero as "you never need the Rule of Five." Someone has to write the five correctly at least once — `std::vector`'s own implementation is not exempt from Section G.3's reasoning, it simply already did the work Section G.3 did by hand. A codebase with no resource-owning type more primitive than `std::vector`/`std::unique_ptr` can follow the Rule of Zero everywhere; a codebase that talks to `cudaMalloc` directly, the way Section G.4 does, needs at least one hand-written Rule of Five at the boundary.

### Worked Example G.3.4 — Copy-and-swap: self-assignment safety and exception safety, together

Worked Solution 8 named the technique real implementations use instead of an explicit `this == &other` check: build the new state fully before releasing the old one. **Copy-and-swap** is that technique made concrete, and it solves a second problem Section G.3's hand-written version never addressed at all — what happens when the *allocation itself* fails partway through an assignment.

Section G.3's copy assignment operator frees the old resource, THEN allocates the new one:

```cpp
RiskPathBuffer& operator=(const RiskPathBuffer& other) {
    if (this == &other) return *this;
    delete[] paths;                          // old resource freed FIRST
    count = other.count;
    paths = new float[count];                // if THIS throws, paths is now DANGLING
    std::memcpy(paths, other.paths, count * sizeof(float));
    return *this;
}
```

If `new` throws partway through — genuinely possible under memory pressure, not a contrived scenario — `paths` is left pointing at memory that was already freed one line earlier, and the object is now broken for the rest of its life. This section injects that exact failure, on purpose, using a one-shot fault-injecting allocator:

```cpp
bool g_should_throw = false;
float* fault_new_float_array(int n) {
    if (g_should_throw) { g_should_throw = false; throw std::bad_alloc(); }
    return new float[n];
}
```

```cpp
UnsafeBuffer a(5);
UnsafeBuffer b(3);
g_should_throw = true;
try {
    a = b;   // throws INSIDE operator=, AFTER a's old buffer was already freed
} catch (const std::bad_alloc&) { /* ... */ }
printf("a.paths[0] = %f\n", a.paths[0]);   // reading from `a` now
```

Genuinely compiled with `g++ -Wall -Wextra -fsanitize=address -g` and run:

```
--- UnsafeBuffer: what happens when allocation fails MID-assignment ---
caught std::bad_alloc from a = b, as expected
attempting to read a.paths[0] now (a's old buffer was already freed
before the throw, and nothing valid replaced it):
```

```
==1593==ERROR: AddressSanitizer: heap-use-after-free on address 0x503000000040 ...
READ of size 4 at 0x503000000040 thread T0
    #0 ... main /tmp/cuda_appendix_g/09_copy_and_swap.cpp:105
freed by thread T0 here:
    #1 ... UnsafeBuffer::operator=(UnsafeBuffer const&) /tmp/cuda_appendix_g/09_copy_and_swap.cpp:40
```

The exception is caught cleanly — but `a` itself is now permanently broken, to the point that simply *reading* `a.paths[0]` afterward is a genuine, ASan-confirmed heap-use-after-free. Catching the exception did not save the object.

Copy-and-swap fixes this by building the entire new state inside a **by-value parameter**, and only exchanging it into `*this` — via a `swap` that cannot itself fail — once construction has fully succeeded:

```cpp
struct SafeBuffer {
    float* paths;
    int count;

    friend void swap(SafeBuffer& a, SafeBuffer& b) noexcept {
        std::swap(a.paths, b.paths);
        std::swap(a.count, b.count);
    }

    SafeBuffer(const SafeBuffer& other) : count(other.count) {
        paths = fault_new_float_array(count);
        std::memcpy(paths, other.paths, count * sizeof(float));
    }
    SafeBuffer(SafeBuffer&& other) noexcept : paths(nullptr), count(0) {
        swap(*this, other);
    }

    // ONE function handles BOTH copy assignment and move assignment: an
    // lvalue argument invokes the copy constructor to build `other`
    // (allocating); an rvalue argument invokes the move constructor
    // (just swapping pointers, no allocation) -- ordinary overload
    // resolution picks the right one.
    SafeBuffer& operator=(SafeBuffer other) {
        swap(*this, other);
        return *this;
    }   // `other` (now holding *this's OLD resource) is destroyed HERE
};
```

The identical injected failure, against `SafeBuffer` instead:

```
=== SafeBuffer (copy-and-swap): SAME injected failure, DIFFERENT outcome ===

a BEFORE the failed assignment: count=5, paths=[0.0 1.0 2.0 3.0 4.0]
caught std::bad_alloc from a = b, as expected
a AFTER the failed assignment : count=5, paths=[0.0 1.0 2.0 3.0 4.0]
```

`a` is completely unchanged — not merely "not crashed," but provably holding the exact same five values it started with. The exception happens while constructing the by-value parameter `other` (inside `SafeBuffer`'s copy constructor), which is *before* `operator=`'s own body — the `swap` — ever runs; `*this` is never touched until `other` is fully built. This is the **strong exception guarantee**: either the operation completes entirely, or it has no visible effect at all.

The same `SafeBuffer` run also confirms copy-and-swap's other property — one function, dispatching to copy or move automatically:

```
--- ONE operator=, dispatching to copy OR move via ordinary overload resolution ---
c = d  (lvalue):        allocations 4 -> 5 (copy-constructing `other` allocated)
e = std::move(f) (rvalue): allocations 7 -> 7 (move-constructing `other` allocated nothing)
```

`c = d` (an lvalue) allocates, because building `other` from `d` calls the copy constructor; `e = std::move(f)` (an rvalue) allocates nothing, because building `other` from `std::move(f)` calls the move constructor instead — the identical `operator=(SafeBuffer other)` handled both, with no separate copy-assignment and move-assignment overloads ever written, and no self-assignment guard either: for `a = a`, `other` would be a freshly-allocated deep copy of `a`'s own data, and `swap` would exchange it in — safe, if not a complete no-op the way Section G.3's guarded version is.

> `[COMMON TRAP]` It's tempting to treat copy-and-swap as a strict upgrade with no cost. It does cost something Section G.3's guarded version avoids for the true self-assignment case: `a = a` under copy-and-swap still performs a full allocation and copy (to build `other`) before swapping, where Section G.3's `if (this == &other) return *this;` makes self-assignment a complete no-op. Copy-and-swap trades that one narrow optimization for a correctness guarantee — self-assignment safety AND exception safety, together, from one mechanism — that generalizes to failures Section G.3's guard was never designed to handle at all.

### Worked Example G.3.5 — Why the move members need `noexcept`

Every move constructor and move assignment operator in this appendix — `RiskPathBuffer`'s, `GPUPathBuffer`'s, `SafeBuffer`'s — is marked `noexcept`. The reason is `std::vector`. When a `std::vector<T>` needs to grow beyond its current capacity, it must relocate every existing element into new, larger storage — and to preserve its own strong exception guarantee (if relocation fails partway, the vector must still be left in its original, valid state), it uses `std::move_if_noexcept` internally: an existing element is moved into the new storage *only if* `T`'s move constructor is `noexcept` (or `T` has no accessible copy constructor at all); otherwise, `std::vector` silently falls back to *copying* every element instead, since a copy leaves the original untouched if it fails, while a throwing move could leave the vector holding neither the old state nor the new one.

```cpp
struct MoveThrowingBuffer {
    int id;
    static int copy_ctor_calls;
    static int move_ctor_calls;
    MoveThrowingBuffer(const MoveThrowingBuffer& other) : id(other.id) { copy_ctor_calls++; }
    MoveThrowingBuffer(MoveThrowingBuffer&& other) : id(other.id) { move_ctor_calls++; }   // NOT noexcept
};

struct MoveNoexceptBuffer {
    int id;
    static int copy_ctor_calls;
    static int move_ctor_calls;
    MoveNoexceptBuffer(const MoveNoexceptBuffer& other) : id(other.id) { copy_ctor_calls++; }
    MoveNoexceptBuffer(MoveNoexceptBuffer&& other) noexcept : id(other.id) { move_ctor_calls++; }   // noexcept
};
```

Genuinely compiled with `g++ -Wall -Wextra` and run — both structs' move constructors are otherwise byte-for-byte identical:

```
--- MoveThrowingBuffer: move ctor exists, but is NOT noexcept ---
after filling to capacity (2): copy_ctor_calls=0, move_ctor_calls=0
after emplace_back(3) forces reallocation:
  copy_ctor_calls=2, move_ctor_calls=0
  (the 2 EXISTING elements were relocated by COPY, not move, specifically
  because the move constructor is not noexcept)

--- MoveNoexceptBuffer: IDENTICAL move constructor, ONLY difference: noexcept ---
after filling to capacity (2): copy_ctor_calls=0, move_ctor_calls=0
after emplace_back(3) forces reallocation:
  copy_ctor_calls=0, move_ctor_calls=2
  (the 2 existing elements were relocated by MOVE this time -- the ONLY
  difference between the two structs is the `noexcept` keyword)
```

`v1.reserve(2)` fixes each vector's initial capacity at exactly 2; the third `emplace_back` genuinely exceeds it, forcing a real reallocation, not a hypothetical one. For `MoveThrowingBuffer`, the two elements already in the vector are relocated by **copying** — `copy_ctor_calls` jumps from `0` to `2`, `move_ctor_calls` stays `0` — purely because the compiler cannot prove the move won't throw. For `MoveNoexceptBuffer`, with a move constructor identical in every way except the `noexcept` keyword, the same reallocation relocates both elements by **moving** instead — `move_ctor_calls` jumps to `2`, `copy_ctor_calls` stays `0`. Nothing else about the two structs differs.

Had `RiskPathBuffer`'s own move constructor (Section G.3.1) been left without `noexcept`, putting `RiskPathBuffer` objects into a `std::vector` and triggering growth would silently deep-copy every existing element on every reallocation instead of cheaply stealing their pointers — no compile error, no warning, just quietly worse performance that compounds with every element the vector ever holds.

> `[COMMON TRAP]` It's tempting to assume a missing `noexcept` on a move constructor is purely a documentation nicety. `std::move_if_noexcept`'s behavior, demonstrated genuinely above, means it's a measurable performance decision the compiler makes FOR you, silently, based on that one keyword — with no diagnostic pointing at the cause if you never went looking for it.

## G.4 The Rule of Five for Device Memory: `GPUPathBuffer` `[FOUNDATIONAL]`

### Intuition

The identical Rule of Five pattern applies when the owned resource is a `cudaMalloc`'d device buffer instead of a `new[]`'d host array — copying a struct like this without deep-copying the device buffer produces the same double free Section G.2 demonstrated, except the second `cudaFree` corrupts device-side allocator state instead of heap allocator state. There is one structural difference from `Point2D`, though, and it isn't a style choice: `cudaMalloc` and `cudaFree` are host-side CUDA Runtime API entry points, not callable from device code, so a struct whose constructor and destructor call them is *necessarily* host-only. Marking any of `GPUPathBuffer`'s special members `__host__ __device__` would not compile — it would be trying to call a host-only function from a context that must also support the device.

### Worked Example G.4.1 — The same pattern, verified by counting `cudaMalloc` calls

With no GPU in this sandbox, every `cudaMalloc` call genuinely executes and honestly returns `cudaErrorNoDevice`, leaving every `device_paths` pointer `nullptr`. Comparing pointer addresses directly, the way Section G.3 compared `paths`, would therefore be *misleading*: every pointer is null, so any two would trivially compare equal — for the wrong reason (nullness, not shared ownership), not the right one. The honest fix used here is a global call counter: does the copy path genuinely issue a *second*, separate `cudaMalloc` call? Does the move path issue *zero* additional calls? Both questions have real, countable answers even though every call fails.

```cpp
int g_cudaMalloc_calls = 0;   // a global counter, purely for this appendix's
                              // own evidence -- real code would never need this.

struct GPUPathBuffer {
    float* device_paths;
    int count;

    GPUPathBuffer(int n) : count(n) {
        g_cudaMalloc_calls++;
        cudaError_t err = cudaMalloc(&device_paths, n * sizeof(float));
        if (err != cudaSuccess) device_paths = nullptr;
    }

    ~GPUPathBuffer() { cudaFree(device_paths); }

    GPUPathBuffer(const GPUPathBuffer& other) : count(other.count) {
        g_cudaMalloc_calls++;
        cudaError_t alloc_err = cudaMalloc(&device_paths, count * sizeof(float));
        if (alloc_err != cudaSuccess) device_paths = nullptr;
        cudaMemcpy(device_paths, other.device_paths, count * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    GPUPathBuffer& operator=(const GPUPathBuffer& other) {
        if (this == &other) return *this;
        cudaFree(device_paths);
        count = other.count;
        g_cudaMalloc_calls++;
        cudaMalloc(&device_paths, count * sizeof(float));
        cudaMemcpy(device_paths, other.device_paths, count * sizeof(float), cudaMemcpyDeviceToDevice);
        return *this;
    }

    // Move: steal the pointer, null the source -- ZERO CUDA API calls
    // needed, since nothing is allocated, freed, or copied; ownership of
    // the SAME allocation just changes which host-side struct owns it.
    GPUPathBuffer(GPUPathBuffer&& other) noexcept : device_paths(other.device_paths), count(other.count) {
        other.device_paths = nullptr;
        other.count = 0;
    }

    GPUPathBuffer& operator=(GPUPathBuffer&& other) noexcept {
        if (this == &other) return *this;
        cudaFree(device_paths);
        device_paths = other.device_paths;
        count = other.count;
        other.device_paths = nullptr;
        other.count = 0;
        return *this;
    }
};
```

Genuinely compiled with `nvcc -arch=sm_80` and run:

```
--- copy construction: does it call cudaMalloc a SECOND time? ---
  [ctor]        count=1000, cudaMalloc call #1: cudaErrorNoDevice
  [copy ctor]   cudaMalloc call #2: cudaErrorNoDevice, cudaMemcpy(D2D): cudaErrorNoDevice
total cudaMalloc calls so far: 2 (one for a, one MORE for b's copy ctor)

--- move construction: does it call cudaMalloc at all? ---
  [ctor]        count=500, cudaMalloc call #3: cudaErrorNoDevice
  [move ctor]   g_cudaMalloc_calls still 3 -- no CUDA API call made
cudaMalloc calls: before c's ctor=2, after c's ctor=3, after moving c into d=3
move added ZERO new cudaMalloc calls: true
c.device_paths after move: (nil) (nullptr -- c's destructor will call cudaFree(nullptr), a documented no-op)

--- move assignment: target's OLD allocation freed via cudaFree, then source's pointer stolen ---
  [ctor]        count=200, cudaMalloc call #4: cudaErrorNoDevice
  [ctor]        count=50, cudaMalloc call #5: cudaErrorNoDevice
  [move assign] g_cudaMalloc_calls still 5 -- no allocation/copy needed
cudaMalloc calls: before move-assign=5, after=5 (move assignment allocates nothing)
f.device_paths after move: (nil) (nullptr -- f's destructor will call cudaFree(nullptr))

--- copy assignment: the ONE member left unexercised above -- does IT call cudaMalloc too? ---
  [ctor]        count=9, cudaMalloc call #6: cudaErrorNoDevice
  [ctor]        count=3, cudaMalloc call #7: cudaErrorNoDevice
  [copy assign] cudaMalloc call #8: cudaErrorNoDevice, cudaMemcpy(D2D): cudaErrorNoDevice
cudaMalloc calls: before copy-assign=7, after=8 (copy assignment allocates exactly ONE new buffer)
g_obj.count is now: 3 (was 9, now matches h_obj's count)

--- self-assignment and self-move: safe with the guard, genuinely exercised ---
  [ctor]        count=42, cudaMalloc call #9: cudaErrorNoDevice
after i_obj=i_obj: device_paths unchanged=true, cudaMalloc calls unchanged=true (9->9)
after i_obj=std::move(i_obj): device_paths unchanged=true, count unchanged=true (42)
```

The call count rises by exactly one for every copy (construction or assignment) and by exactly zero for every move — the honest, sandbox-appropriate evidence that this struct's copy path genuinely allocates a new, independent device buffer while its move path genuinely does not, regardless of whether any individual `cudaMalloc` call succeeds. `cudaFree(nullptr)` is a documented no-op in the CUDA Runtime API, which is exactly why nulling a moved-from `device_paths` is safe, the same role `nullptr` plays for `delete[]` on the host in Section G.3. The copy-assignment run closes the one gap the earlier four demonstrations left open — every one of the five members is now genuinely exercised for `GPUPathBuffer`, not four of five — and the self-assignment/self-move run confirms `GPUPathBuffer`'s own guards behave identically to `RiskPathBuffer`'s in Worked Example G.3.2: unchanged pointer, unchanged call count, unchanged `count`.

**[COMMON TRAP]** Comparing `device_paths` addresses directly here (`a.device_paths == b.device_paths`) would print `true` for *every* pair of objects in this sandbox, because every allocation fails and every pointer is null — a result that looks like "the bug from Section G.2 is still present" when the real explanation is simply "there's no GPU." Evidence has to be chosen with the actual environment in mind, not copy-pasted from a technique that happened to work in a different one.

## G.5 Value-at-Risk and Its Variants `[FOUNDATIONAL]`

### Intuition

Value-at-Risk answers one question: at a given confidence level, how bad could tomorrow's loss get? Different VaR methodologies answer it with different assumptions about the distribution of outcomes — this section computes it two genuinely different ways from the same underlying data and checks that they agree approximately, for a reason that is itself informative rather than a discrepancy to explain away.

### Background

This section reuses Chapter 22.4's `box_muller`/xorshift-PRNG/GBM-update machinery byte-for-byte, at a 1-day horizon (`dt = 1/252`) instead of the full-year horizon Chapter 22.4 used for option pricing. Three methodologies are computed:

**Simulated ("historical") VaR** sorts a large sample of simulated 1-day P&L outcomes ascending and reads off the loss at the 1st percentile for 99% confidence. This appendix has no real external historical price series to draw from, so — honestly — "historical VaR" here is applied to the same GBM-simulated sample "Monte Carlo VaR" would use; in real practice these are two different data sources (an actual historical return series vs. a model-simulated one), but the *extraction algorithm* — sort, take the tail quantile — is the genuine, general-purpose technique either data source would use.

**Parametric (variance-covariance) VaR** assumes P&L is normally distributed and uses a closed form: `VaR = S0 * sigma_1day * z_99`, where `z_99 = 2.326348` is the standard normal 99th-percentile critical value and `sigma_1day = sigma * sqrt(dt)`.

**Conditional VaR (CVaR / Expected Shortfall)** is the average of the P&L values *beyond* the VaR cutoff — not just the one point at the boundary, but the full tail. CVaR is always at least as large as VaR by construction: it averages a set of losses every one of which is at least as large as the VaR loss itself, since that's the criterion for a loss belonging to the tail at all.

### Worked Example G.5.1 — Both methodologies, cross-checked

```cpp
std::vector<float> one_day_prices(NUM_PATHS);
simulate_gbm_paths_host(one_day_prices, S0, mu, sigma, dt_1day, 1, NUM_PATHS, 42ULL);

std::vector<float> pnl(NUM_PATHS);
for (int i = 0; i < NUM_PATHS; i++) pnl[i] = one_day_prices[i] - S0;
std::sort(pnl.begin(), pnl.end());

int var_index = (int)((1.0 - 0.99) * NUM_PATHS);   // 1% tail
float simulated_var_99 = -pnl[var_index];

float sigma_1day = sigma * sqrtf(dt_1day);
float parametric_var_99 = S0 * sigma_1day * 2.326348f;

double sum_tail_losses = 0.0;
for (int i = 0; i < var_index; i++) sum_tail_losses += -pnl[i];
float cvar_99 = (float)(sum_tail_losses / var_index);
```

Genuinely compiled with `g++ -std=c++17 -Wall -Wextra` (zero warnings) and run, with `S0=100.0, mu=0.03, sigma=0.20`, `NUM_PATHS=200000`:

```
--- Simulated ("Monte Carlo"/"Historical") VaR, 1-day horizon, 99% confidence ---
holding: 1 unit of the underlying, S0=100.00, mu=sigma=0.03/0.20, 200000 simulated 1-day scenarios
worst-case index for 99% VaR: floor(0.01 * 200000) = 2000
P&L at that index: -2.8665  ->  simulated 99% 1-day VaR = 2.8665

--- Parametric (Variance-Covariance) VaR, same horizon ---
sigma_1day = sigma * sqrt(dt) = 0.012599, z_99 = 2.326348
parametric VaR = S0 * sigma_1day * z_99 = 2.9309

--- Cross-check: parametric vs. simulated, and why they should be CLOSE, not identical ---
relative difference: 0.0225 (2.25%)
(GBM log-returns are exactly normally distributed by construction, but VaR here is
measured on ARITHMETIC P&L (S_T - S0), which is log-normally, not normally,
distributed -- parametric VaR's normal-P&L assumption is therefore an approximation
even against this appendix's own fully-consistent GBM data, not a bug in either method.)

--- Conditional VaR (CVaR / Expected Shortfall): the average loss BEYOND the VaR cutoff ---
averaging the 2000 worst-case P&L values (indices 0..1999): CVaR_99 = 3.2938
CVaR >= VaR: true (3.2938 >= 2.8665)
(this must always hold: CVaR is the average of a set of losses EVERY one of which is
at least as large as the VaR cutoff itself, by definition of which losses qualify.)
```

The two methodologies land close (2.86 vs. 2.93, a 2.25% relative difference) precisely *because* the underlying GBM data is internally consistent — and they don't land identically because arithmetic P&L (`S_T - S0`) is log-normally, not normally, distributed even when the *log-returns* driving it are exactly normal by construction. Parametric VaR's normality assumption is therefore always an approximation, even measured against data this appendix itself generated to be as favorable to it as possible. CVaR's invariant (`3.2938 ≥ 2.8665`) is checked, not assumed, directly from the same sorted tail the simulated VaR was read from.

**[COMMON TRAP]** It's tempting to read a close match between simulated and parametric VaR as validation that "the model is right." What it actually validates is narrower: that arithmetic P&L over a short, low-volatility, one-day horizon is *close enough* to normal for the parametric shortcut to be a reasonable approximation — a property of the horizon and volatility level, not a general guarantee that would survive a longer horizon or a fatter-tailed underlying process.

## G.6 XVA and Its Variants `[FOUNDATIONAL]`

### Intuition

VaR asks about the risk of holding a position. XVA asks a related but different question: given that the counterparty on the other side of a trade, or the bank itself, might default before the trade matures, what is that default risk worth, in the price of the trade itself? Answering it requires more than a single terminal payoff — it requires the trade's *exposure profile over time*, since what matters is how much would be lost if default happened at each point along the way, not only at maturity.

### Background

An exposure profile requires simulating a path's value at multiple checkpoints, not just its terminal value — this section extends Chapter 22.4's simulation to record the price at four quarterly checkpoints per path, continuing each path's own state from one checkpoint to the next rather than resimulating from `t=0` each time. For a long forward contract with value `V(t) = S(t) - K`:

- **Expected Positive Exposure**, `EE(t) = E[max(V(t), 0)]`, is what the bank stands to lose if the *counterparty* defaults at `t` — averaged over paths where the trade is currently in the bank's favor.
- **Expected Negative Exposure**, `ENE(t) = E[min(V(t), 0)]`, is the mirror: what the bank currently owes, relevant to the bank's *own* default risk.
- **CVA** (Credit Valuation Adjustment) is the expected loss from counterparty default: `CVA = (1-R_C) * Σᵢ EE(tᵢ) * [Q_C(tᵢ₋₁) - Q_C(tᵢ)] * DF(tᵢ)`, where `Q_C(t) = exp(-λ_C·t)` is the counterparty's survival probability under a flat hazard rate `λ_C`, `R_C` its recovery rate, and `DF(t) = exp(-r·t)` the risk-free discount factor.
- **DVA** (Debit Valuation Adjustment) is the mirror using the bank's *own* hazard rate and `ENE`: `DVA = (1-R_B) * Σᵢ |ENE(tᵢ)| * [Q_B(tᵢ₋₁) - Q_B(tᵢ)] * DF(tᵢ)` — an expected *gain* to the bank's own valuation, since the bank's own default would extinguish an obligation it owed.
- **FVA** (Funding Valuation Adjustment) is the cost of funding expected positive exposure at a spread over the risk-free rate: `FVA = Σᵢ s_funding * EE(tᵢ) * DF(tᵢ) * Δtᵢ`.

### Worked Example G.6.1 — A genuine exposure profile, and CVA/DVA/FVA from it

```cpp
void simulate_gbm_paths_checkpointed(std::vector<std::vector<float>>& checkpoint_prices,
                                      const std::vector<float>& checkpoint_times,
                                      float s0, float mu, float sigma,
                                      int steps_per_checkpoint, int num_paths,
                                      unsigned long long seed) {
    int num_checkpoints = (int)checkpoint_times.size();
    for (int idx = 0; idx < num_paths; idx++) {
        float s = s0;
        unsigned long long state = seed + (unsigned long long)idx * 7919ULL + 1ULL;
        float t_prev = 0.0f;
        for (int c = 0; c < num_checkpoints; c++) {
            float t_cur = checkpoint_times[c];
            float dt = (t_cur - t_prev) / steps_per_checkpoint;
            for (int step = 0; step < steps_per_checkpoint; step++) {
                // ... identical xorshift + box_muller + GBM update as Chapter 22.4 ...
                s *= expf((mu - sigma * sigma / 2.0f) * dt + sigma * sqrtf(dt) * z);
            }
            checkpoint_prices[c][idx] = s;   // record the price AT this checkpoint
            t_prev = t_cur;
        }
    }
}
```

Genuinely compiled with `g++ -std=c++17 -Wall -Wextra` (zero warnings) and run, for an at-the-money (`K = S0 = 100`) one-year forward, `r = 3%`, quarterly checkpoints, 200,000 paths:

```
Instrument: a long forward contract, S0=100.00, K=100.00 (at-the-money), T=1.00y, r=3.00%
t        EE(t)        ENE(t)       EE(t)+ENE(t)
0.25     4.3690       -3.6382      0.7308
0.50     6.4255       -4.9706      1.4549
0.75     8.1167       -5.9087      2.2081
1.00     9.6584       -6.6322      3.0262

--- Independent cross-check: EE(T)+ENE(T) must equal E[S(T)]-K, and E[S(T)] must
    match the theoretical GBM real-world mean S0*exp(mu*T) ---
simulated  E[S(T)] = 103.0262
theoretical S0*exp(mu*T) = 103.0454  (relative diff 0.0187%)
EE(T)+ENE(T) = 3.0262  vs.  E[S(T)]-K = 3.0262  (must match: max(v,0)+min(v,0)=v identically)

--- CVA (Credit Valuation Adjustment): expected loss from counterparty default ---
CVA = (1-R_C) * sum_i EE(t_i) * [Q_C(t_i-1)-Q_C(t_i)] * DF(t_i) = 0.0830

--- DVA (Debit Valuation Adjustment): expected GAIN from OUR OWN default ---
DVA = (1-R_B) * sum_i |ENE(t_i)| * [Q_B(t_i-1)-Q_B(t_i)] * DF(t_i) = 0.0309

--- FVA (Funding Valuation Adjustment): cost of funding expected positive exposure ---
FVA = sum_i funding_spread * EE(t_i) * DF(t_i) * dt_i = 0.0350

--- Hand cross-check of the FIRST interval's CVA term (t: 0.00 -> 0.25) ---
Q_C(0)=1.000000, Q_C(0.25)=0.995012, DF(0.25)=0.992528, EE(0.25)=4.3690
term = (1-0.40) * 4.3690 * (1.000000-0.995012) * 0.992528 = 0.012976

--- Net XVA adjustment to the trade's value (bank's perspective) ---
Net = -CVA + DVA - FVA = -0.0830 + 0.0309 - 0.0350 = -0.0870
```

Two independent checks confirm the exposure profile itself before any credit or funding assumption is applied to it. First, `EE(t)+ENE(t)` must equal `E[V(t)]` identically, since `max(v,0)+min(v,0)=v` for any `v` — the table's own arithmetic confirms this at every checkpoint. Second, at the final checkpoint, `E[S(T)]` from the simulation (`103.0262`) is compared against the closed-form theoretical GBM mean `S0·exp(μT) = 103.0454`, a relative difference of `0.0187%` — evidence the checkpointed simulator is behaving like the same GBM process Chapter 22.4 established, not a new, unrelated one. The first interval's CVA contribution is then recomputed by hand from the same `EE(0.25)`, survival probabilities, and discount factor already printed, landing on the identical `0.012976` one of the four terms the full sum embeds.

`DVA` comes out smaller than `CVA` here specifically because the bank's own hazard rate (`1%`) was set lower than the counterparty's (`2%`) — a better-credit bank has a smaller DVA, all else equal, since DVA scales with the bank's *own* probability of defaulting. The net adjustment, `-CVA + DVA - FVA = -0.087`, is a genuine, if small, negative adjustment to this particular trade's value under these particular assumptions.

**Scope note — MVA and KVA.** Margin Valuation Adjustment (the funding cost of posting initial margin under bilateral or cleared margin rules) and Capital Valuation Adjustment (the cost of holding regulatory capital against the trade) are real, widely-quoted XVA variants alongside CVA/DVA/FVA above. Both require inputs this appendix does not model — an initial-margin schedule for MVA, a regulatory capital profile and cost-of-equity rate for KVA — so, honestly: they are named and defined here, not computed, rather than approximated with inputs invented for the occasion.

**Scope note — wrong-way risk.** The CVA formula above treats `EE(t)` and the counterparty's default probability as independent — `EE(tᵢ)` is computed once from the exposure profile, then multiplied by a survival-probability *difference* that never depends on which path produced that exposure. Real counterparties frequently violate this: **wrong-way risk** is the case where a counterparty's own default probability *rises* precisely when the bank's exposure to them is rising too (the standard example: an oil producer as the counterparty on a contract whose value depends on oil prices, where a sustained oil-price move that increases the bank's exposure is also exactly the scenario that stresses the producer's own credit). Modeling this genuinely requires coupling the hazard rate to the simulated path itself — computing a path-dependent `λ_C(t, S)` rather than the single flat `λ_C` used above — which this appendix does not build. The CVA figure computed here is therefore the *independent-exposure* baseline the industry starts from, not a wrong-way-risk-adjusted figure; where wrong-way risk is material, the real CVA is larger than what a flat-hazard model like this one reports.

**[COMMON TRAP]** It's tempting to compute CVA using the *terminal* exposure alone (`EE(T)`) rather than the full profile. That would ignore every default scenario before maturity — a counterparty that defaults at `t=0.25` when `EE(0.25)=4.37` is a genuinely different (and, here, smaller) loss than one that defaults at `T` when `EE(T)=9.66`, which is exactly why CVA is a *sum over the exposure profile*, weighted by the probability of default arriving in each specific interval, rather than a single point evaluated once.

## G.7 Complete Runnable Code

### File: `01_point2d_kernel.cu` (Section G.1)

```cpp
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

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

__global__ void compute_distances_kernel(float* out_from_origin, float* out_between,
                                          const float* xs, const float* ys, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Point2D p(xs[idx], ys[idx]);
        Point2D origin_reference(0.0f, 0.0f);
        out_from_origin[idx] = p.distance_from_origin();
        out_between[idx] = p.distance_to(origin_reference);
    }
}

int main() {
    printf("=== G.1: Point2D -- identical struct, host call and device call ===\n\n");

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
    printf("\n(point1 was NOT affected by mutating point2b -- the compiler-generated\n");
    printf("copy constructor copied x and y by VALUE, and neither member is a pointer\n");
    printf("to anything shared, so this default copy is completely safe. Section G.2\n");
    printf("shows exactly what changes the moment a struct owns a pointer instead.)\n\n");

    printf("--- calling the SAME Point2D methods FROM a kernel ---\n");
    const int N = 5;
    float xs[N] = {3.0f, 0.0f, 1.0f, -2.0f, 5.0f};
    float ys[N] = {4.0f, 0.0f, 1.0f, 2.0f, 12.0f};

    printf("host-side reference, using the exact same Point2D methods:\n");
    for (int i = 0; i < N; i++) {
        Point2D p(xs[i], ys[i]);
        printf("  Point2D(%.1f, %.1f).distance_from_origin() = %f\n", xs[i], ys[i], p.distance_from_origin());
    }

    float *d_xs = nullptr, *d_ys = nullptr, *d_out_origin = nullptr, *d_out_between = nullptr;
    cudaError_t err = cudaMalloc(&d_xs, N * sizeof(float));
    printf("\ncudaMalloc: %s\n", cudaGetErrorName(err));
    compute_distances_kernel<<<1, N>>>(d_out_origin, d_out_between, d_xs, d_ys, N);
    printf("compute_distances_kernel launch (constructs Point2D ON THE DEVICE): %s\n",
           cudaGetErrorName(cudaGetLastError()));
    printf("(no CUDA-capable device in this sandbox -- genuinely attempted, honestly failed.\n");
    printf("What compiled cleanly: Point2D's constructor and both methods running INSIDE a\n");
    printf("__global__ function, constructing objects with `Point2D p(xs[idx], ys[idx]);`\n");
    printf("directly in device code -- ordinary C++ object construction, on the device.)\n");

    return 0;
}
```

### File: `02_double_free_trap.cpp` (Section G.2)

```cpp
#include <cstdio>

struct RiskPathBuffer {
    float* paths;
    int count;

    RiskPathBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = (float)i;
    }
    ~RiskPathBuffer() {
        delete[] paths;
    }
};

int main() {
    RiskPathBuffer a(10);
    printf("a.paths[0] = %f\n", a.paths[0]);
    fflush(stdout);   // ASan will abort this process mid-run; flush now or lose buffered stdout
    {
        RiskPathBuffer b = a;
        printf("b.paths[0] = %f (b.paths and a.paths are the SAME address: %s)\n",
               b.paths[0], (b.paths == a.paths) ? "true" : "false");
        fflush(stdout);
    }
    printf("a.paths[0] = %f (reading through a AFTER b's destructor already freed this memory)\n", a.paths[0]);
    fflush(stdout);
    return 0;
}
```

### File: `03_rule_of_five_correct.cpp` (Section G.3)

```cpp
#include <cstdio>
#include <cstring>
#include <utility>

struct RiskPathBuffer {
    float* paths;
    int count;

    RiskPathBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = (float)i;
        printf("  [ctor]        count=%d, paths=%p\n", count, (void*)paths);
    }

    ~RiskPathBuffer() {
        printf("  [dtor]        count=%d, paths=%p\n", count, (void*)paths);
        delete[] paths;
    }

    RiskPathBuffer(const RiskPathBuffer& other) : count(other.count) {
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
        printf("  [copy ctor]   count=%d, NEW paths=%p (source was %p)\n", count, (void*)paths, (void*)other.paths);
    }

    RiskPathBuffer& operator=(const RiskPathBuffer& other) {
        printf("  [copy assign] target paths=%p, source paths=%p\n", (void*)paths, (void*)other.paths);
        if (this == &other) return *this;
        delete[] paths;
        count = other.count;
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
        return *this;
    }

    RiskPathBuffer(RiskPathBuffer&& other) noexcept : paths(other.paths), count(other.count) {
        printf("  [move ctor]   stole paths=%p from source, nulling source\n", (void*)paths);
        other.paths = nullptr;
        other.count = 0;
    }

    RiskPathBuffer& operator=(RiskPathBuffer&& other) noexcept {
        printf("  [move assign] target paths=%p, stealing source paths=%p\n", (void*)paths, (void*)other.paths);
        if (this == &other) return *this;
        delete[] paths;
        paths = other.paths;
        count = other.count;
        other.paths = nullptr;
        other.count = 0;
        return *this;
    }
};

int main() {
    printf("=== G.3: The Rule of Five, correctly implemented ===\n\n");

    printf("--- copy construction: independent, deep-copied buffers ---\n");
    RiskPathBuffer a(5);
    RiskPathBuffer b = a;
    printf("a.paths == b.paths? %s (deep copy: must be false)\n", (a.paths == b.paths) ? "true" : "false");
    b.paths[0] = 999.0f;
    printf("after b.paths[0]=999: a.paths[0]=%.1f (untouched), b.paths[0]=%.1f\n\n", a.paths[0], b.paths[0]);

    printf("--- move construction: ownership transfers, source is nulled ---\n");
    RiskPathBuffer c(3);
    float* c_original_ptr = c.paths;
    RiskPathBuffer d = std::move(c);
    printf("d.paths == c's ORIGINAL pointer? %s\n", (d.paths == c_original_ptr) ? "true" : "false");
    printf("c.paths is now: %p (nullptr, so c's destructor will free nothing)\n\n", (void*)c.paths);

    printf("--- copy assignment: old resource freed, new one deep-copied ---\n");
    RiskPathBuffer e(2);
    RiskPathBuffer f(7);
    f = e;
    printf("f.count is now: %d (was 7, now matches e's count)\n", f.count);
    printf("f.paths == e.paths? %s (deep copy: must be false)\n\n", (f.paths == e.paths) ? "true" : "false");

    printf("--- move assignment: target's OLD resource freed, then source's pointer stolen ---\n");
    RiskPathBuffer g(4);
    RiskPathBuffer h(1);
    float* h_original_ptr = h.paths;
    g = std::move(h);
    printf("g.paths == h's ORIGINAL pointer? %s\n", (g.paths == h_original_ptr) ? "true" : "false");
    printf("h.paths is now: %p (nullptr, so h's destructor will free nothing)\n\n", (void*)h.paths);

    printf("(destructors for all live objects run automatically below, as main returns)\n");
    return 0;
}
```

### File: `04_gpu_buffer_rule_of_five.cu` (Section G.4)

```cpp
#include <cstdio>
#include <cuda_runtime.h>

int g_cudaMalloc_calls = 0;

struct GPUPathBuffer {
    float* device_paths;
    int count;

    GPUPathBuffer(int n) : count(n) {
        g_cudaMalloc_calls++;
        cudaError_t err = cudaMalloc(&device_paths, n * sizeof(float));
        if (err != cudaSuccess) device_paths = nullptr;
        printf("  [ctor]        count=%d, cudaMalloc call #%d: %s\n", count, g_cudaMalloc_calls, cudaGetErrorName(err));
    }

    ~GPUPathBuffer() {
        cudaError_t err = cudaFree(device_paths);
        printf("  [dtor]        count=%d, cudaFree: %s\n", count, cudaGetErrorName(err));
    }

    GPUPathBuffer(const GPUPathBuffer& other) : count(other.count) {
        g_cudaMalloc_calls++;
        cudaError_t alloc_err = cudaMalloc(&device_paths, count * sizeof(float));
        if (alloc_err != cudaSuccess) device_paths = nullptr;
        cudaError_t copy_err = cudaMemcpy(device_paths, other.device_paths,
                                           count * sizeof(float), cudaMemcpyDeviceToDevice);
        printf("  [copy ctor]   cudaMalloc call #%d: %s, cudaMemcpy(D2D): %s\n",
               g_cudaMalloc_calls, cudaGetErrorName(alloc_err), cudaGetErrorName(copy_err));
    }

    GPUPathBuffer& operator=(const GPUPathBuffer& other) {
        if (this == &other) return *this;
        cudaFree(device_paths);
        count = other.count;
        g_cudaMalloc_calls++;
        cudaError_t alloc_err = cudaMalloc(&device_paths, count * sizeof(float));
        cudaError_t copy_err = cudaMemcpy(device_paths, other.device_paths,
                                           count * sizeof(float), cudaMemcpyDeviceToDevice);
        printf("  [copy assign] cudaMalloc call #%d: %s, cudaMemcpy(D2D): %s\n",
               g_cudaMalloc_calls, cudaGetErrorName(alloc_err), cudaGetErrorName(copy_err));
        return *this;
    }

    GPUPathBuffer(GPUPathBuffer&& other) noexcept : device_paths(other.device_paths), count(other.count) {
        printf("  [move ctor]   g_cudaMalloc_calls still %d -- no CUDA API call made\n", g_cudaMalloc_calls);
        other.device_paths = nullptr;
        other.count = 0;
    }

    GPUPathBuffer& operator=(GPUPathBuffer&& other) noexcept {
        if (this == &other) return *this;
        cudaFree(device_paths);
        device_paths = other.device_paths;
        count = other.count;
        other.device_paths = nullptr;
        other.count = 0;
        printf("  [move assign] g_cudaMalloc_calls still %d -- no allocation/copy needed\n", g_cudaMalloc_calls);
        return *this;
    }
};

int main() {
    printf("=== G.4: The Rule of Five, applied to a cudaMalloc'd device buffer ===\n\n");

    printf("--- copy construction: does it call cudaMalloc a SECOND time? ---\n");
    GPUPathBuffer a(1000);
    GPUPathBuffer b = a;
    printf("total cudaMalloc calls so far: %d (one for a, one MORE for b's copy ctor)\n\n", g_cudaMalloc_calls);

    printf("--- move construction: does it call cudaMalloc at all? ---\n");
    int calls_before_move = g_cudaMalloc_calls;
    GPUPathBuffer c(500);
    int calls_after_c_ctor = g_cudaMalloc_calls;
    GPUPathBuffer d = std::move(c);
    int calls_after_move = g_cudaMalloc_calls;
    printf("cudaMalloc calls: before c's ctor=%d, after c's ctor=%d, after moving c into d=%d\n",
           calls_before_move, calls_after_c_ctor, calls_after_move);
    printf("move added ZERO new cudaMalloc calls: %s\n\n",
           (calls_after_c_ctor == calls_after_move) ? "true" : "false");
    printf("c.device_paths after move: %p (nullptr -- c's destructor will call cudaFree(nullptr), a documented no-op)\n\n",
           (void*)c.device_paths);

    printf("--- move assignment: target's OLD allocation freed via cudaFree, then source's pointer stolen ---\n");
    GPUPathBuffer e(200);
    GPUPathBuffer f(50);
    int calls_before_move_assign = g_cudaMalloc_calls;
    e = std::move(f);
    int calls_after_move_assign = g_cudaMalloc_calls;
    printf("cudaMalloc calls: before move-assign=%d, after=%d (move assignment allocates nothing)\n",
           calls_before_move_assign, calls_after_move_assign);
    printf("f.device_paths after move: %p (nullptr -- f's destructor will call cudaFree(nullptr))\n\n",
           (void*)f.device_paths);

    printf("--- copy assignment: the ONE member left unexercised above -- does IT call cudaMalloc too? ---\n");
    GPUPathBuffer g_obj(9);
    GPUPathBuffer h_obj(3);
    int calls_before_copy_assign = g_cudaMalloc_calls;
    g_obj = h_obj;
    int calls_after_copy_assign = g_cudaMalloc_calls;
    printf("cudaMalloc calls: before copy-assign=%d, after=%d (copy assignment allocates exactly ONE new buffer)\n",
           calls_before_copy_assign, calls_after_copy_assign);
    printf("g_obj.count is now: %d (was 9, now matches h_obj's count)\n\n", g_obj.count);

    printf("--- self-assignment and self-move: safe with the guard, genuinely exercised ---\n");
    GPUPathBuffer i_obj(42);
    float* i_ptr_before = i_obj.device_paths;
    int calls_before_self = g_cudaMalloc_calls;
    i_obj = i_obj;
    printf("after i_obj=i_obj: device_paths unchanged=%s, cudaMalloc calls unchanged=%s (%d->%d)\n",
           (i_obj.device_paths == i_ptr_before) ? "true" : "false",
           (g_cudaMalloc_calls == calls_before_self) ? "true" : "false",
           calls_before_self, g_cudaMalloc_calls);
    i_obj = std::move(i_obj);
    printf("after i_obj=std::move(i_obj): device_paths unchanged=%s, count unchanged=%s (%d)\n\n",
           (i_obj.device_paths == i_ptr_before) ? "true" : "false",
           (i_obj.count == 42) ? "true" : "false", i_obj.count);

    printf("(destructors for all live objects run automatically below, as main returns)\n");
    return 0;
}
```

### File: `07_self_assignment_self_move.cpp` (Worked Example G.3.2)

```cpp
#include <cstdio>
#include <cstring>
#include <utility>

struct RiskPathBuffer {
    float* paths;
    int count;

    RiskPathBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = (float)i;
    }
    ~RiskPathBuffer() { delete[] paths; }

    RiskPathBuffer(const RiskPathBuffer& other) : count(other.count) {
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
    }
    RiskPathBuffer& operator=(const RiskPathBuffer& other) {
        if (this == &other) return *this;
        delete[] paths;
        count = other.count;
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
        return *this;
    }
    RiskPathBuffer(RiskPathBuffer&& other) noexcept : paths(other.paths), count(other.count) {
        other.paths = nullptr;
        other.count = 0;
    }
    RiskPathBuffer& operator=(RiskPathBuffer&& other) noexcept {
        if (this == &other) return *this;
        delete[] paths;
        paths = other.paths;
        count = other.count;
        other.paths = nullptr;
        other.count = 0;
        return *this;
    }
};

// The SAME copy-assignment logic, but WITHOUT the self-assignment guard.
struct UnguardedCopyBuffer {
    float* paths;
    int count;

    UnguardedCopyBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = (float)i;
    }
    ~UnguardedCopyBuffer() { delete[] paths; }

    UnguardedCopyBuffer& operator=(const UnguardedCopyBuffer& other) {
        // NO "if (this == &other) return *this;" guard here.
        delete[] paths;
        count = other.count;
        paths = new float[count];
        std::memcpy(paths, other.paths, count * sizeof(float));
        return *this;
    }
};

int main() {
    printf("=== Self-assignment and self-move: guarded (correct) vs. unguarded (buggy) ===\n\n");

    printf("--- Guarded RiskPathBuffer: a = a (self copy-assignment) ---\n");
    RiskPathBuffer a(5);
    float* a_ptr_before = a.paths;
    a = a;
    printf("a.paths unchanged: %s (%p -> %p)\n", (a.paths == a_ptr_before) ? "true" : "false",
           (void*)a_ptr_before, (void*)a.paths);
    printf("a.count unchanged: %s (%d)\n\n", (a.count == 5) ? "true" : "false", a.count);

    printf("--- Guarded RiskPathBuffer: b = std::move(b) (self move-assignment) ---\n");
    RiskPathBuffer b(7);
    float* b_ptr_before = b.paths;
    b = std::move(b);
    printf("b.paths unchanged: %s (%p -> %p)\n", (b.paths == b_ptr_before) ? "true" : "false",
           (void*)b_ptr_before, (void*)b.paths);
    printf("b.count unchanged: %s (%d)\n\n", (b.count == 7) ? "true" : "false", b.count);

    printf("--- UNGUARDED copy assignment: x = x (no self-assignment guard) ---\n");
    UnguardedCopyBuffer x(4);
    printf("x.paths BEFORE self-assignment: %.1f %.1f %.1f %.1f\n", x.paths[0], x.paths[1], x.paths[2], x.paths[3]);
    x = x;
    printf("x.paths AFTER  self-assignment: %.1f %.1f %.1f %.1f\n", x.paths[0], x.paths[1], x.paths[2], x.paths[3]);
    printf("(NOT a crash, and NOT what was expected going in -- see the appendix text)\n\n");

    return 0;
}
```

### File: `07b_unguarded_self_move.cpp` (Worked Example G.3.2)

```cpp
#include <cstdio>
#include <utility>

struct UnguardedMoveBuffer {
    float* paths;
    int count;

    UnguardedMoveBuffer(int n) : count(n) {
        paths = new float[n];
        for (int i = 0; i < n; i++) paths[i] = (float)i;
    }
    ~UnguardedMoveBuffer() { delete[] paths; }

    UnguardedMoveBuffer& operator=(UnguardedMoveBuffer&& other) noexcept {
        // NO "if (this == &other) return *this;" guard here.
        delete[] paths;
        paths = other.paths;
        count = other.count;
        other.paths = nullptr;
        other.count = 0;
        return *this;
    }
};

int main() {
    printf("=== UNGUARDED move assignment: y = std::move(y) (no self-move guard) ===\n\n");

    UnguardedMoveBuffer y(6);
    printf("before self-move: y.paths=%p, y.count=%d\n", (void*)y.paths, y.count);
    y = std::move(y);
    printf("after  self-move: y.paths=%p, y.count=%d\n\n", (void*)y.paths, y.count);

    printf("no crash, no ASan report -- a silent-emptying failure mode, not a use-after-free.\n");
    return 0;
}
```

### File: `08_rule_of_zero.cpp` (Worked Example G.3.3)

```cpp
#include <cstdio>
#include <vector>
#include <utility>

struct RiskPathBufferV2 {
    std::vector<float> paths;

    RiskPathBufferV2(int n) : paths(n) {
        for (int i = 0; i < n; i++) paths[i] = (float)i;
    }
    // No destructor, no copy constructor, no copy assignment, no move
    // constructor, no move assignment written ANYWHERE in this struct.
};

int main() {
    printf("=== The Rule of Zero: RiskPathBufferV2, zero hand-written special members ===\n\n");

    printf("--- copy construction: still deep, still independent -- for free ---\n");
    RiskPathBufferV2 a(5);
    RiskPathBufferV2 b = a;
    printf("a.paths.data() == b.paths.data()? %s (deep copy: must be false)\n",
           (a.paths.data() == b.paths.data()) ? "true" : "false");
    b.paths[0] = 999.0f;
    printf("after b.paths[0]=999: a.paths[0]=%.1f (untouched), b.paths[0]=%.1f\n\n",
           a.paths[0], b.paths[0]);

    printf("--- move construction: still steals the buffer, still nulls the source -- for free ---\n");
    RiskPathBufferV2 c(3);
    const float* c_original_ptr = c.paths.data();
    RiskPathBufferV2 d = std::move(c);
    printf("d.paths.data() == c's ORIGINAL pointer? %s\n", (d.paths.data() == c_original_ptr) ? "true" : "false");
    printf("c.paths.size() after move: %zu (empty -- std::vector's own move leaves the source valid but empty)\n\n",
           c.paths.size());

    printf("--- self-assignment and self-move: SAFE, with no guard written anywhere ---\n");
    RiskPathBufferV2 e(4);
    const float* e_ptr_before = e.paths.data();
    e = e;
    printf("e.paths.data() unchanged after e=e: %s\n", (e.paths.data() == e_ptr_before) ? "true" : "false");
    printf("e.paths[0..3] after e=e: %.1f %.1f %.1f %.1f\n\n",
           e.paths[0], e.paths[1], e.paths[2], e.paths[3]);

    return 0;
}
```

### File: `09_copy_and_swap.cpp` (Worked Example G.3.4)

```cpp
#include <cstdio>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <new>

bool g_should_throw = false;
float* fault_new_float_array(int n) {
    if (g_should_throw) {
        g_should_throw = false;
        throw std::bad_alloc();
    }
    return new float[n];
}

struct UnsafeBuffer {
    float* paths;
    int count;

    UnsafeBuffer(int n) : count(n) {
        paths = fault_new_float_array(n);
        for (int i = 0; i < n; i++) paths[i] = (float)i;
    }
    ~UnsafeBuffer() { delete[] paths; }

    UnsafeBuffer(const UnsafeBuffer& other) : count(other.count) {
        paths = fault_new_float_array(count);
        std::memcpy(paths, other.paths, count * sizeof(float));
    }

    UnsafeBuffer& operator=(const UnsafeBuffer& other) {
        if (this == &other) return *this;
        delete[] paths;
        count = other.count;
        paths = fault_new_float_array(count);
        std::memcpy(paths, other.paths, count * sizeof(float));
        return *this;
    }
};

int main() {
    printf("=== Copy-and-swap: self-assignment safety AND exception safety, together ===\n\n");

    printf("--- UnsafeBuffer: what happens when allocation fails MID-assignment ---\n");
    UnsafeBuffer a(5);
    UnsafeBuffer b(3);
    g_should_throw = true;
    try {
        a = b;
        printf("(unreached -- the assignment should have thrown)\n");
    } catch (const std::bad_alloc&) {
        printf("caught std::bad_alloc from a = b, as expected\n");
    }
    printf("attempting to read a.paths[0] now (a's old buffer was already freed\n");
    printf("before the throw, and nothing valid replaced it):\n");
    fflush(stdout);
    printf("a.paths[0] = %f\n", a.paths[0]);
    printf("(unreached if ASan caught the read above)\n");

    return 0;
}
```

### File: `09b_copy_and_swap_safe.cpp` (Worked Example G.3.4)

```cpp
#include <cstdio>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <new>

bool g_should_throw = false;
int g_alloc_count = 0;
float* fault_new_float_array(int n) {
    if (g_should_throw) {
        g_should_throw = false;
        throw std::bad_alloc();
    }
    g_alloc_count++;
    return new float[n];
}

struct SafeBuffer {
    float* paths;
    int count;

    friend void swap(SafeBuffer& a, SafeBuffer& b) noexcept {
        std::swap(a.paths, b.paths);
        std::swap(a.count, b.count);
    }

    SafeBuffer(int n) : count(n) {
        paths = fault_new_float_array(n);
        for (int i = 0; i < n; i++) paths[i] = (float)i;
    }
    ~SafeBuffer() { delete[] paths; }

    SafeBuffer(const SafeBuffer& other) : count(other.count) {
        paths = fault_new_float_array(count);
        std::memcpy(paths, other.paths, count * sizeof(float));
    }
    SafeBuffer(SafeBuffer&& other) noexcept : paths(nullptr), count(0) {
        swap(*this, other);
    }

    SafeBuffer& operator=(SafeBuffer other) {
        swap(*this, other);
        return *this;
    }
};

static void print_buffer(const char* label, const SafeBuffer& buf) {
    printf("%s: count=%d, paths=[", label, buf.count);
    for (int i = 0; i < buf.count; i++) printf("%s%.1f", (i ? " " : ""), buf.paths[i]);
    printf("]\n");
}

int main() {
    printf("=== SafeBuffer (copy-and-swap): SAME injected failure, DIFFERENT outcome ===\n\n");

    SafeBuffer a(5);
    SafeBuffer b(3);
    print_buffer("a BEFORE the failed assignment", a);

    g_should_throw = true;
    try {
        a = b;
        printf("(unreached -- the assignment should have thrown)\n");
    } catch (const std::bad_alloc&) {
        printf("caught std::bad_alloc from a = b, as expected\n");
    }

    print_buffer("a AFTER the failed assignment ", a);

    printf("\n--- ONE operator=, dispatching to copy OR move via ordinary overload resolution ---\n");
    SafeBuffer c(4);
    SafeBuffer d(2);
    int calls_before_copy = g_alloc_count;
    c = d;
    int calls_after_copy = g_alloc_count;
    printf("c = d  (lvalue):        allocations %d -> %d (copy-constructing `other` allocated)\n",
           calls_before_copy, calls_after_copy);

    SafeBuffer e(6);
    SafeBuffer f(9);
    int calls_before_move = g_alloc_count;
    e = std::move(f);
    int calls_after_move = g_alloc_count;
    printf("e = std::move(f) (rvalue): allocations %d -> %d (move-constructing `other` allocated nothing)\n",
           calls_before_move, calls_after_move);

    return 0;
}
```

### File: `10_noexcept_vector_reallocation.cpp` (Worked Example G.3.5)

```cpp
#include <cstdio>
#include <vector>

struct MoveThrowingBuffer {
    int id;
    static int copy_ctor_calls;
    static int move_ctor_calls;

    explicit MoveThrowingBuffer(int i) : id(i) {}
    MoveThrowingBuffer(const MoveThrowingBuffer& other) : id(other.id) { copy_ctor_calls++; }
    MoveThrowingBuffer(MoveThrowingBuffer&& other) : id(other.id) { move_ctor_calls++; }
};
int MoveThrowingBuffer::copy_ctor_calls = 0;
int MoveThrowingBuffer::move_ctor_calls = 0;

struct MoveNoexceptBuffer {
    int id;
    static int copy_ctor_calls;
    static int move_ctor_calls;

    explicit MoveNoexceptBuffer(int i) : id(i) {}
    MoveNoexceptBuffer(const MoveNoexceptBuffer& other) : id(other.id) { copy_ctor_calls++; }
    MoveNoexceptBuffer(MoveNoexceptBuffer&& other) noexcept : id(other.id) { move_ctor_calls++; }
};
int MoveNoexceptBuffer::copy_ctor_calls = 0;
int MoveNoexceptBuffer::move_ctor_calls = 0;

int main() {
    printf("=== Why the Rule of Five's move members need `noexcept` ===\n\n");

    printf("--- MoveThrowingBuffer: move ctor exists, but is NOT noexcept ---\n");
    std::vector<MoveThrowingBuffer> v1;
    v1.reserve(2);
    v1.emplace_back(1);
    v1.emplace_back(2);
    printf("after filling to capacity (2): copy_ctor_calls=%d, move_ctor_calls=%d\n",
           MoveThrowingBuffer::copy_ctor_calls, MoveThrowingBuffer::move_ctor_calls);
    v1.emplace_back(3);
    printf("after emplace_back(3) forces reallocation:\n");
    printf("  copy_ctor_calls=%d, move_ctor_calls=%d\n",
           MoveThrowingBuffer::copy_ctor_calls, MoveThrowingBuffer::move_ctor_calls);

    printf("\n--- MoveNoexceptBuffer: IDENTICAL move constructor, ONLY difference: noexcept ---\n");
    std::vector<MoveNoexceptBuffer> v2;
    v2.reserve(2);
    v2.emplace_back(1);
    v2.emplace_back(2);
    printf("after filling to capacity (2): copy_ctor_calls=%d, move_ctor_calls=%d\n",
           MoveNoexceptBuffer::copy_ctor_calls, MoveNoexceptBuffer::move_ctor_calls);
    v2.emplace_back(3);
    printf("after emplace_back(3) forces reallocation:\n");
    printf("  copy_ctor_calls=%d, move_ctor_calls=%d\n",
           MoveNoexceptBuffer::copy_ctor_calls, MoveNoexceptBuffer::move_ctor_calls);

    return 0;
}
```

### File: `05_var_engine.cpp` (Section G.5)

```cpp
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

float box_muller_host(float u1, float u2) {
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

void simulate_gbm_paths_host(std::vector<float>& paths, float s0, float mu, float sigma, float dt,
                              int num_steps, int num_paths, unsigned long long seed) {
    for (int idx = 0; idx < num_paths; idx++) {
        float s = s0;
        unsigned long long state = seed + (unsigned long long)idx * 7919ULL + 1ULL;
        for (int step = 0; step < num_steps; step++) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            float u1 = (float)((state >> 11) & 0xFFFFFF) / (float)0x1000000 + 1e-7f;
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            float u2 = (float)((state >> 11) & 0xFFFFFF) / (float)0x1000000;
            float z = box_muller_host(u1, u2);
            s *= expf((mu - sigma * sigma / 2.0f) * dt + sigma * sqrtf(dt) * z);
        }
        paths[idx] = s;
    }
}

int main() {
    printf("=== G.5: Value-at-Risk and Its Variants ===\n\n");

    const int NUM_PATHS = 200000;
    float S0 = 100.0f, mu = 0.03f, sigma = 0.20f;
    float horizon_days = 1.0f;
    float dt_1day = horizon_days / 252.0f;

    printf("--- Simulated (\"Monte Carlo\"/\"Historical\") VaR, 1-day horizon, 99%% confidence ---\n");
    std::vector<float> one_day_prices(NUM_PATHS);
    simulate_gbm_paths_host(one_day_prices, S0, mu, sigma, dt_1day, 1, NUM_PATHS, 42ULL);

    std::vector<float> pnl(NUM_PATHS);
    for (int i = 0; i < NUM_PATHS; i++) pnl[i] = one_day_prices[i] - S0;
    std::sort(pnl.begin(), pnl.end());

    double confidence = 0.99;
    int var_index = (int)((1.0 - confidence) * NUM_PATHS);
    float simulated_var_99 = -pnl[var_index];

    printf("holding: 1 unit of the underlying, S0=%.2f, mu=sigma=%.2f/%.2f, %d simulated 1-day scenarios\n",
           S0, mu, sigma, NUM_PATHS);
    printf("worst-case index for 99%% VaR: floor(0.01 * %d) = %d\n", NUM_PATHS, var_index);
    printf("P&L at that index: %.4f  ->  simulated 99%% 1-day VaR = %.4f\n\n", pnl[var_index], simulated_var_99);

    printf("--- Parametric (Variance-Covariance) VaR, same horizon ---\n");
    float sigma_1day = sigma * sqrtf(dt_1day);
    float z_99 = 2.326348f;
    float parametric_var_99 = S0 * sigma_1day * z_99;
    printf("sigma_1day = sigma * sqrt(dt) = %.6f, z_99 = %.6f\n", sigma_1day, z_99);
    printf("parametric VaR = S0 * sigma_1day * z_99 = %.4f\n\n", parametric_var_99);

    printf("--- Cross-check: parametric vs. simulated, and why they should be CLOSE, not identical ---\n");
    float relative_diff = fabsf(parametric_var_99 - simulated_var_99) / simulated_var_99;
    printf("relative difference: %.4f (%.2f%%)\n", relative_diff, relative_diff * 100.0f);
    printf("(GBM log-returns are exactly normally distributed by construction, but VaR here is\n");
    printf("measured on ARITHMETIC P&L (S_T - S0), which is log-normally, not normally,\n");
    printf("distributed -- parametric VaR's normal-P&L assumption is therefore an approximation\n");
    printf("even against this appendix's own fully-consistent GBM data, not a bug in either method.)\n\n");

    printf("--- Conditional VaR (CVaR / Expected Shortfall): the average loss BEYOND the VaR cutoff ---\n");
    double sum_tail_losses = 0.0;
    for (int i = 0; i < var_index; i++) sum_tail_losses += -pnl[i];
    float cvar_99 = (float)(sum_tail_losses / var_index);
    printf("averaging the %d worst-case P&L values (indices 0..%d): CVaR_99 = %.4f\n", var_index, var_index - 1, cvar_99);
    printf("CVaR >= VaR: %s (%.4f >= %.4f)\n", (cvar_99 >= simulated_var_99) ? "true" : "false", cvar_99, simulated_var_99);
    printf("(this must always hold: CVaR is the average of a set of losses EVERY one of which is\n");
    printf("at least as large as the VaR cutoff itself, by definition of which losses qualify.)\n");

    return 0;
}
```

### File: `06_xva_engine.cpp` (Section G.6)

```cpp
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

float box_muller_host(float u1, float u2) {
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

void simulate_gbm_paths_checkpointed(std::vector<std::vector<float>>& checkpoint_prices,
                                      const std::vector<float>& checkpoint_times,
                                      float s0, float mu, float sigma,
                                      int steps_per_checkpoint, int num_paths,
                                      unsigned long long seed) {
    int num_checkpoints = (int)checkpoint_times.size();
    for (int idx = 0; idx < num_paths; idx++) {
        float s = s0;
        unsigned long long state = seed + (unsigned long long)idx * 7919ULL + 1ULL;
        float t_prev = 0.0f;
        for (int c = 0; c < num_checkpoints; c++) {
            float t_cur = checkpoint_times[c];
            float dt = (t_cur - t_prev) / steps_per_checkpoint;
            for (int step = 0; step < steps_per_checkpoint; step++) {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                float u1 = (float)((state >> 11) & 0xFFFFFF) / (float)0x1000000 + 1e-7f;
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                float u2 = (float)((state >> 11) & 0xFFFFFF) / (float)0x1000000;
                float z = box_muller_host(u1, u2);
                s *= expf((mu - sigma * sigma / 2.0f) * dt + sigma * sqrtf(dt) * z);
            }
            checkpoint_prices[c][idx] = s;
            t_prev = t_cur;
        }
    }
}

int main() {
    printf("=== G.6: XVA and Its Variants ===\n\n");

    const int NUM_PATHS = 200000;
    const int STEPS_PER_CHECKPOINT = 13;
    float S0 = 100.0f, mu = 0.03f, sigma = 0.20f, r = 0.03f;
    float K = 100.0f;

    std::vector<float> checkpoint_times = {0.25f, 0.5f, 0.75f, 1.0f};
    int NC = (int)checkpoint_times.size();

    std::vector<std::vector<float>> checkpoint_prices(NC, std::vector<float>(NUM_PATHS));
    simulate_gbm_paths_checkpointed(checkpoint_prices, checkpoint_times, S0, mu, sigma,
                                     STEPS_PER_CHECKPOINT, NUM_PATHS, 42ULL);

    printf("Instrument: a long forward contract, S0=%.2f, K=%.2f (at-the-money), T=%.2fy, r=%.2f%%\n",
           S0, K, checkpoint_times[NC - 1], r * 100.0f);
    printf("V(t) = S(t) - K.  Exposure profile from %d simulated paths, quarterly checkpoints:\n\n",
           NUM_PATHS);

    std::vector<float> EE(NC), ENE(NC);
    printf("%-8s %-12s %-12s %-14s\n", "t", "EE(t)", "ENE(t)", "EE(t)+ENE(t)");
    for (int c = 0; c < NC; c++) {
        double sum_pos = 0.0, sum_neg = 0.0;
        for (int i = 0; i < NUM_PATHS; i++) {
            float v = checkpoint_prices[c][i] - K;
            if (v > 0.0f) sum_pos += v; else sum_neg += v;
        }
        EE[c] = (float)(sum_pos / NUM_PATHS);
        ENE[c] = (float)(sum_neg / NUM_PATHS);
        printf("%-8.2f %-12.4f %-12.4f %-14.4f\n", checkpoint_times[c], EE[c], ENE[c], EE[c] + ENE[c]);
    }

    printf("\n--- Independent cross-check: EE(T)+ENE(T) must equal E[S(T)]-K, and E[S(T)] must\n");
    printf("    match the theoretical GBM real-world mean S0*exp(mu*T) ---\n");
    double sum_terminal = 0.0;
    for (int i = 0; i < NUM_PATHS; i++) sum_terminal += checkpoint_prices[NC - 1][i];
    float mean_terminal_simulated = (float)(sum_terminal / NUM_PATHS);
    float mean_terminal_theoretical = S0 * expf(mu * checkpoint_times[NC - 1]);
    printf("simulated  E[S(T)] = %.4f\n", mean_terminal_simulated);
    printf("theoretical S0*exp(mu*T) = %.4f  (relative diff %.4f%%)\n",
           mean_terminal_theoretical,
           100.0f * fabsf(mean_terminal_simulated - mean_terminal_theoretical) / mean_terminal_theoretical);
    printf("EE(T)+ENE(T) = %.4f  vs.  E[S(T)]-K = %.4f  (must match: max(v,0)+min(v,0)=v identically)\n\n",
           EE[NC - 1] + ENE[NC - 1], mean_terminal_simulated - K);

    float hazard_counterparty = 0.02f;
    float recovery_counterparty = 0.40f;
    float hazard_own = 0.01f;
    float recovery_own = 0.40f;
    float funding_spread = 0.005f;

    auto survival = [](float hazard, float t) { return expf(-hazard * t); };
    auto discount = [&](float t) { return expf(-r * t); };

    printf("--- Credit/funding assumptions ---\n");
    printf("counterparty: flat hazard=%.4f, recovery=%.2f\n", hazard_counterparty, recovery_counterparty);
    printf("own (bank):   flat hazard=%.4f, recovery=%.2f\n", hazard_own, recovery_own);
    printf("funding spread over r: %.4f\n\n", funding_spread);

    double cva = 0.0, dva = 0.0, fva = 0.0;
    float t_prev = 0.0f;
    for (int c = 0; c < NC; c++) {
        float t_cur = checkpoint_times[c];
        float q_c_prev = survival(hazard_counterparty, t_prev);
        float q_c_cur = survival(hazard_counterparty, t_cur);
        float q_b_prev = survival(hazard_own, t_prev);
        float q_b_cur = survival(hazard_own, t_cur);
        float df = discount(t_cur);
        float dt = t_cur - t_prev;

        cva += (1.0 - recovery_counterparty) * EE[c] * (q_c_prev - q_c_cur) * df;
        dva += (1.0 - recovery_own) * (-ENE[c]) * (q_b_prev - q_b_cur) * df;
        fva += funding_spread * EE[c] * df * dt;

        t_prev = t_cur;
    }

    printf("--- CVA (Credit Valuation Adjustment): expected loss from counterparty default ---\n");
    printf("CVA = (1-R_C) * sum_i EE(t_i) * [Q_C(t_i-1)-Q_C(t_i)] * DF(t_i) = %.4f\n\n", cva);

    printf("--- DVA (Debit Valuation Adjustment): expected GAIN from OUR OWN default ---\n");
    printf("DVA = (1-R_B) * sum_i |ENE(t_i)| * [Q_B(t_i-1)-Q_B(t_i)] * DF(t_i) = %.4f\n\n", dva);

    printf("--- FVA (Funding Valuation Adjustment): cost of funding expected positive exposure ---\n");
    printf("FVA = sum_i funding_spread * EE(t_i) * DF(t_i) * dt_i = %.4f\n\n", fva);

    printf("--- Hand cross-check of the FIRST interval's CVA term (t: 0.00 -> 0.25) ---\n");
    {
        float q_c_prev = survival(hazard_counterparty, 0.0f);
        float q_c_cur = survival(hazard_counterparty, 0.25f);
        float df = discount(0.25f);
        double term1 = (1.0 - recovery_counterparty) * EE[0] * (q_c_prev - q_c_cur) * df;
        printf("Q_C(0)=%.6f, Q_C(0.25)=%.6f, DF(0.25)=%.6f, EE(0.25)=%.4f\n",
               q_c_prev, q_c_cur, df, EE[0]);
        printf("term = (1-0.40) * %.4f * (%.6f-%.6f) * %.6f = %.6f\n",
               EE[0], q_c_prev, q_c_cur, df, term1);
    }

    printf("\n--- Net XVA adjustment to the trade's value (bank's perspective) ---\n");
    double net_xva = -cva + dva - fva;
    printf("Net = -CVA + DVA - FVA = -%.4f + %.4f - %.4f = %.4f\n\n", cva, dva, fva, net_xva);

    printf("--- Scope note: MVA and KVA ---\n");
    printf("Margin Valuation Adjustment (MVA) and Capital Valuation Adjustment (KVA) are real,\n");
    printf("widely-quoted XVA variants alongside CVA/DVA/FVA above, requiring an initial-margin\n");
    printf("schedule and a regulatory-capital profile respectively -- inputs this appendix does\n");
    printf("not model, so, honestly: they are named and defined here, not computed.\n");

    return 0;
}
```

### Expected Output

Each file's genuine output is embedded inline in its own section above (G.1–G.6). All twelve files compile cleanly — `01_point2d_kernel.cu` and `04_gpu_buffer_rule_of_five.cu` with `nvcc -arch=sm_80`, the other ten with `g++ -std=c++17 -Wall -Wextra` (`02`, `03`, `07`, `07b`, `09`, and `09b` additionally with `-fsanitize=address -g`) — with zero warnings across all twelve, except `07_self_assignment_self_move.cpp` and `07b_unguarded_self_move.cpp` each producing their own single, expected `-Wself-move` warning on their deliberate `std::move`-onto-itself lines, discussed inline in Worked Example G.3.2 rather than suppressed.

## Chapter Summary

`Point2D`'s compiler-generated copy constructor is safe for exactly one reason: it owns nothing beyond the two `float`s it directly contains, so copying those two numbers twice produces two fully independent, correct objects — demonstrated in Section G.1 both on the host and by constructing `Point2D` objects directly inside a `__global__` kernel using its existing `__host__ __device__` methods, unchanged. The moment a struct owns a resource instead — Section G.2's `RiskPathBuffer`, wrapping a `new[]`'d array — the identical compiler-generated copy stops being safe, copying a pointer instead of the memory it addresses, and genuinely triggers a `heap-use-after-free` under AddressSanitizer the instant the first copy's destructor runs. The Rule of Five fixes this by making all five special members explicit: Section G.3 implements destructor, copy constructor, copy assignment, move constructor, and move assignment for the same struct, each one genuinely exercised and independently checked — deep-copy independence, a stolen-and-nulled moved-from pointer, a freed old resource before a new one is taken on. Section G.4 applies the identical pattern to a `cudaMalloc`'d device buffer, with one structural difference that is a direct consequence of what the struct owns rather than a stylistic choice: `cudaMalloc`/`cudaFree` are host-only Runtime API entry points, so `GPUPathBuffer`'s special members cannot be `__host__ __device__` — and, with no GPU in this sandbox to produce real distinct pointer addresses, verification shifts honestly from address comparison to counting `cudaMalloc` calls, evidence that survives every allocation failing, with every one of the five members — including copy assignment, and including self-assignment/self-move — now genuinely exercised. Section G.3's own self-assignment and self-move guards get the same scrutiny in Worked Example G.3.2: removing them doesn't reproduce Section G.2's crash, but produces two different, quieter failures instead — silent corruption from copying a brand-new uninitialized allocation onto itself, and silent data loss from one assignment's own two statements overwriting each other — neither one anything AddressSanitizer's heap checks would catch. Worked Example G.3.3 then shows the Rule of Five is not the only way out: wrapping the owned resource in `std::vector` instead of a raw pointer restores `Point2D`'s original situation — zero hand-written special members, all five (and the two self-assignment guards) correct by construction — the Rule of Zero this book's own `Point2D` example embodied from the very first section without naming it. Worked Example G.3.4 closes a second gap Section G.3's hand-written version left open: a genuinely injected allocation failure leaves the free-then-allocate version holding a pointer to already-freed memory (an ASan-confirmed heap-use-after-free just from reading it back afterward), while the copy-and-swap alternative — build the new state fully inside a by-value parameter, then swap it in via an operation that cannot itself fail — comes back from the identical failure completely unchanged, and, as a bonus, collapses copy assignment and move assignment into one function that dispatches between them via ordinary overload resolution. Worked Example G.3.5 explains the one keyword every move member in this appendix carries: `noexcept`, demonstrated by putting two otherwise byte-identical structs into a `std::vector` and forcing reallocation — the non-`noexcept` one is silently relocated by copying every element, the `noexcept` one by moving them, a real, measurable, and otherwise undiagnosed performance difference traced to `std::move_if_noexcept`. The same discipline extends into quantitative finance: Section G.5 computes Value-at-Risk two genuinely different ways — a simulated/historical quantile extraction and a closed-form parametric estimate — from the same GBM-simulated data Chapter 22.4 established, landing close but not identical for a specific, checked reason (arithmetic P&L on a log-normal terminal price is only approximately normal), with Conditional VaR's `CVaR ≥ VaR` invariant verified rather than assumed. Section G.6 extends that same machinery to record an exposure profile across time rather than only at maturity, computing CVA, DVA, and FVA from genuine `EE(t)`/`ENE(t)` values cross-checked against the GBM process's own closed-form mean, with honest scope notes that MVA and KVA are named but not computed, since this appendix has no initial-margin schedule or capital profile to model them from, and that the CVA figure itself assumes exposure and counterparty default are independent — real wrong-way risk, where the two are correlated, would make the true CVA larger than what this flat-hazard model reports.

## Self-Check Questions

1. Section G.1 gives `Point2D`'s constructor and both methods `__host__ __device__`, but Section G.4 explicitly does NOT give `GPUPathBuffer`'s special members that annotation. Using Section G.4's own reasoning about what `cudaMalloc`/`cudaFree` are, explain why adding `__host__ __device__` to `GPUPathBuffer`'s constructor would be a genuine compile error, not just an unnecessary annotation.
2. Section G.2's bug requires TWO things to go wrong together: a memberwise pointer copy, AND a scope where the copy's destructor runs before the original's. Suppose `RiskPathBuffer b = a;` in Section G.2's code were changed to a reference, `RiskPathBuffer& b = a;`, instead. Would the double-free bug still occur? Why or why not?
3. Section G.4 verifies the Rule of Five for `GPUPathBuffer` using a call counter instead of comparing `device_paths` addresses directly, because every address is null in this sandbox. If this exact code were run on a machine WITH a working GPU, would comparing `a.device_paths == b.device_paths` directly (after `GPUPathBuffer b = a;`) now become a valid, sufficient way to verify the copy constructor is correct — or would it still be missing something the call-counter approach catches?
4. Section G.5 reports simulated VaR (2.8665) and parametric VaR (2.9309) as "close but not identical," attributing the gap to arithmetic P&L being log-normal rather than normal. If the same VaR calculation were instead applied to LOG-returns (`ln(S_T/S0)`) rather than arithmetic P&L (`S_T - S0`), would you expect the simulated-vs-parametric gap to shrink, grow, or stay about the same? Justify your answer using what GBM assumes is normally distributed by construction.
5. Section G.6 computes `DVA < CVA` here specifically because the bank's own hazard rate (1%) was set lower than the counterparty's (2%). Using the DVA formula's own structure (`DVA = (1-R_B) * Σᵢ |ENE(tᵢ)| * [Q_B(tᵢ₋₁)-Q_B(tᵢ)] * DF(tᵢ)`), explain what would happen to DVA, holding everything else fixed, if the bank's OWN credit quality got WORSE (a higher hazard rate) — and why a bank's DVA increasing when its own credit gets worse has historically been considered a controversial property of DVA accounting.
6. Section G.6's exposure profile shows `EE(t)` growing from 4.3690 at `t=0.25` to 9.6584 at `t=1.00`, roughly proportional to `sqrt(t)` rather than growing linearly with `t`. Using what you know about GBM's volatility term (`sigma * sqrt(dt)` in the update step), explain why an at-the-money forward's expected positive exposure would be expected to grow with the SQUARE ROOT of time rather than linearly.
7. Worked Example G.3.2's `UnguardedCopyBuffer` was expected, going in, to reproduce Section G.2's heap-use-after-free — but genuinely didn't. Explain, using the exact order of statements in its `operator=`, why `other.paths` had already stopped pointing at the freed memory by the time `memcpy` read it — and what SPECIFIC single-line reordering (moving one statement earlier or later, without adding a guard) would make it into a genuine use-after-free instead.
8. Worked Example G.3.3's `RiskPathBufferV2` gets correct self-assignment behavior (`e = e` leaves `e` unchanged) without writing any guard at all, while Section G.3's `RiskPathBuffer` needs an explicit `if (this == &other) return *this;` to get the same guarantee. Both structs ultimately bottom out in a `new[]`/`delete[]`-based allocation somewhere. Where does `std::vector`'s OWN internal implementation actually solve this problem — does it avoid the issue entirely, or does it just contain the same kind of guard Section G.3 wrote by hand, hidden one layer down?
9. Worked Example G.3.4's `SafeBuffer::operator=(SafeBuffer other)` takes its parameter BY VALUE rather than by `const&` (the signature Section G.3's `RiskPathBuffer::operator=` uses). Explain why passing by value is not an incidental style choice here but the specific mechanism that makes copy-and-swap dispatch to either the copy constructor or the move constructor automatically — what would break about that dispatch if the parameter were changed to `const SafeBuffer&`?
10. Worked Example G.3.5 shows `MoveThrowingBuffer` being copied instead of moved during `std::vector` reallocation, purely because its move constructor lacks `noexcept`. Both `MoveThrowingBuffer` and `MoveNoexceptBuffer` have accessible copy constructors in this appendix's example. The text mentions `std::move_if_noexcept` also allows a move when a type has "no accessible copy constructor at all," even without `noexcept`. Explain why that specific exception is safe for `std::vector`'s strong exception guarantee, using the same reasoning Worked Example G.3.4 used to explain why a THROWING move would be unsafe there.

## Where We Go Next

This appendix closes a loop this book opened with `Point2D` in its main chapters: a struct simple enough that the compiler's default special members are already correct, contrasted here against the point where that stops being true and the Rule of Five becomes load-bearing rather than optional. The verification techniques compound across sections — AddressSanitizer, first used in Appendix D.9 for a dangling reference, catches Section G.2's double free with the same deterministic honesty; the call-count technique Section G.4 introduces for a no-GPU sandbox is a direct descendant of this book's running principle, since Chapter 18, of showing a real `cudaErrorNoDevice` rather than skipping what a missing device can't run. The financial sections extend Chapter 22.4's GBM machinery rather than replacing it, so a change to that chapter's PRNG or GBM update would propagate its effect identically into every VaR and XVA number here — a property worth remembering if this book's own simulation parameters are ever revisited.

## Worked Solutions

**1.** `cudaMalloc` and `cudaFree` are host-side CUDA Runtime API functions — they have no device-callable implementation, because allocating and freeing device memory is a driver-level operation issued from the host, not something a single GPU thread does for itself. `__host__ __device__` on a function tells `nvcc` to compile TWO versions of it, one of which must be valid device code — and a device-code version of `GPUPathBuffer`'s constructor would need to call `cudaMalloc` from inside a kernel, which is simply not a legal call `nvcc` can compile, producing a genuine compile error rather than a warning or a runtime failure. This is exactly the reasoning Section G.4 gives for why `GPUPathBuffer`'s host-only-ness is a consequence of what it owns, not a stylistic choice `Point2D` happened to make differently.

**2.** No — the double-free would NOT occur, but for a reason worth being precise about: `RiskPathBuffer& b = a;` doesn't call the copy constructor at all. A reference is not a new object; `b` becomes simply another name for the exact same `a`, with no second `RiskPathBuffer` ever constructed and therefore no second destructor ever running. When `b`'s "scope" ends, nothing is destroyed, because `b` was never an independent object to destroy — only `a` is, once, when `main` actually returns. This is a different escape from the bug than the Rule of Five provides: it avoids the double-free by avoiding a copy entirely, not by making the copy safe, and it would not help in the (far more common) case where a genuine independent copy is actually needed, e.g. to store two buffers that must later diverge.

**3.** It would still be missing something. Even with a working GPU giving `a.device_paths` and `b.device_paths` two genuinely different, non-null addresses, the address comparison alone only proves the copy constructor allocated a *different* buffer — it says nothing about whether `cudaMemcpy(..., cudaMemcpyDeviceToDevice)` actually succeeded in copying the *contents* correctly, and it says nothing about the move path (a real GPU wouldn't change the reasoning there at all: move should still add zero allocations, which address comparison alone can't detect since a move constructor's whole point is that the pointer VALUE is identical between source and destination). The call-counter approach and address comparison check different, complementary things — a real GPU makes address comparison meaningful for the copy-independence question, but the call-count evidence for "did the move path skip allocation entirely" remains necessary regardless of whether a GPU is present.

**4.** The gap would shrink, likely substantially. GBM's defining assumption is that LOG-returns, `ln(S_T/S0) = (μ - σ²/2)T + σ√T·Z` for standard normal `Z`, are exactly normally distributed by construction — that's the entire mechanism the `box_muller`-driven update step implements. Arithmetic P&L, `S_T - S0`, is a nonlinear (exponential) transformation of that normal log-return, which is only approximately normal itself, and that approximation is precisely what Section G.5 identifies as the source of the 2.25% gap. Applying parametric VaR's normal-distribution assumption directly to log-returns instead removes that transformation entirely — parametric VaR would then be assuming normality for the exact quantity GBM actually makes normal, so the simulated and parametric figures should coincide far more closely, with the small remaining gap attributable only to finite-sample Monte Carlo noise from using 200,000 paths rather than the true underlying distribution.

**5.** Holding `R_B`, `ENE(t)`, and `DF(t)` fixed, a higher `hazard_own` makes `Q_B(t)` fall faster with time, which makes each `[Q_B(tᵢ₋₁)-Q_B(tᵢ)]` term — the incremental probability of the bank itself defaulting in that specific interval — LARGER, which makes DVA larger. This is exactly the property that has made DVA controversial in practice: it means a bank's own reported earnings can go UP as its own credit quality gets WORSE, since a higher own-default probability increases the "expected gain from extinguishing our own obligations" DVA represents — an accounting outcome many practitioners and regulators have criticized as counterintuitive (a firm should not look more profitable purely because the market believes it is more likely to fail), which is part of why post-financial-crisis accounting and regulatory treatments of DVA have been repeatedly revisited.

**6.** GBM's volatility contribution to each step accumulates through the `sigma * sqrt(dt)` term, and for INDEPENDENT increments, variance is additive across time while standard deviation — the `sqrt` of variance — is not: the standard deviation of the price after time `t` scales with `sqrt(t)`, not `t` itself (this is the same square-root-of-time scaling behind the `sigma_1day = sigma * sqrt(dt)` conversion Section G.5 uses to go from an annual to a 1-day volatility). Since an at-the-money forward's exposure is driven almost entirely by how far `S(t)` has plausibly wandered from `K = S0` — which is governed by that same standard deviation, not the mean drift alone at these parameter magnitudes — `EE(t)`, being roughly proportional to the spread of the distribution at time `t`, inherits that same `sqrt(t)` growth rather than growing linearly, which is exactly the shape the table's own numbers (4.37, 6.43, 8.12, 9.66 at `t=0.25, 0.5, 0.75, 1.0`) trace out.

**7.** The four statements run in this order: `delete[] paths;` frees the old buffer; `count = other.count;` copies an `int`, unaffected by anything freed; `paths = new float[count];` allocates a brand-new buffer and assigns its address into `paths` — and since `other` and `*this` are the same object for `x = x`, this line *also* just changed the value of `other.paths`, because it's the identical variable. Only after that does `memcpy(paths, other.paths, ...)` run, by which point `other.paths` already holds the *new* allocation's address, not the freed one — so the copy reads uninitialized memory, not freed memory. The single-line reordering that WOULD produce a genuine use-after-free: move the `memcpy(...)` call to run *before* `paths = new float[count];`, i.e., swap those two statements. With that swap, at the moment `memcpy` runs, `paths` (and `other.paths`, same variable) still holds the pointer `delete[]` just freed one line earlier — `memcpy(paths, other.paths, ...)` would then read from AND write to that already-freed address, a genuine, ASan-catchable heap-use-after-free on both sides of the copy at once.

**8.** It depends on which specific standard library implementation you read, since the C++ standard requires only that self-assignment leave a `std::vector` unchanged, not any particular technique for guaranteeing it — but the technique real implementations generally favor avoids the issue structurally rather than checking for it explicitly. Instead of Section G.3's order (free the old resource, THEN allocate and copy the new one), a self-assignment-safe copy assignment builds the new state FIRST — allocate fresh storage, copy every element into it — and only releases the OLD storage after the new one is fully and successfully built. Under that ordering, `e = e` never has a moment where `e`'s data has been destroyed before it's been read, because reading happens entirely before anything is freed; no identity check is needed because the two objects being "the same object" never causes the same statement to see two different states of that object's data mid-operation, the way Worked Example G.3.2's UnguardedCopyBuffer did. Some implementations may still include an explicit `this == &other` fast path purely to skip redundant work on self-assignment (a performance choice, not a correctness requirement) — but the underlying safety guarantee comes from the allocate-before-free ordering itself, which is also exactly the single-line fix Question 7 identified as the difference between a safe copy and a genuine use-after-free.

**9.** With `SafeBuffer& operator=(const SafeBuffer& other)`, the body `swap(*this, other)` would not compile at all: `swap` takes both arguments by mutable reference (`swap(SafeBuffer& a, SafeBuffer& b)`), and a `const SafeBuffer&` cannot bind to a non-`const` reference parameter. Passing by value is what makes copy-and-swap work at all, for two reasons at once. First, it's what supplies a genuinely swappable, mutable local object — `other` — regardless of whether the caller's argument was an lvalue or an rvalue. Second, and this is the dispatch mechanism itself: constructing that local `other` from the function's actual argument is exactly where overload resolution happens — an lvalue argument (`c = d`) selects `SafeBuffer`'s copy constructor to build `other`, an rvalue argument (`e = std::move(f)`) selects the move constructor instead — and this selection happens automatically, via the same overload-resolution rules that pick any constructor, without `operator=`'s own body needing to know or care which one ran. A `const SafeBuffer&` parameter would still be aliasing the CALLER's actual object, meaning `swap(*this, other)` — if it somehow compiled — would mutate the caller's own variable as a side effect of assignment, which is not what `c = d` is supposed to do to `d`.

**10.** A type with no accessible copy constructor at all is **move-only** — copying it isn't merely undesirable, it's not an operation that exists. Worked Example G.3.4's reasoning for why `std::vector` prefers copy over a possibly-throwing move is that copying leaves the ORIGINAL element untouched if it fails, so the vector can still recover to its pre-reallocation state; but that reasoning only matters as a choice between two available options. For a move-only type, there is no second option — `std::move_if_noexcept` selecting move here isn't a claim that the move is safe, it's a recognition that moving is the *only* mechanically possible way to relocate the element at all, throwing or not. This is, honestly, a real, known gap in the strong-exception-guarantee story: `std::vector`'s reallocation genuinely cannot promise the same recovery-on-failure behavior for a move-only type with a throwing move constructor that it can for a copyable type — the standard's guarantee is correspondingly weaker in that specific corner, not because it was overlooked, but because no alternative to a possibly-throwing move exists to fall back to.
