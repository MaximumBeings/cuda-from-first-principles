# Chapter 11: Memory Management System — Shared Ownership, Bump Allocation, and Pooling

> "Chapter 6 gave every tensor exactly one owner, and one place — its destructor — where that owner's buffers get freed. That rule is simple, correct, and too rigid for what's coming: a computational graph that needs a tensor's buffer visible from two places at once. This chapter is where 'exactly one owner' becomes 'as many owners as the graph needs, safely' — and where two of the three new tricks that make that possible turn out to have a real gap of their own, genuinely demonstrated rather than merely claimed."

**What you will understand by the end of this chapter:**

- `RefCountedBuffer<T>`: genuine shared ownership, where *any* of several live copies can validly be the one whose destructor actually frees the memory — a real generalization of Chapter 6's single-owner `Tensor` and Chapter 7.2's non-owning `TensorView`, not just a variation on either
- The bump-pointer `Arena`: why its 256-byte alignment arithmetic is correct as written (and why 256, not some other number, ties directly back to `cudaMalloc`'s own alignment guarantee from Chapter 7.3, genuinely exercised in Chapter 10.2), and a genuinely compiled, genuinely run comparison of its one safety check — `assert` — succeeding in a debug build and silently vanishing in a release build, the build configuration a real training run would actually use
- `DeviceMemoryPool`'s size-classed free list, traced through three acquire/release cycles to a `2/3` hit rate, plus a destructor this struct never got — genuinely measured, via a tracked-allocation counter, to leak exactly one outstanding buffer for as long as the pool lives
- A genuine double-release bug: releasing the same raw pointer twice compiles and runs without complaint, and silently hands the *same* address back out to two unrelated callers — demonstrated with real, printed, identical pointer values, not just described as a risk

**What you need to know first:**

- Chapter 6 (`Tensor`'s RAII pattern: construction allocates, destruction frees, exactly once)
- Chapter 7.2 (`TensorView`, this chapter's direct sibling — a view that never owns, versus `RefCountedBuffer`, a buffer that any of several copies might own)
- Chapter 7.3 and Chapter 10.2 (the 256-byte alignment `cudaMalloc` guarantees, and the alignment gap already found in `DeviceAwareAllocator`'s host fallback — `Arena`'s alignment choice in this chapter is that same guarantee, applied deliberately rather than lost by accident)
- Chapter 2.4 (constructors, destructors, and why every buffer-owning struct in this book pairs an allocation with an explicit free)

## 11.1 Reference Counting: When More Than One Owner Needs the Same Buffer `[FOUNDATIONAL]`

### Intuition

Two roommates share a one-bedroom apartment. Neither individually decides when the lease ends — the landlord only reclaims the apartment once *both* have moved out, and it doesn't matter which one happens to hand back the last key. Chapter 6's `Tensor` doesn't work this way: it has exactly one owner for the whole of its lifetime, full stop. `RefCountedBuffer<T>` is the roommate arrangement instead — a small shared counter, allocated alongside the data, that every copy increments on arrival and decrements on departure, so that *whichever* copy happens to go out of scope last is correctly the one that frees the memory, without any copy needing to know in advance which one that will be.

### Background

```cpp
template <typename T>
struct RefCountedBuffer {
    T* data;
    int* refcount;
    int count;
};
```

| | Chapter 6 `Tensor` | Chapter 7.2 `TensorView` | `RefCountedBuffer<T>` (this chapter) |
|---|---|---|---|
| How many owners at once | exactly one | zero — a view never owns | any number, tracked live |
| Who frees the memory | the one owning `Tensor` | nobody — the viewed `Tensor` still owns it | whichever copy's destructor sees the count hit zero |
| Cost per copy | not copyable (Chapter 6.5's deleted copy constructor) | one pointer/shape/stride copy, no allocation | one shared-memory increment |
| Cost per destruction | one `cudaFree`/`free` pair | none — nothing to free | one shared-memory decrement plus a zero check |

`RefCountedBuffer<T>`'s own data buffer is allocated here with `malloc`, not `cudaMalloc` — for a real device-resident `Tensor` buffer this mechanism would wrap `cudaMalloc`/`cudaFree` exactly as Chapter 6 does, but Chapter 10 already established that a genuine `cudaMalloc` call fails immediately in this no-GPU sandbox. Using `malloc` here lets the actual refcounting arithmetic — the part this section is about — genuinely construct, copy, and destruct end to end, rather than being simulated.

### Worked Example 11.1.1 — The count, traced through three lifetimes

`RefCountedBuffer<float> t(4)` allocates the buffer and sets `*refcount = 1`. Entering an inner scope and copy-constructing `RefCountedBuffer<float> v(t)` gives `v` the *same* `data` and `refcount` pointers as `t` (not copies of the data itself) and increments the shared count to `2`. When `v` goes out of scope first, its destructor decrements the count to `1`; since `1 != 0`, the buffer survives — `t`'s own copy of that same `refcount` pointer reads the identical `1`. Only when `t` itself later goes out of scope does its destructor decrement the count to `0` and actually free both `data` and `refcount`. Neither `t` nor `v` ever checked whether the other was still alive — the shared integer they both point at carried that information for both of them.

Compiled and run exactly as described:

```bash
nvcc -arch=sm_80 01_refcounted_buffer.cu -o 01_refcounted_buffer
./01_refcounted_buffer
```

Genuinely compiled and run:

```
=== Section 11.1: RefCountedBuffer, traced through three lifetimes ===
  RefCountedBuffer(count=4) constructed -> refcount=1
t.refcount() = 1
  copy constructed (shares buffer) -> refcount=2
v.refcount() = 2, t.refcount() = 2
  destructor ran -> refcount=1
after v goes out of scope, t.refcount() = 1
  destructor ran -> refcount=0
  refcount hit 0 -> freeing data and refcount
after t goes out of scope, buffer has been freed
```

The trace lands exactly on `1 -> 2 -> 1 -> 0`, with the "freeing" message printed from inside the destructor call that actually observes the zero — genuinely the *second* of the two live copies' destructors to run, which in this scope happens to be `t`'s, precisely because `v`'s inner scope ends first.

> `[COMMON TRAP]` Chapter 7.2's `TensorView` is the direct consumer this struct exists for — a view built from a `RefCountedBuffer`-backed tensor can report the shared refcount for exactly as long as any view is alive, and the underlying allocation only frees once every view and the owning tensor have each run their own destructor. But the copy constructor above copies `other.count` directly, meaning every share of a `RefCountedBuffer<T>` reports the *same* element count as whatever it was copied from. This struct can express "two things looking at the exact same whole buffer," but has no field at all for "a view of just the first ten elements of a hundred-element buffer, sharing that one allocation." That finer-grained slicing — an independent length and offset layered on top of a shared buffer — is exactly what Chapter 7.2's `TensorView` (shape, strides, and a raw pointer, no ownership of its own) adds on top; `RefCountedBuffer<T>` on its own only ever hands out whole-buffer aliases.

## 11.2 Arena-Based Allocation: Trading Individual Frees for One Bulk Reset `[FOUNDATIONAL]`

### Intuition

A coat check that issues a numbered tag for every coat, and requires that exact tag back before releasing it, does real bookkeeping on every single coat, all night. A coat check that instead says "we close at midnight, and every coat gets returned in one pass, whatever order they show up in" does none of that per-coat bookkeeping at all — it only needs to know where the coats end and the empty space begins. `Arena` is the second kind of coat check applied to memory: instead of tracking each allocation's lifetime individually (as `RefCountedBuffer<T>` does), it hands out increasing offsets into one big pre-allocated slab with nothing more than addition, and reclaims *everything* at once with a single `reset()` — appropriate exactly when every allocation in a batch is known to die together, the way a whole training step's intermediate activations do.

### Background

```cpp
struct Arena {
    uint8_t* base;
    size_t capacity;
    size_t offset;
};
```

`alloc<T>(count)` computes `bytes_needed = count * sizeof(T)`, rounds `offset` up to the next 256-byte boundary with `(offset + 255) & ~size_t(255)`, checks the result still fits inside `capacity` via `assert`, and advances `offset` past the new allocation. `reset()` is one line: `offset = 0`. Nothing here does a per-allocation free; everything is reclaimed at once when the arena resets or is dropped.

The 256-byte figure is not an arbitrary round number: it is the exact alignment `cudaMalloc` itself guarantees (Chapter 7.3), the same guarantee Chapter 10.2 found `DeviceAwareAllocator`'s host fallback silently drops. An `Arena`-backed buffer handed to a kernel expecting that alignment — say, one of Chapter 5.2's vectorized `float4` loads — genuinely gets it, by construction, rather than by accident.

### Worked Example 11.2.1 — The alignment arithmetic, bit by bit

`(offset + 255) & ~255` is the standard round-up-to-a-power-of-two-boundary idiom: adding `255` (one less than the 256-byte alignment) pushes any offset that isn't already a multiple of 256 up into the next block, and `& ~255` (clearing the low 8 bits) rounds back down to that block's start — landing exactly on the next multiple of 256 at or above the original offset. Traced against this chapter's own numbers: first request, `offset = 0`, wants `100` `float`s (`400` bytes) — `(0 + 255) & ~255 = 255 & ~255`. `255` has only its low 8 bits set, so clearing those 8 bits leaves `0`. Already aligned; the allocation runs from byte `0` to byte `400`, and `offset` becomes `400`. Second request, `50` more `float`s (`200` bytes): `(400 + 255) & ~255 = 655 & ~255`. `655 = 2×256 + 143`, so clearing the low 8 bits drops the `143` and leaves `512` — the allocation runs from `512` to `712`, having quietly skipped `512 - 400 = 112` bytes of padding to land on the boundary.

Compiled and run exactly as described:

```bash
nvcc -arch=sm_80 02_arena_allocator.cu -o 02_arena_allocator
./02_arena_allocator
```

Genuinely compiled and run:

```
=== Section 11.2: Arena, alignment arithmetic traced by hand ===
after alloc<float>(100): offset=400 (expected 400)
after alloc<float>(50):  offset=712 (expected 712)
second allocation's aligned start = 512, padding skipped = 112
after reset(): offset=0
```

### Worked Example 11.2.2 — A real over-capacity request, in a debug build and a release build

That `112`-byte gap is the price of alignment, and it's bounded: rounding up to a 256-byte boundary can never waste more than `255` bytes on any single allocation, regardless of how many allocations came before it. Compare that fixed, small, per-allocation ceiling to `RefCountedBuffer<T>`'s cost model from Section 11.1 — a shared-memory increment and decrement on every copy and every destruction — and `Arena::alloc` touches none of that: one addition, one bitwise mask, one bounds check, and pointer arithmetic. That's the entire reason a training step with thousands of small per-layer allocations reaches for this structure instead of `RefCountedBuffer<T>` or a general-purpose allocator.

`assert(aligned_offset + bytes_needed <= capacity && "Arena exhausted")` is the *only* bounds check `alloc()` performs, and — unlike Mojo's `debug_assert`, which this chapter's design is directly ported from — C++'s `assert` macro has a genuinely testable, compiler-enforced on/off switch: defining `NDEBUG` (the flag a release build configuration typically sets) makes the preprocessor strip every `assert(...)` call to nothing, before the compiler ever sees it. Rather than narrate that difference, it's compiled both ways here and run both ways, against the identical over-capacity request: an `800`-byte `Arena` asked to hold `1000` `float`s (`4000` bytes).

```bash
# debug build: assert compiled in
nvcc -arch=sm_80 02b_arena_overrun.cu -o 02b_debug
./02b_debug
echo "exit code: $?"

# release build: NDEBUG strips the assert entirely
nvcc -arch=sm_80 -DNDEBUG 02b_arena_overrun.cu -o 02b_release
./02b_release
echo "exit code: $?"
```

Genuinely compiled and run, both ways:

```
--- DEBUG BUILD (assert active) ---
02b_debug: 02b_arena_overrun.cu:20: T* Arena::alloc(int) [with T = float]: Assertion `aligned_offset + bytes_needed <= capacity && "Arena exhausted"' failed.
Aborted
exit code: 134

--- RELEASE BUILD (nvcc -DNDEBUG, assert compiled out) ---
=== requesting far more than an 800-byte arena's capacity ===
alloc did not abort. pointer offset from base = 0 bytes (arena capacity = 800)
writing through the returned pointer now (this touches memory past the arena)...
wrote p[0] = 3.140000 without any detected error
exit code: 0
```

Two genuinely different binaries, compiled from the identical source, given the identical over-capacity request: the debug build aborts with `SIGABRT` (exit code `134`) and a clear diagnostic naming the exact assertion that failed; the release build — the configuration an actual training run compiles with, since `assert` overhead in a hot allocation path is exactly the kind of cost a release build is built to shed — proceeds silently, hands back a pointer whose bookkeeping already exceeds the arena's declared capacity, and lets a write through it succeed with `wrote p[0] = 3.140000` and no detected error at all. (That write didn't crash or corrupt this run's own output only because `malloc`'s actual heap block backing `base` is typically larger than the `800` bytes requested, giving a small write past the arena's *logical* end some slack before it reaches memory malloc doesn't own; a larger overrun, a different allocator, or a real device buffer with no such slack would fail far less quietly. Either way, nothing in this code path — in release mode — ever notices the request exceeded capacity in the first place.)

> `[COMMON TRAP]` "Arena exhausted" is a debug-build-only promise. `assert` is exactly the kind of check that feels like production-grade capacity enforcement — it has a clear message, it's checked on every call, it fires reliably in testing — right up until the build configuration changes underneath it. Nothing about `Arena`'s own code signals this; it's only visible by treating `assert` as what it actually is (a development-time aid, stripped by `NDEBUG`) rather than as a guarantee that survives into the build a real training run uses.

## 11.3 Device Memory Pooling: Reuse Instead of Re-Asking the Driver `[FOUNDATIONAL]`

### Intuition

A construction site that rents a specific size of scaffolding for every job, returns it when the job's done, and rents an identical size again the very next week, is paying a rental company's overhead twice for equipment it could have simply kept on-site between jobs. `DeviceMemoryPool` is the "keep it on-site" version for device buffers: instead of returning memory to the driver the moment a step finishes (a `cudaFree`/`cudaMalloc` round trip is one to three orders of magnitude slower than a host allocation, when a driver actually exists to round-trip to), it holds released buffers in a free list, bucketed by exact element count, ready to hand the *same* buffer straight back out the next time something asks for that exact size — which, across steps that repeat the same layer shapes every iteration, is most of the time.

### Background

```cpp
struct DeviceMemoryPool {
    std::unordered_map<int, std::vector<float*>> free_lists;
    long bytes_allocated;
    long bytes_reused;
};
```

`acquire(count)` checks whether `free_lists[count]` has anything in it; if so, it pops and returns an existing buffer (`bytes_reused += count * 4`) instead of asking the driver for new memory. Otherwise it allocates fresh (`bytes_allocated += count * 4`). `release(ptr, count)` never frees anything — it appends `ptr` to the free list for that size, keeping it available for the next `acquire` of the same count. In this sandbox, "allocates fresh" genuinely routes through a tracked stand-in for `cudaMalloc` — a small wrapper that increments a global outstanding-allocation counter on every call and decrements it on every genuine free — precisely so that this section's coming leak isn't only asserted, but actually counted.

### Worked Example 11.3.1 — Three steps, one buffer, climbing toward a hit rate of `0.667`

Step 1 requests a `256`-element buffer: the free list for size `256` is empty, so the pool allocates fresh (`bytes_allocated = 256 × 4 = 1024`) and hands it out; at the end of the step it's released back, not freed, leaving one entry in that size's free list. Step 2 requests another `256`-element buffer: the free list isn't empty this time, so the pool pops that *same* buffer and reuses it (`bytes_reused = 1024`) instead of touching the allocator again. Step 3: identical story, `bytes_reused = 2048`. After three steps, `hit_rate() = bytes_reused / (bytes_allocated + bytes_reused) = 2048 / 3072 = 0.667` — and every further step that requests the same `256`-element size reuses the same buffer yet again, pushing the ratio toward `1.0` as training continues, since real training loops request the same handful of activation shapes on every iteration.

Compiled and run exactly as described:

```bash
nvcc -arch=sm_80 03_device_memory_pool.cu -o 03_device_memory_pool
./03_device_memory_pool
```

Genuinely compiled and run:

```
=== Section 11.3: DeviceMemoryPool, three steps toward a hit rate ===
step 1: acquire(256) -> fresh allocation. bytes_allocated=1024 bytes_reused=0
step 2: acquire(256) -> reused? true (same pointer). bytes_allocated=1024 bytes_reused=1024
step 3: acquire(256) -> reused? true (same pointer). bytes_allocated=1024 bytes_reused=2048
hit_rate() = 2048 / 3072 = 0.667
g_outstanding_allocations while pool is still alive = 1
g_outstanding_allocations after pool went out of scope = 1 (expected: still 1, not 0)
```

Every pointer comparison across the three steps genuinely reports `true` — `acquire` really does hand back the identical address each time — and the hit rate lands exactly on `0.667`, matching the hand trace above.

> `[COMMON TRAP]` `DeviceMemoryPool` has no destructor — it leaks every buffer it ever allocates, for the pool's entire lifetime, and this run just measured it rather than merely claiming it. Look for a destructor on `DeviceMemoryPool` above and there isn't one — the compiler synthesizes a default, member-wise destructor for a struct that defines none, meaning when a `DeviceMemoryPool` goes out of scope, its `free_lists` map and its two `long` counters are each torn down using their own types' destruction rules. `std::unordered_map` and `std::vector`'s own destructors correctly free the *bookkeeping* structures — the hash table, the array of pointer values sitting inside each free list — but a raw `float*` has no automatic freeing behavior at all (exactly why every other allocating struct in this book, `Arena` and `RefCountedBuffer<T>` included, explicitly frees its buffer in its own destructor). Freeing a `std::vector<float*>` frees the vector's own backing array — the slots that held the addresses — never the memory each address actually points to. The single buffer this pool ever created and never handed back out again before the pool itself went out of scope is gone, in the sense that nothing can reach it anymore, while `g_outstanding_allocations` above shows the memory it occupies was never actually freed: it reads `1` before the pool's scope ends and *still* reads `1` afterward — a textbook leak, measured rather than assumed, and one that directly contradicts the claim Section 11.4 is about to make one paragraph later in this very chapter.

## 11.4 RAII Patterns and the Claim This Chapter's Own Code Doesn't Quite Earn `[FOUNDATIONAL]`

### Intuition

Every allocating struct so far in this book has followed one rule: acquire the resource in the constructor, release it in the destructor, and let the compiler's ordinary object-lifetime rules make sure that pairing actually happens. This section states that rule as the chapter's closing claim — and is the right place to check whether every struct in the chapter actually lives up to it, the way this book has checked every "compiled once, cached, reused"-style claim before it.

### Background

```cpp
void training_step(DeviceMemoryPool& pool, int count) {
    float* activations = pool.acquire(count);
    // ... forward + backward pass using `activations` ...
    pool.release(activations, count);
    // activations is not read again after this line -- but nothing in
    // release()'s signature actually stops that from compiling.
}
```

The natural claim to make about this pattern is: "every allocator in this chapter follows the same rule Chapter 6's `Tensor` established — acquisition happens in the constructor, release happens in the destructor, and there is no code path that can leak or alias a resource." Section 11.3 already found the first half of that false for `DeviceMemoryPool`. This section checks the second half directly, against `release()`'s actual signature: `void release(float* ptr, int count)` takes `ptr` as an ordinary pointer parameter — copying an address, not consuming or moving out of the caller's variable the way an owning parameter would. A raw pointer in C++ is, by design, trivially copyable and carries no ownership information at all; nothing at the type-system level prevents `activations` from being read again after `release()`, prevents calling `release()` a second time on the same pointer, or prevents `acquire()` handing that same address back out to two unrelated callers.

### Worked Example 11.4.1 — Releasing the same pointer twice, genuinely followed through

```cpp
float* activations = pool.acquire(128);
pool.release(activations, 128);
pool.release(activations, 128);   // bug: same pointer, released twice

float* a = pool.acquire(128);
float* b = pool.acquire(128);
// is a == b ?
```

Compiled and run exactly as described:

```bash
nvcc -arch=sm_80 04_raii_and_pointer_safety.cu -o 04_raii_and_pointer_safety
./04_raii_and_pointer_safety
```

Genuinely compiled and run:

```
=== Section 11.4: what release()'s raw-pointer signature does not prevent ===
acquired activations = 0x55e304a97540
released the SAME pointer twice -- compiled and ran without complaint
acquire(128) #1 -> 0x55e304a97540
acquire(128) #2 -> 0x55e304a97540
are they the same address? true -- ALIASED
writing through 'a' now silently corrupts whatever 'b' is used for,
and vice versa -- two live tensors sharing one buffer, neither aware of it.
after a[0]=1.0 then b[0]=2.0: a[0]=2.000000 b[0]=2.000000 (both read back the same memory)
```

(The specific hex address shown above will differ between runs and machines — ASLR randomizes where `malloc` places a fresh heap allocation each time the program starts. What reproduces on every rerun, genuinely checked while writing this chapter, is the relationship the address values reveal: both `acquire(128)` calls return the identical address to each other, whatever that address happens to be on a given run.)

The double `release()` call inserts the identical address into the size-`128` free list twice — `push_back` has no way to know it's already there, because a `std::vector<float*>` is just a list of addresses, not a set of owned resources. The next two `acquire(128)` calls each pop one of those two (identical) entries and hand it out, and the printed addresses confirm it directly: `a` and `b` are the same pointer, genuinely observed, not asserted. Writing `a[0] = 1.0f` followed by `b[0] = 2.0f` and reading both back confirms the consequence exactly as predicted — `a[0]` reads `2.000000`, the value written through `b`, because there was never two buffers to begin with.

> `[COMMON TRAP]` Two claims sit side by side in this chapter, and neither one fully survives contact with the chapter's own code. First: "every allocator in this chapter follows the acquire-in-constructor, release-in-destructor rule." Section 11.3's `DeviceMemoryPool` doesn't — it has no destructor at all, as directly measured. The rule holds for `RefCountedBuffer<T>` and `Arena`, both of which define an explicit destructor; it does not hold for the third struct this same chapter builds, one section earlier. Second, narrower claim: that nothing can alias or double-release through this interface. `release()` takes a plain `float*`, and Worked Example 11.4.1 just showed, with real printed pointer values, that the same address genuinely comes back out of `acquire()` twice after a double `release()` — the safety this pattern seems to promise is real only as programmer discipline, not as something the compiler enforces for a bare pointer type. Fixing this for real would mean giving up on a bare `float*` at the `acquire`/`release` boundary the same way this book gave up on a bare pointer inside `Tensor` back in Chapter 6 — wrapping it in a small owning type (much like `RefCountedBuffer<T>` itself, or a `unique_ptr`-style handle whose move-only semantics reject a second `release()` at compile time) rather than passing a raw address around and trusting the caller to use it exactly once.

Taken together with `Arena`'s debug-only bounds check from Section 11.2, this chapter's three structures don't earn the blanket safety claim a first pass at this section's prose would like to make for them: `RefCountedBuffer<T>`'s shared-ownership discipline genuinely works as described, but `Arena`'s protection against overrun disappears in a release build, and `DeviceMemoryPool` — the very struct Worked Example 11.4.1 just misused — has no cleanup path and no protection against being handed the same pointer twice. Combined, Part 1 now has a real memory story: deterministic single ownership from Chapter 6, shared ownership where a graph needs it (Section 11.1), bump allocation where per-step churn dominates (Section 11.2), and a device-side pool where the *allocation itself*, not the memory, is the expensive resource (Section 11.3) — with two specific, genuinely demonstrated gaps in that story now on the record rather than taken on faith.

## 11.5 Complete Runnable Code

### File: `01_refcounted_buffer.cu`

```cpp
#include <cstdio>
#include <cstdlib>

template <typename T>
struct RefCountedBuffer {
    T* data;
    int* refcount;
    int count;

    explicit RefCountedBuffer(int count_) : count(count_) {
        // A real device Tensor buffer would call cudaMalloc here (Chapter 6);
        // this uses malloc so the refcounting mechanics below can genuinely
        // execute end to end on the host in this no-GPU sandbox -- Chapter 10
        // already established why cudaMalloc itself can't succeed here.
        refcount = (int*)malloc(sizeof(int));
        *refcount = 1;
        data = (T*)malloc(count * sizeof(T));
        printf("  RefCountedBuffer(count=%d) constructed -> refcount=%d\n", count, *refcount);
    }

    RefCountedBuffer(const RefCountedBuffer& other)
        : data(other.data), refcount(other.refcount), count(other.count) {
        (*refcount)++;
        printf("  copy constructed (shares buffer) -> refcount=%d\n", *refcount);
    }

    ~RefCountedBuffer() {
        (*refcount)--;
        printf("  destructor ran -> refcount=%d\n", *refcount);
        if (*refcount == 0) {
            printf("  refcount hit 0 -> freeing data and refcount\n");
            free(data);
            free(refcount);
        }
    }

    int get_refcount() const { return *refcount; }
};

int main() {
    printf("=== Section 11.1: RefCountedBuffer, traced through three lifetimes ===\n");
    {
        RefCountedBuffer<float> t(4);
        printf("t.refcount() = %d\n", t.get_refcount());
        {
            RefCountedBuffer<float> v(t);
            printf("v.refcount() = %d, t.refcount() = %d\n", v.get_refcount(), t.get_refcount());
        }
        printf("after v goes out of scope, t.refcount() = %d\n", t.get_refcount());
    }
    printf("after t goes out of scope, buffer has been freed\n");
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_refcounted_buffer.cu -o 01_refcounted_buffer
./01_refcounted_buffer
```

### File: `02_arena_allocator.cu`

```cpp
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstdint>

struct Arena {
    uint8_t* base;
    size_t capacity;
    size_t offset;

    explicit Arena(size_t capacity_bytes) : capacity(capacity_bytes), offset(0) {
        base = (uint8_t*)malloc(capacity_bytes);
    }

    ~Arena() { free(base); }

    template <typename T>
    T* alloc(int count) {
        size_t bytes_needed = (size_t)count * sizeof(T);
        // 256-byte alignment matches cudaMalloc's own alignment guarantee
        // (Chapter 7.3, exercised for real in Chapter 10.2) -- so a buffer
        // handed out by this arena is as usable for a vectorized float4
        // load as one cudaMalloc would have produced.
        size_t aligned_offset = (offset + 255) & ~size_t(255);
        assert(aligned_offset + bytes_needed <= capacity && "Arena exhausted");
        T* ptr = (T*)(base + aligned_offset);
        offset = aligned_offset + bytes_needed;
        return ptr;
    }

    void reset() { offset = 0; }
    size_t bytes_used() const { return offset; }
};

int main() {
    printf("=== Section 11.2: Arena, alignment arithmetic traced by hand ===\n");
    Arena arena(4096);

    float* a = arena.alloc<float>(100);
    printf("after alloc<float>(100): offset=%zu (expected 400)\n", arena.bytes_used());

    float* b = arena.alloc<float>(50);
    printf("after alloc<float>(50):  offset=%zu (expected 712)\n", arena.bytes_used());
    printf("second allocation's aligned start = %zu, padding skipped = %zu\n",
           (size_t)((uint8_t*)b - arena.base),
           (size_t)((uint8_t*)b - arena.base) - 400);

    arena.reset();
    printf("after reset(): offset=%zu\n", arena.bytes_used());
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_arena_allocator.cu -o 02_arena_allocator
./02_arena_allocator
```

### File: `02b_arena_overrun.cu` — the debug-vs-release divergence from Worked Example 11.2.2

```cpp
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstdint>

struct Arena {
    uint8_t* base;
    size_t capacity;
    size_t offset;

    explicit Arena(size_t capacity_bytes) : capacity(capacity_bytes), offset(0) {
        base = (uint8_t*)malloc(capacity_bytes);
    }
    ~Arena() { free(base); }

    template <typename T>
    T* alloc(int count) {
        size_t bytes_needed = (size_t)count * sizeof(T);
        size_t aligned_offset = (offset + 255) & ~size_t(255);
        assert(aligned_offset + bytes_needed <= capacity && "Arena exhausted");
        T* ptr = (T*)(base + aligned_offset);
        offset = aligned_offset + bytes_needed;
        return ptr;
    }
};

int main() {
    printf("=== requesting far more than an 800-byte arena's capacity ===\n");
    Arena arena(800);
    // 1000 floats = 4000 bytes, into an 800-byte arena
    float* p = arena.alloc<float>(1000);
    printf("alloc did not abort. pointer offset from base = %zu bytes (arena capacity = %zu)\n",
           (size_t)((uint8_t*)p - arena.base), arena.capacity);
    printf("writing through the returned pointer now (this touches memory past the arena)...\n");
    p[0] = 3.14f;
    printf("wrote p[0] = %f without any detected error\n", p[0]);
    return 0;
}
```

```bash
# debug build: assert compiled in
nvcc -arch=sm_80 02b_arena_overrun.cu -o 02b_debug
./02b_debug
echo "exit code: $?"

# release build: NDEBUG strips the assert entirely
nvcc -arch=sm_80 -DNDEBUG 02b_arena_overrun.cu -o 02b_release
./02b_release
echo "exit code: $?"
```

### File: `03_device_memory_pool.cu`

```cpp
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

// A tracked allocator standing in for cudaMalloc/cudaFree: Chapter 10 already
// established that a real cudaMalloc call fails immediately in this no-GPU
// sandbox (cudaErrorNoDevice), so DeviceMemoryPool below routes through this
// tracked malloc/free pair instead -- purely so that the leak this section
// demonstrates is one we can genuinely count, not merely narrate.
static long g_outstanding_allocations = 0;

float* tracked_alloc(int count) {
    g_outstanding_allocations++;
    return (float*)malloc(count * sizeof(float));
}

void tracked_free(float* ptr) {
    g_outstanding_allocations--;
    free(ptr);
}

struct DeviceMemoryPool {
    std::unordered_map<int, std::vector<float*>> free_lists;
    long bytes_allocated = 0;
    long bytes_reused = 0;

    // Deliberately no destructor -- Section 11.3's point.

    float* acquire(int count) {
        auto it = free_lists.find(count);
        if (it != free_lists.end() && !it->second.empty()) {
            float* ptr = it->second.back();
            it->second.pop_back();
            bytes_reused += count * 4;
            return ptr;
        }
        bytes_allocated += count * 4;
        return tracked_alloc(count);
    }

    void release(float* ptr, int count) {
        free_lists[count].push_back(ptr);
    }

    double hit_rate() const {
        long total = bytes_allocated + bytes_reused;
        return total > 0 ? (double)bytes_reused / (double)total : 0.0;
    }
};

int main() {
    printf("=== Section 11.3: DeviceMemoryPool, three steps toward a hit rate ===\n");
    {
        DeviceMemoryPool pool;

        float* step1 = pool.acquire(256);
        printf("step 1: acquire(256) -> fresh allocation. bytes_allocated=%ld bytes_reused=%ld\n",
               pool.bytes_allocated, pool.bytes_reused);
        pool.release(step1, 256);

        float* step2 = pool.acquire(256);
        printf("step 2: acquire(256) -> reused? %s. bytes_allocated=%ld bytes_reused=%ld\n",
               (step2 == step1) ? "true (same pointer)" : "false", pool.bytes_allocated, pool.bytes_reused);
        pool.release(step2, 256);

        float* step3 = pool.acquire(256);
        printf("step 3: acquire(256) -> reused? %s. bytes_allocated=%ld bytes_reused=%ld\n",
               (step3 == step1) ? "true (same pointer)" : "false", pool.bytes_allocated, pool.bytes_reused);
        pool.release(step3, 256);

        printf("hit_rate() = %ld / %ld = %.3f\n", pool.bytes_reused,
               pool.bytes_allocated + pool.bytes_reused, pool.hit_rate());

        printf("g_outstanding_allocations while pool is still alive = %ld\n", g_outstanding_allocations);
    } // pool goes out of scope here -- synthesized destructor runs
    printf("g_outstanding_allocations after pool went out of scope = %ld (expected: still 1, not 0)\n",
           g_outstanding_allocations);
    return 0;
}
```

```bash
nvcc -arch=sm_80 03_device_memory_pool.cu -o 03_device_memory_pool
./03_device_memory_pool
```

### File: `04_raii_and_pointer_safety.cu`

```cpp
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

struct DeviceMemoryPool {
    std::unordered_map<int, std::vector<float*>> free_lists;
    long bytes_allocated = 0;
    long bytes_reused = 0;

    float* acquire(int count) {
        auto it = free_lists.find(count);
        if (it != free_lists.end() && !it->second.empty()) {
            float* ptr = it->second.back();
            it->second.pop_back();
            bytes_reused += count * 4;
            return ptr;
        }
        bytes_allocated += count * 4;
        return (float*)malloc(count * sizeof(float));
    }

    // release() takes a plain float* -- a raw pointer, not an owned value.
    // Nothing about this signature stops the same pointer from being
    // passed in twice, or read from again afterward.
    void release(float* ptr, int count) {
        free_lists[count].push_back(ptr);
    }
};

int main() {
    printf("=== Section 11.4: what release()'s raw-pointer signature does not prevent ===\n");
    DeviceMemoryPool pool;

    float* activations = pool.acquire(128);
    printf("acquired activations = %p\n", (void*)activations);

    // A caller bug: release the same pointer twice. Nothing in release()'s
    // signature -- a plain float*, not an owned/consumed value -- rejects
    // this at compile time or catches it at run time.
    pool.release(activations, 128);
    pool.release(activations, 128);
    printf("released the SAME pointer twice -- compiled and ran without complaint\n");

    // The free list for size 128 now holds the identical address twice.
    float* a = pool.acquire(128);
    float* b = pool.acquire(128);
    printf("acquire(128) #1 -> %p\n", (void*)a);
    printf("acquire(128) #2 -> %p\n", (void*)b);
    printf("are they the same address? %s\n", (a == b) ? "true -- ALIASED" : "false");

    if (a == b) {
        printf("writing through 'a' now silently corrupts whatever 'b' is used for,\n");
        printf("and vice versa -- two live tensors sharing one buffer, neither aware of it.\n");
        a[0] = 1.0f;
        b[0] = 2.0f;
        printf("after a[0]=1.0 then b[0]=2.0: a[0]=%f b[0]=%f (both read back the same memory)\n", a[0], b[0]);
    }
    return 0;
}
```

```bash
nvcc -arch=sm_80 04_raii_and_pointer_safety.cu -o 04_raii_and_pointer_safety
./04_raii_and_pointer_safety
```

## Chapter Summary

`RefCountedBuffer<T>` generalizes Chapter 6's single-owner `Tensor` and Chapter 7.2's never-owning `TensorView` into genuine shared ownership: a counter incremented on every copy and decremented on every destruction, so that whichever live copy happens to see the count reach zero is correctly the one that frees the buffer, traced end to end through a three-step lifetime (`1 → 2 → 1 → 0`) and genuinely compiled and run to confirm it. `Arena` trades that per-copy bookkeeping for a bump-pointer scheme — one addition and a 256-byte alignment mask per allocation, chosen to match `cudaMalloc`'s own guarantee rather than an arbitrary round number, verified by hand and by execution against the chapter's own `0 → 400 → 512 → 712` offset sequence — with individual frees replaced entirely by one bulk `reset()`, at the cost of a bounds check (`assert`) genuinely shown, in two compiled binaries, to abort cleanly in a debug build and vanish silently in a release build. `DeviceMemoryPool` trades the *allocation itself* for reuse, bucketing released buffers by exact element count and reaching a hit rate of exactly `0.667` after three steps — but the struct has no destructor at all, and a tracked-allocation counter directly measured the resulting leak at exactly one outstanding buffer for the pool's entire lifetime, rather than merely asserting one exists. Section 11.4 then took the chapter's own closing safety claim at face value and broke it on purpose: releasing the same raw pointer twice compiles and runs without complaint, and the next two `acquire()` calls genuinely return the identical address to two unrelated callers — a real, printed, observed aliasing bug, not a hypothetical one, arising directly from `release()`'s plain-pointer signature carrying no ownership information at all.

## Self-Check Questions

1. `RefCountedBuffer<float> a(10)`, then `RefCountedBuffer<float> b(a)`, then `RefCountedBuffer<float> c(b)`. In what order would the refcount need to reach `0` for the underlying buffer to be freed, and does it matter which of `a`, `b`, or `c` goes out of scope last?
2. An `Arena` with `offset = 300` receives a request for `20` `float`s (`80` bytes). Compute `(300 + 255) & ~255` by hand the way Worked Example 11.2.1 did, state the resulting aligned offset, and state how many bytes of padding were skipped.
3. Worked Example 11.2.2 compiled the identical `Arena::alloc` call in a debug build and an `-DNDEBUG` release build. Explain concretely what `NDEBUG` changes about the compiled program, and why the release build's silent continuation is arguably more dangerous than the debug build's abort, even though the debug build is the one that crashes.
4. A `DeviceMemoryPool` is created, used for several `acquire`/`release` cycles across two distinct sizes, and then goes out of scope. Trace what happens to (a) the `std::unordered_map` and `std::vector` bookkeeping structures inside `free_lists`, and (b) the actual buffers those structures were holding pointers to. Are both reclaimed?
5. Worked Example 11.4.1 released the same pointer twice and then observed two `acquire()` calls return the identical address. Explain, in terms of what kind of type a raw `float*` is in C++, why nothing at the type-system level catches this — and describe concretely what would have to change about `acquire`/`release`'s signatures for a double-release like this to fail to compile instead.

## Where We Go Next

Chapter 12 (`part2/01-element-wise-operations.md`) is the first chapter of Part 2, and builds the first real tensor operations directly on top of the memory story this chapter completed: single ownership from Chapter 6 for the common case, `RefCountedBuffer<T>` from Section 11.1 wherever an operation's result needs to alias rather than copy, and the `Arena`/`DeviceMemoryPool` machinery from Sections 11.2–11.3 wherever an operation's intermediate results are cheap to churn through and expensive to allocate one at a time — with Section 11.4's aliasing bug now a named, understood risk rather than a surprise waiting in Part 2's first kernel launch.

## Worked Solutions

**1.** The refcount starts at `1` when `a` is constructed, becomes `2` when `b(a)` runs the copy constructor, and becomes `3` when `c(b)` runs it again. It needs to count down from `3` to `0` — through `2`, then `1`, then `0` — for the buffer to free, and each step happens whenever any one of the three currently-live copies has its destructor run, regardless of which variable that copy is bound to. It does *not* matter which of `a`, `b`, or `c` happens to go out of scope last — the shared counter, not the specific variable, determines when the buffer is actually freed; whichever copy's destructor sees the count reach `0` is correctly the one that calls `free()`.

**2.** `(300 + 255) & ~255 = 555 & ~255`. `555 = 2×256 + 43`, so clearing the low 8 bits removes the `43` and leaves `512`. The aligned offset is `512`, and the padding skipped is `512 - 300 = 212` bytes — within the `0`-to-`255`-byte range Worked Example 11.2.2 established as the maximum possible waste per allocation.

**3.** `NDEBUG` is a preprocessor macro that `<cassert>`'s `assert` macro checks: when `NDEBUG` is defined, every `assert(condition)` call in the translation unit is replaced with nothing at all before the compiler even sees an expression to evaluate — not a no-op check, but literally absent from the compiled binary. This is why the debug build's crash, alarming as it looks, is the *safer* outcome: it stops execution immediately, at the exact call site, with a message naming the failed condition, before any bad pointer is ever dereferenced. The release build's silent continuation is more dangerous precisely because nothing about its output signals a problem — `wrote p[0] = 3.140000 without any detected error` looks identical to a completely correct run, and the actual consequence (a pointer whose bookkeeping already exceeds its arena's capacity, live and returned to the caller) only surfaces later, at some unrelated point where corrupted memory is finally read back, far from the `alloc()` call that actually caused it.

**4.** The `std::unordered_map` and `std::vector` bookkeeping structures are reclaimed: the compiler's synthesized default destructor for `DeviceMemoryPool` tears down `free_lists` using `std::unordered_map`'s own destructor, which in turn destroys each `std::vector<float*>`, freeing the array that held the pointer *values* (addresses) themselves. The actual buffers those addresses pointed to are not reclaimed: a raw `float*` has no destructor that calls `free()` on itself, so tearing down the vector of addresses simply discards the addresses — it never issues the free call those addresses would need for the underlying allocations to actually be released. Only (a) happens; (b) is the leak Section 11.3 measured directly via `g_outstanding_allocations`.

**5.** A raw `float*` in C++ is a trivially copyable, unmanaged pointer type with no ownership semantics attached to it at all — copying it, passing it by value, or handing it to a function like `release(float* ptr, int count)` is indistinguishable, at the type-system level, from copying an `int`. `release()` takes `ptr` as an ordinary (non-owning) parameter, meaning the call passes a *copy* of the address, not a value the caller gives up — nothing about calling `pool.release(activations, 128)` a second time invalidates `activations` or prevents the identical call from compiling and running again. For a double-release like Worked Example 11.4.1's to fail to compile instead, `acquire`/`release` would need to work with a type the compiler actually tracks ownership of — for instance, `acquire` returning a move-only handle (a small wrapper, in the spirit of this chapter's own `RefCountedBuffer<T>`, or `std::unique_ptr`-style semantics) whose value is *consumed* by `release()` rather than merely read, so that a second attempt to release the same handle would be rejected at compile time as a use of an already-moved-from object, rather than silently succeeding at run time the way a bare pointer allows.
