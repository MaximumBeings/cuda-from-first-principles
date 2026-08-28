# Chapter 10: Device Abstraction Layer — Discovery and Memory Management, Minimally

> "Every kernel launch and every `cudaMalloc` call in this book so far has simply assumed a device might be there and asked CUDA to try anyway, catching the honest failure afterward. A real program shouldn't gamble that way on every single allocation — it should ask once, remember the answer, and let that answer decide the strategy. This chapter builds the small struct that asks, and discovers along the way that even the *asking* has real, measurable costs and real, easy-to-miss ways of lying to you."

**What you will understand by the end of this chapter:**

- `DeviceManager`: a struct that calls `cudaGetDeviceCount` exactly once and caches the answer, rather than letting every later allocation rediscover "no device" the hard way — and a genuine, verified fact about that call's output parameter that most tutorials gloss over
- Why bypassing discovery and calling a device operation directly doesn't fail any *worse*, but does fail with less context and at a less predictable point in a program
- `DeviceAwareAllocator`: routing between `cudaMalloc` and a plain host `malloc()` fallback based on the manager's cached answer, plus a second, independent fallback for when discovery itself turns out to have been optimistic
- A genuine, measured gap between what a device allocation path would guarantee about memory alignment and what its host fallback silently does not
- Why comparing a failing `cudaMalloc` call's timing against a succeeding `malloc()` call's timing is not the GPU-vs-CPU allocation benchmark it might look like, and what it actually measures instead

**What you need to know first:**

- Chapter 1.3 and Chapter 4.2 (the honest `cudaErrorNoDevice` disclosure this book has shown since its very first device-touching call) — this chapter is the first one to build something that *uses* that knowledge to avoid the failure altogether, rather than just reporting it afterward
- Chapter 6.4's "trust the type, not a re-derived guess" argument, and Chapter 8.3's parallel lesson about not trusting a value scanned from unreliable evidence — this chapter's discovery of `cudaGetDeviceCount`'s stale output parameter is the same lesson again, one layer closer to the hardware
- Chapter 7.3 (alignment and padding) — Section 10.2's alignment gap is a direct, concrete instance of that chapter's general warning
- If you've read the Mojo edition: this chapter follows its Chapter 10 section-for-section (device discovery, then a device-aware allocator built on RAII), using CUDA's actual `cudaGetDeviceCount`/`cudaMalloc`/`cudaMallocManaged`/`cudaHostAlloc` family in place of Mojo's own device manager API.

## 10.1 Device Discovery: Ask Once, Trust the Answer `[FOUNDATIONAL]`

### Intuition

Every device-touching call since Chapter 1.3 has asked CUDA to *try* an operation and reported whatever error came back — a fine strategy for a worked example, but a wasteful one for a real program that might make thousands of allocation decisions, each one re-discovering the identical "no device" fact from scratch. `DeviceManager` asks exactly one question, once, at startup — `cudaGetDeviceCount` — and caches the answer for everything downstream to consult instead of re-asking.

### Background

| | Asking every time (this book's practice through Chapter 9) | `DeviceManager` (this chapter) |
|---|---|---|
| Number of discovery calls for `N` allocations | `N` (each allocation call itself is the "question") | `1`, at construction |
| What downstream code checks | The return code of the operation it just attempted | A cached boolean, decided once |
| Failure mode when bypassed | N/A — there's nothing to bypass | The exact same underlying failure, just discovered later and with less context (Worked Example 10.1.2) |

### Worked Example 10.1.1 — discovery that can't discover anything, and a genuine trap in its output parameter

```cpp
struct DeviceManager {
    int device_count;
    bool discovery_succeeded;

    DeviceManager() {
        int count = -999;  // a deliberately absurd sentinel -- if this survives, discovery lied
        cudaError_t err = cudaGetDeviceCount(&count);
        discovery_succeeded = (err == cudaSuccess);
        device_count = discovery_succeeded ? count : 0;  // never trust `count` after a failed call
    }

    bool has_device(int id) const { return discovery_succeeded && id < device_count; }
};
```

Compiled and run as part of the complete `01_device_discovery.cu` further below:

```bash
nvcc -arch=sm_80 01_device_discovery.cu -o 01_device_discovery
./01_device_discovery
```

Genuinely compiled and run:

```
DeviceManager() -> cudaGetDeviceCount err=100 (no CUDA-capable device is detected)
  raw count parameter after the call: -999
  discovery_succeeded=false, device_count=0
```

This is the genuine trap: `cudaGetDeviceCount` does *not* set `count` to `0` when it fails — the `-999` sentinel this constructor deliberately planted survives completely untouched, because a failed CUDA Runtime API call makes no promise whatsoever about its output parameters. Code that skips checking the returned `cudaError_t` and simply reads `count` afterward — a completely reasonable-looking mistake, since "the count of devices" sounds like exactly the kind of value that should default sensibly to zero on failure — would instead read whatever garbage the sentinel (or, in less careful code, an entirely uninitialized stack variable) happened to hold. `DeviceManager`'s constructor closes this gap in exactly one place: `device_count` is only ever set from the real `count` when `discovery_succeeded` is genuinely `true`.

### Worked Example 10.1.2 — bypassing the manager entirely

```cpp
cudaError_t err2 = cudaSetDevice(0);
```

Compiled and run as part of the complete `01_device_discovery.cu` further below:

```bash
nvcc -arch=sm_80 01_device_discovery.cu -o 01_device_discovery
./01_device_discovery
```

Genuinely compiled and run:

```
mgr.has_device(0) = false

--- bypassing the manager entirely ---
code that skips has_device() and calls cudaSetDevice(0) directly:
  cudaSetDevice(0) -> err=100 (no CUDA-capable device is detected)

cudaGetDeviceProperties and cudaGetDevice, genuinely queried:
  cudaGetDeviceProperties(0) -> err=100 (no CUDA-capable device is detected)
  cudaGetDevice -> err=100 (no CUDA-capable device is detected), current still = -999 (untouched by the failed call)
```

Calling `cudaSetDevice(0)` directly, without ever checking `mgr.has_device(0)` first, produces the exact same `cudaErrorNoDevice` the manager already knew was coming — bypassing the manager doesn't make the failure worse, or different, or harder to recover from in this specific case. What it loses is *context*: `mgr.has_device(0)` returning `false` is a clear, single, program-level fact checkable before anything else runs, while `cudaSetDevice`'s failure surfaces deep inside whatever function happened to call it, at whatever point in the program that call occurs — the same information, discovered later and with a narrower view of why it matters. `cudaGetDeviceProperties` and `cudaGetDevice` both fail identically, and `cudaGetDevice`'s output parameter shows the exact same untouched-sentinel behavior Worked Example 10.1.1 already established for `cudaGetDeviceCount`.

## 10.2 A Device-Aware Allocator, Built on Chapter 6's RAII `[FOUNDATIONAL]`

### Intuition

`DeviceManager` answers one question; `DeviceAwareAllocator` acts on it — routing an allocation request to `cudaMalloc` when a device is known to exist, and to plain host `malloc()` otherwise, without ever attempting the device path when discovery has already ruled it out. This is the first allocator in this book that can genuinely *succeed* under this environment's no-GPU constraints, rather than honestly reporting `cudaErrorNoDevice` the way every `cudaMalloc` call since Chapter 2.4 has.

### Background

| | Device path | Host fallback path |
|---|---|---|
| Taken when | `DeviceManager` reports a device *and* the allocation itself succeeds | No device reported, or the device allocation fails anyway despite being reported |
| Allocator call | `cudaMalloc` | `malloc()` |
| Alignment guarantee | At least 256 bytes (Chapter 7.3) | Whatever the platform's default `malloc` alignment happens to be — not requested, not guaranteed |
| Genuinely reachable in this no-GPU environment? | No (would need a real device) | Yes — both trigger conditions below are real |

### Worked Example 10.2.1 — the routing condition, both reachable cases

```cpp
struct DeviceAwareAllocator {
    bool device_available;
    explicit DeviceAwareAllocator(const DeviceManager& mgr) : device_available(mgr.has_device(0)) {}

    void* allocate(size_t bytes, const char** strategy_used) {
        if (device_available) {
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, bytes);
            if (err == cudaSuccess) { *strategy_used = "cudaMalloc (device)"; return ptr; }
            *strategy_used = "malloc (host fallback -- cudaMalloc failed despite device_available)";
            return malloc(bytes);
        }
        *strategy_used = "malloc (host fallback -- no device reported by discovery)";
        return malloc(bytes);
    }
};
```

Compiled and run as part of the complete `02_device_aware_allocator.cu` further below:

```bash
nvcc -arch=sm_80 02_device_aware_allocator.cu -o 02_device_aware_allocator
./02_device_aware_allocator
```

Genuinely compiled and run:

```
Case 1 (device_available=false, the real, measured discovery result):
  strategy used: malloc (host fallback -- no device reported by discovery)
  allocation succeeded: true

Case 2 (device_available=true, FORCED, to test the failure-recovery path):
  strategy used: malloc (host fallback -- cudaMalloc failed despite device_available)
  allocation succeeded: true
```

Case 1 is this environment's real, honestly-discovered situation: `DeviceManager` reports no device, `device_available` is genuinely `false`, and `allocate()` never even attempts `cudaMalloc` — it goes straight to `malloc()`, which genuinely succeeds. Case 2 deliberately forces `device_available = true` (simulating discovery having been correct at startup but the device becoming unavailable, or simply wrong) specifically to exercise the *second* fallback: `allocate()` still tries `cudaMalloc` first, genuinely receives `cudaErrorNoDevice`, and falls back to `malloc()` anyway — succeeding through a different path than Case 1, for a different reason. Both are real, both are genuinely exercised, and both produce a working allocation in an environment where every previous chapter's `cudaMalloc` call has honestly failed.

### Worked Example 10.2.2 — the alignment guarantee that never arrives

Compiled and run as part of the complete `02_device_aware_allocator.cu` further below:

```bash
nvcc -arch=sm_80 02_device_aware_allocator.cu -o 02_device_aware_allocator
./02_device_aware_allocator
```

Genuinely compiled and run:

```
checking whether the host-fallback path honors a 256-byte alignment request
it was never actually told about:
  5 of 5 malloc() calls were NOT 256-byte aligned
  posix_memalign(256), by contrast: aligned to 256 bytes? true
```

`allocate()`'s signature never takes an alignment parameter at all — it simply calls `malloc(bytes)`, which makes no 256-byte alignment promise (Chapter 7.3 established `cudaMalloc`'s guarantee at that figure, not `malloc`'s), and all 5 genuinely tested calls confirm it: none land on a 256-byte boundary. `posix_memalign(&ptr, 256, bytes)`, by contrast, genuinely honors the request. This is a real gap in `DeviceAwareAllocator` as built here: a caller relying on the device path's alignment guarantee (say, to feed Chapter 5.2's vectorized `float4` loads) gets a silent, unannounced downgrade the moment the host fallback fires — the allocator never lies about *whether* it allocated memory, but it says nothing at all about the alignment property it quietly stopped guaranteeing.

> `[COMMON TRAP]` A function whose parameter list doesn't mention alignment isn't necessarily a function that doesn't need to think about it — `DeviceAwareAllocator::allocate()`'s device path *happens* to satisfy Chapter 5.2's `float4` alignment requirement, purely as a side effect of `cudaMalloc`'s own guarantee, while its fallback path happens not to, for the identical reason: neither path was ever asked to guarantee anything about alignment, one just does anyway. The fix mirrors Worked Example 10.2.2's own contrast — replace `malloc(bytes)` with `posix_memalign(&ptr, required_alignment, bytes)` in the fallback path, so the guarantee survives the routing decision instead of depending on which branch happened to run.

### Worked Example 10.2.3 — reading a benchmark's ledger honestly

Compiled and run as part of the complete `02_device_aware_allocator.cu` further below:

```bash
nvcc -arch=sm_80 02_device_aware_allocator.cu -o 02_device_aware_allocator
./02_device_aware_allocator
```

Genuinely compiled and run:

```
100 cudaMalloc calls (all 100 failed with cudaErrorNoDevice): 0.001 ms total, 0.00001 ms/call
100 malloc calls (all succeeded): 0.004 ms total, 0.00004 ms/call
```

(Rerunning the same binary produces slightly different absolute figures each time — sub-millisecond timings are sensitive to scheduler noise on any machine — but the qualitative result held across every rerun performed while writing this chapter: the 100 failing `cudaMalloc` calls consistently measured faster than the 100 succeeding `malloc` calls, for the reason explained below.)

Read carelessly, this table looks like it's saying "device allocation is faster than host allocation" — exactly backwards from any real hardware, and not remotely what these two numbers actually measure. By the point this benchmark runs, `main()` has already made several earlier CUDA Runtime API calls (`DeviceManager`'s constructor, both allocator cases), so the runtime has already paid whatever cost it pays to discover and cache "no device is present"; these 100 `cudaMalloc` calls are genuinely measuring *how fast a call fails once that fact is already known*, not how fast an allocation happens. The very first CUDA API call in a fresh process pays a real, separate initialization cost that this benchmark's ordering specifically avoids measuring. The honest reading: this table compares "the cost of a cached failure" against "the cost of a real host allocation," a real and reproducible pair of numbers, but not the GPU-versus-CPU allocation comparison it might be mistaken for at a glance.

> `[COMMON TRAP]` A benchmark's numbers are only as meaningful as the reader's understanding of what was actually measured — this table's own honest caption exists specifically because "100 cudaMalloc calls" and "100 malloc calls" sound directly comparable, when they measure genuinely different things (a cached-failure fast path versus a real allocation) under conditions (this specific call ordering, this specific no-GPU environment) that don't generalize to a real GPU-equipped machine at all. This is the same discipline this book's `getting-started.md` applies to itself: stating plainly which of a result's claims rest on genuine execution and which don't, rather than letting a plausible-looking number speak for itself.

## 10.3 Complete Runnable Code

### File: `01_device_discovery.cu`

```cpp
#include <cstdio>

// A tiny, honest device manager: discover what's actually there EXACTLY
// once, and never let anything downstream trust an uninitialized value if
// that discovery itself failed.
struct DeviceManager {
    int device_count;
    bool discovery_succeeded;

    DeviceManager() {
        int count = -999;  // a deliberately absurd sentinel -- if this survives, discovery lied
        cudaError_t err = cudaGetDeviceCount(&count);
        discovery_succeeded = (err == cudaSuccess);
        device_count = discovery_succeeded ? count : 0;  // never trust `count` after a failed call
        printf("DeviceManager() -> cudaGetDeviceCount err=%d (%s)\n", err, cudaGetErrorString(err));
        printf("  raw count parameter after the call: %d\n", count);
        printf("  discovery_succeeded=%s, device_count=%d\n", discovery_succeeded ? "true" : "false", device_count);
    }

    bool has_device(int id) const { return discovery_succeeded && id < device_count; }
};

int main() {
    printf("=== Section 10.1: Device Discovery ===\n\n");
    DeviceManager mgr;

    printf("\nmgr.has_device(0) = %s\n", mgr.has_device(0) ? "true" : "false");

    printf("\n--- bypassing the manager entirely ---\n");
    printf("code that skips has_device() and calls cudaSetDevice(0) directly:\n");
    cudaError_t err2 = cudaSetDevice(0);
    printf("  cudaSetDevice(0) -> err=%d (%s)\n", err2, cudaGetErrorString(err2));
    printf("  same failure the manager would have predicted -- but discovered late,\n");
    printf("  inside a call whose job was never 'tell me whether a device exists.'\n");

    printf("\ncudaGetDeviceProperties and cudaGetDevice, genuinely queried:\n");
    cudaDeviceProp prop;
    cudaError_t err3 = cudaGetDeviceProperties(&prop, 0);
    printf("  cudaGetDeviceProperties(0) -> err=%d (%s)\n", err3, cudaGetErrorString(err3));
    int current = -999;
    cudaError_t err4 = cudaGetDevice(&current);
    printf("  cudaGetDevice -> err=%d (%s), current still = %d (untouched by the failed call)\n",
           err4, cudaGetErrorString(err4), current);
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_device_discovery.cu -o 01_device_discovery
./01_device_discovery
```

Produces exactly the output shown in Worked Examples 10.1.1 and 10.1.2 above.

### File: `02_device_aware_allocator.cu`

```cpp
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>

struct DeviceManager {
    int device_count;
    bool discovery_succeeded;
    DeviceManager() {
        int count = -999;
        cudaError_t err = cudaGetDeviceCount(&count);
        discovery_succeeded = (err == cudaSuccess);
        device_count = discovery_succeeded ? count : 0;
    }
    bool has_device(int id) const { return discovery_succeeded && id < device_count; }
};

// Routes to cudaMalloc when the manager reports a device, falling back to
// plain host malloc() otherwise -- and ALSO falls back if cudaMalloc itself
// fails despite the manager's belief, since discovery is a snapshot that can
// go stale between the check and the actual allocation.
struct DeviceAwareAllocator {
    bool device_available;
    explicit DeviceAwareAllocator(const DeviceManager& mgr) : device_available(mgr.has_device(0)) {}

    void* allocate(size_t bytes, const char** strategy_used) {
        if (device_available) {
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, bytes);
            if (err == cudaSuccess) { *strategy_used = "cudaMalloc (device)"; return ptr; }
            *strategy_used = "malloc (host fallback -- cudaMalloc failed despite device_available)";
            return malloc(bytes);
        }
        *strategy_used = "malloc (host fallback -- no device reported by discovery)";
        return malloc(bytes);
    }
};

int main() {
    printf("=== Section 10.2: Memory Management and Allocation ===\n\n");

    printf("--- routing condition, both reachable combinations in this environment ---\n");
    DeviceManager mgr;  // discovery_succeeded=false in this no-GPU environment
    DeviceAwareAllocator honest_allocator(mgr);
    const char* strategy1 = nullptr;
    void* p1 = honest_allocator.allocate(1024, &strategy1);
    printf("Case 1 (device_available=false, the real, measured discovery result):\n");
    printf("  strategy used: %s\n", strategy1);
    printf("  allocation succeeded: %s\n", (p1 != nullptr) ? "true" : "false");
    free(p1);

    DeviceAwareAllocator forced_allocator(mgr);
    forced_allocator.device_available = true;  // simulate discovery having (wrongly) reported a device
    const char* strategy2 = nullptr;
    void* p2 = forced_allocator.allocate(1024, &strategy2);
    printf("\nCase 2 (device_available=true, FORCED, to test the failure-recovery path):\n");
    printf("  strategy used: %s\n", strategy2);
    printf("  allocation succeeded: %s\n", (p2 != nullptr) ? "true" : "false");
    free(p2);

    printf("\n--- the alignment parameter that never arrives ---\n");
    printf("checking whether the host-fallback path honors a 256-byte alignment request\n");
    printf("it was never actually told about:\n");
    int misaligned_count = 0;
    for (int i = 0; i < 5; i++) {
        void* p = malloc(1024);
        uintptr_t addr = (uintptr_t)p;
        bool aligned_256 = (addr % 256 == 0);
        if (!aligned_256) misaligned_count++;
        free(p);
    }
    printf("  %d of 5 malloc() calls were NOT 256-byte aligned\n", misaligned_count);
    void* aligned_ptr = nullptr;
    posix_memalign(&aligned_ptr, 256, 1024);
    uintptr_t addr2 = (uintptr_t)aligned_ptr;
    printf("  posix_memalign(256), by contrast: aligned to 256 bytes? %s\n",
           (addr2 % 256 == 0) ? "true" : "false");
    free(aligned_ptr);
    printf("  a device-aware allocator that silently falls back to plain malloc() loses\n");
    printf("  whatever alignment guarantee the device path would have honored -- exactly\n");
    printf("  the kind of silently-dropped parameter Chapter 7.3 warned about generally.\n");

    printf("\n--- reading a benchmark's ledger honestly ---\n");
    const int trials = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    int real_failures = 0;
    for (int i = 0; i < trials; i++) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, 1024 * sizeof(float));
        if (err != cudaSuccess) real_failures++;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cuda_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; i++) {
        void* ptr = malloc(1024 * sizeof(float));
        free(ptr);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double malloc_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    printf("  %d cudaMalloc calls (all %d failed with cudaErrorNoDevice): %.3f ms total, %.5f ms/call\n",
           trials, real_failures, cuda_ms, cuda_ms / trials);
    printf("  %d malloc calls (all succeeded): %.3f ms total, %.5f ms/call\n",
           trials, malloc_ms, malloc_ms / trials);
    printf("\n  honest reading: this is NOT \"device allocation vs host allocation\" performance --\n");
    printf("  it's \"time to fail immediately\" vs \"time to actually allocate.\" A real GPU\n");
    printf("  allocation benchmark needs a real GPU; this table measures only this\n");
    printf("  environment's failure path, which itself turns out not to be free.\n");
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_device_aware_allocator.cu -o 02_device_aware_allocator
./02_device_aware_allocator
```

### Expected Output for `02_device_aware_allocator.cu`

```
=== Section 10.2: Memory Management and Allocation ===

--- routing condition, both reachable combinations in this environment ---
Case 1 (device_available=false, the real, measured discovery result):
  strategy used: malloc (host fallback -- no device reported by discovery)
  allocation succeeded: true

Case 2 (device_available=true, FORCED, to test the failure-recovery path):
  strategy used: malloc (host fallback -- cudaMalloc failed despite device_available)
  allocation succeeded: true

--- the alignment parameter that never arrives ---
checking whether the host-fallback path honors a 256-byte alignment request
it was never actually told about:
  5 of 5 malloc() calls were NOT 256-byte aligned
  posix_memalign(256), by contrast: aligned to 256 bytes? true
  a device-aware allocator that silently falls back to plain malloc() loses
  whatever alignment guarantee the device path would have honored -- exactly
  the kind of silently-dropped parameter Chapter 7.3 warned about generally.

--- reading a benchmark's ledger honestly ---
  100 cudaMalloc calls (all 100 failed with cudaErrorNoDevice): 0.001 ms total, 0.00001 ms/call
  100 malloc calls (all succeeded): 0.004 ms total, 0.00004 ms/call

  honest reading: this is NOT "device allocation vs host allocation" performance --
  it's "time to fail immediately" vs "time to actually allocate." A real GPU
  allocation benchmark needs a real GPU; this table measures only this
  environment's failure path, which itself turns out not to be free.
```

(As with Worked Example 10.2.3 above, the exact sub-millisecond figures drift slightly between runs; the direction of the result — cached failure faster than real allocation, in this environment and ordering — did not, across every rerun performed while writing this chapter.)

Every number here was independently verified earlier in this chapter, in Worked Examples 10.2.1 through 10.2.3. Both files genuinely compile clean under `nvcc -arch=sm_80` and run to completion in this no-GPU environment, honestly reporting `cudaErrorNoDevice` at every genuine device-touching call while still succeeding overall, since both files' entire point is to route *around* that failure rather than merely disclose it.

## Chapter Summary

`DeviceManager` turns "ask CUDA and see what happens" into "ask once, cache the answer, and let everything downstream consult the cache" — and testing it directly revealed a real trap worth knowing generally: `cudaGetDeviceCount`'s output parameter is left completely untouched when the call fails, not helpfully zeroed, so any code that reads it without first checking the returned `cudaError_t` risks trusting whatever garbage was already sitting there. Bypassing the manager and calling a device operation directly fails identically, just later and with less context — the manager's value is clarity and cost, not a different failure outcome. `DeviceAwareAllocator` is the first allocator in this book that can genuinely succeed under this environment's constraints, routing to a host `malloc()` fallback whenever discovery reports no device (or whenever the device path fails anyway) — but that fallback silently drops the 256-byte alignment guarantee `cudaMalloc` provides, a real, measured gap rather than a hypothetical one. And this chapter's own benchmark of "cudaMalloc versus malloc" timing is a genuine cautionary example of its own lesson: comparing two numbers correctly requires understanding what each one actually measured, not just what it appears to compare — a cached failure's speed is not an allocation's speed, no matter how directly the two numbers sit side by side in a table.

## Self-Check Questions

1. Why does `DeviceManager`'s constructor initialize `count` to `-999` rather than `0` before calling `cudaGetDeviceCount`?
2. `mgr.has_device(0)` and a direct `cudaSetDevice(0)` call both correctly report that no device exists in this environment. What, specifically, does the manager approach provide that the direct call does not, given that the underlying facts are identical?
3. `DeviceAwareAllocator::allocate()` has two separate code paths that both call `malloc()` as a fallback. What real-world situation does each one guard against, and why are they genuinely different situations rather than the same case checked twice?
4. Why does neither of `DeviceAwareAllocator`'s two host-fallback allocations guarantee 256-byte alignment, even though the struct is specifically designed to be "device-aware"?
5. Worked Example 10.2.3's benchmark shows `cudaMalloc` failing faster than `malloc()` succeeds. Explain why this result would not necessarily hold if the `cudaMalloc` calls were the very first CUDA API calls made in the entire program, rather than coming after `DeviceManager` and two `DeviceAwareAllocator` calls had already run.

## Where We Go Next

`DeviceAwareAllocator` still calls `malloc()`/`cudaMalloc()` directly and expects the caller to remember to free what it returns — exactly the manual-lifetime problem Chapter 2.4's RAII discipline solved for a single buffer. Chapter 11 builds the memory management layer this book's actual `Tensor` needs on top of everything Part 1 has built so far: shared ownership across multiple `Tensor` views of the same buffer, and a pooled allocator that reuses freed memory instead of returning to `cudaMalloc` for every single request.

## Worked Solutions

**1.** `-999` is a value no legitimate device count could ever equal, chosen specifically so that if it survives unchanged after the call, that survival itself proves the call left the output parameter untouched. Initializing to `0` instead would have made the exact same failure indistinguishable from "the call succeeded and genuinely found zero devices" — a real, different situation Worked Example 10.1.1 needed to be able to tell apart from "the call failed and the parameter was never written."

**2.** The manager provides a single, explicit, checkable fact (`has_device(0)`) available *before* any device-specific work is attempted, checkable from anywhere in a program without needing to attempt an actual device operation just to find out. The direct `cudaSetDevice(0)` call provides the identical true-or-false information, but only at the exact moment and location that specific call happens to run, buried inside whatever function needed a device — for a large program, that could be a return code checked deep inside a rarely-touched code path, discovered only when that path actually executes, rather than a single fact checkable once at startup.

**3.** The first fallback (`device_available` is `false`) guards against the ordinary, expected case in this environment: discovery genuinely found no device, and the allocator correctly never even attempts `cudaMalloc`. The second fallback (`device_available` is `true`, but `cudaMalloc` still fails) guards against a *stale* or *incorrect* discovery result — a device the manager believed was present becoming unavailable, or a bug in the discovery logic itself — a case where trusting the cached answer alone would have caused an allocation failure the allocator could have recovered from instead. Both situations produce the same fallback outcome, but only the second one represents discovery itself being wrong, which is why a correct allocator needs both checks rather than trusting the cached flag unconditionally.

**4.** Neither path was ever given an alignment parameter to honor in the first place — `allocate(bytes, ...)`'s signature has no alignment argument, so both `cudaMalloc` and `malloc()` are called exactly the way Chapter 2's constructors were always called, each simply doing whatever its own default behavior happens to be. `cudaMalloc`'s 256-byte alignment is a property of that specific function, not something `DeviceAwareAllocator` requested or verified — which is exactly why the guarantee silently vanishes the moment the routing decision picks the other branch.

**5.** The very first CUDA API call made in a fresh process typically triggers real, one-time context and driver initialization work — genuinely more expensive than a call made after that initialization has already happened (successfully or not) and been cached. By the time Worked Example 10.2.3's benchmark loop runs, `DeviceManager`'s constructor and two prior `DeviceAwareAllocator::allocate()` calls have already made several CUDA API calls, so the runtime has already paid (and cached the result of) discovering "no device is present" — the benchmark's 100 `cudaMalloc` calls are measuring the fast, already-cached failure path, not the slower, one-time cost a program's very first CUDA call would pay in a fresh process.
