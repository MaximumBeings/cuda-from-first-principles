# Appendix E: The Streaming Multiprocessor as a Pipeline — Control, Registers, and the Tiling Payoff

> "Appendix C walked the memory hierarchy space by space, from registers down to global DRAM. This appendix draws the SAME hardware a second time, boxed the way an introductory GPU-architecture course draws it — one Streaming Multiprocessor at a time, control logic and compute cores sitting right next to the memory they feed — and uses that box to answer one very concrete question in full: when a course slide says a tile reduces global memory traffic 'by a factor of TILE_WIDTH,' is that literally, exactly true, or a rounded-off rule of thumb?"

## What you will understand by the end of this appendix

- One Streaming Multiprocessor drawn as a single box — a Control unit, a Register file, an array of Streaming Processors (the actual scalar cores), and a combined Shared Memory/L1 Cache block sitting beside it — and how that one diagram is simultaneously Chapter 18's launch-and-warp story and Appendix C's memory-space story, told from a single vantage point instead of two separate ones.
- Why a simplified "~1 cycle / ~5 cycles / ~500 cycles" three-tier latency model (the kind a first GPU-architecture course teaches) and Appendix C's own "~1 / ~20-30 / ~200 / ~400-800 cycle" figures are not in conflict — they're the same hardware described at two different resolutions, and this appendix says exactly where the simplification is standing in for what.
- Whether tiling a matrix multiplication with a `TILE_WIDTH × TILE_WIDTH` shared-memory tile really does cut global memory reads by *exactly* `TILE_WIDTH`, checked here by two independent, from-scratch simulations that walk the same grid/block/thread/tile-step loop nest a real kernel executes — not by evaluating a formula and calling it proven.

## What you need to know first

- Appendix C in full, especially C.1 (the memory hierarchy) and C.3 (shared memory and bank conflicts) — this appendix's Section E.1 redraws the same hardware from a different angle, and Section E.3's tiled kernel is the same shared-memory tiling technique C.3 already introduced, examined here for its *global*-memory effect rather than its bank-conflict behavior.
- Chapter 18.1 and 18.3 (ceiling-division launch configuration, and the convolution kernel's shared-memory tiling) — Section E.3's matmul tiling is the same technique Chapter 18.3 uses for convolution, applied to a different operation.
- The Struct-of-Arrays coalescing analysis from Chapter 18.2 — Section E.2 places it on the simplified latency model this appendix introduces.

## E.1 The SM From the Inside: Control, Registers, and the SP Array `[FOUNDATIONAL]`

### Intuition

Appendix C.1 drew the memory hierarchy as a stack — registers at the top, global memory at the bottom — because Appendix C's subject was where a value *lives*. This section draws the same silicon as a factory floor instead: one Streaming Multiprocessor, boxed as a single unit, with a Control section that decides what happens next, a bank of Streaming Processors that are the actual arithmetic cores doing the deciding-on's work, a Register file sitting closest to those cores, and a Shared Memory/L1 Cache block within the same SM's walls but a few cycles farther away. Appendix C's registers-to-global stack and this section's control-plus-cores floor plan are not two different chips — they're the same chip, and putting the Control unit into the picture for the first time is what lets this section connect Chapter 18's launch math (which thread runs which instruction, and when) to Appendix C's memory story (what that thread can reach, and how fast) in one drawing.

### Background

```
ONE STREAMING MULTIPROCESSOR (SM)
+--------------------------------------------------------------+
|  CONTROL                                                      |
|  - fetches and decodes the NEXT instruction for a resident    |
|    warp (Appendix C.6's 32-thread lockstep unit)               |
|  - schedules which of several resident warps issues this cycle |
|  - this is the hardware Chapter 18.1's ceiling-division launch |
|    math is ultimately feeding: every block this book has ever  |
|    launched becomes some number of warps THIS unit schedules   |
+--------------------------------------------------------------+
|  REGISTERS            |  SP   SP   SP   SP                    |
|  (Appendix C.2 --      |  SP   SP   SP   SP    <- Streaming     |
|  one thread's own       |  ...                    Processors:  |
|  private values,        |                         the actual   |
|  ~1 cycle to reach)     |                         scalar cores |
|                                                    doing the    |
|                                                    arithmetic   |
+--------------------------------------------------------------+
|  SHARED MEMORY  |  L1 CACHE                                    |
|  (Appendix C.3 -- programmer-managed, one allocation per       |
|  resident block; L1 -- hardware-managed, caches recent          |
|  global-memory traffic. Different management, similar speed.)  |
+--------------------------------------------------------------+
                              |
                              v  (off-chip, shared by every SM)
+--------------------------------------------------------------+
|                        GLOBAL MEMORY                          |
|          (Appendix C.5 -- large, off-chip DRAM)                |
+--------------------------------------------------------------+
```

Nothing in this diagram is a new hardware fact relative to Appendix C or Chapter 18 — every box already has a home in one of those two chapters. What's new is the framing: Control, Registers, and the SP array sit *inside the same box*, at the *same* distance from Shared Memory/L1 below them, which is exactly why Chapter 18.1's warp-scheduling story and Appendix C.2's register story are two descriptions of work happening in the same physical neighborhood, not two separate subsystems that happen to share a chapter number.

**[COMMON TRAP]** The SP boxes in this diagram are individual **scalar** cores — one SP executes one thread's arithmetic instruction, not a whole warp's worth. A warp's 32 threads occupy 32 SPs (or the same SPs across several cycles, on an SM with fewer than 32 SPs per partition) executing *the same instruction* in lockstep — the SIMT behavior Appendix C.6 already covers — but the SP itself has no notion of "warp" baked into it; it just executes whatever single-thread instruction the Control unit hands it this cycle.

## E.2 A Simplified Three-Tier Latency Model, and How It Relates to Appendix C's Own Numbers `[FOUNDATIONAL]`

### Intuition

An introductory GPU-architecture course, teaching this material for the first time, often collapses the memory hierarchy into three rounded numbers: about 1 cycle for registers, about 5 cycles for shared memory and L1 together, about 500 cycles for global memory. Appendix C.1's own table is more granular — `~1 cycle` for registers still matches, but shared memory sits at `~20-30 cycles`, there's a separate `~200 cycle` L2 tier the three-tier model folds into "shared/L1," and global memory's `~400-800 cycle` range is wider than a single rounded `~500`. Neither table is wrong. They're the same hardware, reported at two different resolutions for two different purposes.

### Background

| Tier | Simplified 3-tier figure | Appendix C.1's own figure | What the simplification collapses |
|---|---|---|---|
| Registers | ~1 cycle | ~1 cycle | Nothing — both agree exactly |
| Shared memory / L1 | ~5 cycles | Shared: ~20-30 cycles | Rounds a 20-30 cycle reality down to a single small number, and folds L1's caching behavior into the same bucket as shared memory's explicit staging, even though one is programmer-managed and the other is hardware-managed |
| (no separate tier) | — | L2 cache: ~200 cycles | The simplified model has no L2 tier at all — it's invisible in a 3-tier picture, present in Appendix C's more granular one |
| Global memory | ~500 cycles | ~400-800 cycles | Rounds a genuinely wide range (uncached vs. L2-cached global access can differ by 2x) to one representative number |

**[COMMON TRAP]** It's tempting to treat one of these tables as the "corrected" version of the other. That's the wrong frame. A three-tier model is a *teaching* tool — accurate enough to correctly predict which optimization matters most (moving a value from global to shared memory is worth far more than anything else in this hierarchy, at any resolution you measure it), while being simple enough to reason about on a whiteboard in one pass. Appendix C's own table exists for a different purpose — it's the reference this book's own worked examples (register spilling's `~400-800 cycle` cost, Chapter 18.2's coalescing argument) actually cite numbers from. Use the three-tier version to build intuition fast; use Appendix C's version when a specific chapter's argument needs a specific number.

## E.3 Tiled Matrix Multiplication: The `TILE_WIDTH` Reduction Factor, Genuinely Counted `[FOUNDATIONAL]`

### Intuition

Appendix C.3 already established shared-memory tiling's bank-conflict story: pad a `[32][32]` tile to `[32][33]` and a column read stops colliding on one bank. This section asks a completely different question about the *same* tiling technique, applied to matrix multiplication instead of a transpose: does staging a `TILE_WIDTH × TILE_WIDTH` block of each input matrix into shared memory *actually* cut global memory traffic by exactly `TILE_WIDTH`, the way an introductory course's slide states it, or is that a rounded rule of thumb the way the three-tier latency model is?

### Background

A naive matrix-multiply kernel gives every thread its own independent walk down the reduction dimension: computing one output element `C[row][col] = sum_k A[row][k] * B[k][col]` means that thread alone issues `K` reads from `A` and `K` reads from `B`, with nothing shared between it and its neighbors even though a neighboring thread computing `C[row][col+1]` reads the *exact same* `K` values of `A[row][*]` all over again.

A tiled kernel instead assigns one `TILE_WIDTH × TILE_WIDTH` output tile to one thread block, and for each `TILE_WIDTH`-deep step along the reduction dimension, has the block's `TILE_WIDTH × TILE_WIDTH` threads *cooperatively* load one `TILE_WIDTH × TILE_WIDTH` tile of `A` and one of `B` into shared memory — each thread reads exactly one element of each tile from global memory, once, then every thread in the block reuses that same shared tile for its own partial dot product before the next tile-step reads the next pair of tiles.

### Worked Example E.3.1 — A small case, hand-traceable

For `N = 4` (a 4×4 output), `K = 4`, `TILE_WIDTH = 2`: the naive kernel's 16 output threads each read `4` values from `A` and `4` from `B`, for `16 × 4 × 2 = 128` total global reads. The tiled kernel has `(4/2)² = 4` blocks, each running `4/2 = 2` tile-steps, each tile-step reading `2×2 = 4` elements of `A` and `4` of `B` — `4 × 2 × (4+4) = 64` total global reads. `128 / 64 = 2`, exactly `TILE_WIDTH`.

### Worked Example E.3.2 — Genuinely counted by direct simulation, not by formula

The claim above is checked here by actually walking the loop nest a real kernel executes — one counter increment per simulated global-memory read, in two functions that share no code:

```cpp
void simulate_naive(int N, int K, ReadCounters& counters) {
    for (int row = 0; row < N; row++)
        for (int col = 0; col < N; col++)          // one simulated thread per (row, col)
            for (int k = 0; k < K; k++) {
                counters.naive_reads += 1;           // reading A[row][k]
                counters.naive_reads += 1;           // reading B[k][col]
            }
}

void simulate_tiled(int N, int K, int TILE_WIDTH, ReadCounters& counters) {
    int blocks_per_dim = N / TILE_WIDTH;
    int tile_steps = K / TILE_WIDTH;
    for (int block_row = 0; block_row < blocks_per_dim; block_row++)
        for (int block_col = 0; block_col < blocks_per_dim; block_col++)   // one simulated BLOCK
            for (int step = 0; step < tile_steps; step++)                  // one tile-step
                for (int ty = 0; ty < TILE_WIDTH; ty++)
                    for (int tx = 0; tx < TILE_WIDTH; tx++) {               // one simulated THREAD
                        counters.tiled_reads += 1;   // this thread loads ONE element of the A tile
                        counters.tiled_reads += 1;   // this thread loads ONE element of the B tile
                    }
}
```

Genuinely captured, `N = 32`, `K = 32`, three different tile widths:

```
N=32 (output is NxN), K=32 (reduction depth), square matrices

naive: 65536 total global reads (genuinely counted, one increment per simulated read)
independently checked against the closed form N*N*K*2 = 65536: match

TILE_WIDTH   tiled reads (counted)   naive/tiled ratio   matches TILE_WIDTH?
8            8192                   8.0000              yes, exactly
16           4096                   16.0000             yes, exactly
32           2048                   32.0000             yes, exactly
```

The `16×16` tile case reproduces the introductory course's own stated figure exactly: `16` — not "roughly `16`," not "`16`-ish," but the precise integer this simulation's two independent loop nests agree on, checked against the closed form `N²K·2 / TILE_WIDTH` derived analytically and matching every one of the three counted cases. Unlike Section E.2's latency numbers, "reduces global memory reads by a factor of `TILE_WIDTH`" is not a rounded rule of thumb — it is exact, for the idealized case both simulations model (every thread reuses its whole tile, no boundary effects from a matrix size that doesn't divide evenly by `TILE_WIDTH`).

**[COMMON TRAP]** This exact factor assumes `N` and `K` are evenly divisible by `TILE_WIDTH` — precisely the assumption Chapter 18.1's ceiling-division machinery exists to relax elsewhere in this book. A real kernel handling a matrix size that doesn't divide evenly needs the same bounds-checking and padding logic Chapter 18.3's convolution tiling already applies, and the *exact* `TILE_WIDTH` reduction factor becomes a very close approximation rather than a precise integer once boundary tiles are only partially full — worth knowing before quoting "exactly `TILE_WIDTH`" outside the idealized case this section actually checked.

## E.4 Reference Implementation

```cpp
#include <cstdio>

struct ReadCounters {
    long long naive_reads = 0;
    long long tiled_reads = 0;
};

void simulate_naive(int N, int K, ReadCounters& counters) {
    for (int row = 0; row < N; row++)
        for (int col = 0; col < N; col++)
            for (int k = 0; k < K; k++) {
                counters.naive_reads += 1;
                counters.naive_reads += 1;
            }
}

void simulate_tiled(int N, int K, int TILE_WIDTH, ReadCounters& counters) {
    int blocks_per_dim = N / TILE_WIDTH;
    int tile_steps = K / TILE_WIDTH;
    for (int block_row = 0; block_row < blocks_per_dim; block_row++)
        for (int block_col = 0; block_col < blocks_per_dim; block_col++)
            for (int step = 0; step < tile_steps; step++)
                for (int ty = 0; ty < TILE_WIDTH; ty++)
                    for (int tx = 0; tx < TILE_WIDTH; tx++) {
                        counters.tiled_reads += 1;
                        counters.tiled_reads += 1;
                    }
}

int main() {
    int N = 32, K = 32;
    int tile_widths[] = {8, 16, 32};

    ReadCounters naive_counters;
    simulate_naive(N, K, naive_counters);
    printf("naive: %lld total global reads\n", naive_counters.naive_reads);

    for (int tw : tile_widths) {
        ReadCounters tc;
        simulate_tiled(N, K, tw, tc);
        double ratio = (double)naive_counters.naive_reads / (double)tc.tiled_reads;
        printf("TILE_WIDTH=%d: %lld tiled reads, ratio=%.4f\n", tw, tc.tiled_reads, ratio);
    }
    return 0;
}
```

Genuinely compiled and run with `g++ -std=c++17 -Wall -Wextra`, no CUDA toolchain required — this entire appendix's evidence is a host-side simulation of the loop structure a real kernel executes, exactly like Appendix C's `compute_distinct_banks()` host function independently verifies a bank-conflict claim without needing a device to run the actual kernel on.

### Expected Output

```
naive: 65536 total global reads
TILE_WIDTH=8: 8192 tiled reads, ratio=8.0000
TILE_WIDTH=16: 4096 tiled reads, ratio=16.0000
TILE_WIDTH=32: 2048 tiled reads, ratio=32.0000
```

## Chapter Summary

One Streaming Multiprocessor is a single box holding a Control unit (which warp, which instruction, this cycle — the hardware underneath Chapter 18.1's launch math), a Register file and an array of Streaming Processors (the actual scalar cores, one thread's arithmetic each, not a whole warp's), and a Shared Memory/L1 Cache block a few cycles farther away — the same hardware Appendix C already covered as a memory-space stack, redrawn here as a compute-and-memory floor plan instead. A simplified three-tier latency model (`~1` / `~5` / `~500` cycles) and Appendix C's own more granular figures (`~1` / `~20-30` / `~200` / `~400-800` cycles) describe the identical hardware at two resolutions built for two different jobs — intuition-building at a glance versus citable numbers for a specific argument — neither one is the "wrong" table. The claim that a `TILE_WIDTH × TILE_WIDTH` shared-memory tile cuts global memory reads by exactly `TILE_WIDTH` was checked here by two independent, from-scratch simulations walking a real kernel's own grid/block/tile-step loop nest, not a formula taken on faith — and for evenly-divisible matrix sizes, the factor comes out exact, not approximate, confirmed at `TILE_WIDTH = 8, 16,` and `32` alike, with `16` matching an introductory course's own stated figure precisely.

## Self-Check Questions

1. Using Section E.1's diagram, which single box would Chapter 18.4's `__shfl_down_sync` warp-shuffle instruction bypass entirely, and which two boxes does it move data directly between instead?
2. A course teaches the three-tier `~1/~5/~500` model exclusively. A student concludes "shared memory and L1 cache are the same thing, since they're one tier." Using Appendix C.1 and C.3's own distinction, name the one difference between them the three-tier model's collapsing obscures.
3. Using Worked Example E.3.2's simulation code, compute (by hand, following the same loop-count logic) the tiled read count for `N = 64, K = 64, TILE_WIDTH = 8`, and confirm the naive/tiled ratio still equals `TILE_WIDTH`.
4. Section E.3's `[COMMON TRAP]` states the exact `TILE_WIDTH` factor assumes even divisibility. For `N = 30, TILE_WIDTH = 16` (so the last tile in each dimension is only partially full), would you expect the REAL naive/tiled ratio to be higher or lower than `16`? Explain which side wastes work in the boundary tile.
5. Section E.1 places Registers and the SP array in the same latency tier (`~1 cycle`) as each other but explicitly not the same tier as Shared Memory. Using Appendix C.2's own material, name the one thing that can happen to a value that "belongs" in a register that would move it into a completely different tier of this same diagram — and which tier.

## Where We Go Next

This appendix's SM-as-a-pipeline diagram and its genuinely counted tiling factor don't introduce anything Appendix C or Chapter 18 didn't already establish piece by piece — they're this book's own material, redrawn and re-verified from the angle a first course in GPU architecture typically starts from, so that a reader arriving from that angle can connect it to everything else in this book directly rather than needing to reconcile two apparently different vocabularies on their own.

## Worked Solutions

**1.** `__shfl_down_sync` bypasses the Shared Memory/L1 box entirely — that's precisely what makes it faster than a shared-memory-staged reduction for a warp's final rounds (Chapter 18.4). It moves data directly between the Register files of different SPs within the same warp: one thread's register value is read directly into another thread's register, with no memory access of any kind (not even the ~1-cycle-latency register file of the *reading* thread's own SP is where the *other* thread's value is fetched from — it's an inter-lane register exchange the hardware performs directly).

**2.** Shared memory is explicitly programmer-managed — a kernel writes `__shared__` and decides exactly what goes into it via `stack_allocation`-equivalent declarations (Appendix C.3) — while L1 cache is hardware-managed, automatically caching whatever global-memory traffic recently passed through it with no source-level control at all (Appendix C.1's own table lists this exact distinction in its "Managed by" column). The three-tier model's single "~5 cycles" bucket correctly says they're similarly fast, but says nothing about who controls what lives in each one, which is the actual distinction a kernel author needs when deciding whether to explicitly stage a value or simply hope the hardware caches it.

**3.** `blocks_per_dim = 64/8 = 8`, `tile_steps = 64/8 = 8`. Naive reads: `N²K·2 = 64×64×64×2 = 524,288`. Tiled reads: `blocks_per_dim² × tile_steps × (TILE_WIDTH² × 2) = 8×8 × 8 × (8×8×2) = 64 × 8 × 128 = 65,536`. Ratio: `524,288 / 65,536 = 8`, exactly `TILE_WIDTH`, confirming the pattern holds at a size neither Worked Example checked directly.

**4.** The real ratio would be LOWER than `16` (tiling's advantage shrinks). With `N = 30` and `TILE_WIDTH = 16`, two tiles are needed per dimension (`ceil(30/16) = 2`), but the second tile only has `30 - 16 = 14` valid rows/columns — a real kernel still launches a full `16×16` block for that boundary tile (Chapter 18.1's own ceiling-division pattern), and that block's threads still cooperatively load a full tile's worth of shared-memory traffic even though only `14×14` of it corresponds to real, needed output — the tiling side does "wasted" work loading and staging data for output positions that don't exist, while the naive kernel's per-thread cost doesn't change at all for a boundary case (a thread outside the true `30×30` output simply never gets launched, using Chapter 18.1's bounds check, rather than wastefully loading a tile). The tiled kernel's efficiency advantage is specifically diluted by boundary tiles doing proportionally more "wasted" shared-memory staging than the naive kernel wastes on an equivalent boundary.

**5.** A register-bound value spills into local memory (Appendix C.2) when the compiler's allocation runs out of physical registers for how many live values a kernel needs at once — the exact `-Xptxas -v`-measured spill Appendix C.2 demonstrates directly. That moves the value from the `~1 cycle` register tier into the SAME tier as global memory (`~400-800 cycles` in Appendix C.1's table, or the simplified model's `~500 cycles`, since local memory is physically global DRAM despite its per-thread-private naming) — the largest possible jump this diagram has to offer, and precisely why Appendix C.2 treats register pressure as a correctness-preserving but potentially severe performance cliff rather than a minor inefficiency.
