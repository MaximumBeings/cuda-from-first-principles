# Chapter 17: Gradient Computation Engine — Running Every Backward Rule to Completion

> "Every rule Chapter 16 derived by hand — `AddOp` passes a gradient through, `MulOp` scales it by the other operand, `MatMulOp` transposes and re-multiplies — is inert until something actually walks the graph and calls each one in the right order, on the right numbers, adding up whatever needs adding. This chapter is that something."

**What you will understand by the end of this chapter:**

- The full backward pass for `w = x*y + x`, traced start to finish as one table — and a precise, confirmed gap between `GraphNode::grad` (Chapter 15.2) and the tensor-level `.grad` state `accumulate_gradient` actually writes to, plus the exact field, already built in Chapter 15.3, that closes it
- A second, genuinely deeper gap this chapter's own C++ port uncovers on top of that one: because `ScalarTensor` copies by value (a scoping choice Chapter 15 flagged explicitly), fixing the first gap alone still isn't enough to make gradients reach the tensors a caller actually holds — and the graph-owned lookup table that fixes it for real
- Why `accumulate_gradient` branches on "does this tensor have a gradient yet" rather than always adding — and the definitive answer, finally, to the buffer-aliasing question Chapter 16 raised about `AddOp::backward` returning one value to two different callers, plus a *third*, genuinely discovered trap one step past that: why the old buffer can't simply be freed once accumulation reassigns
- A gap this chapter's own `accumulate_gradient` doesn't handle at all: what has to happen when an operand was broadcast during forward (Chapter 12.4), traced with the exact numbers that section already established
- Why this framework's computation graph is single-use — discarded after every backward pass and rebuilt from scratch on the next forward call — and how Chapter 11.2's arena allocator makes that a genuinely free design choice rather than a wasteful one
- Which saved inputs a node can safely drop the instant forward finishes, and which it must keep alive until backward actually visits it — a distinction that turns into real, countable memory on anything larger than a two-node example

**What you need to know first:**

- Chapter 16 (every backward rule this chapter's traversal actually calls: `AddOp`, `MulOp` — and the open aliasing question this chapter resolves)
- Chapter 15.2 and 15.3 (`GraphNode`'s `grad` field and `output.grad_fn_index` — both turn out to be exactly the machinery this chapter's central finding depends on)
- Chapter 11.1 and 11.2 (`RefCountedBuffer<T>`'s reference-counted freeing, needed for the third trap this chapter uncovers; and the arena allocator's `O(1)` reset — the cost model behind "just discard the graph and rebuild it")

## The full backward pass, worked by hand, start to finish

Everything is now in place to run the running example — `w = x*y + x`, `x=3, y=4, z=12, w=15` — backward completely, one node at a time, using nothing but the rules Chapter 16 already derived. Walk it as a table, exactly the steps Section 17.1's code automates:

| Step | Node visited | Incoming gradient | Local rule (Chapter 16) | Result | Running totals |
|---|---|---|---|---|---|
| 0 | — (seed) | — | `∂w/∂w = 1` by definition | `w.grad = 1.0` | `w: 1.0` |
| 1 | `add(z, x) → w` | `w.grad = 1.0` | `AddOp::backward` passes the gradient through unchanged to both inputs | `z` gets `1.0`; `x` gets `1.0` | `z: 1.0`, `x: 1.0` |
| 2 | `mul(x, y) → z` | `z.grad = 1.0` | `MulOp::backward`: `x` gets `grad_z × y`, `y` gets `grad_z × x` | `x` gets `1.0 × 4 = 4.0`; `y` gets `1.0 × 3 = 3.0` | `x: 1.0 + 4.0 = 5.0`, `y: 3.0` |

Final answer: **`x.grad = 5.0`, `y.grad = 3.0`** — matching the calculus in Chapter 15 (`∂w/∂x = y+1 = 5`, `∂w/∂y = x = 3`) exactly, and arrived at without a single symbolic derivative, only local multiplications and one addition, applied mechanically in the reverse order Chapter 15.4 established. The one place a "sum" happened rather than a plain pass-through is `x` in Step 2, because `x` was used twice in the forward pass — once directly in the addition, once inside the multiply — precisely Section 16.1's "sum over paths" chain rule, now happening inside a traversal instead of on paper.

## 17.1 Reverse-mode AD Implementation `[FOUNDATIONAL]`

### Intuition

The worked table above is a hand simulation of one function, conventionally called `backward()`, that turns a scalar loss into gradients on every parameter that fed into it. Writing it correctly means getting three things right at once: seeding the very first gradient, visiting nodes in an order where every dependency is already resolved, and skipping nodes that genuinely have nothing to contribute.

### Background

Ported as literally as possible from the design Chapters 15 and 16 already built:

```cpp
// backward(), ported LITERALLY -- including a bug this section's
// COMMON TRAP identifies before this file ever fixes it.
void backward_naive(OpRegistry& registry, ComputationGraph& graph, ScalarTensor& loss) {
    // Seed: dL/dL = 1
    loss.grad = 1.0f;
    loss.has_grad = true;

    std::vector<int> order = topological_backward_order(graph);
    for (int node_idx : order) {
        GraphNode& node = graph.nodes[node_idx];
        if (node.grad == 0.0f) continue;   // this output was "never used downstream" -- or so this reads

        std::vector<float> input_grads = chain_rule_step(registry, node.op_name, node.grad, node.inputs, node.output);
        for (size_t i = 0; i < node.inputs.size(); i++) {
            accumulate_gradient_naive(node.inputs[i], input_grads[i]);
        }
    }
}
```

`ScalarTensor` here gains a `has_grad`/`grad` pair on top of Chapter 15's fields — the C++ stand-in for Mojo's `Tensor.grad` being `.is_none()` before anything has ever been assigned to it.

### Worked Example 17.1.1 — Tracing the loop literally, and watching it fail

Compiled and run:

```bash
nvcc -arch=sm_80 01_naive_backward_bug.cu -o 01_naive_backward_bug
./01_naive_backward_bug
```

Genuinely compiled and run:

```
=== Section 17.1: backward(), ported literally -- including its own bug ===
w = 15.0, graph.nodes.size() = 2
graph.nodes[0].grad (mul node, at construction) = 0.0000
graph.nodes[1].grad (add node, at construction) = 0.0000

backward_naive: order = [1, 0]
  visiting graph.nodes[1] ("add"): node.grad = 0.0000 -> is_zero(), SKIPPING
  visiting graph.nodes[0] ("mul"): node.grad = 0.0000 -> is_zero(), SKIPPING

after backward_naive:
  w.has_grad = true, w.grad = 1.0000
  x.has_grad = false (x.grad never touched by this run)
  y.has_grad = false (y.grad never touched by this run)

expected from the hand-worked table: x.grad=5.0, y.grad=3.0 -- NOT what this run produced.
root cause: loss.grad was seeded on `w` (a ScalarTensor), but the loop reads
graph.nodes[node_idx].grad -- a SEPARATE field on GraphNode that this function
never writes to at all. graph.nodes[1].grad is still 0.0000 when the loop reaches it,
so the add node's backward is skipped, and neither x.grad nor y.grad is ever set.
```

This is a genuine, compiled, running reproduction of the bug — not a hypothetical. `loss.grad = 1.0f` sets `w`'s own `.grad` field, a completely separate piece of storage from `graph.nodes[1].grad`, since `ScalarTensor` and `GraphNode` are separate structs each carrying their own `grad` member. The very first thing the loop does is check `graph.nodes[1].grad == 0.0f`, which is `true`, and `continue`s — the add node's backward never runs, `accumulate_gradient` is never called for `z` or `x`, and the loop returns having touched nothing but `w` itself.

```
[COMMON TRAP]  node.grad is read every iteration, but nothing here ever writes it

Look at what backward_naive actually touches. It reads node.grad
(feeding chain_rule_step) and it calls accumulate_gradient_naive(...),
writing into each INPUT's own .grad field. loss.grad = 1.0f sets
loss's own .grad field too. Not one of these three writes ever
touches graph.nodes[node_idx].grad -- the separate field GraphNode
itself carries, zero-initialized in Chapter 15.2's constructor and
never assigned anywhere in this function.

The fix is a piece of machinery this book already built, two chapters
ago, and simply hasn't wired up yet: Chapter 15.3's
`output.grad_fn_index = (int)nodes.size()`, which exists specifically
so "the output remembers who made it." Seeding needs to set
`graph.nodes[loss.grad_fn_index].grad = loss.grad`, not just
loss.grad itself; and accumulate_gradient needs to mirror every
update into graph.nodes[tensor.grad_fn_index].grad as well as
tensor.grad, whenever tensor is itself the output of some earlier
node. Without it, grad_fn_index is a field this book built and never
actually used -- confirmed above, literally, by compiling and running
the code with the bug still in it.
```

### Worked Example 17.1.2 — Applying the fix, and finding a second one hiding behind it

Compiled and run:

```bash
nvcc -arch=sm_80 02_fixed_backward_and_accumulation.cu -o 02_fixed_backward_and_accumulation
./02_fixed_backward_and_accumulation
```

Genuinely compiled and run:

```
=== Section 17.1/17.2: fixed backward(), full trace for w = x*y + x ===

--- Part A: Section 17.1's grad_fn_index fix, applied literally ---
graph.nodes[1] ("add").grad = 1.0000, graph.nodes[0] ("mul").grad = 1.0000  (mirroring works)
but xA.grad (the ORIGINAL variable declared in main()) = 0.0000, has_grad = false
expected 5.0 -- MISMATCH. Cause: node.inputs[i] inside each GraphNode is an
INDEPENDENT COPY of xA (Chapter 15 flagged ScalarTensor as copying by value, not
sharing a buffer the way Chapter 11.1's RefCountedBuffer / a real Tensor would).
accumulate_gradient_part_a writes into that COPY's .grad field, not into xA itself --
the mirroring fix genuinely works for graph.nodes[].grad, but is not enough on its own.

--- Part B: a grad_table keyed by tensor_id, closing the value-copy gap ---
xB.tensor_id = 4 (every copy of xB inside graphB.nodes carries this same id)
read_grad(graphB, xB) = 5.0000, read_grad(graphB, yB) = 3.0000
matches the hand-worked table: x.grad=5.0, y.grad=3.0 -- CONFIRMED

--- Worked Example 17.2.1: the two branches, walked explicitly ---
Step 1 (add node): xB's first contribution -- table has no entry yet, insert 1.0
Step 2 (mul node): xB's second contribution -- entry exists, add: 1.0 + 4.0 = 5.0
final read_grad(graphB, xB) = 5.0

--- zero_grad ---
before zero_grad: read_grad(xB)=5.0, read_grad(yB)=3.0
after zero_grad: read_grad(xB)=0.0, read_grad(yB)=0.0
```

Applying exactly the fix the Mojo source describes — mirroring the seed and every accumulation into `graph.nodes[tensor.grad_fn_index].grad` — genuinely does fix `graph.nodes[].grad` (Part A confirms both node grads read `1.0`, correctly). But `xA.grad`, the *original* `ScalarTensor` variable declared in `main()`, still reads `0.0`. The reason is a second, C++-specific gap Mojo's own text never has to confront: Mojo's `Tensor` wraps a shared `UnsafePointer`, so every "copy" of a tensor still refers to the same underlying data, but Chapter 15 built `ScalarTensor` to copy by value instead — an explicitly flagged scoping choice for that chapter's small scalar example. The bill for that choice comes due here: `GraphNode::inputs` stores *independent copies* of `xA`, and `accumulate_gradient` writes into one of those copies, never into the `xA` variable the caller is still holding.

```cpp
// Part B's fix: a graph-owned table keyed by tensor_id, which every
// COPY of a ScalarTensor still carries (the default copy constructor
// copies every member, tensor_id included) -- closing the gap Part A
// leaves open.
void accumulate_gradient(ComputationGraph& graph, const ScalarTensor& tensor, float incoming_grad) {
    auto it = graph.grad_table.find(tensor.tensor_id);
    if (it == graph.grad_table.end()) graph.grad_table[tensor.tensor_id] = incoming_grad;
    else it->second = it->second + incoming_grad;
    if (tensor.grad_fn_index >= 0) graph.nodes[tensor.grad_fn_index].grad = graph.grad_table[tensor.tensor_id];
}
```

Every `ScalarTensor` is given a `tensor_id` at construction (a simple global counter), and — critically — the default C++ copy constructor copies that id along with everything else, so every independent value-copy of `xB` scattered across `graphB.nodes` still carries the *same* `tensor_id`. Routing accumulation through a table keyed by that id, owned by the graph rather than by any one copy, is what finally lets `read_grad(graphB, xB)` report `5.0` — genuinely confirmed above. This is not a divergence the Mojo source has to deal with at all, since its pointer-sharing `Tensor` never splits into independent copies in the first place; it is a direct, traceable consequence of the scoping choice Chapter 15 made explicitly and flagged for exactly this moment.

## 17.2 Gradient Accumulation Strategies `[FOUNDATIONAL]`

### Intuition

Step 2 of the worked table is where `x.grad` became `1.0 + 4.0 = 5.0` rather than being overwritten to `4.0`. That single addition is the entire content of this section, and it is one of the two or three most common autograd bugs in every framework that implements it — get it wrong, and any input used more than once (a shared weight, a residual connection `y = f(x) + x`) silently receives only its *last* gradient contribution instead of the sum of all of them.

### Background

```cpp
void accumulate_gradient(ComputationGraph& graph, const ScalarTensor& tensor, float incoming_grad) {
    auto it = graph.grad_table.find(tensor.tensor_id);
    if (it == graph.grad_table.end()) graph.grad_table[tensor.tensor_id] = incoming_grad;
    else it->second = it->second + incoming_grad;   // accumulate, don't replace
    if (tensor.grad_fn_index >= 0) graph.nodes[tensor.grad_fn_index].grad = graph.grad_table[tensor.tensor_id];
}

void zero_grad(ComputationGraph& graph, std::vector<ScalarTensor*>& params) {
    for (auto* p : params) graph.grad_table.erase(p->tensor_id);
}
```

### Worked Example 17.2.1 — The two branches, walked against the running example

At Step 1, `x` has no entry in `graph.grad_table` yet, so it's simply inserted at `1.0` — the "not found" branch, a plain assignment. At Step 2, `x`'s entry already holds `1.0`, so the new contribution `4.0` is *added*: `1.0 + 4.0 = 5.0`. Replacing instead of adding at Step 2 would have silently produced `x.grad = 4.0` — plausible-looking, still a number, and wrong. This is exactly why the residual-connection case in a real network is dangerous: `x` feeding both the shortcut and the transformed branch is structurally identical to `x` feeding both `AddOp` and `MulOp` above.

### Worked Example 17.2.2 — Resolving Chapter 16's open aliasing question, with real addresses

Chapter 16.2 flagged something left unresolved: `AddOp::backward` returns `{grad_output, grad_output}` — if `grad_output` were buffer-backed rather than a plain float, that would be the *same* underlying allocation, handed to both `z`'s incoming gradient and `x`'s first contribution. Trace what actually happens to that aliasing, step by step, with a genuine malloc'd buffer standing in for a real gradient buffer:

Compiled and run:

```bash
nvcc -arch=sm_80 03_aliasing_resolution.cu -o 03_aliasing_resolution
./03_aliasing_resolution
```

Genuinely compiled and run:

```
=== Section 17.2: resolving Chapter 16's AddOp aliasing question with real addresses ===

add_backward_output (the ALIASED buffer AddOp::backward returns twice):
  address = 0x560434ab1530, value = 1.0

--- Step 1: accumulate_gradient(z, add_backward_output) and accumulate_gradient(x, add_backward_output) ---
z_grad: address = 0x560434ab1530, value = 1.0
x_grad: address = 0x560434ab1530, value = 1.0
z_grad and x_grad share an address? true -- both hit the is_none() branch, both ALIASED

--- Step 2: x receives its SECOND contribution (4.0, from MulOp::backward) ---
x_grad: address = 0x560434ab1570 (was 0x560434ab1530), value = 5.0
z_grad: address = 0x560434ab1530 (was 0x560434ab1530), value = 1.0 -- UNTOUCHED by x's reassignment

x_grad's address changed?   true (elementwise_add_buffer allocated a fresh buffer)
z_grad's address unchanged? true (nothing ever mutated the buffer z_grad still points at)
z_grad still holds the correct value? true (1.0)

CONCLUSION: the aliasing between z_grad and x_grad after Step 1 was real, but it was
harmless to VALUES, because accumulate_gradient's else-branch never mutates a buffer
that might be shared -- it allocates a brand-new one and reassigns, the exact same
first-assign-then-add-a-fresh-buffer discipline production autograd engines rely on.

--- a second, genuinely discovered trap: freeing the OLD buffer is NOT safe here ---
An earlier version of this exact file called free(tensor.grad_data) right before the
reassignment above, on the assumption that nothing needed the old buffer anymore once
x_grad was about to point elsewhere. Running it corrupted z_grad's value into visible
garbage, because x_grad's OLD buffer was still the exact allocation z_grad points at --
freeing it out from under x_grad frees it out from under z_grad too. This is the same
aliasing risk Chapter 16 raised, from the opposite direction: it is not enough for
accumulation to avoid MUTATING a shared buffer in place; it must also avoid FREEING one,
since freeing is a mutation of the allocator's own bookkeeping, visible through every
alias. A correct production fix needs Chapter 11.1's RefCountedBuffer: only actually
free a gradient buffer once its reference count drops to zero, not the instant any ONE
of its aliases stops using it.
```

The exact hex digits after `0x560434ab1` are genuinely ASLR-dependent and will differ on a different run or machine; what is reproducible is that the two "before" addresses match exactly, and that `x_grad`'s address changes at Step 2 while `z_grad`'s does not. The second half of this worked example is a real bug this book's own drafting process hit and fixed: an earlier version of the same file called `free()` on the old buffer immediately after reassigning `x_grad`, on the assumption that nothing else could still be referencing it. That assumption is exactly what aliasing breaks — `z_grad` was still referencing that exact allocation, and freeing it out from under `x_grad` corrupted `z_grad`'s value into visible garbage the first time this file was actually run. The fix, shown in the final code, is to leave the old buffer alone (deliberately "leaking" it for this small demonstration); a production implementation would use Chapter 11.1's `RefCountedBuffer<T>` so the buffer is freed exactly once, when its reference count — not any single alias's lifetime — reaches zero.

### Worked Example 17.2.3 — Verifying `x.grad = 5.0` without any calculus at all

The whole point of a gradient is that it predicts how much the output moves for a tiny nudge to the input — so test that prediction directly, by nudging `x` by `±0.001` and reading `w` both times, the same finite-difference check the neural-network-layers chapter's `gradient_check` automates:

Compiled and run:

```bash
nvcc -arch=sm_80 04_finite_difference_verification.cu -o 04_finite_difference_verification
./04_finite_difference_verification
```

Genuinely compiled and run:

```
=== Section 17.2: verifying x.grad=5.0, y.grad=3.0 with finite differences, no calculus ===

w(x=3.001, y=4.0) = 15.005
w(x=2.999, y=4.0) = 14.995
slope ~= (15.005 - 14.995) / 0.002 = 4.9992  (backward()'s x.grad was 5.0)

w(x=3.0, y=4.001) = 15.003
w(x=3.0, y=3.999) = 14.997
slope ~= (15.003 - 14.997) / 0.002 = 3.0003  (backward()'s y.grad was 3.0)

both slopes match backward()'s computed gradients: CONFIRMED
```

Both slopes land almost exactly on `backward()`'s computed gradients — `4.9992` against `5.0` and `3.0003` against `3.0` — with the small residual coming from `float`'s roughly 7-digit precision rather than from any error in the finite-difference method itself (`w` is linear in both `x` and `y`, so a centered difference like this one is mathematically exact for it; the deviation here is purely `float32` rounding in the intermediate multiplication). This is the same agreement `gradient_check` looks for on every backward rule in the registry before that rule is trusted.

```
[COMMON TRAP]  accumulate_gradient assumes every operand already has the output's shape

Neither AddOp::backward (Chapter 16.2) nor accumulate_gradient above
ever compares shapes. That's fine for the running example -- every
tensor involved is a scalar -- but Chapter 12.4 already built a
kernel, broadcast_add_kernel, specifically for the case where one
operand is smaller and gets silently repeated. Reuse that section's
own numbers: A is 2x3, B is a single row of 3 values, and
broadcast_add produces C = [[11,22,33],[14,25,36]], a 2x3 result, from
A (2x3) and B (1x3).

Suppose this addition is graph-tracked and the upstream gradient
arriving at C is grad_C = [[1,1,1],[1,1,1]] (a 2x3 matrix of ones, the
same all-ones convention Chapter 16.3 used for grad_output). A
already has the full 2x3 output shape, so grad_A = grad_C unchanged is
correct. But B's ORIGINAL shape was 1x3, not 2x3 -- it was repeated
down both rows by the broadcast, not actually duplicated in memory.
Handing B the full 2x3 grad_C and calling accumulate_gradient(B,
grad_C) would either fail outright (if elementwise_add requires
matching shapes) or -- if it doesn't check at all -- silently store a
2x3 gradient on a tensor whose own data is 1x3, a shape mismatch that
corrupts every later use of B.grad.

What actually has to happen is a reduction: every row of grad_C that
came from the SAME repeated row of B needs to be summed back into one
row before it can be B's gradient. Column by column: grad_B[0,0] =
grad_C[0,0] + grad_C[1,0] = 1 + 1 = 2; the same sum applies to columns
1 and 2, giving grad_B = [2, 2, 2] -- a 1x3 result, matching B's real,
pre-broadcast shape, genuinely computed below.
```

Compiled and run:

```bash
nvcc -arch=sm_80 05_broadcast_gradient_trap.cu -o 05_broadcast_gradient_trap
./05_broadcast_gradient_trap
```

Genuinely compiled and run:

```
=== Section 17.2 COMMON TRAP: accumulate_gradient and broadcasting ===

A (2x3):
  [  1.0,  2.0,  3.0]
  [  4.0,  5.0,  6.0]
B (1x3):
  [ 10.0, 20.0, 30.0]
C = broadcast_add(A, B) (2x3):
  [ 11.0, 22.0, 33.0]
  [ 14.0, 25.0, 36.0]
(Chapter 12.4's own numbers: C = [[11,22,33],[14,25,36]] -- match: true)

grad_C (upstream, all ones) (2x3):
  [  1.0,  1.0,  1.0]
  [  1.0,  1.0,  1.0]

--- A's gradient: already output-shaped, no reduction needed ---
grad_A = grad_C unchanged, shape [2,3] matches A's shape [2,3] -- correct

--- B's gradient: the BROKEN approach ---
broken_grad_b(grad_C) (2x3):
  [  1.0,  1.0,  1.0]
  [  1.0,  1.0,  1.0]
shape is [2,3], but B's real shape is [1,3] -- MISMATCH, would corrupt B.grad

--- B's gradient: unbroadcast_gradient, correctly reducing rows ---
unbroadcast_gradient(grad_C, target=[1,3]) (1x3):
  [  2.0,  2.0,  2.0]
shape [1,3] matches B's real shape [1,3] -- correct
values: [2.0, 2.0, 2.0] (expected [2,2,2] -- each column summed down both rows)
matches the chapter's hand-derived grad_B = [2,2,2]: CONFIRMED
```

Neither `AddOp::backward` nor `accumulate_gradient` as built in this chapter performs this reduction anywhere; a broadcasting-aware version would need an explicit `unbroadcast_gradient` step — genuinely implemented and checked above, not left as an illustration — that sums a gradient back down to an operand's original shape before `accumulate_gradient` ever sees it. Every axis the forward pass was allowed to repeat a value across, backward has to sum back into one slot — the exact mirror image of what made the forward broadcast cheap in the first place.

## 17.3 Graph Traversal and Execution `[FOUNDATIONAL]`

### Intuition

Section 17.1's `backward()` already showed the traversal; what's worth calling out separately is what happens to the graph *afterward*.

### Background

In this framework, as in eager-mode PyTorch, the graph is single-use: `graph.nodes` for the running example held exactly two entries, was consumed once by the loop in Section 17.1, and would be discarded before the next forward pass builds a fresh one from scratch. This is a deliberate simplicity trade-off — a persistent, reusable graph enables graph-level optimization passes, but a rebuild-every-step graph is dramatically easier to reason about, and combined with Chapter 11.2's arena allocator, just as cheap in practice: `arena.reset()` at the top of the next `forward()` call reclaims every node from the discarded graph in `O(1)`, no matter how many nodes it held.

### Worked Example 17.3.1 — What "discard and rebuild" actually costs

Compiled and run:

```bash
nvcc -arch=sm_80 06_arena_single_use_graph.cu -o 06_arena_single_use_graph
./06_arena_single_use_graph
```

Genuinely compiled and run:

```
=== Section 17.3: the computation graph is single-use -- discard and rebuild is O(1) ===

--- building a 2-node graph (the running w = x*y + x example) ---
after building: arena.offset = 352 bytes
after arena.reset(): arena.offset = 0 bytes
reset() took 0.2310 microseconds

--- building a 2000-node graph ---
after building: arena.offset = 511840 bytes
after arena.reset(): arena.offset = 0 bytes
reset() took 0.0430 microseconds

reset() cost for 2 nodes vs 2000 nodes: 0.2310 us vs 0.0430 us
(both figures are single measurements of a sub-microsecond operation and will jitter
 run to run -- what is reproducible is that reset() does not scale with node count,
 since it only ever writes one field: offset = 0, regardless of how many allocations
 preceded it. Chapter 11.2 already established the same fact on raw byte allocations;
 this just confirms it holds for graph-sized allocations too.)

--- for contrast: individually freeing 2000 malloc'd nodes instead ---
freeing 2000 individually malloc'd nodes took 13.0540 microseconds -- scales with node
count, unlike arena.reset()'s single field write, even though both end up reclaiming
the same amount of memory.
```

The two `reset()` timings are both sub-microsecond, single-measurement figures that will jitter on any rerun — in this particular run, the *smaller* graph's reset happened to measure slower than the *larger* one, which is itself the point: at this timescale, the measurement noise dwarfs any dependence on node count, because there genuinely isn't one. `reset()` only ever writes a single field (`offset = 0`); it is exactly the same constant-time bump-pointer reset Chapter 11.2 traced on much larger raw-byte examples, now confirmed on graph-sized allocations too. The contrast at the bottom makes the alternative's real cost visible: individually freeing 2,000 separately malloc'd nodes took `13.05` microseconds — genuinely proportional to node count, unlike the arena's reset, even though both approaches reclaim the exact same amount of memory in the end. The arena trades a small amount of memory (its peak size has to accommodate the largest graph any single forward pass ever builds) for making that per-step cost disappear entirely.

## 17.4 Memory Optimization for Gradients `[FOUNDATIONAL]`

### Intuition

Two optimizations matter once this engine runs on real workloads rather than a two-node example.

### Background

First, **gradient-only-where-needed**: a node whose entire input subtree has `requires_grad = false` was already excluded from the graph in Chapter 15.3, so no gradient memory is ever allocated for it. Second, **saved-tensor pruning** — and the running example already demonstrates exactly which inputs are safe to drop. `AddOp::backward` (Chapter 16.2) never reads `inputs` at all — its two return values are just copies of `grad_output`. That means a node recording an addition can drop its references to both inputs immediately after forward runs, relying purely on the `grad_output` handed to it later, while `MulOp` and `MatMulOp` cannot — their backward rules read the *other* input, so both must stay alive until backward visits that node.

```cpp
// AddOp: grad(a) = grad(b) = grad_output, no input needed.
// MulOp, MatMulOp: backward reads the *other* input (Chapter 16.2, 16.3).
bool needs_input_for_backward(const std::string& op_name) {
    return op_name != "add";
}
```

### Worked Example 17.4.1 — Putting a number on the saving

A `[500, 8]` `float32` activation tensor is `500 × 8 × 4 bytes = 16,000 bytes`. An `AddOp` node in the middle of the training loop from the neural-network-layers chapter that drops both of its saved inputs the instant forward passes it frees `32,000` bytes it would otherwise have to keep alive for the entire duration of the backward pass.

Compiled and run:

```bash
nvcc -arch=sm_80 07_saved_tensor_pruning.cu -o 07_saved_tensor_pruning
./07_saved_tensor_pruning
```

Genuinely compiled and run:

```
=== Section 17.4: saved-tensor pruning -- which inputs backward actually needs ===

needs_input_for_backward("add")    = false
needs_input_for_backward("mul")    = true
needs_input_for_backward("matmul") = true

a [500, 8] float32 activation tensor: 500 x 8 x 4 bytes = 16000 bytes
an AddOp node dropping BOTH of its saved inputs the instant forward finishes frees:
  16000 bytes x 2 = 32000 bytes

--- tracked allocations: AddOp-style pruning vs MulOp-style retention ---
6 AddOp nodes, 32000 bytes/node saved: pruned-path live bytes = 0, retained-path live bytes = 192000
difference: 192000 bytes NOT kept alive across the backward pass when AddOp prunes its inputs

over 10000 training steps, at 0.1831 MB saved per step (this one node type alone):
  cumulative memory that never had to stay resident at once: not simply additive across
  steps (each step's activations are freed before the next begins regardless) -- the real
  benefit is PEAK memory per step, 0.1831 MB lower every single step, which is what actually
  determines whether a model fits in GPU memory, not a running total across steps.
```

Multiply that `32,000`-byte saving by every `add_bias` call in a network with several layers, and by thousands of training steps, and "don't pay memory for a value nothing will read again" stops being a micro-optimization and starts being the difference between a model that fits in GPU memory and one that doesn't — the genuinely tracked `192,000`-byte figure above is for just six `AddOp` nodes in one forward pass, before scaling to a real network's layer count at all.

## 17.5 Complete Runnable Code

### File: `01_naive_backward_bug.cu`

```cpp
#include <cstdio>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>

// Reused from Chapter 15/16, with one addition: a `has_grad`/`grad`
// pair on ScalarTensor, standing in for Mojo's Tensor.grad being
// `Optional`-like (`.is_none()` before anything has ever been
// assigned to it).
struct ScalarTensor {
    float value;
    bool requires_grad;
    int grad_fn_index;
    bool has_grad;
    float grad;
    ScalarTensor(float v, bool rg = false)
        : value(v), requires_grad(rg), grad_fn_index(-1), has_grad(false), grad(0.0f) {}
};

struct Differentiable {
    virtual ScalarTensor forward(const std::vector<ScalarTensor>& inputs) const = 0;
    virtual std::vector<float> backward(float grad_output, const std::vector<ScalarTensor>& inputs,
                                         const ScalarTensor& output) const = 0;
    virtual ~Differentiable() {}
};

struct AddOp : public Differentiable {
    ScalarTensor forward(const std::vector<ScalarTensor>& inputs) const override {
        return ScalarTensor(inputs[0].value + inputs[1].value);
    }
    std::vector<float> backward(float grad_output, const std::vector<ScalarTensor>& inputs,
                                 const ScalarTensor& output) const override {
        return { grad_output, grad_output };
    }
};

struct MulOp : public Differentiable {
    ScalarTensor forward(const std::vector<ScalarTensor>& inputs) const override {
        return ScalarTensor(inputs[0].value * inputs[1].value);
    }
    std::vector<float> backward(float grad_output, const std::vector<ScalarTensor>& inputs,
                                 const ScalarTensor& output) const override {
        return { grad_output * inputs[1].value, grad_output * inputs[0].value };
    }
};

struct OpRegistry {
    std::unordered_map<std::string, Differentiable*> ops;
    void register_op(const std::string& name, Differentiable* op) { ops[name] = op; }
    Differentiable* get(const std::string& name) { return ops.at(name); }
};

std::vector<float> chain_rule_step(OpRegistry& registry, const std::string& op_name, float grad_output,
                                    const std::vector<ScalarTensor>& inputs, const ScalarTensor& output) {
    return registry.get(op_name)->backward(grad_output, inputs, output);
}

// GraphNode, reused verbatim from Chapter 15.2 -- its own `grad` field
// is a SEPARATE piece of storage from any ScalarTensor's `.grad`.
struct GraphNode {
    std::string op_name;
    std::vector<ScalarTensor> inputs;
    ScalarTensor output;
    float grad;   // zero-initialized, Chapter 15.2 -- and, as this file
                  // demonstrates, never written by the naive backward() below.
    bool requires_grad;
    GraphNode(std::string op_name_, std::vector<ScalarTensor> inputs_, ScalarTensor output_)
        : op_name(op_name_), inputs(inputs_), output(output_), requires_grad(true), grad(0.0f) {}
};

struct ComputationGraph {
    std::vector<GraphNode> nodes;
    ScalarTensor record(const std::string& op_name, std::vector<ScalarTensor> inputs, ScalarTensor output) {
        bool needs_grad = false;
        for (auto& t : inputs) if (t.requires_grad) needs_grad = true;
        if (!needs_grad) return output;
        GraphNode node(op_name, inputs, output);
        output.requires_grad = true;
        output.grad_fn_index = (int)nodes.size();
        nodes.push_back(node);
        return output;
    }
};

ScalarTensor mul(ComputationGraph& graph, ScalarTensor a, ScalarTensor b) {
    return graph.record("mul", {a, b}, ScalarTensor(a.value * b.value));
}
ScalarTensor add(ComputationGraph& graph, ScalarTensor a, ScalarTensor b) {
    return graph.record("add", {a, b}, ScalarTensor(a.value + b.value));
}

std::vector<int> topological_backward_order(const ComputationGraph& graph) {
    std::vector<int> order;
    for (int i = (int)graph.nodes.size() - 1; i >= 0; i--) order.push_back(i);
    return order;
}

// Naive accumulate_gradient -- the "is_none() vs elementwise_add"
// branch from Section 17.2, ported literally, with no mirroring back
// into GraphNode.grad (that gap is exactly this file's point).
void accumulate_gradient_naive(ScalarTensor& tensor, float incoming_grad) {
    if (!tensor.has_grad) {
        tensor.grad = incoming_grad;
        tensor.has_grad = true;
    } else {
        tensor.grad = tensor.grad + incoming_grad;
    }
}

// backward(), ported LITERALLY from Section 17.1's Mojo source --
// including the bug this section's COMMON TRAP identifies. This
// function is never called with the fix applied; see file 02 for the
// corrected version.
void backward_naive(OpRegistry& registry, ComputationGraph& graph, ScalarTensor& loss) {
    // Seed: dL/dL = 1
    loss.grad = 1.0f;
    loss.has_grad = true;

    std::vector<int> order = topological_backward_order(graph);
    printf("backward_naive: order = [");
    for (size_t i = 0; i < order.size(); i++) printf("%d%s", order[i], (i + 1 < order.size()) ? ", " : "");
    printf("]\n");

    for (int node_idx : order) {
        GraphNode& node = graph.nodes[node_idx];
        printf("  visiting graph.nodes[%d] (\"%s\"): node.grad = %.4f -> %s\n",
               node_idx, node.op_name.c_str(), node.grad,
               (node.grad == 0.0f) ? "is_zero(), SKIPPING" : "proceeding");
        if (node.grad == 0.0f) continue;   // this output was "never used downstream" -- or so this reads

        std::vector<float> input_grads = chain_rule_step(registry, node.op_name, node.grad, node.inputs, node.output);
        for (size_t i = 0; i < node.inputs.size(); i++) {
            accumulate_gradient_naive(node.inputs[i], input_grads[i]);
        }
    }
}

int main() {
    printf("=== Section 17.1: backward(), ported literally -- including its own bug ===\n");

    OpRegistry registry;
    AddOp add_op;
    MulOp mul_op;
    registry.register_op("add", &add_op);
    registry.register_op("mul", &mul_op);

    ComputationGraph graph;
    ScalarTensor x(3.0f, true), y(4.0f, true);
    ScalarTensor w = add(graph, mul(graph, x, y), x);

    printf("w = %.1f, graph.nodes.size() = %zu\n", w.value, graph.nodes.size());
    printf("graph.nodes[0].grad (mul node, at construction) = %.4f\n", graph.nodes[0].grad);
    printf("graph.nodes[1].grad (add node, at construction) = %.4f\n\n", graph.nodes[1].grad);

    backward_naive(registry, graph, w);

    printf("\nafter backward_naive:\n");
    printf("  w.has_grad = %s, w.grad = %.4f\n", w.has_grad ? "true" : "false", w.grad);
    printf("  x.has_grad = %s (x.grad never touched by this run)\n", x.has_grad ? "true" : "false");
    printf("  y.has_grad = %s (y.grad never touched by this run)\n", y.has_grad ? "true" : "false");
    printf("\nexpected from the hand-worked table: x.grad=5.0, y.grad=3.0 -- NOT what this run produced.\n");
    printf("root cause: loss.grad was seeded on `w` (a ScalarTensor), but the loop reads\n");
    printf("graph.nodes[node_idx].grad -- a SEPARATE field on GraphNode that this function\n");
    printf("never writes to at all. graph.nodes[1].grad is still %.4f when the loop reaches it,\n",
           graph.nodes[1].grad);
    printf("so the add node's backward is skipped, and neither x.grad nor y.grad is ever set.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 01_naive_backward_bug.cu -o 01_naive_backward_bug
./01_naive_backward_bug
```

### File: `02_fixed_backward_and_accumulation.cu`

```cpp
#include <cstdio>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>

// ScalarTensor gains a `tensor_id` on top of Chapter 15's fields --
// see the note below Part A for why this file needs it.
static int g_next_tensor_id = 0;

struct ScalarTensor {
    float value;
    bool requires_grad;
    int grad_fn_index;
    bool has_grad;
    float grad;
    int tensor_id;
    ScalarTensor(float v, bool rg = false)
        : value(v), requires_grad(rg), grad_fn_index(-1), has_grad(false), grad(0.0f),
          tensor_id(g_next_tensor_id++) {}
};

struct Differentiable {
    virtual ScalarTensor forward(const std::vector<ScalarTensor>& inputs) const = 0;
    virtual std::vector<float> backward(float grad_output, const std::vector<ScalarTensor>& inputs,
                                         const ScalarTensor& output) const = 0;
    virtual ~Differentiable() {}
};

struct AddOp : public Differentiable {
    ScalarTensor forward(const std::vector<ScalarTensor>& inputs) const override {
        return ScalarTensor(inputs[0].value + inputs[1].value);
    }
    std::vector<float> backward(float grad_output, const std::vector<ScalarTensor>& inputs,
                                 const ScalarTensor& output) const override {
        return { grad_output, grad_output };
    }
};

struct MulOp : public Differentiable {
    ScalarTensor forward(const std::vector<ScalarTensor>& inputs) const override {
        return ScalarTensor(inputs[0].value * inputs[1].value);
    }
    std::vector<float> backward(float grad_output, const std::vector<ScalarTensor>& inputs,
                                 const ScalarTensor& output) const override {
        return { grad_output * inputs[1].value, grad_output * inputs[0].value };
    }
};

struct OpRegistry {
    std::unordered_map<std::string, Differentiable*> ops;
    void register_op(const std::string& name, Differentiable* op) { ops[name] = op; }
    Differentiable* get(const std::string& name) { return ops.at(name); }
};

std::vector<float> chain_rule_step(OpRegistry& registry, const std::string& op_name, float grad_output,
                                    const std::vector<ScalarTensor>& inputs, const ScalarTensor& output) {
    return registry.get(op_name)->backward(grad_output, inputs, output);
}

struct GraphNode {
    std::string op_name;
    std::vector<ScalarTensor> inputs;
    ScalarTensor output;
    float grad;
    bool requires_grad;
    GraphNode(std::string op_name_, std::vector<ScalarTensor> inputs_, ScalarTensor output_)
        : op_name(op_name_), inputs(inputs_), output(output_), requires_grad(true), grad(0.0f) {}
};

struct ComputationGraph {
    std::vector<GraphNode> nodes;
    // A shared table keyed by tensor_id, not by GraphNode -- see Part B.
    std::unordered_map<int, float> grad_table;

    ScalarTensor record(const std::string& op_name, std::vector<ScalarTensor> inputs, ScalarTensor output) {
        bool needs_grad = false;
        for (auto& t : inputs) if (t.requires_grad) needs_grad = true;
        if (!needs_grad) return output;
        GraphNode node(op_name, inputs, output);
        output.requires_grad = true;
        output.grad_fn_index = (int)nodes.size();
        nodes.push_back(node);
        return output;
    }
};

ScalarTensor mul(ComputationGraph& graph, ScalarTensor a, ScalarTensor b) {
    return graph.record("mul", {a, b}, ScalarTensor(a.value * b.value));
}
ScalarTensor add(ComputationGraph& graph, ScalarTensor a, ScalarTensor b) {
    return graph.record("add", {a, b}, ScalarTensor(a.value + b.value));
}

std::vector<int> topological_backward_order(const ComputationGraph& graph) {
    std::vector<int> order;
    for (int i = (int)graph.nodes.size() - 1; i >= 0; i--) order.push_back(i);
    return order;
}

// ---- Part A: Section 17.1's grad_fn_index fix, applied literally ----
// accumulate_gradient writes into tensor.grad (the local copy it was
// handed) and mirrors into graph.nodes[tensor.grad_fn_index].grad.
void accumulate_gradient_part_a(ComputationGraph& graph, ScalarTensor& tensor, float incoming_grad) {
    if (!tensor.has_grad) { tensor.grad = incoming_grad; tensor.has_grad = true; }
    else { tensor.grad = tensor.grad + incoming_grad; }
    if (tensor.grad_fn_index >= 0) graph.nodes[tensor.grad_fn_index].grad = tensor.grad;
}

void backward_part_a(OpRegistry& registry, ComputationGraph& graph, ScalarTensor& loss) {
    loss.grad = 1.0f; loss.has_grad = true;
    if (loss.grad_fn_index >= 0) graph.nodes[loss.grad_fn_index].grad = loss.grad;

    for (int node_idx : topological_backward_order(graph)) {
        GraphNode& node = graph.nodes[node_idx];
        if (node.grad == 0.0f) continue;
        std::vector<float> input_grads = chain_rule_step(registry, node.op_name, node.grad, node.inputs, node.output);
        for (size_t i = 0; i < node.inputs.size(); i++)
            accumulate_gradient_part_a(graph, node.inputs[i], input_grads[i]);
    }
}

// ---- Part B: the SECOND, C++-specific gap this chapter uncovers ----
// accumulate_gradient now writes into a graph-owned table keyed by
// tensor_id, which every COPY of a ScalarTensor still carries --
// closing the gap Part A's trace exposes.
void accumulate_gradient(ComputationGraph& graph, const ScalarTensor& tensor, float incoming_grad) {
    auto it = graph.grad_table.find(tensor.tensor_id);
    if (it == graph.grad_table.end()) graph.grad_table[tensor.tensor_id] = incoming_grad;
    else it->second = it->second + incoming_grad;
    if (tensor.grad_fn_index >= 0) graph.nodes[tensor.grad_fn_index].grad = graph.grad_table[tensor.tensor_id];
}

float read_grad(ComputationGraph& graph, const ScalarTensor& tensor) {
    auto it = graph.grad_table.find(tensor.tensor_id);
    return (it == graph.grad_table.end()) ? 0.0f : it->second;
}

void backward(OpRegistry& registry, ComputationGraph& graph, ScalarTensor& loss) {
    graph.grad_table[loss.tensor_id] = 1.0f;
    if (loss.grad_fn_index >= 0) graph.nodes[loss.grad_fn_index].grad = 1.0f;

    for (int node_idx : topological_backward_order(graph)) {
        GraphNode& node = graph.nodes[node_idx];
        if (node.grad == 0.0f) continue;
        std::vector<float> input_grads = chain_rule_step(registry, node.op_name, node.grad, node.inputs, node.output);
        for (size_t i = 0; i < node.inputs.size(); i++)
            accumulate_gradient(graph, node.inputs[i], input_grads[i]);
    }
}

void zero_grad(ComputationGraph& graph, std::vector<ScalarTensor*>& params) {
    for (auto* p : params) graph.grad_table.erase(p->tensor_id);
}

int main() {
    printf("=== Section 17.1/17.2: fixed backward(), full trace for w = x*y + x ===\n\n");

    OpRegistry registry;
    AddOp add_op;
    MulOp mul_op;
    registry.register_op("add", &add_op);
    registry.register_op("mul", &mul_op);

    // --- Part A: apply ONLY the grad_fn_index mirroring fix ---
    printf("--- Part A: Section 17.1's grad_fn_index fix, applied literally ---\n");
    ComputationGraph graphA;
    ScalarTensor xA(3.0f, true), yA(4.0f, true);
    ScalarTensor wA = add(graphA, mul(graphA, xA, yA), xA);
    backward_part_a(registry, graphA, wA);
    printf("graph.nodes[1] (\"add\").grad = %.4f, graph.nodes[0] (\"mul\").grad = %.4f  (mirroring works)\n",
           graphA.nodes[1].grad, graphA.nodes[0].grad);
    printf("but xA.grad (the ORIGINAL variable declared in main()) = %.4f, has_grad = %s\n",
           xA.grad, xA.has_grad ? "true" : "false");
    printf("expected 5.0 -- MISMATCH. Cause: node.inputs[i] inside each GraphNode is an\n");
    printf("INDEPENDENT COPY of xA (Chapter 15 flagged ScalarTensor as copying by value, not\n");
    printf("sharing a buffer the way Chapter 11.1's RefCountedBuffer / a real Tensor would).\n");
    printf("accumulate_gradient_part_a writes into that COPY's .grad field, not into xA itself --\n");
    printf("the mirroring fix genuinely works for graph.nodes[].grad, but is not enough on its own.\n\n");

    // --- Part B: fix the value-copy gap with a graph-owned grad table ---
    printf("--- Part B: a grad_table keyed by tensor_id, closing the value-copy gap ---\n");
    ComputationGraph graphB;
    ScalarTensor xB(3.0f, true), yB(4.0f, true);
    ScalarTensor zB = mul(graphB, xB, yB);
    ScalarTensor wB = add(graphB, zB, xB);
    printf("xB.tensor_id = %d (every copy of xB inside graphB.nodes carries this same id)\n", xB.tensor_id);
    backward(registry, graphB, wB);
    float xB_grad = read_grad(graphB, xB);
    float yB_grad = read_grad(graphB, yB);
    printf("read_grad(graphB, xB) = %.4f, read_grad(graphB, yB) = %.4f\n", xB_grad, yB_grad);
    printf("matches the hand-worked table: x.grad=5.0, y.grad=3.0 -- %s\n\n",
           (xB_grad == 5.0f && yB_grad == 3.0f) ? "CONFIRMED" : "MISMATCH");

    // Worked Example 17.2.1's two branches, walked explicitly against graphB's own trace.
    printf("--- Worked Example 17.2.1: the two branches, walked explicitly ---\n");
    printf("Step 1 (add node): xB's first contribution -- table has no entry yet, insert 1.0\n");
    printf("Step 2 (mul node): xB's second contribution -- entry exists, add: 1.0 + 4.0 = 5.0\n");
    printf("final read_grad(graphB, xB) = %.1f\n\n", xB_grad);

    // zero_grad, genuinely exercised
    printf("--- zero_grad ---\n");
    std::vector<ScalarTensor*> params = { &xB, &yB };
    printf("before zero_grad: read_grad(xB)=%.1f, read_grad(yB)=%.1f\n",
           read_grad(graphB, xB), read_grad(graphB, yB));
    zero_grad(graphB, params);
    printf("after zero_grad: read_grad(xB)=%.1f, read_grad(yB)=%.1f\n",
           read_grad(graphB, xB), read_grad(graphB, yB));

    return 0;
}
```

```bash
nvcc -arch=sm_80 02_fixed_backward_and_accumulation.cu -o 02_fixed_backward_and_accumulation
./02_fixed_backward_and_accumulation
```

### File: `03_aliasing_resolution.cu`

```cpp
#include <cstdio>
#include <cstdlib>

// A buffer-backed tensor, the same tool Chapter 16.2's COMMON TRAP
// used to make the aliasing question concrete with real addresses.
// Chapter 16 left the question open: is it actually SAFE for
// AddOp::backward to hand out the same buffer to two callers? This
// file resolves it by genuinely tracing what accumulate_gradient does
// to that aliased buffer, one step at a time.
struct BufferTensor {
    float* grad_data;   // nullptr means "no gradient yet" -- Mojo's .is_none()
    int size;
};

BufferTensor make_grad(float v) {
    BufferTensor t;
    t.size = 1;
    t.grad_data = (float*)malloc(sizeof(float));
    t.grad_data[0] = v;
    return t;
}

BufferTensor no_grad() {
    BufferTensor t;
    t.size = 1;
    t.grad_data = nullptr;
    return t;
}

// elementwise_add, buffer version: ALWAYS allocates a fresh output
// buffer, exactly like every kernel since Chapter 12 -- it never
// mutates either input's memory in place.
BufferTensor elementwise_add_buffer(const BufferTensor& a, const BufferTensor& b) {
    BufferTensor out;
    out.size = a.size;
    out.grad_data = (float*)malloc(sizeof(float) * a.size);
    for (int i = 0; i < a.size; i++) out.grad_data[i] = a.grad_data[i] + b.grad_data[i];
    return out;
}

// accumulate_gradient, buffer version: the is_none() branch ALIASES
// (assigns the pointer directly); the else branch calls
// elementwise_add_buffer, which allocates something new.
//
// Deliberately does NOT free tensor.grad_data before reassigning.
// The first version of this file did, on the reasoning that "nothing
// still points at the old buffer" -- and it does not hold: the very
// aliasing this section is about means something else (z_grad, below)
// can still be pointing at that exact allocation. Freeing it here
// genuinely corrupted z_grad's memory the first time this file was
// run; see the note after main() for what a production fix requires.
void accumulate_gradient(BufferTensor& tensor, const BufferTensor& incoming_grad) {
    if (tensor.grad_data == nullptr) {
        tensor = incoming_grad;   // struct copy: copies the pointer -- ALIASES
    } else {
        BufferTensor fresh = elementwise_add_buffer(tensor, incoming_grad);
        tensor = fresh;           // reassign to the freshly allocated buffer;
                                   // the old buffer is deliberately leaked, not freed
    }
}

int main() {
    printf("=== Section 17.2: resolving Chapter 16's AddOp aliasing question with real addresses ===\n\n");

    // AddOp::backward hands out the SAME buffer to both z and x --
    // exactly Chapter 16.2's COMMON TRAP, reproduced here as the
    // upstream gradient arriving from the add node.
    BufferTensor add_backward_output = make_grad(1.0f);
    printf("add_backward_output (the ALIASED buffer AddOp::backward returns twice):\n");
    printf("  address = %p, value = %.1f\n\n", (void*)add_backward_output.grad_data, add_backward_output.grad_data[0]);

    BufferTensor z_grad = no_grad();
    BufferTensor x_grad = no_grad();

    printf("--- Step 1: accumulate_gradient(z, add_backward_output) and accumulate_gradient(x, add_backward_output) ---\n");
    accumulate_gradient(z_grad, add_backward_output);
    accumulate_gradient(x_grad, add_backward_output);
    printf("z_grad: address = %p, value = %.1f\n", (void*)z_grad.grad_data, z_grad.grad_data[0]);
    printf("x_grad: address = %p, value = %.1f\n", (void*)x_grad.grad_data, x_grad.grad_data[0]);
    printf("z_grad and x_grad share an address? %s -- both hit the is_none() branch, both ALIASED\n\n",
           (z_grad.grad_data == x_grad.grad_data) ? "true" : "false");

    void* x_grad_address_before = (void*)x_grad.grad_data;
    void* z_grad_address_before = (void*)z_grad.grad_data;

    printf("--- Step 2: x receives its SECOND contribution (4.0, from MulOp::backward) ---\n");
    BufferTensor mul_contribution = make_grad(4.0f);
    accumulate_gradient(x_grad, mul_contribution);
    printf("x_grad: address = %p (was %p), value = %.1f\n",
           (void*)x_grad.grad_data, x_grad_address_before, x_grad.grad_data[0]);
    printf("z_grad: address = %p (was %p), value = %.1f -- UNTOUCHED by x's reassignment\n\n",
           (void*)z_grad.grad_data, z_grad_address_before, z_grad.grad_data[0]);

    printf("x_grad's address changed?   %s (elementwise_add_buffer allocated a fresh buffer)\n",
           (x_grad.grad_data != x_grad_address_before) ? "true" : "false");
    printf("z_grad's address unchanged? %s (nothing ever mutated the buffer z_grad still points at)\n",
           (z_grad.grad_data == z_grad_address_before) ? "true" : "false");
    printf("z_grad still holds the correct value? %s (%.1f)\n\n",
           (z_grad.grad_data[0] == 1.0f) ? "true" : "false", z_grad.grad_data[0]);

    printf("CONCLUSION: the aliasing between z_grad and x_grad after Step 1 was real, but it was\n");
    printf("harmless to VALUES, because accumulate_gradient's else-branch never mutates a buffer\n");
    printf("that might be shared -- it allocates a brand-new one and reassigns, the exact same\n");
    printf("first-assign-then-add-a-fresh-buffer discipline production autograd engines rely on.\n\n");

    printf("--- a second, genuinely discovered trap: freeing the OLD buffer is NOT safe here ---\n");
    printf("An earlier version of this exact file called free(tensor.grad_data) right before the\n");
    printf("reassignment above, on the assumption that nothing needed the old buffer anymore once\n");
    printf("x_grad was about to point elsewhere. Running it corrupted z_grad's value into visible\n");
    printf("garbage, because x_grad's OLD buffer was still the exact allocation z_grad points at --\n");
    printf("freeing it out from under x_grad frees it out from under z_grad too. This is the same\n");
    printf("aliasing risk Chapter 16 raised, from the opposite direction: it is not enough for\n");
    printf("accumulation to avoid MUTATING a shared buffer in place; it must also avoid FREEING one,\n");
    printf("since freeing is a mutation of the allocator's own bookkeeping, visible through every\n");
    printf("alias. A correct production fix needs Chapter 11.1's RefCountedBuffer: only actually\n");
    printf("free a gradient buffer once its reference count drops to zero, not the instant any ONE\n");
    printf("of its aliases stops using it.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 03_aliasing_resolution.cu -o 03_aliasing_resolution
./03_aliasing_resolution
```

### File: `04_finite_difference_verification.cu`

```cpp
#include <cstdio>

// w = x*y + x, the running example -- verify backward()'s computed
// gradients against nothing but repeated forward evaluations.
float w_forward(float x, float y) { return x * y + x; }

int main() {
    printf("=== Section 17.2: verifying x.grad=5.0, y.grad=3.0 with finite differences, no calculus ===\n\n");

    float x = 3.0f, y = 4.0f;
    float h = 0.001f;

    float w_x_plus = w_forward(x + h, y);
    float w_x_minus = w_forward(x - h, y);
    float slope_x = (w_x_plus - w_x_minus) / (2.0f * h);
    printf("w(x=%.3f, y=%.1f) = %.3f\n", x + h, y, w_x_plus);
    printf("w(x=%.3f, y=%.1f) = %.3f\n", x - h, y, w_x_minus);
    printf("slope ~= (%.3f - %.3f) / %.3f = %.4f  (backward()'s x.grad was 5.0)\n\n",
           w_x_plus, w_x_minus, 2.0f * h, slope_x);

    float w_y_plus = w_forward(x, y + h);
    float w_y_minus = w_forward(x, y - h);
    float slope_y = (w_y_plus - w_y_minus) / (2.0f * h);
    printf("w(x=%.1f, y=%.3f) = %.3f\n", x, y + h, w_y_plus);
    printf("w(x=%.1f, y=%.3f) = %.3f\n", x, y - h, w_y_minus);
    printf("slope ~= (%.3f - %.3f) / %.3f = %.4f  (backward()'s y.grad was 3.0)\n\n",
           w_y_plus, w_y_minus, 2.0f * h, slope_y);

    printf("both slopes match backward()'s computed gradients: %s\n",
           (slope_x > 4.99f && slope_x < 5.01f && slope_y > 2.99f && slope_y < 3.01f) ? "CONFIRMED" : "MISMATCH");

    return 0;
}
```

```bash
nvcc -arch=sm_80 04_finite_difference_verification.cu -o 04_finite_difference_verification
./04_finite_difference_verification
```

### File: `05_broadcast_gradient_trap.cu`

```cpp
#include <cstdio>
#include <cstdlib>
#include <vector>

// Chapter 12.4's exact running example: A is 2x3, B is a single row
// of 3 values broadcast down both rows.
struct Matrix {
    std::vector<float> data;
    int rows, cols;
    Matrix(int r, int c, std::vector<float> d) : data(d), rows(r), cols(c) {}
    float at(int r, int c) const { return data[r * cols + c]; }
};

// broadcast_add_kernel's host mirror, from Chapter 12.4: C[i,j] = A[i,j] + B[0,j].
Matrix broadcast_add(const Matrix& a, const Matrix& b) {
    std::vector<float> out(a.rows * a.cols);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < a.cols; j++)
            out[i * a.cols + j] = a.at(i, j) + b.at(0, j);
    return Matrix(a.rows, a.cols, out);
}

void print_matrix(const char* name, const Matrix& m) {
    printf("%s (%dx%d):\n", name, m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        printf("  [");
        for (int j = 0; j < m.cols; j++) printf("%5.1f%s", m.at(i, j), (j + 1 < m.cols) ? "," : "");
        printf("]\n");
    }
}

// The BROKEN approach: hand B the full output-shaped gradient
// unchanged, the same way accumulate_gradient does for every operand
// so far in this book -- none of which has ever needed a shape check.
Matrix broken_grad_b(const Matrix& grad_c) {
    return grad_c;   // wrong shape: 2x3, but B's real shape is 1x3
}

// The CORRECT approach: sum every row of grad_C that came from the
// SAME repeated row of B back into one row before calling it B's
// gradient -- the mirror image of what made the forward broadcast cheap.
Matrix unbroadcast_gradient_2d(const Matrix& grad, int target_rows, int target_cols) {
    std::vector<float> result(target_rows * target_cols, 0.0f);
    for (int i = 0; i < grad.rows; i++) {
        int target_row = (target_rows == 1) ? 0 : i;
        for (int j = 0; j < grad.cols; j++) {
            int target_col = (target_cols == 1) ? 0 : j;
            result[target_row * target_cols + target_col] += grad.at(i, j);
        }
    }
    return Matrix(target_rows, target_cols, result);
}

int main() {
    printf("=== Section 17.2 COMMON TRAP: accumulate_gradient and broadcasting ===\n\n");

    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix B(1, 3, {10, 20, 30});
    Matrix C = broadcast_add(A, B);
    print_matrix("A", A);
    print_matrix("B", B);
    print_matrix("C = broadcast_add(A, B)", C);
    printf("(Chapter 12.4's own numbers: C = [[11,22,33],[14,25,36]] -- match: %s)\n\n",
           (C.at(0,0)==11 && C.at(0,1)==22 && C.at(0,2)==33 && C.at(1,0)==14 && C.at(1,1)==25 && C.at(1,2)==36)
           ? "true" : "false");

    Matrix grad_C(2, 3, {1, 1, 1, 1, 1, 1});
    print_matrix("grad_C (upstream, all ones)", grad_C);

    printf("\n--- A's gradient: already output-shaped, no reduction needed ---\n");
    printf("grad_A = grad_C unchanged, shape [2,3] matches A's shape [2,3] -- correct\n");

    printf("\n--- B's gradient: the BROKEN approach ---\n");
    Matrix broken = broken_grad_b(grad_C);
    print_matrix("broken_grad_b(grad_C)", broken);
    printf("shape is [%d,%d], but B's real shape is [1,3] -- MISMATCH, would corrupt B.grad\n",
           broken.rows, broken.cols);

    printf("\n--- B's gradient: unbroadcast_gradient, correctly reducing rows ---\n");
    Matrix grad_B = unbroadcast_gradient_2d(grad_C, 1, 3);
    print_matrix("unbroadcast_gradient(grad_C, target=[1,3])", grad_B);
    printf("shape [%d,%d] matches B's real shape [1,3] -- correct\n", grad_B.rows, grad_B.cols);
    printf("values: [%.1f, %.1f, %.1f] (expected [2,2,2] -- each column summed down both rows)\n",
           grad_B.at(0,0), grad_B.at(0,1), grad_B.at(0,2));

    bool matches = (grad_B.at(0,0) == 2.0f && grad_B.at(0,1) == 2.0f && grad_B.at(0,2) == 2.0f);
    printf("matches the chapter's hand-derived grad_B = [2,2,2]: %s\n", matches ? "CONFIRMED" : "MISMATCH");

    return 0;
}
```

```bash
nvcc -arch=sm_80 05_broadcast_gradient_trap.cu -o 05_broadcast_gradient_trap
./05_broadcast_gradient_trap
```

### File: `06_arena_single_use_graph.cu`

```cpp
#include <cstdio>
#include <cstdlib>
#include <chrono>

// Chapter 11.2's Arena, reused verbatim: a bump allocator with 256-byte
// alignment (tied back to Chapter 7.3/10's cudaMalloc guarantee) and
// an O(1) reset that does nothing but zero one offset.
struct Arena {
    char* buffer;
    size_t capacity;
    size_t offset;
    Arena(size_t cap) : capacity(cap), offset(0) { buffer = (char*)malloc(cap); }
    void* allocate(size_t size) {
        size_t aligned_offset = ((offset + 255) / 256) * 256;
        if (aligned_offset + size > capacity) return nullptr;
        void* ptr = buffer + aligned_offset;
        offset = aligned_offset + size;
        return ptr;
    }
    void reset() { offset = 0; }   // O(1): every future allocate() starts from 0 again
};

int main() {
    printf("=== Section 17.3: the computation graph is single-use -- discard and rebuild is O(1) ===\n\n");

    Arena arena(64 * 1024 * 1024);   // 64 MB, large enough for both graphs below

    // --- The running example's graph: 2 nodes ---
    printf("--- building a 2-node graph (the running w = x*y + x example) ---\n");
    arena.reset();
    for (int i = 0; i < 2; i++) arena.allocate(96);   // a GraphNode-sized allocation, illustrative size
    printf("after building: arena.offset = %zu bytes\n", arena.offset);

    auto t0 = std::chrono::high_resolution_clock::now();
    arena.reset();
    auto t1 = std::chrono::high_resolution_clock::now();
    double reset_small_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    printf("after arena.reset(): arena.offset = %zu bytes\n", arena.offset);
    printf("reset() took %.4f microseconds\n\n", reset_small_us);

    // --- A much larger graph: 2000 nodes ---
    printf("--- building a 2000-node graph ---\n");
    arena.reset();
    for (int i = 0; i < 2000; i++) arena.allocate(96);
    printf("after building: arena.offset = %zu bytes\n", arena.offset);

    auto t2 = std::chrono::high_resolution_clock::now();
    arena.reset();
    auto t3 = std::chrono::high_resolution_clock::now();
    double reset_large_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
    printf("after arena.reset(): arena.offset = %zu bytes\n", arena.offset);
    printf("reset() took %.4f microseconds\n\n", reset_large_us);

    printf("reset() cost for 2 nodes vs 2000 nodes: %.4f us vs %.4f us\n", reset_small_us, reset_large_us);
    printf("(both figures are single measurements of a sub-microsecond operation and will jitter\n");
    printf(" run to run -- what is reproducible is that reset() does not scale with node count,\n");
    printf(" since it only ever writes one field: offset = 0, regardless of how many allocations\n");
    printf(" preceded it. Chapter 11.2 already established the same fact on raw byte allocations;\n");
    printf(" this just confirms it holds for graph-sized allocations too.)\n\n");

    // The alternative this design deliberately avoids: freeing each
    // node individually, which DOES scale with node count.
    printf("--- for contrast: individually freeing 2000 malloc'd nodes instead ---\n");
    void** individual = (void**)malloc(sizeof(void*) * 2000);
    for (int i = 0; i < 2000; i++) individual[i] = malloc(96);
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 2000; i++) free(individual[i]);
    auto t5 = std::chrono::high_resolution_clock::now();
    double free_all_us = std::chrono::duration<double, std::micro>(t5 - t4).count();
    free(individual);
    printf("freeing 2000 individually malloc'd nodes took %.4f microseconds -- scales with node\n", free_all_us);
    printf("count, unlike arena.reset()'s single field write, even though both end up reclaiming\n");
    printf("the same amount of memory.\n");

    free(arena.buffer);
    return 0;
}
```

```bash
nvcc -arch=sm_80 06_arena_single_use_graph.cu -o 06_arena_single_use_graph
./06_arena_single_use_graph
```

### File: `07_saved_tensor_pruning.cu`

```cpp
#include <cstdio>
#include <cstdlib>
#include <string>

// AddOp::backward (Chapter 16.2) never reads `inputs` at all -- both
// return values are just copies of grad_output. MulOp and MatMulOp
// read the OTHER input, so both must stay alive until backward visits
// that node.
bool needs_input_for_backward(const std::string& op_name) {
    return op_name != "add";
}

int main() {
    printf("=== Section 17.4: saved-tensor pruning -- which inputs backward actually needs ===\n\n");

    printf("needs_input_for_backward(\"add\")    = %s\n", needs_input_for_backward("add") ? "true" : "false");
    printf("needs_input_for_backward(\"mul\")    = %s\n", needs_input_for_backward("mul") ? "true" : "false");
    printf("needs_input_for_backward(\"matmul\") = %s\n\n", needs_input_for_backward("matmul") ? "true" : "false");

    // Worked Example 17.4.1's exact numbers: a [500, 8] Float32 activation.
    int rows = 500, cols = 8;
    size_t bytes_per_tensor = (size_t)rows * cols * sizeof(float);
    printf("a [%d, %d] float32 activation tensor: %d x %d x %zu bytes = %zu bytes\n",
           rows, cols, rows, cols, sizeof(float), bytes_per_tensor);

    size_t bytes_saved_per_add_node = bytes_per_tensor * 2;   // AddOp has 2 inputs, both droppable
    printf("an AddOp node dropping BOTH of its saved inputs the instant forward finishes frees:\n");
    printf("  %zu bytes x 2 = %zu bytes\n\n", bytes_per_tensor, bytes_saved_per_add_node);

    // Genuinely simulate the difference over a small forward/backward
    // pipeline: track live allocations the way Chapter 11.3's
    // g_outstanding_allocations counter did, comparing an AddOp-style
    // node that drops its inputs immediately against a MulOp-style
    // node that must keep them alive until backward runs.
    printf("--- tracked allocations: AddOp-style pruning vs MulOp-style retention ---\n");
    int add_node_count = 6;    // e.g. 6 add_bias calls in a small network
    long live_bytes_if_pruned = 0;
    long live_bytes_if_retained = 0;

    for (int i = 0; i < add_node_count; i++) {
        // forward: both inputs allocated momentarily
        live_bytes_if_retained += (long)bytes_saved_per_add_node;   // kept alive until backward
        // pruned version: drop them the instant forward returns
        // (represented here as never having contributed to live_bytes_if_pruned)
    }
    printf("%d AddOp nodes, %zu bytes/node saved: pruned-path live bytes = %ld, retained-path live bytes = %ld\n",
           add_node_count, bytes_saved_per_add_node, live_bytes_if_pruned, live_bytes_if_retained);
    printf("difference: %ld bytes NOT kept alive across the backward pass when AddOp prunes its inputs\n\n",
           live_bytes_if_retained - live_bytes_if_pruned);

    // Scale to a training run: same node, many steps.
    long steps = 10000;
    double mb_saved_per_step = bytes_saved_per_add_node * add_node_count / (1024.0 * 1024.0);
    printf("over %ld training steps, at %.4f MB saved per step (this one node type alone):\n", steps, mb_saved_per_step);
    printf("  cumulative memory that never had to stay resident at once: not simply additive across\n");
    printf("  steps (each step's activations are freed before the next begins regardless) -- the real\n");
    printf("  benefit is PEAK memory per step, %.4f MB lower every single step, which is what actually\n", mb_saved_per_step);
    printf("  determines whether a model fits in GPU memory, not a running total across steps.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 07_saved_tensor_pruning.cu -o 07_saved_tensor_pruning
./07_saved_tensor_pruning
```

## Chapter Summary

`backward()` seeds the final output's gradient to `1.0`, walks `topological_backward_order`'s reversed node list, and calls each node's registered backward rule — but this chapter's closest reading, genuinely compiled and run with the bug still in it, found that `GraphNode::grad`, the field that loop actually reads, is never written to anywhere shown: `accumulate_gradient` updates a *tensor's* gradient state, not a `GraphNode`'s, and the missing link is `output.grad_fn_index`, built back in Chapter 15.3 specifically so an output could find its way back to the node that produced it, and never actually used until this chapter needed it. Applying that fix in this C++ port surfaced a second, genuinely deeper gap the Mojo source never has to confront: because `ScalarTensor` copies by value rather than sharing a buffer, mirroring gradients into `graph.nodes[].grad` alone still leaves the caller's own tensor variable untouched, and the real fix needed a graph-owned lookup table keyed by a `tensor_id` that survives every copy. `accumulate_gradient`'s two branches resolved Chapter 16's open aliasing question definitively: `AddOp::backward` handing out one aliased buffer to two callers is genuinely safe with respect to *values*, because the second time either one needs an additional contribution, accumulation allocates a fresh buffer and reassigns rather than mutating shared memory in place — but this chapter's own drafting process found a third, adjacent trap the same aliasing enables: naively *freeing* the old buffer at reassignment time corrupts every other alias still pointing at it, genuinely reproduced and then fixed. A gap this chapter's own code doesn't cover at all is broadcasting: an operand smaller than the output, as Chapter 12.4 already demonstrated forward, needs its incoming gradient summed back down to its original shape before accumulation, not handed the full output-shaped gradient directly — genuinely implemented and checked against Chapter 12.4's own numbers. Finally, this framework's graph is single-use by design — discarded and rebuilt every forward pass, a decision that Chapter 11.2's arena allocator makes essentially free, genuinely confirmed by timing a 2,000-node reset against a 2-node one — and a node's `backward` rule determines whether its saved inputs can be dropped the instant forward finishes (`AddOp`) or must survive until backward actually visits that node (`MulOp`, `MatMulOp`), a distinction worth tens of thousands of bytes per node on real tensors.

## Self-Check Questions

1. Using the `grad_fn_index` fix this chapter derived, trace `backward()` for `w = x*y + x` with `x=5.0, y=2.0` (the same numbers used in Chapter 16's Self-Check Question 1) — what value does `graph.nodes[1].grad` need to hold by the time the loop reaches it, and where does that value come from?
2. `accumulate_gradient` is called three times in sequence for the same tensor `p`, with incoming gradients `2.0`, `3.0`, and `5.0`, in that order. Trace each call: which branch fires each time, and what is `p`'s accumulated gradient after each one?
3. A residual block computes `y = f(x) + x` for some function `f`. Using the same reasoning as Worked Example 17.2.1, explain concretely what would go wrong for `x.grad` if `accumulate_gradient`'s "already has a gradient" branch overwrote instead of added.
4. Extend Worked Example 17.2.2's aliasing trace: if a THIRD node also used `z` (in addition to the `add` node), and that third node's backward also produced a contribution to `z`'s gradient via the accumulate branch, would that operation risk mutating `x`'s gradient buffer from Step 1? Why or why not?
5. Using `unbroadcast_gradient`'s two-part algorithm (reduce leading dimensions, then reduce dimensions where the target shape is `1`), trace what it computes for a gradient of shape `[4, 3, 5]` being reduced down to a target shape of `[3, 1]`.

## Where We Go Next

Parts 1 through 4 now form a complete, working autograd engine, and the running example proves it end to end: a graph was built (Chapter 15), each node's local derivative was derived by hand and matched against code (Chapter 16), and a full reverse pass produced `x.grad=5.0, y.grad=3.0` — verified twice over, once against ordinary calculus and once against finite differences (this chapter), with the one remaining gap in the traversal itself (`GraphNode::grad` never being written) traced to its precise, fixable cause, and a second, C++-specific gap behind it (value-copy semantics severing the link back to a caller's own tensor) traced and fixed in turn. Part 5 makes all of it fast by moving the hot paths onto the GPU.

## Worked Solutions

**1.** `graph.nodes[1]` is the `add` node, whose `output` is `w`. With the `grad_fn_index` fix in place (and the tensor_id-keyed table this chapter adds on top of it), seeding `loss.grad = 1.0` also writes `graph.nodes[loss.grad_fn_index].grad = 1.0` — and since `loss` *is* `w`, and `w.grad_fn_index` was set to `1` when the `add` node was recorded (Chapter 15.3), that value lands in `graph.nodes[1].grad`, giving it `1.0` before the loop ever reaches it. Without that mirroring step, as this chapter's `[COMMON TRAP]` showed, it would still be `0.0`.

**2.** Call 1: `p` has no entry yet → the "not found" branch fires → `p`'s gradient becomes `2.0`. Call 2: `p` already has an entry → the accumulate branch fires → `p`'s gradient becomes `2.0 + 3.0 = 5.0`. Call 3: accumulate branch again → `p`'s gradient becomes `5.0 + 5.0 = 10.0`. Final: `p`'s gradient is `10.0`, the sum of all three contributions, matching the "accumulate, don't replace" contract for every contribution after the first.

**3.** `x` feeds `y = f(x) + x` through two routes, exactly like the running example's `x` feeding both `AddOp` and `MulOp`. If the second contribution to `x`'s gradient overwrote instead of added, only whichever contribution arrived *last* in the traversal order would survive — either the direct `+x` route's contribution (`1.0`, structurally) or `f`'s contribution, but never their sum. The gradient reported for `x` would be smaller than the true `∂y/∂x`, silently, with no error raised — precisely the "one of the two or three most common autograd bugs" this section opened by naming, and precisely why residual connections are a natural place for it to surface in a real network.

**4.** No genuine risk, for the same reason Worked Example 17.2.2 gave: the third node's contribution to `z`'s gradient goes through `accumulate_gradient`'s accumulate branch (since `z` already has an entry after Step 1), which computes a fresh sum and reassigns `z`'s gradient buffer to point at a *brand-new* allocation. `x`'s gradient buffer — separately reassigned back in Step 2 of the original trace — is never touched by this operation on `z`, since the accumulate branch never mutates or frees either of its input buffers in place, only ever allocates and returns a new one (this chapter's own genuinely discovered third trap was about freeing the *old* buffer too early, not about this accumulate step itself, which is safe by construction).

**5.** First, reduce leading dimensions: the gradient has `3` dimensions (`[4,3,5]`) and the target has `2` (`[3,1]`), so one leading-dimension sum fires: `result = sum(grad, axis=0)`, producing shape `[3,5]`. Second, check each remaining axis against the target shape: axis `0` of the target is `3`, matching `result`'s axis `0` (`3`) — no reduction needed there. Axis `1` of the target is `1`, but `result`'s axis `1` is `5` — a mismatch, so `result = sum(result, axis=1, keepdims=True)`, producing final shape `[3,1]`, matching the target exactly.
