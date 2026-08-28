# Chapter 15: Graph Node Architecture — Recording What the Forward Pass Throws Away

> "The moment the program finishes computing `15.0`, none of the history that produced it is still around — `w` is just a `float`, indistinguishable from a `float` that arrived from anywhere else. A computational graph is nothing more than that missing history, kept around on purpose."

**What you will understand by the end of this chapter:**

- Why an operation needs to implement *both* a forward and a backward method, bundled into one `Differentiable` interface, before a graph can ever record it — and why `MulOp::backward` specifically cannot be written from `grad_output` alone, without also seeing what the *other* input was
- `GraphNode`'s exact fields, traced field-by-field into the two nodes this chapter's running `w = x*y + x` example actually produces, down to the literal numbers `3.0`, `4.0`, `12.0`, and `15.0`
- The precise condition under which `ComputationGraph::record` skips creating a node at all — which is narrower than "this particular input is frozen," and worth stating exactly rather than loosely
- Why this chapter's `topological_backward_order` can get away with simply reversing the order nodes were appended, instead of running a real topological-sort algorithm from a specific output — and the one architectural assumption that safety depends on, checked against how reverse-mode automatic differentiation is implemented in production frameworks generally
- A genuine, compiled divergence between what an unregistered-op lookup does in a debug build versus a release build — and a further, real divergence between C++'s `std::unordered_map` and the associative container this design is ported from, worth knowing about on its own terms

**What you need to know first:**

- Chapter 12 (`elementwise_add` and `elementwise_mul` — the kernels `AddOp` and `MulOp` wrap; this chapter is about the bookkeeping layer built *on top of* those kernels, not a replacement for them)
- Chapter 11.1 (shared, non-copying buffer references — `GraphNode::inputs` conceptually holds the same tensors an operation was called with, not independent copies, the same sharing discipline `RefCountedBuffer<T>` established; this chapter's `ScalarTensor` stand-in copies by value for simplicity at this small scale, a scoping choice flagged explicitly below)
- Chapter 6 (the `Tensor` struct this chapter's `requires_grad` flag and `grad_fn_index` field extend, in spirit — implemented here against a minimal scalar stand-in rather than the full device-backed `Tensor`)
- Chapter 11.2 (the `assert`/`NDEBUG` divergence this chapter's `OpRegistry` reuses directly, genuinely compiled both ways again)

## Why a graph, and not just a function call?

Suppose you write the expression `w = x*y + x` in C++, with `x = 3.0f` and `y = 4.0f`. Running it forward is trivial:

```
z = x * y = 3.0 * 4.0 = 12.0
w = z + x = 12.0 + 3.0 = 15.0
```

Now ask a different question: *if `x` nudges up by a tiny amount, how much does `w` move?* Ordinary calculus answers this in one line — `w = xy + x`, so `∂w/∂x = y + 1 = 4 + 1 = 5` — but notice everything that answer silently depended on: that `w` was built from `z`, that `z` was built from `x` and `y` by multiplication, and that `x` feeds into `w` a *second* time, directly, through the addition. The instant the program finishes computing `15.0`, none of that structure survives; `w` is one `float`, with no record of how it got there.

**A computational graph is nothing more than that missing history, kept around on purpose.** Every time an operation runs, instead of only producing a value, it also records a note: "I am a multiply, my inputs were `x` and `y`, and I produced `12.0`." Chapter 16 shows that once every operation leaves a note like this, the chain rule applies *mechanically* to the whole chain of notes — no human derives `∂w/∂x = y + 1` symbolically ever again. This chapter builds the note itself (`GraphNode`), the mechanism that writes one during every forward computation, and the ordering rule for which notes to read first when walking backward.

Keep `x=3, y=4, z=12, w=15` in mind — every section below builds the graph these two operations actually produce, genuinely compiled and run, one field at a time.

## 15.1 Function Registration System `[FOUNDATIONAL]`

### Intuition

An operation can only be part of a graph if the framework knows two things about it: how to run it forward, and how to turn an "upstream sensitivity" into sensitivities for each of its inputs. Neither is optional on its own — an op with only a forward implementation can compute `w`, but can never explain how `w` depends on `x`, which makes it useless to a graph that exists specifically to answer that question. Bundling both directions into one interface, registered together, is what guarantees a graph can never end up holding a forward computation it has no idea how to differentiate.

### Background

```cpp
// A minimal scalar stand-in for Chapter 6's Tensor, sized to this
// chapter's running scalar example.
struct ScalarTensor {
    float value;
    bool requires_grad;
    int grad_fn_index;
    ScalarTensor(float v, bool rg = false) : value(v), requires_grad(rg), grad_fn_index(-1) {}
};

// Every op the graph can record implements both directions -- the C++
// analogue of a Mojo-style trait, implemented here as an abstract base.
struct Differentiable {
    virtual ScalarTensor forward(const std::vector<ScalarTensor>& inputs) const = 0;
    virtual std::vector<float> backward(float grad_output, const std::vector<ScalarTensor>& inputs,
                                         const ScalarTensor& output) const = 0;
    virtual ~Differentiable() {}
};

// Maps an op name to its registered Differentiable implementation.
struct OpRegistry {
    std::unordered_map<std::string, Differentiable*> ops;

    void register_op(const std::string& name, Differentiable* op) {
        ops[name] = op;
    }

    Differentiable* get(const std::string& name) {
        assert(ops.count(name) > 0 && "Unregistered op");
        return ops[name];
    }
};
```

`backward`'s signature is worth reading carefully: it takes `grad_output`, but also `inputs` and `output` — not just the upstream sensitivity by itself. That is not incidental plumbing. For `MulOp`, knowing only the upstream sensitivity isn't enough to answer anything: the sensitivity of `z = x*y` to `x` is literally the *value of `y`*, and the sensitivity to `y` is the value of `x`. There is no way to produce either number without seeing both operands, which is exactly why `backward` receives `inputs` at all — a design choice every reverse-mode autodiff implementation makes the same way, because the local derivative of a product is fundamentally "the other operand," not something derivable from the output alone.

### Worked Example 15.1.1 — What the registry needs before recording anything, and what `MulOp::backward` genuinely computes

For the running example, two ops must already exist in the registry before `w = x*y + x` can be recorded at all: a `MulOp` and an `AddOp`. Compiled and run:

```bash
nvcc -arch=sm_80 01_function_registration.cu -o 01_function_registration
./01_function_registration
```

Genuinely compiled and run:

```
registry.get("mul") succeeded: true
MulOp.backward(grad_output=1.0, x=3.0, y=4.0): dz/dx=4.0 (=y), dz/dy=3.0 (=x)
```

`MulOp::backward` genuinely returns `[y, x] = [4.0, 3.0]` when scaled by an upstream gradient of `1.0` — exactly the "sensitivity to `x` is the value of `y`, and vice versa" claim above, not asserted but computed. Chapter 16 derives this formally; the shape of the answer is already visible here.

### Worked Example 15.1.2 — The registry's own trap, genuinely compiled both ways

Calling `registry.get("relu")` before any `ReluOp` has ever been registered would trip the `assert` in `OpRegistry::get` — but only in a build where `assert` hasn't been compiled out. Chapter 11.2 already established that `assert` is a debug-build-only check in C++; the identical divergence is reproduced here, on an empty registry, compiled both ways:

```bash
# debug build: assert active
nvcc -arch=sm_80 01b_registry_trap.cu -o 01b_debug
./01b_debug
echo "exit code: $?"

# release build: NDEBUG strips the assert entirely
nvcc -arch=sm_80 -DNDEBUG 01b_registry_trap.cu -o 01b_release
./01b_release
echo "exit code: $?"
```

Genuinely compiled and run, both ways:

```
--- DEBUG BUILD (assert active) ---
01b_debug: 01b_registry_trap.cu:11: Differentiable* OpRegistry::get(const std::string&): Assertion `ops.count(name) > 0 && "Unregistered op"' failed.
Aborted
exit code: 134

--- RELEASE BUILD (-DNDEBUG, assert compiled out) ---
calling registry.get("relu") on an empty registry...
get() returned without aborting. result == nullptr? true
registry.ops now has 1 entries (operator[] silently inserted one)
exit code: 0
```

The debug build aborts cleanly with a diagnostic naming the exact failed condition. The release build falls straight through to `return ops[name]` — and here C++'s own container semantics add a second, sharper trap on top of the one this design is ported from: `std::unordered_map::operator[]` never throws on a missing key. It silently default-constructs one, inserts it, and returns that default (`nullptr`, for a pointer value type) — genuinely confirmed above: `result == nullptr` is `true`, and the registry's own size grew from `0` to `1` entries as a side effect of a lookup that was only supposed to be reading. A caller in a release build gets a `nullptr` back with no signal anything was missing at all, and the registry itself now silently contains a phantom empty entry under the name `"relu"` — worse than a Mojo-style associative container, whose indexing operator would at least raise on a missing key regardless of build mode; `operator[]`'s specific "insert a default on miss" behavior is a real, C++-specific hazard layered on top of the debug-only `assert` gap.

## 15.2 Gradient Function Traits `[FOUNDATIONAL]`

### Intuition

Once an op is registered, each time it actually *runs* inside a graph-tracked computation, the framework needs somewhere to keep the specific inputs and output from that one call — not the op's code (there's only one `MulOp` in existence), but this particular invocation of it, on these particular tensors. That per-call record is a `GraphNode`.

### Background

```cpp
// Captures one specific invocation of an op: its inputs (conceptually
// shared, Chapter 11.1 -- copied by value here for this scalar-sized
// demo), its output, and a grad field zero-initialized up front.
struct GraphNode {
    std::string op_name;
    std::vector<ScalarTensor> inputs;
    ScalarTensor output;
    float grad;
    bool requires_grad;

    GraphNode(std::string op_name_, std::vector<ScalarTensor> inputs_, ScalarTensor output_)
        : op_name(op_name_), inputs(inputs_), output(output_), requires_grad(true), grad(0.0f) {}
};
```

`grad` is initialized to zero at construction, not left unset — a small design choice with a large consequence Section 15.4 depends on directly: a node whose output is never actually consulted during a real backward pass still holds a well-defined, harmless `0.0` rather than uninitialized memory, so accidentally running its `backward` contributes exactly nothing to anyone's accumulated gradient.

### Worked Example 15.2.1 — The two nodes `w = x*y + x` actually produces, genuinely constructed

Compiled and run:

```bash
nvcc -arch=sm_80 02_graph_node.cu -o 02_graph_node
./02_graph_node
```

Genuinely compiled and run:

```
GraphNode #0: op_name=mul, inputs=[x=3.0, y=4.0], output=z=12.0, grad=0.0
GraphNode #1: op_name=add, inputs=[z=12.0, x=3.0], output=w=15.0, grad=0.0
```

The multiply produces exactly one `GraphNode`, and the addition produces a second one, whose `inputs` conceptually include `z` — the tensor node `#0` produced. Both nodes' `grad` fields read `0.0` at construction, exactly as designed.

### Worked Example 15.2.2 — What `requires_grad` actually gates, genuinely tested

Suppose `y` had instead been a fixed hyperparameter, `requires_grad = false`, while `x` still needs gradients. It's tempting to read this as "the graph now skips building anything for `y`" — but a `GraphNode` is recorded once *per operation*, not once per input, and Section 15.3's `record` method checks whether *any* input needs a gradient, not whether *every* input does. Genuinely compiled and run (from Section 15.3's file, reproduced here for context):

```
--- requires_grad gating: y frozen, x trainable ---
mul(x [requires_grad=true], y [requires_grad=false]) -> node created? true (nodes=1)
```

Since `x.requires_grad` is still `true`, the `mul` node above is still created in full, `y` included in its `inputs` — the framework still needs it to compute `∂z/∂x = y`, even though nobody will ever ask for `∂z/∂y`. The condition that actually skips a node entirely is stricter: *every single input* to that operation must have `requires_grad = false` simultaneously. One frozen operand next to one trainable operand still gets a full node; the constant-folding-out only happens when a whole operation is built entirely from values nobody will ever differentiate — the case Section 15.3 makes precise.

## 15.3 Graph Construction During Forward Pass `[FOUNDATIONAL]`

### Intuition

There is no separate "build the graph" step in this design — the graph is a side effect of running the forward pass once, normally. Every operation call either appends exactly one node or, if nothing about the call could ever need a gradient, appends nothing at all and returns a plain, untracked tensor.

### Background

```cpp
struct ComputationGraph {
    std::vector<GraphNode> nodes;

    ScalarTensor record(const std::string& op_name, std::vector<ScalarTensor> inputs, ScalarTensor output) {
        bool needs_grad = false;
        for (auto& t : inputs) if (t.requires_grad) needs_grad = true;
        if (!needs_grad) return output;   // constant-folded: no node created
        GraphNode node(op_name, inputs, output);
        output.requires_grad = true;
        output.grad_fn_index = (int)nodes.size();
        nodes.push_back(node);
        return output;
    }
};

ScalarTensor mul(ComputationGraph& graph, ScalarTensor a, ScalarTensor b) {
    ScalarTensor result(a.value * b.value);   // Chapter 12's elementwise_mul, scalar case
    return graph.record("mul", {a, b}, result);
}

ScalarTensor add(ComputationGraph& graph, ScalarTensor a, ScalarTensor b) {
    ScalarTensor result(a.value + b.value);
    return graph.record("add", {a, b}, result);
}
```

`record`'s loop is exactly the condition Worked Example 15.2.2 worked through: `needs_grad` becomes `true` the moment *any* input in the list has `requires_grad = true`, so skipping a node requires the loop to finish without ever setting it — every input frozen, simultaneously. `output.grad_fn_index = nodes.size()` is the other half of the bookkeeping: rather than the output tensor holding a pointer or reference back to the node that made it, it holds a plain integer — the position that node is about to occupy in `graph.nodes` — a flat-array-of-nodes design, as opposed to each tensor directly owning a reference to its own producing computation. The trade-off is real: a flat list threaded through every call (`ComputationGraph&` appears in every op's signature) makes the entire trace trivially inspectable — printing `graph.nodes` shows the whole computation — at the cost of every operation needing that graph object passed in explicitly, everywhere, rather than gradient machinery living entirely inside the tensor type itself.

### Worked Example 15.3.1 — Building the graph, one call at a time

Compiled and run:

```bash
nvcc -arch=sm_80 03_graph_construction.cu -o 03_graph_construction
./03_graph_construction
```

Genuinely compiled and run:

```
w = 15.0
graph.nodes.size() = 2
graph.nodes[0] = GraphNode("mul", output=12.0)
graph.nodes[1] = GraphNode("add", output=15.0)
```

Evaluating `w = add(graph, mul(graph, x, y), x)` with `x=3.0, y=4.0` does two things at once, exactly the way running the arithmetic by hand did: it produces `15.0`, and it leaves `graph.nodes` holding precisely the two `GraphNode`s from Worked Example 15.2.1, in the order they were computed. That list is the entire "history" this chapter opened by saying was normally thrown away — a literal, numeric trace of the computation that just ran, not anything symbolic or abstract.

Genuinely compiled and run in the same file, both boundary cases:

```
--- both operands frozen ---
mul(both requires_grad=false) -> node created? false (nodes=0), returned value=12.0
```

When *every* input is frozen, `record` returns the plain output with no node at all — `nodes=0` — genuinely confirming the exact boundary Worked Example 15.2.2 described in words.

## 15.4 Topological Sorting Implementation `[FOUNDATIONAL]`

### Intuition

Backward must undo the computation in the opposite order it was built: you cannot ask "how sensitive is `w` to `z`" until `w` exists, and `w`'s node was necessarily appended to `graph.nodes` *after* `z`'s node, because `z` had to be computed first to be usable as an input to the addition. That single fact — every node's inputs were computed, and therefore appended, strictly before it — means `graph.nodes` is already sorted in a valid forward order, and reversing it gives a valid order for backward with no separate sorting algorithm required, for this book's specific construction.

### Background

```cpp
// Forward execution order is already a valid topo-sort; backward just
// walks it in reverse.
std::vector<int> topological_backward_order(const ComputationGraph& graph) {
    std::vector<int> order;
    for (int i = (int)graph.nodes.size() - 1; i >= 0; i--) order.push_back(i);
    return order;
}
```

Notice what this function's signature does *not* take: a specific output tensor. It walks the entire `graph.nodes` list, unconditionally, from last-appended to first — there is no way for it to distinguish "give me the ancestors of `w`" from "give me the ancestors of some other tensor entirely." That is a real simplification, and it is safe *specifically* because every example in this book calls `backward()` from the single true final output of the whole graph that was ever built, with nothing else recorded alongside it. Reverse-mode automatic differentiation, implemented generally, doesn't get to assume that: a production engine builds its backward order with a depth-first search that starts *from the specific tensor `.backward()` was called on* and walks only that tensor's recorded parents, so nodes that were computed but never actually feed into the differentiated output are excluded from the walk entirely, by construction — not merely rendered harmless after the fact.

### Worked Example 15.4.1 — The running example's backward order, genuinely computed

```bash
nvcc -arch=sm_80 04_topological_order.cu -o 04_topological_order
./04_topological_order
```

Genuinely compiled and run:

```
forward order: [mul, add]
backward order (indices): [1, 0]
```

`topological_backward_order` returns `[1, 0]` — node `1` (`add`) first, node `0` (`mul`) second. Read that as a to-do list: first figure out how sensitive the final `w` is to `z` and to `x` through the addition; only afterward, once `z`'s sensitivity is known, ask how sensitive `z` is to `x` and `y` through the multiplication. Doing it the other way around would mean asking "how does `z` affect `x` and `y`" before knowing how much `z` itself matters to the answer — not yet a meaningful question.

### Worked Example 15.4.2 — A node that is recorded but isn't an ancestor of `w`, genuinely traced

Extend the example with one more line, computed but never used: `q = sub(graph, x, one)`, run *between* the `mul` call and the `add` call. Genuinely computed in the same run:

```
--- a node recorded but not an ancestor of w ---
graph2.nodes in append order: [mul, sub, add]
topological_backward_order: [2, 1, 0]
w2 = 15.0 (built from z2 and x2 only -- q2 never touched)
graph2.nodes[1] ("sub", producing q2=2.0).grad = 0.0 (never assigned, still 0)
```

`graph.nodes` now holds three entries in append order: `[mul(x,y)→z, sub(x,1)→q, add(z,x)→w]`. `topological_backward_order` still just reverses the whole list: `[2, 1, 0]` — `add` first, then `sub`, then `mul`. Node `1` (`sub`, producing `q`) is not an ancestor of `w` at all; `w` was built from `z` and `x`, never from `q`. Running `sub`'s `backward` anyway isn't wrong here, only wasted: `GraphNode`'s constructor zero-initializes every node's `grad`, and nothing in this trace ever assigns `q`'s node a nonzero gradient — genuinely confirmed above, `graph2.nodes[1].grad` reads exactly `0.0` — so `sub`'s backward would compute contributions of exactly `0.0` into `x`: harmless, but genuinely unnecessary work, and a concrete illustration of exactly the gap this section's Intuition described: reversing *every recorded node* is not the same operation as walking the *ancestors of one specific output*, even though the two happen to agree whenever nothing dead ever gets recorded alongside the computation you actually care about.

## 15.5 Complete Runnable Code

### File: `01_function_registration.cu`

```cpp
#include <cstdio>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>

// A minimal scalar stand-in for Chapter 6's Tensor, sized to this chapter's
// running scalar example -- the real framework's inputs/output would be
// full Tensors sharing buffers via Chapter 11.1's RefCountedBuffer, not
// copied by value the way this demo's ScalarTensor is.
struct ScalarTensor {
    float value;
    bool requires_grad;
    int grad_fn_index;
    ScalarTensor(float v, bool rg = false) : value(v), requires_grad(rg), grad_fn_index(-1) {}
};

// Every op the graph can record implements both directions -- the C++
// analogue of Mojo's `trait Differentiable`.
struct Differentiable {
    virtual ScalarTensor forward(const std::vector<ScalarTensor>& inputs) const = 0;
    virtual std::vector<float> backward(float grad_output, const std::vector<ScalarTensor>& inputs,
                                         const ScalarTensor& output) const = 0;
    virtual ~Differentiable() {}
};

struct MulOp : public Differentiable {
    ScalarTensor forward(const std::vector<ScalarTensor>& inputs) const override {
        return ScalarTensor(inputs[0].value * inputs[1].value);
    }
    std::vector<float> backward(float grad_output, const std::vector<ScalarTensor>& inputs,
                                 const ScalarTensor& output) const override {
        // dz/dx = y, dz/dy = x -- cannot be derived from grad_output alone.
        return { grad_output * inputs[1].value, grad_output * inputs[0].value };
    }
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

// Maps an op name to its registered Differentiable implementation.
struct OpRegistry {
    std::unordered_map<std::string, Differentiable*> ops;

    void register_op(const std::string& name, Differentiable* op) {
        ops[name] = op;
    }

    Differentiable* get(const std::string& name) {
        assert(ops.count(name) > 0 && "Unregistered op");
        return ops[name];   // operator[]: on a missing key in release mode, see below
    }
};

int main() {
    printf("=== Section 15.1: function registration, forward+backward bundled ===\n");

    OpRegistry registry;
    MulOp mul_op;
    AddOp add_op;
    registry.register_op("mul", &mul_op);
    registry.register_op("add", &add_op);

    Differentiable* found = registry.get("mul");
    printf("registry.get(\"mul\") succeeded: %s\n", (found == &mul_op) ? "true" : "false");

    // MulOp.backward genuinely needs the other operand, not just grad_output
    ScalarTensor x(3.0f, true), y(4.0f, true);
    ScalarTensor z = mul_op.forward({x, y});
    std::vector<float> grads = mul_op.backward(1.0f, {x, y}, z);
    printf("MulOp.backward(grad_output=1.0, x=3.0, y=4.0): dz/dx=%.1f (=y), dz/dy=%.1f (=x)\n",
           grads[0], grads[1]);
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_function_registration.cu -o 01_function_registration
./01_function_registration
```

### File: `01b_registry_trap.cu` — the debug-vs-release divergence from Worked Example 15.1.2

```cpp
#include <cstdio>
#include <cassert>
#include <string>
#include <unordered_map>

struct Differentiable { virtual ~Differentiable() {} };

struct OpRegistry {
    std::unordered_map<std::string, Differentiable*> ops;
    Differentiable* get(const std::string& name) {
        assert(ops.count(name) > 0 && "Unregistered op");
        return ops[name];
    }
};

int main() {
    OpRegistry registry;   // nothing registered at all
    printf("calling registry.get(\"relu\") on an empty registry...\n");
    Differentiable* result = registry.get("relu");
    printf("get() returned without aborting. result == nullptr? %s\n",
           (result == nullptr) ? "true" : "false");
    printf("registry.ops now has %zu entries (operator[] silently inserted one)\n", registry.ops.size());
    return 0;
}
```

```bash
# debug build: assert active
nvcc -arch=sm_80 01b_registry_trap.cu -o 01b_debug
./01b_debug
echo "exit code: $?"

# release build: NDEBUG strips the assert entirely
nvcc -arch=sm_80 -DNDEBUG 01b_registry_trap.cu -o 01b_release
./01b_release
echo "exit code: $?"
```

### File: `02_graph_node.cu`

```cpp
#include <cstdio>
#include <string>
#include <vector>

struct ScalarTensor {
    float value;
    bool requires_grad;
    int grad_fn_index;
    ScalarTensor(float v, bool rg = false) : value(v), requires_grad(rg), grad_fn_index(-1) {}
};

// Captures one specific invocation of an op: its inputs (conceptually
// shared, Chapter 11.1 -- copied by value here for this scalar-sized
// demo), its output, and a grad field zero-initialized up front.
struct GraphNode {
    std::string op_name;
    std::vector<ScalarTensor> inputs;
    ScalarTensor output;
    float grad;
    bool requires_grad;

    GraphNode(std::string op_name_, std::vector<ScalarTensor> inputs_, ScalarTensor output_)
        : op_name(op_name_), inputs(inputs_), output(output_), requires_grad(true), grad(0.0f) {}
};

int main() {
    printf("=== Section 15.2: GraphNode, the two nodes w = x*y + x actually produces ===\n");

    ScalarTensor x(3.0f, true), y(4.0f, true);
    float z_val = x.value * y.value;
    ScalarTensor z(z_val, true);
    GraphNode node0("mul", {x, y}, z);

    printf("GraphNode #0: op_name=%s, inputs=[x=%.1f, y=%.1f], output=z=%.1f, grad=%.1f\n",
           node0.op_name.c_str(), node0.inputs[0].value, node0.inputs[1].value,
           node0.output.value, node0.grad);

    float w_val = z.value + x.value;
    ScalarTensor w(w_val, true);
    GraphNode node1("add", {z, x}, w);

    printf("GraphNode #1: op_name=%s, inputs=[z=%.1f, x=%.1f], output=w=%.1f, grad=%.1f\n",
           node1.op_name.c_str(), node1.inputs[0].value, node1.inputs[1].value,
           node1.output.value, node1.grad);
    return 0;
}
```

```bash
nvcc -arch=sm_80 02_graph_node.cu -o 02_graph_node
./02_graph_node
```

### File: `03_graph_construction.cu`

```cpp
#include <cstdio>
#include <string>
#include <vector>

struct ScalarTensor {
    float value;
    bool requires_grad;
    int grad_fn_index;
    ScalarTensor(float v, bool rg = false) : value(v), requires_grad(rg), grad_fn_index(-1) {}
};

struct GraphNode {
    std::string op_name;
    std::vector<ScalarTensor> inputs;
    ScalarTensor output;
    float grad;
    bool requires_grad;
    GraphNode(std::string op_name_, std::vector<ScalarTensor> inputs_, ScalarTensor output_)
        : op_name(op_name_), inputs(inputs_), output(output_), requires_grad(true), grad(0.0f) {}
};

// There is no separate "build the graph" step -- the graph is a side
// effect of running the forward pass once, normally.
struct ComputationGraph {
    std::vector<GraphNode> nodes;

    ScalarTensor record(const std::string& op_name, std::vector<ScalarTensor> inputs, ScalarTensor output) {
        bool needs_grad = false;
        for (auto& t : inputs) if (t.requires_grad) needs_grad = true;
        if (!needs_grad) return output;   // constant-folded: no node created
        GraphNode node(op_name, inputs, output);
        output.requires_grad = true;
        output.grad_fn_index = (int)nodes.size();
        nodes.push_back(node);
        return output;
    }
};

ScalarTensor mul(ComputationGraph& graph, ScalarTensor a, ScalarTensor b) {
    ScalarTensor result(a.value * b.value);   // Chapter 12's elementwise_mul, scalar case
    return graph.record("mul", {a, b}, result);
}

ScalarTensor add(ComputationGraph& graph, ScalarTensor a, ScalarTensor b) {
    ScalarTensor result(a.value + b.value);
    return graph.record("add", {a, b}, result);
}

int main() {
    printf("=== Section 15.3: graph construction as a forward-pass side effect ===\n");

    ComputationGraph graph;
    ScalarTensor x(3.0f, true), y(4.0f, true);
    ScalarTensor w = add(graph, mul(graph, x, y), x);

    printf("w = %.1f\n", w.value);
    printf("graph.nodes.size() = %zu\n", graph.nodes.size());
    for (size_t i = 0; i < graph.nodes.size(); i++) {
        auto& n = graph.nodes[i];
        printf("graph.nodes[%zu] = GraphNode(\"%s\", output=%.1f)\n", i, n.op_name.c_str(), n.output.value);
    }

    // Worked Example 15.2.2: requires_grad gating -- one frozen operand
    // next to one trainable operand still produces a full node.
    printf("\n--- requires_grad gating: y frozen, x trainable ---\n");
    ComputationGraph graph2;
    ScalarTensor x2(3.0f, true), y2_frozen(4.0f, false);
    ScalarTensor z2 = mul(graph2, x2, y2_frozen);
    printf("mul(x [requires_grad=true], y [requires_grad=false]) -> node created? %s (nodes=%zu)\n",
           (graph2.nodes.size() > 0) ? "true" : "false", graph2.nodes.size());

    // Both frozen: no node at all.
    printf("\n--- both operands frozen ---\n");
    ComputationGraph graph3;
    ScalarTensor a3(3.0f, false), b3(4.0f, false);
    ScalarTensor c3 = mul(graph3, a3, b3);
    printf("mul(both requires_grad=false) -> node created? %s (nodes=%zu), returned value=%.1f\n",
           (graph3.nodes.size() > 0) ? "true" : "false", graph3.nodes.size(), c3.value);
    return 0;
}
```

```bash
nvcc -arch=sm_80 03_graph_construction.cu -o 03_graph_construction
./03_graph_construction
```

### File: `04_topological_order.cu`

```cpp
#include <cstdio>
#include <string>
#include <vector>

struct ScalarTensor {
    float value;
    bool requires_grad;
    int grad_fn_index;
    ScalarTensor(float v, bool rg = false) : value(v), requires_grad(rg), grad_fn_index(-1) {}
};

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
ScalarTensor sub(ComputationGraph& graph, ScalarTensor a, ScalarTensor b) {
    return graph.record("sub", {a, b}, ScalarTensor(a.value - b.value));
}

// Forward execution order is already a valid topo-sort; backward just
// walks it in reverse -- safe only because backward() is always called
// from the graph's single true final output in this book's usage.
std::vector<int> topological_backward_order(const ComputationGraph& graph) {
    std::vector<int> order;
    for (int i = (int)graph.nodes.size() - 1; i >= 0; i--) order.push_back(i);
    return order;
}

int main() {
    printf("=== Section 15.4: topological order is forward order, reversed ===\n");

    ComputationGraph graph;
    ScalarTensor x(3.0f, true), y(4.0f, true);
    ScalarTensor w = add(graph, mul(graph, x, y), x);

    std::vector<int> order = topological_backward_order(graph);
    printf("forward order: [");
    for (size_t i = 0; i < graph.nodes.size(); i++) printf("%s%s", graph.nodes[i].op_name.c_str(), i+1<graph.nodes.size()?", ":"");
    printf("]\n");
    printf("backward order (indices): [");
    for (size_t i = 0; i < order.size(); i++) printf("%d%s", order[i], i+1<order.size()?", ":"");
    printf("]\n");

    // Worked Example 15.4.2: an unrelated node recorded but not an ancestor of w
    printf("\n--- a node recorded but not an ancestor of w ---\n");
    ComputationGraph graph2;
    ScalarTensor x2(3.0f, true), y2(4.0f, true), one2(1.0f, false);
    ScalarTensor z2 = mul(graph2, x2, y2);
    ScalarTensor q2 = sub(graph2, x2, one2);   // computed, never used again
    ScalarTensor w2 = add(graph2, z2, x2);

    printf("graph2.nodes in append order: [");
    for (size_t i = 0; i < graph2.nodes.size(); i++) printf("%s%s", graph2.nodes[i].op_name.c_str(), i+1<graph2.nodes.size()?", ":"");
    printf("]\n");
    std::vector<int> order2 = topological_backward_order(graph2);
    printf("topological_backward_order: [");
    for (size_t i = 0; i < order2.size(); i++) printf("%d%s", order2[i], i+1<order2.size()?", ":"");
    printf("]\n");
    printf("w2 = %.1f (built from z2 and x2 only -- q2 never touched)\n", w2.value);
    printf("graph2.nodes[1] (\"%s\", producing q2=%.1f).grad = %.1f (never assigned, still 0)\n",
           graph2.nodes[1].op_name.c_str(), graph2.nodes[1].output.value, graph2.nodes[1].grad);
    return 0;
}
```

```bash
nvcc -arch=sm_80 04_topological_order.cu -o 04_topological_order
./04_topological_order
```

## Chapter Summary

A computational graph exists to keep, on purpose, the history a forward pass would otherwise throw away the moment it finishes. `Differentiable` bundles a `forward` and `backward` method into one interface so a graph can never record an operation it has no idea how to differentiate, and `backward`'s signature takes `inputs` (not just `grad_output`) because a local derivative like multiplication's — the *other* operand — genuinely cannot be produced any other way, genuinely confirmed by `MulOp::backward` returning `[y, x]`. `GraphNode` captures one specific invocation of an op: its inputs, its output, and a `grad` field zero-initialized up front rather than left unset. `ComputationGraph::record` appends a node only when *at least one* input requires a gradient — a looser condition than "this particular input is frozen," genuinely confirmed on both boundary cases: one trainable operand next to one frozen one still produces a full node, and only when every input is frozen does `record` skip creating a node at all. Because every node's inputs are computed, and therefore appended, strictly before it, the append order is already a valid forward topological order, and `topological_backward_order` gets away with simply reversing it — a simplification that holds exactly because this book never calls `backward()` on anything but the graph's true final output, and never leaves an unrelated, unused node's presence corrupt the result, genuinely confirmed by tracing a three-node graph where an unused `sub` node's `grad` field stayed at its initialized `0.0`. Finally, `OpRegistry::get`'s `assert`-based bounds check was genuinely compiled and run in both a debug build (a clean abort, exit code `134`) and a release build (a silent `nullptr`, plus a genuinely worse, C++-specific hazard: `std::unordered_map::operator[]` quietly inserting a phantom entry for the missing key it was only supposed to be reading).

## Self-Check Questions

1. `w = x*y + x` is built as `add(graph, mul(graph, x, y), x)` with `x=5.0, y=2.0`. Trace both `GraphNode`s exactly as Worked Example 15.2.1 did — report every field of `GraphNode #0` and `GraphNode #1`, including the final values of `z` and `w`.
2. Suppose *both* `x` and `y` have `requires_grad = false` for a call to `mul(graph, x, y)`. Walk through `record`'s loop step by step. Is a `GraphNode` created? What does the function return instead?
3. Extend Worked Example 15.4.2's three-node graph with a fourth call, `r = mul(graph, q, x)`, run after `add`. Does `r`'s node make `q`'s `sub` node an ancestor of `w`? Does it matter to `w`'s backward pass correctness that `sub`'s node exists in `graph.nodes` either way?
4. `OpRegistry::get("relu")` is called before any `ReluOp` has ever been registered, in a release build where `assert` has been compiled out via `NDEBUG`. What actually happens, in contrast to what would happen in a debug build — and what does `std::unordered_map::operator[]` specifically do to the registry's own contents as a side effect?
5. Why does `GraphNode`'s constructor initialize `grad` to `0.0f` rather than leaving it in whatever state a freshly constructed `float` member would otherwise hold? Connect your answer directly to what Worked Example 15.4.2 showed about running an unrelated node's `backward` by mistake.

## Where We Go Next

Chapter 16 (`part4/01-backward-function-implementation.md`) derives what each `backward` method in this chapter's `Differentiable` interface actually computes — starting with the exact numbers this chapter has been building toward, `x.grad = 5.0` and `y.grad = 3.0` for the running `w = x*y + x` example — by walking `topological_backward_order`'s list and applying the chain rule at each node in turn.

## Worked Solutions

**1.** `z = x*y = 5.0 × 2.0 = 10.0`; `w = z + x = 10.0 + 5.0 = 15.0`. `GraphNode #0`: `op_name="mul"`, `inputs=[x=5.0, y=2.0]`, `output=z=10.0`, `grad=0.0`. `GraphNode #1`: `op_name="add"`, `inputs=[z=10.0, x=5.0]`, `output=w=15.0`, `grad=0.0`.

**2.** The loop checks each of `x` and `y` in turn: `x.requires_grad` is `false`, so the `if` body never runs for it; `y.requires_grad` is also `false`, same result. After the loop, `needs_grad` is still `false` (its initial value), so the function hits `return output` immediately — no `GraphNode` is created, and the returned tensor is the plain, untracked multiplication result, indistinguishable from a value computed with no graph involved at all — genuinely confirmed by Worked Example 15.3.1's own "both operands frozen" test, which reports `nodes=0`.

**3.** No — `r = mul(graph, q, x)` makes `q` (and therefore `sub`'s node) an ancestor of `r`, not of `w`. `w` was already fully computed by the earlier `add(graph, z, x)` call, using only `z` and `x`; nothing about `w`'s own definition changes because a later, unrelated computation happens to reuse `q` and `x` afterward. It does not matter to `w`'s backward pass correctness whether `sub`'s node exists in `graph.nodes` — per Worked Example 15.4.2, an unrelated node's `backward` always contributes exactly `0.0` to whatever it touches, since its own `grad` field is never set to anything nonzero by a walk that never reaches it through `w`'s actual dependency chain.

**4.** In the debug build, `assert(ops.count(name) > 0 && "Unregistered op")` fires before the lookup, halting with `SIGABRT` (exit code `134`) and a message naming the exact failed condition. In a release build, the assertion is compiled out entirely — execution falls straight through to `return ops[name]`. `std::unordered_map::operator[]` specifically never throws on a missing key: it default-constructs one (`nullptr`, for a pointer value type) and *inserts* it into the map before returning it — genuinely confirmed above, the registry's own entry count grew from `0` to `1` as a side effect of a call that was only supposed to be reading. This is a real, C++-specific hazard beyond the generic "debug-only check disappears" pattern Chapter 11.2 first identified for `Arena`'s bounds checking.

**5.** Leaving `grad` unset (or relying on whatever a freshly constructed `float` defaults to, which in C++ is genuinely indeterminate for a non-initialized primitive member) would make a node's gradient state ambiguous the moment nothing writes to it — is `0.0` from initialization, or is `0.0` because a caller genuinely computed a zero contribution, or is it uninitialized memory that happens to hold something else entirely? Zero-initializing at construction removes that ambiguity outright, and it's precisely what makes Worked Example 15.4.2 safe: `sub`'s node never has its `grad` touched by anything downstream, so its `backward` reads a well-defined `0.0` and produces a well-defined, harmless zero contribution — not a gamble on whatever bytes happened to be sitting in memory when the node was constructed, the same gamble Chapter 14.1's `tensor_sum_incomplete` genuinely lost three different ways.
