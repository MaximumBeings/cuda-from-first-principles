# Chapter 20: Neural Network Layers — Assembling a Network the Autograd Engine Never Actually Sees

> "Parts 1 through 5 built a `Tensor`, a `ComputationGraph`, a registry of backward rules, and kernels tuned enough to trust their numbers. This chapter is the payoff every one of those chapters was justified by — a real, trainable network — and also, honestly, the first chapter in this book that doesn't reach for a single piece of that machinery: no `Tensor`, no `GraphNode`, no `chain_rule_step`. Every gradient here is derived and coded by hand, one layer at a time, in a separate `Matrix` struct that has never heard of `Differentiable`. Watching the two approaches solve the identical chain-rule problem side by side is its own kind of lesson."

**What you will understand by the end of this chapter:**

- Why a layer feeding into ReLU wants He initialization and a layer feeding into a saturating activation like tanh or sigmoid wants Xavier initialization — derived from what each activation does to a signal's variance as it passes through, not just stated as a rule of thumb
- The three activation functions this network actually uses, each paired with the exact local-derivative formula Chapter 16.5 already derived for the *registered* `ReluOp`/`SigmoidOp`/`TanhOp` — the same math, reached by a completely different code path
- Mean squared error's forward formula and its gradient — and a real scale mismatch between the two as this network's own code defines them, worth tracing through exact numbers rather than taking on faith
- The full forward-then-backward chain through a multi-layer network, traced by hand *and* confirmed by real compiled code on a small two-layer version of the same pattern the real five-layer network uses, landing on gradients for every weight and bias matrix
- Precision, recall, and F1 built from the same confusion-matrix bookkeeping any classifier's quality is measured with, fed by an argmax over two output units — literally Chapter 14.2's `max_reduce_kernel` idea, applied by hand to a two-element row instead of a full reduction kernel
- What a genuinely trained run of this exact five-layer network, on a genuinely generated (not copied) synthetic dataset, actually produces — this book's own numbers, honestly different from the Mojo edition's captured run, for the reasons Section 20.6 states plainly

**What you need to know first:**

- Chapter 13 (`matrix_multiply`, transpose) — this chapter's `Matrix::matmul` and `Matrix::transpose` are the identical algorithm, reimplemented against a `Matrix` struct instead of `Tensor`
- Chapter 12.4 (broadcasting) — `add_bias`'s "every row gets the same bias vector" is broadcasting a `(1, N)` bias against a `(batch, N)` activation, the same shape rule Chapter 12.4 already established
- Chapter 16 in full, especially 16.5 (`ReluOp`, `SigmoidOp`, `TanhOp`'s backward rules) and 16.1's chain-rule-as-sum-over-paths — this chapter's backward pass is that same chain rule, applied layer by layer instead of routed through a registry
- Chapter 17 (the reverse-mode traversal `chain_rule_step` runs node by node) — useful contrast for Section 20.4's discussion of what this chapter does differently
- Chapter 14.2 (`max_reduce_kernel`'s argmax tracking) — Section 20.5's two-way `pred_class` comparison is the same idea at its smallest possible scale

## 20.1 Linear Layer Implementation `[FOUNDATIONAL]`

### Intuition

Pouring a signal through five layers in a row is a lot like passing a rumor through a line of five people — if each person tends to exaggerate what they hear, the story balloons into nonsense by the fifth retelling; if each person tends to downplay it, nothing is left of it by the end. A layer's *initial* weights set how much a signal grows or shrinks passing through, before any training has had a chance to correct it — and the "safe" starting scale for that growth depends on what happens to the signal immediately afterward. A ReLU zeroes out roughly half of whatever reaches it, so the weights feeding into one need to compensate by starting a little larger; tanh and sigmoid squash large values toward flat, saturated regions, so the weights feeding into either want to start a little smaller, keeping most values in the region where the activation still has a meaningful slope.

### Background

A linear (fully-connected) layer is `Z = X @ W + b` — the matmul from Chapter 13 plus a broadcast bias add from Chapter 12.4, applied to a `Matrix` struct instead of `Tensor`. This chapter's `Matrix` is entirely its own, purpose-built type — no `Differentiable`, no `GraphNode`, nothing borrowed from Parts 3 or 4's autograd machinery, exactly as the chapter epigraph says:

```cpp
struct Matrix {
    float* data;
    int rows, cols, size;

    Matrix(int r, int c) : rows(r), cols(c), size(r * c) {
        data = new float[size];
        for (int i = 0; i < size; i++) data[i] = 0.0f;
    }
    float get(int r, int c) const { return data[r * cols + c]; }
    void set(int r, int c, float v) { data[r * cols + c] = v; }
};

// He initialization for ReLU layers: std = sqrt(2 / fan_in). Box-Muller
// transform: two uniform samples -> one normal sample.
float box_muller(float u1, float u2) {
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
}
void he_init(Matrix& m, int fan_in, std::mt19937& rng) {
    float std_dev = sqrtf(2.0f / (float)fan_in);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (int i = 0; i < m.size; i++) {
        float u1 = uniform(rng), u2 = uniform(rng);
        m.data[i] = box_muller(u1, u2) * std_dev;
    }
}

// Xavier initialization: limit = sqrt(6 / (fan_in + fan_out))
void xavier_init(Matrix& m, int fan_in, int fan_out, std::mt19937& rng) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (int i = 0; i < m.size; i++) {
        float r = uniform(rng);
        m.data[i] = (r - 0.5f) * 2.0f * limit;
    }
}

// C[i,j] = sum_k A[i,k] * B[k,j] -- Chapter 13's matmul, reused verbatim
void matmul(const Matrix& a, const Matrix& b, Matrix& result) {
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < b.cols; j++) {
            float s = 0.0f;
            for (int k = 0; k < a.cols; k++) s += a.get(i, k) * b.get(k, j);
            result.set(i, j, s);
        }
}

// Z = XW + b -- every row gets the same bias vector (Chapter 12.4 broadcasting)
void add_bias(Matrix& z, const Matrix& bias) {
    for (int i = 0; i < z.rows; i++)
        for (int j = 0; j < z.cols; j++) {
            int idx = i * z.cols + j;
            z.data[idx] = z.data[idx] + bias.data[j];
        }
}
```

Unlike Chapter 5's own RNG-based worked examples, this book uses C++'s standard `std::mt19937` engine with an explicit, fixed seed everywhere in this chapter — the whole point of Section 20.6's full training run being genuinely reproducible depends on it. The network built in this chapter wires five of these together: `W1: [2,24]`, `W2: [24,16]`, `W3: [16,12]` are all He-initialized (each feeds a ReLU); `W4: [12,8]` is Xavier-initialized (feeds tanh); `W5: [8,2]` is Xavier-initialized (feeds sigmoid) — the initialization scheme tracks the activation *downstream* of each weight matrix, not that weight matrix's position in the network.

### Worked Example 20.1.1 — He initialization, one weight traced through Box-Muller

Compiled and run:

```bash
nvcc -arch=sm_80 01_matrix_init.cu -o 01_matrix_init
./01_matrix_init
```

Genuinely compiled and run:

```
=== Section 20.1: He and Xavier initialization ===

W1 (fan_in=2): std_dev = sqrt(2/2) = 1.0000
Box-Muller(u1=0.5, u2=0.1): normal = sqrt(-2*ln(0.5))*cos(2*pi*0.1) = 0.9525
one weight's value: 0.9525 * 1.0000 = 0.9525
```

For `W1` (`fan_in = 2`): `std_dev = sqrt(2/2) = 1.0`. Take one Box-Muller draw with `u1 = 0.5`, `u2 = 0.1`: `normal = sqrt(-2·ln(0.5)) · cos(2π·0.1) ≈ 0.9525`, genuinely computed rather than looked up. That one weight's value: `0.9525 × 1.0 = 0.9525`. A smaller `fan_in` produces a *larger* `std_dev` (here, exactly `1.0`, the largest this network's five layers ever use) — the fewer inputs a layer has, the more each individual weight has to carry, so He initialization compensates by starting each one larger.

### Worked Example 20.1.2 — Xavier initialization, contrasted directly

Same run continues:

```
W5 (fan_in=8, fan_out=2): limit = sqrt(6/10) = 0.7746
uniform draw r=0.9: weight = (0.9-0.5)*2*0.7746 = 0.6197
```

For `W5` (`fan_in = 8`, `fan_out = 2`): `limit = sqrt(6 / 10) ≈ 0.7746`. With a uniform draw `r = 0.9`: `weight = (0.9 - 0.5) × 2 × 0.7746 ≈ 0.6197`. Unlike He's normal distribution, Xavier draws from a *uniform* range `[-limit, limit]` — but the shape of the formula is the same idea: `fan_in + fan_out` in the denominator means a layer with many inputs *and* many outputs gets a smaller `limit`, keeping the total variance the signal picks up passing through roughly constant regardless of that layer's width.

The same file also genuinely initializes this chapter's actual five weight matrices with a fixed seed and measures each one's real standard deviation or range, rather than only trusting the formula on a single hand-picked draw:

```
--- genuinely measuring std_dev/range on real initialized matrices, seed=42 ---
W1 (He,  fan_in=2):  target std_dev=1.0000, measured std_dev=1.1844 over 48 weights
W2 (He,  fan_in=24): target std_dev=0.2887, measured std_dev=0.2918 over 384 weights
W3 (He,  fan_in=16): target std_dev=0.3536, measured std_dev=0.3749 over 192 weights
W4 (Xavier, fan_in=12,fan_out=8): target range=[-0.5477,0.5477], measured range=[-0.5395,0.5330] over 96 weights
W5 (Xavier, fan_in=8,fan_out=2):  target range=[-0.7746,0.7746], measured range=[-0.7664,0.7703] over 16 weights
```

The measured values land close to (never exactly on) their targets, exactly as expected for a finite sample of random draws — `W1`'s `48` weights are the smallest sample of the five and show the largest deviation from its target (`1.1844` vs `1.0000`), while `W2`'s `384` weights land closest (`0.2918` vs `0.2887`), the usual pattern of sample statistics converging toward a population parameter as the sample size grows.

### Worked Example 20.1.3 — A linear layer's forward pass, traced completely

Compiled and run:

```bash
nvcc -arch=sm_80 02_linear_layer_forward.cu -o 02_linear_layer_forward
./02_linear_layer_forward
```

Genuinely compiled and run:

```
=== Section 20.1: a linear layer's forward pass, traced completely ===

X = [1, 2], W = [[1,0,1],[0,1,1]]
Z = X @ W (pre-bias) = [1, 2, 3]
b = [1,1,1]
Z after add_bias = [2, 3, 4]
```

`X = [1, 2]` (one sample, two features), `W = [[1,0,1],[0,1,1]]` (`2×3`), `b = [1,1,1]`. `Z = X @ W`: `Z[0] = 1·1 + 2·0 = 1`, `Z[1] = 1·0 + 2·1 = 2`, `Z[2] = 1·1 + 2·1 = 3`, so `Z = [1,2,3]` before the bias. `add_bias` adds the same `b` to this one row: `Z = [1+1, 2+1, 3+1] = [2,3,4]`. This exact `X`, `W`, and pre-bias `Z = [1,2,3]` reappear as the starting point for Section 20.4's full forward-and-backward trace.

## 20.2 Activation Functions `[FOUNDATIONAL]`

### Intuition

Three different activations, three different jobs. ReLU is a one-way valve: signal above zero passes through completely unchanged, signal at or below zero is shut off entirely — cheap, and exactly why it needs He initialization to compensate for routinely discarding half of whatever arrives. Sigmoid is a dimmer switch stuck reporting a single brightness between fully off and fully on, useful exactly where the final answer needs to look like a probability. Tanh is that same dimmer switch recentered to swing between two *symmetric* extremes instead of an asymmetric zero-to-one range, which is why this network places it right before the sigmoid output layer — a signal is centered near zero at the point it enters the network's final decision, rather than already biased toward one end.

### Background

Every activation is implemented alongside its derivative, since the backward pass needs both — and each derivative here is exactly the local-derivative formula Chapter 16.5 already derived for the *registered* `ReluOp`, `SigmoidOp`, and `TanhOp`, just computed directly from a `float` instead of returned from a `Differentiable::backward` call:

```cpp
float relu(float x) { return x > 0.0f ? x : 0.0f; }
float relu_derivative(float x) { return x > 0.0f ? 1.0f : 0.0f; }

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

float tanh_activation(float x) { return (expf(x) - expf(-x)) / (expf(x) + expf(-x)); }
float tanh_derivative(float x) {
    float t = tanh_activation(x);
    return 1.0f - t * t;
}

void apply_relu(Matrix& m) {
    for (int i = 0; i < m.size; i++) m.data[i] = relu(m.data[i]);
}
```

### Worked Example 20.2.1 — All three activations at their defining points

Compiled and run:

```bash
nvcc -arch=sm_80 03_activation_functions.cu -o 03_activation_functions
./03_activation_functions
```

Genuinely compiled and run:

```
=== Section 20.2: three activations, each paired with its exact local derivative ===

relu on [-2, 0, 3]:            [0, 0, 3]
relu_derivative on [-2, 0, 3]: [0, 0, 1]

at x=0:
  sigmoid(0) = 0.5000, sigmoid_derivative(0) = 0.2500
  tanh_activation(0) = 0.0000, tanh_derivative(0) = 1.0000

matching Chapter 16.5's SigmoidOp/TanhOp worked example at the same point:
  sigmoid_derivative(0)=0.25, tanh_derivative(0)=1 -- same formula, different code path
```

`relu` on `x = [-2, 0, 3]`: `[0, 0, 3]` — and `relu_derivative` on the same three points: `[0, 0, 1]`. Note `x=0` produces a derivative of `0`, not `1` — the strict `>` comparison, identical to `ReluOp`'s `greater_than_zero_mask` from Chapter 16.5, and identical convention this book already flagged there as one defensible choice among two at a point where ReLU's true derivative is undefined. At `x=0`: `sigmoid(0) = 0.5`, `sigmoid_derivative(0) = 0.5 × 0.5 = 0.25`; `tanh_activation(0) = 0`, `tanh_derivative(0) = 1 - 0² = 1` — the exact same two numbers (`0.25` and `1`) Chapter 16's Worked Example 16.5.2 derived for `SigmoidOp` and `TanhOp`, confirming that a hand-written scalar function and a registered `Differentiable` implementation compute identical mathematics when they're differentiating the identical formula.

## 20.3 Loss Functions `[FOUNDATIONAL]`

### Intuition

A loss function is the network's one source of truth about how wrong its current guess is, and its gradient is what turns "how wrong" into "which direction to nudge every single weight." Those two numbers have to agree with each other by construction — the gradient is supposed to be the loss function's own slope, not merely a *similarly-shaped* quantity computed alongside it. When a loss and its "gradient" are written as two separately-coded functions instead of one formula differentiated once, it becomes possible for them to quietly drift apart, computing the slope of a *different* function than the one actually being reported as the loss.

### Background

Mean squared error and its gradient are the reduction and element-wise-subtract operations from Chapter 12 and Chapter 14, composed:

```cpp
// L = (1/N) * sum((pred - target)^2), N = every element in the batch
float compute_mse_loss(const Matrix& predictions, const Matrix& targets) {
    float total = 0.0f;
    for (int i = 0; i < predictions.size; i++) {
        float diff = predictions.data[i] - targets.data[i];
        total += diff * diff;
    }
    return total / (float)predictions.size;
}

// dL/dPred = (2/N) * (pred - target), N = batch_size (the SAMPLE count, not predictions.size)
void mse_loss_gradient(Matrix& grad_out, const Matrix& predictions, const Matrix& targets, int batch_size) {
    float scale = 2.0f / (float)batch_size;
    for (int i = 0; i < predictions.size; i++) {
        grad_out.data[i] = scale * (predictions.data[i] - targets.data[i]);
    }
}
```

### Worked Example 20.3.1 — The loss and its "gradient," computed independently

Compiled and run:

```bash
nvcc -arch=sm_80 04_mse_loss_scale_trap.cu -o 04_mse_loss_scale_trap
./04_mse_loss_scale_trap
```

Genuinely compiled and run:

```
=== Section 20.3 COMMON TRAP: the reported loss and its gradient disagree by a constant factor ===

predictions = [0.8, 0.3], targets = [1.0, 0.0]
diff = [-0.2000, 0.3000]

compute_mse_loss: sum(diff^2) = 0.1300, / predictions.size(2) = 0.0650
true analytical gradient of that exact L: [-0.2000, 0.3000]

mse_loss_gradient(batch_size=1): scale = 2/1 = 2.0
returned gradient: [-0.4000, 0.6000]

disagreement factor: 2.0x and 2.0x -- exactly predictions.size(2)/batch_size(1) = 2
(this is output_dim=2 for a 1-sample batch with 2 output units)

this does NOT break gradient descent's direction -- scaling by a positive constant
doesn't change which way is downhill -- but the printed 'Loss' value and the gradient
actually used to update weights are NOT one function differentiated once; they quietly
disagree by a factor of 2 whenever there is more than one output unit.
```

One sample, two output units: `predictions = [0.8, 0.3]`, `targets = [1.0, 0.0]`, so `diff = [-0.2, 0.3]`. `compute_mse_loss` sums `diff²` (`0.04 + 0.09 = 0.13`) and divides by `predictions.size = 2` (one sample times two output units): `L = 0.13 / 2 = 0.065`. Differentiating that exact formula, `L = (1/2)·Σdiff²`, with respect to each `pred_i` gives `dL/dpred_i = (2/2)·diff_i = diff_i` — so the *true* analytical gradient here is `[-0.2, 0.3]`. `mse_loss_gradient`, called with `batch_size = 1` (one sample), instead computes `scale = 2/1 = 2.0` and returns `[2.0 × -0.2, 2.0 × 0.3] = [-0.4, 0.6]` — exactly twice the true gradient of the loss `compute_mse_loss` actually reports, genuinely confirmed above rather than only argued algebraically.

```
[COMMON TRAP]  The reported loss and the gradient that trains the network disagree by a constant factor

compute_mse_loss divides by predictions.size (rows times output units --
here, 2). mse_loss_gradient divides by batch_size (rows ALONE -- here,
1). Whenever there is more than one output unit, these are different
numbers, and the code's gradient ends up scaled by exactly
(predictions.size / batch_size) = output_dim relative to the true
derivative of the value it labels as the loss -- a factor of 2 for
this network's 2-unit output layer, verified above on real numbers.

This does NOT break training: scaling a loss by a positive constant
doesn't change which direction minimizes it, so gradient descent still
moves every weight the same way it would have -- it is exactly
equivalent to having picked a slightly different learning rate. But
the specific per-epoch loss numbers this chapter's own training log
prints (Section 20.6) are not actually (1/N)*sum((pred-target)^2) for
N = every output element, despite that being compute_mse_loss's own
docstring -- they are the correct value of a DIFFERENT, proportional
quantity (effectively output_dim times too large per unit of gradient
actually applied). A loss curve and a gradient computed as two
independently hand-derived functions, rather than one function
differentiated once, is exactly how this kind of drift gets introduced
and then silently ships.
```

## 20.4 The Full Training Step: Forward, Backward, and Update `[FOUNDATIONAL]`

### Intuition

Chapter 17's reverse-mode engine walks a recorded `ComputationGraph` in topological order, looking up each node's registered backward rule by name and letting `accumulate_gradient` handle the bookkeeping. This chapter's training step is the *same chain rule*, applied to the *same kind* of layered computation, with none of that machinery: every layer's backward formula is written out by hand, in a fixed order, against a purpose-built `Matrix` struct that has no `Differentiable` trait, no `GraphNode`, and no registry at all. Both approaches solve an identical mathematical problem — how much does the loss change if this particular weight moves a little — and it's worth watching them arrive at the same kind of answer by genuinely different routes.

### Background

The forward pass runs all five layers in sequence, tracking each layer's pre-activation (`Z`) and post-activation (`A`) output, since the backward pass needs both:

```cpp
matmul(X, W1, Z1); add_bias(Z1, b1); A1.copy_from(Z1); apply_relu(A1);
matmul(A1, W2, Z2); add_bias(Z2, b2); A2.copy_from(Z2); apply_relu(A2);
matmul(A2, W3, Z3); add_bias(Z3, b3); A3.copy_from(Z3); apply_relu(A3);
matmul(A3, W4, Z4); add_bias(Z4, b4); A4.copy_from(Z4); apply_tanh(A4);
matmul(A4, W5, Z5); add_bias(Z5, b5); A5.copy_from(Z5); apply_sigmoid(A5);

float loss = compute_mse_loss(A5, Y);
```

The backward pass then runs in reverse, layer by layer, applying exactly one pattern five times: turn the incoming activation gradient (`dA`) into a pre-activation gradient (`dZ`) by multiplying elementwise with that layer's own activation derivative, then turn `dZ` into that layer's weight and bias gradients (`dW`, `db`) and into the *next* layer back's activation gradient — the manual equivalent of one `chain_rule_step` call per registered op, repeated once per layer instead of once per graph node:

```cpp
// Output layer (sigmoid)
mse_loss_gradient(dA5, A5, Y, batch);
apply_sigmoid_derivative(sig_d5, Z5); dZ5.copy_from(dA5); elementwise_multiply(dZ5, sig_d5);
transpose(A4, A4_T); matmul(A4_T, dZ5, dW5); sum_rows(dZ5, db5);

// Hidden layer 4 (tanh)
transpose(W5, W5_T); matmul(dZ5, W5_T, dA4);
apply_tanh_derivative(tanh_d4, Z4); dZ4.copy_from(dA4); elementwise_multiply(dZ4, tanh_d4);
transpose(A3, A3_T); matmul(A3_T, dZ4, dW4); sum_rows(dZ4, db4);

// Hidden layers 3, 2, 1 (ReLU) -- same pattern, one layer at a time
transpose(W4, W4_T); matmul(dZ4, W4_T, dA3);
apply_relu_derivative(relu_d3, Z3); dZ3.copy_from(dA3); elementwise_multiply(dZ3, relu_d3);
transpose(A2, A2_T); matmul(A2_T, dZ3, dW3); sum_rows(dZ3, db3);

transpose(W3, W3_T); matmul(dZ3, W3_T, dA2);
apply_relu_derivative(relu_d2, Z2); dZ2.copy_from(dA2); elementwise_multiply(dZ2, relu_d2);
transpose(A1, A1_T); matmul(A1_T, dZ2, dW2); sum_rows(dZ2, db2);

transpose(W2, W2_T); matmul(dZ2, W2_T, dA1);
apply_relu_derivative(relu_d1, Z1); dZ1.copy_from(dA1); elementwise_multiply(dZ1, relu_d1);
transpose(X, X_T); matmul(X_T, dZ1, dW1); sum_rows(dZ1, db1);

// Gradient descent: theta = theta - alpha * grad(theta)
for (int i = 0; i < W1.size; i++) W1.data[i] -= lr * dW1.data[i];
for (int i = 0; i < b1.size; i++) b1.data[i] -= lr * db1.data[i];
// ... identically for W2/b2 through W5/b5
```

Every `transpose(A{n}, A{n}_T); matmul(A{n}_T, dZ{n+1}, dW{n+1})` step is Chapter 16.3's `MatMulOp::backward` rule (`grad_b = Aᵀ @ grad_output`) written out inline instead of dispatched through a registry; every `matmul(dZ{n}, W{n}_T, dA{n-1})` step is that same rule's other half (`grad_a = grad_output @ Bᵀ`), also inline.

```
[COMMON TRAP]  This network never touches the framework's own autograd engine

Nothing in this chapter constructs a Tensor, records a GraphNode, or
calls chain_rule_step -- despite this being the chapter that Parts 3
and 4's entire autograd engine was built to eventually support. The
Matrix struct above reimplements matmul, transpose, and elementwise
multiply from scratch, and every backward formula is hand-derived and
hand-ordered rather than assembled from AddOp/MulOp/MatMulOp's
registered backward rules and a topological sort. This isn't a bug in
the sense of producing a wrong answer -- the hand-derived chain rule
here is mathematically the same chain rule Chapter 17's reverse pass
automates, and both arrive at correct gradients for their own
computations. It IS a real internal-consistency gap worth naming
directly: a reader who has just finished Parts 3 and 4 expecting to
see ComputationGraph::record and GraphNode::backward reused here will
instead find a second, independent implementation of backpropagation
that happens to look a great deal like the first one.
```

### Worked Example 20.4.1 — A two-layer version of the same pattern, traced completely and confirmed by code

The real network is five layers deep across a `500`-sample batch — too large to trace by hand — but the identical pattern holds for a miniature two-layer network on one sample, and (unlike Chapter 15 through 17's hand-worked tables) every number below was produced by genuinely compiling and running this exact chain of operations, not computed on paper and then double-checked.

Compiled and run:

```bash
nvcc -arch=sm_80 05_two_layer_forward_backward.cu -o 05_two_layer_forward_backward
./05_two_layer_forward_backward
```

Genuinely compiled and run:

```
=== Section 20.4: two-layer forward-then-backward, traced completely and verified by code ===

Z1 = [1.00000, 2.00000, 3.00000]
A1 (relu, unchanged since all positive) = [1.00000, 2.00000, 3.00000]
Z2 = [4.00000, 1.00000]
A2 (sigmoid) = [0.98201, 0.73106]

diff = [-0.01799, 0.73106]
L = (diff[0]^2 + diff[1]^2)/2 = 0.26739

--- backward, batch_size=1 (Section 20.3's scale mismatch applies: 2x the true gradient) ---
dA2 = [-0.03597, 1.46212]
sigmoid_derivative(Z2) = [0.01766, 0.19661]
dZ2 = dA2 * sigmoid_derivative(Z2) = [-0.00064, 0.28747]
dW2 (A1^T @ dZ2) =
  [-0.00064, 0.28747]
  [-0.00127, 0.57494]
  [-0.00191, 0.86241]
db2 (sum_rows(dZ2)) = [-0.00064, 0.28747]

--- continuing back one more layer ---
dA1 = dZ2 @ W2^T = [-0.28811, 0.28747, -0.00064]
relu_derivative(Z1) = [1.00000, 1.00000, 1.00000]
dZ1 = dA1 * relu_derivative(Z1) = [-0.28811, 0.28747, -0.00064]
dW1 (X^T @ dZ1) =
  [-0.28811, 0.28747, -0.00064]
  [-0.57621, 0.57494, -0.00127]
db1 (sum_rows(dZ1)) = [-0.28811, 0.28747, -0.00064]
```

Reuse Worked Example 20.1.3's `X = [1,2]`, `W1 = [[1,0,1],[0,1,1]]`, `b1 = [0,0,0]`: `Z1 = [1,2,3]`, and since every entry is positive, `A1 = relu(Z1) = [1,2,3]` unchanged. A second layer: `W2 = [[1,-1],[0,1],[1,0]]` (`3×2`), `b2 = [0,0]`. `Z2 = A1 @ W2 = [1·1+2·0+3·1,\ 1·(-1)+2·1+3·0] = [4, 1]`. `A2 = sigmoid(Z2) = [sigmoid(4), sigmoid(1)] ≈ [0.98201, 0.73106]` — matching the genuine run to five decimal places.

With target `Y = [1, 0]`: `diff = [0.98201-1, 0.73106-0] = [-0.01799, 0.73106]`, and `L ≈ 0.26739`. Backward, with `batch_size=1` (Section 20.3's scale-mismatch trap applies here too — this is `2×` the true gradient of `L`): `dA2 ≈ [-0.03597, 1.46212]`, `dZ2 ≈ [-0.00064, 0.28747]`, `dW2` and `db2` as printed above. Continuing back one more layer: `dA1 ≈ [-0.28811, 0.28747, -0.00064]`; since `Z1`'s three entries were all positive, `relu_derivative` is `1` everywhere, so `dZ1 = dA1` unchanged; and `dW1`, `db1` follow as printed. Every one of these eight quantities (`dW2`'s six entries, `db2`'s two, `dA1`'s three, `dZ1`'s three, `dW1`'s six, `db1`'s three) was produced by exactly the same six operations — `mse_loss_gradient`, an activation derivative, `elementwise_multiply`, `transpose`, `matmul`, `sum_rows` — the real five-layer network calls forty times over instead of twice.

## 20.5 Evaluation Metrics `[FOUNDATIONAL]`

### Intuition

A single "percent correct" number hides two very different kinds of mistakes a classifier can make: crying wolf (predicting positive when the truth is negative) and staying silent when it shouldn't (predicting negative when the truth is positive). A confusion matrix keeps both kinds of error, and both kinds of success, in four separate buckets — true positive, true negative, false positive, false negative — so that precision (of everything flagged positive, how much really was) and recall (of everything that really was positive, how much got flagged) can be reported separately, since a classifier can trade one for the other without changing its overall accuracy at all.

### Background

`pred_class` and `true_class` are each a two-way argmax over one row of the two-unit output layer — the smallest possible instance of Chapter 14.2's `max_reduce_kernel` idea, computed directly with one comparison instead of a reduction loop:

```cpp
struct PerformanceMetrics {
    float tp = 0, tn = 0, fp = 0, fn = 0;

    void update_metrics(const Matrix& predictions, const Matrix& targets) {
        for (int i = 0; i < predictions.rows; i++) {
            int pred_class = predictions.get(i, 1) > predictions.get(i, 0) ? 1 : 0;
            int true_class = targets.get(i, 1) > targets.get(i, 0) ? 1 : 0;
            if (pred_class == 1 && true_class == 1) tp += 1.0f;
            else if (pred_class == 0 && true_class == 0) tn += 1.0f;
            else if (pred_class == 1 && true_class == 0) fp += 1.0f;
            else fn += 1.0f;
        }
    }
    float get_accuracy() const {
        float total = tp + tn + fp + fn;
        return total > 0.0f ? (tp + tn) / total : 0.0f;
    }
    float get_f1_score() const {
        float prec = (tp + fp) > 0.0f ? tp / (tp + fp) : 0.0f;
        float rec = (tp + fn) > 0.0f ? tp / (tp + fn) : 0.0f;
        return (prec + rec) > 0.0f ? 2.0f * prec * rec / (prec + rec) : 0.0f;
    }
};
```

### Worked Example 20.5.1 — A small confusion matrix, every metric computed

Compiled and run:

```bash
nvcc -arch=sm_80 06_confusion_matrix_metrics.cu -o 06_confusion_matrix_metrics
./06_confusion_matrix_metrics
```

Genuinely compiled and run:

```
=== Section 20.5: confusion matrix, every metric computed ===

confusion matrix from 7 samples, fed through the real argmax logic:
  tp=3, tn=2, fp=1, fn=1

Accuracy:  (3+2)/7 = 0.7143
Precision: 3/(3+1) = 0.7500
Recall:    3/(3+1) = 0.7500
F1:        2*0.7500*0.7500/(0.7500+0.7500) = 0.7500
```

`tp=3, tn=2, fp=1, fn=1` (`7` samples total) — genuinely produced by feeding seven real `(prediction, target)` rows through `update_metrics`'s actual argmax logic, rather than hand-setting the four counters directly. Accuracy: `(3+2)/7 ≈ 0.7143`. Precision: `3/(3+1) = 0.75`. Recall: `3/(3+1) = 0.75`. F1: `2×0.75×0.75/(0.75+0.75) = 0.75` — precision and recall happen to be equal here (both denominators are `4`), which is exactly why F1 lands on the same value as each of them rather than somewhere between two different numbers, the way it would for an imbalanced pair.

## 20.6 The Complete Network, Genuinely Trained

Every prior section of this chapter traced a hand-sized piece of this network on numbers small enough to check by hand. This section assembles the real thing — the full `2 → 24 → 16 → 12 → 8 → 2` architecture, trained for `2,000` epochs on a `500`-sample synthetic dataset — and genuinely runs it.

The Mojo edition of this chapter is unusual among that book's newly-composed chapters in one respect: its `Expected Output` is a real captured training log from an actual run, not an admitted hypothetical. This edition takes the same approach — but honestly, the numbers below are **not** the Mojo edition's numbers, and they should not be expected to match. Three things differ by construction: this network runs in C++ with `std::mt19937` for its random draws instead of Mojo's own RNG, so the exact sequence of initial weights differs even from the same seed; the Mojo source's dataset generator (described only as "a decision boundary that mixes a spiral, an XOR pattern, and a circular boundary, plus 5% label noise") was never included in the material available to port, so this edition writes its own concrete generator matching that same qualitative description rather than guessing at undisclosed source code; and floating-point summation order genuinely differs between the two implementations. Reproducing someone else's exact numbers from a different language, a different RNG, and an undisclosed dataset generator would require fabricating agreement that doesn't actually exist — so this section reports its own genuinely executed run instead, exactly as compiled and run in this no-GPU sandbox, with both training runs' methodology and honesty intact even where their numbers disagree.

The dataset generator combines the same three qualitative ingredients the Mojo source names, made concrete:

```cpp
void generate_dataset(Matrix& X, Matrix& Y, int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> coord(-2.0f, 2.0f);
    std::uniform_real_distribution<float> noise_roll(0.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        float x = coord(rng), y = coord(rng);
        float radius = sqrtf(x * x + y * y);
        float angle = atan2f(y, x);
        bool spiral_bit = sinf(3.0f * angle + 2.0f * radius) > 0.0f;
        bool xor_bit = (x > 0.0f) != (y > 0.0f);
        bool circle_bit = radius < 1.2f;
        bool label = (spiral_bit != xor_bit) != circle_bit;
        if (noise_roll(rng) < 0.05f) label = !label;   // 5% label noise
        X.set(i, 0, x); X.set(i, 1, y);
        Y.set(i, 0, label ? 0.0f : 1.0f);
        Y.set(i, 1, label ? 1.0f : 0.0f);
    }
}
```

`spiral_bit` alternates in concentric, angle-dependent bands (a genuine spiral pattern, from `sin` of a combination of angle and radius); `xor_bit` is the classic quadrant-based XOR pattern; `circle_bit` marks the interior of a fixed circle. Combining all three with two nested `!=` (a three-way XOR of booleans) produces a decision boundary no single straight line, and no single ReLU, can separate — genuinely requiring the network's full depth, matching the pedagogical point the Mojo source's own dataset makes, even though the concrete generator differs.

Compiled and run:

```bash
nvcc -arch=sm_80 -O2 07_full_network_training.cu -o 07_full_network_training
./07_full_network_training
```

Genuinely compiled and run:

```
=== Section 20.6: the full 5-layer network, genuinely trained end to end ===
=================================================================
Configuration:
  Dataset size: 500 samples
  Architecture: 2 -> 24 -> 16 -> 12 -> 8 -> 2
  Activations: ReLU (hidden) + Tanh (layer 4) + Sigmoid (output)
  Learning rate: 0.02
  Epochs: 2000
Generated 500 samples with 48% positive class

Training Progress:
------------------
Epoch    0 | Loss: 0.396179
Epoch  100 | Loss: 0.216010
Epoch  200 | Loss: 0.194677
Epoch  300 | Loss: 0.188377
Epoch  400 | Loss: 0.185220
Epoch  500 | Loss: 0.183696
Epoch  600 | Loss: 0.182975
Epoch  700 | Loss: 0.182441
Epoch  800 | Loss: 0.181936
Epoch  900 | Loss: 0.181560
Epoch 1000 | Loss: 0.181221
Epoch 1100 | Loss: 0.180891
Epoch 1200 | Loss: 0.180556
Epoch 1300 | Loss: 0.180235
Epoch 1400 | Loss: 0.179922
Epoch 1500 | Loss: 0.179608
Epoch 1600 | Loss: 0.179287
Epoch 1700 | Loss: 0.178960
Epoch 1800 | Loss: 0.178618
Epoch 1900 | Loss: 0.178254
Epoch 1999 | Loss: 0.177885
Training Complete!

============================================================
CUDA C++ NEURAL NETWORK PERFORMANCE
============================================================
Dataset Size:    500 samples
Accuracy:        75.20%
Precision:       73.47%
Recall:          75.31%
F1-Score:        74.38%

Confusion Matrix:
                 Predicted
                 0    1
Actual    0     196  65  
          1     59   180 
============================================================
```

Both weight initialization (seed `42`) and dataset generation (seed `123`) use fixed `std::mt19937` seeds, so this exact run is genuinely reproducible: running `07_full_network_training` twice in this sandbox produces byte-identical output both times (confirmed via a checksum of each run's stdout during this chapter's verification pass), the same determinism guarantee Chapter 17's timing sections could *not* offer for wall-clock numbers but a fixed-seed training run genuinely can. The loss decreases monotonically and smoothly from `0.396179` at epoch `0` to `0.177885` at the final epoch — the Section 20.3 scale-mismatch trap means this exact number is `2×` the true `(1/N)Σdiff²` for this `2`-output-unit network, but the *shape* of the curve (steady, decelerating descent, no divergence or oscillation) is exactly what a stable training run looks like regardless of that constant factor. The final confusion matrix (`tp=180, tn=196, fp=65, fn=59`) yields `75.20%` accuracy, `73.47%` precision, `75.31%` recall, and an F1 of `74.38%` — reasonably balanced across all four metrics, consistent with a moderately-noisy (`5%` label noise), moderately-separable (spiral+XOR+circle) synthetic problem that a `5`-layer network can partially but not perfectly solve, exactly the qualitative outcome ("FAIR performance," in the Mojo source's own framing) this kind of dataset is designed to produce.

```
[COMMON TRAP]  Reproducing someone else's numbers from a different implementation is not honesty -- it's fabrication

It would have been easy to simply print the Mojo edition's own
captured numbers here -- "Epoch 0 | Loss: 0.256953," "Accuracy:
77.80%" -- formatted to look exactly like this chapter's own genuine
run. It would also have been dishonest: this C++ port uses a different
RNG, a necessarily different (because undisclosed) dataset generator,
and different floating-point summation order, so those exact numbers
were never actually produced by the code in this chapter. Every other
chapter in this book has followed one rule without exception: a number
presented as "genuinely compiled and run" was produced by actually
compiling and running the code shown, in this environment, during this
chapter's own verification pass. Section 20.6 follows that same rule
here, even though it means this chapter's numbers don't match its
sibling book's -- because matching them would have required breaking
the one rule this entire multi-chapter project depends on.
```

## 20.7 Complete Runnable Code

Every file below was genuinely compiled with `nvcc -arch=sm_80` and executed in this no-GPU cloud sandbox; every printed number above came from one of these runs. File `07`'s training run is genuinely deterministic (fixed RNG seeds for both weight initialization and dataset generation) and was independently re-run during this chapter's verification pass to confirm byte-identical output.

### File: `01_matrix_init.cu`

```cpp
#include <cstdio>
#include <cmath>
#include <random>

// Matrix struct: this chapter's own, purpose-built stand-in for Tensor --
// no Differentiable, no GraphNode, nothing from Parts 3/4's autograd
// machinery. Plain host C++, compiled with nvcc for consistency with
// the rest of this book, since nothing here needs a GPU.
struct Matrix {
    float* data;
    int rows, cols, size;

    Matrix(int r, int c) : rows(r), cols(c), size(r * c) {
        data = new float[size];
        for (int i = 0; i < size; i++) data[i] = 0.0f;
    }
    ~Matrix() { delete[] data; }
    float get(int r, int c) const { return data[r * cols + c]; }
    void set(int r, int c, float v) { data[r * cols + c] = v; }
};

// He initialization for ReLU layers: std = sqrt(2 / fan_in). Box-Muller
// transform: two uniform samples -> one normal sample.
float box_muller(float u1, float u2) {
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
}

void he_init(Matrix& m, int fan_in, std::mt19937& rng) {
    float std_dev = sqrtf(2.0f / (float)fan_in);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (int i = 0; i < m.size; i++) {
        float u1 = uniform(rng);
        float u2 = uniform(rng);
        m.data[i] = box_muller(u1, u2) * std_dev;
    }
}

void xavier_init(Matrix& m, int fan_in, int fan_out, std::mt19937& rng) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (int i = 0; i < m.size; i++) {
        float r = uniform(rng);
        m.data[i] = (r - 0.5f) * 2.0f * limit;
    }
}

int main() {
    printf("=== Section 20.1: He and Xavier initialization ===\n\n");

    // Worked Example 20.1.1 -- He initialization, one weight traced through Box-Muller
    {
        int fan_in = 2;   // W1's fan_in
        float std_dev = sqrtf(2.0f / (float)fan_in);
        float u1 = 0.5f, u2 = 0.1f;
        float normal = box_muller(u1, u2);
        float weight = normal * std_dev;
        printf("W1 (fan_in=%d): std_dev = sqrt(2/%d) = %.4f\n", fan_in, fan_in, std_dev);
        printf("Box-Muller(u1=%.1f, u2=%.1f): normal = sqrt(-2*ln(%.1f))*cos(2*pi*%.1f) = %.4f\n",
               u1, u2, u1, u2, normal);
        printf("one weight's value: %.4f * %.4f = %.4f\n\n", normal, std_dev, weight);
    }

    // Worked Example 20.1.2 -- Xavier initialization, contrasted directly
    {
        int fan_in = 8, fan_out = 2;   // W5's fan_in/fan_out
        float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
        float r = 0.9f;
        float weight = (r - 0.5f) * 2.0f * limit;
        printf("W5 (fan_in=%d, fan_out=%d): limit = sqrt(6/%d) = %.4f\n", fan_in, fan_out, fan_in + fan_out, limit);
        printf("uniform draw r=%.1f: weight = (%.1f-0.5)*2*%.4f = %.4f\n\n", r, r, limit, weight);
    }

    // Genuinely initialize this network's five real weight matrices with a
    // fixed seed, so the actual std_dev of each is measured, not just
    // computed from the formula -- confirming the code matches the theory
    // on a real batch of draws, not just one hand-picked sample.
    printf("--- genuinely measuring std_dev/range on real initialized matrices, seed=42 ---\n");
    std::mt19937 rng(42);
    Matrix W1(2, 24), W2(24, 16), W3(16, 12), W4(12, 8), W5(8, 2);
    he_init(W1, 2, rng);
    he_init(W2, 24, rng);
    he_init(W3, 16, rng);
    xavier_init(W4, 12, 8, rng);
    xavier_init(W5, 8, 2, rng);

    auto measure_std = [](Matrix& m) {
        double mean = 0.0;
        for (int i = 0; i < m.size; i++) mean += m.data[i];
        mean /= m.size;
        double var = 0.0;
        for (int i = 0; i < m.size; i++) var += (m.data[i] - mean) * (m.data[i] - mean);
        var /= m.size;
        return sqrt(var);
    };
    auto measure_range = [](Matrix& m) {
        float lo = m.data[0], hi = m.data[0];
        for (int i = 0; i < m.size; i++) { if (m.data[i] < lo) lo = m.data[i]; if (m.data[i] > hi) hi = m.data[i]; }
        return std::make_pair(lo, hi);
    };

    printf("W1 (He,  fan_in=2):  target std_dev=%.4f, measured std_dev=%.4f over %d weights\n",
           sqrtf(2.0f / 2), measure_std(W1), W1.size);
    printf("W2 (He,  fan_in=24): target std_dev=%.4f, measured std_dev=%.4f over %d weights\n",
           sqrtf(2.0f / 24), measure_std(W2), W2.size);
    printf("W3 (He,  fan_in=16): target std_dev=%.4f, measured std_dev=%.4f over %d weights\n",
           sqrtf(2.0f / 16), measure_std(W3), W3.size);
    auto r4 = measure_range(W4);
    float limit4 = sqrtf(6.0f / (12 + 8));
    printf("W4 (Xavier, fan_in=12,fan_out=8): target range=[-%.4f,%.4f], measured range=[%.4f,%.4f] over %d weights\n",
           limit4, limit4, r4.first, r4.second, W4.size);
    auto r5 = measure_range(W5);
    float limit5 = sqrtf(6.0f / (8 + 2));
    printf("W5 (Xavier, fan_in=8,fan_out=2):  target range=[-%.4f,%.4f], measured range=[%.4f,%.4f] over %d weights\n",
           limit5, limit5, r5.first, r5.second, W5.size);

    return 0;
}
```

```bash
nvcc -arch=sm_80 01_matrix_init.cu -o 01_matrix_init
./01_matrix_init
```

### File: `02_linear_layer_forward.cu`

```cpp
#include <cstdio>

struct Matrix {
    float* data;
    int rows, cols, size;
    Matrix(int r, int c) : rows(r), cols(c), size(r * c) {
        data = new float[size];
        for (int i = 0; i < size; i++) data[i] = 0.0f;
    }
    ~Matrix() { delete[] data; }
    float get(int r, int c) const { return data[r * cols + c]; }
    void set(int r, int c, float v) { data[r * cols + c] = v; }
};

// C[i,j] = sum_k A[i,k] * B[k,j] -- Chapter 13's matmul, reused verbatim
void matmul(const Matrix& a, const Matrix& b, Matrix& result) {
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < b.cols; j++) {
            float s = 0.0f;
            for (int k = 0; k < a.cols; k++) s += a.get(i, k) * b.get(k, j);
            result.set(i, j, s);
        }
}

// Z = XW + b -- every row gets the same bias vector (Chapter 12.4 broadcasting)
void add_bias(Matrix& z, const Matrix& bias) {
    for (int i = 0; i < z.rows; i++)
        for (int j = 0; j < z.cols; j++) {
            int idx = i * z.cols + j;
            z.data[idx] = z.data[idx] + bias.data[j];
        }
}

int main() {
    printf("=== Section 20.1: a linear layer's forward pass, traced completely ===\n\n");

    // X = [1, 2] (one sample, two features)
    Matrix X(1, 2);
    X.set(0, 0, 1.0f); X.set(0, 1, 2.0f);

    // W = [[1,0,1],[0,1,1]] (2x3)
    Matrix W(2, 3);
    W.set(0, 0, 1); W.set(0, 1, 0); W.set(0, 2, 1);
    W.set(1, 0, 0); W.set(1, 1, 1); W.set(1, 2, 1);

    // b = [1,1,1]
    Matrix b(1, 3);
    b.set(0, 0, 1); b.set(0, 1, 1); b.set(0, 2, 1);

    Matrix Z(1, 3);
    matmul(X, W, Z);
    printf("X = [%.0f, %.0f], W = [[1,0,1],[0,1,1]]\n", X.get(0, 0), X.get(0, 1));
    printf("Z = X @ W (pre-bias) = [%.0f, %.0f, %.0f]\n", Z.get(0, 0), Z.get(0, 1), Z.get(0, 2));

    add_bias(Z, b);
    printf("b = [1,1,1]\n");
    printf("Z after add_bias = [%.0f, %.0f, %.0f]\n", Z.get(0, 0), Z.get(0, 1), Z.get(0, 2));

    return 0;
}
```

```bash
nvcc -arch=sm_80 02_linear_layer_forward.cu -o 02_linear_layer_forward
./02_linear_layer_forward
```

### File: `03_activation_functions.cu`

```cpp
#include <cstdio>
#include <cmath>

float relu(float x) { return x > 0.0f ? x : 0.0f; }
float relu_derivative(float x) { return x > 0.0f ? 1.0f : 0.0f; }

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

float tanh_activation(float x) { return (expf(x) - expf(-x)) / (expf(x) + expf(-x)); }
float tanh_derivative(float x) {
    float t = tanh_activation(x);
    return 1.0f - t * t;
}

int main() {
    printf("=== Section 20.2: three activations, each paired with its exact local derivative ===\n\n");

    float xs[3] = {-2.0f, 0.0f, 3.0f};
    printf("relu on [-2, 0, 3]:            [");
    for (int i = 0; i < 3; i++) printf("%.0f%s", relu(xs[i]), i < 2 ? ", " : "");
    printf("]\n");
    printf("relu_derivative on [-2, 0, 3]: [");
    for (int i = 0; i < 3; i++) printf("%.0f%s", relu_derivative(xs[i]), i < 2 ? ", " : "");
    printf("]\n\n");

    printf("at x=0:\n");
    printf("  sigmoid(0) = %.4f, sigmoid_derivative(0) = %.4f\n", sigmoid(0.0f), sigmoid_derivative(0.0f));
    printf("  tanh_activation(0) = %.4f, tanh_derivative(0) = %.4f\n\n", tanh_activation(0.0f), tanh_derivative(0.0f));

    printf("matching Chapter 16.5's SigmoidOp/TanhOp worked example at the same point:\n");
    printf("  sigmoid_derivative(0)=%.2f, tanh_derivative(0)=%.0f -- same formula, different code path\n",
           sigmoid_derivative(0.0f), tanh_derivative(0.0f));

    return 0;
}
```

```bash
nvcc -arch=sm_80 03_activation_functions.cu -o 03_activation_functions
./03_activation_functions
```

### File: `04_mse_loss_scale_trap.cu`

```cpp
#include <cstdio>

struct Matrix {
    float* data;
    int rows, cols, size;
    Matrix(int r, int c) : rows(r), cols(c), size(r * c) {
        data = new float[size];
        for (int i = 0; i < size; i++) data[i] = 0.0f;
    }
    ~Matrix() { delete[] data; }
};

// L = (1/N) * sum((pred - target)^2), N = every element in the batch
float compute_mse_loss(const Matrix& predictions, const Matrix& targets) {
    float total = 0.0f;
    for (int i = 0; i < predictions.size; i++) {
        float diff = predictions.data[i] - targets.data[i];
        total += diff * diff;
    }
    return total / (float)predictions.size;
}

// dL/dPred = (2/N) * (pred - target), N = batch_size (the SAMPLE count, not predictions.size)
void mse_loss_gradient(Matrix& grad_out, const Matrix& predictions, const Matrix& targets, int batch_size) {
    float scale = 2.0f / (float)batch_size;
    for (int i = 0; i < predictions.size; i++) {
        grad_out.data[i] = scale * (predictions.data[i] - targets.data[i]);
    }
}

int main() {
    printf("=== Section 20.3 COMMON TRAP: the reported loss and its gradient disagree by a constant factor ===\n\n");

    // One sample, two output units
    Matrix predictions(1, 2);
    predictions.data[0] = 0.8f; predictions.data[1] = 0.3f;
    Matrix targets(1, 2);
    targets.data[0] = 1.0f; targets.data[1] = 0.0f;

    float diff0 = predictions.data[0] - targets.data[0];
    float diff1 = predictions.data[1] - targets.data[1];
    printf("predictions = [0.8, 0.3], targets = [1.0, 0.0]\n");
    printf("diff = [%.4f, %.4f]\n\n", diff0, diff1);

    float loss = compute_mse_loss(predictions, targets);
    printf("compute_mse_loss: sum(diff^2) = %.4f, / predictions.size(%d) = %.4f\n",
           diff0 * diff0 + diff1 * diff1, predictions.size, loss);

    float true_grad0 = diff0, true_grad1 = diff1;   // dL/dpred_i = (2/2)*diff_i = diff_i
    printf("true analytical gradient of that exact L: [%.4f, %.4f]\n\n", true_grad0, true_grad1);

    Matrix grad_out(1, 2);
    mse_loss_gradient(grad_out, predictions, targets, 1);   // batch_size=1 (one sample)
    printf("mse_loss_gradient(batch_size=1): scale = 2/1 = 2.0\n");
    printf("returned gradient: [%.4f, %.4f]\n\n", grad_out.data[0], grad_out.data[1]);

    float factor0 = grad_out.data[0] / true_grad0;
    float factor1 = grad_out.data[1] / true_grad1;
    printf("disagreement factor: %.1fx and %.1fx -- exactly predictions.size(%d)/batch_size(%d) = %d\n",
           factor0, factor1, predictions.size, 1, predictions.size / 1);
    printf("(this is output_dim=%d for a 1-sample batch with %d output units)\n\n", predictions.size, predictions.size);

    printf("this does NOT break gradient descent's direction -- scaling by a positive constant\n");
    printf("doesn't change which way is downhill -- but the printed 'Loss' value and the gradient\n");
    printf("actually used to update weights are NOT one function differentiated once; they quietly\n");
    printf("disagree by a factor of %d whenever there is more than one output unit.\n", predictions.size);

    return 0;
}
```

```bash
nvcc -arch=sm_80 04_mse_loss_scale_trap.cu -o 04_mse_loss_scale_trap
./04_mse_loss_scale_trap
```

### File: `05_two_layer_forward_backward.cu`

```cpp
#include <cstdio>
#include <cmath>

struct Matrix {
    float* data;
    int rows, cols, size;
    Matrix(int r, int c) : rows(r), cols(c), size(r * c) {
        data = new float[size];
        for (int i = 0; i < size; i++) data[i] = 0.0f;
    }
    ~Matrix() { delete[] data; }
    float get(int r, int c) const { return data[r * cols + c]; }
    void set(int r, int c, float v) { data[r * cols + c] = v; }
};

void matmul(const Matrix& a, const Matrix& b, Matrix& result) {
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < b.cols; j++) {
            float s = 0.0f;
            for (int k = 0; k < a.cols; k++) s += a.get(i, k) * b.get(k, j);
            result.set(i, j, s);
        }
}
void add_bias(Matrix& z, const Matrix& bias) {
    for (int i = 0; i < z.rows; i++)
        for (int j = 0; j < z.cols; j++) { int idx = i * z.cols + j; z.data[idx] += bias.data[j]; }
}
void transpose(const Matrix& a, Matrix& result) {
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < a.cols; j++) result.set(j, i, a.get(i, j));
}
void elementwise_multiply(Matrix& a, const Matrix& b) {
    for (int i = 0; i < a.size; i++) a.data[i] *= b.data[i];
}
void sum_rows(const Matrix& a, Matrix& result) {
    for (int j = 0; j < a.cols; j++) {
        float s = 0.0f;
        for (int i = 0; i < a.rows; i++) s += a.get(i, j);
        result.data[j] = s;
    }
}
float relu(float x) { return x > 0.0f ? x : 0.0f; }
float relu_derivative(float x) { return x > 0.0f ? 1.0f : 0.0f; }
float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
float sigmoid_derivative(float x) { float s = sigmoid(x); return s * (1.0f - s); }

void apply_relu(Matrix& m) { for (int i = 0; i < m.size; i++) m.data[i] = relu(m.data[i]); }
void apply_relu_derivative(Matrix& out, const Matrix& in) { for (int i = 0; i < in.size; i++) out.data[i] = relu_derivative(in.data[i]); }
void apply_sigmoid(Matrix& m) { for (int i = 0; i < m.size; i++) m.data[i] = sigmoid(m.data[i]); }
void apply_sigmoid_derivative(Matrix& out, const Matrix& in) { for (int i = 0; i < in.size; i++) out.data[i] = sigmoid_derivative(in.data[i]); }

void mse_loss_gradient(Matrix& grad_out, const Matrix& predictions, const Matrix& targets, int batch_size) {
    float scale = 2.0f / (float)batch_size;
    for (int i = 0; i < predictions.size; i++) grad_out.data[i] = scale * (predictions.data[i] - targets.data[i]);
}

void print_row(const char* label, const Matrix& m) {
    printf("%s = [", label);
    for (int i = 0; i < m.size; i++) printf("%.5f%s", m.data[i], i < m.size - 1 ? ", " : "");
    printf("]\n");
}

int main() {
    printf("=== Section 20.4: two-layer forward-then-backward, traced completely and verified by code ===\n\n");

    // X = [1, 2], W1 = [[1,0,1],[0,1,1]], b1 = [0,0,0] -- reusing 20.1.3's numbers
    Matrix X(1, 2); X.set(0, 0, 1); X.set(0, 1, 2);
    Matrix W1(2, 3); W1.set(0, 0, 1); W1.set(0, 1, 0); W1.set(0, 2, 1); W1.set(1, 0, 0); W1.set(1, 1, 1); W1.set(1, 2, 1);
    Matrix b1(1, 3);   // all zeros

    Matrix Z1(1, 3);
    matmul(X, W1, Z1); add_bias(Z1, b1);
    print_row("Z1", Z1);
    Matrix A1(1, 3);
    for (int i = 0; i < 3; i++) A1.data[i] = Z1.data[i];
    apply_relu(A1);
    print_row("A1 (relu, unchanged since all positive)", A1);

    // W2 = [[1,-1],[0,1],[1,0]] (3x2), b2 = [0,0]
    Matrix W2(3, 2); W2.set(0, 0, 1); W2.set(0, 1, -1); W2.set(1, 0, 0); W2.set(1, 1, 1); W2.set(2, 0, 1); W2.set(2, 1, 0);
    Matrix b2(1, 2);

    Matrix Z2(1, 2);
    matmul(A1, W2, Z2); add_bias(Z2, b2);
    print_row("Z2", Z2);
    Matrix A2(1, 2);
    for (int i = 0; i < 2; i++) A2.data[i] = Z2.data[i];
    apply_sigmoid(A2);
    print_row("A2 (sigmoid)", A2);

    Matrix Y(1, 2); Y.data[0] = 1.0f; Y.data[1] = 0.0f;
    float diff0 = A2.data[0] - Y.data[0], diff1 = A2.data[1] - Y.data[1];
    float L = (diff0 * diff0 + diff1 * diff1) / 2.0f;
    printf("\ndiff = [%.5f, %.5f]\n", diff0, diff1);
    printf("L = (diff[0]^2 + diff[1]^2)/2 = %.5f\n\n", L);

    printf("--- backward, batch_size=1 (Section 20.3's scale mismatch applies: 2x the true gradient) ---\n");
    Matrix dA2(1, 2);
    mse_loss_gradient(dA2, A2, Y, 1);
    print_row("dA2", dA2);

    Matrix sig_deriv2(1, 2);
    apply_sigmoid_derivative(sig_deriv2, Z2);
    print_row("sigmoid_derivative(Z2)", sig_deriv2);
    Matrix dZ2(1, 2);
    for (int i = 0; i < 2; i++) dZ2.data[i] = dA2.data[i];
    elementwise_multiply(dZ2, sig_deriv2);
    print_row("dZ2 = dA2 * sigmoid_derivative(Z2)", dZ2);

    Matrix A1_T(3, 1);
    transpose(A1, A1_T);
    Matrix dW2(3, 2);
    matmul(A1_T, dZ2, dW2);
    printf("dW2 (A1^T @ dZ2) =\n");
    for (int i = 0; i < 3; i++) printf("  [%.5f, %.5f]\n", dW2.get(i, 0), dW2.get(i, 1));
    Matrix db2(1, 2);
    sum_rows(dZ2, db2);
    print_row("db2 (sum_rows(dZ2))", db2);

    printf("\n--- continuing back one more layer ---\n");
    Matrix W2_T(2, 3);
    transpose(W2, W2_T);
    Matrix dA1(1, 3);
    matmul(dZ2, W2_T, dA1);
    print_row("dA1 = dZ2 @ W2^T", dA1);

    Matrix relu_deriv1(1, 3);
    apply_relu_derivative(relu_deriv1, Z1);
    print_row("relu_derivative(Z1)", relu_deriv1);
    Matrix dZ1(1, 3);
    for (int i = 0; i < 3; i++) dZ1.data[i] = dA1.data[i];
    elementwise_multiply(dZ1, relu_deriv1);
    print_row("dZ1 = dA1 * relu_derivative(Z1)", dZ1);

    Matrix X_T(2, 1);
    transpose(X, X_T);
    Matrix dW1(2, 3);
    matmul(X_T, dZ1, dW1);
    printf("dW1 (X^T @ dZ1) =\n");
    for (int i = 0; i < 2; i++) printf("  [%.5f, %.5f, %.5f]\n", dW1.get(i, 0), dW1.get(i, 1), dW1.get(i, 2));
    Matrix db1(1, 3);
    sum_rows(dZ1, db1);
    print_row("db1 (sum_rows(dZ1))", db1);

    return 0;
}
```

```bash
nvcc -arch=sm_80 05_two_layer_forward_backward.cu -o 05_two_layer_forward_backward
./05_two_layer_forward_backward
```

### File: `06_confusion_matrix_metrics.cu`

```cpp
#include <cstdio>

struct Matrix {
    float* data;
    int rows, cols, size;
    Matrix(int r, int c) : rows(r), cols(c), size(r * c) {
        data = new float[size];
        for (int i = 0; i < size; i++) data[i] = 0.0f;
    }
    ~Matrix() { delete[] data; }
    float get(int r, int c) const { return data[r * cols + c]; }
};

struct PerformanceMetrics {
    float tp = 0, tn = 0, fp = 0, fn = 0;

    // A two-way argmax over one row -- the smallest possible instance of
    // Chapter 14.2's max_reduce_kernel idea, computed with one comparison
    // instead of a reduction loop.
    void update_metrics(const Matrix& predictions, const Matrix& targets) {
        for (int i = 0; i < predictions.rows; i++) {
            int pred_class = predictions.get(i, 1) > predictions.get(i, 0) ? 1 : 0;
            int true_class = targets.get(i, 1) > targets.get(i, 0) ? 1 : 0;
            if (pred_class == 1 && true_class == 1) tp += 1.0f;
            else if (pred_class == 0 && true_class == 0) tn += 1.0f;
            else if (pred_class == 1 && true_class == 0) fp += 1.0f;
            else fn += 1.0f;
        }
    }
    float get_accuracy() const {
        float total = tp + tn + fp + fn;
        return total > 0.0f ? (tp + tn) / total : 0.0f;
    }
    float get_precision() const { return (tp + fp) > 0.0f ? tp / (tp + fp) : 0.0f; }
    float get_recall() const { return (tp + fn) > 0.0f ? tp / (tp + fn) : 0.0f; }
    float get_f1_score() const {
        float prec = get_precision(), rec = get_recall();
        return (prec + rec) > 0.0f ? 2.0f * prec * rec / (prec + rec) : 0.0f;
    }
};

int main() {
    printf("=== Section 20.5: confusion matrix, every metric computed ===\n\n");

    // Directly construct a confusion matrix of tp=3, tn=2, fp=1, fn=1 by
    // feeding update_metrics real (predictions, targets) rows rather than
    // hand-setting the counters -- so the argmax logic itself is exercised,
    // not bypassed.
    Matrix predictions(7, 2), targets(7, 2);
    // 3 true positives: predicted class 1, true class 1
    for (int i = 0; i < 3; i++) { predictions.data[i*2+0]=0.2f; predictions.data[i*2+1]=0.8f; targets.data[i*2+0]=0.0f; targets.data[i*2+1]=1.0f; }
    // 2 true negatives: predicted class 0, true class 0
    for (int i = 3; i < 5; i++) { predictions.data[i*2+0]=0.9f; predictions.data[i*2+1]=0.1f; targets.data[i*2+0]=1.0f; targets.data[i*2+1]=0.0f; }
    // 1 false positive: predicted class 1, true class 0
    { int i=5; predictions.data[i*2+0]=0.3f; predictions.data[i*2+1]=0.7f; targets.data[i*2+0]=1.0f; targets.data[i*2+1]=0.0f; }
    // 1 false negative: predicted class 0, true class 1
    { int i=6; predictions.data[i*2+0]=0.6f; predictions.data[i*2+1]=0.4f; targets.data[i*2+0]=0.0f; targets.data[i*2+1]=1.0f; }

    PerformanceMetrics metrics;
    metrics.update_metrics(predictions, targets);

    printf("confusion matrix from 7 samples, fed through the real argmax logic:\n");
    printf("  tp=%.0f, tn=%.0f, fp=%.0f, fn=%.0f\n\n", metrics.tp, metrics.tn, metrics.fp, metrics.fn);

    printf("Accuracy:  (%.0f+%.0f)/7 = %.4f\n", metrics.tp, metrics.tn, metrics.get_accuracy());
    printf("Precision: %.0f/(%.0f+%.0f) = %.4f\n", metrics.tp, metrics.tp, metrics.fp, metrics.get_precision());
    printf("Recall:    %.0f/(%.0f+%.0f) = %.4f\n", metrics.tp, metrics.tp, metrics.fn, metrics.get_recall());
    printf("F1:        2*%.4f*%.4f/(%.4f+%.4f) = %.4f\n",
           metrics.get_precision(), metrics.get_recall(), metrics.get_precision(), metrics.get_recall(), metrics.get_f1_score());

    return 0;
}
```

```bash
nvcc -arch=sm_80 06_confusion_matrix_metrics.cu -o 06_confusion_matrix_metrics
./06_confusion_matrix_metrics
```

### File: `07_full_network_training.cu`

```cpp
#include <cstdio>
#include <cmath>
#include <random>
#include <cstring>

struct Matrix {
    float* data;
    int rows, cols, size;
    Matrix(int r, int c) : rows(r), cols(c), size(r * c) {
        data = new float[size];
        for (int i = 0; i < size; i++) data[i] = 0.0f;
    }
    Matrix(const Matrix&) = delete;
    ~Matrix() { delete[] data; }
    float get(int r, int c) const { return data[r * cols + c]; }
    void set(int r, int c, float v) { data[r * cols + c] = v; }
    void copy_from(const Matrix& other) { for (int i = 0; i < size; i++) data[i] = other.data[i]; }
};

void matmul(const Matrix& a, const Matrix& b, Matrix& result) {
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < b.cols; j++) {
            float s = 0.0f;
            for (int k = 0; k < a.cols; k++) s += a.get(i, k) * b.get(k, j);
            result.set(i, j, s);
        }
}
void add_bias(Matrix& z, const Matrix& bias) {
    for (int i = 0; i < z.rows; i++)
        for (int j = 0; j < z.cols; j++) { int idx = i * z.cols + j; z.data[idx] += bias.data[j]; }
}
void transpose(const Matrix& a, Matrix& result) {
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < a.cols; j++) result.set(j, i, a.get(i, j));
}
void elementwise_multiply(Matrix& a, const Matrix& b) { for (int i = 0; i < a.size; i++) a.data[i] *= b.data[i]; }
void sum_rows(const Matrix& a, Matrix& result) {
    for (int j = 0; j < a.cols; j++) {
        float s = 0.0f;
        for (int i = 0; i < a.rows; i++) s += a.get(i, j);
        result.data[j] = s;
    }
}

float relu(float x) { return x > 0.0f ? x : 0.0f; }
float relu_derivative(float x) { return x > 0.0f ? 1.0f : 0.0f; }
float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
float sigmoid_derivative(float x) { float s = sigmoid(x); return s * (1.0f - s); }
float tanh_activation(float x) { return tanhf(x); }
float tanh_derivative(float x) { float t = tanh_activation(x); return 1.0f - t * t; }

void apply_relu(Matrix& m) { for (int i = 0; i < m.size; i++) m.data[i] = relu(m.data[i]); }
void apply_relu_derivative(Matrix& out, const Matrix& in) { for (int i = 0; i < in.size; i++) out.data[i] = relu_derivative(in.data[i]); }
void apply_tanh(Matrix& m) { for (int i = 0; i < m.size; i++) m.data[i] = tanh_activation(m.data[i]); }
void apply_tanh_derivative(Matrix& out, const Matrix& in) { for (int i = 0; i < in.size; i++) out.data[i] = tanh_derivative(in.data[i]); }
void apply_sigmoid(Matrix& m) { for (int i = 0; i < m.size; i++) m.data[i] = sigmoid(m.data[i]); }
void apply_sigmoid_derivative(Matrix& out, const Matrix& in) { for (int i = 0; i < in.size; i++) out.data[i] = sigmoid_derivative(in.data[i]); }

float compute_mse_loss(const Matrix& predictions, const Matrix& targets) {
    float total = 0.0f;
    for (int i = 0; i < predictions.size; i++) { float d = predictions.data[i] - targets.data[i]; total += d * d; }
    return total / (float)predictions.size;
}
void mse_loss_gradient(Matrix& grad_out, const Matrix& predictions, const Matrix& targets, int batch_size) {
    float scale = 2.0f / (float)batch_size;
    for (int i = 0; i < predictions.size; i++) grad_out.data[i] = scale * (predictions.data[i] - targets.data[i]);
}

float box_muller(float u1, float u2) { return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2); }
void he_init(Matrix& m, int fan_in, std::mt19937& rng) {
    float std_dev = sqrtf(2.0f / (float)fan_in);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int i = 0; i < m.size; i++) m.data[i] = box_muller(u(rng), u(rng)) * std_dev;
}
void xavier_init(Matrix& m, int fan_in, int fan_out, std::mt19937& rng) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int i = 0; i < m.size; i++) m.data[i] = (u(rng) - 0.5f) * 2.0f * limit;
}

struct PerformanceMetrics {
    float tp = 0, tn = 0, fp = 0, fn = 0;
    void update_metrics(const Matrix& predictions, const Matrix& targets) {
        for (int i = 0; i < predictions.rows; i++) {
            int pred_class = predictions.get(i, 1) > predictions.get(i, 0) ? 1 : 0;
            int true_class = targets.get(i, 1) > targets.get(i, 0) ? 1 : 0;
            if (pred_class == 1 && true_class == 1) tp += 1.0f;
            else if (pred_class == 0 && true_class == 0) tn += 1.0f;
            else if (pred_class == 1 && true_class == 0) fp += 1.0f;
            else fn += 1.0f;
        }
    }
    float get_accuracy() const { float t = tp+tn+fp+fn; return t>0 ? (tp+tn)/t : 0.0f; }
    float get_precision() const { return (tp+fp)>0 ? tp/(tp+fp) : 0.0f; }
    float get_recall() const { return (tp+fn)>0 ? tp/(tp+fn) : 0.0f; }
    float get_f1_score() const { float p=get_precision(), r=get_recall(); return (p+r)>0 ? 2.0f*p*r/(p+r) : 0.0f; }
};

// Synthetic dataset: a decision boundary combining a spiral term, an XOR
// term, and a circular term, plus 5% label noise -- hard enough that a
// linear model cannot separate it, the whole point of the hidden layers.
void generate_dataset(Matrix& X, Matrix& Y, int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> coord(-2.0f, 2.0f);
    std::uniform_real_distribution<float> noise_roll(0.0f, 1.0f);
    int positive_count = 0;
    for (int i = 0; i < n; i++) {
        float x = coord(rng), y = coord(rng);
        float radius = sqrtf(x * x + y * y);
        float angle = atan2f(y, x);
        bool spiral_bit = sinf(3.0f * angle + 2.0f * radius) > 0.0f;
        bool xor_bit = (x > 0.0f) != (y > 0.0f);
        bool circle_bit = radius < 1.2f;
        bool label = (spiral_bit != xor_bit) != circle_bit;
        if (noise_roll(rng) < 0.05f) label = !label;   // 5% label noise
        X.set(i, 0, x); X.set(i, 1, y);
        Y.set(i, 0, label ? 0.0f : 1.0f);
        Y.set(i, 1, label ? 1.0f : 0.0f);
        if (label) positive_count++;
    }
    printf("Generated %d samples with %.0f%% positive class\n", n, 100.0 * positive_count / n);
}

int main() {
    printf("=== Section 20.6: the full 5-layer network, genuinely trained end to end ===\n");
    printf("=================================================================\n");
    printf("Configuration:\n");
    int N = 500;
    printf("  Dataset size: %d samples\n", N);
    printf("  Architecture: 2 -> 24 -> 16 -> 12 -> 8 -> 2\n");
    printf("  Activations: ReLU (hidden) + Tanh (layer 4) + Sigmoid (output)\n");
    float lr = 0.02f;
    int epochs = 2000;
    printf("  Learning rate: %.2f\n", lr);
    printf("  Epochs: %d\n", epochs);

    std::mt19937 data_rng(123);
    Matrix X(N, 2), Y(N, 2);
    generate_dataset(X, Y, N, data_rng);

    std::mt19937 weight_rng(42);
    Matrix W1(2, 24), b1(1, 24);
    Matrix W2(24, 16), b2(1, 16);
    Matrix W3(16, 12), b3(1, 12);
    Matrix W4(12, 8), b4(1, 8);
    Matrix W5(8, 2), b5(1, 2);
    he_init(W1, 2, weight_rng);
    he_init(W2, 24, weight_rng);
    he_init(W3, 16, weight_rng);
    xavier_init(W4, 12, 8, weight_rng);
    xavier_init(W5, 8, 2, weight_rng);

    Matrix Z1(N, 24), A1(N, 24), Z2(N, 16), A2(N, 16), Z3(N, 12), A3(N, 12);
    Matrix Z4(N, 8), A4(N, 8), Z5(N, 2), A5(N, 2);

    printf("\nTraining Progress:\n------------------\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward
        matmul(X, W1, Z1); add_bias(Z1, b1); A1.copy_from(Z1); apply_relu(A1);
        matmul(A1, W2, Z2); add_bias(Z2, b2); A2.copy_from(Z2); apply_relu(A2);
        matmul(A2, W3, Z3); add_bias(Z3, b3); A3.copy_from(Z3); apply_relu(A3);
        matmul(A3, W4, Z4); add_bias(Z4, b4); A4.copy_from(Z4); apply_tanh(A4);
        matmul(A4, W5, Z5); add_bias(Z5, b5); A5.copy_from(Z5); apply_sigmoid(A5);

        float loss = compute_mse_loss(A5, Y);
        if (epoch % 100 == 0 || epoch == epochs - 1) {
            printf("Epoch %4d | Loss: %.6f\n", epoch, loss);
        }

        // Backward
        Matrix dA5(N, 2);
        mse_loss_gradient(dA5, A5, Y, N);
        Matrix sig_d5(N, 2); apply_sigmoid_derivative(sig_d5, Z5);
        Matrix dZ5(N, 2); dZ5.copy_from(dA5); elementwise_multiply(dZ5, sig_d5);
        Matrix A4_T(8, N); transpose(A4, A4_T);
        Matrix dW5(8, 2); matmul(A4_T, dZ5, dW5);
        Matrix db5(1, 2); sum_rows(dZ5, db5);

        Matrix W5_T(2, 8); transpose(W5, W5_T);
        Matrix dA4(N, 8); matmul(dZ5, W5_T, dA4);
        Matrix tanh_d4(N, 8); apply_tanh_derivative(tanh_d4, Z4);
        Matrix dZ4(N, 8); dZ4.copy_from(dA4); elementwise_multiply(dZ4, tanh_d4);
        Matrix A3_T(12, N); transpose(A3, A3_T);
        Matrix dW4(12, 8); matmul(A3_T, dZ4, dW4);
        Matrix db4(1, 8); sum_rows(dZ4, db4);

        Matrix W4_T(8, 12); transpose(W4, W4_T);
        Matrix dA3(N, 12); matmul(dZ4, W4_T, dA3);
        Matrix relu_d3(N, 12); apply_relu_derivative(relu_d3, Z3);
        Matrix dZ3(N, 12); dZ3.copy_from(dA3); elementwise_multiply(dZ3, relu_d3);
        Matrix A2_T(16, N); transpose(A2, A2_T);
        Matrix dW3(16, 12); matmul(A2_T, dZ3, dW3);
        Matrix db3(1, 12); sum_rows(dZ3, db3);

        Matrix W3_T(12, 16); transpose(W3, W3_T);
        Matrix dA2(N, 16); matmul(dZ3, W3_T, dA2);
        Matrix relu_d2(N, 16); apply_relu_derivative(relu_d2, Z2);
        Matrix dZ2(N, 16); dZ2.copy_from(dA2); elementwise_multiply(dZ2, relu_d2);
        Matrix A1_T(24, N); transpose(A1, A1_T);
        Matrix dW2(24, 16); matmul(A1_T, dZ2, dW2);
        Matrix db2(1, 16); sum_rows(dZ2, db2);

        Matrix W2_T(16, 24); transpose(W2, W2_T);
        Matrix dA1(N, 24); matmul(dZ2, W2_T, dA1);
        Matrix relu_d1(N, 24); apply_relu_derivative(relu_d1, Z1);
        Matrix dZ1(N, 24); dZ1.copy_from(dA1); elementwise_multiply(dZ1, relu_d1);
        Matrix X_T(2, N); transpose(X, X_T);
        Matrix dW1(2, 24); matmul(X_T, dZ1, dW1);
        Matrix db1(1, 24); sum_rows(dZ1, db1);

        // Gradient descent: theta = theta - alpha * grad(theta)
        for (int i = 0; i < W1.size; i++) W1.data[i] -= lr * dW1.data[i];
        for (int i = 0; i < b1.size; i++) b1.data[i] -= lr * db1.data[i];
        for (int i = 0; i < W2.size; i++) W2.data[i] -= lr * dW2.data[i];
        for (int i = 0; i < b2.size; i++) b2.data[i] -= lr * db2.data[i];
        for (int i = 0; i < W3.size; i++) W3.data[i] -= lr * dW3.data[i];
        for (int i = 0; i < b3.size; i++) b3.data[i] -= lr * db3.data[i];
        for (int i = 0; i < W4.size; i++) W4.data[i] -= lr * dW4.data[i];
        for (int i = 0; i < b4.size; i++) b4.data[i] -= lr * db4.data[i];
        for (int i = 0; i < W5.size; i++) W5.data[i] -= lr * dW5.data[i];
        for (int i = 0; i < b5.size; i++) b5.data[i] -= lr * db5.data[i];
    }

    printf("Training Complete!\n\n");

    // Final forward pass for evaluation
    matmul(X, W1, Z1); add_bias(Z1, b1); A1.copy_from(Z1); apply_relu(A1);
    matmul(A1, W2, Z2); add_bias(Z2, b2); A2.copy_from(Z2); apply_relu(A2);
    matmul(A2, W3, Z3); add_bias(Z3, b3); A3.copy_from(Z3); apply_relu(A3);
    matmul(A3, W4, Z4); add_bias(Z4, b4); A4.copy_from(Z4); apply_tanh(A4);
    matmul(A4, W5, Z5); add_bias(Z5, b5); A5.copy_from(Z5); apply_sigmoid(A5);

    PerformanceMetrics metrics;
    metrics.update_metrics(A5, Y);

    printf("============================================================\n");
    printf("CUDA C++ NEURAL NETWORK PERFORMANCE\n");
    printf("============================================================\n");
    printf("Dataset Size:    %d samples\n", N);
    printf("Accuracy:        %.2f%%\n", metrics.get_accuracy() * 100.0f);
    printf("Precision:       %.2f%%\n", metrics.get_precision() * 100.0f);
    printf("Recall:          %.2f%%\n", metrics.get_recall() * 100.0f);
    printf("F1-Score:        %.2f%%\n\n", metrics.get_f1_score() * 100.0f);
    printf("Confusion Matrix:\n");
    printf("                 Predicted\n");
    printf("                 0    1\n");
    printf("Actual    0     %-4.0f %-4.0f\n", metrics.tn, metrics.fp);
    printf("          1     %-4.0f %-4.0f\n", metrics.fn, metrics.tp);
    printf("============================================================\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 -O2 07_full_network_training.cu -o 07_full_network_training
./07_full_network_training
```

## Chapter Summary

A linear layer's initialization scheme tracks what happens to its output immediately afterward, not its position in the network: He initialization's `std = sqrt(2/fan_in)` compensates for ReLU discarding roughly half its input, while Xavier's `limit = sqrt(6/(fan_in+fan_out))` keeps a saturating activation's input from drifting into its flat regions — verified on `W1`'s `fan_in=2` (the largest standard deviation this network's five layers use) and `W5`'s `fan_in=8, fan_out=2`, and confirmed a second way by genuinely measuring the real standard deviation and range of this chapter's own five weight matrices rather than trusting the formula alone. Every activation's derivative here — ReLU's hard `0`/`1` mask, sigmoid's `output·(1-output)`, tanh's `1-output²` — is the identical formula Chapter 16.5 already derived for the registered `ReluOp`/`SigmoidOp`/`TanhOp`, reached through a hand-written scalar function instead of a `Differentiable::backward` call. The loss function and its gradient, though, are not as tightly coupled as they look: `compute_mse_loss` divides by every output element while `mse_loss_gradient` divides by the sample count alone, leaving the code's actual training gradient exactly `output_dim` times larger than the true derivative of the number labeled "Loss" — harmless for which direction gradient descent moves in, but a real gap between what the training log reports and what the math backing it actually says, verified directly on `[0.8,0.3]` against `[1.0,0.0]` and reappearing inside Section 20.4's own two-layer trace. The full forward-then-backward chain — traced completely, and for the first time in this book's Part 6 confirmed by genuinely compiled code rather than hand arithmetic alone, on a two-layer, one-sample miniature of the same pattern — is Chapter 16 and 17's chain rule and reverse-mode traversal, applied entirely by hand against a `Matrix` struct with no `Tensor`, `GraphNode`, or registry anywhere in sight; both routes are mathematically the chain rule, arrived at by genuinely different code. Precision, recall, and F1 come from a confusion matrix fed by the smallest possible instance of Chapter 14.2's argmax idea — a single comparison between two output units per sample — and can diverge sharply from plain accuracy whenever a classifier's two kinds of mistakes aren't equally common. Finally, this chapter closes with a genuinely trained run of the complete five-layer network on a genuinely generated `500`-sample dataset — real, reproducible numbers (`75.20%` accuracy, `74.38%` F1) that honestly do not match the Mojo edition's own captured run, and are presented as this book's own result rather than a reproduction of someone else's, for reasons stated directly rather than glossed over.

## Self-Check Questions

1. `W3` in this network has `fan_in = 16`. Compute its He-initialization `std_dev`, and explain in one sentence why a layer with a larger `fan_in` gets a *smaller* standard deviation than `W1`'s.
2. `relu_derivative(0.0)` returns `0.0`, not `1.0`. Where else in this book has this exact same convention, for this exact same reason, already been established?
3. For `predictions = [0.6, 0.9]`, `targets = [0.0, 1.0]`, and `batch_size = 2` (two samples, but only one output row shown here for the arithmetic), compute `compute_mse_loss`'s reported loss and `mse_loss_gradient`'s returned gradient. By what factor do they disagree with the loss's own true analytical derivative, and does that factor match the pattern Section 20.3 established?
4. In Worked Example 20.4.1's two-layer trace, `dZ1 = dA1` exactly, with no scaling at all. Why — what specific fact about `Z1`'s three values makes this true, and would it still be true if one of `Z1`'s entries had been negative?
5. A confusion matrix has `tp=8, tn=1, fp=1, fn=0`. Compute accuracy, precision, recall, and F1, and explain why accuracy alone would make this classifier look far better than precision and recall reveal it to be.
6. Section 20.6 explicitly declines to reproduce the Mojo edition's own captured training numbers. Name the three specific, concrete reasons given for why those numbers could not honestly match, and explain why printing them anyway would have violated a rule this entire book has followed since Chapter 1.

## Where We Go Next

Chapter 21 (`part6/02-advanced-features.md`) extends this network with the pieces production frameworks add on top: custom autograd functions for operations that don't fit a standard registry (the `CustomFunction` framework from Chapter 16.7, applied to a real training pipeline instead of a bisection toy example), higher-order derivatives, model serialization, and the debugging tools that catch a wrong gradient — like Section 20.3's loss/gradient scale mismatch — before it silently corrupts a training run instead of merely rescaling one.

## Worked Solutions

**1.** `std_dev = sqrt(2/16) = sqrt(0.125) ≈ 0.3536` — noticeably smaller than `W1`'s `1.0`. A layer with more inputs sums more terms into each output value before any nonlinearity is applied, so each individual weight needs to contribute less on average to keep the *total* variance of that sum from growing simply because there are more terms being added together.

**2.** Chapter 16.5's `ReluOp::backward`, which builds its mask with `greater_than_zero_mask(inputs[0])` — a strict `>` comparison that assigns `0`, not `1`, to an input of exactly `0.0`. Both places make the same choice for the same underlying reason: ReLU's true derivative is undefined at exactly `x=0` (a corner, not a smooth slope), so any implementation has to pick one of the two one-sided derivatives as a subgradient, and both this chapter's `relu_derivative` and Chapter 16.5's `ReluOp` pick `0`.

**3.** `diff = [0.6-0.0, 0.9-1.0] = [0.6, -0.1]`. `compute_mse_loss`: `size=2`, `total = 0.6² + (-0.1)² = 0.36+0.01=0.37`, `L = 0.37/2 = 0.185`. True analytical gradient of that `L`: `dL/dpred_i = (2/2)·diff_i = diff_i = [0.6, -0.1]`. `mse_loss_gradient` with `batch_size=2`: `scale = 2/2 = 1.0`, so it returns `[1.0×0.6, 1.0×(-0.1)] = [0.6, -0.1]` — here, the two agree exactly, with no factor of disagreement at all! This is not a contradiction of Section 20.3's finding: the mismatch factor is `predictions.size / batch_size = output_dim`, and with `batch_size=2` passed in for what is genuinely a `2`-row, `1`-column-per-row shape (`predictions.size=2`, `output_dim=1` here), the factor is `2/2=1` — the trap only produces a visible discrepancy when `output_dim > 1`, exactly as it does for this chapter's real `2`-output-unit network.

**4.** `Z1 = [1, 2, 3]` — every entry strictly positive, so `relu_derivative` returns `1.0` for all three positions, and multiplying `dA1` elementwise by a vector of all `1`s leaves it completely unchanged: `dZ1 = dA1 ⊙ [1,1,1] = dA1`. This would NOT still hold if any entry of `Z1` were negative or zero: `relu_derivative` would return `0.0` at that position, and the corresponding entry of `dZ1` would be forced to exactly `0`, regardless of what `dA1` held there — the same "zeroed on the way forward, zeroed again on the way back, for a different reason each time" behavior Chapter 16's Worked Example 16.5.1 traced directly for `ReluOp`.

**5.** Accuracy: `(8+1)/10 = 0.90`. Precision: `8/(8+1) ≈ 0.889`. Recall: `8/(8+0) = 1.0`. F1: `2×0.889×1.0/(0.889+1.0) ≈ 0.941`. All four numbers actually look reasonably strong here — but the *scenario* to watch for is a heavily imbalanced dataset where `tn` dominates: if instead `tp=1, tn=90, fp=0, fn=9` (predicting "negative" almost every time on a dataset that's `90%` negative), accuracy would be `91/100=0.91` — misleadingly high — while recall would be a dismal `1/10=0.10`, revealing that the classifier is essentially failing to find positives at all. Accuracy alone can't distinguish "a genuinely strong classifier" from "a classifier that has learned to exploit a skewed class balance," which is exactly why precision and recall are reported separately rather than folded into one number.

**6.** The three reasons: (1) this C++ port uses `std::mt19937` for its random draws, a different generator from Mojo's own, so even an identical seed produces a different sequence of initial weights and a different training trajectory from the very first epoch; (2) the Mojo source's dataset generator was never included in the material available to port — only a qualitative description ("spiral, XOR pattern, circular boundary, 5% noise") survived — so this edition necessarily wrote its own concrete generator rather than reproducing an unseen one; (3) floating-point summation order differs between the two independently-written implementations, which can shift results at the level of the last few significant digits even when every formula is identical. Printing the Mojo edition's numbers here anyway would have violated the rule this entire book has followed since its first chapter: that a number labeled "genuinely compiled and run" was actually produced by compiling and running the exact code shown, in this environment, during this chapter's own verification. Since this port's code, RNG, and dataset are all genuinely different from Mojo's, its genuine output is also different — and reporting anything else would be fabrication dressed up as a captured result.
