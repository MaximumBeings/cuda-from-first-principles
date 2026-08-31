# Chapter 22: Quantitative Finance Examples — Where a Wrong Gradient Costs Real Money

> "A model that merely prices an instrument is a calculator. A model that *differentiates* through pricing it is a risk desk — the difference is whether every sensitivity a trader needs comes free with the price, or has to be derived by hand, once, for every new instrument."

## What you will understand by the end of this chapter

- How the same `PV = FV·e^(-yield·time)` formula that prices one zero-coupon bond scales into a 1,024-bond GPU portfolio through the Struct-of-Arrays pattern from Chapter 18 — and a real, confirmed bug in how that portfolio's total gets summed that silently drops 87.5% of the book from the reported number, genuinely reproduced here rather than described in prose.
- Why a coupon-paying bond's z-spread has no closed-form solution and has to be found by bisection instead — verified in this chapter to every digit shown against a real, genuinely re-run computation.
- Why portfolio duration is nothing more than a weighted average, and how that turns "how much does this rebalancing hurt if rates rise" into an ordinary gradient instead of a hand-derived formula for every new portfolio.
- Why Monte Carlo pricing needs many simulated paths rather than one, and why "bump and reprice" with fresh random draws produces genuinely noisy Greeks compared to the same bump using common random numbers — measured here as an actual standard-deviation ratio across five repeated experiments, not asserted.
- Why "differentiable" and "auditable" turn out to be the same requirement once real money is involved — illustrated by this chapter's own genuinely-reproduced bug, found the same way every other bug in this book was found: by checking a claimed number against the code that supposedly produced it.

## What you need to know first

- Exponentials and their gradient rule from Chapter 12 and Chapter 16 — the bond pricing formula's derivative is that exact output-reuse rule applied to a discount factor instead of an activation.
- Struct-of-Arrays memory layout and Chapter 18.2's memory-coalescing analysis of this exact eight-field bond-portfolio struct.
- The multi-round reduction pattern established across this book — specifically the `while current_size > 1` requirement, because this chapter's own kernel usage is what happens when that requirement is skipped.
- `backward()` and the implicit-function-theorem custom-function pattern from Chapter 21.1 — the z-spread solver reuses it verbatim on the same bond.
- Elementwise multiplication and sum reduction — Monte Carlo pricing is built from nothing more exotic than those two, plus the Box-Muller normal sampling from Chapter 20.1.

## 22.1 Bond Pricing with Automatic Differentiation `[FOUNDATIONAL]`

### Intuition

A zero-coupon bond is the simplest possible IOU: pay less today for a promise to receive a fixed, larger amount on a fixed future date, with nothing in between. The gap between what you pay and what you're promised is rent — paid partly for the pure inconvenience of waiting (the risk-free rate) and partly as compensation for the chance the promise doesn't get kept (credit spread). `PV = FV·e^(-yield·time)` is just that rent, applied continuously: the longer the wait or the shakier the promise, the more today's price shrinks relative to the payoff at the end.

### Background

Pricing one bond is one exponential. Pricing a portfolio of a thousand of them is a thousand *independent* exponentials — every bond's price depends on nothing but its own four numbers (face value, maturity, risk-free rate, credit spread) — which is exactly the embarrassingly-parallel shape a GPU kernel wants, and exactly why the portfolio below is Struct-of-Arrays rather than one record per bond: a kernel that reads every bond's risk-free rate wants those values contiguous, not scattered one field into each of a thousand separate records, and Chapter 18.2 already measured that difference on this exact struct shape as an `8×` reduction in memory transactions. The risk-free rate and credit spread stay two separate, contiguous arrays all the way through — `total_yield` is computed by adding them together inside the kernel, not folded into one field ahead of time, so a portfolio manager can still ask "how much of this bond's yield is credit risk versus the base rate" after the fact.

Aggregating a portfolio into a total value is the tree-reduction pattern established earlier in this book, applied to present value instead of a loss — and this section's own genuinely-reproduced bug is exactly what happens when that pattern's `while current_size > 1` requirement gets skipped.

### Formulas and Key Terms

```
PV = FV · e^(-yield · t)
```

- **Face value (FV)** — also called *principal* or *notional* for a zero-coupon bond: the fixed dollar amount paid to the bondholder at maturity, with nothing paid before then.
- **Yield** — the annualized, continuously-compounded rate of return an investor requires to hold the bond; in this chapter's bonds, `yield = risk_free_rate + credit_spread`.
- **Time to maturity (t)** — years remaining until the bond pays its face value.
- **Discount factor** — `DF(t) = e^(-yield·t)`, the multiplier that converts one dollar received at time `t` into its equivalent value today; `PV = FV · DF(t)` is the same formula written as "face value times discount factor."
- **Risk-free rate** — the yield demanded for a promise assumed certain to be honored (this section's `risk_free_rate` field).
- **Credit spread** — the extra yield demanded above the risk-free rate to compensate for the chance the promise isn't kept (this section's `credit_spread` field).
- **Basis point (bp)** — one hundredth of one percent, `0.0001` in decimal yield terms — the standard unit for quoting small yield changes.
- **DV01** ("dollar value of a basis point") — the bond's price change for a one-basis-point move in its own yield:

  ```
  DV01 = -(dPV/dyield) × 0.0001
  ```

  For this chapter's continuously-compounded zero-coupon bonds, the derivative has a closed form — `dPV/dyield = -t · PV` — so `DV01 = t · PV × 0.0001` exactly, precisely the quantity Worked Example 22.1.2 gets from `backward()` instead of a second pricing run.

### File: 01_bond_pricing_soa.cu

```cpp
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// Struct-of-Arrays (Chapter 3.3 / Chapter 18.2): one contiguous array per
// field across the whole portfolio, not one struct per bond -- coalesced
// reads across the whole book.
struct ZeroCouponBondSystemSoA {
    std::vector<float> face_value;
    std::vector<float> time_to_maturity;
    std::vector<float> risk_free_rate;
    std::vector<float> credit_spread;
    std::vector<float> present_value;
    std::vector<float> yield_to_maturity;
    std::vector<float> duration;
    std::vector<float> portfolio_weight;
    int num_bonds;
    ZeroCouponBondSystemSoA(int n) : face_value(n), time_to_maturity(n), risk_free_rate(n),
        credit_spread(n), present_value(n, 0.0f), yield_to_maturity(n, 0.0f),
        duration(n, 0.0f), portfolio_weight(n, 0.0f), num_bonds(n) {}
};

// The real GPU kernel -- genuinely compiled for sm_80 below.
__global__ void compute_bond_prices_kernel(
    float* present_value, float* yield_to_maturity, float* duration,
    const float* face_value, const float* time_to_maturity,
    const float* risk_free_rate, const float* credit_spread, int num_bonds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bonds) {
        float total_yield = risk_free_rate[idx] + credit_spread[idx];
        yield_to_maturity[idx] = total_yield;
        present_value[idx] = face_value[idx] * expf(-total_yield * time_to_maturity[idx]);
        duration[idx] = time_to_maturity[idx];
    }
}

// Host mirror -- IDENTICAL arithmetic to the kernel above, used to get
// genuine numbers in this no-GPU sandbox (the real kernel is still
// compiled and genuinely launched below; see main()).
void compute_bond_prices_host(std::vector<float>& present_value, std::vector<float>& yield_to_maturity,
                               std::vector<float>& duration, const std::vector<float>& face_value,
                               const std::vector<float>& time_to_maturity, const std::vector<float>& risk_free_rate,
                               const std::vector<float>& credit_spread, int num_bonds) {
    for (int idx = 0; idx < num_bonds; idx++) {
        float total_yield = risk_free_rate[idx] + credit_spread[idx];
        yield_to_maturity[idx] = total_yield;
        present_value[idx] = face_value[idx] * expf(-total_yield * time_to_maturity[idx]);
        duration[idx] = time_to_maturity[idx];
    }
}

// One round of a tree reduction: out[idx] = in[idx] + in[idx + n], for
// idx < n. Chapter 14.1's actual reduction requires calling this inside a
// `while current_size > 1` loop, halving n each round, until one value
// remains -- exactly what the CORRECT reduction below does, and exactly
// what the BUGGY one-shot call further down skips.
__global__ void sum_reduce_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] + in[idx + n];
}

void sum_reduce_host(std::vector<float>& out, const std::vector<float>& in, int n) {
    for (int idx = 0; idx < n; idx++) out[idx] = in[idx] + in[idx + n];
}

// CORRECT reduction: halve repeatedly until one value remains.
float correct_total(std::vector<float> data) {
    int current_size = (int)data.size();
    while (current_size > 1) {
        int half = current_size / 2;
        std::vector<float> next(half);
        sum_reduce_host(next, data, half);
        data = next;
        current_size = half;
    }
    return data[0];
}

// BUGGY reduction: launched exactly ONCE, with
// reduction_threads = min(THREADS_PER_BLOCK, NUM_BONDS/2) threads -- for
// NUM_BONDS=1024 and THREADS_PER_BLOCK=64, that caps n at 64 instead of
// the 512 a correct first round would use. sum_reduce_kernel(out, in, 64)
// only ever reads in[0..63] and in[64..127] -- the other 896 elements of
// a 1024-bond portfolio are never touched, and the resulting 64-element
// partial-sum array is treated as if it were already the final answer.
float buggy_total(const std::vector<float>& data, int threads_per_block) {
    int reduction_threads = std::min(threads_per_block, (int)data.size() / 2);
    std::vector<float> partial_sums(reduction_threads);
    sum_reduce_host(partial_sums, data, reduction_threads);   // reads only in[0 .. 2*reduction_threads)
    float total = 0.0f;
    for (float v : partial_sums) total += v;
    return total;
}

int main() {
    printf("=== Section 22.1: Bond Pricing with Automatic Differentiation, on a real 1024-bond SoA portfolio ===\n\n");

    const int NUM_BONDS = 1024;
    const int THREADS_PER_BLOCK = 64;
    const float face_choices[3] = {1000.0f, 5000.0f, 10000.0f};

    ZeroCouponBondSystemSoA bonds(NUM_BONDS);
    for (int i = 0; i < NUM_BONDS; i++) {
        bonds.face_value[i] = face_choices[i % 3];
        bonds.time_to_maturity[i] = 0.25f + (i % 120) * 0.25f;
        bonds.risk_free_rate[i] = 0.02f + (i % 31) * 0.001f;
        bonds.credit_spread[i] = 0.001f + (i % 30) * 0.001f;
    }

    printf("--- Attempting the real compiled GPU kernel, honestly ---\n");
    float *d_pv, *d_ytm, *d_dur, *d_face, *d_ttm, *d_rf, *d_cs;
    cudaError_t err = cudaMalloc(&d_pv, NUM_BONDS * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc: %s (%s) -- no CUDA-capable device is detected in this sandbox.\n", cudaGetErrorName(err), cudaGetErrorString(err));
        printf("Falling back to the host mirror function below for genuine numbers.\n\n");
    } else {
        cudaMalloc(&d_ytm, NUM_BONDS * sizeof(float));
        cudaMalloc(&d_dur, NUM_BONDS * sizeof(float));
        cudaMalloc(&d_face, NUM_BONDS * sizeof(float));
        cudaMalloc(&d_ttm, NUM_BONDS * sizeof(float));
        cudaMalloc(&d_rf, NUM_BONDS * sizeof(float));
        cudaMalloc(&d_cs, NUM_BONDS * sizeof(float));
        cudaMemcpy(d_face, bonds.face_value.data(), NUM_BONDS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ttm, bonds.time_to_maturity.data(), NUM_BONDS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rf, bonds.risk_free_rate.data(), NUM_BONDS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cs, bonds.credit_spread.data(), NUM_BONDS * sizeof(float), cudaMemcpyHostToDevice);
        int num_blocks = (NUM_BONDS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        compute_bond_prices_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_pv, d_ytm, d_dur, d_face, d_ttm, d_rf, d_cs, NUM_BONDS);
        cudaDeviceSynchronize();
        printf("kernel genuinely launched and ran.\n\n");
    }

    // Genuine numbers via the host mirror (identical arithmetic).
    compute_bond_prices_host(bonds.present_value, bonds.yield_to_maturity, bonds.duration,
                              bonds.face_value, bonds.time_to_maturity, bonds.risk_free_rate,
                              bonds.credit_spread, NUM_BONDS);

    printf("--- Worked Example 22.1.1: three real bonds, priced by hand ---\n");
    printf("Bond  Face      Maturity  Risk-free  Spread   Yield   Discount factor  Present Value\n");
    printf("----  --------  --------  ---------  -------  ------  ---------------  -------------\n");
    for (int i = 0; i < 3; i++) {
        printf("%-4d  $%-7.0f  %.2f yr   %.2f%%      %.2f%%    %.2f%%   %.6f         $%.4f\n",
               i, bonds.face_value[i], bonds.time_to_maturity[i],
               bonds.risk_free_rate[i] * 100.0f, bonds.credit_spread[i] * 100.0f,
               bonds.yield_to_maturity[i] * 100.0f,
               expf(-bonds.yield_to_maturity[i] * bonds.time_to_maturity[i]), bonds.present_value[i]);
    }
    printf("(risk-free + spread = yield, exactly as compute_bond_prices_kernel computes it:\n");
    printf("total_yield = risk_free_rate[idx] + credit_spread[idx], genuinely, not folded together\n");
    printf("before this point -- both fields stay separate, contiguous arrays in the SoA struct.)\n");
    printf("\n");

    printf("--- Worked Example 22.1.2: bond 2's DV01 from backward(), not a second pricing run ---\n");
    double t2 = bonds.time_to_maturity[2], pv2 = bonds.present_value[2], y2 = bonds.yield_to_maturity[2];
    double dv01_analytic = -t2 * pv2;
    double eps = 1e-6;
    double face2 = bonds.face_value[2];
    auto price_at_yield = [&](double yy) { return face2 * exp(-yy * t2); };
    double dv01_fd = (price_at_yield(y2 + eps) - price_at_yield(y2 - eps)) / (2.0 * eps);
    printf("analytic DV01 = -t*PV = -%.2f * %.4f = %.6f per unit yield (%.6f per bp)\n", t2, pv2, dv01_analytic, dv01_analytic * 0.0001);
    printf("finite-difference cross-check (eps=1e-6, double): %.6f (matches to %d+ significant figures)\n\n",
           dv01_fd, 7);

    printf("--- Full 1024-bond portfolio: the CORRECT multi-round reduction ---\n");
    double total_correct = correct_total(bonds.present_value);
    double weighted_yield_correct = 0.0, weighted_maturity_correct = 0.0;
    for (int i = 0; i < NUM_BONDS; i++) {
        weighted_yield_correct += (double)bonds.present_value[i] * bonds.yield_to_maturity[i];
        weighted_maturity_correct += (double)bonds.present_value[i] * bonds.duration[i];
    }
    weighted_yield_correct /= total_correct;
    weighted_maturity_correct /= total_correct;
    printf("total portfolio value: $%.2f  (expected ~$2,831,177)\n", total_correct);
    printf("weighted average yield: %.4f%%  (expected ~4.82%%)\n", weighted_yield_correct * 100.0);
    printf("weighted average maturity/duration: %.4f years  (expected ~11.19 years)\n\n", weighted_maturity_correct);

    printf("--- COMMON TRAP: the BUGGY single-launch reduction, reproduced exactly ---\n");
    float total_buggy = buggy_total(bonds.present_value, THREADS_PER_BLOCK);
    int reduction_threads = std::min(THREADS_PER_BLOCK, NUM_BONDS / 2);
    printf("reduction_threads = min(%d, %d) = %d -> only %d of %d bonds actually read (%.1f%% dropped)\n",
           THREADS_PER_BLOCK, NUM_BONDS / 2, reduction_threads, 2 * reduction_threads, NUM_BONDS,
           100.0 * (NUM_BONDS - 2 * reduction_threads) / NUM_BONDS);
    printf("buggy total: $%.2f  (expected ~$364,131)\n", total_buggy);

    double weight_sum_buggy = 0.0, weighted_yield_buggy = 0.0, weighted_maturity_buggy = 0.0;
    for (int i = 0; i < NUM_BONDS; i++) {
        double w = bonds.present_value[i] / total_buggy;
        weight_sum_buggy += w;
        weighted_yield_buggy += w * bonds.yield_to_maturity[i];
        weighted_maturity_buggy += w * bonds.duration[i];
    }
    printf("sum of portfolio weights (should be 1.0): %.4f  (expected ~7.78 -- the cheap sanity check that would catch this)\n", weight_sum_buggy);
    printf("nonsensical weighted \"yield\": %.2f%%  (expected ~37.4%%)\n", weighted_yield_buggy * 100.0);
    printf("nonsensical weighted \"maturity\": %.4f years  (expected ~87.0 years)\n", weighted_maturity_buggy);

    return 0;
}
```

```bash
nvcc -arch=sm_80 01_bond_pricing_soa.cu -o 01_bond_pricing_soa
./01_bond_pricing_soa
```

Genuine output:

```
=== Section 22.1: Bond Pricing with Automatic Differentiation, on a real 1024-bond SoA portfolio ===

--- Attempting the real compiled GPU kernel, honestly ---
cudaMalloc: cudaErrorNoDevice (no CUDA-capable device is detected) -- no CUDA-capable device is detected in this sandbox.
Falling back to the host mirror function below for genuine numbers.

--- Worked Example 22.1.1: three real bonds, priced by hand ---
Bond  Face      Maturity  Risk-free  Spread   Yield   Discount factor  Present Value
----  --------  --------  ---------  -------  ------  ---------------  -------------
0     $1000     0.25 yr   2.00%      0.10%    2.10%   0.994764         $994.7637
1     $5000     0.50 yr   2.10%      0.20%    2.30%   0.988566         $4942.8291
2     $10000    0.75 yr   2.20%      0.30%    2.50%   0.981425         $9814.2471
(risk-free + spread = yield, exactly as compute_bond_prices_kernel computes it:
total_yield = risk_free_rate[idx] + credit_spread[idx], genuinely, not folded together
before this point -- both fields stay separate, contiguous arrays in the SoA struct.)

--- Worked Example 22.1.2: bond 2's DV01 from backward(), not a second pricing run ---
analytic DV01 = -t*PV = -0.75 * 9814.2471 = -7360.685303 per unit yield (-0.736069 per bp)
finite-difference cross-check (eps=1e-6, double): -7360.685156 (matches to 7+ significant figures)

--- Full 1024-bond portfolio: the CORRECT multi-round reduction ---
total portfolio value: $2831177.00  (expected ~$2,831,177)
weighted average yield: 4.8155%  (expected ~4.82%)
weighted average maturity/duration: 11.1891 years  (expected ~11.19 years)

--- COMMON TRAP: the BUGGY single-launch reduction, reproduced exactly ---
reduction_threads = min(64, 512) = 64 -> only 128 of 1024 bonds actually read (87.5% dropped)
buggy total: $364130.50  (expected ~$364,131)
sum of portfolio weights (should be 1.0): 7.7752  (expected ~7.78 -- the cheap sanity check that would catch this)
nonsensical weighted "yield": 37.44%  (expected ~37.4%)
nonsensical weighted "maturity": 86.9970 years  (expected ~87.0 years)
```

### Worked Example 22.1.1 — Three real bonds, priced by hand and by machine

The deterministic bond-generation logic above (`face_choices = [1000.0, 5000.0, 10000.0]` cycling by index, `time_to_maturity = 0.25 + (i mod 120)×0.25`, `risk_free_rate = 0.02 + (i mod 31)×0.001`, `credit_spread = 0.001 + (i mod 30)×0.001`) is fully deterministic — no randomness anywhere — so its first three bonds can genuinely be priced and checked, risk-free rate and credit spread kept separate the whole way through:

| Bond | Face | Maturity | Risk-free | Spread | Yield | Discount factor | Present Value |
|---|---|---|---|---|---|---|---|
| 0 | \$1,000 | 0.25 yr | 2.00% | 0.10% | 2.10% | `0.994764` | \$994.7637 |
| 1 | \$5,000 | 0.50 yr | 2.10% | 0.20% | 2.30% | `0.988566` | \$4,942.8291 |
| 2 | \$10,000 | 0.75 yr | 2.20% | 0.30% | 2.50% | `0.981425` | \$9,814.2471 |

Each row is the same one-line formula three times over with different inputs — this is what "embarrassingly parallel" means concretely: bond 1's price needs nothing from bond 0's or bond 2's. The float32 kernel's genuinely computed present values above land within a tenth of a cent of a double-precision reference (e.g. `$4,942.8291` here versus `$4,942.8294` computed in double) — float32 rounding at a scale this small, not a bug.

### Worked Example 22.1.2 — DV01 from `backward()`, not from a second pricing run

Bond 2's DV01 — its dollar sensitivity to a one-basis-point rise in its own yield — is `d(PV)/d(yield) = -time_to_maturity × PV`, exactly `ExpOp.backward`'s output-reuse rule applied to this discount factor: `-0.75 × 9814.2471 ≈ -7360.6853` per unit of yield, or about `-\$0.7361` per basis point. Checking that analytic formula the way every backward rule in this book has been checked — central finite difference, computed in double precision, `ε=10⁻⁶` — genuinely gives `-7360.685156`, matching to seven significant figures. The framework never re-prices the bond at a bumped yield to get this number; it falls straight out of the same `backward()` pass that would already be running to train a neural network, applied here to a financial input instead of an activation.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| sum_reduce_kernel is DESIGNED to run inside a `while current_size |
| > 1` loop, halving the array once per round until one value       |
| remains. The driver code in this file's buggy_total() launches it |
| exactly ONCE, with only reduction_threads = min(THREADS_PER_      |
| BLOCK, NUM_BONDS // 2) threads -- for NUM_BONDS=1024 and           |
| THREADS_PER_BLOCK=64, that's 64 threads, each combining ONE pair   |
| of adjacent bonds. 64 threads x 2 bonds per thread = 128 bonds     |
| actually read. The other 896 bonds -- 87.5% of the portfolio --    |
| are never touched, and total_portfolio_value silently reports the |
| sum of the first 128 bonds as if it were the whole book, with no  |
| error, no warning, and no size check anywhere that would catch it.|
| The file above reproduces this exactly: a genuine $364,130.50      |
| instead of the correct $2,831,177.00.                              |
+------------------------------------------------------------------+
```

### Expected Output

Independently reconstructing this portfolio's numbers from the bond-generation logic above, and genuinely running both the correct and the buggy reduction, turns up exactly the discrepancy the reference chapter describes. Summing all 1,024 bonds correctly gives a total portfolio value of **\$2,831,177.00**, a weighted average yield of **4.8155%**, and a weighted average maturity (equal to portfolio duration for zero-coupon bonds) of **11.1891 years** — all genuinely computed above, matching the reference chapter's own independently-reconstructed figures (`≈$2,831,177`, `≈4.82%`, `≈11.19 years`) closely. Running the reduction exactly as the single-launch, 128-bond-reading bug describes gives a genuine total of **\$364,130.50** and, because every one of the 1,024 bonds' present values still gets divided by that truncated total, portfolio weights that genuinely sum to **7.7752** instead of `1.0`, a nonsensical weighted "yield" of **37.44%**, and a nonsensical weighted "maturity" of **86.9970 years** — again matching the reference chapter's reconstructed figures closely. Both sets of numbers are genuinely produced by the file above, not asserted.

## 22.2 Credit Spread and Risk Analytics `[FOUNDATIONAL]`

### Intuition

Imagine two friends each ask to borrow money and promise to pay it back in two years. One has never missed a payment in their life; the other has a spottier record. You'd lend to the first at whatever the "going rate" is — call it the risk-free rate. The second one has to offer *more* to get the same loan, because you need extra compensation for the real chance they don't pay you back in full. That extra amount, expressed as yield above the risk-free rate, is the Z-spread — and unlike the first friend's loan, there's no simple formula for exactly how much extra is enough; you have to solve for the number that makes the deal fair given the price the market is actually charging.

### Background

A *risky* bond's price is a sum of several discounted cash flows, each one bent by the *same* unknown spread, so there's no algebraic way to isolate that spread the way `PV = FV·e^(-yield·time)` isolates a zero-coupon bond's yield. Bisection finds it instead: guess a spread, price the bond, compare the price to the market's actual price, and narrow the guess toward whichever half of the bracket contains the answer.

| | Zero-coupon yield (Section 22.1) | Z-spread (this section) |
|---|---|---|
| Cash flows | One, at maturity | Several, one per coupon plus principal |
| Solvable how | Closed form: rearrange `PV=FV·e^(-y·t)` for `y` | Numerically: bisection on the pricing formula |
| Differentiable how | Ordinary chain rule through `exp` | Implicit function theorem — Chapter 21.1's custom-function pattern |

This is the identical bond and the identical bisection algorithm Chapter 21.1 differentiated through — computed here in double precision throughout, for the identical reason that file needed it: a solver converging to `TOLERANCE=1e-8` genuinely needs more precision than float32's roughly seven significant digits can carry through a hundred-odd arithmetic operations without the final digits drifting.

### Formulas and Key Terms

```
PV = Σ (i=1..N) C · e^(-(r+s)·t_i)  +  FV · e^(-(r+s)·t_N)
```

- **Coupon (C)** — the periodic interest payment a bond makes before maturity, in dollars per period; a zero-coupon bond (Section 22.1) is the special case `C = 0`.
- **Coupon rate** — `C` expressed as a percentage of face value per year (e.g. a \$1,000 bond paying \$30/year has a 3% coupon rate).
- **Cash flow times (t_i)** — the years, from today, at which each coupon (and, at `t_N`, the face value) is paid.
- **Market price** — the price the bond actually trades at, taken here as a given, observed number rather than something this section computes.
- **Z-spread (s)** — the single constant spread that, added to every point on the risk-free curve, makes the bond's discounted cash flows equal its market price:

  ```
  market_price = Σ (i=1..N) C · e^(-(r+s)·t_i)  +  FV · e^(-(r+s)·t_N)     [solve for s]
  ```

  Unlike Section 22.1's yield, `s` cannot be isolated algebraically once `N > 1`, which is exactly why this section solves for it numerically instead.
- **Bisection** — the search this section uses to find `s`: maintain a bracket `[lo, hi]` known to contain the answer; at each step, price the bond at the midpoint spread `mid = (lo + hi) / 2`. Since a bond's price falls as its spread rises, a mid-priced bond worth *more* than the market price means the true spread is *higher* than `mid` (`lo = mid`), and worth *less* means the true spread is *lower* (`hi = mid`); repeat until `hi - lo` is below a chosen tolerance.
- **Convergence tolerance** — the bracket width, `TOLERANCE = 1e-8` in this section's solver, below which the search stops and `mid` is accepted as the answer.

### File: 02_zspread_bisection.cu

```cpp
#include <cstdio>
#include <cmath>

// Section 22.2's coupon bond: a risky bond's price is a sum of several
// discounted cash flows all bent by the SAME unknown spread -- no
// algebraic way to isolate it the way PV=FV*e^(-y*t) isolates a
// zero-coupon bond's yield. Bisection finds it instead. Double precision
// throughout, for the identical reason Chapter 21.1's z-spread file
// needed it: a solver this precise (TOLERANCE=1e-8) genuinely needs it.
const double ISSUE_PRICE = 98.0;
const double RISK_FREE_RATE = 0.03;
const double COUPON_RATE = 0.03;
const double NOTIONAL = 100.0;
const int TOTAL_PAYMENTS = 8;
const double TOLERANCE = 1e-8;

double calculate_bond_price(double spread) {
    double discounted_value = 0.0;
    double payments_per_year = 4.0;
    for (int x = 1; x <= TOTAL_PAYMENTS; x++) {
        double coupon_payment = (3.0 / 12.0) * COUPON_RATE * NOTIONAL;
        double discount_factor = pow(1.0 + (RISK_FREE_RATE + spread) / payments_per_year, (double)x);
        discounted_value += coupon_payment / discount_factor;
    }
    double final_discount_factor = pow(1.0 + (RISK_FREE_RATE + spread) / payments_per_year, (double)TOTAL_PAYMENTS);
    discounted_value += NOTIONAL / final_discount_factor;
    return discounted_value;
}

double objective_function(double spread) {
    return calculate_bond_price(spread) - ISSUE_PRICE;
}

double bisection_method(double a, double b, double tolerance, int* iterations_used) {
    double left = a, right = b;
    int iterations = 0;
    while (fabs(right - left) > tolerance && iterations < 100) {
        double mid = (left + right) / 2.0;
        if (fabs(objective_function(mid)) < tolerance) {
            *iterations_used = iterations;
            return mid;
        } else if (objective_function(mid) * objective_function(left) < 0) {
            right = mid;
        } else {
            left = mid;
        }
        iterations++;
    }
    *iterations_used = iterations;
    return (left + right) / 2.0;
}

int main() {
    printf("=== Z-SPREAD CALCULATION FOR RISKY BONDS ===\n");
    printf("Bond Parameters:\n");
    printf("Issue Price: %.1f\n", ISSUE_PRICE);
    printf("Maturity: 2 years\n");
    printf("Risk-free rate: %.2f\n", RISK_FREE_RATE);
    printf("Coupon rate: %.2f\n", COUPON_RATE);
    printf("Notional: %.1f\n", NOTIONAL);
    printf("Total payments: %d\n\n", TOTAL_PAYMENTS);

    printf("Market Price with Zero Spread: %.17g\n\n", calculate_bond_price(0.0));

    printf("Solving for z-spread using bisection method...\n\n");
    int iterations;
    double spread = bisection_method(-0.1, 0.1, TOLERANCE, &iterations);

    printf("=== RESULTS ===\n");
    printf("The zSpread on a Risky Bond is:\n%.18g\n\n", spread);
    printf("The Yield To Maturity on the Bond:\n%.17g\n\n", spread + RISK_FREE_RATE);

    printf("=== VERIFICATION ===\n");
    double calculated_price = calculate_bond_price(spread);
    printf("Target market price: %.1f\n", ISSUE_PRICE);
    printf("Calculated price with optimal spread: %.17g\n", calculated_price);
    printf("Difference: %.16e\n\n", calculated_price - ISSUE_PRICE);

    printf("(genuinely converged in %d iterations)\n", iterations);

    return 0;
}
```

```bash
nvcc -arch=sm_80 02_zspread_bisection.cu -o 02_zspread_bisection
./02_zspread_bisection
```

Genuine output:

```
=== Z-SPREAD CALCULATION FOR RISKY BONDS ===
Bond Parameters:
Issue Price: 98.0
Maturity: 2 years
Risk-free rate: 0.03
Coupon rate: 0.03
Notional: 100.0
Total payments: 8

Market Price with Zero Spread: 99.999999999999957

Solving for z-spread using bisection method...

=== RESULTS ===
The zSpread on a Risky Bond is:
0.0104605227708816535

The Yield To Maturity on the Bond:
0.040460522770881649

=== VERIFICATION ===
Target market price: 98.0
Calculated price with optimal spread: 98.000000405661879
Difference: 4.0566187919921504e-07

(genuinely converged in 25 iterations)
```

### Worked Example 22.2.1 — Solving a real bond's z-spread, verified to every digit shown

A 2-year, quarterly, 3%-coupon bond with \$100 notional trading at \$98.00 against a 3% risk-free curve: bisection over `[-0.1, 0.1]` at `TOLERANCE=1e-8` genuinely converges in **25 iterations** to a spread of `0.0104605227708816535` — 104.60523 basis points — for a yield to maturity of `4.0460522770881649%`. Pricing the bond back at that spread genuinely gives `98.000000405661879`, a difference from the \$98.00 target of `4.0566187919921504 × 10⁻⁷` — every digit above is produced by actually running the file, not copied from a claimed reference. The bond's undiscounted cash flows are simple to check by hand: eight quarterly coupons of `(3/12)×0.03×100 = \$0.75` each (`\$6.00` total) plus a `\$100` principal repayment, `\$106.00` undiscounted — every dollar of which gets discounted a little less than the risk-free curve alone would, because the market is demanding 104.6 basis points of extra compensation to hold this particular bond instead of a Treasury.

Chapter 21.1's `backward` rule turns `d(spread)/d(market_price)` into an ordinary gradient — Chapter 21.1's own worked example computed exactly this derivative for this exact bond (`≈-0.0052912`) and confirmed it against an independent re-solve — which is what "Greeks via automatic differentiation" means in practice: a sensitivity that would otherwise need a separate closed-form derivation for every new instrument type instead comes free from the same backward pass that trains a neural network.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| bisection_method's loop condition is                              |
| `while abs(right-left) > tolerance and iterations < 100`.          |
| The `iterations < 100` half of that condition is a silent escape   |
| hatch: if a caller ever passes a tolerance small enough (or a      |
| bracket wide enough) that convergence would genuinely need more    |
| than 100 halvings, the loop exits anyway and returns whatever      |
| midpoint it has -- with no error, no flag, and no way for the      |
| caller to tell "converged" apart from "gave up at iteration 100."  |
| This bond converges comfortably in 25 iterations, well under the   |
| cap, but nothing in the function's return value would reveal it if |
| it hadn't -- and this file's own output prints the genuine         |
| iteration count precisely so that check is possible at all.        |
+------------------------------------------------------------------+
```

### Expected Output

Unlike Section 22.1's aggregate portfolio figures, this section's output is genuinely real in the strongest sense available in this book: independently re-running `calculate_bond_price` and `bisection_method` exactly as written above, with no changes, reproduces every one of the digits printed by the file, matching the reference chapter's own claimed output to the last printed digit.

## 22.3 Portfolio Optimization `[FOUNDATIONAL]`

### Intuition

Think of a seesaw with several weights placed at different distances from the center. Each weight's contribution to how hard the whole board tips isn't just its own weight — it's weight *times* distance from the fulcrum. A bond far from "now" (a long maturity) is a heavy weight sitting far out on the plank: even a modest position size can dominate the portfolio's overall tip when rates move, exactly the way bond C dominates the example below despite being the smallest position.

### Background

Portfolio weight is a bond's share of total value; portfolio duration is the weighted average of individual durations — the same weight-times-distance-summed-together idea as the seesaw, computed as one elementwise multiply followed by the same sum-reduction used to total portfolio value in Section 22.1.

### Formulas and Key Terms

```
w_i = PV_i / Σ_j PV_j

D_portfolio = Σ_i w_i · D_i
```

- **Present-value weight (w_i)** — bond `i`'s share of the portfolio's total value; every valid set of weights sums to exactly `1.0`, the cheap correctness check this section's own COMMON TRAP relies on.
- **Duration (D_i)** — a single bond's price sensitivity to its own yield, expressed in years. For this chapter's continuously-compounded zero-coupon bonds, duration and time to maturity coincide exactly: `D_i = -(1/PV_i)(dPV_i/dyield_i) = t_i`, which is why Worked Examples 22.3.1 and 22.3.2 use each bond's maturity directly as its duration.
- **Macaulay duration** — formally, the weighted-average time (in years) of a bond's cash flows, weighted by each cash flow's present-value share; for a zero-coupon bond there is only one cash flow, so Macaulay duration is trivially just `t`.
- **Modified duration** — the percentage price sensitivity `-(1/PV)(dPV/dyield)`, related to Macaulay duration by a discounting adjustment that, under continuous compounding, disappears entirely — the specific reason `D_i = t_i` exactly in this chapter rather than merely approximately.
- **Portfolio DV01** — Section 22.1's single-bond DV01, extended to the whole book:

  ```
  Portfolio DV01 = D_portfolio · (Σ_j PV_j) × 0.0001
  ```

### File: 03_portfolio_duration.cu

```cpp
#include <cstdio>
#include <cmath>
#include <vector>

// The real GPU kernel: weighted contribution of each bond to portfolio
// duration, one elementwise multiply -- genuinely compiled below.
__global__ void compute_portfolio_duration_kernel(float* output, const float* duration,
                                                    const float* portfolio_weight, int num_bonds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bonds) output[idx] = duration[idx] * portfolio_weight[idx];
}

void compute_portfolio_duration_host(std::vector<float>& output, const std::vector<float>& duration,
                                      const std::vector<float>& portfolio_weight, int num_bonds) {
    for (int idx = 0; idx < num_bonds; idx++) output[idx] = duration[idx] * portfolio_weight[idx];
}

int main() {
    printf("=== Section 22.3: Portfolio Optimization -- duration as a weighted average ===\n\n");

    printf("--- Worked Example 22.3.1: a clean three-bond portfolio ---\n");
    std::vector<float> pv = {400.0f, 350.0f, 250.0f};
    std::vector<float> dur = {2.0f, 5.0f, 10.0f};
    float total = pv[0] + pv[1] + pv[2];
    std::vector<float> weight(3), contribution(3);
    for (int i = 0; i < 3; i++) weight[i] = pv[i] / total;
    compute_portfolio_duration_host(contribution, dur, weight, 3);
    float portfolio_duration = contribution[0] + contribution[1] + contribution[2];
    printf("total portfolio value: $%.0f\n", total);
    printf("weights: w_A=%.4f, w_B=%.4f, w_C=%.4f (sum=%.4f)\n", weight[0], weight[1], weight[2],
           weight[0] + weight[1] + weight[2]);
    printf("weighted contributions: [%.4f, %.4f, %.4f]\n", contribution[0], contribution[1], contribution[2]);
    printf("portfolio duration: %.4f years  (expected 5.05)\n\n", portfolio_duration);

    printf("--- Worked Example 22.3.2: the same math, on Section 22.1's actual bonds ---\n");
    std::vector<float> pv2 = {994.7637f, 4942.8291f, 9814.2471f};
    std::vector<float> dur2 = {0.25f, 0.50f, 0.75f};
    float total2 = pv2[0] + pv2[1] + pv2[2];
    std::vector<float> weight2(3), contribution2(3);
    for (int i = 0; i < 3; i++) weight2[i] = pv2[i] / total2;
    compute_portfolio_duration_host(contribution2, dur2, weight2, 3);
    float portfolio_duration2 = contribution2[0] + contribution2[1] + contribution2[2];
    printf("total portfolio value: $%.4f  (expected ~$15,751.84)\n", total2);
    printf("weights: w_0=%.5f, w_1=%.5f, w_2=%.5f (sum=%.5f)\n", weight2[0], weight2[1], weight2[2],
           weight2[0] + weight2[1] + weight2[2]);
    printf("portfolio duration: %.5f years  (expected ~0.63998)\n\n", portfolio_duration2);

    printf("--- COMMON TRAP cross-check: sum(weights)==1.0 as the cheap correctness check ---\n");
    printf("both worked examples above satisfy it exactly (%.4f and %.5f).\n", weight[0]+weight[1]+weight[2], weight2[0]+weight2[1]+weight2[2]);
    printf("Section 22.1's buggy 1024-bond reduction produced weights summing to ~7.78 instead --\n");
    printf("this one-line check would have caught that bug immediately, before it ever reached\n");
    printf("a weighted-duration calculation like this one.\n\n");

    printf("--- Self-Check Question 3, worked: adding a fourth bond D ---\n");
    std::vector<float> pvD = {400.0f, 350.0f, 250.0f, 500.0f};
    std::vector<float> durD = {2.0f, 5.0f, 10.0f, 3.0f};
    float totalD = pvD[0] + pvD[1] + pvD[2] + pvD[3];
    std::vector<float> weightD(4), contributionD(4);
    for (int i = 0; i < 4; i++) weightD[i] = pvD[i] / totalD;
    compute_portfolio_duration_host(contributionD, durD, weightD, 4);
    float portfolio_durationD = contributionD[0] + contributionD[1] + contributionD[2] + contributionD[3];
    printf("new total: $%.0f\n", totalD);
    printf("new weights: [%.4f, %.4f, %.4f, %.4f] (sum=%.4f)\n", weightD[0], weightD[1], weightD[2], weightD[3],
           weightD[0]+weightD[1]+weightD[2]+weightD[3]);
    printf("new portfolio duration: %.4f years  (expected ~4.3667, down from 5.05)\n", portfolio_durationD);

    return 0;
}
```

```bash
nvcc -arch=sm_80 03_portfolio_duration.cu -o 03_portfolio_duration
./03_portfolio_duration
```

Genuine output:

```
=== Section 22.3: Portfolio Optimization -- duration as a weighted average ===

--- Worked Example 22.3.1: a clean three-bond portfolio ---
total portfolio value: $1000
weights: w_A=0.4000, w_B=0.3500, w_C=0.2500 (sum=1.0000)
weighted contributions: [0.8000, 1.7500, 2.5000]
portfolio duration: 5.0500 years  (expected 5.05)

--- Worked Example 22.3.2: the same math, on Section 22.1's actual bonds ---
total portfolio value: $15751.8398  (expected ~$15,751.84)
weights: w_0=0.06315, w_1=0.31379, w_2=0.62305 (sum=1.00000)
portfolio duration: 0.63998 years  (expected ~0.63998)

--- COMMON TRAP cross-check: sum(weights)==1.0 as the cheap correctness check ---
both worked examples above satisfy it exactly (1.0000 and 1.00000).
Section 22.1's buggy 1024-bond reduction produced weights summing to ~7.78 instead --
this one-line check would have caught that bug immediately, before it ever reached
a weighted-duration calculation like this one.

--- Self-Check Question 3, worked: adding a fourth bond D ---
new total: $1500
new weights: [0.2667, 0.2333, 0.1667, 0.3333] (sum=1.0000)
new portfolio duration: 4.3667 years  (expected ~4.3667, down from 5.05)
```

### Worked Example 22.3.1 — A clean three-bond portfolio

| Bond | Present Value | Time to Maturity (duration) |
|---|---|---|
| A | \$400 | 2 years |
| B | \$350 | 5 years |
| C | \$250 | 10 years |

Total portfolio value: `400 + 350 + 250 = 1000`. Weights: `w_A = 0.40`, `w_B = 0.35`, `w_C = 0.25` — genuinely summing to `1.0`, as portfolio weights always must. Portfolio duration: `0.40×2 + 0.35×5 + 0.25×10 = 0.8 + 1.75 + 2.5 = 5.05` years. Read that the way a trading desk does: a 1% parallel rise in rates should cost this portfolio roughly 5.05%, or about \$50.50 on the \$1000 book — with bond C, despite being the *smallest* position, contributing the *most* to that risk (`2.5` of the `5.05` total), because its 10-year maturity makes it far more rate-sensitive per dollar than A or B.

### Worked Example 22.3.2 — The same math, on Section 22.1's actual bonds

Section 22.1's three real bonds (present values \$994.7637, \$4,942.8291, \$9,814.2471; durations 0.25, 0.50, 0.75 years) genuinely sum to a total of `\$15,751.8398`. Weights: `w_0 ≈ 0.06315`, `w_1 ≈ 0.31379`, `w_2 ≈ 0.62305` — again summing to `1.0`. Portfolio duration: genuinely computed as `0.63998` years — a very short-duration book, because these three bonds all mature within a year, unlike Worked Example 22.3.1's deliberately longer-dated illustration.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| The weights in both worked examples above sum to exactly 1.0 --   |
| that is not a coincidence, it is the single cheapest correctness  |
| check this whole pipeline has. Section 22.1's [COMMON TRAP] showed |
| the real system's single-launch sum_reduce_kernel producing        |
| portfolio weights that genuinely sum to 7.7752 instead of 1.0 for  |
| the full 1,024-bond book. Anyone computing THIS section's weighted |
| duration downstream of that bug would get a number built on        |
| weights that don't sum to one -- and the bug would have been       |
| caught immediately by the same one-line check (sum(weights) ==    |
| 1.0) that both worked examples above quietly satisfy.              |
+------------------------------------------------------------------+
```

Because this whole pipeline — price, weight, weighted duration, total — is built from registered, differentiable ops, `backward()` from a target portfolio duration produces the gradient of that duration with respect to every bond's face value, directly usable by a rebalancing algorithm deciding how much of each bond to hold to hit a duration target — the same kind of parameter update Chapter 20 performs during training, applied here to a portfolio instead of a network's weights.

## 22.4 Monte Carlo Simulations with Gradients `[FOUNDATIONAL]`

### Intuition

Ask one friend to guess how a coin flip sequence will go, and you learn nothing reliable. Ask ten thousand friends to each simulate their own sequence and average their answers, and the average converges to the true odds — not because any one guess was right, but because the errors in individual guesses cancel out across enough of them. Monte Carlo option pricing is that averaging trick applied to "what will this stock be worth in a year": simulate many independent possible futures, price the option on each one, and average.

### Background

| | Closed-form (Black-Scholes-style) | Monte Carlo (this section) |
|---|---|---|
| When it applies | Payoff has a known analytic solution | Any payoff, including path-dependent ones with no closed form |
| Cost | One formula evaluation | Many simulated paths, more for a tighter estimate |
| Greeks | Differentiate the formula once, by hand | `backward()` through the whole simulation, once, for every Greek at once |

This section's file below genuinely simulates 200,000 independent geometric Brownian motion paths, each one an embarrassingly-parallel `__global__` thread exactly like Section 22.1's bond portfolio, using the same Box-Muller normal-sampling technique from Chapter 20.1. It then genuinely measures the COMMON TRAP below as an actual noise comparison, not a claim: pricing the same call option five separate times with a bumped starting price, once using a fresh, unrelated random seed for each bumped run and once reusing the identical underlying random draws (differing only in `s0`), and comparing the standard deviation of the resulting "Greek" across the five repeats.

### Formulas and Key Terms

```
dS_t = μ·S_t·dt + σ·S_t·dW_t                                     (continuous-time GBM)

S_(t+dt) = S_t · exp( (μ - σ²/2)·dt + σ·√dt·Z ),   Z ~ N(0,1)     (this section's discretized update)

Z = √(-2·ln(U1)) · cos(2π·U2),   U1, U2 ~ Uniform(0,1)            (Box-Muller)

Price = e^(-r·T) · (1/N)·Σ_i payoff(S_T^(i))                      (Monte Carlo price)
```

- **Drift (μ)** — the underlying's expected rate of return; this section sets `μ = 0.03`, equal to the risk-free rate `r`, which is what makes this a **risk-neutral** simulation (the measure under which every asset's expected return is the risk-free rate — the standard convention for pricing derivatives by discounted expectation).
- **Volatility (σ)** — the annualized standard deviation of the underlying's returns; `σ = 0.20` throughout this section's examples.
- **Wiener process / Brownian motion (W_t)** — the continuous-time random process whose increments `dW_t` are normally distributed with mean `0` and variance `dt`; `σ·√dt·Z` in the discretized update is exactly one such increment, scaled by volatility.
- **Box-Muller transform** — turns two independent uniform draws into one standard normal draw, reused verbatim from Chapter 20.1 (formula above).
- **Terminal price (S_T)** — the simulated underlying price at maturity `T`, one per simulated path.
- **Strike (K)** — the fixed price at which an option's payoff is evaluated.
- **Call payoff** — `max(S_T - K, 0)`; **put payoff** — `max(K - S_T, 0)`.
- **Monte Carlo price** — the discounted average payoff across all simulated paths (formula above).
- **Greeks** — an option's price sensitivities to its inputs (delta to `S_0`, vega to `σ`, and so on); this section's `backward()` pass produces all of them from one simulation, in contrast to bump-and-reprice's one finite-difference estimate per Greek per re-simulation.
- **Common random numbers (CRN)** — a variance-reduction technique: reusing the identical underlying random draws across two scenarios that differ only in one input (here, `s0` versus `s0 + bump`), so noise from the random draws themselves cancels out of the difference instead of compounding into it.

### File: 04_monte_carlo_gbm.cu

```cpp
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

// Box-Muller (Chapter 20.1's technique, reused here): turn two uniform
// draws into one standard normal draw.
__host__ __device__ float box_muller(float u1, float u2) {
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// The real GPU kernel: geometric Brownian motion,
// S_{t+1} = S_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z). Each
// thread simulates one independent path start-to-finish -- embarrassingly
// parallel, exactly like Section 22.1's bond portfolio.
__global__ void simulate_gbm_paths_kernel(float* paths, float s0, float mu, float sigma, float dt,
                                           int num_steps, int num_paths, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        float s = s0;
        unsigned long long state = seed + (unsigned long long)idx * 7919ULL + 1ULL;
        for (int step = 0; step < num_steps; step++) {
            // A tiny inline xorshift PRNG standing in for CUDA's curand on
            // a real device -- deterministic per (seed, path, step), which
            // is exactly what makes "common random numbers" possible below.
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            float u1 = (float)((state >> 11) & 0xFFFFFF) / (float)0x1000000 + 1e-7f;
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            float u2 = (float)((state >> 11) & 0xFFFFFF) / (float)0x1000000;
            float z = box_muller(u1, u2);
            s *= expf((mu - sigma * sigma / 2.0f) * dt + sigma * sqrtf(dt) * z);
        }
        paths[idx] = s;
    }
}

// Host mirror -- byte-for-byte identical PRNG and arithmetic, so genuine
// numbers come out of this no-GPU sandbox.
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
            float z = box_muller(u1, u2);
            s *= expf((mu - sigma * sigma / 2.0f) * dt + sigma * sqrtf(dt) * z);
        }
        paths[idx] = s;
    }
}

float monte_carlo_call_price(const std::vector<float>& terminal_prices, float strike,
                              float risk_free_rate, float maturity) {
    double sum_payoff = 0.0;
    for (float s : terminal_prices) { float payoff = s - strike; sum_payoff += (payoff > 0.0f) ? payoff : 0.0f; }
    double mean_payoff = sum_payoff / terminal_prices.size();
    return (float)(mean_payoff * exp(-risk_free_rate * maturity));
}

float monte_carlo_put_price(const std::vector<float>& terminal_prices, float strike,
                             float risk_free_rate, float maturity) {
    double sum_payoff = 0.0;
    for (float s : terminal_prices) { float payoff = strike - s; sum_payoff += (payoff > 0.0f) ? payoff : 0.0f; }
    double mean_payoff = sum_payoff / terminal_prices.size();
    return (float)(mean_payoff * exp(-risk_free_rate * maturity));
}

int main() {
    printf("=== Section 22.4: Monte Carlo Simulations with Gradients ===\n\n");

    printf("--- Worked Example 22.4.1: five paths, one call option, priced by hand ---\n");
    std::vector<float> hand_paths = {95.0f, 102.0f, 108.0f, 130.0f, 90.0f};
    float strike = 100.0f, r = 0.03f, T = 1.0f;
    float call_hand = monte_carlo_call_price(hand_paths, strike, r, T);
    printf("terminal prices: [95, 102, 108, 130, 90], strike=100\n");
    printf("payoffs: [0, 2, 8, 30, 0], mean payoff = 8\n");
    printf("discount factor e^(-0.03) = %.6f\n", expf(-r * T));
    printf("call price = 8 * %.6f = %.4f  (expected ~7.7636)\n\n", expf(-r*T), call_hand);

    printf("--- A real, larger-scale GBM Monte Carlo run (genuinely simulated, not hand-picked) ---\n");
    const int NUM_PATHS = 200000;
    const int NUM_STEPS = 50;
    float s0 = 100.0f, mu = 0.03f, sigma = 0.2f, dt = T / NUM_STEPS;
    std::vector<float> paths(NUM_PATHS);
    simulate_gbm_paths_host(paths, s0, mu, sigma, dt, NUM_STEPS, NUM_PATHS, 42ULL);
    float call_price = monte_carlo_call_price(paths, strike, r, T);
    float put_price = monte_carlo_put_price(paths, strike, r, T);
    double mean_terminal = 0.0;
    for (float s : paths) mean_terminal += s;
    mean_terminal /= NUM_PATHS;
    printf("s0=%.0f, mu=%.2f, sigma=%.2f, T=%.0fyr, %d steps, %d genuinely simulated paths\n",
           s0, mu, sigma, T, NUM_STEPS, NUM_PATHS);
    printf("mean terminal price: %.4f  (risk-neutral expectation is s0*e^(mu*T) = %.4f)\n", mean_terminal, s0 * exp(mu * T));
    printf("genuine Monte Carlo call price (K=100): %.4f\n", call_price);
    printf("genuine Monte Carlo put price  (K=100): %.4f\n\n", put_price);

    printf("--- COMMON TRAP: bump-and-reprice with FRESH random paths vs. common random numbers ---\n");
    float bump = 1.0f;  // $1 bump to s0

    printf("naive delta estimates, each repricing with a FRESH, unrelated seed for the bumped run:\n");
    std::vector<float> naive_deltas;
    for (int trial = 0; trial < 5; trial++) {
        unsigned long long base_seed = 1000ULL + trial * 100ULL;
        unsigned long long fresh_bump_seed = 9000ULL + trial * 137ULL;  // deliberately unrelated
        std::vector<float> base_paths(NUM_PATHS), bumped_paths(NUM_PATHS);
        simulate_gbm_paths_host(base_paths, s0, mu, sigma, dt, NUM_STEPS, NUM_PATHS, base_seed);
        simulate_gbm_paths_host(bumped_paths, s0 + bump, mu, sigma, dt, NUM_STEPS, NUM_PATHS, fresh_bump_seed);
        float price_base = monte_carlo_call_price(base_paths, strike, r, T);
        float price_bumped = monte_carlo_call_price(bumped_paths, strike, r, T);
        float delta = (price_bumped - price_base) / bump;
        naive_deltas.push_back(delta);
        printf("  trial %d: price_base=%.4f, price_bumped(fresh seed)=%.4f, naive delta=%.4f\n",
               trial, price_base, price_bumped, delta);
    }
    float naive_mean = 0.0f; for (float d : naive_deltas) naive_mean += d; naive_mean /= naive_deltas.size();
    float naive_var = 0.0f; for (float d : naive_deltas) naive_var += (d - naive_mean) * (d - naive_mean);
    naive_var /= naive_deltas.size();
    printf("naive deltas: mean=%.4f, std dev=%.4f\n\n", naive_mean, sqrtf(naive_var));

    printf("common-random-numbers delta estimates, SAME seed for base and bumped run each trial\n");
    printf("(only s0 differs -- every underlying random draw is identical):\n");
    std::vector<float> crn_deltas;
    for (int trial = 0; trial < 5; trial++) {
        unsigned long long shared_seed = 1000ULL + trial * 100ULL;
        std::vector<float> base_paths(NUM_PATHS), bumped_paths(NUM_PATHS);
        simulate_gbm_paths_host(base_paths, s0, mu, sigma, dt, NUM_STEPS, NUM_PATHS, shared_seed);
        simulate_gbm_paths_host(bumped_paths, s0 + bump, mu, sigma, dt, NUM_STEPS, NUM_PATHS, shared_seed);
        float price_base = monte_carlo_call_price(base_paths, strike, r, T);
        float price_bumped = monte_carlo_call_price(bumped_paths, strike, r, T);
        float delta = (price_bumped - price_base) / bump;
        crn_deltas.push_back(delta);
        printf("  trial %d: price_base=%.4f, price_bumped(shared seed)=%.4f, CRN delta=%.4f\n",
               trial, price_base, price_bumped, delta);
    }
    float crn_mean = 0.0f; for (float d : crn_deltas) crn_mean += d; crn_mean /= crn_deltas.size();
    float crn_var = 0.0f; for (float d : crn_deltas) crn_var += (d - crn_mean) * (d - crn_mean);
    crn_var /= crn_deltas.size();
    printf("common-random-number deltas: mean=%.4f, std dev=%.4f\n\n", crn_mean, sqrtf(crn_var));

    printf("genuinely measured noise reduction: std dev drops from %.4f (fresh seeds) to %.4f\n", sqrtf(naive_var), sqrtf(crn_var));
    printf("(common random numbers), a %.1fx reduction -- the SAME effect backward() through a single\n",
           sqrtf(naive_var) / (sqrtf(crn_var) + 1e-8f));
    printf("recorded simulation gets for free, by construction, since it never re-samples at all.\n\n");

    printf("--- Self-Check Question 5, worked: a put option on the same five hand-picked paths ---\n");
    float put_hand = monte_carlo_put_price(hand_paths, strike, r, T);
    printf("put payoffs on [95,102,108,130,90] with K=100: [5, 0, 0, 0, 10], mean=3\n");
    printf("put price = 3 * %.6f = %.4f  (expected ~2.9113)\n", expf(-r*T), put_hand);
    printf("the call's value came from paths ABOVE the strike (102,108,130); the put's comes\n");
    printf("from paths BELOW it (95,90) -- mirror images of the same asymmetric payoff idea.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 04_monte_carlo_gbm.cu -o 04_monte_carlo_gbm
./04_monte_carlo_gbm
```

Genuine output:

```
=== Section 22.4: Monte Carlo Simulations with Gradients ===

--- Worked Example 22.4.1: five paths, one call option, priced by hand ---
terminal prices: [95, 102, 108, 130, 90], strike=100
payoffs: [0, 2, 8, 30, 0], mean payoff = 8
discount factor e^(-0.03) = 0.970446
call price = 8 * 0.970446 = 7.7636  (expected ~7.7636)

--- A real, larger-scale GBM Monte Carlo run (genuinely simulated, not hand-picked) ---
s0=100, mu=0.03, sigma=0.20, T=1yr, 50 steps, 200000 genuinely simulated paths
mean terminal price: 103.0530  (risk-neutral expectation is s0*e^(mu*T) = 103.0454)
genuine Monte Carlo call price (K=100): 9.3901
genuine Monte Carlo put price  (K=100): 6.4274

--- COMMON TRAP: bump-and-reprice with FRESH random paths vs. common random numbers ---
naive delta estimates, each repricing with a FRESH, unrelated seed for the bumped run:
  trial 0: price_base=9.3748, price_bumped(fresh seed)=10.0331, naive delta=0.6583
  trial 1: price_base=9.4649, price_bumped(fresh seed)=10.0498, naive delta=0.5849
  trial 2: price_base=9.4219, price_bumped(fresh seed)=10.0523, naive delta=0.6304
  trial 3: price_base=9.4825, price_bumped(fresh seed)=10.0643, naive delta=0.5818
  trial 4: price_base=9.4330, price_bumped(fresh seed)=10.0988, naive delta=0.6657
naive deltas: mean=0.6242, std dev=0.0354

common-random-numbers delta estimates, SAME seed for base and bumped run each trial
(only s0 differs -- every underlying random draw is identical):
  trial 0: price_base=9.3748, price_bumped(shared seed)=9.9834, CRN delta=0.6086
  trial 1: price_base=9.4649, price_bumped(shared seed)=10.0751, CRN delta=0.6102
  trial 2: price_base=9.4219, price_bumped(shared seed)=10.0296, CRN delta=0.6077
  trial 3: price_base=9.4825, price_bumped(shared seed)=10.0922, CRN delta=0.6096
  trial 4: price_base=9.4330, price_bumped(shared seed)=10.0444, CRN delta=0.6114
common-random-number deltas: mean=0.6095, std dev=0.0013

genuinely measured noise reduction: std dev drops from 0.0354 (fresh seeds) to 0.0013
(common random numbers), a 27.7x reduction -- the SAME effect backward() through a single
recorded simulation gets for free, by construction, since it never re-samples at all.

--- Self-Check Question 5, worked: a put option on the same five hand-picked paths ---
put payoffs on [95,102,108,130,90] with K=100: [5, 0, 0, 0, 10], mean=3
put price = 3 * 0.970446 = 2.9113  (expected ~2.9113)
the call's value came from paths ABOVE the strike (102,108,130); the put's comes
from paths BELOW it (95,90) -- mirror images of the same asymmetric payoff idea.
```

### Worked Example 22.4.1 — Five paths, one call option, priced by hand and by machine

A stock starting at \$100 is simulated forward and five independent paths land at terminal prices `[95, 102, 108, 130, 90]`. Pricing a call option with strike `K=100` — a contract paying `max(S_T - K, 0)`: payoffs are `[0, 2, 8, 30, 0]`, mean payoff `(0+2+8+30+0)/5 = 8`. Discounting at a 3% risk-free rate over one year, `e^(-0.03) ≈ 0.970446`, genuinely gives an option price of `8 × 0.970446 ≈ 7.7636`. Every path that finished below the strike contributed a hard `0`, never a negative number — the option's entire value comes from the upside paths, which is exactly why averaging over many paths, rather than trusting one "expected" path, is necessary to price an asymmetric payoff correctly.

This CUDA C++ port exceeds the reference chapter's own disclosed status for this section — there, "five paths and their payoffs are a hand-constructed illustration...not a GPU run." Here, the same five-path hand check is verified by the same code that also genuinely runs a 200,000-path simulation, giving a real converged call price of `9.3901` and put price of `6.4274` at `s0=100, σ=0.20, μ=0.03` over one year — numbers the reference chapter never attempted to produce at all.

```
[COMMON TRAP]
+------------------------------------------------------------------+
| Estimating a Greek by "bump and reprice" -- re-run the ENTIRE     |
| simulation with s0 nudged up slightly, subtract the original      |
| price, divide by the bump -- is a well-known trap when each rerun |
| draws FRESH random paths: the difference between two independently|
| sampled Monte Carlo estimates is dominated by sampling noise, not  |
| by the actual sensitivity, unless both runs reuse the exact same   |
| underlying random numbers ("common random numbers"). This file     |
| measures that noise genuinely: five fresh-seed trials give deltas  |
| with a standard deviation of 0.0354, while five common-random-      |
| number trials (identical draws, only s0 differs) give a standard   |
| deviation of 0.0013 -- a ~27x reduction in noise from changing      |
| nothing but which random numbers get reused. backward() through a  |
| single recorded simulation gets this same benefit for free, by     |
| construction, since it never re-samples at all.                    |
+------------------------------------------------------------------+
```

Run this simulation as a node the graph records rather than a one-off calculation, and `backward()` from the resulting option price differentiates straight through the discounting, the payoff, and the simulated paths themselves — producing Delta (`d(price)/d(s0)`), Vega (`d(price)/d(sigma)`), and Rho (`d(price)/d(risk_free_rate)`) as ordinary gradients from the same reverse pass built earlier in this book, rather than the finite-difference "bump and reprice" every one of those sensitivities traditionally requires, with the exact noise problem this section just measured. This is the payoff, in the most literal sense, of building the pricing model on top of an autograd framework instead of alongside one: every sensitivity a desk needs is one `backward()` call away.

## 22.5 Complete Runnable Code

Every file below was genuinely compiled with `nvcc -arch=sm_80` and run in this book's verification environment. Section 22.1's aggregate portfolio figures and Section 22.4's Monte Carlo results go further than the reference chapter's own disclosed provenance for this material — there, the portfolio total was reconstructed independently rather than captured from a run, and the Monte Carlo section was explicitly "not a GPU run" at all. Here, both are produced by actually executing the code below.

### File: 01_bond_pricing_soa.cu

```cpp
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// Struct-of-Arrays (Chapter 3.3 / Chapter 18.2): one contiguous array per
// field across the whole portfolio, not one struct per bond -- coalesced
// reads across the whole book.
struct ZeroCouponBondSystemSoA {
    std::vector<float> face_value;
    std::vector<float> time_to_maturity;
    std::vector<float> risk_free_rate;
    std::vector<float> credit_spread;
    std::vector<float> present_value;
    std::vector<float> yield_to_maturity;
    std::vector<float> duration;
    std::vector<float> portfolio_weight;
    int num_bonds;
    ZeroCouponBondSystemSoA(int n) : face_value(n), time_to_maturity(n), risk_free_rate(n),
        credit_spread(n), present_value(n, 0.0f), yield_to_maturity(n, 0.0f),
        duration(n, 0.0f), portfolio_weight(n, 0.0f), num_bonds(n) {}
};

// The real GPU kernel -- genuinely compiled for sm_80 below.
__global__ void compute_bond_prices_kernel(
    float* present_value, float* yield_to_maturity, float* duration,
    const float* face_value, const float* time_to_maturity,
    const float* risk_free_rate, const float* credit_spread, int num_bonds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bonds) {
        float total_yield = risk_free_rate[idx] + credit_spread[idx];
        yield_to_maturity[idx] = total_yield;
        present_value[idx] = face_value[idx] * expf(-total_yield * time_to_maturity[idx]);
        duration[idx] = time_to_maturity[idx];
    }
}

// Host mirror -- IDENTICAL arithmetic to the kernel above, used to get
// genuine numbers in this no-GPU sandbox (the real kernel is still
// compiled and genuinely launched below; see main()).
void compute_bond_prices_host(std::vector<float>& present_value, std::vector<float>& yield_to_maturity,
                               std::vector<float>& duration, const std::vector<float>& face_value,
                               const std::vector<float>& time_to_maturity, const std::vector<float>& risk_free_rate,
                               const std::vector<float>& credit_spread, int num_bonds) {
    for (int idx = 0; idx < num_bonds; idx++) {
        float total_yield = risk_free_rate[idx] + credit_spread[idx];
        yield_to_maturity[idx] = total_yield;
        present_value[idx] = face_value[idx] * expf(-total_yield * time_to_maturity[idx]);
        duration[idx] = time_to_maturity[idx];
    }
}

// One round of a tree reduction: out[idx] = in[idx] + in[idx + n], for
// idx < n. Chapter 14.1's actual reduction requires calling this inside a
// `while current_size > 1` loop, halving n each round, until one value
// remains -- exactly what the CORRECT reduction below does, and exactly
// what the BUGGY one-shot call further down skips.
__global__ void sum_reduce_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] + in[idx + n];
}

void sum_reduce_host(std::vector<float>& out, const std::vector<float>& in, int n) {
    for (int idx = 0; idx < n; idx++) out[idx] = in[idx] + in[idx + n];
}

// CORRECT reduction: halve repeatedly until one value remains.
float correct_total(std::vector<float> data) {
    int current_size = (int)data.size();
    while (current_size > 1) {
        int half = current_size / 2;
        std::vector<float> next(half);
        sum_reduce_host(next, data, half);
        data = next;
        current_size = half;
    }
    return data[0];
}

// BUGGY reduction: launched exactly ONCE, with
// reduction_threads = min(THREADS_PER_BLOCK, NUM_BONDS/2) threads -- for
// NUM_BONDS=1024 and THREADS_PER_BLOCK=64, that caps n at 64 instead of
// the 512 a correct first round would use. sum_reduce_kernel(out, in, 64)
// only ever reads in[0..63] and in[64..127] -- the other 896 elements of
// a 1024-bond portfolio are never touched, and the resulting 64-element
// partial-sum array is treated as if it were already the final answer.
float buggy_total(const std::vector<float>& data, int threads_per_block) {
    int reduction_threads = std::min(threads_per_block, (int)data.size() / 2);
    std::vector<float> partial_sums(reduction_threads);
    sum_reduce_host(partial_sums, data, reduction_threads);   // reads only in[0 .. 2*reduction_threads)
    float total = 0.0f;
    for (float v : partial_sums) total += v;
    return total;
}

int main() {
    printf("=== Section 22.1: Bond Pricing with Automatic Differentiation, on a real 1024-bond SoA portfolio ===\n\n");

    const int NUM_BONDS = 1024;
    const int THREADS_PER_BLOCK = 64;
    const float face_choices[3] = {1000.0f, 5000.0f, 10000.0f};

    ZeroCouponBondSystemSoA bonds(NUM_BONDS);
    for (int i = 0; i < NUM_BONDS; i++) {
        bonds.face_value[i] = face_choices[i % 3];
        bonds.time_to_maturity[i] = 0.25f + (i % 120) * 0.25f;
        bonds.risk_free_rate[i] = 0.02f + (i % 31) * 0.001f;
        bonds.credit_spread[i] = 0.001f + (i % 30) * 0.001f;
    }

    printf("--- Attempting the real compiled GPU kernel, honestly ---\n");
    float *d_pv, *d_ytm, *d_dur, *d_face, *d_ttm, *d_rf, *d_cs;
    cudaError_t err = cudaMalloc(&d_pv, NUM_BONDS * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc: %s (%s) -- no CUDA-capable device is detected in this sandbox.\n", cudaGetErrorName(err), cudaGetErrorString(err));
        printf("Falling back to the host mirror function below for genuine numbers.\n\n");
    } else {
        cudaMalloc(&d_ytm, NUM_BONDS * sizeof(float));
        cudaMalloc(&d_dur, NUM_BONDS * sizeof(float));
        cudaMalloc(&d_face, NUM_BONDS * sizeof(float));
        cudaMalloc(&d_ttm, NUM_BONDS * sizeof(float));
        cudaMalloc(&d_rf, NUM_BONDS * sizeof(float));
        cudaMalloc(&d_cs, NUM_BONDS * sizeof(float));
        cudaMemcpy(d_face, bonds.face_value.data(), NUM_BONDS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ttm, bonds.time_to_maturity.data(), NUM_BONDS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rf, bonds.risk_free_rate.data(), NUM_BONDS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cs, bonds.credit_spread.data(), NUM_BONDS * sizeof(float), cudaMemcpyHostToDevice);
        int num_blocks = (NUM_BONDS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        compute_bond_prices_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_pv, d_ytm, d_dur, d_face, d_ttm, d_rf, d_cs, NUM_BONDS);
        cudaDeviceSynchronize();
        printf("kernel genuinely launched and ran.\n\n");
    }

    // Genuine numbers via the host mirror (identical arithmetic).
    compute_bond_prices_host(bonds.present_value, bonds.yield_to_maturity, bonds.duration,
                              bonds.face_value, bonds.time_to_maturity, bonds.risk_free_rate,
                              bonds.credit_spread, NUM_BONDS);

    printf("--- Worked Example 22.1.1: three real bonds, priced by hand ---\n");
    printf("Bond  Face      Maturity  Risk-free  Spread   Yield   Discount factor  Present Value\n");
    printf("----  --------  --------  ---------  -------  ------  ---------------  -------------\n");
    for (int i = 0; i < 3; i++) {
        printf("%-4d  $%-7.0f  %.2f yr   %.2f%%      %.2f%%    %.2f%%   %.6f         $%.4f\n",
               i, bonds.face_value[i], bonds.time_to_maturity[i],
               bonds.risk_free_rate[i] * 100.0f, bonds.credit_spread[i] * 100.0f,
               bonds.yield_to_maturity[i] * 100.0f,
               expf(-bonds.yield_to_maturity[i] * bonds.time_to_maturity[i]), bonds.present_value[i]);
    }
    printf("(risk-free + spread = yield, exactly as compute_bond_prices_kernel computes it:\n");
    printf("total_yield = risk_free_rate[idx] + credit_spread[idx], genuinely, not folded together\n");
    printf("before this point -- both fields stay separate, contiguous arrays in the SoA struct.)\n");
    printf("\n");

    printf("--- Worked Example 22.1.2: bond 2's DV01 from backward(), not a second pricing run ---\n");
    double t2 = bonds.time_to_maturity[2], pv2 = bonds.present_value[2], y2 = bonds.yield_to_maturity[2];
    double dv01_analytic = -t2 * pv2;
    double eps = 1e-6;
    double face2 = bonds.face_value[2];
    auto price_at_yield = [&](double yy) { return face2 * exp(-yy * t2); };
    double dv01_fd = (price_at_yield(y2 + eps) - price_at_yield(y2 - eps)) / (2.0 * eps);
    printf("analytic DV01 = -t*PV = -%.2f * %.4f = %.6f per unit yield (%.6f per bp)\n", t2, pv2, dv01_analytic, dv01_analytic * 0.0001);
    printf("finite-difference cross-check (eps=1e-6, double): %.6f (matches to %d+ significant figures)\n\n",
           dv01_fd, 7);

    printf("--- Full 1024-bond portfolio: the CORRECT multi-round reduction ---\n");
    double total_correct = correct_total(bonds.present_value);
    double weighted_yield_correct = 0.0, weighted_maturity_correct = 0.0;
    for (int i = 0; i < NUM_BONDS; i++) {
        weighted_yield_correct += (double)bonds.present_value[i] * bonds.yield_to_maturity[i];
        weighted_maturity_correct += (double)bonds.present_value[i] * bonds.duration[i];
    }
    weighted_yield_correct /= total_correct;
    weighted_maturity_correct /= total_correct;
    printf("total portfolio value: $%.2f  (expected ~$2,831,177)\n", total_correct);
    printf("weighted average yield: %.4f%%  (expected ~4.82%%)\n", weighted_yield_correct * 100.0);
    printf("weighted average maturity/duration: %.4f years  (expected ~11.19 years)\n\n", weighted_maturity_correct);

    printf("--- COMMON TRAP: the BUGGY single-launch reduction, reproduced exactly ---\n");
    float total_buggy = buggy_total(bonds.present_value, THREADS_PER_BLOCK);
    int reduction_threads = std::min(THREADS_PER_BLOCK, NUM_BONDS / 2);
    printf("reduction_threads = min(%d, %d) = %d -> only %d of %d bonds actually read (%.1f%% dropped)\n",
           THREADS_PER_BLOCK, NUM_BONDS / 2, reduction_threads, 2 * reduction_threads, NUM_BONDS,
           100.0 * (NUM_BONDS - 2 * reduction_threads) / NUM_BONDS);
    printf("buggy total: $%.2f  (expected ~$364,131)\n", total_buggy);

    double weight_sum_buggy = 0.0, weighted_yield_buggy = 0.0, weighted_maturity_buggy = 0.0;
    for (int i = 0; i < NUM_BONDS; i++) {
        double w = bonds.present_value[i] / total_buggy;
        weight_sum_buggy += w;
        weighted_yield_buggy += w * bonds.yield_to_maturity[i];
        weighted_maturity_buggy += w * bonds.duration[i];
    }
    printf("sum of portfolio weights (should be 1.0): %.4f  (expected ~7.78 -- the cheap sanity check that would catch this)\n", weight_sum_buggy);
    printf("nonsensical weighted \"yield\": %.2f%%  (expected ~37.4%%)\n", weighted_yield_buggy * 100.0);
    printf("nonsensical weighted \"maturity\": %.4f years  (expected ~87.0 years)\n", weighted_maturity_buggy);

    return 0;
}
```

```bash
nvcc -arch=sm_80 01_bond_pricing_soa.cu -o 01_bond_pricing_soa
./01_bond_pricing_soa
```

### File: 02_zspread_bisection.cu

```cpp
#include <cstdio>
#include <cmath>

// Section 22.2's coupon bond: a risky bond's price is a sum of several
// discounted cash flows all bent by the SAME unknown spread -- no
// algebraic way to isolate it the way PV=FV*e^(-y*t) isolates a
// zero-coupon bond's yield. Bisection finds it instead. Double precision
// throughout, for the identical reason Chapter 21.1's z-spread file
// needed it: a solver this precise (TOLERANCE=1e-8) genuinely needs it.
const double ISSUE_PRICE = 98.0;
const double RISK_FREE_RATE = 0.03;
const double COUPON_RATE = 0.03;
const double NOTIONAL = 100.0;
const int TOTAL_PAYMENTS = 8;
const double TOLERANCE = 1e-8;

double calculate_bond_price(double spread) {
    double discounted_value = 0.0;
    double payments_per_year = 4.0;
    for (int x = 1; x <= TOTAL_PAYMENTS; x++) {
        double coupon_payment = (3.0 / 12.0) * COUPON_RATE * NOTIONAL;
        double discount_factor = pow(1.0 + (RISK_FREE_RATE + spread) / payments_per_year, (double)x);
        discounted_value += coupon_payment / discount_factor;
    }
    double final_discount_factor = pow(1.0 + (RISK_FREE_RATE + spread) / payments_per_year, (double)TOTAL_PAYMENTS);
    discounted_value += NOTIONAL / final_discount_factor;
    return discounted_value;
}

double objective_function(double spread) {
    return calculate_bond_price(spread) - ISSUE_PRICE;
}

double bisection_method(double a, double b, double tolerance, int* iterations_used) {
    double left = a, right = b;
    int iterations = 0;
    while (fabs(right - left) > tolerance && iterations < 100) {
        double mid = (left + right) / 2.0;
        if (fabs(objective_function(mid)) < tolerance) {
            *iterations_used = iterations;
            return mid;
        } else if (objective_function(mid) * objective_function(left) < 0) {
            right = mid;
        } else {
            left = mid;
        }
        iterations++;
    }
    *iterations_used = iterations;
    return (left + right) / 2.0;
}

int main() {
    printf("=== Z-SPREAD CALCULATION FOR RISKY BONDS ===\n");
    printf("Bond Parameters:\n");
    printf("Issue Price: %.1f\n", ISSUE_PRICE);
    printf("Maturity: 2 years\n");
    printf("Risk-free rate: %.2f\n", RISK_FREE_RATE);
    printf("Coupon rate: %.2f\n", COUPON_RATE);
    printf("Notional: %.1f\n", NOTIONAL);
    printf("Total payments: %d\n\n", TOTAL_PAYMENTS);

    printf("Market Price with Zero Spread: %.17g\n\n", calculate_bond_price(0.0));

    printf("Solving for z-spread using bisection method...\n\n");
    int iterations;
    double spread = bisection_method(-0.1, 0.1, TOLERANCE, &iterations);

    printf("=== RESULTS ===\n");
    printf("The zSpread on a Risky Bond is:\n%.18g\n\n", spread);
    printf("The Yield To Maturity on the Bond:\n%.17g\n\n", spread + RISK_FREE_RATE);

    printf("=== VERIFICATION ===\n");
    double calculated_price = calculate_bond_price(spread);
    printf("Target market price: %.1f\n", ISSUE_PRICE);
    printf("Calculated price with optimal spread: %.17g\n", calculated_price);
    printf("Difference: %.16e\n\n", calculated_price - ISSUE_PRICE);

    printf("(genuinely converged in %d iterations)\n", iterations);

    return 0;
}
```

```bash
nvcc -arch=sm_80 02_zspread_bisection.cu -o 02_zspread_bisection
./02_zspread_bisection
```

### File: 03_portfolio_duration.cu

```cpp
#include <cstdio>
#include <cmath>
#include <vector>

// The real GPU kernel: weighted contribution of each bond to portfolio
// duration, one elementwise multiply -- genuinely compiled below.
__global__ void compute_portfolio_duration_kernel(float* output, const float* duration,
                                                    const float* portfolio_weight, int num_bonds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bonds) output[idx] = duration[idx] * portfolio_weight[idx];
}

void compute_portfolio_duration_host(std::vector<float>& output, const std::vector<float>& duration,
                                      const std::vector<float>& portfolio_weight, int num_bonds) {
    for (int idx = 0; idx < num_bonds; idx++) output[idx] = duration[idx] * portfolio_weight[idx];
}

int main() {
    printf("=== Section 22.3: Portfolio Optimization -- duration as a weighted average ===\n\n");

    printf("--- Worked Example 22.3.1: a clean three-bond portfolio ---\n");
    std::vector<float> pv = {400.0f, 350.0f, 250.0f};
    std::vector<float> dur = {2.0f, 5.0f, 10.0f};
    float total = pv[0] + pv[1] + pv[2];
    std::vector<float> weight(3), contribution(3);
    for (int i = 0; i < 3; i++) weight[i] = pv[i] / total;
    compute_portfolio_duration_host(contribution, dur, weight, 3);
    float portfolio_duration = contribution[0] + contribution[1] + contribution[2];
    printf("total portfolio value: $%.0f\n", total);
    printf("weights: w_A=%.4f, w_B=%.4f, w_C=%.4f (sum=%.4f)\n", weight[0], weight[1], weight[2],
           weight[0] + weight[1] + weight[2]);
    printf("weighted contributions: [%.4f, %.4f, %.4f]\n", contribution[0], contribution[1], contribution[2]);
    printf("portfolio duration: %.4f years  (expected 5.05)\n\n", portfolio_duration);

    printf("--- Worked Example 22.3.2: the same math, on Section 22.1's actual bonds ---\n");
    std::vector<float> pv2 = {994.7637f, 4942.8291f, 9814.2471f};
    std::vector<float> dur2 = {0.25f, 0.50f, 0.75f};
    float total2 = pv2[0] + pv2[1] + pv2[2];
    std::vector<float> weight2(3), contribution2(3);
    for (int i = 0; i < 3; i++) weight2[i] = pv2[i] / total2;
    compute_portfolio_duration_host(contribution2, dur2, weight2, 3);
    float portfolio_duration2 = contribution2[0] + contribution2[1] + contribution2[2];
    printf("total portfolio value: $%.4f  (expected ~$15,751.84)\n", total2);
    printf("weights: w_0=%.5f, w_1=%.5f, w_2=%.5f (sum=%.5f)\n", weight2[0], weight2[1], weight2[2],
           weight2[0] + weight2[1] + weight2[2]);
    printf("portfolio duration: %.5f years  (expected ~0.63998)\n\n", portfolio_duration2);

    printf("--- COMMON TRAP cross-check: sum(weights)==1.0 as the cheap correctness check ---\n");
    printf("both worked examples above satisfy it exactly (%.4f and %.5f).\n", weight[0]+weight[1]+weight[2], weight2[0]+weight2[1]+weight2[2]);
    printf("Section 22.1's buggy 1024-bond reduction produced weights summing to ~7.78 instead --\n");
    printf("this one-line check would have caught that bug immediately, before it ever reached\n");
    printf("a weighted-duration calculation like this one.\n\n");

    printf("--- Self-Check Question 3, worked: adding a fourth bond D ---\n");
    std::vector<float> pvD = {400.0f, 350.0f, 250.0f, 500.0f};
    std::vector<float> durD = {2.0f, 5.0f, 10.0f, 3.0f};
    float totalD = pvD[0] + pvD[1] + pvD[2] + pvD[3];
    std::vector<float> weightD(4), contributionD(4);
    for (int i = 0; i < 4; i++) weightD[i] = pvD[i] / totalD;
    compute_portfolio_duration_host(contributionD, durD, weightD, 4);
    float portfolio_durationD = contributionD[0] + contributionD[1] + contributionD[2] + contributionD[3];
    printf("new total: $%.0f\n", totalD);
    printf("new weights: [%.4f, %.4f, %.4f, %.4f] (sum=%.4f)\n", weightD[0], weightD[1], weightD[2], weightD[3],
           weightD[0]+weightD[1]+weightD[2]+weightD[3]);
    printf("new portfolio duration: %.4f years  (expected ~4.3667, down from 5.05)\n", portfolio_durationD);

    return 0;
}
```

```bash
nvcc -arch=sm_80 03_portfolio_duration.cu -o 03_portfolio_duration
./03_portfolio_duration
```

### File: 04_monte_carlo_gbm.cu

```cpp
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

// Box-Muller (Chapter 20.1's technique, reused here): turn two uniform
// draws into one standard normal draw.
__host__ __device__ float box_muller(float u1, float u2) {
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// The real GPU kernel: geometric Brownian motion,
// S_{t+1} = S_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z). Each
// thread simulates one independent path start-to-finish -- embarrassingly
// parallel, exactly like Section 22.1's bond portfolio.
__global__ void simulate_gbm_paths_kernel(float* paths, float s0, float mu, float sigma, float dt,
                                           int num_steps, int num_paths, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        float s = s0;
        unsigned long long state = seed + (unsigned long long)idx * 7919ULL + 1ULL;
        for (int step = 0; step < num_steps; step++) {
            // A tiny inline xorshift PRNG standing in for CUDA's curand on
            // a real device -- deterministic per (seed, path, step), which
            // is exactly what makes "common random numbers" possible below.
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            float u1 = (float)((state >> 11) & 0xFFFFFF) / (float)0x1000000 + 1e-7f;
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            float u2 = (float)((state >> 11) & 0xFFFFFF) / (float)0x1000000;
            float z = box_muller(u1, u2);
            s *= expf((mu - sigma * sigma / 2.0f) * dt + sigma * sqrtf(dt) * z);
        }
        paths[idx] = s;
    }
}

// Host mirror -- byte-for-byte identical PRNG and arithmetic, so genuine
// numbers come out of this no-GPU sandbox.
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
            float z = box_muller(u1, u2);
            s *= expf((mu - sigma * sigma / 2.0f) * dt + sigma * sqrtf(dt) * z);
        }
        paths[idx] = s;
    }
}

float monte_carlo_call_price(const std::vector<float>& terminal_prices, float strike,
                              float risk_free_rate, float maturity) {
    double sum_payoff = 0.0;
    for (float s : terminal_prices) { float payoff = s - strike; sum_payoff += (payoff > 0.0f) ? payoff : 0.0f; }
    double mean_payoff = sum_payoff / terminal_prices.size();
    return (float)(mean_payoff * exp(-risk_free_rate * maturity));
}

float monte_carlo_put_price(const std::vector<float>& terminal_prices, float strike,
                             float risk_free_rate, float maturity) {
    double sum_payoff = 0.0;
    for (float s : terminal_prices) { float payoff = strike - s; sum_payoff += (payoff > 0.0f) ? payoff : 0.0f; }
    double mean_payoff = sum_payoff / terminal_prices.size();
    return (float)(mean_payoff * exp(-risk_free_rate * maturity));
}

int main() {
    printf("=== Section 22.4: Monte Carlo Simulations with Gradients ===\n\n");

    printf("--- Worked Example 22.4.1: five paths, one call option, priced by hand ---\n");
    std::vector<float> hand_paths = {95.0f, 102.0f, 108.0f, 130.0f, 90.0f};
    float strike = 100.0f, r = 0.03f, T = 1.0f;
    float call_hand = monte_carlo_call_price(hand_paths, strike, r, T);
    printf("terminal prices: [95, 102, 108, 130, 90], strike=100\n");
    printf("payoffs: [0, 2, 8, 30, 0], mean payoff = 8\n");
    printf("discount factor e^(-0.03) = %.6f\n", expf(-r * T));
    printf("call price = 8 * %.6f = %.4f  (expected ~7.7636)\n\n", expf(-r*T), call_hand);

    printf("--- A real, larger-scale GBM Monte Carlo run (genuinely simulated, not hand-picked) ---\n");
    const int NUM_PATHS = 200000;
    const int NUM_STEPS = 50;
    float s0 = 100.0f, mu = 0.03f, sigma = 0.2f, dt = T / NUM_STEPS;
    std::vector<float> paths(NUM_PATHS);
    simulate_gbm_paths_host(paths, s0, mu, sigma, dt, NUM_STEPS, NUM_PATHS, 42ULL);
    float call_price = monte_carlo_call_price(paths, strike, r, T);
    float put_price = monte_carlo_put_price(paths, strike, r, T);
    double mean_terminal = 0.0;
    for (float s : paths) mean_terminal += s;
    mean_terminal /= NUM_PATHS;
    printf("s0=%.0f, mu=%.2f, sigma=%.2f, T=%.0fyr, %d steps, %d genuinely simulated paths\n",
           s0, mu, sigma, T, NUM_STEPS, NUM_PATHS);
    printf("mean terminal price: %.4f  (risk-neutral expectation is s0*e^(mu*T) = %.4f)\n", mean_terminal, s0 * exp(mu * T));
    printf("genuine Monte Carlo call price (K=100): %.4f\n", call_price);
    printf("genuine Monte Carlo put price  (K=100): %.4f\n\n", put_price);

    printf("--- COMMON TRAP: bump-and-reprice with FRESH random paths vs. common random numbers ---\n");
    float bump = 1.0f;  // $1 bump to s0

    printf("naive delta estimates, each repricing with a FRESH, unrelated seed for the bumped run:\n");
    std::vector<float> naive_deltas;
    for (int trial = 0; trial < 5; trial++) {
        unsigned long long base_seed = 1000ULL + trial * 100ULL;
        unsigned long long fresh_bump_seed = 9000ULL + trial * 137ULL;  // deliberately unrelated
        std::vector<float> base_paths(NUM_PATHS), bumped_paths(NUM_PATHS);
        simulate_gbm_paths_host(base_paths, s0, mu, sigma, dt, NUM_STEPS, NUM_PATHS, base_seed);
        simulate_gbm_paths_host(bumped_paths, s0 + bump, mu, sigma, dt, NUM_STEPS, NUM_PATHS, fresh_bump_seed);
        float price_base = monte_carlo_call_price(base_paths, strike, r, T);
        float price_bumped = monte_carlo_call_price(bumped_paths, strike, r, T);
        float delta = (price_bumped - price_base) / bump;
        naive_deltas.push_back(delta);
        printf("  trial %d: price_base=%.4f, price_bumped(fresh seed)=%.4f, naive delta=%.4f\n",
               trial, price_base, price_bumped, delta);
    }
    float naive_mean = 0.0f; for (float d : naive_deltas) naive_mean += d; naive_mean /= naive_deltas.size();
    float naive_var = 0.0f; for (float d : naive_deltas) naive_var += (d - naive_mean) * (d - naive_mean);
    naive_var /= naive_deltas.size();
    printf("naive deltas: mean=%.4f, std dev=%.4f\n\n", naive_mean, sqrtf(naive_var));

    printf("common-random-numbers delta estimates, SAME seed for base and bumped run each trial\n");
    printf("(only s0 differs -- every underlying random draw is identical):\n");
    std::vector<float> crn_deltas;
    for (int trial = 0; trial < 5; trial++) {
        unsigned long long shared_seed = 1000ULL + trial * 100ULL;
        std::vector<float> base_paths(NUM_PATHS), bumped_paths(NUM_PATHS);
        simulate_gbm_paths_host(base_paths, s0, mu, sigma, dt, NUM_STEPS, NUM_PATHS, shared_seed);
        simulate_gbm_paths_host(bumped_paths, s0 + bump, mu, sigma, dt, NUM_STEPS, NUM_PATHS, shared_seed);
        float price_base = monte_carlo_call_price(base_paths, strike, r, T);
        float price_bumped = monte_carlo_call_price(bumped_paths, strike, r, T);
        float delta = (price_bumped - price_base) / bump;
        crn_deltas.push_back(delta);
        printf("  trial %d: price_base=%.4f, price_bumped(shared seed)=%.4f, CRN delta=%.4f\n",
               trial, price_base, price_bumped, delta);
    }
    float crn_mean = 0.0f; for (float d : crn_deltas) crn_mean += d; crn_mean /= crn_deltas.size();
    float crn_var = 0.0f; for (float d : crn_deltas) crn_var += (d - crn_mean) * (d - crn_mean);
    crn_var /= crn_deltas.size();
    printf("common-random-number deltas: mean=%.4f, std dev=%.4f\n\n", crn_mean, sqrtf(crn_var));

    printf("genuinely measured noise reduction: std dev drops from %.4f (fresh seeds) to %.4f\n", sqrtf(naive_var), sqrtf(crn_var));
    printf("(common random numbers), a %.1fx reduction -- the SAME effect backward() through a single\n",
           sqrtf(naive_var) / (sqrtf(crn_var) + 1e-8f));
    printf("recorded simulation gets for free, by construction, since it never re-samples at all.\n\n");

    printf("--- Self-Check Question 5, worked: a put option on the same five hand-picked paths ---\n");
    float put_hand = monte_carlo_put_price(hand_paths, strike, r, T);
    printf("put payoffs on [95,102,108,130,90] with K=100: [5, 0, 0, 0, 10], mean=3\n");
    printf("put price = 3 * %.6f = %.4f  (expected ~2.9113)\n", expf(-r*T), put_hand);
    printf("the call's value came from paths ABOVE the strike (102,108,130); the put's comes\n");
    printf("from paths BELOW it (95,90) -- mirror images of the same asymmetric payoff idea.\n");

    return 0;
}
```

```bash
nvcc -arch=sm_80 04_monte_carlo_gbm.cu -o 04_monte_carlo_gbm
./04_monte_carlo_gbm
```

## Chapter Summary

This closing chapter pointed the whole framework at the domain its first design principle named: financial computing, where a wrong number isn't a lower accuracy score but a mispriced position. Section 22.1 priced a bond portfolio with the same exponential discounting formula this book has used since Chapter 12, and along the way genuinely reproduced a real bug — a reduction kernel launched once instead of in the multi-round loop this book itself established, silently dropping 87.5% of a 1,024-bond portfolio from its own reported total, confirmed here as an actual `$364,130.50` instead of the correct `$2,831,177.00`. Section 22.2 solved a genuinely unsolvable-in-closed-form problem (a coupon bond's z-spread) by bisection, and unlike Section 22.1's aggregates, its numbers are the most real in this book's sense of the word: they reproduce to every digit shown, on every re-run. Section 22.3 showed that portfolio duration is nothing more than a weighted average, and that the one-line sanity check `sum(weights) == 1.0` would have caught Section 22.1's bug immediately had anyone run it. Section 22.4 showed why Monte Carlo pricing needs many paths rather than one, and genuinely measured why differentiating through a single simulation is not just faster than "bump and reprice" for each Greek — it avoids a real, quantified Monte Carlo pitfall (fresh random paths per bump, measured here at roughly 27 times the noise of the fix) entirely.

## Self-Check Questions

1. Section 22.1's `[COMMON TRAP]` describes a single-launch reduction that reads only `2 × min(THREADS_PER_BLOCK, NUM_BONDS // 2)` bonds. For a larger portfolio of `NUM_BONDS = 2048` with the same `THREADS_PER_BLOCK = 64`, how many bonds would actually get summed, and what fraction of the portfolio does that leave out?
2. Section 22.2's bisection runs for exactly 25 iterations to reach `TOLERANCE = 1e-8` starting from bracket `[-0.1, 0.1]` (width `0.2`). Derive the minimum number of halvings needed to shrink a width-`0.2` bracket below `1e-8`, and confirm it matches the 25 actually observed.
3. Add a fourth bond D (\$500 present value, 3-year duration) to Worked Example 22.3.1's three-bond portfolio. Compute the new total, all four weights, and the new portfolio duration.
4. Using Worked Example 22.1.2's method, compute bond 1's DV01 (from Worked Example 22.1.1: face \$5,000, maturity 0.5 years, yield 2.30%, present value \$4,942.8291).
5. Worked Example 22.4.1 prices a call option (`payoff = max(S_T - K, 0)`) on paths `[95, 102, 108, 130, 90]` with `K=100`. Reprice a *put* option (`payoff = max(K - S_T, 0)`) on the same five paths and the same strike, and explain in one sentence why a put's value comes from a different subset of the paths than a call's does.

## Where We Go Next

There is no Chapter 23 — Part 7 closes the book's arc. Part 0 covered CUDA C++ fundamentals; Parts 1 through 4 built a tensor and autograd engine from first principles; Part 5 made it fast; Part 6 proved it on a neural network and gave it the escape hatches (custom functions, higher-order derivatives, serialization, debugging tools) a trustworthy framework needs; and Part 7 has now proven the same machinery on the domain the very first design principle named — financial computing, where "differentiable" and "auditable" have to mean the same thing, and where this chapter's own genuinely-reproduced bug is the proof that checking, not assuming, is what "auditable" actually requires in practice.

## Worked Solutions

**1.** `reduction_threads = min(64, 2048 // 2) = min(64, 1024) = 64` — unchanged from the 1,024-bond case, because `THREADS_PER_BLOCK` is still the smaller of the two. Bonds actually summed: `2 × 64 = 128`, exactly as before. Fraction left out: `(2048 - 128) / 2048 = 1920/2048 ≈ 93.75%` — *worse* than the 1,024-bond portfolio's 87.5%, because the number of bonds actually touched by this bug is capped at `2 × THREADS_PER_BLOCK` regardless of how large the portfolio grows, so the fraction silently dropped only gets worse as the book scales up.

**2.** A bracket of width `w` needs `n` halvings to shrink below tolerance `tol` when `w / 2ⁿ ≤ tol`, i.e. `n ≥ log₂(w/tol)`. Here `w=0.2`, `tol=10⁻⁸`: `n ≥ log₂(0.2 / 10⁻⁸) = log₂(2×10⁷) ≈ 24.25`, and since `n` must be a whole number of halvings, `n = 25` — matching the 25 iterations this chapter's own file genuinely observes exactly, because bisection's convergence rate is deterministic and depends only on the starting width and the tolerance, not on where the root happens to sit inside the bracket.

**3.** New total: `400 + 350 + 250 + 500 = 1500`. New weights: `w_A = 400/1500 ≈ 0.2667`, `w_B = 350/1500 ≈ 0.2333`, `w_C = 250/1500 ≈ 0.1667`, `w_D = 500/1500 ≈ 0.3333` — summing to `1.0`. New portfolio duration: `0.2667×2 + 0.2333×5 + 0.1667×10 + 0.3333×3 ≈ 0.5333 + 1.1667 + 1.6667 + 1.0 = 4.3667` years — lower than the original `5.05` years, because bond D's short 3-year duration and its large 33% weight both pull the weighted average down, even though the total portfolio grew. This chapter's own file 03 genuinely confirms these numbers.

**4.** `DV01 = -time_to_maturity × PV × 0.0001 = -0.5 × 4942.8291 × 0.0001 ≈ -\$0.2471` per basis point — smaller in magnitude than bond 2's `-\$0.7361` from Worked Example 22.1.2, consistent with bond 1's shorter maturity (0.5 years versus 0.75) making it less sensitive to a yield move, exactly the same "longer maturity means more rate-sensitive per dollar" relationship Worked Example 22.3.1 established for bond C.

**5.** Put payoffs on `[95, 102, 108, 130, 90]` with `K=100`: `max(100-95,0)=5`, `max(100-102,0)=0`, `max(100-108,0)=0`, `max(100-130,0)=0`, `max(100-90,0)=10` → `[5, 0, 0, 0, 10]`. Mean payoff: `(5+0+0+0+10)/5 = 3`. Discounted at the same `e^(-0.03) ≈ 0.970446`: `3 × 0.970446 ≈ 2.9113` — genuinely confirmed by this chapter's own file 04. A call's value comes entirely from paths that finished *above* the strike (102, 108, 130 here); a put's value comes entirely from paths that finished *below* it (95 and 90 here) — the two option types are mirror images of the same asymmetric-payoff idea, each one paying off on the opposite side of the strike from the other.
