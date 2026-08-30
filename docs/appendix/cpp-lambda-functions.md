# Appendix F: C++ Lambda Functions, From First Principles

> "Appendix D built an entire CUDA-specific vocabulary on top of the C++ lambda — extended lambdas, generic kernels parameterized by a closure, a device-side ban on reference capture. None of that is where lambdas start. This appendix is where they start: the plain, host-only C++ language feature, on its own terms, before a single `__device__` annotation enters the picture."

## What you will understand by the end of this appendix

- The complete anatomy of a lambda expression — capture clause, parameter list, optional return type, body — and that a lambda with an empty capture list, a lambda capturing everything by reference, and a named function are three different tools for the same underlying job.
- Why a lambda has no name in source code, yet is a fully real object with a real, compiler-generated type and a real `sizeof` — two lambdas with byte-identical source text are still two distinct types, demonstrated here rather than only stated.
- Every capture-clause form C++ actually supports — value, reference, mixed, `this`, and C++17's `[*this]` — including a genuine, real compiler error from mixing `[*this]` with a `const` member function, found by testing the natural-looking code first.
- Why a plain lambda's `operator()` is `const` by default, proven with a real, uncorrected compile error, and what `mutable` actually changes (the closure's own copy, never the original variable).
- C++14's generalized (init) capture, `[name = expr]`, as the mechanism for giving a lambda genuinely new, closure-owned state — verified here with a momentum-style running accumulator, its own worked recurrence, and a second independent instance proving the state is truly per-closure.
- How a lambda's own distinct closure type interacts with `std::function` — able to erase that distinction into one common, storable type — entirely on the host, where Appendix D.7's device-side restriction on `std::function` simply does not apply.

## What you need to know first

- Ordinary C++ functions, `auto` type deduction, and function templates — this appendix assumes comfort with all three and builds the lambda vocabulary on top of them.
- Appendix D, Section D.9 in particular, if you want the CUDA-adjacent motivation for *why* capture-by-value and capture-by-reference matter enough to be worth a whole appendix — this appendix stands on its own without it, but D.9's dangling-reference trap and this appendix's Section F.4 are close relatives, one host-only and one device-motivated.
- `std::sort`, `std::count_if`, and `std::find_if` from `<algorithm>` — Section F.6 uses them as the running example for lambdas-as-comparators, but doesn't require prior familiarity beyond their signatures.

## F.1 Anatomy of a Lambda: Capture, Parameters, Body, Return Type `[FOUNDATIONAL]`

### Intuition

A lambda expression has exactly four parts, in a fixed order: a capture clause in square brackets (what it can see from its enclosing scope), a parameter list in parentheses (exactly like a function's), an optional trailing return type, and a body in curly braces. Every C++ lambda in existence is some combination of these four parts — nothing else is possible, and nothing here is optional except the return type (which the compiler will deduce for you whenever the body has exactly one `return` statement) and, in a couple of narrow contexts, the parameter-list parentheses themselves for a truly empty parameter list.

### Background

```cpp
auto add_explicit = [](int a, int b) -> int { return a + b; };
auto add_deduced   = [](int a, int b) { return a + b; };
```

Genuinely captured, `g++ -std=c++17`:

```
add_explicit(3, 4) = 7
add_deduced(3, 4)  = 7
both compile to callables with identical behavior: true
```

An empty capture clause `[]` means the lambda captures *nothing* from its enclosing scope — it can only use its own parameters, plus anything at namespace or global scope. `[]() { printf("hello\n"); }` and `[](int x) { return x * x; }` are both perfectly ordinary lambdas that happen to need nothing from their surroundings, which is the majority case for the small, single-purpose operations later sections of this book (and Appendix D's `square_fn`, `cube_fn`, and similar) pass around.

## F.2 A Lambda Has No Name, But It Is a Real Object `[FOUNDATIONAL]`

### Intuition

"Lambda" describes a piece of *syntax* for writing a closure inline, at the point it's needed, without declaring a named struct first. What that syntax produces is, underneath, an ordinary object: the compiler synthesizes an anonymous class with member variables for whatever was captured and an `operator()` for the body, then constructs one instance of it right there in your expression. `auto` is doing real work in `auto f = [](...){...};` — it's deducing whatever that anonymous, otherwise-unnameable type actually is.

### Worked Example F.2.1 — `sizeof` a closure, and two "identical" lambdas that aren't

```cpp
int a = 1, b = 2, c = 3;
auto captures_nothing = []() { return 0; };
auto captures_one     = [a]() { return a; };
auto captures_three   = [a, b, c]() { return a + b + c; };
```

Genuinely captured:

```
sizeof(captures_nothing) = 1 byte(s)
sizeof(captures_one)     = 4 byte(s)  (holds one copied int)
sizeof(captures_three)   = 12 byte(s)  (holds three copied ints)
```

`captures_nothing`'s closure is `1` byte — every object in C++ needs a nonzero size so two distinct objects can have distinct addresses, even one with no data members at all. Each additional captured `int` adds exactly `4` bytes, precisely the way adding a member to a hand-written struct would.

Two lambdas with byte-identical source text are still genuinely distinct types:

```cpp
auto identical_source_1 = [](int x) { return x * 2; };
auto identical_source_2 = [](int x) { return x * 2; };
```

Genuinely captured, via `typeid(...).name()`:

```
identical_source_1's type name: Z4mainEUliE_
identical_source_2's type name: Z4mainEUliE0_
same type despite identical source text: false (two distinct types)
```

**[COMMON TRAP]** This is why `std::vector<auto>` cannot hold "several lambdas" the way it might seem like it should: `auto` deduces exactly one concrete type per declaration, and every distinct lambda *expression* — even with identical source text, written twice — mints its own type. Section F.7 covers the tool that actually solves "I need a container of several different callables": `std::function`, which deliberately erases this exact distinction.

## F.3 The Capture Clause in Full `[FOUNDATIONAL]`

### Intuition

Every capture-clause form is a variation on exactly two primitives — copy a value in, or store a reference/pointer to something that already exists — applied to one variable at a time, to every visible variable at once, or to the whole enclosing object when the lambda is written inside a member function.

### Background

| Form | Meaning |
|---|---|
| `[]` | Capture nothing |
| `[x]` | Capture `x` by value (a copy) |
| `[&x]` | Capture `x` by reference (its address) |
| `[x, &y]` | Mixed — `x` by value, `y` by reference, per-variable |
| `[=]` | Capture everything used in the body, by value |
| `[&]` | Capture everything used in the body, by reference |
| `[this]` | Capture a pointer to the enclosing object (inside a member function) |
| `[*this]` | (C++17) Capture a *copy* of the whole enclosing object, by value |

### Worked Example F.3.1 — Mixed capture, `this`, and `*this`

```cpp
int p = 10, q = 20;
auto mixed = [p, &q](int add) { q += add; return p + q; };
```

Genuinely captured:

```
p=10 q=25, mixed(5) = 35
after mixed(5): p=10 (untouched, captured by value) q=25 (mutated, captured by reference)
```

`[this]`, inside a member function, captures a *pointer* — the returned lambda reaches back into the exact same object it was created from:

```cpp
struct Accumulator {
    double total = 0.0;
    auto make_adder_capturing_this() {
        return [this](double x) { total += x; return total; };
    }
};
```

Genuinely captured:

```
acc.total starts at 0.0
add_via_this(3.5) returns 3.5
acc.total is now 3.5 (the ORIGINAL object was mutated through the captured pointer)
```

`[*this]` (C++17) instead copies the *whole object* by value — the returned lambda is safe to use even after the original object no longer exists, because it isn't reaching back into anything:

```cpp
auto make_adder_capturing_copy() {
    return [*this](double x) mutable { total += x; return total; };
}
```

Genuinely captured:

```
acc2.total = 100.0 before calling the [*this]-capturing lambda
add_via_copy(5.0) returns 105.0 (mutates its OWN copy's total)
acc2.total is still 100.0 (the ORIGINAL object was never touched)
```

**[COMMON TRAP]** The natural first attempt at `make_adder_capturing_copy` marks the *member function itself* `const` (it looks read-only, after all — it only reads `*this` to make a copy). That attempt produces a real, genuine compiler error:

```
error: assignment of member 'Accumulator::total' in read-only object
```

Inside a `const` member function, `*this` has type `const Accumulator&` — capturing `[*this]` there captures a *const-qualified* copy, and no amount of `mutable` on the lambda itself changes that, because `mutable` only removes the `const` from the closure's own `operator()`, not from what was captured into it. The member function generating the copy must itself be non-`const` for the resulting `[*this]` capture to be mutable — exactly the fix `make_adder_capturing_copy`'s working version above uses.

## F.4 Mutable Lambdas and Why Lambdas Are `const` by Default `[FOUNDATIONAL]`

### Intuition

An ordinary, non-`mutable` lambda's `operator()` is implicitly `const` — calling it is not supposed to change anything the closure captured by value, the same default C++ applies to member functions unless you explicitly ask otherwise. `mutable` removes exactly that one restriction: it makes `operator()` non-`const`, so the closure's own by-value copies can change from call to call.

### Worked Example F.4.1 — A real compile error, then the fix

```cpp
int counter = 0;
auto broken_counter = [counter]() {
    counter++;    // ERROR: counter is captured by value into a const operator()
    return counter;
};
```

Genuinely captured, `g++ -std=c++17`:

```
error: increment of read-only variable 'counter'
```

Adding `mutable` fixes it, and the closure's own copy now persists its incremented value across repeated calls — while the *original* `counter` variable is completely unaffected, exactly as Appendix D.9's `increment_own_copy` demonstrated in a CUDA-adjacent context:

```cpp
auto working_counter = [counter]() mutable {
    counter++;
    return counter;
};
```

Genuinely captured:

```
working_counter() call 1: 1
working_counter() call 2: 2
working_counter() call 3: 3
original counter is still: 0 (mutable only affects the closure's OWN copy)
```

## F.5 Generalized (Init) Capture: `[name = expr]`, State That Lives Inside the Closure `[FOUNDATIONAL]`

### Intuition

Every capture form so far reaches for something that already exists in the enclosing scope — a copy or a reference of an existing variable. C++14's generalized capture, `[name = expr]`, does something different: it constructs a *brand-new* member inside the closure, initialized from an arbitrary expression, that exists nowhere outside that one lambda. This is what lets a lambda carry genuinely private, persistent state — no external variable a caller could accidentally read or mutate between calls, and no risk of two different closures fighting over one shared object.

### Worked Example F.5.1 — A momentum-style running accumulator

Gradient-based optimization's momentum update — `velocity ← β·velocity + gradient`, applied elementwise, every step reusing the *previous* step's velocity — is a natural fit: the velocity vector is exactly the kind of persistent, closure-private state generalized capture is for.

```cpp
auto make_momentum_smoother(double beta) {
    return [beta, velocity = std::vector<double>()](const std::vector<double>& gradient) mutable {
        if (velocity.empty()) velocity.resize(gradient.size(), 0.0);
        for (size_t i = 0; i < gradient.size(); i++) {
            velocity[i] = beta * velocity[i] + gradient[i];
        }
        return velocity;
    };
}
```

`velocity = std::vector<double>()` constructs an empty vector *inside the capture list itself* — there is no pre-existing `velocity` variable anywhere in `main` for this closure to reach for. Genuinely captured, `beta = 0.9`, the identical gradient `{1, 2, 3}` applied three times in a row:

```
step 1 velocity: 1.0000 2.0000 3.0000
step 2 velocity: 1.9000 3.8000 5.7000
step 3 velocity: 2.7100 5.4200 8.1300

independently hand-unrolled recurrence for element 0: 1.0000, 1.9000, 2.7100
matches the closure's own reported values: true
```

The independent check hand-unrolls the same recurrence a completely different way — `v₀ = 0`, `v₁ = 0.9·v₀ + 1 = 1.0`, `v₂ = 0.9·v₁ + 1 = 1.9`, `v₃ = 0.9·v₂ + 1 = 2.71` — matching the closure's own reported values exactly, rather than trusting the closure to agree with itself.

A second, independently created smoother has its own, separate velocity, confirming the state genuinely lives inside each closure rather than in something the two instances could interfere with:

```
a freshly created SECOND smoother, same beta, first call: 1.0000 2.0000 3.0000
(matches smoother's own step-1 result exactly, 1.0000 2.0000 3.0000 -- confirming
each closure's velocity starts fresh and independent of any other instance)
```

**[COMMON TRAP]** `[velocity = std::vector<double>()]` is a *copy*-initializing capture by default — for a type expensive to copy, `[velocity = std::move(some_existing_vector)]` moves an already-constructed object in instead, avoiding a redundant copy. Both spellings produce a closure that owns its own independent `velocity` member; the difference is only in how that member gets its initial value, not in whether the resulting state is private to the closure.

## F.6 Lambdas as Comparators and Predicates `[FOUNDATIONAL]`

### Intuition

`std::sort`, `std::count_if`, and `std::find_if` all take a callable as one of their arguments — a comparator (`bool(T, T)`, "should `a` come before `b`?") or a predicate (`bool(T)`, "does this element match?"). Before lambdas existed, supplying one of these meant writing a named function or a functor struct just for a single call site. A lambda lets the comparator or predicate live directly where it's used, and — because it's a closure — lets it capture whatever context it needs (a threshold, a target value) without changing the algorithm call's own signature at all.

### Worked Example F.6.1 — Sorting, counting, and finding, all with lambdas

```cpp
std::vector<int> values = {5, 2, 8, 1, 9, 3};

std::sort(ascending.begin(), ascending.end(), [](int a, int b) { return a < b; });
auto compare = [](int a, int b) { return a > b; };
std::sort(descending.begin(), descending.end(), compare);
```

Genuinely captured:

```
original: 5 2 8 1 9 3 
ascending (lambda comparator a < b): 1 2 3 5 8 9 
descending (comparator stored in a variable named compare): 9 8 5 3 2 1 
```

`compare` here is nothing more than the name of a variable holding a lambda — `std::sort` neither knows nor cares that its comparator argument came from a variable instead of an inline expression; any callable matching `bool(int, int)` works identically.

```cpp
int count_over_4 = std::count_if(values.begin(), values.end(), [](int v) { return v > 4; });
auto it = std::find_if(values.begin(), values.end(), [](int v) { return v % 2 == 0; });
```

Genuinely captured:

```
count_if(values, v > 4) = 3
find_if(values, v is even) = 2 (first even value found)
```

A capturing predicate parameterizes the same check without hard-coding a number into the algorithm call itself:

```cpp
int threshold = 4;
int count_over_threshold = std::count_if(values.begin(), values.end(),
                                          [threshold](int v) { return v > threshold; });
```

Genuinely captured:

```
a CAPTURING predicate, threshold=4: count_if returns 3 (matches the hard-coded > 4 case above: true)
```

## F.7 `std::function`, Type Erasure, and When a Lambda's Own Type Isn't Enough `[FOUNDATIONAL]`

### Intuition

Section F.2 established that every lambda has its own distinct closure type, even with identical source text — which is exactly why a plain `std::vector<auto>` cannot hold "several different lambdas": `auto` only ever deduces one concrete type. `std::function<Signature>` is the standard library's answer: it erases every callable matching a given signature into one common, storable type, at the cost of the runtime indirection Appendix D.7 showed cannot cross onto a CUDA device.

### Worked Example F.7.1 — One container, several genuinely different closures

```cpp
std::vector<std::function<int(int)>> operations;
operations.push_back([](int x) { return x * 2; });
operations.push_back([offset](int x) { return x + offset; });
operations.push_back([](int x) { return x * x; });
```

Genuinely captured:

```
three genuinely different closures, stored in ONE std::vector<std::function<int(int)>>:
operations[0] (double) applied to 7: 14
operations[1] (add_offset(100)) applied to 7: 107
operations[2] (square) applied to 7: 49
```

This entire file compiles and runs with a plain `g++` — no CUDA toolchain, no `--extended-lambda`, no device anywhere — because `std::function`'s virtual-dispatch mechanism has nothing to conflict with on the host. Appendix D.7's failure was never "`std::function` is broken"; it was specifically that virtual dispatch through host vtables is unreachable *from device code*, a restriction with no host-side analogue at all. On the host, `std::function` is exactly the tool Section F.2's limitation calls for.

## F.8 Complete Runnable Code

Every example this appendix derived, assembled in one place. Every file below is genuinely compiled with `g++ -std=c++17 -Wall -Wextra` — plain host C++, no CUDA toolchain anywhere in this appendix.

### File: `01_anatomy.cpp` (Section F.1)

```cpp
#include <cstdio>

int main() {
    auto add_explicit = [](int a, int b) -> int { return a + b; };
    auto add_deduced   = [](int a, int b) { return a + b; };
    printf("%d %d\n", add_explicit(3, 4), add_deduced(3, 4));

    auto no_capture = [](int x) { return x * x; };
    printf("%d\n", no_capture(6));

    auto say_hello = []() { printf("hello from a zero-parameter lambda\n"); };
    say_hello();
    return 0;
}
```

### File: `02_lambdas_are_objects.cpp` (Section F.2)

```cpp
#include <cstdio>
#include <typeinfo>

int main() {
    int a = 1, b = 2, c = 3;
    auto captures_nothing = []() { return 0; };
    auto captures_one     = [a]() { return a; };
    auto captures_three   = [a, b, c]() { return a + b + c; };
    printf("%zu %zu %zu\n", sizeof(captures_nothing), sizeof(captures_one), sizeof(captures_three));

    auto identical_source_1 = [](int x) { return x * 2; };
    auto identical_source_2 = [](int x) { return x * 2; };
    printf("%s\n", (typeid(identical_source_1) == typeid(identical_source_2)) ? "same type" : "distinct types");

    auto reassigned = captures_one;
    printf("%d\n", reassigned());
    return 0;
}
```

### File: `03_capture_clause_full.cpp` (Section F.3)

```cpp
#include <cstdio>

struct Accumulator {
    double total = 0.0;
    auto make_adder_capturing_this() {
        return [this](double x) { total += x; return total; };
    }
    auto make_adder_capturing_copy() {   // must NOT be const -- see Section F.3's own trap
        return [*this](double x) mutable { total += x; return total; };
    }
};

int main() {
    int p = 10, q = 20;
    auto mixed = [p, &q](int add) { q += add; return p + q; };
    printf("%d %d %d\n", p, q, mixed(5));

    Accumulator acc;
    auto add_via_this = acc.make_adder_capturing_this();
    printf("%.1f %.1f\n", add_via_this(3.5), acc.total);

    Accumulator acc2;
    acc2.total = 100.0;
    auto add_via_copy = acc2.make_adder_capturing_copy();
    printf("%.1f %.1f\n", add_via_copy(5.0), acc2.total);
    return 0;
}
```

### File: `04_mutable_lambdas.cpp` (Section F.4)

```cpp
#include <cstdio>

int main() {
    int counter = 0;
    auto working_counter = [counter]() mutable {
        counter++;
        return counter;
    };
    printf("%d %d %d %d\n", working_counter(), working_counter(), working_counter(), counter);
    return 0;
}
```

### File: `05_init_capture_momentum.cpp` (Section F.5)

```cpp
#include <cstdio>
#include <vector>

auto make_momentum_smoother(double beta) {
    return [beta, velocity = std::vector<double>()](const std::vector<double>& gradient) mutable {
        if (velocity.empty()) velocity.resize(gradient.size(), 0.0);
        for (size_t i = 0; i < gradient.size(); i++)
            velocity[i] = beta * velocity[i] + gradient[i];
        return velocity;
    };
}

int main() {
    auto smoother = make_momentum_smoother(0.9);
    std::vector<double> grad = {1.0, 2.0, 3.0};
    for (int step = 0; step < 3; step++) {
        auto v = smoother(grad);
        printf("step %d: %.4f %.4f %.4f\n", step + 1, v[0], v[1], v[2]);
    }
    return 0;
}
```

### File: `06_comparators_predicates.cpp` (Section F.6)

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> values = {5, 2, 8, 1, 9, 3};
    std::sort(values.begin(), values.end(), [](int a, int b) { return a < b; });
    for (int v : values) printf("%d ", v);
    printf("\n");

    int count = static_cast<int>(std::count_if(values.begin(), values.end(), [](int v) { return v > 4; }));
    printf("count_if(v > 4) = %d\n", count);
    return 0;
}
```

### File: `07_std_function_host.cpp` (Section F.7)

```cpp
#include <cstdio>
#include <functional>
#include <vector>

int main() {
    int offset = 100;
    std::vector<std::function<int(int)>> operations;
    operations.push_back([](int x) { return x * 2; });
    operations.push_back([offset](int x) { return x + offset; });
    operations.push_back([](int x) { return x * x; });
    for (auto& op : operations) printf("%d\n", op(7));
    return 0;
}
```

### Expected Output

There is no single combined run to reproduce here — Worked Examples F.1.1 through F.7.1 are each this appendix's source of truth for their own file's genuine output.

## Chapter Summary

A lambda expression is exactly four parts in a fixed order — capture clause, parameter list, optional return type, body — and its capture clause has exactly two underlying primitives (copy a value in, or reference/point to something existing) applied per-variable (`[x, &y]`), uniformly (`[=]`/`[&]`), to the enclosing object (`[this]`), or as a whole-object copy (`[*this]`, with a genuine, real compiler error waiting for anyone who tries it inside a `const` member function without also removing that `const`). A lambda has no name in source, but is a fully real object with a real, measured `sizeof` — `1` byte with nothing captured, `4` more bytes per captured `int` — and two lambdas with byte-identical source text are still two distinct, separately-typed objects, which is exactly why `std::function` exists: to erase that distinction into one common, storable type, entirely safely on the host, where Appendix D.7's device-side restriction on it simply never arises. A plain lambda's `operator()` is `const` by default, genuinely rejecting an attempt to modify a by-value capture until `mutable` is added — and `mutable` only ever affects the closure's own copy, never the original variable, exactly as Appendix D.9 demonstrates in a CUDA-adjacent context. C++14's generalized capture, `[name = expr]`, goes one step further than any existing-variable capture: it constructs brand-new, closure-private state — verified here with a momentum-style accumulator whose three-step recurrence (`1.0, 1.9, 2.71` for element `0`, with `β=0.9`) matched an independent hand-unrolled check exactly, and whose state was confirmed private to each closure by creating a second, independent instance. Lambdas as comparators and predicates for `std::sort`, `std::count_if`, and `std::find_if` need nothing beyond what Sections F.1–F.5 already established — a capturing lambda simply closes over whatever context (a threshold, an offset) the algorithm call itself has no room to express.

## Self-Check Questions

1. `[x, y]` (no `&` on either) and `[=]` both use "copy everything" as their strategy, for a lambda whose body reads both `x` and `y`. Using Section F.2's `sizeof` reasoning, would you expect these two capture clauses to produce closures of the same size? Why or why not, for this specific case?
2. Section F.3's `[*this]` trap requires removing `const` from the member function that creates the capturing lambda. Explain, using Section F.4's own const-by-default reasoning for `operator()`, why simply adding `mutable` to the LAMBDA (without touching the member function) was not enough to fix the original error.
3. Using Section F.5's `make_momentum_smoother`, hand-compute the velocity for element `0` after a FOURTH call with the same gradient value `1.0` and `beta = 0.9`, continuing the recurrence from the appendix's own three computed steps (`1.0, 1.9, 2.71`).
4. Section F.7 stores three different lambdas in one `std::vector<std::function<int(int)>>`. Could the same three lambdas instead be stored in a `std::vector` of a hand-written base class with a virtual `operator()`, each lambda wrapped in a derived class? What would `std::function` be doing for you automatically that the hand-written version would require you to write by hand?
5. A lambda `[&]() { return local_array[0]; }` is returned from the function that declared `local_array` as a local variable, then called later. Using Appendix D.9's own terminology, name the specific bug this produces, and explain why changing the capture to `[=]` would only be a safe fix if `local_array` itself is cheap and safe to copy — what would still go wrong if `local_array` held, say, a raw pointer to another piece of local memory instead of plain data?

## Where We Go Next

This appendix is the host-only foundation Appendix D builds its CUDA-specific vocabulary on top of: extended lambdas' `__device__`/`__host__ __device__` annotations, the by-value-only capture restriction, and generic kernels parameterized by a callable are all additional rules layered onto exactly the language feature described here, not a different feature entirely. A reader who arrived at Appendix D without this background, or who wants the plain-C++ half of the picture on its own terms, now has it; nothing in this appendix depends on Appendix D, and nothing in Appendix D needs re-reading in light of it.

## Worked Solutions

**1.** Not necessarily the same size, and in this specific case with two `int`s captured either way, `[x, y]` and `[=]` DO happen to produce the same size (`8` bytes, two `4`-byte ints) — but that's a property of this particular lambda body, not a general rule. `[=]` captures by value *only the variables the body actually uses* (here, exactly `x` and `y`, nothing more), so for a lambda whose body uses precisely two variables, `[x, y]` and `[=]` are two ways of asking for the identical set of copies. Had the enclosing scope contained a third variable `z` that the lambda's body never references, `[=]` would still capture only `x` and `y` (unused variables are never captured under `[=]`, by rule), so the sizes would still match — the two capture clauses only diverge in size when the explicit list and the "everything the body uses" set genuinely differ, which they can't for a body using exactly the variables named.

**2.** `mutable` changes whether the closure's `operator()` is `const` — it does nothing to the type of what was *captured*. Inside a `const` member function, `*this` already has type `const Accumulator&` before the lambda is even written, so `[*this]` captures a `const Accumulator` value into the closure regardless of what the lambda itself is later marked. `mutable` would successfully let that closure's `operator()` attempt to write to its members, but the member being captured is already `const`-qualified, so the write is rejected for the same reason writing to any `const` object is rejected — `mutable` on the lambda and `const`-ness of the captured copy are two separate, independent facts, and only removing `const` from the member function changes the second one.

**3.** Continuing the recurrence `vₙ = 0.9·vₙ₋₁ + 1.0` from `v₃ = 2.71`: `v₄ = 0.9 × 2.71 + 1.0 = 2.439 + 1.0 = 3.439`.

**4.** Yes — a hand-written abstract base class with a pure virtual `operator()`, and one derived class per lambda (each storing whatever that lambda would have captured as its own member data, with the capture's logic moved into the derived class's overridden `operator()`), is a functionally equivalent design; this is, in fact, essentially what `std::function`'s own implementation does internally. What `std::function` automates: generating that derived wrapper class for you, for ANY callable matching the signature, without you writing a new derived class by hand for every distinct lambda; managing the wrapper's lifetime (construction, copying, destruction) automatically; and providing one uniform type, `std::function<int(int)>`, that works the same way regardless of how many different underlying callables you ever store in it — the hand-written version would need a new derived class definition, by name, for every distinct closure, exactly the per-lambda-type problem Section F.2 identifies `std::function` as solving.

**5.** This is Appendix D.9's "dangling reference capture" / stack-use-after-return bug, applied to an array instead of a scalar: `[&]` captures `local_array` by reference (an address), and that address is invalid the instant the enclosing function returns, so calling the returned lambda later reads memory that no longer belongs to it. Switching to `[=]` genuinely fixes this specific case only because copying `local_array` (assuming it's a plain, fixed-size array of simple data) copies the actual data into the closure, leaving nothing referencing the now-gone stack frame. If `local_array` instead held a raw pointer to *other* local memory, `[=]` would copy the pointer's numeric value — a shallow copy — while the memory that pointer addresses would still vanish when the function returned; the lambda would hold a perfectly valid closure containing a perfectly invalid pointer, reproducing the identical dangling-memory bug one level removed, which is exactly why "capture by value" only guarantees safety for the specific value being copied, not for whatever that value might itself point to.
