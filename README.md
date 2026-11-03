# LoopNet

A tiny language with persistent float globals, compiled directly into a recurrent
residual **ReGLU** network. No training and no normalization. The compiler produces
weights; the network evaluator does not interpret source instructions.

Feed the network output back as its next input. Primitive `.loop` functions run
once per forward pass, preserving globals between invocations. Structured `.flow`
programs preserve their control position as well, advancing until a persistent
**`terminate` neuron** signals that the whole program has finished.

## Run it

Python 3.10+; the compiler and sparse network runtime have **no dependencies**.

```sh
python3 -m loopnet run examples/counter.loop --ticks 12 --compare
python3 -m loopnet run examples/logistic.loop --set x=0.25 --ticks 20 --compare
python3 -m loopnet run examples/countdown.loop --max-steps 14 --ticks 3 --compare
python3 -m loopnet run examples/arrays.loop --ticks 12 --compare
python3 -m loopnet run examples/functions.flow --max-steps 8 --compare
python3 -m loopnet run examples/array_loop.flow --max-steps 8 --compare
python3 -m loopnet run examples/unrolled.flow --max-steps 8 --compare
python3 -m loopnet run examples/recursive.flow --stack-depth 8 --max-steps 8 --compare
```

`--compare` checks each invocation against a separate reference interpreter.

Play the local two-player Connect Four GUI:

```sh
python3 -m loopnet.connect4 --open
```

Open `http://127.0.0.1:8765` if it does not open automatically. Click a column or
press `1`–`7`; `U` undoes a move. The rules in `examples/connect4.flow` compile to
float64 neural weights evaluated in a browser worker. Each move runs the recurrence
until `terminate`; the board and game globals persist into the next move.

Play the **64×48 neural first-person shooter**:

```sh
python3 -m loopnet.doom --open
```

Open `http://127.0.0.1:8766`. `W/S` move, `A/D` strafe, arrow keys or `Q/E`
turn, and Space fires. The game starts automatically; `R` or **reset** starts a
fresh room. The on-screen controls also support pointer and touch input. Clear
the three red targets; pillars block both movement and shots. This is a small
Doom-inspired gallery shooter with original geometry and
graphics, rather than a port of the original engine.

The entire simulation and all 3072 palette-indexed pixels come from
`examples/doom.flow`, compiled into the same float64 ReGLU network. The browser
only passes keys, maintains the frame cadence, and blits the returned palette
coordinates with interpolated colors. The bottom eight rows are a neural status
bar: its 3×5 pixel font draws spaced ammunition and enemy counters. When the room
is clear, the counters disappear and a wider, centered **WIN** replaces them.
These are framebuffer pixels, generated in `.flow` along with the world and
weapon. One fresh function invocation produces a complete frame; persistent
player, camera, ammunition, and target globals carry into the next invocation.
The renderer takes 32 quadratically spaced samples per ray, with finer spacing
near the camera, then bisects each hit interval five times. It projects the
refined wall and target distances into pixels, reducing close-up stair steps;
walls also remain visible behind the shorter target silhouettes.
The model uses 3144 recurrent channels, 399 full feedforward stages, and one
execution region per rendered frame (plus an unreachable fallthrough region).
A warm sparse JavaScript benchmark in Node here takes about 9 ms per frame;
the UI schedules 20 simulation frames per second. On startup, a JavaScript
compiler in the browser worker constructs the same float64 weights locally.
This takes about two seconds in Node here; browser/device timings vary. The
static site is about 8.2 MB, compared with 108 MB for the full weight JSON.
It ships a compressed source instruction stream and symbol/constant tables,
not a compressed copy of the weights. The browser performs common-expression
elimination, builds the activation DAG, removes dead nodes, schedules layers,
and generates typed sparse ReGLU matrices. No source opcodes execute during
game frames, and all generated weights exactly match the Python compiler.

To deploy Doom on **GitHub Pages**, generate the static folder:

```sh
python3 -m loopnet.doom --export docs
```

Commit `docs/` and configure your repository's **Settings → Pages → Deploy from
a branch** to use your branch and `/docs`. See the
[GitHub Pages publishing instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
The export includes `.nojekyll` and works under a repository subpath. It needs
no backend, CDN, Python runtime, or downloaded neural weights. Modern browsers
use their built-in gzip decompressor; synthesis runs once per page load in a
worker so the interface remains responsive. Run the export again after changing
the source, compiler, or UI. Exporting lowers the source but does not build the NN.

Preview the exported files with any static HTTP server:

```sh
python3 -m http.server 8000 --directory docs
# Open http://localhost:8000
```

Across the top of the screen, the **neurons** activation strip displays the
input and every neural stage as one column each, in execution order. Each column
shows the first 60 neurons top-to-bottom; larger layers are truncated in the
display. All layers still compute their complete outputs; clipping affects only
the visualization. The native image is 801×62 across 400 columns. Hover to
identify a stage and exact float64 neuron value. Cyan is positive, red is
negative, and black is zero. Each layer has its own logarithmic color range,
normalized by its largest displayed absolute activation in the current frame.
Small-magnitude layers remain visible alongside large-magnitude layers; clipped
neurons and values from previous frames do not dim the view. This affects only
visualization colors, and never changes the actual network activations.
The atlas records the same forward pass that renders the displayed game frame.
It reuses float64 recording buffers and returns image buffers to the worker for
reuse. The UI schedules 20 simulation frames per second. The strip stays at the
top, with its label underneath and a gap before Doom. The game scales to fit
the viewport, including on narrow screens, and page scrolling is disabled.

The page has a black background, the game on the left, and a description on the
right with source and author links. Its **4,730,103 parameters** count the stored
sparse weights and biases, including zero biases; its **399 layers** count the
stages in one network pass. Both counts come from the generated network. The
description moves below the game on narrow screens. The other elements are the
live neuron strip, movement and fire controls, **reset**, and **twitter mode**. The source link in
the description opens the actual `.flow` code. There is no start screen or pause control.

**Twitter mode** hides all page text and controls and maximizes the game beside
a clockwise-rotated neuron strip. The strip uses a 1:2.5 width-to-height ratio
and shows up to 100 neurons from every third layer; the normal view shows up to
60 neurons from every layer. Keyboard
controls and clicking the game to fire keep working. Press **Escape** to return
to the normal layout. Switching views keeps the current game state and uses the
same actual recorded activations; it does not compile or run a separate network.

One rendered frame uses one recurrence with 399 network stages. Intermediate
computations live in the feedforward branch; framebuffer and global values are
committed at residual writeback. The browser feeds persistent game values and
current keys into the next invocation. The live neuron strip reuses recording
buffers from that forward pass.

Compile it separately with its larger expansion budget:

```sh
python3 -m loopnet compile examples/doom.flow --unroll-limit 80000 -o /tmp/doom.json
```

Export a network, then run it without the source:

```sh
python3 -m loopnet compile examples/counter.loop -o counter.net.json
python3 -m loopnet run-net counter.net.json --ticks 12
python3 -m loopnet info examples/counter.loop
```

## Structured functions, while loops, and termination

Write Python-like structured code in a `.flow` file:

```python
def pair(x, y):
    return (x + y, x * y)

remaining = 4
total = 0
while remaining > 0:
    (outa, outb) = pair(remaining, 2)
    total = total + outa + outb
    remaining = remaining - 1
```

The final `total` is `38`. There is no training or host-side execution of these
functions: the compiler reduces them to the primitive language, then generates
the same residual ReGLU weights as before. `.flow` files are detected automatically;
pass `--structured` for structured source with another extension.

Inspect the reduction, compile it, or run it:

```sh
python3 -m loopnet lower examples/functions.flow -o functions.loop
python3 -m loopnet run functions.loop --max-steps 8 --compare
python3 -m loopnet compile examples/functions.flow --max-steps 8 -o functions.net.json
python3 -m loopnet run-net functions.net.json
```

The reduction starts with a `continuation` directive. Its program counter persists
between network calls. A `while` loop becomes labels, conditional jumps, and a
backward jump. An ordinary `def` emits one shared function body. Calls push a
return instruction address and use a new activation frame; returns pop that frame
and jump indirectly to the saved address. Arguments, locals, intermediate
expression values, and multiple outputs survive nested calls and recursion.
At top-level completion, a primitive
`halt` instruction sets a dedicated residual-stream channel, `terminate`, from
`0.0` to `1.0`.

Use the **`inline` keyword** when you want expansion at the call site instead:

```python
inline def pair(x, y):
    return x + y, x * y

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
a, b = pair(result, 2)
```

`factorial` uses runtime calls and supports recursion. `pair` is expanded into its
caller and consumes no call-stack frame. Inline functions can call ordinary
functions, and ordinary functions can call inline helpers. A cycle made entirely
of inline expansions is rejected; a cycle through an ordinary function is a
normal bounded recursive call.

The network has a fixed memory size, so its **runtime stack is bounded**, by
default to 16 simultaneously active ordinary calls. Set `--stack-depth` when
compiling/lowering structured source, or pass `max_stack_depth` to `lower` or
`compile_structured`. This sizes activation storage, including expression
temporaries; it does not duplicate function bodies. The root program uses depth
zero. Overflow, underflow, or fractional stack pointers produce an in-network
diagnostic checked by the runner. Increasing the stack depth increases memory and
neural work. A lowered `.loop` file already encodes its chosen stack depth.

`break` exits the innermost `while` and skips that loop's `else` block.
`continue` returns to its condition. Both work in ordinary and inline functions;
they do not affect a caller's loop.

### Automatically unrolled for loops

`for` loops over a fixed `range` expand at compile time:

```python
array squares[8]
for i in range(8):
    squares[i] = i * i

for row in range(2):
    for column in range(4):
        squares[row * 4 + column] += row + column
```

Use `range(stop)`, `range(start, stop)`, or `range(start, stop, step)`. Bounds and
steps must be compile-time integers, with magnitude at most `2**53`; the step
cannot be zero. Arithmetic on literals and enclosing unrolled loop indices is
allowed. Negative steps, empty ranges, nesting, `break`, `continue`, and `for ...
else` work. An empty range leaves the induction variable unchanged; otherwise it
retains its last assigned value. Range expansion has a total limit of 4096
iterations per compilation by default, including nested expansion and inline
copies. Set `--unroll-limit` or `max_unrolled_iterations` on `lower` /
`compile_structured` for deliberately larger renderers.

The compiler substitutes immutable induction variables, folds constant arithmetic
and branches, and connects constant array indices directly to their cells. The
first loop above needs no dynamic indexing selectors, counter updates, comparisons,
or backward jumps. If the body reassigns the induction variable, or a function
could modify it through a global, its reads remain runtime reads to preserve
ordinary loop semantics. Functions, `while` loops, and `break` inside an unrolled
loop keep their normal control flow. Runtime-sized loops still use `while`.

`--max-steps` controls **execution regions per network call**, defaulting to one
for continuation programs. Structured programs now use basic blocks by default:
a straight-line stretch runs through its terminating branch, jump, or return in
one slot. Independent operations in that stretch share hidden layers; dependent
operations occupy successive layers. Branches, calls, and loop backedges retain
their normal semantics. The final source-level values are unchanged, while each
recurrent call can advance more primitive instructions.

Use `--execution instruction` (or `compile_structured(..., execution="instruction")`)
for exact primitive-by-primitive debugging. Primitive `.loop` compilation defaults
to this mode. `--execution block` also works on primitive source; arbitrary
computed jumps conservatively keep instruction-sized regions. The structured
frontend knows the possible runtime return addresses and can safely form larger
regions around dynamic calls.

The execution budget determines network depth, not a limit on the number of
iterations of a while loop. The CLI feeds back the full state until
`terminate` becomes one, then stops scheduling network calls. `--ticks` sets a
host budget (default `10000` for continuations); reaching it before termination is
an error. The terminal state is a fixed point, so extra forward passes preserve it.
CLI records expose `"terminate": 0.0` or `1.0` separately from program globals.

The Python API can run to completion or expose the recurrence explicitly:

```python
from pathlib import Path
from loopnet import compile_structured, lower

source = Path("examples/functions.flow").read_text()
net = compile_structured(source, max_steps=8, branch_margin=1e-4, max_stack_depth=16)
assert net.run(max_ticks=1000)["total"] == 38.0
print(lower(source))

state = net.initial_state()
while not net.terminated(state):
    state = net(state)
    net.assert_valid(state)
assert state[net.terminate_channel] == 1.0
print(net.globals(state))
```

Always feed back the **full state vector** when stepping a continuation program.
A globals-only dictionary omits its control position. `net.step(...)` is reserved
for primitive function mode; use `net.run(...)` or the explicit loop above for
continuations. To inspect generated variable slots, call
`net.globals(state, include_internal=True)`; normal output hides the reserved
`__ln_` names. `Interpreter(parse(lower(source)))` is the independent primitive
reference machine; its `step(max_steps=...)` method also preserves control.
For a basic-block trace, use `loopnet.regions.reference_step(reference,
net.regions, net.max_steps)`. The CLI's `--compare` handles either granularity.
Transient compiler values do not appear in the residual state or globals dump.

The structured syntax supports:

- Positional float parameters, zero/one/multiple outputs, nested calls, and early
  returns. Functions with outputs must explicitly return the same number of
  values on every path. Ordinary `def` uses runtime call/return control and supports
  recursion and mutual recursion. `inline def` opts into static expansion.
- Parameters and assigned scalar names are local to each call and become global
  float cells selected by the frame pointer. Locals start at zero on each call,
  like primitive globals. Compiler temporaries also belong to their activation.
  Other names read top-level globals; `global name` enables scalar global writes.
  Arrays are always global and are accessed directly, rather than passed as values.
- `while`, automatically unrolled constant `for ... in range(...)`, nested loops,
  `if`/`else`/`elif`, `break`, `continue`, loop `else`,
  `pass`, and bare top-level `return` for early termination.
- Float expressions using `+`, `-`, `*`, `relu(x)`, parentheses, and unary signs; tuple
  assignment; and `+=`, `-=`, `*=`. Reads and argument values preserve evaluation
  order, including calls that change globals.
- Conditions with `<`, `>`, `<=`, `>=`, `==`, `!=`, short-circuit `and`/`or`/`not`,
  and `True`/`False` in conditions. Comparison differences must respect the
  primitive branch gap. A bare float condition means **positive**, consistent with
  the primitive language; a negative float takes the false path.
- Static arrays declared with either `array a[16]` or `a = array(16)` at top level.
  Index expressions such as `a[i + 1]` and `a[b[i]]` are reduced to scalar
  temporaries, then checked by the existing index circuits.

This is a restricted parsed syntax, not general Python execution. Imports,
objects, strings, division, runtime-sized `for` loops, keyword/default arguments, decorators,
and first-class functions/arrays are not supported. All runtime data remains
float64, with the same `1e-4` branch contract. Function definitions alone do not
invoke anything; execution starts with the top-level statements.

## Language

A file is the body of exactly one function. Its five assignment forms are:

```text
x = 2.5
x = y + -1
x = y + z
x = y * 0.5
x = y * z
```

An additional arithmetic intrinsic, `x = relu(y)`, stores `max(0, y)` exactly.
It also accepts array operands and destinations; structured expressions can use
`relu(...)` anywhere a float expression is allowed. This is a continuous
arithmetic operation and needs no branch gap. The shader's `clip` and geometry
helpers are composed from ReLU, addition, and multiplication.

All values are floats. All variables are global memory addresses, initially zero
unless the caller supplies a value. Addresses may be names (`x`, `state`), numbered
slots (`[0]`, `[12]`), or cells in a declared array (`a[0]`, `a[i]`). There are no local
variables or additional runtime data types. Constants may be negative or use
scientific notation. A direct copy, `x = y`, is shorthand for `x = y + 0`.

Control flow:

```text
count = count + 1
remaining = count + -5
if remaining > 0 goto reset
return

reset:
count = 0
cycles = cycles + 1
return
```

- `if x > 0 goto label` branches only when `x` is positive. `if x goto label` is shorthand.
- `goto label` jumps unconditionally. Forward and backward jumps are supported.
- `return` ends the current invocation. Falling off the file also returns.
- `halt` ends the entire recurrent program and latches its `terminate` output.
  An ordinary function-mode `return` leaves `terminate` at zero.
- `ip = &label` loads a label's numeric instruction address. `goto *ip` jumps to
  an integer instruction address stored in a float. Invalid targets are diagnosed
  in the network. The structured frontend uses this for dynamic returns.
- `stack sp 16` declares a checked stack-pointer range, `0` through `16` inclusive.
  It is allocation/checking metadata and executes no instruction.
- A label occupies its own line and ends in `:`. Labels are compile-time metadata.
- `#` starts a comment.

Assignment executes when reached; putting `x = 0` at the beginning resets `x` on
every call. Supply initial values externally when they should persist instead.
Subtraction is expressed as addition of a negative constant or multiplication by
`-1` followed by addition.

## Fixed-size arrays

```text
array a[16]
array b[8]

a[i] = x + 1
x = a[j] * y
b[k] = a[i]
if a[j] > 0 goto positive
return
positive:
a[0] = 3
```

`array a[16]` reserves sixteen global float cells, `a[0]` through `a[15]`. The size
is a positive integer literal known at compile time. Declarations are allocation
metadata: they execute no instructions, may appear anywhere in the file, and
never reset existing contents. Array cells persist between calls just like scalars.

Array references are accepted everywhere a scalar address is accepted: assignment
destinations, both arithmetic operands, copies, and branch conditions. The index
must be either a constant or one scalar global address. Named and numbered global
addresses both work, including `a[[0]]`. Index expressions and nested indexing
are not supported; compute the index in a scalar first.

Indices remain floats but **must hold exact integers in `[0, length)`**. There is
no rounding, interpolation, wrapping, or negative indexing. Invalid constant
indices fail when parsing; invalid dynamic indices are diagnosed only when their
instruction executes. The interpreter raises `ArrayIndexError`; the network's
`step`, `assert_valid`, and CLI report `NetworkContractError`. No extra branch gap
is needed for integer indices. Branching on an array cell's *value* still uses the
usual `1e-4` positive-value gap.

All operands and the destination index use memory from before the assignment.
Thus `a[i] = a[i] + 1` works, and `i = a[i]` reads using the old `i`.

The host API and JSON output expose array cells as flat global names:

```python
net = compile_program("array a[3]\na[i] = a[j] * 2", max_steps=2)
values = net.step({"i": 1, "j": 2, "a[2]": -3})
assert values["a[1]"] == -6.0
```

Quote array names when setting cells from the shell:

```sh
python3 -m loopnet run examples/arrays.loop --set 'bins[0]=5' --ticks 8 --compare
```

Constant indexing compiles directly to a cell address. A dynamic access generates
one selector per cell:

```text
mask[k] = ReLU(1 - ReLU(i-k) - ReLU(k-i))
read = sum(mask[k] * a[k])
```

Valid integer indices produce exactly one selected cell. Stores use the masks to
clear the selected cell and write its saved result in separate residual blocks.
Additional ReLU circuits detect bounds violations and fractional indices. All
selection and index diagnostics execute as neural weights, with **O(array length)**
work and workspace per distinct dynamic access in an instruction. Constant indices
do not need these selectors. Small fixed-size arrays are the intended use.

## Python API

```python
from pathlib import Path
from loopnet import compile_program

net = compile_program(
    Path("examples/counter.loop").read_text(),
    branch_margin=1e-4,
    max_steps=64,
)

state = net.initial_state({"count": 0})
for _ in range(12):
    state = net(state)             # the same residual network, called repeatedly
    net.assert_valid(state)       # checks diagnostics; does not execute source code
    print(net.globals(state))

net.save("counter.net.json")
```

`net.step({"count": 2})` is a convenience adapter for primitive function mode that
returns updated globals. The full `state` vector contains persistent global cells,
values live across execution-region boundaries, one scalar instruction pointer,
and six completion/termination/diagnostic channels. Transient neural activations
belong to the feedforward branch and are never returned as recurrent state.
Diagnostics reset inside the network; continuation control and `terminate`
persist. These channels do not introduce additional data types.

## The generated network

The network uses ReGLU feedforward branches and residual writeback stages, with
sparse, compiler-generated weights. A ReGLU stage has the form:

```text
h_out = W_out (ReLU(W_gate h_in + b_gate) * (W_value h_in + b_value)) + b_out
z_cleared = z + clear_projection(z, h)
z_next = z_cleared + write_projection(z, h)
```

The `*` inside this block is elementwise multiplication, as in
[ReGLU](https://arxiv.org/abs/2002.05202). Signed multiplication is represented by
two channels:

```text
x*y = ReLU(x)*y - ReLU(-x)*y
```

No multiplication approximation or bounded variable range is needed. The compiler
stores the instruction pointer as one float, computes one-hot selection masks in
hidden layers, and emits the scalar address of the next region. Arithmetic,
branch selection, writes, and control changes
all happen through ReGLU weights. There is no runtime source-level `if`, opcode
dispatch, or data-dependent layer scheduling.

Each execution slot reads the old residual state into a feedforward branch. The
compiler creates fresh versions of assigned values, shares common expressions,
removes unused computations, and schedules independent operations into the same
hidden layer. Hidden activation storage can be wide without widening the recurrent
state. A liveness pass retains only compiler values that cross a region boundary;
temporary frame banks are removed when every read follows a write in the same
region and frame. Values live across calls remain banked, preserving recursion.

After the branch finishes, two residual stages clear the active destinations
and write their computed values. Both projections read the saved branch results;
no hidden activation is fed back on the next recurrence. Using
separate clear and write blocks avoids the precision loss of implementing an
overwrite as `x + (new_value - x)` when `x` is much larger than `new_value`.

The compiler unrolls `max_steps` execution slots into a fixed feedforward
schedule. The exported JSON shares block weights across these slots to save space;
this is still an explicit, fixed-depth forward pass. In function mode, a return
preserves globals for the remaining slots. In continuation mode, execution resumes
from its saved control position on the next call, and completion sets `terminate`.
Only the caller performs recurrence. Weight exports retain the execution mode and
termination-channel index, stack bound, and stack/jump diagnostics. Earlier
v1/v2/v3/v4 exports still load in Python; the browser supports v4 and v5. The v5
format describes feedforward layers, residual projections, and static execution
boundaries. It contains no source operations or interpreter instructions.

`net.summary()` reports recurrent `state_width`, ephemeral `transient_channels`,
the widest ReGLU layer (`peak_hidden_width`), residual writeback count, and the full
`feedforward_layers` depth including the branch. Residual block count alone is
not a computational-depth measurement when a branch contains multiple stages.

Connect Four now uses **126 recurrent channels**, including 109 user global
cells, ten compiler values live across regions, and seven control/diagnostic
channels. Its branch has 5579 transient channels and a peak ReGLU width of 1224;
full depth is **17 ReGLU stages per call** with `max_steps=1`. The prior compiler
used 5879 recurrent channels and 11 shallow stages. A warm, five-sample median
for a seven-move game in the CPU JavaScript runtime measured about 25 ms, versus
402 ms for that wide version and 175 ms for the original 926-stage version.
These timings are machine- and workload-dependent. Reduced layer count by itself
does not guarantee a wall-clock speedup.

Reproduce the JavaScript benchmark on exported models (including older v4 models):

```sh
python3 -m loopnet compile examples/connect4.flow -o /tmp/connect4.json
node benchmarks/connect4.mjs /tmp/connect4.json
```

## Precision and execution contract

- **Float64 by default.** Python uses double precision, and PyTorch export defaults
  to `torch.float64`. There is no quantization. Arithmetic is mathematically exact
  within this construction, subject to ordinary floating-point rounding and
  overflow; different execution kernels need not give bit-identical results.
- **Branch gap: `1e-4`.** An executed branch condition must be `x <= 0` or
  `x >= 1e-4`. Zero takes the false path. Negative values arbitrarily close to zero
  also work. Conditions strictly between zero and the declared gap are outside the
  contract and produce a diagnostic checked by `assert_valid` / `step` / the CLI.
  Arithmetic values that are not used for branching have no gap requirement.
- **Continuous branch circuit.** A clipped linear ramp is exactly zero for
  nonpositive inputs and exactly one above an internal threshold no greater than
  the declared margin. The internal threshold is a power of two so both control
  endpoints are exactly representable in binary floating point. A separate ReGLU
  diagnostic checks the caller's actual declared margin.
- **Primitive function budget: 64 instructions per invocation by default.** This includes executed
  `return` instructions. Set `--max-steps` / `max_steps` high enough for backward
  jumps or long functions. Failure to return is reported by a floating-point halt
  output; the caller checks it after the fixed forward pass. No finite network can
  finish arbitrary, unbounded loops in one call. Continuation programs instead
  preserve control between calls, with one basic block per call by default for
  structured programs (one instruction for primitive programs), and an independent
  host recurrence budget.
- Inputs and intermediate results must remain finite. As with ordinary float
  programs, extreme values can overflow and rounding can move a condition across
  a boundary. Recurrence can amplify numerical differences.

The reference interpreter evaluates strict `> 0` without a gap. It is useful for
checking valid programs, while the neural runtime enforces the additional branch
and execution-mode contract.

## Optional PyTorch module

Install `torch` in your Python environment, then:

```python
import torch

model = net.to_torch()  # actual nn.Module; float64, no normalization
state = torch.tensor(net.initial_state(), dtype=torch.float64)
with torch.no_grad():
    for _ in range(12):
        state = model(state)
        net.assert_valid(state.tolist())
        print(net.globals(state.tolist()))
```

The module accepts `[state_width]` or `[batch, state_width]` tensors. Its weights
are ordinary trainable parameters and its forward pass is differentiable wherever
ReLU is. Each transient layer exports dense matrices over its connected input and
output ports; temporary activations remain internal tensors. The dependency-free
evaluator uses sparse weights. Dense size and work grow with program size and the
chosen execution budget. Changing weights through training can invalidate the
compiled program's semantics.

For structured programs, PyTorch uses the same persistent control and termination
neuron. `model.terminate_channel` exposes its index:

```python
net = compile_structured(source, max_steps=8)
model = net.to_torch()
state = torch.tensor(net.initial_state(), dtype=torch.float64)
with torch.no_grad():
    for _ in range(1000):
        state = model(state)
        net.assert_valid(state.tolist())
        if state[model.terminate_channel].item() == 1.0:
            break
    else:
        raise RuntimeError("Recurrence budget exceeded")
```

In a batch, completed rows remain fixed while other rows continue. Stop scheduling
calls when all rows' termination channels are one.

## Checks

```sh
python3 -m unittest discover -s tests -v
```

Tests cover signed multiplication, all assignment forms, recurrent globals,
backward and unconditional jumps, strict-zero and margin boundary behavior,
unreachable code, large-value overwrites, randomized programs, and weight-only
export/reload. Array checks cover constant and dynamic reads/writes, aliases,
index mutation, indexed branches, fractional and out-of-bounds indices, persistent
contents, and CLI/export behavior. When PyTorch is installed, the same suite also
checks the dense module, batches, and gradients, including array gather/scatter.
Structured checks compare against independent executions of trusted Python test
programs and also verify the primitive interpreter at every recurrent step. They
cover multiple outputs, call isolation, evaluation order, nested loops, early
returns, termination budgets, export/reload, terminal fixed points, and gradients
through recurrent function calls. Dynamic-call tests cover recursion, mutual
recursion, live expression values, stack capacity/overflow, shared function bodies,
opt-in inlining, indirect returns, and `break` in both call modes.
