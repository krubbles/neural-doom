"""A weight-only residual ReGLU evaluator. No language instructions run here."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .language import ExecutionLimitError


class NetworkContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Affine:
    terms: tuple[tuple[int, float], ...] = ()
    bias: float = 0.0

    def evaluate(self, state: Sequence[float]) -> float:
        return math.fsum([self.bias, *(weight * state[index] for index, weight in self.terms)])

    def scaled(self, scale: float) -> "Affine":
        return Affine(tuple((index, weight * scale) for index, weight in self.terms),
                      self.bias * scale)


def channel(index: int, weight: float = 1.0) -> Affine:
    return Affine(((index, weight),))


@dataclass(frozen=True)
class Unit:
    gate: Affine
    value: Affine
    outputs: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class ResidualBlock:
    """z' = z + W_o (ReLU(W_g z + b_g) * (W_v z + b_v)) + b_o."""

    units: tuple[Unit, ...]
    bias: tuple[tuple[int, float], ...] = ()

    def __call__(self, state: Sequence[float]) -> list[float]:
        contributions: dict[int, list[float]] = {}
        for unit in self.units:
            hidden = max(0.0, unit.gate.evaluate(state)) * unit.value.evaluate(state)
            for index, weight in unit.outputs:
                contributions.setdefault(index, []).append(weight * hidden)
        for index, value in self.bias:
            contributions.setdefault(index, []).append(value)
        result = list(state)
        for index, values in contributions.items():
            result[index] = math.fsum([state[index], *values])
        return result


@dataclass(frozen=True)
class TransientBlock:
    """A deep feedforward branch followed by two residual writeback stages.

    Branch channels are fresh activations, not recurrent neurons. Affine input
    indices address [residual input, transient activations]. Each branch output
    is assigned once. Clear and write both read that immutable branch input,
    preserving small overwrites of large old values without cancellation.
    """
    transient_width: int
    layers: tuple[ResidualBlock, ...]
    clear: ResidualBlock
    write: ResidualBlock

    @property
    def units(self):
        return tuple(unit for layer in (*self.layers, self.clear, self.write) for unit in layer.units)

    def __call__(self, state):
        work = [*state, *([0.0] * self.transient_width)]

        def projection(block):
            contributions = {}
            for unit in block.units:
                gate = unit.gate.evaluate(work)
                if gate <= 0.0:
                    continue
                hidden = gate * unit.value.evaluate(work)
                if not math.isfinite(hidden):
                    raise NetworkContractError("Network arithmetic overflowed or became nonfinite")
                for index, weight in unit.outputs:
                    contributions.setdefault(index, []).append(weight * hidden)
            for index, bias in block.bias:
                contributions.setdefault(index, []).append(bias)
            return contributions

        try:
            for layer in self.layers:
                for index, values in projection(layer).items():
                    work[index] = math.fsum(values)
                    if not math.isfinite(work[index]):
                        raise NetworkContractError("Network arithmetic overflowed or became nonfinite")
            result = list(state)
            for block in (self.clear, self.write):
                for index, values in projection(block).items():
                    result[index] = math.fsum([result[index], *values])
            return result
        except OverflowError as error:
            raise NetworkContractError("Network arithmetic overflowed or became nonfinite") from error


@dataclass(frozen=True)
class Network:
    width: int
    variables: tuple[str, ...]
    done_channel: int
    margin_channel: int
    branch_margin: float
    max_steps: int
    blocks: tuple[ResidualBlock | TransientBlock, ...]
    schedule: tuple[int, ...]
    arrays: tuple[tuple[str, int], ...] = ()
    index_channel: int | None = None
    terminate_channel: int | None = None
    continuation: bool = False
    jump_channel: int | None = None
    stack_channel: int | None = None
    max_stack_depth: int | None = None
    pc_channel: int | None = None
    execution: str = "instruction"
    regions: tuple[tuple[int, int], ...] = ()

    def __call__(self, state: Sequence[float]) -> list[float]:
        """One fixed forward pass; continuation mode preserves control state."""
        if len(state) != self.width:
            raise ValueError(f"Expected {self.width} channels, received {len(state)}")
        result = list(state)
        for block_id in self.schedule:
            result = self.blocks[block_id](result)
        return result

    def initial_state(self, values: Mapping[str, float] | None = None) -> list[float]:
        values = dict(values or {})
        unknown = values.keys() - set(self.variables)
        if unknown:
            raise ValueError(f"Unknown globals: {', '.join(sorted(unknown))}")
        state = [0.0] * self.width
        for index, name in enumerate(self.variables):
            value = float(values.get(name, 0.0))
            if not math.isfinite(value):
                raise ValueError("Globals must be finite floats")
            state[index] = value
        return state

    def globals(self, state: Sequence[float], *, include_internal: bool = False) -> dict[str, float]:
        return {name: state[index] for index, name in enumerate(self.variables)
                if include_internal or not (self.continuation and name.startswith("__ln_"))}

    def terminated(self, state: Sequence[float]) -> bool:
        return self.terminate_channel is not None and state[self.terminate_channel] > 0.0

    def run(self, values: Mapping[str, float] | None = None, *, max_ticks: int = 10000) -> dict[str, float]:
        """Run recurrence until the terminate output is set, with a host budget."""
        if not isinstance(max_ticks, int) or isinstance(max_ticks, bool) or max_ticks < 1:
            raise ValueError("max_ticks must be a positive integer")
        state = self.initial_state(values)
        for _ in range(max_ticks):
            state = self(state)
            self.assert_valid(state)
            if self.terminated(state):
                return self.globals(state)
        raise ExecutionLimitError(f"Program did not terminate within {max_ticks} network calls")

    def assert_valid(self, state: Sequence[float]) -> None:
        """Host-side diagnostics only; these checks never execute source code."""
        if len(state) != self.width:
            raise ValueError(f"Expected {self.width} channels, received {len(state)}")
        if not all(math.isfinite(value) for value in state):
            raise NetworkContractError("Network arithmetic overflowed or became nonfinite")
        if self.stack_channel is not None and state[self.stack_channel] > 0.0:
            raise NetworkContractError(f"Call stack overflow, underflow, or fractional pointer (maximum depth {self.max_stack_depth})")
        if self.jump_channel is not None and state[self.jump_channel] > 0.0:
            raise NetworkContractError("An indirect jump target was not a valid integer instruction address")
        if self.index_channel is not None and state[self.index_channel] > 0.0:
            raise NetworkContractError("An active array index was fractional or out of bounds")
        if state[self.margin_channel] > 0.0:
            raise NetworkContractError(
                f"An active positive branch condition was below the {self.branch_margin:g} margin")
        if self.terminate_channel is not None and state[self.terminate_channel] not in (0.0, 1.0):
            raise NetworkContractError("The terminate channel must be zero or one")
        if not self.continuation and state[self.done_channel] != 1.0:
            units = "instructions" if self.execution == "instruction" else "basic blocks"
            raise NetworkContractError(
                f"Function did not return within {self.max_steps} {units}")

    def step(self, values: Mapping[str, float] | None = None) -> dict[str, float]:
        if self.continuation:
            raise ValueError("Use network(state) for persistent control, or run() until termination")
        state = self(self.initial_state(values))
        self.assert_valid(state)
        return self.globals(state)

    def summary(self) -> dict:
        return {
            "architecture": "residual ReGLU, no normalization",
            "arithmetic": "float64",
            "globals": list(self.variables),
            "arrays": dict(self.arrays),
            "state_width": self.width,
            "residual_blocks": sum(2 if isinstance(self.blocks[i], TransientBlock) else 1
                                   for i in self.schedule),
            "residual_modules": len(self.schedule),
            "feedforward_layers": sum(len(self.blocks[i].layers) + 2
                                      if isinstance(self.blocks[i], TransientBlock) else 1
                                      for i in self.schedule),
            "transient_channels": max((b.transient_width for b in self.blocks
                                        if isinstance(b, TransientBlock)), default=0),
            "peak_hidden_width": max((len(layer.units) for b in self.blocks
                                      for layer in ((*b.layers, b.clear, b.write)
                                                    if isinstance(b, TransientBlock) else (b,))), default=0),
            "unique_blocks": len(self.blocks),
            "gated_units_per_call": sum(len(self.blocks[i].units) for i in self.schedule),
            "max_steps": self.max_steps,
            "execution": self.execution,
            "execution_regions": len(self.regions),
            "pc_channel": self.pc_channel,
            "branch_margin": self.branch_margin,
            "mode": "continuation" if self.continuation else "function",
            "terminate_channel": self.terminate_channel,
            "max_stack_depth": self.max_stack_depth,
        }

    def save(self, path: str | Path) -> None:
        """Serialize only weights, layout, and the fixed layer schedule."""
        def affine(expr):
            return {"terms": expr.terms, "bias": expr.bias}

        def block_data(block):
            if isinstance(block, TransientBlock):
                return {"kind": "transient", "transient_width": block.transient_width,
                        "layers": [block_data(layer) for layer in block.layers],
                        "clear": block_data(block.clear), "write": block_data(block.write)}
            return {"units": [{"gate": affine(unit.gate), "value": affine(unit.value),
                               "outputs": unit.outputs} for unit in block.units], "bias": block.bias}

        data = {
            "format": "loopnet.reglu.v5", "width": self.width,
            "variables": self.variables, "done_channel": self.done_channel,
            "arrays": dict(self.arrays), "index_channel": self.index_channel,
            "terminate_channel": self.terminate_channel, "continuation": self.continuation,
            "jump_channel": self.jump_channel, "stack_channel": self.stack_channel,
            "max_stack_depth": self.max_stack_depth,
            "margin_channel": self.margin_channel, "branch_margin": self.branch_margin,
            "max_steps": self.max_steps, "schedule": self.schedule,
            "pc_channel": self.pc_channel, "execution": self.execution, "regions": self.regions,
            "blocks": [block_data(block) for block in self.blocks],
        }
        Path(path).write_text(json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Network":
        data = json.loads(Path(path).read_text())
        if data.get("format") not in {f"loopnet.reglu.v{version}" for version in range(1, 6)}:
            raise ValueError("Unsupported network format")

        def affine(item):
            return Affine(tuple((int(i), float(w)) for i, w in item["terms"]),
                          float(item["bias"]))

        def read_block(block):
            if block.get("kind") == "transient":
                return TransientBlock(block["transient_width"],
                                      tuple(read_block(layer) for layer in block["layers"]),
                                      read_block(block["clear"]), read_block(block["write"]))
            return ResidualBlock(tuple(
                Unit(affine(unit["gate"]), affine(unit["value"]),
                     tuple((int(i), float(w)) for i, w in unit["outputs"]))
                for unit in block["units"]), tuple((int(i), float(w)) for i, w in block["bias"]))

        blocks = tuple(read_block(block) for block in data["blocks"])
        return cls(data["width"], tuple(data["variables"]), data["done_channel"],
                   data["margin_channel"], data["branch_margin"], data["max_steps"],
                   blocks, tuple(data["schedule"]),
                   arrays=tuple(data.get("arrays", {}).items()),
                   index_channel=data.get("index_channel"),
                   terminate_channel=data.get("terminate_channel"),
                   continuation=data.get("continuation", False),
                   jump_channel=data.get("jump_channel"), stack_channel=data.get("stack_channel"),
                   max_stack_depth=data.get("max_stack_depth"), pc_channel=data.get("pc_channel"),
                   execution=data.get("execution", "instruction"),
                   regions=tuple(tuple(region) for region in data.get("regions", ())))

    def to_torch(self, *, dtype=None, device=None):
        """Optional differentiable dense torch.nn.Module, defaulting to float64.

        PyTorch is optional and is not needed by the compiler or sparse evaluator.
        Dense export is convenient for experimentation, rather than memory efficient.
        """
        import torch
        from torch import nn

        dtype = dtype or torch.float64

        class TorchBlock(nn.Module):
            def __init__(self, block, width):
                super().__init__()
                hidden = len(block.units)
                self.gate = nn.Linear(width, hidden, dtype=dtype, device=device)
                self.value = nn.Linear(width, hidden, dtype=dtype, device=device)
                self.output = nn.Linear(hidden, width, dtype=dtype, device=device)
                with torch.no_grad():
                    for parameter in self.parameters():
                        parameter.zero_()
                    for row, unit in enumerate(block.units):
                        for column, weight in unit.gate.terms:
                            self.gate.weight[row, column] += weight
                        self.gate.bias[row] = unit.gate.bias
                        for column, weight in unit.value.terms:
                            self.value.weight[row, column] += weight
                        self.value.bias[row] = unit.value.bias
                        for target, weight in unit.outputs:
                            self.output.weight[target, row] += weight
                    for target, value in block.bias:
                        self.output.bias[target] += value

            def forward(self, state):
                return state + self.output(torch.relu(self.gate(state)) * self.value(state))

        class TorchProjection(nn.Module):
            """Dense trainable matrices over this layer's connected input ports."""
            def __init__(self, block):
                super().__init__()
                inputs = sorted({i for u in block.units for e in (u.gate, u.value) for i, _ in e.terms})
                outputs = sorted({i for u in block.units for i, _ in u.outputs} | {i for i, _ in block.bias})
                self.register_buffer("inputs", torch.tensor(inputs, dtype=torch.long, device=device))
                self.register_buffer("outputs", torch.tensor(outputs, dtype=torch.long, device=device))
                input_map = {index: column for column, index in enumerate(inputs)}
                output_map = {index: row for row, index in enumerate(outputs)}
                hidden = len(block.units)
                self.gate = nn.Linear(len(inputs), hidden, dtype=dtype, device=device)
                self.value = nn.Linear(len(inputs), hidden, dtype=dtype, device=device)
                self.output = nn.Linear(hidden, len(outputs), dtype=dtype, device=device)
                with torch.no_grad():
                    for parameter in self.parameters():
                        parameter.zero_()
                    for row, unit in enumerate(block.units):
                        for index, weight in unit.gate.terms:
                            self.gate.weight[row, input_map[index]] += weight
                        self.gate.bias[row] = unit.gate.bias
                        for index, weight in unit.value.terms:
                            self.value.weight[row, input_map[index]] += weight
                        self.value.bias[row] = unit.value.bias
                        for index, weight in unit.outputs:
                            self.output.weight[output_map[index], row] += weight
                    for index, bias in block.bias:
                        self.output.bias[output_map[index]] += bias

            def forward(self, work):
                inputs = work.index_select(-1, self.inputs)
                return self.output(torch.relu(self.gate(inputs)) * self.value(inputs))

        class TorchTransientBlock(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.transient_width = block.transient_width
                self.layers = nn.ModuleList(TorchProjection(layer) for layer in block.layers)
                self.clear = TorchProjection(block.clear)
                self.write = TorchProjection(block.write)

            def forward(self, state):
                hidden = state.new_zeros((*state.shape[:-1], self.transient_width))
                work = torch.cat((state, hidden), dim=-1)
                for layer in self.layers:
                    work = work.index_copy(-1, layer.outputs, layer(work))
                result = state.index_add(-1, self.clear.outputs, self.clear(work))
                return result.index_add(-1, self.write.outputs, self.write(work))

        class TorchNetwork(nn.Module):
            def __init__(self, network):
                super().__init__()
                self.blocks = nn.ModuleList(TorchTransientBlock(block) if isinstance(block, TransientBlock)
                                            else TorchBlock(block, network.width)
                                            for block in network.blocks)
                self.schedule = network.schedule
                self.terminate_channel = network.terminate_channel
                self.continuation = network.continuation
                self.pc_channel = network.pc_channel
                self.execution = network.execution

            def forward(self, state):
                for block_id in self.schedule:
                    state = self.blocks[block_id](state)
                return state

        return TorchNetwork(self)
