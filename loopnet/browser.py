"""Export source instructions for browser-side weight synthesis, never weights."""

from array import array
import gzip
import json
import struct
import sys

from .language import ArrayAccess
from .regions import execution_regions, recurrent_variables


def browser_program(program, *, max_steps=1, branch_margin=1e-4,
                    execution="block", optimize_temporaries=True):
    """Encode lowered source as uint32 words plus a small JSON symbol table.

    Region selection and liveness are frontend work. CSE, the activation DAG,
    dead-node elimination, layer scheduling, and all neural weights are built
    on the visitor's computer by web/compiler.mjs.
    """
    targets = program.address_targets
    regions = execution_regions(program, execution, targets)
    variables = recurrent_variables(program, regions, eliminate=execution == "block" and optimize_temporaries,
                                    indirect_targets=targets)
    symbols = {name: index for index, name in enumerate(program.variables)}
    arrays = [{"name": name, "cells": [symbols[f"{name}[{i}]"] for i in range(size)]}
              for name, size in program.arrays]
    array_ids = {item["name"]: index for index, item in enumerate(arrays)}
    accesses, access_ids = [], {}
    constants, constant_ids = [], {}

    def address(value):
        if not isinstance(value, ArrayAccess):
            return symbols[value]
        key = (array_ids[value.name], symbols[value.index])
        if key not in access_ids:
            access_ids[key] = len(symbols) + len(accesses)
            accesses.append(key)
        return access_ids[key]

    def constant(value):
        key = struct.pack("<d", value)
        if key not in constant_ids:
            constant_ids[key] = len(constants)
            constants.append(value)
        return constant_ids[key]

    code = array("I")
    boundaries = {index for region in regions for index in region}
    offsets = {}
    for index, instruction in enumerate(program.instructions):
        if index in boundaries:
            offsets[index] = len(code)
        op = instruction.op
        if op == "set":
            code.extend((0, address(instruction.dest), constant(instruction.right)))
        elif op in {"add", "mul"}:
            immediate = isinstance(instruction.right, float)
            if op == "add" and immediate and instruction.right == 0:
                code.extend((11, address(instruction.dest), address(instruction.left)))
            else:
                opcode = (1 if op == "add" else 3) + (0 if immediate else 1)
                right = constant(instruction.right) if immediate else address(instruction.right)
                code.extend((opcode, address(instruction.dest), address(instruction.left), right))
        elif op == "relu":
            code.extend((5, address(instruction.dest), address(instruction.left)))
        elif op == "goto":
            code.extend((6, instruction.target))
        elif op == "if":
            code.extend((7, address(instruction.left), instruction.target))
        elif op == "indirect":
            code.extend((8, address(instruction.left)))
        else:
            code.append({"return": 9, "halt": 10}[op])
    offsets[len(program.instructions)] = len(code)
    if sys.byteorder != "little":
        code.byteswap()
    manifest = {
        "format": "loopnet.program.v1", "code": "program.bin.gz",
        "word_count": len(code), "variable_count": len(symbols),
        "instruction_count": len(program.instructions),
        "globals": [(name, symbols[name]) for name in variables],
        "arrays": arrays, "accesses": accesses, "constants": constants,
        "regions": [(start, end, offsets[start], offsets[end]) for start, end in regions],
        "continuation": program.continuation,
        "stack_pointer": symbols.get(program.stack_pointer),
        "max_stack_depth": program.max_stack_depth,
        "max_steps": max_steps, "branch_margin": branch_margin, "execution": execution,
    }
    return manifest, gzip.compress(code.tobytes(), compresslevel=9, mtime=0)


def manifest_bytes(manifest):
    return (json.dumps(manifest, allow_nan=False, separators=(",", ":")) + "\n").encode()
