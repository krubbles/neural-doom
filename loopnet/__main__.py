import argparse
import json
import math
from pathlib import Path
import sys

from . import ExecutionLimitError, Interpreter, Network, compile_program, lower, parse
from .regions import reference_step


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compile a persistent float state machine into a ReGLU residual net")
    commands = parser.add_subparsers(dest="command", required=True)

    def compiler_options(command):
        command.add_argument("source", type=Path, help="a .loop function or .flow structured program")
        command.add_argument("--structured", action="store_true", help="parse structured syntax (automatic for .flow)")
        command.add_argument("--max-steps", type=int,
                             help="execution regions per call (default: 64 for functions, 1 for continuations)")
        command.add_argument("--execution", choices=("instruction", "block"),
                             help="stepping granularity (default: block for structured, instruction for primitive)")
        command.add_argument("--branch-margin", type=float, default=1e-4)
        command.add_argument("--stack-depth", type=int, default=16, help="maximum nested runtime calls (default: 16)")
        command.add_argument("--unroll-limit", type=int, default=4096, help="maximum compile-time loop expansion (default: 4096)")

    def run_options(command):
        command.add_argument("--ticks", type=int, help="maximum network calls (default: 10 for functions, 10000 for continuations)")
        command.add_argument("--set", action="append", default=[], metavar="GLOBAL=FLOAT",
                             help="initial global value; repeat for multiple globals")

    compile_command = commands.add_parser("compile", help="export weight-only JSON")
    compiler_options(compile_command)
    compile_command.add_argument("-o", "--output", type=Path, required=True)
    run_command = commands.add_parser("run", help="compile and run the recurrent network")
    compiler_options(run_command)
    run_options(run_command)
    run_command.add_argument("--compare", action="store_true", help="check against the reference interpreter")
    info_command = commands.add_parser("info", help="print the compiled architecture")
    compiler_options(info_command)
    exported_command = commands.add_parser("run-net", help="run an exported network without source code")
    exported_command.add_argument("network", type=Path)
    run_options(exported_command)
    lower_command = commands.add_parser("lower", help="reduce structured code to a primitive continuation program")
    lower_command.add_argument("source", type=Path)
    lower_command.add_argument("-o", "--output", type=Path)
    lower_command.add_argument("--stack-depth", type=int, default=16)
    lower_command.add_argument("--unroll-limit", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        if args.command == "lower":
            lowered = lower(args.source.read_text(), max_stack_depth=args.stack_depth,
                            max_unrolled_iterations=args.unroll_limit)
            if args.output:
                args.output.write_text(lowered)
                print(f"Wrote {args.output}")
            else:
                print(lowered, end="")
            return 0
        if args.command == "run-net":
            network = Network.load(args.network)
            program = None
        else:
            source = args.source.read_text()
            structured = args.structured or args.source.suffix == ".flow"
            if structured:
                source = lower(source, max_stack_depth=args.stack_depth,
                               max_unrolled_iterations=args.unroll_limit)
            program = parse(source)
            max_steps = args.max_steps if args.max_steps is not None else (1 if program.continuation else 64)
            network = compile_program(program, max_steps=max_steps,
                                      branch_margin=args.branch_margin,
                                      execution=args.execution or ("block" if structured else "instruction"),
                                      indirect_targets=program.address_targets if structured else None,
                                      optimize_temporaries=structured)
        if args.command == "compile":
            network.save(args.output)
            print(f"Wrote {args.output}")
        elif args.command == "info":
            print(json.dumps(network.summary(), indent=2))
        else:
            ticks = args.ticks if args.ticks is not None else (10000 if network.continuation else 10)
            if ticks < 0:
                raise ValueError("--ticks must be nonnegative")
            initial = {}
            for item in args.set:
                if "=" not in item:
                    raise ValueError("--set requires GLOBAL=FLOAT")
                name, value = item.split("=", 1)
                initial[name] = float(value)
            state = network.initial_state(initial)
            compare = getattr(args, "compare", False)
            reference = Interpreter(program, initial) if compare else None
            worst_error = 0.0
            calls = 0
            for tick in range(1, ticks + 1):
                calls = tick
                state = network(state)
                network.assert_valid(state)
                values = network.globals(state)
                if compare:
                    expected = reference_step(reference, network.regions, network.max_steps)
                    error = max((abs(values[name] - expected[name]) for name in values), default=0.0)
                    worst_error = max(error, worst_error)
                    if any(not math.isclose(values[name], expected[name],
                                            rel_tol=1e-10, abs_tol=1e-12)
                           for name in values) or network.terminated(state) != reference.terminated:
                        raise ArithmeticError(f"Tick {tick}: network and interpreter disagree (error {error:g})")
                print(json.dumps({"tick": tick, "globals": values,
                                  "terminate": float(network.terminated(state))}, allow_nan=False))
                if network.terminated(state):
                    break
            if network.continuation and not network.terminated(state):
                raise ExecutionLimitError(f"Program did not terminate within {ticks} network calls")
            if compare:
                print(f"Matched interpreter for {calls} calls; max absolute error {worst_error:g}",
                      file=sys.stderr)
    except (ValueError, RuntimeError, ArithmeticError, OSError) as error:
        print(f"loopnet: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
