# SPDX-License-Identifier: Apache-2.0
"""Restricted policy process. JSON lines on stdin/stdout; no Agentick imports.

This limits accidental misuse, not a security boundary for hostile Python.
"""

import ast
import builtins
import json
import sys
from typing import Any

MAX_CODE_BYTES = 32_768
MAX_MEMORY_BYTES = 65_536
BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)
METHOD_NAMES = {
    "add",
    "append",
    "clear",
    "copy",
    "count",
    "discard",
    "extend",
    "get",
    "index",
    "items",
    "keys",
    "pop",
    "remove",
    "reverse",
    "sort",
    "values",
}
FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.TryStar,
    ast.Raise,
    ast.Yield,
    ast.YieldFrom,
)


def compile_policy(code: str) -> Any:
    """Compile only a small, import-free Python policy API."""
    if not code.strip() or len(code.encode()) > MAX_CODE_BYTES:
        raise ValueError("Empty or oversized policy")
    tree = ast.parse(code)
    if not tree.body or any(not isinstance(n, ast.FunctionDef) for n in tree.body):
        raise ValueError("Only function definitions are allowed at module level")
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError(f"Unsupported Python construct: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("Private names are not allowed")
        if isinstance(node, ast.Attribute) and node.attr not in METHOD_NAMES:
            raise ValueError(f"Unsupported attribute: {node.attr}")
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_") or node.decorator_list:
                raise ValueError("Private functions and decorators are not allowed")
    namespace = {"__builtins__": {n: getattr(builtins, n) for n in BUILTIN_NAMES}}
    exec(compile(tree, "<policy>", "exec"), namespace)
    if not callable(namespace.get("act")):
        raise ValueError("Define act(obs, memory)")
    return namespace["act"]


def validate_result(result: Any) -> tuple[int, Any]:
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise ValueError("act must return (action, memory)")
    action, memory = result
    if type(action) is not int or action not in range(9):
        raise ValueError("Action must be an integer from 0 to 8")
    encoded = json.dumps(memory, allow_nan=False)
    if len(encoded.encode()) > MAX_MEMORY_BYTES:
        raise ValueError("Memory exceeds 64 KiB")
    return action, json.loads(encoded)


def main() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    try:
        request = json.loads(sys.stdin.readline())
        act = compile_policy(request["code"])
        sys.stdout.write('{"ready": true}\n')
        sys.stdout.flush()
        memory = None
        for line in sys.stdin:
            action, memory = validate_result(act(json.loads(line), memory))
            sys.stdout.write(json.dumps({"action": action}) + "\n")
            sys.stdout.flush()
    except Exception as exc:
        sys.stdout.write(
            json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:300]}"}) + "\n"
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
