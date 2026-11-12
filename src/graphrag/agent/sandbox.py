"""Deterministic arithmetic sandbox.

LLMs are unreliable at token-level arithmetic and at accounting sign
conventions. Every number the system *computes* (growth rates, ratios,
deltas, unit conversions) goes through this evaluator instead.

Security: no ``eval``/``exec``. The expression is parsed to a Python AST and
only numeric literals, arithmetic operators and a whitelist of pure functions
are interpreted. Names, attributes, subscripts, calls to anything else,
comprehensions and lambdas are rejected. Exponents and magnitudes are bounded.
"""

from __future__ import annotations

import ast
import math
import operator
import re

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _pct_change(old: float, new: float) -> float:
    if old == 0:
        raise ZeroDivisionError("pct_change from zero")
    return (new - old) / abs(old) * 100.0


def _cagr(start: float, end: float, years: float) -> float:
    if start <= 0 or years <= 0:
        raise ValueError("cagr needs positive start and years")
    return ((end / start) ** (1.0 / years) - 1.0) * 100.0


_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": lambda *a: sum(a[0]) if len(a) == 1 and isinstance(a[0], (list, tuple)) else sum(a),
    "sqrt": math.sqrt, "log": math.log, "pct_change": _pct_change, "cagr": _cagr,
    "avg": lambda *a: sum(a) / len(a),
}

MAX_ABS = 1e18
MAX_EXP = 100


class SandboxError(ValueError):
    pass


def normalize_numbers(expr: str) -> str:
    """Accounting -> Python: '(1,234)' -> '(-1234)', '1,234.5' -> '1234.5', '12%' -> '(12/100)', '$' stripped."""
    expr = expr.replace("$", "").replace("€", "").replace("−", "-")
    # Parenthetical negatives: a parenthesised bare number (not a function call argument).
    expr = re.sub(r"(?<![\w)])\(\s*(\d[\d,]*(?:\.\d+)?)\s*\)", lambda m: f"(-{m.group(1)})", expr)
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    return expr


def safe_eval(expression: str) -> float:
    src = normalize_numbers(expression.strip())
    if len(src) > 500:
        raise SandboxError("expression too long")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise SandboxError(f"invalid expression: {exc.msg}") from None
    result = _eval(tree.body)
    if isinstance(result, (list, tuple)):
        raise SandboxError("expression must evaluate to a single number")
    if not math.isfinite(result):
        raise SandboxError("non-finite result")
    return float(result)


def _eval(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXP:
            raise SandboxError("exponent too large")
        try:
            out = _BIN[type(node.op)](left, right)
        except ZeroDivisionError:
            raise SandboxError("division by zero") from None
        if abs(out) > MAX_ABS:
            raise SandboxError("magnitude too large")
        return out
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        args = [_eval(a) for a in node.args]
        try:
            return _FUNCS[node.func.id](*args)
        except (ValueError, ZeroDivisionError, TypeError) as exc:
            raise SandboxError(f"{node.func.id}: {exc}") from None
    raise SandboxError(f"disallowed syntax: {type(node).__name__}")


def run_calculation(expression: str, label: str = "") -> dict:
    try:
        value = safe_eval(expression)
        return {"label": label, "expression": expression, "normalized": normalize_numbers(expression),
                "result": round(value, 6), "ok": True}
    except SandboxError as exc:
        return {"label": label, "expression": expression, "error": str(exc), "ok": False}
