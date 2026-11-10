"""
Safe arithmetic for the `calculator` tool.

Why not eval()? eval("__import__('os').remove('x')") runs real code. Instead we
parse the text into a syntax tree and walk it ourselves, allowing only numbers,
+ - * / ** and brackets. Anything else (names, calls, attributes, strings) is
rejected before any evaluation happens.

    evaluate("(1577 - 1373) / 1373 * 100")  ->  14.857975236707938
    evaluate("__import__('os')")            ->  CalculatorError
"""

import ast
import math
import operator

MAX_LENGTH = 300     # characters; long enough for any real financial formula
MAX_EXPONENT = 100   # 10**100 is fine, 9**9**9 (a 370-million-digit int) is not
DECIMALS = 10        # hide float noise: 0.1 + 0.2 -> 0.3, not 0.30000000000000004

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalculatorError(ValueError):
    """The expression is not allowed or cannot be computed. The message says why."""


def evaluate(expression: str) -> float:
    if len(expression) > MAX_LENGTH:
        raise CalculatorError(f"Expression is longer than {MAX_LENGTH} characters.")
    if "," in expression:
        # "1,577" would parse as the tuple (1, 577) -- a silent wrong answer.
        raise CalculatorError("Remove thousands separators: write 1577, not 1,577.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise CalculatorError(f"Not a valid arithmetic expression: {expression!r}") from None

    result = _eval(tree.body)
    if isinstance(result, float) and not math.isfinite(result):
        raise CalculatorError("Result is too large.")
    return round(float(result), DECIMALS)


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        # bool is a subclass of int in Python; True + 1 is not arithmetic we want.
        if type(node.value) in (int, float):
            return node.value
        raise CalculatorError(f"Only numbers are allowed, not {node.value!r}.")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"Exponent is larger than {MAX_EXPONENT}.")
        try:
            return _BINARY[type(node.op)](left, right)
        except ZeroDivisionError:
            raise CalculatorError("Division by zero.") from None
        except OverflowError:
            raise CalculatorError("Result is too large.") from None

    raise CalculatorError(
        f"{type(node).__name__} is not allowed. Use numbers, + - * / ** and brackets only."
    )
