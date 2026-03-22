"""
Safe recursive-descent expression parser and evaluator.

Parses trading strategy conditions like:
    "close < bb_20.lower AND rsi_14 < 30"

NO eval() — all evaluation is through explicit AST traversal.

Grammar (operator precedence, lowest to highest):
    expression  -> or_expr
    or_expr     -> and_expr ( "OR" and_expr )*
    and_expr    -> not_expr ( "AND" not_expr )*
    not_expr    -> "NOT" not_expr | comparison
    comparison  -> arithmetic ( ( "<" | ">" | "<=" | ">=" | "==" | "!=" ) arithmetic )?
    arithmetic  -> term ( ( "+" | "-" ) term )*
    term        -> factor ( ( "*" | "/" ) factor )*
    factor      -> NUMBER | VARIABLE | "(" expression ")"
    VARIABLE    -> IDENTIFIER ( "." IDENTIFIER )*
    NUMBER      -> FLOAT_LITERAL
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ExpressionParseError(Exception):
    """Raised when an expression cannot be parsed."""


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

@dataclass
class NumberNode:
    value: float


@dataclass
class VariableNode:
    parts: list[str]  # ["bb_20", "lower"] for bb_20.lower


@dataclass
class BinaryOpNode:
    op: str  # "+", "-", "*", "/", "<", ">", "<=", ">=", "==", "!="
    left: ASTNode
    right: ASTNode


@dataclass
class BooleanOpNode:
    op: str  # "AND", "OR"
    left: ASTNode
    right: ASTNode


@dataclass
class NotNode:
    operand: ASTNode


ASTNode = Union[NumberNode, VariableNode, BinaryOpNode, BooleanOpNode, NotNode]

# Fix forward references in dataclass fields
BinaryOpNode.__annotations__["left"] = ASTNode
BinaryOpNode.__annotations__["right"] = ASTNode
BooleanOpNode.__annotations__["left"] = ASTNode
BooleanOpNode.__annotations__["right"] = ASTNode
NotNode.__annotations__["operand"] = ASTNode


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TokenType:
    NUMBER = "NUMBER"
    IDENTIFIER = "IDENTIFIER"
    OPERATOR = "OPERATOR"     # comparison and arithmetic operators
    KEYWORD = "KEYWORD"       # AND, OR, NOT
    DOT = "DOT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    EOF = "EOF"


@dataclass
class Token:
    type: str
    value: str
    pos: int  # position in the original expression for error messages


_KEYWORDS = {"AND", "OR", "NOT"}

# Two-char operators must come before their single-char prefixes in the regex
_TOKEN_SPEC = [
    ("NUMBER",   r"\d+(?:\.\d+)?"),
    ("OP2",      r"<=|>=|==|!="),
    ("OP1",      r"[<>+\-*/]"),
    ("DOT",      r"\."),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("IDENT",    r"[A-Za-z_][A-Za-z0-9_]*"),
    ("SKIP",     r"\s+"),
    ("MISMATCH", r"."),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC))


def tokenize(expression: str) -> list[Token]:
    """Convert an expression string into a list of tokens."""
    tokens: list[Token] = []
    for m in _TOKEN_RE.finditer(expression):
        kind = m.lastgroup
        value = m.group()
        pos = m.start()

        if kind == "SKIP":
            continue
        elif kind == "MISMATCH":
            raise ExpressionParseError(f"Unexpected character {value!r} at position {pos}")
        elif kind == "NUMBER":
            tokens.append(Token(TokenType.NUMBER, value, pos))
        elif kind in ("OP1", "OP2"):
            tokens.append(Token(TokenType.OPERATOR, value, pos))
        elif kind == "DOT":
            tokens.append(Token(TokenType.DOT, value, pos))
        elif kind == "LPAREN":
            tokens.append(Token(TokenType.LPAREN, value, pos))
        elif kind == "RPAREN":
            tokens.append(Token(TokenType.RPAREN, value, pos))
        elif kind == "IDENT":
            if value in _KEYWORDS:
                tokens.append(Token(TokenType.KEYWORD, value, pos))
            else:
                tokens.append(Token(TokenType.IDENTIFIER, value, pos))

    tokens.append(Token(TokenType.EOF, "", len(expression)))
    return tokens


# ---------------------------------------------------------------------------
# Parser — recursive descent
# ---------------------------------------------------------------------------

class Parser:
    """Recursive-descent parser that builds an AST from tokens."""

    def __init__(self, tokens: list[Token], expression: str):
        self.tokens = tokens
        self.expression = expression
        self.pos = 0

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, token_type: str, value: str | None = None) -> Token:
        tok = self._peek()
        if tok.type != token_type or (value is not None and tok.value != value):
            expected = f"{token_type}" + (f" {value!r}" if value else "")
            raise ExpressionParseError(
                f"Expected {expected} but got {tok.type} {tok.value!r} "
                f"at position {tok.pos} in: {self.expression!r}"
            )
        return self._advance()

    def parse(self) -> ASTNode:
        if self._peek().type == TokenType.EOF:
            raise ExpressionParseError("Empty expression")
        node = self._or_expr()
        if self._peek().type != TokenType.EOF:
            tok = self._peek()
            raise ExpressionParseError(
                f"Unexpected token {tok.value!r} at position {tok.pos} "
                f"in: {self.expression!r}"
            )
        return node

    # expression -> or_expr
    def _or_expr(self) -> ASTNode:
        left = self._and_expr()
        while self._peek().type == TokenType.KEYWORD and self._peek().value == "OR":
            self._advance()
            right = self._and_expr()
            left = BooleanOpNode(op="OR", left=left, right=right)
        return left

    # or_expr -> and_expr ( "AND" and_expr )*
    def _and_expr(self) -> ASTNode:
        left = self._not_expr()
        while self._peek().type == TokenType.KEYWORD and self._peek().value == "AND":
            self._advance()
            right = self._not_expr()
            left = BooleanOpNode(op="AND", left=left, right=right)
        return left

    # not_expr -> "NOT" not_expr | comparison
    def _not_expr(self) -> ASTNode:
        if self._peek().type == TokenType.KEYWORD and self._peek().value == "NOT":
            self._advance()
            operand = self._not_expr()
            return NotNode(operand=operand)
        return self._comparison()

    # comparison -> arithmetic ( ( "<" | ">" | "<=" | ">=" | "==" | "!=" ) arithmetic )?
    def _comparison(self) -> ASTNode:
        left = self._arithmetic()
        if self._peek().type == TokenType.OPERATOR and self._peek().value in (
            "<", ">", "<=", ">=", "==", "!="
        ):
            op = self._advance().value
            right = self._arithmetic()
            left = BinaryOpNode(op=op, left=left, right=right)
        return left

    # arithmetic -> term ( ( "+" | "-" ) term )*
    def _arithmetic(self) -> ASTNode:
        left = self._term()
        while self._peek().type == TokenType.OPERATOR and self._peek().value in ("+", "-"):
            op = self._advance().value
            right = self._term()
            left = BinaryOpNode(op=op, left=left, right=right)
        return left

    # term -> factor ( ( "*" | "/" ) factor )*
    def _term(self) -> ASTNode:
        left = self._factor()
        while self._peek().type == TokenType.OPERATOR and self._peek().value in ("*", "/"):
            op = self._advance().value
            right = self._factor()
            left = BinaryOpNode(op=op, left=left, right=right)
        return left

    # factor -> ( "-" factor ) | NUMBER | VARIABLE | "(" expression ")"
    def _factor(self) -> ASTNode:
        tok = self._peek()

        # Unary minus: -expr  =>  0 - expr
        if tok.type == TokenType.OPERATOR and tok.value == "-":
            self._advance()
            operand = self._factor()
            return BinaryOpNode(op="-", left=NumberNode(value=0.0), right=operand)

        if tok.type == TokenType.NUMBER:
            self._advance()
            return NumberNode(value=float(tok.value))

        if tok.type == TokenType.IDENTIFIER:
            return self._variable()

        if tok.type == TokenType.LPAREN:
            self._advance()
            node = self._or_expr()
            self._expect(TokenType.RPAREN)
            return node

        raise ExpressionParseError(
            f"Unexpected token {tok.value!r} at position {tok.pos} "
            f"in: {self.expression!r}"
        )

    # VARIABLE -> IDENTIFIER ( "." IDENTIFIER )*
    def _variable(self) -> VariableNode:
        parts = [self._expect(TokenType.IDENTIFIER).value]
        while self._peek().type == TokenType.DOT:
            self._advance()
            parts.append(self._expect(TokenType.IDENTIFIER).value)
        return VariableNode(parts=parts)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class _UndefinedSentinel:
    """Sentinel returned when a variable is undefined, causing comparisons to
    yield False rather than crash."""
    pass


_UNDEFINED = _UndefinedSentinel()


def _resolve_numeric(value) -> float:
    """Resolve a value to a numeric float.

    If the value is an object with a .value attribute (like an Indicator),
    use that attribute. Otherwise convert directly.
    """
    if isinstance(value, _UndefinedSentinel):
        return value  # propagate sentinel
    if isinstance(value, (int, float)):
        return float(value)
    # Indicator-like objects: use .value property
    if hasattr(value, "value"):
        v = value.value
        if v is None:
            return _UNDEFINED
        return float(v)
    return float(value)


def evaluate_ast(node: ASTNode, context: dict) -> float | bool:
    """Evaluate a parsed AST against a context dictionary.

    Returns:
        bool for comparison/boolean nodes, float for arithmetic/variable/number nodes.
        Returns False if an undefined variable is encountered in a comparison.
    """
    if isinstance(node, NumberNode):
        return node.value

    if isinstance(node, VariableNode):
        # Resolve first part from context
        root_name = node.parts[0]
        if root_name not in context:
            logger.warning("Undefined variable %r in expression", root_name)
            return _UNDEFINED

        value = context[root_name]

        # Resolve subsequent parts via attribute access
        for attr_name in node.parts[1:]:
            if isinstance(value, _UndefinedSentinel):
                return _UNDEFINED
            try:
                value = getattr(value, attr_name)
            except AttributeError:
                logger.warning(
                    "Object %r has no attribute %r", root_name, attr_name
                )
                return _UNDEFINED

        # Auto-resolve indicator .value for numeric context (no dot access)
        if len(node.parts) == 1:
            return _resolve_numeric(value)
        else:
            # Dot-accessed attribute — resolve if it's a property
            if isinstance(value, (int, float)):
                return float(value)
            if value is None:
                return _UNDEFINED
            return float(value)

    if isinstance(node, BinaryOpNode):
        left = evaluate_ast(node.left, context)
        right = evaluate_ast(node.right, context)

        # If either side is undefined, comparisons return False
        if isinstance(left, _UndefinedSentinel) or isinstance(right, _UndefinedSentinel):
            if node.op in ("<", ">", "<=", ">=", "==", "!="):
                return False
            return _UNDEFINED

        left_f = float(left)
        right_f = float(right)

        # Arithmetic operators
        if node.op == "+":
            return left_f + right_f
        if node.op == "-":
            return left_f - right_f
        if node.op == "*":
            return left_f * right_f
        if node.op == "/":
            if right_f == 0:
                logger.warning("Division by zero in expression")
                return _UNDEFINED
            return left_f / right_f

        # Comparison operators
        if node.op == "<":
            return left_f < right_f
        if node.op == ">":
            return left_f > right_f
        if node.op == "<=":
            return left_f <= right_f
        if node.op == ">=":
            return left_f >= right_f
        if node.op == "==":
            return left_f == right_f
        if node.op == "!=":
            return left_f != right_f

        raise ExpressionParseError(f"Unknown binary operator: {node.op}")

    if isinstance(node, BooleanOpNode):
        left = evaluate_ast(node.left, context)
        right = evaluate_ast(node.right, context)

        # Coerce to bool — undefined is False
        left_b = False if isinstance(left, _UndefinedSentinel) else bool(left)
        right_b = False if isinstance(right, _UndefinedSentinel) else bool(right)

        if node.op == "AND":
            return left_b and right_b
        if node.op == "OR":
            return left_b or right_b

        raise ExpressionParseError(f"Unknown boolean operator: {node.op}")

    if isinstance(node, NotNode):
        operand = evaluate_ast(node.operand, context)
        operand_b = False if isinstance(operand, _UndefinedSentinel) else bool(operand)
        return not operand_b

    raise ExpressionParseError(f"Unknown AST node type: {type(node)}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(expression: str) -> ASTNode:
    """Parse an expression string into an AST.

    Raises ExpressionParseError if the expression is malformed.
    """
    tokens = tokenize(expression)
    parser = Parser(tokens, expression)
    return parser.parse()


def evaluate(expression: str, context: dict) -> float | bool:
    """Parse and evaluate an expression string.

    Convenience function that combines parse() + evaluate_ast().
    For repeated evaluation of the same expression with different contexts,
    use parse() once then call evaluate_ast() with the returned AST.
    """
    ast = parse(expression)
    result = evaluate_ast(ast, context)
    if isinstance(result, _UndefinedSentinel):
        return False
    return result
