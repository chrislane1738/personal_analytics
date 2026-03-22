"""Tests for the expression parser — safe recursive-descent, NO eval()."""

import logging
import pytest

from strategy.expression_parser import (
    ExpressionParseError,
    evaluate,
    parse,
)


# ---------------------------------------------------------------------------
# Simple comparisons
# ---------------------------------------------------------------------------

class TestSimpleComparison:
    def test_less_than_true(self):
        assert evaluate("close < 100", {"close": 95.0}) is True

    def test_less_than_false(self):
        assert evaluate("close < 100", {"close": 105.0}) is False

    def test_greater_than(self):
        assert evaluate("close > 100", {"close": 105.0}) is True

    def test_less_equal(self):
        assert evaluate("close <= 100", {"close": 100.0}) is True
        assert evaluate("close <= 100", {"close": 100.1}) is False

    def test_greater_equal(self):
        assert evaluate("close >= 100", {"close": 100.0}) is True
        assert evaluate("close >= 100", {"close": 99.9}) is False

    def test_equal(self):
        assert evaluate("close == 100", {"close": 100.0}) is True
        assert evaluate("close == 100", {"close": 99.0}) is False

    def test_not_equal(self):
        assert evaluate("close != 100", {"close": 99.0}) is True
        assert evaluate("close != 100", {"close": 100.0}) is False


# ---------------------------------------------------------------------------
# Boolean operators
# ---------------------------------------------------------------------------

class TestBooleanOps:
    def test_and_both_true(self):
        assert evaluate("close < 100 AND rsi_14 < 30", {"close": 95.0, "rsi_14": 25.0}) is True

    def test_and_one_false(self):
        assert evaluate("close < 100 AND rsi_14 < 30", {"close": 95.0, "rsi_14": 35.0}) is False

    def test_or_one_true(self):
        assert evaluate("close < 90 OR rsi_14 < 30", {"close": 95.0, "rsi_14": 25.0}) is True

    def test_or_both_false(self):
        assert evaluate("close < 90 OR rsi_14 < 30", {"close": 95.0, "rsi_14": 35.0}) is False

    def test_not_true(self):
        assert evaluate("NOT close > 100", {"close": 95.0}) is True

    def test_not_false(self):
        assert evaluate("NOT close > 100", {"close": 105.0}) is False

    def test_chained_and(self):
        ctx = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert evaluate("a < 5 AND b < 5 AND c < 5", ctx) is True
        assert evaluate("a < 5 AND b < 5 AND c < 1", ctx) is False

    def test_chained_or(self):
        ctx = {"a": 10.0, "b": 10.0, "c": 1.0}
        assert evaluate("a < 5 OR b < 5 OR c < 5", ctx) is True


# ---------------------------------------------------------------------------
# Dot notation (attribute access)
# ---------------------------------------------------------------------------

class TestDotNotation:
    def test_single_dot(self):
        class MockBB:
            lower = 90.0
            upper = 110.0
        assert evaluate("close < bb_20.lower", {"close": 85.0, "bb_20": MockBB()}) is True
        assert evaluate("close < bb_20.lower", {"close": 95.0, "bb_20": MockBB()}) is False

    def test_upper_attribute(self):
        class MockBB:
            upper = 110.0
        assert evaluate("close > bb_20.upper", {"close": 115.0, "bb_20": MockBB()}) is True


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_multiplication(self):
        # close * 0.95 = 95.0, which is < 100
        assert evaluate("close * 0.95 < 100", {"close": 100.0}) is True

    def test_division(self):
        assert evaluate("close / 2 < 60", {"close": 100.0}) is True  # 50 < 60

    def test_addition(self):
        assert evaluate("close + 10 > 100", {"close": 95.0}) is True  # 105 > 100

    def test_subtraction(self):
        assert evaluate("close - 10 < 90", {"close": 95.0}) is True  # 85 < 90

    def test_precedence_mul_over_add(self):
        # 2 + 3 * 4 should be 14 not 20
        assert evaluate("a + b * c == 14", {"a": 2.0, "b": 3.0, "c": 4.0}) is True

    def test_parenthesized_expression(self):
        # (2 + 3) * 4 = 20
        assert evaluate("(a + b) * c == 20", {"a": 2.0, "b": 3.0, "c": 4.0}) is True


# ---------------------------------------------------------------------------
# Complex expressions
# ---------------------------------------------------------------------------

class TestComplex:
    def test_bb_and_rsi(self):
        class MockBB:
            lower = 90.0
        assert evaluate(
            "close < bb_20.lower AND rsi_14 < 30",
            {"close": 85.0, "bb_20": MockBB(), "rsi_14": 25.0},
        ) is True

    def test_mixed_arithmetic_and_boolean(self):
        assert evaluate(
            "close * 1.05 > 100 AND volume > 1000",
            {"close": 96.0, "volume": 2000.0},
        ) is True  # 96 * 1.05 = 100.8 > 100


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_undefined_variable_returns_false(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = evaluate("undefined_var < 30", {})
        assert result is False

    def test_parse_error_on_bad_syntax(self):
        with pytest.raises(ExpressionParseError):
            parse("close < < 30")

    def test_parse_error_on_empty_string(self):
        with pytest.raises(ExpressionParseError):
            parse("")

    def test_parse_error_on_unclosed_paren(self):
        with pytest.raises(ExpressionParseError):
            parse("(close < 100")


# ---------------------------------------------------------------------------
# AST caching / parse then evaluate
# ---------------------------------------------------------------------------

class TestParseAndEvaluate:
    def test_parse_then_evaluate(self):
        ast = parse("close < 100 AND rsi_14 < 30")
        from strategy.expression_parser import evaluate_ast
        assert evaluate_ast(ast, {"close": 95.0, "rsi_14": 25.0}) is True
        assert evaluate_ast(ast, {"close": 95.0, "rsi_14": 35.0}) is False


# ---------------------------------------------------------------------------
# Indicator .value attribute resolution
# ---------------------------------------------------------------------------

class TestIndicatorValueResolution:
    """When a context variable is an indicator object (has a .value attribute),
    the evaluator should use .value for numeric comparisons automatically."""

    def test_indicator_value_property(self):
        class MockIndicator:
            @property
            def value(self):
                return 25.0
        ctx = {"rsi_14": MockIndicator()}
        assert evaluate("rsi_14 < 30", ctx) is True
        assert evaluate("rsi_14 > 30", ctx) is False
