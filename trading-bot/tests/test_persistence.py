"""Tests for dashboard persistence: equity curves, get_trades, delete_run, paginated list_runs."""
import os
import tempfile
from datetime import date, datetime

import pytest

from data.storage.database import Database
from data.storage.models import EquityCurvePoint, RunRecord, TradeRecord


@pytest.fixture
def db():
    """Provide a temporary database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    database = Database(db_path)
    database.create_tables()
    yield database
    database.close()
    os.unlink(db_path)


def _make_run(run_id="run-001", strategy_name="MomentumStrategy",
              total_return=0.15, sharpe=1.42, max_drawdown=0.08,
              created_at=None) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        mode="backtest",
        strategy_name=strategy_name,
        config='{"lookback": 20}',
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        initial_capital=100_000.0,
        final_value=115_000.0,
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        created_at=created_at or datetime(2024, 1, 15, 10, 30, 0),
        full_metrics='{"alpha": 0.05}',
    )


def _make_trade(trade_id, run_id="run-001") -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id,
        run_id=run_id,
        symbol="AAPL",
        direction="long",
        entry_date=date(2023, 1, 5),
        exit_date=date(2023, 1, 10),
        entry_price=150.0,
        exit_price=155.0,
        quantity=100,
        pnl=500.0,
        pnl_pct=0.033,
        entry_reason="momentum signal",
        exit_reason="target hit",
    )


# ---------------------------------------------------------------------------
# Task 0.1: EquityCurvePoint insert and get
# ---------------------------------------------------------------------------

class TestEquityCurve:
    def test_insert_and_get_equity_curve(self, db):
        """Insert equity curve points and retrieve them by run_id."""
        run = _make_run()
        db.insert_run(run)

        points = [
            EquityCurvePoint(
                run_id="run-001",
                date=date(2023, 1, 1),
                strategy_value=100_000.0,
                benchmark_value=100_000.0,
            ),
            EquityCurvePoint(
                run_id="run-001",
                date=date(2023, 1, 2),
                strategy_value=100_500.0,
                benchmark_value=100_200.0,
            ),
            EquityCurvePoint(
                run_id="run-001",
                date=date(2023, 1, 3),
                strategy_value=101_000.0,
                benchmark_value=100_400.0,
            ),
        ]
        db.insert_equity_curve(points)

        result = db.get_equity_curve("run-001")
        assert len(result) == 3
        assert result[0].date == date(2023, 1, 1)
        assert result[0].strategy_value == 100_000.0
        assert result[0].benchmark_value == 100_000.0
        assert result[2].date == date(2023, 1, 3)
        assert result[2].strategy_value == 101_000.0

    def test_get_equity_curve_ordered_by_date(self, db):
        """Equity curve points should be returned in date order."""
        run = _make_run()
        db.insert_run(run)

        # Insert out of order
        points = [
            EquityCurvePoint(run_id="run-001", date=date(2023, 1, 3),
                             strategy_value=101_000.0, benchmark_value=100_400.0),
            EquityCurvePoint(run_id="run-001", date=date(2023, 1, 1),
                             strategy_value=100_000.0, benchmark_value=100_000.0),
            EquityCurvePoint(run_id="run-001", date=date(2023, 1, 2),
                             strategy_value=100_500.0, benchmark_value=100_200.0),
        ]
        db.insert_equity_curve(points)

        result = db.get_equity_curve("run-001")
        dates = [p.date for p in result]
        assert dates == sorted(dates)

    def test_get_equity_curve_empty(self, db):
        """Getting equity curve for nonexistent run returns empty list."""
        result = db.get_equity_curve("nonexistent")
        assert result == []

    def test_equity_curves_table_exists(self, db):
        """The equity_curves table should exist after create_tables()."""
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='equity_curves'", ()
        ).fetchall()
        assert len(rows) == 1

    def test_equity_curves_index_exists(self, db):
        """The equity_curves run_id index should exist."""
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_equity_curves_run_id'", ()
        ).fetchall()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Task 0.2: get_trades
# ---------------------------------------------------------------------------

class TestGetTrades:
    def test_get_trades_by_run_id(self, db):
        """Retrieve trades for a specific run_id."""
        run = _make_run()
        db.insert_run(run)

        trades = [
            _make_trade("trade-001"),
            _make_trade("trade-002"),
            _make_trade("trade-003"),
        ]
        db.insert_trades(trades)

        result = db.get_trades("run-001")
        assert len(result) == 3
        assert all(isinstance(t, TradeRecord) for t in result)
        assert {t.trade_id for t in result} == {"trade-001", "trade-002", "trade-003"}

    def test_get_trades_returns_only_requested_run(self, db):
        """get_trades should only return trades for the specified run."""
        run1 = _make_run("run-001")
        run2 = _make_run("run-002")
        db.insert_run(run1)
        db.insert_run(run2)

        trades = [
            _make_trade("trade-001", "run-001"),
            _make_trade("trade-002", "run-001"),
            _make_trade("trade-003", "run-002"),
        ]
        db.insert_trades(trades)

        result = db.get_trades("run-001")
        assert len(result) == 2
        assert all(t.run_id == "run-001" for t in result)

    def test_get_trades_empty(self, db):
        """get_trades for nonexistent run returns empty list."""
        result = db.get_trades("nonexistent")
        assert result == []

    def test_get_trades_preserves_fields(self, db):
        """Verify all TradeRecord fields are preserved through round-trip."""
        run = _make_run()
        db.insert_run(run)

        trade = TradeRecord(
            trade_id="trade-full",
            run_id="run-001",
            symbol="MSFT",
            direction="short",
            entry_date=date(2023, 3, 10),
            exit_date=date(2023, 3, 20),
            entry_price=280.0,
            exit_price=270.0,
            quantity=50,
            pnl=500.0,
            pnl_pct=0.036,
            entry_reason="mean reversion",
            exit_reason="stop loss",
            option_type="put",
            strike=275.0,
            expiration=date(2023, 4, 21),
        )
        db.insert_trades([trade])

        result = db.get_trades("run-001")
        assert len(result) == 1
        t = result[0]
        assert t.trade_id == "trade-full"
        assert t.symbol == "MSFT"
        assert t.direction == "short"
        assert t.entry_date == date(2023, 3, 10)
        assert t.exit_date == date(2023, 3, 20)
        assert t.entry_price == 280.0
        assert t.exit_price == 270.0
        assert t.quantity == 50
        assert t.pnl == 500.0
        assert t.pnl_pct == 0.036
        assert t.entry_reason == "mean reversion"
        assert t.exit_reason == "stop loss"
        assert t.option_type == "put"
        assert t.strike == 275.0
        assert t.expiration == date(2023, 4, 21)


# ---------------------------------------------------------------------------
# Task 0.2: delete_run
# ---------------------------------------------------------------------------

class TestDeleteRun:
    def test_delete_run_cascades(self, db):
        """delete_run removes the run, its trades, and its equity curve points."""
        run = _make_run()
        db.insert_run(run)

        trades = [_make_trade("trade-001"), _make_trade("trade-002")]
        db.insert_trades(trades)

        points = [
            EquityCurvePoint(run_id="run-001", date=date(2023, 1, 1),
                             strategy_value=100_000.0, benchmark_value=100_000.0),
            EquityCurvePoint(run_id="run-001", date=date(2023, 1, 2),
                             strategy_value=100_500.0, benchmark_value=100_200.0),
        ]
        db.insert_equity_curve(points)

        db.delete_run("run-001")

        assert db.get_run("run-001") is None
        assert db.get_trades("run-001") == []
        assert db.get_equity_curve("run-001") == []

    def test_delete_run_does_not_affect_other_runs(self, db):
        """Deleting one run should not affect other runs' data."""
        run1 = _make_run("run-001")
        run2 = _make_run("run-002")
        db.insert_run(run1)
        db.insert_run(run2)

        db.insert_trades([_make_trade("trade-001", "run-001")])
        db.insert_trades([_make_trade("trade-002", "run-002")])

        points1 = [EquityCurvePoint(run_id="run-001", date=date(2023, 1, 1),
                                    strategy_value=100_000.0, benchmark_value=100_000.0)]
        points2 = [EquityCurvePoint(run_id="run-002", date=date(2023, 1, 1),
                                    strategy_value=100_000.0, benchmark_value=100_000.0)]
        db.insert_equity_curve(points1)
        db.insert_equity_curve(points2)

        db.delete_run("run-001")

        # run-002 should be unaffected
        assert db.get_run("run-002") is not None
        assert len(db.get_trades("run-002")) == 1
        assert len(db.get_equity_curve("run-002")) == 1

    def test_delete_nonexistent_run_is_noop(self, db):
        """Deleting a run that doesn't exist should not raise."""
        db.delete_run("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# Task 0.2: list_runs with pagination
# ---------------------------------------------------------------------------

class TestListRunsPaginated:
    def _insert_multiple_runs(self, db):
        """Insert 5 runs with different attributes for pagination testing."""
        runs = [
            _make_run("run-001", strategy_name="Momentum", total_return=0.15,
                       sharpe=1.42, max_drawdown=0.08,
                       created_at=datetime(2024, 1, 10)),
            _make_run("run-002", strategy_name="MeanReversion", total_return=0.25,
                       sharpe=2.10, max_drawdown=0.05,
                       created_at=datetime(2024, 1, 20)),
            _make_run("run-003", strategy_name="Momentum", total_return=0.10,
                       sharpe=0.90, max_drawdown=0.12,
                       created_at=datetime(2024, 1, 15)),
            _make_run("run-004", strategy_name="Breakout", total_return=-0.05,
                       sharpe=-0.30, max_drawdown=0.20,
                       created_at=datetime(2024, 1, 25)),
            _make_run("run-005", strategy_name="MeanReversion", total_return=0.30,
                       sharpe=2.50, max_drawdown=0.03,
                       created_at=datetime(2024, 1, 5)),
        ]
        for r in runs:
            db.insert_run(r)

    def test_list_runs_default_pagination(self, db):
        """Default list_runs returns all runs ordered by created_at DESC."""
        self._insert_multiple_runs(db)
        result = db.list_runs()
        assert len(result) == 5
        # Default order: created_at DESC
        assert result[0].run_id == "run-004"  # Jan 25
        assert result[4].run_id == "run-005"  # Jan 5

    def test_list_runs_with_limit_and_offset(self, db):
        """Test limit and offset for pagination."""
        self._insert_multiple_runs(db)
        page1 = db.list_runs(limit=2, offset=0)
        assert len(page1) == 2

        page2 = db.list_runs(limit=2, offset=2)
        assert len(page2) == 2

        page3 = db.list_runs(limit=2, offset=4)
        assert len(page3) == 1

        # No overlap between pages
        all_ids = [r.run_id for r in page1 + page2 + page3]
        assert len(set(all_ids)) == 5

    def test_list_runs_sort_by_total_return_asc(self, db):
        """Sort by total_return ascending."""
        self._insert_multiple_runs(db)
        result = db.list_runs(sort="total_return", order="asc")
        returns = [r.total_return for r in result]
        assert returns == sorted(returns)

    def test_list_runs_sort_by_sharpe_desc(self, db):
        """Sort by sharpe descending."""
        self._insert_multiple_runs(db)
        result = db.list_runs(sort="sharpe", order="desc")
        sharpes = [r.sharpe for r in result]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_list_runs_filter_by_strategy(self, db):
        """Filter by strategy_name."""
        self._insert_multiple_runs(db)
        result = db.list_runs(strategy_filter="Momentum")
        assert len(result) == 2
        assert all(r.strategy_name == "Momentum" for r in result)

    def test_list_runs_filter_and_sort_combined(self, db):
        """Filter by strategy and sort by sharpe ascending."""
        self._insert_multiple_runs(db)
        result = db.list_runs(strategy_filter="MeanReversion",
                              sort="sharpe", order="asc")
        assert len(result) == 2
        assert all(r.strategy_name == "MeanReversion" for r in result)
        sharpes = [r.sharpe for r in result]
        assert sharpes == sorted(sharpes)

    def test_list_runs_invalid_sort_column_raises(self, db):
        """Using an invalid sort column should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid sort column"):
            db.list_runs(sort="DROP TABLE runs; --")

    def test_list_runs_with_pagination(self, db):
        """Full pagination test: filter + sort + limit + offset."""
        self._insert_multiple_runs(db)
        # Get all Momentum runs sorted by total_return DESC, paginated
        page1 = db.list_runs(strategy_filter="Momentum",
                              sort="total_return", order="desc",
                              limit=1, offset=0)
        assert len(page1) == 1
        assert page1[0].total_return == 0.15  # higher return first

        page2 = db.list_runs(strategy_filter="Momentum",
                              sort="total_return", order="desc",
                              limit=1, offset=1)
        assert len(page2) == 1
        assert page2[0].total_return == 0.10
