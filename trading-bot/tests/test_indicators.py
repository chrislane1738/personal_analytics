"""
Tests for technical indicators.
All expected values are hand-calculated.
"""
import math
import pytest

from indicators.technical import SMA, EMA, RSI, MACD, BollingerBands, ATR


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------

class TestSMA:
    def test_warm_up_period(self):
        sma = SMA(period=3)
        assert sma.warm_up_period == 3

    def test_none_before_warm_up(self):
        sma = SMA(period=3)
        sma.update(10)
        assert sma.value is None
        sma.update(11)
        assert sma.value is None

    def test_is_ready_false_before_warm_up(self):
        sma = SMA(period=3)
        sma.update(10)
        assert sma.is_ready is False

    def test_is_ready_true_after_warm_up(self):
        sma = SMA(period=3)
        for p in [10, 11, 12]:
            sma.update(p)
        assert sma.is_ready is True

    def test_basic_rolling_window(self):
        # prices [10, 11, 12, 13, 14], period=3
        sma = SMA(period=3)
        prices = [10, 11, 12, 13, 14]
        expected = [None, None, 11.0, 12.0, 13.0]
        for price, exp in zip(prices, expected):
            sma.update(price)
            assert sma.value == exp

    def test_period_1(self):
        sma = SMA(period=1)
        sma.update(42.0)
        assert sma.value == 42.0
        sma.update(7.5)
        assert sma.value == 7.5

    def test_equal_prices(self):
        sma = SMA(period=5)
        for _ in range(5):
            sma.update(100.0)
        assert sma.value == 100.0

    def test_fractional_result(self):
        sma = SMA(period=3)
        for p in [1, 2, 4]:
            sma.update(p)
        # (1+2+4)/3 = 7/3 ≈ 2.3333...
        assert abs(sma.value - 7 / 3) < 1e-9


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_warm_up_period(self):
        ema = EMA(period=5)
        assert ema.warm_up_period == 5

    def test_none_before_warm_up(self):
        ema = EMA(period=3)
        ema.update(10)
        assert ema.value is None
        ema.update(11)
        assert ema.value is None

    def test_is_ready_false_before_warm_up(self):
        ema = EMA(period=3)
        ema.update(10)
        assert ema.is_ready is False

    def test_is_ready_true_after_warm_up(self):
        ema = EMA(period=3)
        for p in [10, 11, 12]:
            ema.update(p)
        assert ema.is_ready is True

    def test_first_value_is_sma(self):
        # EMA of period 3 with prices [10, 20, 30]:
        # first EMA = SMA(10,20,30) = 20.0
        ema = EMA(period=3)
        for p in [10, 20, 30]:
            ema.update(p)
        assert ema.value == pytest.approx(20.0)

    def test_subsequent_ema(self):
        # period=3, mult = 2/(3+1) = 0.5
        # prices: 10, 20, 30, 40
        # After 3 prices: EMA = SMA(10,20,30) = 20.0
        # After 4th price (40): EMA = 40*0.5 + 20.0*0.5 = 30.0
        ema = EMA(period=3)
        mult = 2 / (3 + 1)  # 0.5
        for p in [10, 20, 30]:
            ema.update(p)
        ema.update(40)
        expected = 40 * mult + 20.0 * (1 - mult)
        assert ema.value == pytest.approx(expected)

    def test_five_steps(self):
        # period=3, mult=0.5
        # prices: 10, 20, 30, 40, 50
        # EMA_3 = 20.0
        # EMA_4 = 40*0.5 + 20.0*0.5 = 30.0
        # EMA_5 = 50*0.5 + 30.0*0.5 = 40.0
        ema = EMA(period=3)
        for p in [10, 20, 30, 40, 50]:
            ema.update(p)
        assert ema.value == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_warm_up_period(self):
        rsi = RSI(period=14)
        assert rsi.warm_up_period == 15  # period + 1

    def test_none_before_warm_up(self):
        rsi = RSI(period=14)
        for i in range(14):
            rsi.update(float(i))
            assert rsi.value is None

    def test_is_ready_after_warm_up(self):
        rsi = RSI(period=14)
        for i in range(15):
            rsi.update(float(i + 10))
        assert rsi.is_ready is True

    def test_all_gains_gives_100(self):
        # All prices rising → avg_loss = 0 → RSI = 100
        rsi = RSI(period=5)
        for i in range(6):  # period+1 prices
            rsi.update(float(i * 10))
        assert rsi.value == pytest.approx(100.0)

    def test_all_losses_gives_0(self):
        # All prices falling → avg_gain = 0 → RSI = 0
        rsi = RSI(period=5)
        for i in range(6):
            rsi.update(float(100 - i * 10))
        assert rsi.value == pytest.approx(0.0)

    def test_rsi_hand_calculation(self):
        # period=3, warm_up = 4 prices needed
        # Prices: 10, 12, 11, 13
        # Changes: +2, -1, +2
        # Initial avg_gain = (2 + 0 + 2) / 3 = 4/3
        # Initial avg_loss = (0 + 1 + 0) / 3 = 1/3
        # RS = (4/3) / (1/3) = 4
        # RSI = 100 - 100/(1+4) = 100 - 20 = 80.0
        rsi = RSI(period=3)
        for p in [10, 12, 11, 13]:
            rsi.update(float(p))
        assert rsi.value == pytest.approx(80.0)

    def test_wilders_smoothing(self):
        # period=3
        # Prices: 10, 12, 11, 13, 15
        # After 4 prices (rsi is ready):
        #   avg_gain = 4/3, avg_loss = 1/3  (as above)
        # 5th price: change = +2 (gain=2, loss=0)
        #   avg_gain = (4/3 * 2 + 2) / 3 = (8/3 + 6/3) / 3 = (14/3) / 3 = 14/9
        #   avg_loss = (1/3 * 2 + 0) / 3 = (2/3) / 3 = 2/9
        #   RS = (14/9) / (2/9) = 7
        #   RSI = 100 - 100/8 = 87.5
        rsi = RSI(period=3)
        for p in [10, 12, 11, 13, 15]:
            rsi.update(float(p))
        assert rsi.value == pytest.approx(87.5)

    def test_rsi_bounds(self):
        rsi = RSI(period=5)
        prices = [44, 46, 44, 48, 42, 50]
        for p in prices:
            rsi.update(float(p))
        assert 0.0 <= rsi.value <= 100.0


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_warm_up_period(self):
        macd = MACD(fast=12, slow=26, signal=9)
        # slow + signal - 1 = 26 + 9 - 1 = 34
        assert macd.warm_up_period == 34

    def test_none_before_warm_up(self):
        macd = MACD(fast=3, slow=5, signal=3)
        # warm_up = 5 + 3 - 1 = 7
        for i in range(6):
            macd.update(float(i + 10))
        assert macd.value is None
        assert macd.signal_line is None
        assert macd.histogram is None

    def test_is_ready_after_warm_up(self):
        macd = MACD(fast=3, slow=5, signal=3)
        # warm_up = 7
        for i in range(7):
            macd.update(float(i + 10))
        assert macd.is_ready is True
        assert macd.signal_line is not None
        assert macd.histogram is not None

    def test_histogram_equals_macd_minus_signal(self):
        macd = MACD(fast=3, slow=5, signal=3)
        for i in range(20):
            macd.update(float(i * 2 + 10))
        assert macd.histogram == pytest.approx(macd.value - macd.signal_line)

    def test_rising_prices_positive_macd(self):
        # Steadily rising prices → fast EMA > slow EMA → MACD > 0
        macd = MACD(fast=3, slow=5, signal=3)
        for i in range(30):
            macd.update(float(i + 1))
        assert macd.value > 0

    def test_flat_prices_zero_macd(self):
        # Perfectly flat prices → all EMAs equal → MACD ≈ 0
        macd = MACD(fast=3, slow=5, signal=3)
        for _ in range(30):
            macd.update(100.0)
        assert macd.value == pytest.approx(0.0, abs=1e-9)
        assert macd.signal_line == pytest.approx(0.0, abs=1e-9)
        assert macd.histogram == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# BollingerBands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_warm_up_period(self):
        bb = BollingerBands(period=20, std_dev=2.0)
        assert bb.warm_up_period == 20

    def test_none_before_warm_up(self):
        bb = BollingerBands(period=5)
        for i in range(4):
            bb.update(float(i + 10))
        assert bb.value is None
        assert bb.upper is None
        assert bb.lower is None
        assert bb.middle is None

    def test_is_ready_after_warm_up(self):
        bb = BollingerBands(period=5)
        for i in range(5):
            bb.update(float(i + 10))
        assert bb.is_ready is True

    def test_equal_prices_zero_width(self):
        # 20 identical prices → std_dev = 0 → upper = lower = middle
        bb = BollingerBands(period=20, std_dev=2.0)
        for _ in range(20):
            bb.update(100.0)
        assert bb.upper == pytest.approx(100.0)
        assert bb.lower == pytest.approx(100.0)
        assert bb.middle == pytest.approx(100.0)
        assert bb.value == pytest.approx(100.0)

    def test_middle_equals_sma(self):
        # Middle band should equal SMA of period
        bb = BollingerBands(period=3, std_dev=2.0)
        prices = [10.0, 20.0, 30.0]
        for p in prices:
            bb.update(p)
        # SMA(10,20,30) = 20.0
        assert bb.middle == pytest.approx(20.0)
        assert bb.value == pytest.approx(20.0)

    def test_bands_hand_calculation(self):
        # period=3, std_dev=2, prices=[10, 20, 30]
        # mean = 20.0
        # population std (ddof=0): sqrt(((10-20)^2+(20-20)^2+(30-20)^2)/3)
        #   = sqrt((100+0+100)/3) = sqrt(200/3) ≈ 8.16496...
        # upper = 20 + 2*sqrt(200/3)
        # lower = 20 - 2*sqrt(200/3)
        bb = BollingerBands(period=3, std_dev=2.0)
        for p in [10.0, 20.0, 30.0]:
            bb.update(p)
        std = math.sqrt(200 / 3)
        assert bb.upper == pytest.approx(20.0 + 2 * std)
        assert bb.lower == pytest.approx(20.0 - 2 * std)

    def test_upper_above_lower(self):
        bb = BollingerBands(period=5, std_dev=2.0)
        prices = [10, 12, 9, 13, 11]
        for p in prices:
            bb.update(float(p))
        assert bb.upper > bb.lower

    def test_rolling_update(self):
        # After 4 prices with period=3, window shifts: [12, 9, 13]
        bb = BollingerBands(period=3, std_dev=2.0)
        for p in [10.0, 12.0, 9.0, 13.0]:
            bb.update(p)
        # SMA of [12, 9, 13] = 34/3 ≈ 11.333...
        assert bb.middle == pytest.approx(34 / 3)

    def test_std_dev_multiplier(self):
        # 1x std_dev vs 2x std_dev: width should double
        bb1 = BollingerBands(period=5, std_dev=1.0)
        bb2 = BollingerBands(period=5, std_dev=2.0)
        prices = [10, 12, 9, 13, 11]
        for p in prices:
            bb1.update(float(p))
            bb2.update(float(p))
        width1 = bb1.upper - bb1.lower
        width2 = bb2.upper - bb2.lower
        assert width2 == pytest.approx(width1 * 2)


# ---------------------------------------------------------------------------
# ATR — Average True Range
# ---------------------------------------------------------------------------

class TestATR:
    def test_warm_up_period(self):
        atr = ATR(period=14)
        assert atr.warm_up_period == 14

    def test_warm_up_period_custom(self):
        atr = ATR(period=3)
        assert atr.warm_up_period == 3

    def test_none_before_warm_up(self):
        # ATR(3) needs 3 bars before producing a value
        atr = ATR(period=3)
        # After 1st bar: still None
        atr.update(high=12.0, low=10.0, close=11.0)
        assert atr.value is None
        # After 2nd bar: still None
        atr.update(high=13.0, low=10.5, close=12.0)
        assert atr.value is None

    def test_is_ready_false_before_warm_up(self):
        atr = ATR(period=3)
        atr.update(high=12.0, low=10.0, close=11.0)
        assert atr.is_ready is False

    def test_is_ready_true_after_warm_up(self):
        atr = ATR(period=3)
        atr.update(high=12.0, low=10.0, close=11.0)
        atr.update(high=13.0, low=10.5, close=12.0)
        atr.update(high=14.0, low=11.0, close=13.0)
        assert atr.is_ready is True

    def test_hand_calculated_atr3(self):
        # ATR(3) with bars: H=12/L=10/C=11, H=13/L=10.5/C=12, H=14/L=11/C=13
        #
        # Bar 1 (no prev close): TR = H - L = 12 - 10 = 2.0
        # Bar 2 (prev_close=11): TR = max(13-10.5, |13-11|, |10.5-11|)
        #                            = max(2.5, 2.0, 0.5) = 2.5
        # Bar 3 (prev_close=12): TR = max(14-11, |14-12|, |11-12|)
        #                            = max(3.0, 2.0, 1.0) = 3.0
        # Initial ATR = SMA of first 3 TRs = (2.0 + 2.5 + 3.0) / 3 = 2.5
        atr = ATR(period=3)
        atr.update(high=12.0, low=10.0, close=11.0)
        atr.update(high=13.0, low=10.5, close=12.0)
        atr.update(high=14.0, low=11.0, close=13.0)
        assert atr.value == pytest.approx(2.5)

    def test_wilder_smoothing(self):
        # Continue from above: ATR after 3 bars = 2.5
        # Bar 4 (prev_close=13): H=15/L=12/C=14
        #   TR = max(15-12, |15-13|, |12-13|) = max(3.0, 2.0, 1.0) = 3.0
        #   ATR = (2.5 * 2 + 3.0) / 3 = (5.0 + 3.0) / 3 = 8.0 / 3 ≈ 2.6667
        atr = ATR(period=3)
        atr.update(high=12.0, low=10.0, close=11.0)
        atr.update(high=13.0, low=10.5, close=12.0)
        atr.update(high=14.0, low=11.0, close=13.0)
        atr.update(high=15.0, low=12.0, close=14.0)
        assert atr.value == pytest.approx(8.0 / 3)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            ATR(period=0)

    def test_invalid_period_negative_raises(self):
        with pytest.raises(ValueError):
            ATR(period=-5)

    def test_period_1(self):
        # ATR(1) should be ready after the first bar; TR = H - L (no prev close)
        atr = ATR(period=1)
        atr.update(high=10.0, low=8.0, close=9.0)
        assert atr.value == pytest.approx(2.0)

    def test_atr_always_non_negative(self):
        atr = ATR(period=3)
        bars = [
            (12.0, 10.0, 11.0),
            (13.0, 10.5, 12.0),
            (14.0, 11.0, 13.0),
            (11.0, 9.0, 10.0),   # price drops
            (10.0, 8.0, 9.0),
        ]
        for h, l, c in bars:
            atr.update(high=h, low=l, close=c)
            if atr.value is not None:
                assert atr.value >= 0.0
