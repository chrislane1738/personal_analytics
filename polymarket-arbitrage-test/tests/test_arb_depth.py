from analytics.arb import PairQuote


def test_pairquote_accepts_depth_fields_default_none():
    pq = PairQuote(pair_name="x")
    assert pq.poly_yes_asks is None
    assert pq.poly_no_asks is None
    assert pq.kalshi_yes_bids is None
    assert pq.kalshi_no_bids is None


def test_pairquote_populates_depth_fields():
    pq = PairQuote(
        pair_name="x",
        poly_yes_asks={0.50: 100.0},
        poly_no_asks={0.48: 200.0},
        kalshi_yes_bids={0.49: 50.0},
        kalshi_no_bids={0.51: 75.0},
    )
    assert pq.poly_yes_asks == {0.50: 100.0}
    assert pq.kalshi_no_bids == {0.51: 75.0}
