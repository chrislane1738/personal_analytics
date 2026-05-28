def test_smoke():
    assert 1 + 1 == 2


def test_can_import_analytics():
    from analytics import arb, book, paper  # noqa: F401
