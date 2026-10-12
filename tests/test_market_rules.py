import pytest
from ge_predictor.market_rules import calculate_ge_tax, calculate_net_margin

def test_ge_tax_calculation():
    # sell_price = 49 -> 0 tax
    assert calculate_ge_tax(49) == 0

    # sell_price = 50 -> 1 tax
    assert calculate_ge_tax(50) == 1

    # sell_price = 100 -> 2 tax
    assert calculate_ge_tax(100) == 2

    # sell_price = 10_000_000 -> 200_000 tax
    assert calculate_ge_tax(10_000_000) == 200_000

    # sell_price >= 250_000_000 -> 5_000_000 tax cap
    assert calculate_ge_tax(250_000_000) == 5_000_000
    assert calculate_ge_tax(500_000_000) == 5_000_000

    # is_exempt = True -> 0 tax
    assert calculate_ge_tax(100_000_000, is_exempt=True) == 0

def test_net_margin_calculation():
    # Buy at 100, sell at 200 (tax = 4) -> net margin = 200 - 4 - 100 = 96
    assert calculate_net_margin(100, 200) == 96

    # Exempt item: Buy at 100, sell at 200 (tax = 0) -> net margin = 100
    assert calculate_net_margin(100, 200, is_exempt=True) == 100
