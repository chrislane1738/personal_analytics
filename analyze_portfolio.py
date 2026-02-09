import csv


def analyze_portfolio(filepath="mock_portfolio.csv"):
    holdings = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            holdings.append({
                "symbol": row["Symbol"],
                "description": row["Description"],
                "market_value": float(row["MarketValue"]),
                "cost_basis": float(row["CostBasis"]),
            })

    total_value = sum(h["market_value"] for h in holdings)
    total_cost = sum(h["cost_basis"] for h in holdings)
    total_unrealized_gain = total_value - total_cost

    top = max(
        holdings,
        key=lambda h: (h["market_value"] - h["cost_basis"]) / h["cost_basis"],
    )
    top_return_pct = (top["market_value"] - top["cost_basis"]) / top["cost_basis"] * 100

    print(f"Total Portfolio Value:  ${total_value:,.2f}")
    print(f"Total Cost Basis:      ${total_cost:,.2f}")
    print(f"Total Unrealized Gain: ${total_unrealized_gain:,.2f} ({total_unrealized_gain / total_cost * 100:.2f}%)")
    print()
    print(f"Top Performer: {top['symbol']} ({top['description']})")
    print(f"  Return: {top_return_pct:.2f}%  |  Gain: ${top['market_value'] - top['cost_basis']:,.2f}")


if __name__ == "__main__":
    analyze_portfolio()
