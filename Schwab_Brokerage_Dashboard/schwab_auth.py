import os
import json
from dotenv import load_dotenv
import schwab

load_dotenv()

APP_KEY = os.getenv("SCHWAB_APP_KEY")
SECRET_KEY = os.getenv("SCHWAB_SECRET_KEY")
CALLBACK_URL = os.getenv("SCHWAB_CALLBACK_URL")
TOKEN_PATH = "token.json"


def authenticate():
    """Authenticate with Schwab API. Uses saved token if available, otherwise
    runs the browser-based login flow."""
    try:
        client = schwab.auth.client_from_token_file(
            TOKEN_PATH, APP_KEY, SECRET_KEY
        )
        print("Loaded existing token from", TOKEN_PATH)
    except FileNotFoundError:
        print("No saved token found. Starting login flow...")
        print("A browser window will open — log in to your Schwab account.")
        print("After the SSL warning, click through to complete the redirect.\n")
        client = schwab.auth.client_from_login_flow(
            APP_KEY, SECRET_KEY, CALLBACK_URL, TOKEN_PATH,
            interactive=False
        )
        print("Token saved to", TOKEN_PATH)
    return client


def print_account_balances(client):
    """Fetch and print account balances and positions."""
    resp = client.get_accounts(fields=[client.Account.Fields.POSITIONS])
    resp.raise_for_status()
    accounts = resp.json()

    for acct in accounts:
        info = acct["securitiesAccount"]
        acct_type = info.get("type", "N/A")
        acct_id = info.get("accountNumber", "N/A")
        balances = info.get("currentBalances", info.get("initialBalances", {}))

        print(f"\n{'='*50}")
        print(f"Account: {acct_id}  ({acct_type})")
        print(f"{'='*50}")
        print(f"  Liquidation Value:  ${balances.get('liquidationValue', 0):,.2f}")
        print(f"  Cash Balance:       ${balances.get('cashBalance', 0):,.2f}")
        print(f"  Available Funds:    ${balances.get('availableFunds', balances.get('cashAvailableForTrading', 0)):,.2f}")
        print(f"  Buying Power:       ${balances.get('buyingPower', 0):,.2f}")

        positions = info.get("positions", [])
        if positions:
            symbols = [p.get("instrument", {}).get("symbol") for p in positions
                       if p.get("instrument", {}).get("symbol")]
            pe_map = {}
            if symbols:
                quote_resp = client.get_quotes(symbols,
                    fields=[client.Quote.Fields.FUNDAMENTAL])
                quote_resp.raise_for_status()
                quotes = quote_resp.json()
                for sym, data in quotes.items():
                    fund = data.get("fundamental", {})
                    pe_map[sym] = fund.get("peRatio", None)

            print(f"\n  {'Symbol':<10} {'Qty':>8} {'Market Value':>14} {'P/E':>10}")
            print(f"  {'-'*10} {'-'*8} {'-'*14} {'-'*10}")
            for pos in positions:
                symbol = pos.get("instrument", {}).get("symbol", "N/A")
                qty = pos.get("longQuantity", 0) - pos.get("shortQuantity", 0)
                market_value = pos.get("marketValue", 0)
                pe = pe_map.get(symbol)
                pe_str = f"{pe:>10.2f}" if pe else "       N/A"
                print(f"  {symbol:<10} {qty:>8.2f} ${market_value:>13,.2f} {pe_str}")


if __name__ == "__main__":
    client = authenticate()
    print_account_balances(client)
