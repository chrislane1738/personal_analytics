# Charles Schwab Developer API Setup Guide

## Prerequisites

- You must have an **existing Charles Schwab brokerage account** before applying for API access.
- Developer credentials are **separate** from your brokerage login. You will create a distinct developer account on the Schwab Developer Portal.

---

## Registration Steps

1. Go to [https://developer.schwab.com/](https://developer.schwab.com/) and click **Register** (top-right corner).
2. Create your developer account credentials. These are separate from your brokerage login.
3. Log in to the Developer Portal, then click **Create App** in the Dashboard.
4. Choose your API products:
   - **Market Data Production** -- real-time and historical quotes, price history, option chains, movers, etc.
   - **Accounts and Trading Production** -- account balances, positions, order placement and management.
5. Fill in the required fields:
   - **App Name** -- any descriptive name (e.g., `trading-bot`).
   - **Callback URL** -- use `https://127.0.0.1:8182` for local development.
6. Submit the application.
   - The app will initially show **"Approved -- Pending"**.
   - It typically takes **a few business days** to reach **"Ready for Use"** status.
7. Once approved, your **Client ID** (also called App Key) and **Client Secret** will be available in the app details page.

---

## OAuth 2.0 Authentication Flow

Schwab uses a standard OAuth 2.0 authorization code flow:

1. **Authorization** -- redirect the user's browser to Schwab's authorization URL with your Client ID and callback URL. The user logs in with their **brokerage credentials** and grants access.
2. **Token Exchange** -- after the redirect back to your callback URL, exchange the authorization code for an access token and refresh token.
3. **Token Lifetimes:**
   - **Access Token** expires every **30 minutes**. Refresh it using the refresh token.
   - **Refresh Token** expires every **7 days**. After expiry you must re-authenticate via the browser-based OAuth redirect.
4. Store tokens securely and refresh proactively before expiration.

---

## Python Library: schwab-py

The recommended Python client is [`schwab-py`](https://schwab-py.readthedocs.io/), the successor to the `tda-api` library that was used with TD Ameritrade before the Schwab migration.

```bash
pip install schwab-py
```

`schwab-py` handles token management, OAuth flow helpers, and provides both synchronous and async clients for REST and streaming APIs.

- **Documentation:** [https://schwab-py.readthedocs.io/](https://schwab-py.readthedocs.io/)

---

## Rate Limits

| Channel | Limit |
|---|---|
| REST API | **120 requests per minute** |
| Trade requests | **2--4 per second** |
| WebSocket streaming | Not subject to per-minute REST limits |

Stay well under these limits in production. Implement backoff/retry logic for 429 responses.

---

## Futures Data Limitations

- **Historical intraday bars are NOT available for futures.** The price history endpoint only supports equities and ETFs.
- **Real-time streaming quotes and 1-minute candles ARE available** for futures via the WebSocket **CHART_FUTURES** service.
- Futures symbols use a **forward slash prefix**: `/ES`, `/NQ`, `/MES`, `/MNQ`.

If your strategy requires historical futures bars, you will need an alternative data source (e.g., Databento, Polygon, or direct CME data).

---

## Environment Variables

Add the following to your trading-bot `.env` file:

```
SCHWAB_CLIENT_ID=your_client_id
SCHWAB_CLIENT_SECRET=your_client_secret
```

Never commit these values to version control. The `.env` file should be listed in `.gitignore`.

---

## Key Links

| Resource | URL |
|---|---|
| Developer Portal | [https://developer.schwab.com/](https://developer.schwab.com/) |
| User Registration Guide | [https://developer.schwab.com/user-guides/get-started/user-registration](https://developer.schwab.com/user-guides/get-started/user-registration) |
| OAuth Authentication Guide | [https://developer.schwab.com/user-guides/get-started/authenticate-with-oauth](https://developer.schwab.com/user-guides/get-started/authenticate-with-oauth) |
| schwab-py Documentation | [https://schwab-py.readthedocs.io/](https://schwab-py.readthedocs.io/) |
