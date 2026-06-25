"""forex_bot — a modular Forex CFD trading bot for the Capital.com API.

The package is split into independent layers so strategy logic stays free of
any broker/IO concerns:

    forex_bot.models      - typed in-memory domain models (stdlib dataclasses)
    forex_bot.config      - configuration & secret loading
    forex_bot.indicators  - pure technical-indicator functions
    forex_bot.strategy    - pluggable signal-generating strategies
    forex_bot.risk        - position sizing & loss/exposure limits
    forex_bot.backtest    - candle-replay engine + performance metrics
    forex_bot.execution   - simulated and live order execution
    forex_bot.api         - Capital.com REST + WebSocket clients
    forex_bot.data        - historical data storage & live candle building
"""

__version__ = "0.1.0"
