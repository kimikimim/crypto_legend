"""Engine-specific exceptions."""


class EngineError(Exception):
    """Base class for engine failures."""


class InsufficientDataError(EngineError):
    """Raised when the exchange returned too few candles to analyze safely."""


class DataFetchError(EngineError):
    """Raised when data could not be fetched after all retries."""


class UnsupportedSymbolError(EngineError):
    """Raised for any symbol outside the hardcoded BTC/ETH/SOL whitelist."""
