"""
Unit Tests for Technical Indicators
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture
def sample_price_series():
    """Generate sample price series for testing indicators"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    prices = 100 + np.cumsum(np.random.randn(200) * 2)
    return pd.Series(prices, index=dates)


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data"""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')

    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1)
    low = close - np.abs(np.random.randn(n) * 1)
    open_price = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000000, 10000000, n)

    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


class TestMovingAverages:
    """Test Moving Average calculations"""

    def test_sma_calculation(self, sample_price_series):
        """Test Simple Moving Average"""
        sma_20 = sample_price_series.rolling(window=20).mean()

        assert len(sma_20) == len(sample_price_series)
        assert sma_20.iloc[19] == sample_price_series.iloc[:20].mean()
        assert pd.isna(sma_20.iloc[0])  # First values should be NaN

    def test_ema_calculation(self, sample_price_series):
        """Test Exponential Moving Average"""
        ema_20 = sample_price_series.ewm(span=20, adjust=False).mean()

        assert len(ema_20) == len(sample_price_series)
        assert not pd.isna(ema_20.iloc[-1])

    def test_moving_average_crossover(self, sample_price_series):
        """Test moving average crossover detection"""
        sma_10 = sample_price_series.rolling(window=10).mean()
        sma_50 = sample_price_series.rolling(window=50).mean()

        # Bullish crossover: SMA10 crosses above SMA50
        crossover = (sma_10 > sma_50) & (sma_10.shift(1) <= sma_50.shift(1))

        # Should be a boolean series
        assert crossover.dtype == bool


class TestRSI:
    """Test Relative Strength Index calculations"""

    def test_rsi_calculation(self, sample_price_series):
        """Test RSI calculation"""
        delta = sample_price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        assert len(rsi) == len(sample_price_series)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_rsi_overbought_oversold(self, sample_price_series):
        """Test RSI overbought/oversold levels"""
        delta = sample_price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        overbought = rsi > 70
        oversold = rsi < 30

        assert overbought.dtype == bool
        assert oversold.dtype == bool


class TestMACD:
    """Test MACD indicator calculations"""

    def test_macd_calculation(self, sample_price_series):
        """Test MACD calculation"""
        ema_12 = sample_price_series.ewm(span=12, adjust=False).mean()
        ema_26 = sample_price_series.ewm(span=26, adjust=False).mean()

        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        assert len(macd_line) == len(sample_price_series)
        assert len(signal_line) == len(sample_price_series)
        assert len(histogram) == len(sample_price_series)

    def test_macd_crossover(self, sample_price_series):
        """Test MACD signal crossover"""
        ema_12 = sample_price_series.ewm(span=12, adjust=False).mean()
        ema_26 = sample_price_series.ewm(span=26, adjust=False).mean()

        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # Bullish crossover
        bullish = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        # Bearish crossover
        bearish = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        assert bullish.dtype == bool
        assert bearish.dtype == bool


class TestBollingerBands:
    """Test Bollinger Bands calculations"""

    def test_bollinger_bands(self, sample_price_series):
        """Test Bollinger Bands calculation"""
        window = 20
        num_std = 2

        sma = sample_price_series.rolling(window=window).mean()
        std = sample_price_series.rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        assert len(upper_band) == len(sample_price_series)
        assert len(lower_band) == len(sample_price_series)

        # Upper band should always be above SMA
        valid_idx = ~pd.isna(upper_band)
        assert (upper_band[valid_idx] >= sma[valid_idx]).all()
        assert (lower_band[valid_idx] <= sma[valid_idx]).all()

    def test_bollinger_squeeze(self, sample_price_series):
        """Test Bollinger Band squeeze detection"""
        window = 20
        sma = sample_price_series.rolling(window=window).mean()
        std = sample_price_series.rolling(window=window).std()

        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)

        band_width = (upper_band - lower_band) / sma

        # Band width should be positive
        valid_width = band_width.dropna()
        assert (valid_width > 0).all()


class TestATR:
    """Test Average True Range calculations"""

    def test_atr_calculation(self, sample_ohlcv_data):
        """Test ATR calculation"""
        high = sample_ohlcv_data['High']
        low = sample_ohlcv_data['Low']
        close = sample_ohlcv_data['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()

        assert len(atr) == len(sample_ohlcv_data)
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()


class TestSupportResistance:
    """Test Support/Resistance calculations"""

    def test_pivot_points(self, sample_ohlcv_data):
        """Test pivot point calculation"""
        high = sample_ohlcv_data['High'].iloc[-1]
        low = sample_ohlcv_data['Low'].iloc[-1]
        close = sample_ohlcv_data['Close'].iloc[-1]

        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        # Resistance should be above pivot, support below
        assert r1 >= pivot
        assert r2 >= pivot
        assert s1 <= pivot
        assert s2 <= pivot

    def test_recent_high_low(self, sample_ohlcv_data):
        """Test finding recent highs and lows"""
        period = 20

        recent_high = sample_ohlcv_data['High'].tail(period).max()
        recent_low = sample_ohlcv_data['Low'].tail(period).min()

        assert recent_high > recent_low


class TestVolumeIndicators:
    """Test Volume-based indicators"""

    def test_obv_calculation(self, sample_ohlcv_data):
        """Test On-Balance Volume"""
        close = sample_ohlcv_data['Close']
        volume = sample_ohlcv_data['Volume']

        obv = (np.sign(close.diff()) * volume).cumsum()

        assert len(obv) == len(sample_ohlcv_data)

    def test_volume_sma(self, sample_ohlcv_data):
        """Test Volume SMA"""
        volume = sample_ohlcv_data['Volume']
        volume_sma = volume.rolling(window=20).mean()

        assert len(volume_sma) == len(sample_ohlcv_data)

        # Compare current volume to average
        volume_ratio = volume / volume_sma
        valid_ratio = volume_ratio.dropna()
        assert (valid_ratio > 0).all()
