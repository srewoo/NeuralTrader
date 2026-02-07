"""
Candlestick Chart Generator
Generates candlestick chart images with technical overlays for LLM visual analysis.
Uses mplfinance for rendering to in-memory PNG bytes.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — MUST come before other matplotlib imports

import io
import base64
import logging
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False
    mpf = None

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CandlestickChartGenerator:
    """
    Generates candlestick chart images using mplfinance.
    Charts are rendered to in-memory PNG bytes for LLM consumption.
    """

    FIGSIZE = (14, 8)
    DPI = 100

    def generate_daily_chart(
        self,
        ohlcv_data: pd.DataFrame,
        symbol: str,
        indicators: Optional[Dict[str, Any]] = None
    ) -> Optional[bytes]:
        """
        Generate a daily candlestick chart (1-month) with SMA, Bollinger Bands, and volume.

        Args:
            ohlcv_data: DataFrame with Open, High, Low, Close, Volume columns and DatetimeIndex
            symbol: Stock symbol for title
            indicators: Dict with sma_20, sma_50, bb_upper, bb_lower values

        Returns:
            PNG image as bytes, or None if generation fails
        """
        if not MPLFINANCE_AVAILABLE:
            logger.warning("mplfinance not installed — skipping chart generation")
            return None

        try:
            data = self._prepare_dataframe(ohlcv_data)
            if data is None or len(data) < 5:
                return None

            addplots = []

            # Add SMA overlays computed from the data
            if len(data) >= 20:
                sma20 = data['Close'].rolling(20).mean()
                if sma20.notna().sum() >= 2:
                    addplots.append(mpf.make_addplot(sma20, color='#00bfff', width=1.2, label='SMA 20'))
            if len(data) >= 50:
                sma50 = data['Close'].rolling(50).mean()
                if sma50.notna().sum() >= 2:
                    addplots.append(mpf.make_addplot(sma50, color='#ff6347', width=1.2, label='SMA 50'))

            # Bollinger Bands
            if len(data) >= 20:
                bb_mid = data['Close'].rolling(20).mean()
                bb_std = data['Close'].rolling(20).std()
                bb_upper = bb_mid + 2 * bb_std
                bb_lower = bb_mid - 2 * bb_std
                if bb_upper.notna().sum() >= 2:
                    addplots.append(mpf.make_addplot(bb_upper, color='#888888', width=0.8, linestyle='--'))
                    addplots.append(mpf.make_addplot(bb_lower, color='#888888', width=0.8, linestyle='--'))

            buf = io.BytesIO()

            mc = mpf.make_marketcolors(
                up='#26a69a', down='#ef5350',
                edge='inherit',
                wick='inherit',
                volume={'up': '#26a69a', 'down': '#ef5350'},
            )
            style = mpf.make_mpf_style(
                marketcolors=mc,
                base_mpf_style='nightclouds',
                gridstyle='-',
                gridcolor='#333333',
            )

            plot_kwargs = dict(
                type='candle',
                volume=True,
                style=style,
                title=f'\n{symbol} — Daily Chart (1 Month)',
                figsize=self.FIGSIZE,
                returnfig=True,
            )
            if addplots:
                plot_kwargs['addplot'] = addplots

            fig, axes = mpf.plot(data, **plot_kwargs)

            fig.savefig(buf, format='png', dpi=self.DPI, bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)

            buf.seek(0)
            image_bytes = buf.read()
            logger.info(f"Generated daily chart for {symbol} ({len(image_bytes)} bytes)")
            return image_bytes

        except Exception as e:
            logger.warning(f"Daily chart generation failed for {symbol}: {e}")
            return None

    def generate_weekly_chart(
        self,
        ohlcv_data: pd.DataFrame,
        symbol: str,
        indicators: Optional[Dict[str, Any]] = None
    ) -> Optional[bytes]:
        """
        Generate a weekly candlestick chart (6-month) with SMA and volume.
        Resamples daily data to weekly OHLCV.

        Args:
            ohlcv_data: 6-month daily DataFrame
            symbol: Stock symbol for title
            indicators: Dict with sma_50, sma_200 values

        Returns:
            PNG image as bytes, or None if generation fails
        """
        if not MPLFINANCE_AVAILABLE:
            logger.warning("mplfinance not installed — skipping chart generation")
            return None

        try:
            data = self._prepare_dataframe(ohlcv_data)
            if data is None or len(data) < 10:
                return None

            # Resample to weekly
            weekly = self._resample_to_weekly(data)
            if weekly is None or len(weekly) < 4:
                return None

            addplots = []

            # SMA overlays on weekly data (only if enough non-NaN values)
            if len(weekly) >= 10:
                sma10w = weekly['Close'].rolling(10).mean()
                if sma10w.notna().sum() >= 2:
                    addplots.append(mpf.make_addplot(sma10w, color='#00bfff', width=1.2, label='SMA 10W'))
            if len(weekly) >= 20:
                sma20w = weekly['Close'].rolling(20).mean()
                if sma20w.notna().sum() >= 2:
                    addplots.append(mpf.make_addplot(sma20w, color='#ff6347', width=1.2, label='SMA 20W'))

            buf = io.BytesIO()

            mc = mpf.make_marketcolors(
                up='#26a69a', down='#ef5350',
                edge='inherit',
                wick='inherit',
                volume={'up': '#26a69a', 'down': '#ef5350'},
            )
            style = mpf.make_mpf_style(
                marketcolors=mc,
                base_mpf_style='nightclouds',
                gridstyle='-',
                gridcolor='#333333',
            )

            plot_kwargs = dict(
                type='candle',
                volume=True,
                style=style,
                title=f'\n{symbol} — Weekly Chart (6 Months)',
                figsize=self.FIGSIZE,
                returnfig=True,
            )
            if addplots:
                plot_kwargs['addplot'] = addplots

            fig, axes = mpf.plot(weekly, **plot_kwargs)

            fig.savefig(buf, format='png', dpi=self.DPI, bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)

            buf.seek(0)
            image_bytes = buf.read()
            logger.info(f"Generated weekly chart for {symbol} ({len(image_bytes)} bytes)")
            return image_bytes

        except Exception as e:
            logger.warning(f"Weekly chart generation failed for {symbol}: {e}")
            return None

    def _prepare_dataframe(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Ensure DataFrame has proper format for mplfinance."""
        try:
            df = data.copy()

            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                elif 'Datetime' in df.columns:
                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                    df.set_index('Datetime', inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)

            # Remove timezone info if present (mplfinance can struggle with tz)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Ensure required columns exist
            required = ['Open', 'High', 'Low', 'Close']
            for col in required:
                if col not in df.columns:
                    return None

            # Drop rows with NaN in OHLC
            df.dropna(subset=required, inplace=True)

            # Ensure Volume exists (fill with 0 if missing)
            if 'Volume' not in df.columns:
                df['Volume'] = 0

            return df

        except Exception as e:
            logger.warning(f"DataFrame preparation failed: {e}")
            return None

    def _resample_to_weekly(self, daily_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Resample daily OHLCV to weekly using proper aggregation."""
        try:
            weekly = daily_data.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            weekly.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            return weekly
        except Exception as e:
            logger.warning(f"Weekly resampling failed: {e}")
            return None

    def generate_4h_chart(
        self,
        ohlcv_data: pd.DataFrame,
        symbol: str,
        indicators: Optional[Dict[str, Any]] = None
    ) -> Optional[bytes]:
        """
        Generate a 4-hour candlestick chart (5-day) with SMA overlays.
        Resamples hourly data to 4-hour OHLCV.

        Args:
            ohlcv_data: 5-day hourly DataFrame
            symbol: Stock symbol for title
            indicators: Optional indicator dict

        Returns:
            PNG image as bytes, or None if generation fails
        """
        if not MPLFINANCE_AVAILABLE:
            logger.warning("mplfinance not installed — skipping chart generation")
            return None

        try:
            data = self._prepare_dataframe(ohlcv_data)
            if data is None or len(data) < 8:
                return None

            # Resample to 4-hour candles
            data_4h = self._resample_to_4h(data)
            if data_4h is None or len(data_4h) < 4:
                return None

            addplots = []

            # SMA overlays on 4h data
            if len(data_4h) >= 5:
                sma5 = data_4h['Close'].rolling(5).mean()
                if sma5.notna().sum() >= 2:
                    addplots.append(mpf.make_addplot(sma5, color='#00bfff', width=1.2, label='SMA 5'))
            if len(data_4h) >= 13:
                sma13 = data_4h['Close'].rolling(13).mean()
                if sma13.notna().sum() >= 2:
                    addplots.append(mpf.make_addplot(sma13, color='#ff6347', width=1.2, label='SMA 13'))

            buf = io.BytesIO()

            mc = mpf.make_marketcolors(
                up='#26a69a', down='#ef5350',
                edge='inherit',
                wick='inherit',
                volume={'up': '#26a69a', 'down': '#ef5350'},
            )
            style = mpf.make_mpf_style(
                marketcolors=mc,
                base_mpf_style='nightclouds',
                gridstyle='-',
                gridcolor='#333333',
            )

            plot_kwargs = dict(
                type='candle',
                volume=True,
                style=style,
                title=f'\n{symbol} — 4-Hour Chart (5 Days)',
                figsize=self.FIGSIZE,
                returnfig=True,
            )
            if addplots:
                plot_kwargs['addplot'] = addplots

            fig, axes = mpf.plot(data_4h, **plot_kwargs)

            fig.savefig(buf, format='png', dpi=self.DPI, bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)

            buf.seek(0)
            image_bytes = buf.read()
            logger.info(f"Generated 4hr chart for {symbol} ({len(image_bytes)} bytes)")
            return image_bytes

        except Exception as e:
            logger.warning(f"4hr chart generation failed for {symbol}: {e}")
            return None

    def _resample_to_4h(self, hourly_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Resample hourly OHLCV to 4-hour using proper aggregation."""
        try:
            data_4h = hourly_data.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            data_4h.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            return data_4h
        except Exception as e:
            logger.warning(f"4hr resampling failed: {e}")
            return None

    @staticmethod
    def image_to_base64(image_bytes: bytes) -> str:
        """Convert image bytes to base64 string."""
        return base64.b64encode(image_bytes).decode('utf-8')


# Singleton
_chart_generator: Optional[CandlestickChartGenerator] = None


def get_chart_generator() -> CandlestickChartGenerator:
    """Get singleton chart generator instance."""
    global _chart_generator
    if _chart_generator is None:
        _chart_generator = CandlestickChartGenerator()
    return _chart_generator
