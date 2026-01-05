"""
Advanced Technical Indicators Module
Includes additional indicators and multi-timeframe analysis.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdvancedIndicators:
    """
    Advanced technical indicators including:
    - Standard indicators (RSI, MACD, SMA, BB, etc.)
    - Additional indicators (Ichimoku, VWAP, Williams %R, Parabolic SAR)
    - Multi-timeframe analysis
    - Volume profile
    """

    def __init__(self):
        pass

    def calculate_all_indicators(
        self,
        df: pd.DataFrame,
        include_advanced: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate all technical indicators.

        Args:
            df: DataFrame with OHLCV data
            include_advanced: Include advanced indicators (Ichimoku, VWAP, etc.)

        Returns:
            Dict with all calculated indicators
        """
        if df.empty or len(df) < 50:
            return {}

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        open_price = df['Open']

        indicators = {}

        # Standard Momentum Indicators
        indicators.update(self._calculate_momentum_indicators(high, low, close, volume))

        # Standard Trend Indicators
        indicators.update(self._calculate_trend_indicators(close))

        # Volatility Indicators
        indicators.update(self._calculate_volatility_indicators(high, low, close))

        # Volume Indicators
        indicators.update(self._calculate_volume_indicators(close, volume))

        # Advanced Indicators
        if include_advanced:
            indicators.update(self._calculate_advanced_indicators(high, low, close, volume))
            indicators.update(self._calculate_ichimoku(high, low, close))

        # Multi-timeframe analysis
        indicators['multi_timeframe'] = self._calculate_multi_timeframe(df)

        return indicators

    def _calculate_momentum_indicators(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> Dict[str, Any]:
        """Calculate momentum indicators"""
        result = {}

        # RSI (14)
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        result['rsi'] = self._safe_float(rsi.iloc[-1])
        result['rsi_prev'] = self._safe_float(rsi.iloc[-2]) if len(rsi) > 1 else None

        # RSI (7) for short-term
        rsi_7 = ta.momentum.RSIIndicator(close, window=7).rsi()
        result['rsi_7'] = self._safe_float(rsi_7.iloc[-1])

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        result['stoch_k'] = self._safe_float(stoch.stoch().iloc[-1])
        result['stoch_d'] = self._safe_float(stoch.stoch_signal().iloc[-1])

        # Williams %R
        williams_r = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
        result['williams_r'] = self._safe_float(williams_r.iloc[-1])

        # Rate of Change (ROC)
        roc = ta.momentum.ROCIndicator(close, window=12).roc()
        result['roc'] = self._safe_float(roc.iloc[-1])

        # CCI
        cci = ta.trend.CCIIndicator(high, low, close, window=20).cci()
        result['cci'] = self._safe_float(cci.iloc[-1])

        # Money Flow Index
        mfi = ta.volume.MFIIndicator(high, low, close, volume, window=14)
        # MFI requires volume, use close as proxy if needed
        try:
            mfi_indicator = ta.volume.MFIIndicator(high, low, close, pd.Series([1000000] * len(close)), window=14)
            result['mfi'] = self._safe_float(mfi_indicator.money_flow_index().iloc[-1])
        except:
            result['mfi'] = None

        # Triple Exponential Average (TRIX)
        try:
            trix = ta.trend.TRIXIndicator(close, window=15)
            result['trix'] = self._safe_float(trix.trix().iloc[-1])
        except:
            result['trix'] = None

        # Mass Index
        try:
            mi = ta.trend.MassIndex(high, low, window_fast=9, window_slow=25)
            result['mass_index'] = self._safe_float(mi.mass_index().iloc[-1])
        except:
            result['mass_index'] = None

        # True Strength Index (TSI)
        try:
            tsi = ta.momentum.TSIIndicator(close, window_slow=25, window_fast=13)
            result['tsi'] = self._safe_float(tsi.tsi().iloc[-1])
        except:
            result['tsi'] = None

        # Detrended Price Oscillator (DPO)
        try:
            dpo = ta.trend.DPOIndicator(close, window=20)
            result['dpo'] = self._safe_float(dpo.dpo().iloc[-1])
        except:
            result['dpo'] = None

        # Know Sure Thing (KST) Oscillator
        try:
            kst = ta.trend.KSTIndicator(close, roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15)
            result['kst'] = self._safe_float(kst.kst().iloc[-1])
            result['kst_signal'] = self._safe_float(kst.kst_sig().iloc[-1])
            result['kst_diff'] = self._safe_float(kst.kst_diff().iloc[-1])
        except:
            result['kst'] = None
            result['kst_signal'] = None
            result['kst_diff'] = None

        # Schaff Trend Cycle (STC)
        try:
            stc = ta.trend.STCIndicator(close, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3)
            result['stc'] = self._safe_float(stc.stc().iloc[-1])
        except:
            result['stc'] = None

        return result

    def _calculate_trend_indicators(self, close: pd.Series) -> Dict[str, Any]:
        """Calculate trend indicators"""
        result = {}

        # MACD
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        result['macd'] = self._safe_float(macd.macd().iloc[-1])
        result['macd_signal'] = self._safe_float(macd.macd_signal().iloc[-1])
        result['macd_histogram'] = self._safe_float(macd.macd_diff().iloc[-1])
        result['macd_histogram_prev'] = self._safe_float(macd.macd_diff().iloc[-2]) if len(macd.macd_diff()) > 1 else None

        # SMAs
        for period in [10, 20, 50, 100, 200]:
            if len(close) >= period:
                sma = ta.trend.SMAIndicator(close, window=period).sma_indicator()
                result[f'sma_{period}'] = self._safe_float(sma.iloc[-1])
            else:
                result[f'sma_{period}'] = None

        # EMAs
        for period in [9, 21, 50, 200]:
            if len(close) >= period:
                ema = ta.trend.EMAIndicator(close, window=period).ema_indicator()
                result[f'ema_{period}'] = self._safe_float(ema.iloc[-1])
            else:
                result[f'ema_{period}'] = None

        # WMA (Weighted Moving Average)
        for period in [10, 20, 50]:
            if len(close) >= period:
                try:
                    wma = ta.trend.WMAIndicator(close, window=period).wma()
                    result[f'wma_{period}'] = self._safe_float(wma.iloc[-1])
                except:
                    result[f'wma_{period}'] = None
            else:
                result[f'wma_{period}'] = None

        # Hull Moving Average (HMA)
        try:
            if len(close) >= 16:
                wma_half = ta.trend.WMAIndicator(close, window=8).wma()
                wma_full = ta.trend.WMAIndicator(close, window=16).wma()
                raw_hma = 2 * wma_half - wma_full
                hma = ta.trend.WMAIndicator(raw_hma.dropna(), window=4).wma()
                result['hma'] = self._safe_float(hma.iloc[-1])
        except:
            result['hma'] = None

        # Vortex Indicator
        try:
            vi = ta.trend.VortexIndicator(close.shift(-1).fillna(close), close.shift(1).fillna(close), close, window=14)
            result['vi_pos'] = self._safe_float(vi.vortex_indicator_pos().iloc[-1])
            result['vi_neg'] = self._safe_float(vi.vortex_indicator_neg().iloc[-1])
            result['vi_diff'] = self._safe_float((vi.vortex_indicator_pos() - vi.vortex_indicator_neg()).iloc[-1])
        except:
            result['vi_pos'] = None
            result['vi_neg'] = None
            result['vi_diff'] = None

        return result

    def _calculate_volatility_indicators(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, Any]:
        """Calculate volatility indicators"""
        result = {}

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        result['bb_upper'] = self._safe_float(bb.bollinger_hband().iloc[-1])
        result['bb_middle'] = self._safe_float(bb.bollinger_mavg().iloc[-1])
        result['bb_lower'] = self._safe_float(bb.bollinger_lband().iloc[-1])
        result['bb_width'] = self._safe_float(bb.bollinger_wband().iloc[-1])
        result['bb_pband'] = self._safe_float(bb.bollinger_pband().iloc[-1])  # %B indicator

        # ATR
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        result['atr'] = self._safe_float(atr.iloc[-1])
        result['atr_pct'] = self._safe_float((atr.iloc[-1] / close.iloc[-1]) * 100)

        # Keltner Channels
        keltner = ta.volatility.KeltnerChannel(high, low, close, window=20)
        result['keltner_upper'] = self._safe_float(keltner.keltner_channel_hband().iloc[-1])
        result['keltner_middle'] = self._safe_float(keltner.keltner_channel_mband().iloc[-1])
        result['keltner_lower'] = self._safe_float(keltner.keltner_channel_lband().iloc[-1])

        # Donchian Channels
        donchian = ta.volatility.DonchianChannel(high, low, close, window=20)
        result['donchian_upper'] = self._safe_float(donchian.donchian_channel_hband().iloc[-1])
        result['donchian_lower'] = self._safe_float(donchian.donchian_channel_lband().iloc[-1])

        # Ulcer Index
        try:
            ui = ta.volatility.UlcerIndex(close, window=14)
            result['ulcer_index'] = self._safe_float(ui.ulcer_index().iloc[-1])
        except:
            result['ulcer_index'] = None

        # Historical Volatility (Standard Deviation)
        try:
            result['hist_volatility_10'] = self._safe_float(close.pct_change().rolling(10).std().iloc[-1] * 100)
            result['hist_volatility_30'] = self._safe_float(close.pct_change().rolling(30).std().iloc[-1] * 100)
        except:
            result['hist_volatility_10'] = None
            result['hist_volatility_30'] = None

        return result

    def _calculate_volume_indicators(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> Dict[str, Any]:
        """Calculate volume indicators"""
        result = {}

        # OBV
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        result['obv'] = self._safe_float(obv.iloc[-1])

        # Volume SMA
        vol_sma_20 = volume.rolling(20).mean().iloc[-1]
        result['volume_sma_20'] = self._safe_float(vol_sma_20)
        result['volume_ratio'] = self._safe_float(volume.iloc[-1] / vol_sma_20) if vol_sma_20 > 0 else 1

        # Accumulation/Distribution
        ad = ta.volume.AccDistIndexIndicator(close.shift(-1).fillna(close), close, close, volume).acc_dist_index()
        result['acc_dist'] = self._safe_float(ad.iloc[-1])

        # Chaikin Money Flow
        try:
            cmf = ta.volume.ChaikinMoneyFlowIndicator(
                close.shift(-1).fillna(close),  # high proxy
                close.shift(1).fillna(close),   # low proxy
                close,
                volume,
                window=20
            ).chaikin_money_flow()
            result['cmf'] = self._safe_float(cmf.iloc[-1])
        except:
            result['cmf'] = None

        return result

    def _calculate_advanced_indicators(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> Dict[str, Any]:
        """Calculate advanced indicators"""
        result = {}

        # ADX
        adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
        result['adx'] = self._safe_float(adx_indicator.adx().iloc[-1])
        result['adx_pos'] = self._safe_float(adx_indicator.adx_pos().iloc[-1])
        result['adx_neg'] = self._safe_float(adx_indicator.adx_neg().iloc[-1])

        # Parabolic SAR
        psar = ta.trend.PSARIndicator(high, low, close)
        result['psar'] = self._safe_float(psar.psar().iloc[-1])
        result['psar_up'] = self._safe_float(psar.psar_up().iloc[-1])
        result['psar_down'] = self._safe_float(psar.psar_down().iloc[-1])

        # Aroon
        aroon = ta.trend.AroonIndicator(high, low, window=25)
        result['aroon_up'] = self._safe_float(aroon.aroon_up().iloc[-1])
        result['aroon_down'] = self._safe_float(aroon.aroon_down().iloc[-1])

        # VWAP (intraday approximation using daily data)
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        result['vwap'] = self._safe_float(vwap.iloc[-1])

        # Ultimate Oscillator
        uo = ta.momentum.UltimateOscillator(high, low, close, window1=7, window2=14, window3=28)
        result['ultimate_oscillator'] = self._safe_float(uo.ultimate_oscillator().iloc[-1])

        # Awesome Oscillator
        try:
            ao = ta.momentum.AwesomeOscillatorIndicator(high, low, window1=5, window2=34)
            result['awesome_oscillator'] = self._safe_float(ao.awesome_oscillator().iloc[-1])
        except:
            result['awesome_oscillator'] = None

        # Kaufman's Adaptive Moving Average (KAMA)
        try:
            kama = ta.momentum.KAMAIndicator(close, window=10, pow1=2, pow2=30)
            result['kama'] = self._safe_float(kama.kama().iloc[-1])
        except:
            result['kama'] = None

        # Percentage Price Oscillator (PPO)
        try:
            ppo = ta.momentum.PPOIndicator(close, window_slow=26, window_fast=12, window_sign=9)
            result['ppo'] = self._safe_float(ppo.ppo().iloc[-1])
            result['ppo_signal'] = self._safe_float(ppo.ppo_signal().iloc[-1])
            result['ppo_hist'] = self._safe_float(ppo.ppo_hist().iloc[-1])
        except:
            result['ppo'] = None
            result['ppo_signal'] = None
            result['ppo_hist'] = None

        # Percentage Volume Oscillator (PVO)
        try:
            pvo = ta.volume.VolumePriceTrendIndicator(close, volume)
            result['vpt'] = self._safe_float(pvo.volume_price_trend().iloc[-1])
        except:
            result['vpt'] = None

        # Force Index
        try:
            fi = ta.volume.ForceIndexIndicator(close, volume, window=13)
            result['force_index'] = self._safe_float(fi.force_index().iloc[-1])
        except:
            result['force_index'] = None

        # Ease of Movement (EoM)
        try:
            eom = ta.volume.EaseOfMovementIndicator(high, low, volume, window=14)
            result['eom'] = self._safe_float(eom.ease_of_movement().iloc[-1])
            result['eom_signal'] = self._safe_float(eom.sma_ease_of_movement().iloc[-1])
        except:
            result['eom'] = None
            result['eom_signal'] = None

        # Negative Volume Index (NVI)
        try:
            nvi = ta.volume.NegativeVolumeIndexIndicator(close, volume)
            result['nvi'] = self._safe_float(nvi.negative_volume_index().iloc[-1])
        except:
            result['nvi'] = None

        return result

    def _calculate_ichimoku(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, Any]:
        """Calculate Ichimoku Cloud indicators"""
        result = {}

        try:
            ichimoku = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)

            result['ichimoku_tenkan'] = self._safe_float(ichimoku.ichimoku_conversion_line().iloc[-1])
            result['ichimoku_kijun'] = self._safe_float(ichimoku.ichimoku_base_line().iloc[-1])
            result['ichimoku_senkou_a'] = self._safe_float(ichimoku.ichimoku_a().iloc[-1])
            result['ichimoku_senkou_b'] = self._safe_float(ichimoku.ichimoku_b().iloc[-1])

            # Cloud color (bullish if Senkou A > Senkou B)
            senkou_a = ichimoku.ichimoku_a().iloc[-1]
            senkou_b = ichimoku.ichimoku_b().iloc[-1]
            result['ichimoku_cloud_bullish'] = senkou_a > senkou_b if not pd.isna(senkou_a) and not pd.isna(senkou_b) else None

            # Price position relative to cloud
            current_price = close.iloc[-1]
            if not pd.isna(senkou_a) and not pd.isna(senkou_b):
                cloud_top = max(senkou_a, senkou_b)
                cloud_bottom = min(senkou_a, senkou_b)
                if current_price > cloud_top:
                    result['ichimoku_position'] = 'above_cloud'
                elif current_price < cloud_bottom:
                    result['ichimoku_position'] = 'below_cloud'
                else:
                    result['ichimoku_position'] = 'in_cloud'
            else:
                result['ichimoku_position'] = None

        except Exception as e:
            logger.warning(f"Ichimoku calculation failed: {e}")
            result['ichimoku_tenkan'] = None
            result['ichimoku_kijun'] = None
            result['ichimoku_senkou_a'] = None
            result['ichimoku_senkou_b'] = None
            result['ichimoku_cloud_bullish'] = None
            result['ichimoku_position'] = None

        return result

    def _calculate_multi_timeframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators across multiple timeframes.
        Uses resampling to simulate weekly and monthly data.
        """
        result = {
            'daily': {},
            'weekly': {},
            'monthly': {}
        }

        close = df['Close']

        # Daily (current)
        rsi_daily = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        macd_daily = ta.trend.MACD(close).macd().iloc[-1]
        sma_20_daily = ta.trend.SMAIndicator(close, window=20).sma_indicator().iloc[-1]

        result['daily'] = {
            'rsi': self._safe_float(rsi_daily),
            'macd': self._safe_float(macd_daily),
            'sma_20': self._safe_float(sma_20_daily),
            'price_vs_sma': 'above' if close.iloc[-1] > sma_20_daily else 'below'
        }

        # Weekly (resample to weekly)
        if len(df) >= 35:  # At least 7 weeks of data
            try:
                weekly_df = df.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

                if len(weekly_df) >= 14:
                    weekly_close = weekly_df['Close']
                    rsi_weekly = ta.momentum.RSIIndicator(weekly_close, window=14).rsi().iloc[-1]
                    sma_10_weekly = ta.trend.SMAIndicator(weekly_close, window=10).sma_indicator().iloc[-1]

                    result['weekly'] = {
                        'rsi': self._safe_float(rsi_weekly),
                        'sma_10': self._safe_float(sma_10_weekly),
                        'price_vs_sma': 'above' if weekly_close.iloc[-1] > sma_10_weekly else 'below'
                    }
            except Exception as e:
                logger.warning(f"Weekly calculation failed: {e}")

        # Monthly (resample to monthly)
        if len(df) >= 150:  # At least 6 months of data
            try:
                monthly_df = df.resample('ME').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

                if len(monthly_df) >= 6:
                    monthly_close = monthly_df['Close']
                    sma_6_monthly = ta.trend.SMAIndicator(monthly_close, window=6).sma_indicator().iloc[-1]

                    result['monthly'] = {
                        'sma_6': self._safe_float(sma_6_monthly),
                        'price_vs_sma': 'above' if monthly_close.iloc[-1] > sma_6_monthly else 'below',
                        'trend': 'up' if monthly_close.iloc[-1] > monthly_close.iloc[-3] else 'down'
                    }
            except Exception as e:
                logger.warning(f"Monthly calculation failed: {e}")

        # Multi-timeframe alignment
        alignment_score = 0
        if result['daily'].get('price_vs_sma') == 'above':
            alignment_score += 1
        if result['weekly'].get('price_vs_sma') == 'above':
            alignment_score += 1
        if result['monthly'].get('price_vs_sma') == 'above':
            alignment_score += 1

        result['alignment'] = {
            'score': alignment_score,
            'max_score': 3,
            'direction': 'bullish' if alignment_score >= 2 else 'bearish' if alignment_score == 0 else 'mixed'
        }

        return result

    def calculate_volume_profile(
        self,
        df: pd.DataFrame,
        num_bins: int = 20
    ) -> Dict[str, Any]:
        """
        Calculate volume profile (volume at price levels).

        Args:
            df: DataFrame with OHLCV data
            num_bins: Number of price bins

        Returns:
            Volume profile analysis
        """
        if df.empty or len(df) < 20:
            return {}

        close = df['Close']
        volume = df['Volume']

        # Create price bins
        price_min = close.min()
        price_max = close.max()
        bin_edges = np.linspace(price_min, price_max, num_bins + 1)

        # Calculate volume at each price level
        volume_profile = []
        for i in range(len(bin_edges) - 1):
            mask = (close >= bin_edges[i]) & (close < bin_edges[i + 1])
            bin_volume = volume[mask].sum()
            bin_price = (bin_edges[i] + bin_edges[i + 1]) / 2
            volume_profile.append({
                'price': round(float(bin_price), 2),
                'volume': int(bin_volume),
                'pct_of_total': round(float(bin_volume / volume.sum() * 100), 2)
            })

        # Find Point of Control (highest volume price)
        poc = max(volume_profile, key=lambda x: x['volume'])

        # Find Value Area (70% of volume)
        sorted_by_volume = sorted(volume_profile, key=lambda x: x['volume'], reverse=True)
        total_volume = volume.sum()
        cumulative = 0
        value_area = []
        for level in sorted_by_volume:
            cumulative += level['volume']
            value_area.append(level['price'])
            if cumulative >= total_volume * 0.7:
                break

        value_area_high = max(value_area)
        value_area_low = min(value_area)

        current_price = close.iloc[-1]

        return {
            'poc': poc['price'],
            'value_area_high': round(value_area_high, 2),
            'value_area_low': round(value_area_low, 2),
            'price_vs_poc': 'above' if current_price > poc['price'] else 'below',
            'in_value_area': value_area_low <= current_price <= value_area_high,
            'profile': volume_profile[:10]  # Top 10 levels
        }

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float, handling NaN and None"""
        if value is None or pd.isna(value):
            return None
        try:
            return round(float(value), 4)
        except:
            return None


# Global instance
_advanced_indicators_instance = None


def get_advanced_indicators() -> AdvancedIndicators:
    """Get or create global advanced indicators instance"""
    global _advanced_indicators_instance
    if _advanced_indicators_instance is None:
        _advanced_indicators_instance = AdvancedIndicators()
    return _advanced_indicators_instance
