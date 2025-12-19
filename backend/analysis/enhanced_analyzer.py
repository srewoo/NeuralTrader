"""
Enhanced Stock Analyzer
Implements advanced analysis with:
- Market regime detection
- Multi-timeframe analysis
- Weighted indicator scoring
- Indicator confluence measurement
- Bayesian confidence scoring
- Data quality checks
- Pattern and sentiment integration
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio

from .market_regime import MarketRegimeDetector, get_regime_detector
from .indicators import AdvancedIndicators, get_advanced_indicators
from .fundamental_screener import FundamentalScreener, get_fundamental_screener

logger = logging.getLogger(__name__)


class EnhancedAnalyzer:
    """
    Enhanced stock analyzer with advanced technical analysis capabilities.
    """

    def __init__(self):
        self.regime_detector = get_regime_detector()
        self.advanced_indicators = get_advanced_indicators()
        self.screener = get_fundamental_screener()

        # Indicator weights based on historical effectiveness
        self.indicator_weights = {
            'rsi': 1.5,
            'macd': 1.3,
            'sma_crossover': 1.4,
            'bollinger': 1.2,
            'stochastic': 1.1,
            'adx': 1.3,
            'volume': 1.2,
            'ichimoku': 1.4,
            'pattern': 1.5,
            'sentiment': 1.0,
            'multi_timeframe': 1.6,
            'fundamentals': 1.8  # High weight for fundamentals to filter out risky stocks
        }

        # Historical accuracy tracking for Bayesian updates
        self.prior_accuracy = {
            'rsi_oversold': 0.65,
            'rsi_overbought': 0.62,
            'macd_bullish': 0.60,
            'macd_bearish': 0.58,
            'sma_golden_cross': 0.68,
            'sma_death_cross': 0.65,
            'bb_oversold': 0.55,
            'bb_overbought': 0.52,
            'stoch_oversold': 0.58,
            'stoch_overbought': 0.55,
            'trend_following': 0.70,
            'mean_reversion': 0.55
        }

    async def analyze_stock(
        self,
        symbol: str,
        include_patterns: bool = True,
        include_sentiment: bool = True,
        include_fundamentals: bool = True,
        include_backtest_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive stock analysis.

        Args:
            symbol: Stock symbol
            include_patterns: Include candlestick pattern analysis
            include_sentiment: Include news sentiment analysis
            include_backtest_validation: Validate signals against historical performance

        Returns:
            Complete analysis result
        """
        try:
            # Fetch data with quality checks
            df, data_quality = await self._fetch_and_validate_data(symbol)

            if df.empty:
                return self._create_error_result(symbol, "No data available")

            if data_quality['issues']:
                logger.warning(f"Data quality issues for {symbol}: {data_quality['issues']}")

            # Market regime detection
            regime_info = self.regime_detector.detect_regime(df)

            # Get adaptive thresholds based on regime
            thresholds = self.regime_detector.get_adaptive_thresholds(regime_info)

            # Calculate all indicators
            indicators = self.advanced_indicators.calculate_all_indicators(df)

            # Volume profile
            volume_profile = self.advanced_indicators.calculate_volume_profile(df)

            # Generate signals with weighted scoring
            signals, scores = self._generate_weighted_signals(
                df, indicators, regime_info, thresholds
            )

            # Calculate indicator confluence
            confluence = self._calculate_confluence(signals)

            # Pattern analysis
            pattern_signals = []
            if include_patterns:
                pattern_signals = await self._analyze_patterns(symbol, df)

            # News sentiment
            sentiment_data = None
            if include_sentiment:
                sentiment_data = await self._analyze_sentiment(symbol)

            # Backtest validation
            backtest_validation = None
            if include_backtest_validation:
                backtest_validation = await self._validate_with_backtest(symbol, signals)

            # Fundamental Analysis
            fundamental_data = None
            fundamental_score = 0
            if include_fundamentals:
                fundamental_data = await self.screener.get_stock_fundamentals(symbol)
                if fundamental_data and 'error' not in fundamental_data:
                    fundamental_score = self._calculate_fundamental_score(fundamental_data)
                else:
                    logger.warning(f"Fundamental analysis failed for {symbol}: {fundamental_data.get('error') if fundamental_data else 'No data'}")
                    fundamental_score = 50  # Neutral if data missing

            # Calculate Bayesian confidence
            confidence, confidence_breakdown = self._calculate_bayesian_confidence(
                signals, scores, confluence, regime_info, pattern_signals, sentiment_data, fundamental_score
            )

            # Determine recommendation
            recommendation = self._determine_recommendation(
                signals, scores, confidence, regime_info
            )

            # Calculate price targets with confidence intervals
            price_targets = self._calculate_price_targets(
                df, indicators, recommendation, regime_info
            )

            # Risk-reward validation
            risk_reward = self._validate_risk_reward(
                price_targets, df['Close'].iloc[-1], recommendation
            )

            # Sector relative strength
            sector_strength = await self._calculate_sector_strength(symbol, df)

            # Corporate events (earnings, dividends, splits)
            corporate_events = await self._detect_corporate_events(symbol)

            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "recommendation": recommendation,
                "confidence": confidence,
                "confidence_breakdown": confidence_breakdown,
                "fundamental_score": fundamental_score,
                "fundamentals": fundamental_data,
                "current_price": round(float(df['Close'].iloc[-1]), 2),
                "price_targets": price_targets,
                "risk_reward": risk_reward,
                "market_regime": regime_info,
                "adaptive_thresholds": thresholds,
                "signals": signals,
                "signal_scores": scores,
                "confluence": confluence,
                "indicators": indicators,
                "volume_profile": volume_profile,
                "multi_timeframe": indicators.get('multi_timeframe', {}),
                "patterns": pattern_signals,
                "sentiment": sentiment_data,
                "backtest_validation": backtest_validation,
                "sector_strength": sector_strength,
                "corporate_events": corporate_events,
                "data_quality": data_quality
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Analysis failed for {symbol}: {e}")
            return self._create_error_result(symbol, str(e))

    async def _fetch_and_validate_data(
        self,
        symbol: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch data and perform quality checks"""
        data_quality = {
            'is_fresh': True,
            'has_gaps': False,
            'sufficient_liquidity': True,
            'sufficient_history': True,
            'issues': []
        }

        try:
            # Add Indian stock suffix
            ticker_symbol = symbol.upper()
            if not ticker_symbol.endswith('.NS') and not ticker_symbol.endswith('.BO'):
                ticker_symbol = f"{ticker_symbol}.NS"

            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period="1y")

            if df.empty:
                # Try BSE
                ticker_symbol = symbol.upper().replace('.NS', '') + '.BO'
                ticker = yf.Ticker(ticker_symbol)
                df = ticker.history(period="1y")

            if df.empty:
                data_quality['issues'].append("No data available")
                return pd.DataFrame(), data_quality

            # Check data freshness (should have data from last trading day)
            last_date = df.index[-1].date()
            today = datetime.now().date()
            days_old = (today - last_date).days

            if days_old > 3:  # Allow for weekends
                data_quality['is_fresh'] = False
                data_quality['issues'].append(f"Data is {days_old} days old")

            # Check for gaps
            date_diff = df.index.to_series().diff().dropna()
            max_gap = date_diff.max().days if len(date_diff) > 0 else 0
            if max_gap > 5:  # More than 5 days gap (excluding weekends)
                data_quality['has_gaps'] = True
                data_quality['issues'].append(f"Data has gaps of {max_gap} days")

            # Check liquidity
            avg_volume = df['Volume'].mean()
            if avg_volume < 100000:
                data_quality['sufficient_liquidity'] = False
                data_quality['issues'].append(f"Low liquidity: avg volume {avg_volume:,.0f}")

            # Check sufficient history
            if len(df) < 100:
                data_quality['sufficient_history'] = False
                data_quality['issues'].append(f"Limited history: only {len(df)} days")

            return df, data_quality

        except Exception as e:
            data_quality['issues'].append(f"Fetch error: {str(e)}")
            return pd.DataFrame(), data_quality

    def _generate_weighted_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        regime_info: Dict[str, Any],
        thresholds: Dict[str, Any]
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """Generate signals with weighted scoring"""
        signals = []
        scores = {
            'buy_score': 0,
            'sell_score': 0,
            'hold_score': 0,
            'weighted_buy': 0,
            'weighted_sell': 0
        }

        current_price = df['Close'].iloc[-1]

        # RSI signals with adaptive thresholds
        rsi = indicators.get('rsi')
        if rsi is not None:
            weight = self.indicator_weights['rsi']

            if rsi < thresholds['rsi_extreme_oversold']:
                signals.append({
                    'indicator': 'RSI',
                    'signal': 'strong_buy',
                    'value': rsi,
                    'threshold': thresholds['rsi_extreme_oversold'],
                    'weight': weight,
                    'description': f"RSI extremely oversold ({rsi:.1f})"
                })
                scores['buy_score'] += 4
                scores['weighted_buy'] += 4 * weight

            elif rsi < thresholds['rsi_oversold']:
                signals.append({
                    'indicator': 'RSI',
                    'signal': 'buy',
                    'value': rsi,
                    'threshold': thresholds['rsi_oversold'],
                    'weight': weight,
                    'description': f"RSI oversold ({rsi:.1f})"
                })
                scores['buy_score'] += 2
                scores['weighted_buy'] += 2 * weight

            elif rsi > thresholds['rsi_extreme_overbought']:
                signals.append({
                    'indicator': 'RSI',
                    'signal': 'strong_sell',
                    'value': rsi,
                    'threshold': thresholds['rsi_extreme_overbought'],
                    'weight': weight,
                    'description': f"RSI extremely overbought ({rsi:.1f})"
                })
                scores['sell_score'] += 4
                scores['weighted_sell'] += 4 * weight

            elif rsi > thresholds['rsi_overbought']:
                signals.append({
                    'indicator': 'RSI',
                    'signal': 'sell',
                    'value': rsi,
                    'threshold': thresholds['rsi_overbought'],
                    'weight': weight,
                    'description': f"RSI overbought ({rsi:.1f})"
                })
                scores['sell_score'] += 2
                scores['weighted_sell'] += 2 * weight

        # MACD signals
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        macd_hist = indicators.get('macd_histogram')
        macd_hist_prev = indicators.get('macd_histogram_prev')

        if all(v is not None for v in [macd, macd_signal, macd_hist, macd_hist_prev]):
            weight = self.indicator_weights['macd']

            # Bullish crossover
            if macd_hist > 0 and macd_hist_prev <= 0:
                signals.append({
                    'indicator': 'MACD',
                    'signal': 'buy',
                    'value': macd_hist,
                    'weight': weight,
                    'description': "MACD bullish crossover"
                })
                scores['buy_score'] += 3
                scores['weighted_buy'] += 3 * weight

            # Bearish crossover
            elif macd_hist < 0 and macd_hist_prev >= 0:
                signals.append({
                    'indicator': 'MACD',
                    'signal': 'sell',
                    'value': macd_hist,
                    'weight': weight,
                    'description': "MACD bearish crossover"
                })
                scores['sell_score'] += 3
                scores['weighted_sell'] += 3 * weight

            # Histogram momentum
            elif macd_hist > 0 and macd_hist > macd_hist_prev:
                signals.append({
                    'indicator': 'MACD',
                    'signal': 'bullish_momentum',
                    'value': macd_hist,
                    'weight': weight * 0.5,
                    'description': "MACD histogram increasing"
                })
                scores['buy_score'] += 1
                scores['weighted_buy'] += 1 * weight * 0.5

        # Moving Average signals
        sma_20 = indicators.get('sma_20')
        sma_50 = indicators.get('sma_50')
        sma_200 = indicators.get('sma_200')

        if sma_20 is not None and sma_50 is not None:
            weight = self.indicator_weights['sma_crossover']

            # Golden Cross / Death Cross potential
            if current_price > sma_20 > sma_50:
                signals.append({
                    'indicator': 'SMA',
                    'signal': 'buy',
                    'value': {'price': current_price, 'sma_20': sma_20, 'sma_50': sma_50},
                    'weight': weight,
                    'description': "Price above aligned SMAs (uptrend)"
                })
                scores['buy_score'] += 2
                scores['weighted_buy'] += 2 * weight

            elif current_price < sma_20 < sma_50:
                signals.append({
                    'indicator': 'SMA',
                    'signal': 'sell',
                    'value': {'price': current_price, 'sma_20': sma_20, 'sma_50': sma_50},
                    'weight': weight,
                    'description': "Price below aligned SMAs (downtrend)"
                })
                scores['sell_score'] += 2
                scores['weighted_sell'] += 2 * weight

        # Bollinger Bands signals
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        bb_pband = indicators.get('bb_pband')

        if bb_upper is not None and bb_lower is not None:
            weight = self.indicator_weights['bollinger'] * thresholds.get('bb_touch_strength', 1.0)

            if current_price < bb_lower:
                signals.append({
                    'indicator': 'Bollinger',
                    'signal': 'buy',
                    'value': {'price': current_price, 'lower': bb_lower},
                    'weight': weight,
                    'description': "Price below lower Bollinger Band"
                })
                scores['buy_score'] += 2
                scores['weighted_buy'] += 2 * weight

            elif current_price > bb_upper:
                signals.append({
                    'indicator': 'Bollinger',
                    'signal': 'sell',
                    'value': {'price': current_price, 'upper': bb_upper},
                    'weight': weight,
                    'description': "Price above upper Bollinger Band"
                })
                scores['sell_score'] += 2
                scores['weighted_sell'] += 2 * weight

        # Stochastic signals
        stoch_k = indicators.get('stoch_k')
        stoch_d = indicators.get('stoch_d')

        if stoch_k is not None:
            weight = self.indicator_weights['stochastic']

            if stoch_k < thresholds['stoch_oversold']:
                signals.append({
                    'indicator': 'Stochastic',
                    'signal': 'buy',
                    'value': stoch_k,
                    'threshold': thresholds['stoch_oversold'],
                    'weight': weight,
                    'description': f"Stochastic oversold ({stoch_k:.1f})"
                })
                scores['buy_score'] += 1
                scores['weighted_buy'] += 1 * weight

            elif stoch_k > thresholds['stoch_overbought']:
                signals.append({
                    'indicator': 'Stochastic',
                    'signal': 'sell',
                    'value': stoch_k,
                    'threshold': thresholds['stoch_overbought'],
                    'weight': weight,
                    'description': f"Stochastic overbought ({stoch_k:.1f})"
                })
                scores['sell_score'] += 1
                scores['weighted_sell'] += 1 * weight

        # ADX trend strength
        adx = indicators.get('adx')
        adx_pos = indicators.get('adx_pos')
        adx_neg = indicators.get('adx_neg')

        if adx is not None and adx > thresholds['adx_trend']:
            weight = self.indicator_weights['adx']

            if adx_pos is not None and adx_neg is not None:
                if adx_pos > adx_neg:
                    signals.append({
                        'indicator': 'ADX',
                        'signal': 'buy',
                        'value': adx,
                        'weight': weight,
                        'description': f"Strong bullish trend (ADX: {adx:.1f})"
                    })
                    scores['buy_score'] += 2
                    scores['weighted_buy'] += 2 * weight
                else:
                    signals.append({
                        'indicator': 'ADX',
                        'signal': 'sell',
                        'value': adx,
                        'weight': weight,
                        'description': f"Strong bearish trend (ADX: {adx:.1f})"
                    })
                    scores['sell_score'] += 2
                    scores['weighted_sell'] += 2 * weight

        # Volume confirmation
        volume_ratio = indicators.get('volume_ratio')
        if volume_ratio is not None and volume_ratio > thresholds['volume_high']:
            weight = self.indicator_weights['volume']
            signals.append({
                'indicator': 'Volume',
                'signal': 'confirmation',
                'value': volume_ratio,
                'weight': weight,
                'description': f"High volume ({volume_ratio:.1f}x average)"
            })

        # Ichimoku signals
        ichimoku_position = indicators.get('ichimoku_position')
        ichimoku_cloud = indicators.get('ichimoku_cloud_bullish')

        if ichimoku_position is not None:
            weight = self.indicator_weights['ichimoku']

            if ichimoku_position == 'above_cloud' and ichimoku_cloud:
                signals.append({
                    'indicator': 'Ichimoku',
                    'signal': 'buy',
                    'value': ichimoku_position,
                    'weight': weight,
                    'description': "Price above bullish Ichimoku cloud"
                })
                scores['buy_score'] += 2
                scores['weighted_buy'] += 2 * weight

            elif ichimoku_position == 'below_cloud' and not ichimoku_cloud:
                signals.append({
                    'indicator': 'Ichimoku',
                    'signal': 'sell',
                    'value': ichimoku_position,
                    'weight': weight,
                    'description': "Price below bearish Ichimoku cloud"
                })
                scores['sell_score'] += 2
                scores['weighted_sell'] += 2 * weight

        # Multi-timeframe alignment
        mtf = indicators.get('multi_timeframe', {})
        alignment = mtf.get('alignment', {})

        if alignment:
            weight = self.indicator_weights['multi_timeframe']

            if alignment.get('direction') == 'bullish':
                signals.append({
                    'indicator': 'MultiTimeframe',
                    'signal': 'buy',
                    'value': alignment.get('score', 0),
                    'weight': weight,
                    'description': f"Multi-timeframe alignment bullish ({alignment.get('score', 0)}/3)"
                })
                scores['buy_score'] += 3
                scores['weighted_buy'] += 3 * weight

            elif alignment.get('direction') == 'bearish':
                signals.append({
                    'indicator': 'MultiTimeframe',
                    'signal': 'sell',
                    'value': alignment.get('score', 0),
                    'weight': weight,
                    'description': f"Multi-timeframe alignment bearish ({alignment.get('score', 0)}/3)"
                })
                scores['sell_score'] += 3
                scores['weighted_sell'] += 3 * weight

        return signals, scores

    def _calculate_confluence(self, signals: List[Dict]) -> Dict[str, Any]:
        """Calculate indicator confluence (agreement)"""
        buy_signals = [s for s in signals if s['signal'] in ['buy', 'strong_buy', 'bullish_momentum']]
        sell_signals = [s for s in signals if s['signal'] in ['sell', 'strong_sell', 'bearish_momentum']]
        neutral_signals = [s for s in signals if s['signal'] in ['confirmation', 'neutral']]

        total = len(buy_signals) + len(sell_signals) + len(neutral_signals)

        if total == 0:
            return {
                'agreement': 0,
                'direction': 'neutral',
                'strength': 'none',
                'conflicting': False
            }

        buy_pct = len(buy_signals) / total if total > 0 else 0
        sell_pct = len(sell_signals) / total if total > 0 else 0

        # Check for conflicting signals
        conflicting = len(buy_signals) > 0 and len(sell_signals) > 0

        # Determine direction and strength
        if buy_pct >= 0.7:
            direction = 'bullish'
            strength = 'strong'
            agreement = buy_pct * 100
        elif buy_pct >= 0.5:
            direction = 'bullish'
            strength = 'moderate'
            agreement = buy_pct * 100
        elif sell_pct >= 0.7:
            direction = 'bearish'
            strength = 'strong'
            agreement = sell_pct * 100
        elif sell_pct >= 0.5:
            direction = 'bearish'
            strength = 'moderate'
            agreement = sell_pct * 100
        else:
            direction = 'neutral'
            strength = 'weak'
            agreement = max(buy_pct, sell_pct) * 100

        return {
            'agreement': round(agreement, 1),
            'direction': direction,
            'strength': strength,
            'conflicting': conflicting,
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'total_signals': total
        }

    async def _analyze_patterns(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Analyze candlestick patterns"""
        try:
            from patterns.candlestick import get_pattern_detector

            detector = get_pattern_detector()
            recent_patterns = detector.get_recent_patterns(df, days=5)

            # Add weight to patterns
            for pattern in recent_patterns:
                pattern['weight'] = self.indicator_weights['pattern']
                # Use rupee symbol for Indian stocks
                pattern['formatted_price'] = f"₹{pattern['price']:,.2f}"

            return recent_patterns

        except Exception as e:
            logger.warning(f"Pattern analysis failed: {e}")
            return []

    async def _analyze_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze news sentiment"""
        try:
            from news.sources import get_news_aggregator
            from news.sentiment import get_sentiment_analyzer

            news_aggregator = get_news_aggregator()
            sentiment_analyzer = get_sentiment_analyzer()

            articles = news_aggregator.fetch_latest_news(symbol=symbol, limit=20)
            analyzed = sentiment_analyzer.analyze_articles(articles)
            aggregate = sentiment_analyzer.get_aggregate_sentiment(analyzed)

            return {
                'aggregate': aggregate,
                'article_count': len(analyzed),
                'weight': self.indicator_weights['sentiment']
            }

        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return None

    async def _validate_with_backtest(
        self,
        symbol: str,
        signals: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Validate current signals against historical backtest performance"""
        try:
            from backtesting.engine import BacktestEngine
            from backtesting.strategies import StrategyRegistry

            # Determine dominant signal type
            buy_signals = sum(1 for s in signals if 'buy' in s['signal'])
            sell_signals = sum(1 for s in signals if 'sell' in s['signal'])

            if buy_signals > sell_signals:
                # Use trend following strategy for validation
                strategy = StrategyRegistry.get_strategy('trend_following')
            else:
                # Use mean reversion
                strategy = StrategyRegistry.get_strategy('mean_reversion')

            engine = BacktestEngine(initial_capital=100000)

            # Quick backtest on recent 6 months
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

            result = engine.run_backtest(symbol, strategy, start_date, end_date)

            return {
                'strategy_used': strategy.name,
                'win_rate': result['metrics'].get('win_rate', 0),
                'total_return': result['metrics'].get('total_return_pct', 0),
                'sharpe_ratio': result['metrics'].get('sharpe_ratio', 0),
                'total_trades': result['metrics'].get('total_trades', 0),
                'validation': 'positive' if result['metrics'].get('total_return_pct', 0) > 0 else 'negative'
            }

        except Exception as e:
            logger.warning(f"Backtest validation failed: {e}")
            return None

    def _calculate_bayesian_confidence(
        self,
        signals: List[Dict],
        scores: Dict[str, float],
        confluence: Dict[str, Any],
        regime_info: Dict[str, Any],
        patterns: List[Dict],
        sentiment: Optional[Dict],
        fundamental_score: float = 50
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate Bayesian confidence score"""

        # Base probability
        base_confidence = 50

        # Start with weighted scores
        if scores['weighted_buy'] > scores['weighted_sell']:
            signal_direction = 'buy'
            weighted_diff = scores['weighted_buy'] - scores['weighted_sell']
        else:
            signal_direction = 'sell'
            weighted_diff = scores['weighted_sell'] - scores['weighted_buy']

        # Apply Bayesian update based on historical accuracy
        prior_accuracy = 0.55  # Base prior

        # Adjust based on signal types present
        for signal in signals:
            indicator = signal['indicator'].lower()
            signal_type = signal['signal']

            key = f"{indicator}_{signal_type}".replace('strong_', '')
            if key in self.prior_accuracy:
                prior_accuracy = (prior_accuracy + self.prior_accuracy[key]) / 2

        # Confluence adjustment
        if confluence['strength'] == 'strong':
            confluence_multiplier = 1.2
        elif confluence['strength'] == 'moderate':
            confluence_multiplier = 1.1
        elif confluence['conflicting']:
            confluence_multiplier = 0.8
        else:
            confluence_multiplier = 1.0

        # Regime adjustment
        regime = regime_info.get('primary_regime', 'sideways')
        if regime in ['strong_bull', 'strong_bear']:
            regime_multiplier = 1.15  # Clear trend is more reliable
        elif regime in ['bull', 'bear']:
            regime_multiplier = 1.05
        elif regime == 'high_volatility':
            regime_multiplier = 0.85  # High vol reduces confidence
        else:
            regime_multiplier = 1.0

        # Pattern confirmation
        pattern_bonus = 0
        if patterns:
            bullish_patterns = sum(1 for p in patterns if 'bullish' in p.get('type', ''))
            bearish_patterns = sum(1 for p in patterns if 'bearish' in p.get('type', ''))

            if signal_direction == 'buy' and bullish_patterns > bearish_patterns:
                pattern_bonus = min(10, bullish_patterns * 3)
            elif signal_direction == 'sell' and bearish_patterns > bullish_patterns:
                pattern_bonus = min(10, bearish_patterns * 3)
            elif (signal_direction == 'buy' and bearish_patterns > bullish_patterns) or \
                 (signal_direction == 'sell' and bullish_patterns > bearish_patterns):
                pattern_bonus = -5  # Contradicting patterns

        # Sentiment adjustment
        sentiment_bonus = 0
        if sentiment and sentiment.get('aggregate'):
            agg = sentiment['aggregate']
            sentiment_score = agg.get('average_score', 0)

            if signal_direction == 'buy' and sentiment_score > 0.2:
                sentiment_bonus = min(5, sentiment_score * 10)
            elif signal_direction == 'sell' and sentiment_score < -0.2:
                sentiment_bonus = min(5, abs(sentiment_score) * 10)
            elif (signal_direction == 'buy' and sentiment_score < -0.2) or \
                 (signal_direction == 'sell' and sentiment_score > 0.2):
                sentiment_bonus = -3

        # Fundamental adjustment
        # If technicals say BUY but fundamentals are trash (<40), heavily penalize
        # If technicals say BUY and fundamentals are great (>70), boost
        fundamental_bonus = 0
        if signal_direction == 'buy':
            if fundamental_score > 70:
                fundamental_bonus = 10
            elif fundamental_score > 60:
                fundamental_bonus = 5
            elif fundamental_score < 30:
                fundamental_bonus = -20  # Critical penalty for junk stocks
            elif fundamental_score < 40:
                fundamental_bonus = -10
        elif signal_direction == 'sell':
            # Selling good fundamentals should be harder? Maybe not for trading.
            # But selling bad fundamentals should be easier.
            if fundamental_score < 30:
                fundamental_bonus = 5
            elif fundamental_score > 70:
                fundamental_bonus = -5  # Careful selling quality

        # Calculate final confidence
        # Start from base, add weighted signal strength
        signal_contribution = min(30, weighted_diff * 2)

        confidence = base_confidence + signal_contribution
        confidence *= prior_accuracy * 2  # Prior adjusts the base
        confidence *= confluence_multiplier
        confidence *= regime_multiplier
        confidence += pattern_bonus
        confidence += sentiment_bonus
        confidence += fundamental_bonus

        # Clamp between 30 and 95
        confidence = max(30, min(95, confidence))

        breakdown = {
            'base': base_confidence,
            'signal_contribution': signal_contribution,
            'prior_accuracy': round(prior_accuracy, 2),
            'confluence_multiplier': confluence_multiplier,
            'regime_multiplier': regime_multiplier,
            'pattern_bonus': pattern_bonus,
            'sentiment_bonus': sentiment_bonus,
            'fundamental_bonus': fundamental_bonus,
            'formula': 'base + signals * prior * confluence * regime * 10 + patterns + sentiment + fundamentals'
        }

        return round(confidence, 1), breakdown

    def _determine_recommendation(
        self,
        signals: List[Dict],
        scores: Dict[str, float],
        confidence: float,
        regime_info: Dict[str, Any]
    ) -> str:
        """Determine final recommendation"""

        weighted_buy = scores['weighted_buy']
        weighted_sell = scores['weighted_sell']
        min_threshold = 6  # Minimum weighted score for recommendation

        # Strong buy conditions
        if weighted_buy >= min_threshold * 1.5 and weighted_buy > weighted_sell * 1.5:
            return "STRONG_BUY"

        # Buy conditions
        if weighted_buy >= min_threshold and weighted_buy > weighted_sell:
            return "BUY"

        # Strong sell conditions
        if weighted_sell >= min_threshold * 1.5 and weighted_sell > weighted_buy * 1.5:
            return "STRONG_SELL"

        # Sell conditions
        if weighted_sell >= min_threshold and weighted_sell > weighted_buy:
            return "SELL"

        # Hold if no clear signal
        return "HOLD"

    def _calculate_price_targets(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        recommendation: str,
        regime_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate price targets with confidence intervals"""

        current_price = float(df['Close'].iloc[-1])
        atr = indicators.get('atr', current_price * 0.02)
        atr = atr if atr else current_price * 0.02

        # Adjust multipliers based on regime
        regime = regime_info.get('primary_regime', 'sideways')
        volatility = regime_info.get('volatility_regime', 'normal')

        if volatility == 'high':
            atr_mult_target = 3.5
            atr_mult_stop = 2.0
        elif volatility == 'low':
            atr_mult_target = 2.0
            atr_mult_stop = 1.0
        else:
            atr_mult_target = 2.5
            atr_mult_stop = 1.5

        # Calculate targets based on recommendation
        if recommendation in ['BUY', 'STRONG_BUY']:
            entry = current_price
            target_low = current_price + (atr * atr_mult_target * 0.8)
            target_mid = current_price + (atr * atr_mult_target)
            target_high = current_price + (atr * atr_mult_target * 1.3)
            stop_loss = current_price - (atr * atr_mult_stop)

        elif recommendation in ['SELL', 'STRONG_SELL']:
            entry = current_price
            target_low = current_price - (atr * atr_mult_target * 1.3)
            target_mid = current_price - (atr * atr_mult_target)
            target_high = current_price - (atr * atr_mult_target * 0.8)
            stop_loss = current_price + (atr * atr_mult_stop)

        else:  # HOLD
            entry = current_price
            target_low = None
            target_mid = None
            target_high = None
            stop_loss = None

        return {
            'entry': round(entry, 2),
            'target_low': round(target_low, 2) if target_low else None,
            'target': round(target_mid, 2) if target_mid else None,
            'target_high': round(target_high, 2) if target_high else None,
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'target_range': f"₹{target_low:,.2f} - ₹{target_high:,.2f}" if target_low and target_high else None,
            'atr_used': round(atr, 2)
        }

    def _validate_risk_reward(
        self,
        targets: Dict[str, Any],
        current_price: float,
        recommendation: str
    ) -> Dict[str, Any]:
        """Validate risk-reward ratio"""

        if recommendation == 'HOLD' or not targets.get('target') or not targets.get('stop_loss'):
            return {
                'ratio': None,
                'is_valid': False,
                'minimum_required': 2.0,
                'message': 'No actionable targets'
            }

        target = targets['target']
        stop_loss = targets['stop_loss']
        entry = targets['entry']

        if recommendation in ['BUY', 'STRONG_BUY']:
            reward = target - entry
            risk = entry - stop_loss
        else:
            reward = entry - target
            risk = stop_loss - entry

        ratio = reward / risk if risk > 0 else 0

        is_valid = ratio >= 2.0  # Minimum 2:1 risk-reward

        return {
            'ratio': round(ratio, 2),
            'reward': round(reward, 2),
            'risk': round(risk, 2),
            'is_valid': is_valid,
            'minimum_required': 2.0,
            'message': 'Good risk-reward' if is_valid else 'Risk-reward below 2:1 threshold'
        }

    def _calculate_fundamental_score(self, fundamentals: Dict[str, Any]) -> float:
        """
        Calculate a composite fundamental score (0-100)
        Combines Valuation and Quality scores.
        """
        try:
            val_score = fundamentals.get('valuation_score', 50) or 50
            qual_score = fundamentals.get('quality_score', 50) or 50

            # Weighted average: Quality is slightly more important for safety
            composite_score = (val_score * 0.4) + (qual_score * 0.6)

            return round(composite_score, 1)

        except Exception as e:
            logger.error(f"Fundamental score calculation failed: {e}")
            return 50.0

    async def _calculate_sector_strength(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate relative strength vs sector"""
        try:
            # Get NIFTY 50 as benchmark
            nifty = yf.Ticker("^NSEI")
            nifty_df = nifty.history(period="3mo")

            if nifty_df.empty:
                return {'available': False}

            # Calculate relative strength
            stock_return = (df['Close'].iloc[-1] / df['Close'].iloc[-60] - 1) * 100 if len(df) >= 60 else None
            nifty_return = (nifty_df['Close'].iloc[-1] / nifty_df['Close'].iloc[-60] - 1) * 100 if len(nifty_df) >= 60 else None

            if stock_return is not None and nifty_return is not None:
                relative_strength = stock_return - nifty_return

                return {
                    'available': True,
                    'stock_return_3m': round(stock_return, 2),
                    'nifty_return_3m': round(nifty_return, 2),
                    'relative_strength': round(relative_strength, 2),
                    'outperforming': relative_strength > 0
                }

            return {'available': False}

        except Exception as e:
            logger.warning(f"Sector strength calculation failed: {e}")
            return {'available': False}

    async def _detect_corporate_events(self, symbol: str) -> Dict[str, Any]:
        """
        Detect upcoming corporate events (earnings, dividends, splits, etc.)
        using yfinance calendar data.
        """
        try:
            ticker_symbol = symbol.upper()
            if not ticker_symbol.endswith('.NS') and not ticker_symbol.endswith('.BO'):
                ticker_symbol = f"{ticker_symbol}.NS"

            ticker = yf.Ticker(ticker_symbol)

            events = {
                'has_upcoming_events': False,
                'earnings': None,
                'dividends': None,
                'splits': None,
                'warnings': []
            }

            # Check for upcoming earnings
            try:
                calendar = ticker.calendar
                if calendar is not None and not calendar.empty:
                    if 'Earnings Date' in calendar.index:
                        earnings_date = calendar.loc['Earnings Date']
                        if isinstance(earnings_date, pd.Series):
                            earnings_date = earnings_date.iloc[0]
                        if pd.notna(earnings_date):
                            earnings_dt = pd.to_datetime(earnings_date)
                            days_to_earnings = (earnings_dt - pd.Timestamp.now()).days
                            if 0 <= days_to_earnings <= 30:
                                events['has_upcoming_events'] = True
                                events['earnings'] = {
                                    'date': earnings_dt.strftime('%Y-%m-%d'),
                                    'days_away': days_to_earnings
                                }
                                if days_to_earnings <= 7:
                                    events['warnings'].append(f"Earnings in {days_to_earnings} days - expect high volatility")
            except Exception as e:
                logger.debug(f"Could not fetch earnings calendar: {e}")

            # Check recent dividends
            try:
                dividends = ticker.dividends
                if dividends is not None and len(dividends) > 0:
                    last_dividend = dividends.iloc[-1]
                    last_div_date = dividends.index[-1]
                    days_since_div = (pd.Timestamp.now() - last_div_date).days

                    if days_since_div <= 30:
                        events['dividends'] = {
                            'last_date': last_div_date.strftime('%Y-%m-%d'),
                            'amount': float(last_dividend),
                            'days_since': days_since_div
                        }
                        if days_since_div <= 5:
                            events['warnings'].append(f"Recent dividend of ₹{last_dividend:.2f} - may be ex-dividend adjusted")
            except Exception as e:
                logger.debug(f"Could not fetch dividend data: {e}")

            # Check for splits
            try:
                splits = ticker.splits
                if splits is not None and len(splits) > 0:
                    last_split = splits.iloc[-1]
                    last_split_date = splits.index[-1]
                    days_since_split = (pd.Timestamp.now() - last_split_date).days

                    if days_since_split <= 30:
                        events['splits'] = {
                            'last_date': last_split_date.strftime('%Y-%m-%d'),
                            'ratio': float(last_split),
                            'days_since': days_since_split
                        }
                        events['warnings'].append(f"Recent stock split {last_split}:1 - technical levels may need adjustment")
            except Exception as e:
                logger.debug(f"Could not fetch split data: {e}")

            return events

        except Exception as e:
            logger.warning(f"Event detection failed for {symbol}: {e}")
            return {
                'has_upcoming_events': False,
                'earnings': None,
                'dividends': None,
                'splits': None,
                'warnings': []
            }

    def _create_error_result(self, symbol: str, error: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'symbol': symbol,
            'error': error,
            'recommendation': 'HOLD',
            'confidence': 0,
            'timestamp': datetime.now().isoformat()
        }


# Global instance
_enhanced_analyzer_instance = None


def get_enhanced_analyzer() -> EnhancedAnalyzer:
    """Get or create global enhanced analyzer instance"""
    global _enhanced_analyzer_instance
    if _enhanced_analyzer_instance is None:
        _enhanced_analyzer_instance = EnhancedAnalyzer()
    return _enhanced_analyzer_instance
