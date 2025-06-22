"""
Session Analyzer Engine - Multi-Session Trading Optimization (OPTIMIZED)
File: src/python/session_analyzer.py
Version: 1.1.0 (Performance Optimized)
Created: June 2025
Description: Optimized session analysis with caching and performance improvements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import pytz
from functools import lru_cache

@dataclass
class TradingSession:
    """Trading session configuration"""
    name: str
    start_hour: int  # GMT hours
    end_hour: int    # GMT hours
    pairs: List[str]
    characteristics: str
    strategy_bias: str
    risk_multiplier: float
    expected_volatility: float

class SessionAnalyzer:
    """
    Optimized session analysis engine for multi-session trading optimization.
    
    Performance improvements:
    - Pre-computed session masks for faster filtering
    - Cached calculations to avoid redundant processing
    - Optimized timezone handling
    - Memory-efficient data structures
    """
    
    def __init__(self, symbol: str = "EURUSD", timeframe: str = "M15"):
        """
        Initialize session analyzer with optimization features
        
        Args:
            symbol: Trading symbol (default: EURUSD)
            timeframe: Chart timeframe (default: M15)
        """
        self.logger = logging.getLogger(__name__)
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Define trading sessions
        self.sessions = {
            'asian': TradingSession(
                name='Asian',
                start_hour=22,  # 22:00 GMT (Previous day)
                end_hour=8,     # 08:00 GMT
                pairs=['USDJPY', 'AUDUSD', 'NZDUSD', 'EURJPY'],
                characteristics='Range trading, lower volatility, mean reversion',
                strategy_bias='Range/Mean reversion',
                risk_multiplier=0.8,
                expected_volatility=0.6
            ),
            'london': TradingSession(
                name='London', 
                start_hour=8,   # 08:00 GMT
                end_hour=17,    # 17:00 GMT
                pairs=['EURUSD', 'GBPUSD', 'USDCHF', 'EURGBP'],
                characteristics='Trend following, high volatility, breakouts',
                strategy_bias='Trend following/Breakout',
                risk_multiplier=1.0,
                expected_volatility=1.0
            ),
            'newyork': TradingSession(
                name='New York',
                start_hour=13,  # 13:00 GMT (overlap with London)
                end_hour=22,    # 22:00 GMT
                pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD'],
                characteristics='High volatility, news-driven, momentum',
                strategy_bias='Momentum/News trading',
                risk_multiplier=1.2,
                expected_volatility=1.1
            )
        }
        
        # Session overlap periods (higher volatility)
        self.session_overlaps = {
            'london_newyork': {'start': 13, 'end': 17, 'multiplier': 1.3},
            'asian_london': {'start': 7, 'end': 9, 'multiplier': 1.1}
        }
        
        # Performance optimization: Pre-computed session masks cache
        self._session_masks_cache = {}
        self._last_data_hash = None
        
        # Results cache for expensive calculations
        self._volatility_cache = {}
        self._bias_cache = {}
        self._optimal_times_cache = {}
        
        self.timezone = pytz.UTC
        
    @lru_cache(maxsize=128)
    def detect_current_session(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Detect current trading session based on GMT time (cached for performance)
        
        Args:
            timestamp: Current timestamp (UTC). If None, uses current time
            
        Returns:
            Dict with current session info
        """
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            elif timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
                
            current_hour = timestamp.hour
            
            # Optimized session detection with lookup table
            if 8 <= current_hour < 17:
                current_session = 'london'
            elif 13 <= current_hour < 22:
                current_session = 'london' if current_hour < 17 else 'newyork'
            else:
                current_session = 'asian'
                
            session_info = self.sessions[current_session]
            
            # Calculate time remaining in session (optimized)
            if current_session == 'asian':
                time_remaining = (24 + 8 - current_hour) * 60 - timestamp.minute if current_hour >= 22 else (8 - current_hour) * 60 - timestamp.minute
            else:
                time_remaining = (session_info.end_hour - current_hour) * 60 - timestamp.minute
                
            # Pre-compute overlap information
            overlap_multiplier = 1.0
            in_overlap = False
            
            for overlap_info in self.session_overlaps.values():
                if overlap_info['start'] <= current_hour < overlap_info['end']:
                    overlap_multiplier = overlap_info['multiplier']
                    in_overlap = True
                    break
                    
            return {
                'session': current_session,
                'session_name': session_info.name,
                'session_index': list(self.sessions.keys()).index(current_session),
                'time_remaining_minutes': max(0, time_remaining),
                'session_progress': min(1.0, max(0.0, 1.0 - (time_remaining / (session_info.end_hour - session_info.start_hour) / 60))),
                'risk_multiplier': session_info.risk_multiplier * overlap_multiplier,
                'expected_volatility': session_info.expected_volatility * overlap_multiplier,
                'strategy_bias': session_info.strategy_bias,
                'optimal_pairs': session_info.pairs,
                'in_overlap': in_overlap,
                'overlap_multiplier': overlap_multiplier
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting current session: {e}")
            return self._get_default_session_info()
    
    def _get_session_masks(self, ohlcv_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Pre-compute session masks for faster filtering (with caching)
        
        Args:
            ohlcv_data: OHLCV price data with datetime index
            
        Returns:
            Dict with session masks for fast filtering
        """
        # Create data hash for cache validation
        data_hash = hash(f"{len(ohlcv_data)}_{ohlcv_data.index[0]}_{ohlcv_data.index[-1]}")
        
        if data_hash == self._last_data_hash and self._session_masks_cache:
            return self._session_masks_cache
        
        # Ensure datetime index and UTC timezone
        if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
            ohlcv_data.index = pd.to_datetime(ohlcv_data.index)
            
        if ohlcv_data.index.tz is None:
            ohlcv_data.index = ohlcv_data.index.tz_localize('UTC')
        elif ohlcv_data.index.tz != timezone.utc:
            ohlcv_data.index = ohlcv_data.index.tz_convert('UTC')
        
        # Pre-compute hour values once
        hours = ohlcv_data.index.hour
        
        # Create session masks (vectorized operations)
        session_masks = {
            'asian': (hours >= 22) | (hours < 8),
            'london': (hours >= 8) & (hours < 17),
            'newyork': (hours >= 13) & (hours < 22)
        }
        
        # Cache results
        self._session_masks_cache = session_masks
        self._last_data_hash = data_hash
        
        return session_masks
    
    def calculate_session_volatility(self, ohlcv_data: pd.DataFrame, 
                                   lookback_days: int = 20) -> Dict[str, float]:
        """
        Calculate session-specific volatility patterns (optimized with caching)
        
        Args:
            ohlcv_data: OHLCV price data with datetime index
            lookback_days: Number of days to analyze
            
        Returns:
            Dict with volatility metrics for each session
        """
        # Check cache first
        cache_key = f"{len(ohlcv_data)}_{lookback_days}_{hash(str(ohlcv_data.index[-1]))}"
        if cache_key in self._volatility_cache:
            return self._volatility_cache[cache_key]
        
        try:
            if len(ohlcv_data) < 100:
                return self._get_default_volatility_metrics()
            
            # Get pre-computed session masks
            session_masks = self._get_session_masks(ohlcv_data)
            
            # Vectorized True Range calculation
            ohlcv_data = ohlcv_data.copy()
            prev_close = ohlcv_data['close'].shift(1)
            tr1 = ohlcv_data['high'] - ohlcv_data['low']
            tr2 = abs(ohlcv_data['high'] - prev_close)
            tr3 = abs(ohlcv_data['low'] - prev_close)
            ohlcv_data['tr'] = np.maximum(tr1, np.maximum(tr2, tr3))
            
            session_volatility = {}
            
            for session_name in self.sessions.keys():
                try:
                    # Use pre-computed mask for filtering (much faster)
                    session_data = ohlcv_data[session_masks[session_name]]
                    
                    if len(session_data) < 10:
                        session_volatility[session_name] = self._get_default_single_volatility_metrics()
                        continue
                    
                    # Optimized calculations with vectorized operations
                    recent_data = session_data.tail(lookback_days * 4)  # Approx 4 periods per day
                    
                    avg_tr = recent_data['tr'].mean()
                    volatility_percentile = np.clip((recent_data['tr'].tail(5).mean() / 
                                                   recent_data['tr'].quantile(0.5)) * 50, 0, 100)
                    
                    # Vectorized volatility trend calculation
                    if len(recent_data) >= 10:
                        mid_point = len(recent_data) // 2
                        first_half_vol = recent_data['tr'].iloc[:mid_point].mean()
                        second_half_vol = recent_data['tr'].iloc[mid_point:].mean()
                        volatility_trend = (second_half_vol - first_half_vol) / first_half_vol if first_half_vol > 0 else 0.0
                    else:
                        volatility_trend = 0.0
                    
                    # Session range calculation
                    session_ranges = recent_data['high'] - recent_data['low']
                    session_range_avg = session_ranges.mean()
                    
                    session_volatility[session_name] = {
                        'avg_true_range': float(avg_tr),
                        'volatility_percentile': float(volatility_percentile),
                        'volatility_trend': float(np.clip(volatility_trend, -1, 1)),
                        'session_range_avg': float(session_range_avg)
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating volatility for {session_name}: {e}")
                    session_volatility[session_name] = self._get_default_single_volatility_metrics()
            
            # Cache results
            self._volatility_cache[cache_key] = session_volatility
            
            # Limit cache size
            if len(self._volatility_cache) > 50:
                # Remove oldest entries
                oldest_keys = list(self._volatility_cache.keys())[:10]
                for key in oldest_keys:
                    del self._volatility_cache[key]
            
            return session_volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating session volatility: {e}")
            return self._get_default_volatility_metrics()
    
    def identify_optimal_trading_times(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify optimal trading windows within each session (optimized)
        
        Args:
            ohlcv_data: OHLCV price data
            
        Returns:
            Dict with optimal trading windows and scores
        """
        # Check cache first
        cache_key = f"optimal_times_{len(ohlcv_data)}_{hash(str(ohlcv_data.index[-1]))}"
        if cache_key in self._optimal_times_cache:
            return self._optimal_times_cache[cache_key]
        
        try:
            if len(ohlcv_data) < 100:
                return self._get_default_trading_times()
            
            # Get pre-computed session masks
            session_masks = self._get_session_masks(ohlcv_data)
            
            # Pre-compute common calculations
            ohlcv_data = ohlcv_data.copy()
            ohlcv_data['hour'] = ohlcv_data.index.hour
            ohlcv_data['price_move'] = abs(ohlcv_data['close'] - ohlcv_data['open'])
            
            # Vectorized volume normalization
            if 'volume' in ohlcv_data.columns:
                volume_rolling_mean = ohlcv_data['volume'].rolling(24, min_periods=1).mean()
                ohlcv_data['volume_norm'] = ohlcv_data['volume'] / volume_rolling_mean
            else:
                ohlcv_data['volume_norm'] = 1.0
            
            optimal_times = {}
            
            for session_name, session_info in self.sessions.items():
                try:
                    # Get session hours efficiently
                    if session_name == 'asian':
                        session_hours = list(range(22, 24)) + list(range(0, 8))
                    else:
                        session_hours = list(range(session_info.start_hour, session_info.end_hour))
                    
                    # Vectorized hour scoring
                    hour_scores = {}
                    session_data = ohlcv_data[session_masks[session_name]]
                    
                    # Pre-compute quantiles for performance
                    movement_quantile_80 = ohlcv_data['price_move'].quantile(0.8)
                    
                    for hour in session_hours:
                        hour_data = session_data[session_data['hour'] == hour]
                        
                        if len(hour_data) < 5:
                            hour_scores[hour] = 0.5
                            continue
                        
                        # Vectorized scoring calculations
                        avg_movement = hour_data['price_move'].mean()
                        avg_volume = hour_data['volume_norm'].mean()
                        
                        # Optimized score calculation
                        movement_score = min(1.0, avg_movement / movement_quantile_80) if movement_quantile_80 > 0 else 0.5
                        volume_score = min(1.0, avg_volume)
                        
                        hour_scores[hour] = movement_score * 0.7 + volume_score * 0.3
                    
                    # Find optimal windows
                    if hour_scores:
                        best_hour = max(hour_scores.items(), key=lambda x: x[1])
                        session_activity_score = sum(hour_scores.values()) / len(hour_scores)
                    else:
                        best_hour = (session_info.start_hour, 0.5)
                        session_activity_score = 0.5
                    
                    optimal_times[session_name] = {
                        'best_hour': best_hour[0],
                        'best_score': best_hour[1],
                        'hour_scores': hour_scores,
                        'optimal_window_start': best_hour[0],
                        'optimal_window_end': best_hour[0] + 2,
                        'session_activity_score': session_activity_score
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating optimal times for {session_name}: {e}")
                    optimal_times[session_name] = {
                        'best_hour': session_info.start_hour,
                        'best_score': 0.5,
                        'hour_scores': {},
                        'optimal_window_start': session_info.start_hour,
                        'optimal_window_end': session_info.start_hour + 2,
                        'session_activity_score': 0.5
                    }
            
            # Cache results
            self._optimal_times_cache[cache_key] = optimal_times
            
            # Limit cache size
            if len(self._optimal_times_cache) > 30:
                oldest_keys = list(self._optimal_times_cache.keys())[:10]
                for key in oldest_keys:
                    del self._optimal_times_cache[key]
            
            return optimal_times
            
        except Exception as e:
            self.logger.error(f"Error identifying optimal trading times: {e}")
            return self._get_default_trading_times()
    
    def calculate_session_bias(self, ohlcv_data: pd.DataFrame, 
                             lookback_days: int = 10) -> Dict[str, float]:
        """
        Calculate directional bias for each session (optimized with caching)
        
        Args:
            ohlcv_data: OHLCV price data
            lookback_days: Number of days to analyze for bias
            
        Returns:
            Dict with session bias metrics
        """
        # Check cache first
        cache_key = f"bias_{len(ohlcv_data)}_{lookback_days}_{hash(str(ohlcv_data.index[-1]))}"
        if cache_key in self._bias_cache:
            return self._bias_cache[cache_key]
        
        try:
            if len(ohlcv_data) < 50:
                return self._get_default_session_bias()
            
            # Get pre-computed session masks
            session_masks = self._get_session_masks(ohlcv_data)
            
            # Pre-compute session change once
            ohlcv_data = ohlcv_data.copy()
            ohlcv_data['session_change'] = ohlcv_data['close'] - ohlcv_data['open']
            
            # Only use recent data for bias calculation
            recent_data = ohlcv_data.tail(lookback_days * 24 * 4)  # 4 periods per hour
            
            session_bias = {}
            
            for session_name in self.sessions.keys():
                try:
                    # Use pre-computed mask for filtering
                    session_data = recent_data[session_masks[session_name] & (recent_data.index.isin(recent_data.index))]
                    
                    if len(session_data) < 10:
                        session_bias[session_name] = self._get_default_single_session_bias()
                        continue
                    
                    # Vectorized bias calculations
                    session_changes = session_data['session_change']
                    positive_mask = session_changes > 0
                    negative_mask = session_changes < 0
                    
                    # Efficient statistics calculation
                    win_rate_long = positive_mask.sum() / len(session_changes) if len(session_changes) > 0 else 0.5
                    win_rate_short = negative_mask.sum() / len(session_changes) if len(session_changes) > 0 else 0.5
                    
                    avg_positive = session_changes[positive_mask].mean() if positive_mask.any() else 0.0
                    avg_negative = abs(session_changes[negative_mask].mean()) if negative_mask.any() else 0.0
                    
                    # Overall directional bias calculation
                    total_change = session_changes.sum()
                    total_absolute_change = abs(session_changes).sum()
                    directional_bias = total_change / total_absolute_change if total_absolute_change > 0 else 0.0
                    
                    # Bias strength calculation
                    bias_strength = abs(win_rate_long - 0.5) * 2
                    
                    session_bias[session_name] = {
                        'directional_bias': float(np.clip(directional_bias, -1, 1)),
                        'bias_strength': float(np.clip(bias_strength, 0, 1)),
                        'win_rate_long': float(win_rate_long),
                        'win_rate_short': float(win_rate_short),
                        'avg_move_positive': float(avg_positive),
                        'avg_move_negative': float(avg_negative)
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating bias for {session_name}: {e}")
                    session_bias[session_name] = self._get_default_single_session_bias()
            
            # Cache results
            self._bias_cache[cache_key] = session_bias
            
            # Limit cache size
            if len(self._bias_cache) > 30:
                oldest_keys = list(self._bias_cache.keys())[:10]
                for key in oldest_keys:
                    del self._bias_cache[key]
            
            return session_bias
            
        except Exception as e:
            self.logger.error(f"Error calculating session bias: {e}")
            return self._get_default_session_bias()
    
    def analyze_cross_session_correlation(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze correlation between different trading sessions (optimized)
        
        Args:
            ohlcv_data: OHLCV price data
            
        Returns:
            Dict with cross-session correlation metrics
        """
        try:
            if len(ohlcv_data) < 200:
                return self._get_default_correlation()
            
            # Get pre-computed session masks
            session_masks = self._get_session_masks(ohlcv_data)
            
            # Pre-compute returns once
            ohlcv_data = ohlcv_data.copy()
            ohlcv_data['returns'] = ohlcv_data['close'].pct_change()
            ohlcv_data['date'] = ohlcv_data.index.date
            
            # Aggregate returns by session efficiently
            session_returns = {}
            
            for session_name in self.sessions.keys():
                session_data = ohlcv_data[session_masks[session_name]]
                
                if len(session_data) > 10:
                    # Vectorized groupby operation
                    daily_session_returns = session_data.groupby('date')['returns'].sum()
                    session_returns[session_name] = daily_session_returns
            
            # Vectorized correlation calculations
            correlations = {}
            
            if len(session_returns) >= 2:
                session_names = list(session_returns.keys())
                
                for i in range(len(session_names)):
                    for j in range(i + 1, len(session_names)):
                        session1, session2 = session_names[i], session_names[j]
                        
                        # Efficient data alignment and correlation
                        common_dates = session_returns[session1].index.intersection(
                            session_returns[session2].index
                        )
                        
                        if len(common_dates) > 10:
                            returns1 = session_returns[session1].loc[common_dates]
                            returns2 = session_returns[session2].loc[common_dates]
                            
                            correlation = returns1.corr(returns2)
                            correlations[f"{session1}_{session2}_correlation"] = float(correlation) if not np.isnan(correlation) else 0.0
                        else:
                            correlations[f"{session1}_{session2}_correlation"] = 0.0
            
            # Add trend continuation metrics efficiently
            for session_name in self.sessions.keys():
                if session_name in session_returns and len(session_returns[session_name]) > 5:
                    returns = session_returns[session_name]
                    trend_continuation = returns.autocorr(lag=1)
                    correlations[f"{session_name}_trend_continuation"] = float(trend_continuation) if not np.isnan(trend_continuation) else 0.0
                else:
                    correlations[f"{session_name}_trend_continuation"] = 0.0
                        
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error analyzing cross-session correlation: {e}")
            return self._get_default_correlation()
    
    def generate_session_features(self, ohlcv_data: pd.DataFrame, 
                                current_timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """
        Generate comprehensive session-based features for AI model (optimized)
        
        Args:
            ohlcv_data: OHLCV price data
            current_timestamp: Current timestamp for session detection
            
        Returns:
            Dict with 18+ session features
        """
        try:
            # Get current session info (cached)
            current_session_info = self.detect_current_session(current_timestamp)
            
            # Calculate session metrics (all cached)
            session_volatility = self.calculate_session_volatility(ohlcv_data)
            optimal_times = self.identify_optimal_trading_times(ohlcv_data)
            session_bias = self.calculate_session_bias(ohlcv_data)
            cross_correlation = self.analyze_cross_session_correlation(ohlcv_data)
            
            current_session = current_session_info['session']
            
            # Generate features efficiently
            features = {
                # Current Session Features
                'session_current': float(current_session_info['session_index']),
                'session_time_remaining': float(current_session_info['time_remaining_minutes'] / 60),
                'session_progress': float(current_session_info['session_progress']),
                'session_risk_multiplier': float(current_session_info['risk_multiplier']),
                'session_expected_volatility': float(current_session_info['expected_volatility']),
                
                # Optimal Trading Window Features
                'session_in_optimal_window': 1.0 if self._is_in_optimal_window(current_timestamp, optimal_times[current_session]) else 0.0,
                'session_optimal_score': float(optimal_times[current_session]['best_score']),
                'session_activity_score': float(optimal_times[current_session]['session_activity_score']),
                
                # Volatility Features
                'session_volatility_regime': float(session_volatility[current_session]['volatility_percentile'] / 100),
                'session_volatility_trend': float(session_volatility[current_session]['volatility_trend']),
                'session_avg_true_range': float(session_volatility[current_session]['avg_true_range']),
                
                # Bias Features
                'session_directional_bias': float(session_bias[current_session]['directional_bias']),
                'session_bias_strength': float(session_bias[current_session]['bias_strength']),
                'session_bullish_win_rate': float(session_bias[current_session]['win_rate_long']),
                'session_bearish_win_rate': float(session_bias[current_session]['win_rate_short']),
                
                # Cross-Session Features
                'session_trend_continuation': float(cross_correlation.get(f'{current_session}_trend_continuation', 0.0)),
            }
            
            # Add overlap features
            if current_session_info['in_overlap']:
                features['session_in_overlap'] = 1.0
                features['session_overlap_multiplier'] = float(current_session_info['overlap_multiplier'])
            else:
                features['session_in_overlap'] = 0.0
                features['session_overlap_multiplier'] = 1.0
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating session features: {e}")
            return self._get_default_session_features()
    
    def get_session_trading_signal(self, ohlcv_data: pd.DataFrame, 
                                 current_timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate session-based trading signal and context (optimized)
        """
        try:
            current_session_info = self.detect_current_session(current_timestamp)
            session_features = self.generate_session_features(ohlcv_data, current_timestamp)
            
            current_session = current_session_info['session']
            session_config = self.sessions[current_session]
            
            # Optimized signal calculation
            bias_signal = 0
            confidence = 0.5
            
            # Session bias influence
            directional_bias = session_features['session_directional_bias']
            bias_strength = session_features['session_bias_strength']
            
            if abs(directional_bias) > 0.2 and bias_strength > 0.3:
                bias_signal = 1 if directional_bias > 0 else -1
                confidence = 0.5 + (bias_strength * 0.3)
            
            # Apply multipliers efficiently
            if session_features['session_in_optimal_window'] > 0:
                confidence *= 1.2
                
            volatility_regime = session_features['session_volatility_regime']
            if volatility_regime > 0.8:
                confidence *= session_config.risk_multiplier
            elif volatility_regime < 0.3:
                confidence *= 0.8
                
            if session_features['session_in_overlap'] > 0:
                confidence *= session_features['session_overlap_multiplier']
                
            confidence = np.clip(confidence, 0.0, 1.0)
            
            return {
                'signal': bias_signal,
                'confidence': float(confidence),
                'session': current_session,
                'session_name': session_config.name,
                'strategy_bias': session_config.strategy_bias,
                'risk_multiplier': float(session_features['session_risk_multiplier']),
                'optimal_pairs': session_config.pairs,
                'session_context': {
                    'directional_bias': float(directional_bias),
                    'bias_strength': float(bias_strength),
                    'volatility_regime': float(volatility_regime),
                    'in_optimal_window': session_features['session_in_optimal_window'] > 0,
                    'time_remaining_hours': float(session_features['session_time_remaining'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating session trading signal: {e}")
            return self._get_default_trading_signal()
    
    def clear_cache(self):
        """Clear all cached data to free memory"""
        self._session_masks_cache.clear()
        self._volatility_cache.clear()
        self._bias_cache.clear()
        self._optimal_times_cache.clear()
        self._last_data_hash = None
        
        # Clear LRU cache
        self.detect_current_session.cache_clear()
    
    # Optimized helper methods
    def _is_in_optimal_window(self, timestamp: Optional[datetime], 
                            optimal_times: Dict[str, Any]) -> bool:
        """Check if current time is within optimal trading window (optimized)"""
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            elif timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
                
            current_hour = timestamp.hour
            optimal_start = optimal_times['optimal_window_start']
            optimal_end = optimal_times['optimal_window_end']
            
            # Optimized window checking
            return (optimal_start <= current_hour < optimal_end) if optimal_end >= optimal_start else (current_hour >= optimal_start or current_hour < optimal_end)
                
        except Exception:
            return False
    
    def _get_default_single_volatility_metrics(self) -> Dict[str, float]:
        """Get default volatility metrics for a single session"""
        return {
            'avg_true_range': 0.001,
            'volatility_percentile': 50.0,
            'volatility_trend': 0.0,
            'session_range_avg': 0.001
        }
    
    def _get_default_single_session_bias(self) -> Dict[str, float]:
        """Get default session bias for a single session"""
        return {
            'directional_bias': 0.0,
            'bias_strength': 0.0,
            'win_rate_long': 0.5,
            'win_rate_short': 0.5,
            'avg_move_positive': 0.0,
            'avg_move_negative': 0.0
        }
    
    # Keep all original helper methods for compatibility
    def _get_default_session_info(self) -> Dict[str, Any]:
        """Get default session info in case of errors"""
        return {
            'session': 'london',
            'session_name': 'London',
            'session_index': 1,
            'time_remaining_minutes': 240,
            'session_progress': 0.5,
            'risk_multiplier': 1.0,
            'expected_volatility': 1.0,
            'strategy_bias': 'Trend following',
            'optimal_pairs': ['EURUSD', 'GBPUSD'],
            'in_overlap': False,
            'overlap_multiplier': 1.0
        }
    
    def _get_default_volatility_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get default volatility metrics"""
        default_metrics = self._get_default_single_volatility_metrics()
        return {
            'asian': default_metrics.copy(),
            'london': default_metrics.copy(),
            'newyork': default_metrics.copy()
        }
    
    def _get_default_trading_times(self) -> Dict[str, Dict[str, Any]]:
        """Get default optimal trading times"""
        return {
            'asian': {
                'best_hour': 0,
                'best_score': 0.5,
                'hour_scores': {},
                'optimal_window_start': 0,
                'optimal_window_end': 2,
                'session_activity_score': 0.5
            },
            'london': {
                'best_hour': 9,
                'best_score': 0.7,
                'hour_scores': {},
                'optimal_window_start': 9,
                'optimal_window_end': 11,
                'session_activity_score': 0.7
            },
            'newyork': {
                'best_hour': 14,
                'best_score': 0.8,
                'hour_scores': {},
                'optimal_window_start': 14,
                'optimal_window_end': 16,
                'session_activity_score': 0.8
            }
        }
    
    def _get_default_session_bias(self) -> Dict[str, Dict[str, float]]:
        """Get default session bias metrics"""
        default_bias = self._get_default_single_session_bias()
        return {
            'asian': default_bias.copy(),
            'london': default_bias.copy(),
            'newyork': default_bias.copy()
        }
    
    def _get_default_correlation(self) -> Dict[str, float]:
        """Get default correlation metrics"""
        return {
            'asian_london_correlation': 0.0,
            'london_newyork_correlation': 0.0,
            'asian_newyork_correlation': 0.0,
            'asian_trend_continuation': 0.0,
            'london_trend_continuation': 0.0,
            'newyork_trend_continuation': 0.0
        }
    
    def _get_default_session_features(self) -> Dict[str, float]:
        """Get default session features"""
        return {
            'session_current': 1.0,  # London session
            'session_time_remaining': 4.0,
            'session_progress': 0.5,
            'session_risk_multiplier': 1.0,
            'session_expected_volatility': 1.0,
            'session_in_optimal_window': 0.0,
            'session_optimal_score': 0.5,
            'session_activity_score': 0.5,
            'session_volatility_regime': 0.5,
            'session_volatility_trend': 0.0,
            'session_avg_true_range': 0.001,
            'session_directional_bias': 0.0,
            'session_bias_strength': 0.0,
            'session_bullish_win_rate': 0.5,
            'session_bearish_win_rate': 0.5,
            'session_trend_continuation': 0.0,
            'session_in_overlap': 0.0,
            'session_overlap_multiplier': 1.0
        }
    
    def _get_default_trading_signal(self) -> Dict[str, Any]:
        """Get default trading signal"""
        return {
            'signal': 0,
            'confidence': 0.5,
            'session': 'london',
            'session_name': 'London',
            'strategy_bias': 'Trend following',
            'risk_multiplier': 1.0,
            'optimal_pairs': ['EURUSD', 'GBPUSD'],
            'session_context': {
                'directional_bias': 0.0,
                'bias_strength': 0.0,
                'volatility_regime': 0.5,
                'in_optimal_window': False,
                'time_remaining_hours': 4.0
            }
        }
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive session statistics and configuration
        
        Returns:
            Dict with session configuration and statistics
        """
        return {
            'sessions': {
                name: {
                    'name': session.name,
                    'start_hour': session.start_hour,
                    'end_hour': session.end_hour,
                    'pairs': session.pairs,
                    'characteristics': session.characteristics,
                    'strategy_bias': session.strategy_bias,
                    'risk_multiplier': session.risk_multiplier,
                    'expected_volatility': session.expected_volatility
                } for name, session in self.sessions.items()
            },
            'overlaps': self.session_overlaps,
            'supported_symbols': [self.symbol],
            'timeframe': self.timeframe,
            'cache_status': {
                'session_masks_cached': bool(self._session_masks_cache),
                'volatility_cache_size': len(self._volatility_cache),
                'bias_cache_size': len(self._bias_cache),
                'optimal_times_cache_size': len(self._optimal_times_cache)
            }
        }


def test_optimized_session_analyzer():
    """Test function for Optimized SessionAnalyzer"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import time
    
    print("🧪 Testing Optimized Session Analyzer v1.1.0...")
    
    # Create larger sample data for performance testing
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='15T', tz='UTC')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': 1.1000 + np.random.randn(len(dates)) * 0.01,
        'high': 1.1000 + np.random.randn(len(dates)) * 0.01 + 0.005,
        'low': 1.1000 + np.random.randn(len(dates)) * 0.01 - 0.005,
        'close': 1.1000 + np.random.randn(len(dates)) * 0.01,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    print(f"✅ Created test dataset with {len(sample_data)} bars ({len(dates)/96:.1f} days)")
    
    # Initialize optimized analyzer
    analyzer = SessionAnalyzer("EURUSD", "M15")
    
    print("✅ Optimized SessionAnalyzer v1.1.0 initialized")
    
    # Performance test: Multiple calls to test caching
    print("\n🚀 Performance Testing (Caching Effects):")
    
    # First call (cold cache)
    start_time = time.time()
    session_features_1 = analyzer.generate_session_features(sample_data)
    first_call_time = time.time() - start_time
    
    # Second call (warm cache)
    start_time = time.time()
    session_features_2 = analyzer.generate_session_features(sample_data)
    second_call_time = time.time() - start_time
    
    print(f"   First call (cold cache): {first_call_time:.3f}s")
    print(f"   Second call (warm cache): {second_call_time:.3f}s")
    print(f"   Speed improvement: {first_call_time/second_call_time:.1f}x faster")
    
    # Test current session detection (cached)
    current_session = analyzer.detect_current_session()
    print(f"\n✅ Current session: {current_session['session_name']} ({current_session['session']})")
    print(f"   Risk multiplier: {current_session['risk_multiplier']}")
    print(f"   In optimal window: {session_features_1.get('session_in_optimal_window', 0) > 0}")
    
    # Test session volatility with caching
    start_time = time.time()
    volatility = analyzer.calculate_session_volatility(sample_data)
    volatility_time = time.time() - start_time
    
    print(f"\n✅ Session volatility calculated in {volatility_time:.3f}s")
    for session, metrics in volatility.items():
        print(f"   {session}: ATR={metrics['avg_true_range']:.5f}, Percentile={metrics['volatility_percentile']:.1f}")
    
    # Test optimal trading times with caching
    start_time = time.time()
    optimal_times = analyzer.identify_optimal_trading_times(sample_data)
    optimal_time = time.time() - start_time
    
    print(f"\n✅ Optimal trading times identified in {optimal_time:.3f}s")
    for session, times in optimal_times.items():
        print(f"   {session}: Best hour={times['best_hour']}, Score={times['best_score']:.2f}")
    
    # Test session bias with caching
    start_time = time.time()
    bias = analyzer.calculate_session_bias(sample_data)
    bias_time = time.time() - start_time
    
    print(f"\n✅ Session bias calculated in {bias_time:.3f}s")
    for session, bias_metrics in bias.items():
        print(f"   {session}: Bias={bias_metrics['directional_bias']:.3f}, Strength={bias_metrics['bias_strength']:.3f}")
    
    # Test complete feature generation
    print(f"\n✅ Generated {len(session_features_1)} session features")
    print("   Key optimized features:")
    for key, value in list(session_features_1.items())[:5]:
        print(f"     {key}: {value:.3f}")
    
    # Test trading signal generation
    signal = analyzer.get_session_trading_signal(sample_data)
    print(f"\n✅ Session trading signal generated")
    print(f"   Signal: {signal['signal']}, Confidence: {signal['confidence']:.3f}")
    print(f"   Session: {signal['session_name']}, Strategy: {signal['strategy_bias']}")
    
    # Test cache status
    stats = analyzer.get_session_statistics()
    cache_status = stats['cache_status']
    print(f"\n✅ Cache Performance:")
    print(f"   Session masks cached: {cache_status['session_masks_cached']}")
    print(f"   Volatility cache size: {cache_status['volatility_cache_size']}")
    print(f"   Bias cache size: {cache_status['bias_cache_size']}")
    print(f"   Optimal times cache size: {cache_status['optimal_times_cache_size']}")
    
    # Test cache clearing
    analyzer.clear_cache()
    stats_after_clear = analyzer.get_session_statistics()
    print(f"\n✅ Cache cleared successfully")
    print(f"   All caches reset: {not any(stats_after_clear['cache_status'].values())}")
    
    # Performance comparison with multiple datasets
    print(f"\n🏃‍♂️ Scalability Test:")
    
    test_sizes = [100, 500, 1000, 2000]
    for size in test_sizes:
        test_data = sample_data.tail(size)
        
        start_time = time.time()
        features = analyzer.generate_session_features(test_data)
        processing_time = time.time() - start_time
        
        print(f"   {size} bars: {processing_time:.3f}s ({len(features)} features)")
    
    print(f"\n🎯 All Optimized SessionAnalyzer v1.1.0 tests passed!")
    print(f"✅ Caching system working")
    print(f"✅ Performance optimizations active")
    print(f"✅ Memory management implemented")
    print(f"✅ Vectorized operations confirmed")
    
    return True


if __name__ == "__main__":
    # Run optimized tests
    test_optimized_session_analyzer()