# src/python/feature_engineer.py
"""
Feature Engineering for ForexAI-EA Project
Transforms technical indicators into ML features and generates trading labels
Author: Claude AI Developer
Version: 1.0.0
Created: 2025-06-11
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from technical_indicators import TechnicalIndicators

class FeatureEngineer:
    """
    Feature engineering pipeline for forex trading AI model
    Converts raw indicators into meaningful ML features
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.indicators_engine = TechnicalIndicators()
        
    def create_features(self, ohlc_data: Dict[str, List[float]], 
                       indicators: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Create feature set from OHLC data and technical indicators
        
        Args:
            ohlc_data: Dictionary with OHLC price data
            indicators: Pre-calculated indicators (optional)
            
        Returns:
            Dictionary of engineered features for ML model
        """
        if indicators is None:
            indicators = self.indicators_engine.calculate_all_indicators(ohlc_data)
        
        # Get latest values
        latest_indicators = self.indicators_engine.get_latest_values(indicators)
        
        features = {}
        
        # === Trend Features ===
        features.update(self._create_trend_features(latest_indicators, indicators))
        
        # === Momentum Features ===
        features.update(self._create_momentum_features(latest_indicators, indicators))
        
        # === Volatility Features ===
        features.update(self._create_volatility_features(latest_indicators, indicators, ohlc_data))
        
        # === Price Action Features ===
        features.update(self._create_price_action_features(ohlc_data, latest_indicators))
        
        # === Crossover Features ===
        features.update(self._create_crossover_features(indicators))
        
        # === Statistical Features ===
        features.update(self._create_statistical_features(ohlc_data, indicators))
        
        # Clean features (remove NaN values)
        features = self._clean_features(features)
        
        self.logger.debug(f"Created {len(features)} features")
        return features
    
    def _create_trend_features(self, latest: Dict, indicators: Dict) -> Dict[str, float]:
        """Create trend-based features"""
        features = {}
        
        # EMA values and relationships
        if all(key in latest for key in ['ema_9', 'ema_21', 'ema_50']):
            features['ema_9'] = latest['ema_9']
            features['ema_21'] = latest['ema_21'] 
            features['ema_50'] = latest['ema_50']
            
            # EMA spreads (normalized)
            features['ema_9_21_spread'] = (latest['ema_9'] - latest['ema_21']) / latest['ema_21']
            features['ema_21_50_spread'] = (latest['ema_21'] - latest['ema_50']) / latest['ema_50']
            features['ema_9_50_spread'] = (latest['ema_9'] - latest['ema_50']) / latest['ema_50']
            
            # EMA alignment (all EMAs in same direction = stronger trend)
            ema_alignment = 0
            if latest['ema_9'] > latest['ema_21'] > latest['ema_50']:
                ema_alignment = 1  # Bullish alignment
            elif latest['ema_9'] < latest['ema_21'] < latest['ema_50']:
                ema_alignment = -1  # Bearish alignment
            features['ema_alignment'] = ema_alignment
        
        # EMA slopes (trend strength)
        if 'ema_9' in indicators:
            features['ema_9_slope'] = self._calculate_slope(indicators['ema_9'], 3)
            features['ema_21_slope'] = self._calculate_slope(indicators['ema_21'], 5)
        
        # MACD trend features
        if all(key in latest for key in ['macd_main', 'macd_signal', 'macd_histogram']):
            features['macd_main'] = latest['macd_main']
            features['macd_signal'] = latest['macd_signal']
            features['macd_histogram'] = latest['macd_histogram']
            
            # MACD momentum
            features['macd_momentum'] = 1 if latest['macd_histogram'] > 0 else -1
            features['macd_strength'] = abs(latest['macd_histogram'])
        
        return features
    
    def _create_momentum_features(self, latest: Dict, indicators: Dict) -> Dict[str, float]:
        """Create momentum-based features"""
        features = {}
        
        # RSI features
        if 'rsi' in latest:
            rsi_value = latest['rsi']
            features['rsi'] = rsi_value
            features['rsi_normalized'] = (rsi_value - 50) / 50  # Normalize to -1 to 1
            
            # RSI zones
            features['rsi_overbought'] = 1 if rsi_value > 70 else 0
            features['rsi_oversold'] = 1 if rsi_value < 30 else 0
            features['rsi_neutral'] = 1 if 30 <= rsi_value <= 70 else 0
            
            # RSI momentum
            if 'rsi' in indicators and len(indicators['rsi']) >= 3:
                rsi_slope = self._calculate_slope(indicators['rsi'], 3)
                features['rsi_momentum'] = rsi_slope
        
        # Stochastic features
        if all(key in latest for key in ['stoch_k', 'stoch_d']):
            features['stoch_k'] = latest['stoch_k']
            features['stoch_d'] = latest['stoch_d']
            features['stoch_spread'] = latest['stoch_k'] - latest['stoch_d']
            
            # Stochastic zones
            features['stoch_overbought'] = 1 if latest['stoch_k'] > 80 else 0
            features['stoch_oversold'] = 1 if latest['stoch_k'] < 20 else 0
        
        # Williams %R features
        if 'williams_r' in latest:
            williams_value = latest['williams_r']
            features['williams_r'] = williams_value
            features['williams_overbought'] = 1 if williams_value > -20 else 0
            features['williams_oversold'] = 1 if williams_value < -80 else 0
        
        return features
    
    def _create_volatility_features(self, latest: Dict, indicators: Dict, ohlc_data: Dict) -> Dict[str, float]:
        """Create volatility-based features"""
        features = {}
        
        # ATR features
        if 'atr' in latest and 'atr_normalized' in latest:
            features['atr'] = latest['atr']
            features['atr_normalized'] = latest['atr_normalized']
            
            # ATR trend (volatility expanding/contracting)
            if 'atr' in indicators and len(indicators['atr']) >= 5:
                atr_slope = self._calculate_slope(indicators['atr'], 5)
                features['atr_trend'] = atr_slope
                
                # Volatility regime
                current_atr = latest['atr']
                avg_atr = np.mean(indicators['atr'][-20:])  # 20-period average
                features['volatility_regime'] = current_atr / avg_atr if avg_atr > 0 else 1
        
        # Bollinger Bands features
        if all(key in latest for key in ['bb_upper', 'bb_lower', 'bb_position', 'bb_width']):
            features['bb_position'] = latest['bb_position']  # 0-1 scale within bands
            features['bb_width'] = latest['bb_width']
            
            # BB squeeze/expansion
            if 'bb_width' in indicators and len(indicators['bb_width']) >= 10:
                current_width = latest['bb_width']
                avg_width = np.mean(indicators['bb_width'][-20:])
                features['bb_squeeze'] = 1 if current_width < avg_width * 0.8 else 0
                features['bb_expansion'] = 1 if current_width > avg_width * 1.2 else 0
        
        # Price volatility features
        close_prices = np.array(ohlc_data['close'][-20:])  # Last 20 bars
        if len(close_prices) >= 10:
            features['price_volatility'] = np.std(close_prices) / np.mean(close_prices)
        
        return features
    
    def _create_price_action_features(self, ohlc_data: Dict, latest: Dict) -> Dict[str, float]:
        """Create price action features"""
        features = {}
        
        # Current bar characteristics
        if len(ohlc_data['open']) > 0:
            open_price = ohlc_data['open'][-1]
            high_price = ohlc_data['high'][-1]
            low_price = ohlc_data['low'][-1]
            close_price = ohlc_data['close'][-1]
            
            # Bar type
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range > 0:
                features['body_to_range_ratio'] = body_size / total_range
                features['upper_shadow_ratio'] = (high_price - max(open_price, close_price)) / total_range
                features['lower_shadow_ratio'] = (min(open_price, close_price) - low_price) / total_range
            
            # Candle direction
            features['candle_direction'] = 1 if close_price > open_price else -1
            features['candle_strength'] = body_size / close_price if close_price > 0 else 0
        
        # Multi-bar patterns
        if len(ohlc_data['close']) >= 5:
            close_prices = ohlc_data['close'][-5:]
            
            # Higher highs/lower lows pattern
            features['higher_highs'] = 1 if all(close_prices[i] >= close_prices[i-1] 
                                              for i in range(1, len(close_prices))) else 0
            features['lower_lows'] = 1 if all(close_prices[i] <= close_prices[i-1] 
                                            for i in range(1, len(close_prices))) else 0
        
        # Price position relative to moving averages
        if 'bb_middle' in latest:
            current_price = ohlc_data['close'][-1]
            features['price_vs_bb_middle'] = (current_price - latest['bb_middle']) / latest['bb_middle']
        
        return features
    
    def _create_crossover_features(self, indicators: Dict) -> Dict[str, float]:
        """Create crossover-based features"""
        features = {}
        
        # EMA crossovers
        if 'ema_9' in indicators and 'ema_21' in indicators:
            crossovers = self.indicators_engine.detect_crossovers(
                indicators['ema_9'], indicators['ema_21']
            )
            
            # Recent crossover signals (last 5 bars)
            recent_bullish = any(idx >= len(indicators['ema_9']) - 5 
                               for idx in crossovers['bullish'])
            recent_bearish = any(idx >= len(indicators['ema_9']) - 5 
                               for idx in crossovers['bearish'])
            
            features['ema_9_21_bullish_cross'] = 1 if recent_bullish else 0
            features['ema_9_21_bearish_cross'] = 1 if recent_bearish else 0
        
        # MACD crossovers
        if 'macd_main' in indicators and 'macd_signal' in indicators:
            macd_crossovers = self.indicators_engine.detect_crossovers(
                indicators['macd_main'], indicators['macd_signal']
            )
            
            recent_macd_bullish = any(idx >= len(indicators['macd_main']) - 3 
                                    for idx in macd_crossovers['bullish'])
            recent_macd_bearish = any(idx >= len(indicators['macd_main']) - 3 
                                    for idx in macd_crossovers['bearish'])
            
            features['macd_bullish_cross'] = 1 if recent_macd_bullish else 0
            features['macd_bearish_cross'] = 1 if recent_macd_bearish else 0
        
        # Stochastic crossovers
        if 'stoch_k' in indicators and 'stoch_d' in indicators:
            stoch_crossovers = self.indicators_engine.detect_crossovers(
                indicators['stoch_k'], indicators['stoch_d']
            )
            
            recent_stoch_bullish = any(idx >= len(indicators['stoch_k']) - 3 
                                     for idx in stoch_crossovers['bullish'])
            recent_stoch_bearish = any(idx >= len(indicators['stoch_k']) - 3 
                                     for idx in stoch_crossovers['bearish'])
            
            features['stoch_bullish_cross'] = 1 if recent_stoch_bullish else 0
            features['stoch_bearish_cross'] = 1 if recent_stoch_bearish else 0
        
        return features
    
    def _create_statistical_features(self, ohlc_data: Dict, indicators: Dict) -> Dict[str, float]:
        """Create statistical features"""
        features = {}
        
        # Price momentum features
        if len(ohlc_data['close']) >= 10:
            close_prices = np.array(ohlc_data['close'])
            
            # Rate of change
            features['roc_5'] = (close_prices[-1] - close_prices[-6]) / close_prices[-6] if len(close_prices) >= 6 else 0
            features['roc_10'] = (close_prices[-1] - close_prices[-11]) / close_prices[-11] if len(close_prices) >= 11 else 0
            
            # Z-score (how many standard deviations from mean)
            if len(close_prices) >= 20:
                recent_prices = close_prices[-20:]
                mean_price = np.mean(recent_prices)
                std_price = np.std(recent_prices)
                if std_price > 0:
                    features['price_zscore'] = (close_prices[-1] - mean_price) / std_price
        
        # Volume-price relationship (if volume available)
        if 'volume' in ohlc_data and len(ohlc_data['volume']) >= 10:
            volumes = np.array(ohlc_data['volume'][-10:])
            avg_volume = np.mean(volumes)
            current_volume = volumes[-1]
            features['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Market structure features
        if len(ohlc_data['high']) >= 10 and len(ohlc_data['low']) >= 10:
            highs = np.array(ohlc_data['high'][-10:])
            lows = np.array(ohlc_data['low'][-10:])
            
            # Support/resistance levels
            resistance_level = np.max(highs)
            support_level = np.min(lows)
            current_price = ohlc_data['close'][-1]
            
            features['distance_to_resistance'] = (resistance_level - current_price) / current_price
            features['distance_to_support'] = (current_price - support_level) / current_price
        
        return features
    
    def _calculate_slope(self, values: np.ndarray, period: int) -> float:
        """Calculate slope of values over specified period"""
        if len(values) < period:
            return 0.0
        
        recent_values = values[-period:]
        if np.any(np.isnan(recent_values)):
            return 0.0
        
        x = np.arange(period)
        slope = np.polyfit(x, recent_values, 1)[0]
        return float(slope)
    
    def _clean_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Clean features by handling NaN and infinite values"""
        cleaned = {}
        
        for key, value in features.items():
            if value is None or np.isnan(value) or np.isinf(value):
                cleaned[key] = 0.0  # Replace with neutral value
            else:
                cleaned[key] = float(value)
        
        return cleaned
    
    def generate_labels(self, ohlc_data: Dict[str, List[float]], 
                       atr_values: np.ndarray, 
                       lookahead: int = 10,
                       profit_threshold: float = 0.5,
                       sideways_threshold: float = 0.2) -> List[int]:
        """
        Generate trading labels based on future price movement
        
        Args:
            ohlc_data: OHLC price data
            atr_values: ATR values for dynamic thresholds
            lookahead: Bars to look ahead for labeling
            profit_threshold: Multiplier of ATR for profit target
            sideways_threshold: Multiplier of ATR for sideways movement
            
        Returns:
            List of labels: 1 (Buy), 0 (Hold), -1 (Sell)
        """
        close_prices = np.array(ohlc_data['close'])
        labels = []
        
        for i in range(len(close_prices) - lookahead):
            current_price = close_prices[i]
            current_atr = atr_values[i] if i < len(atr_values) and not np.isnan(atr_values[i]) else current_price * 0.001
            
            # Look ahead to find max high and min low
            future_prices = close_prices[i+1:i+1+lookahead]
            max_future_price = np.max(future_prices)
            min_future_price = np.min(future_prices)
            
            # Calculate potential profit/loss
            potential_profit = max_future_price - current_price
            potential_loss = current_price - min_future_price
            
            # Dynamic thresholds based on ATR
            profit_target = current_atr * profit_threshold
            sideways_range = current_atr * sideways_threshold
            
            # Generate label
            if potential_profit >= profit_target and potential_profit > potential_loss:
                labels.append(1)  # Buy signal
            elif potential_loss >= profit_target and potential_loss > potential_profit:
                labels.append(-1)  # Sell signal
            else:
                labels.append(0)  # Hold/sideways
        
        # Pad remaining values with 0 (Hold)
        labels.extend([0] * lookahead)
        
        return labels
    
    def prepare_training_data(self, historical_data: List[Dict], 
                            config: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare complete training dataset with features and labels
        
        Args:
            historical_data: List of OHLC data dictionaries
            config: Configuration for feature engineering
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        if config is None:
            config = {
                'lookahead': 10,
                'profit_threshold': 0.5,
                'sideways_threshold': 0.2,
                'min_bars': 100
            }
        
        all_features = []
        all_labels = []
        
        for data_chunk in historical_data:
            if len(data_chunk['close']) < config['min_bars']:
                self.logger.warning(f"Insufficient data: {len(data_chunk['close'])} bars")
                continue
            
            try:
                # Calculate indicators
                indicators = self.indicators_engine.calculate_all_indicators(data_chunk)
                
                # Calculate ATR for labeling
                atr_values = indicators.get('atr', np.ones(len(data_chunk['close'])) * 0.001)
                
                # Generate labels
                labels = self.generate_labels(
                    data_chunk, 
                    atr_values,
                    config['lookahead'],
                    config['profit_threshold'],
                    config['sideways_threshold']
                )
                
                # Create features for each bar
                for i in range(len(data_chunk['close'])):
                    # Prepare data slice up to current bar
                    current_ohlc = {
                        'open': data_chunk['open'][:i+1],
                        'high': data_chunk['high'][:i+1],
                        'low': data_chunk['low'][:i+1],
                        'close': data_chunk['close'][:i+1]
                    }
                    
                    # Only process if we have enough data
                    if len(current_ohlc['close']) >= 50:  # Minimum for reliable indicators
                        current_indicators = {}
                        for name, values in indicators.items():
                            current_indicators[name] = values[:i+1]
                        
                        features = self.create_features(current_ohlc, current_indicators)
                        
                        if features and i < len(labels):  # Ensure we have corresponding label
                            all_features.append(features)
                            all_labels.append(labels[i])
                
            except Exception as e:
                self.logger.error(f"Error processing data chunk: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid features generated from historical data")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        labels_series = pd.Series(all_labels)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        self.logger.info(f"Prepared training data: {len(features_df)} samples, {len(features_df.columns)} features")
        self.logger.info(f"Label distribution: {labels_series.value_counts().to_dict()}")
        
        return features_df, labels_series
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names"""
        return [
            # Trend features
            'ema_9', 'ema_21', 'ema_50',
            'ema_9_21_spread', 'ema_21_50_spread', 'ema_9_50_spread',
            'ema_alignment', 'ema_9_slope', 'ema_21_slope',
            'macd_main', 'macd_signal', 'macd_histogram',
            'macd_momentum', 'macd_strength',
            
            # Momentum features
            'rsi', 'rsi_normalized', 'rsi_overbought', 'rsi_oversold', 'rsi_neutral', 'rsi_momentum',
            'stoch_k', 'stoch_d', 'stoch_spread', 'stoch_overbought', 'stoch_oversold',
            'williams_r', 'williams_overbought', 'williams_oversold',
            
            # Volatility features
            'atr', 'atr_normalized', 'atr_trend', 'volatility_regime',
            'bb_position', 'bb_width', 'bb_squeeze', 'bb_expansion',
            'price_volatility',
            
            # Price action features
            'body_to_range_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
            'candle_direction', 'candle_strength',
            'higher_highs', 'lower_lows', 'price_vs_bb_middle',
            
            # Crossover features
            'ema_9_21_bullish_cross', 'ema_9_21_bearish_cross',
            'macd_bullish_cross', 'macd_bearish_cross',
            'stoch_bullish_cross', 'stoch_bearish_cross',
            
            # Statistical features
            'roc_5', 'roc_10', 'price_zscore', 'volume_ratio',
            'distance_to_resistance', 'distance_to_support'
        ]