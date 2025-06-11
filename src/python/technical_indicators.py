# src/python/technical_indicators.py
"""
Technical Indicators Engine for ForexAI-EA Project
Calculates various technical indicators for AI model features
Author: Claude AI Developer
Version: 1.0.0
Created: 2025-06-11
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculation engine
    Supports all major indicators used in forex trading
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_ema(self, prices: Union[List[float], np.ndarray], period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average (EMA)
        
        Args:
            prices: Price array (typically close prices)
            period: EMA period
            
        Returns:
            numpy array of EMA values
        """
        prices = np.array(prices, dtype=float)
        
        if len(prices) < period:
            raise ValueError(f"Insufficient data: need {period} prices, got {len(prices)}")
        
        # Calculate smoothing factor
        alpha = 2.0 / (period + 1)
        
        # Initialize EMA array
        ema = np.zeros_like(prices)
        ema[0] = prices[0]  # First EMA value equals first price
        
        # Calculate EMA values
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def calculate_sma(self, prices: Union[List[float], np.ndarray], period: int) -> np.ndarray:
        """
        Calculate Simple Moving Average (SMA)
        
        Args:
            prices: Price array
            period: SMA period
            
        Returns:
            numpy array of SMA values
        """
        prices = np.array(prices, dtype=float)
        
        if len(prices) < period:
            raise ValueError(f"Insufficient data: need {period} prices, got {len(prices)}")
        
        sma = np.zeros_like(prices)
        sma[:period-1] = np.nan  # Not enough data for SMA
        
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
        
        return sma
    
    def calculate_rsi(self, prices: Union[List[float], np.ndarray], period: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Price array (typically close prices)
            period: RSI period (default 14)
            
        Returns:
            numpy array of RSI values (0-100)
        """
        prices = np.array(prices, dtype=float)
        
        if len(prices) < period + 1:
            raise ValueError(f"Insufficient data: need {period + 1} prices, got {len(prices)}")
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial average gain and loss
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Initialize RSI array
        rsi = np.zeros(len(prices))
        rsi[:period] = np.nan  # Not enough data
        
        # Calculate RSI for remaining values
        for i in range(period, len(prices)):
            # Smoothed averages (Wilder's method)
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
            
            # Calculate RS and RSI
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: Union[List[float], np.ndarray], 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price array (typically close prices)
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' arrays
        """
        prices = np.array(prices, dtype=float)
        
        if len(prices) < slow + signal:
            raise ValueError(f"Insufficient data: need {slow + signal} prices, got {len(prices)}")
        
        # Calculate fast and slow EMAs
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        signal_line = self.calculate_ema(macd_line, signal)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, prices: Union[List[float], np.ndarray], 
                                 period: int = 20, std_dev: float = 2.0) -> Dict[str, np.ndarray]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price array (typically close prices)
            period: Moving average period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
            
        Returns:
            Dictionary with 'upper', 'middle', 'lower', and 'position' arrays
        """
        prices = np.array(prices, dtype=float)
        
        if len(prices) < period:
            raise ValueError(f"Insufficient data: need {period} prices, got {len(prices)}")
        
        # Calculate middle band (SMA)
        middle_band = self.calculate_sma(prices, period)
        
        # Calculate standard deviation
        std_array = np.zeros_like(prices)
        std_array[:period-1] = np.nan
        
        for i in range(period-1, len(prices)):
            std_array[i] = np.std(prices[i-period+1:i+1])
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * std_array)
        lower_band = middle_band - (std_dev * std_array)
        
        # Calculate price position within bands (0-1 scale)
        band_width = upper_band - lower_band
        position = np.where(band_width != 0, 
                           (prices - lower_band) / band_width, 
                           0.5)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'position': position,
            'width': band_width
        }
    
    def calculate_atr(self, high: Union[List[float], np.ndarray], 
                     low: Union[List[float], np.ndarray], 
                     close: Union[List[float], np.ndarray], 
                     period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high: High price array
            low: Low price array
            close: Close price array
            period: ATR period (default 14)
            
        Returns:
            numpy array of ATR values
        """
        high = np.array(high, dtype=float)
        low = np.array(low, dtype=float)
        close = np.array(close, dtype=float)
        
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("High, low, and close arrays must have same length")
        
        if len(high) < period + 1:
            raise ValueError(f"Insufficient data: need {period + 1} bars, got {len(high)}")
        
        # Calculate true range for each bar
        true_ranges = np.zeros(len(high))
        
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]  # Current high - current low
            tr2 = abs(high[i] - close[i-1])  # Current high - previous close
            tr3 = abs(low[i] - close[i-1])   # Current low - previous close
            true_ranges[i] = max(tr1, tr2, tr3)
        
        # First bar true range is just high - low
        true_ranges[0] = high[0] - low[0]
        
        # Calculate ATR using smoothed average
        atr = np.zeros_like(true_ranges)
        atr[:period-1] = np.nan
        
        # Initial ATR is simple average of first 'period' true ranges
        atr[period-1] = np.mean(true_ranges[:period])
        
        # Subsequent ATR values use Wilder's smoothing
        for i in range(period, len(true_ranges)):
            atr[i] = (atr[i-1] * (period - 1) + true_ranges[i]) / period
        
        return atr
    
    def calculate_stochastic(self, high: Union[List[float], np.ndarray], 
                           low: Union[List[float], np.ndarray], 
                           close: Union[List[float], np.ndarray], 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, np.ndarray]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High price array
            low: Low price array
            close: Close price array
            k_period: %K period (default 14)
            d_period: %D smoothing period (default 3)
            
        Returns:
            Dictionary with 'k' and 'd' arrays
        """
        high = np.array(high, dtype=float)
        low = np.array(low, dtype=float)
        close = np.array(close, dtype=float)
        
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("High, low, and close arrays must have same length")
        
        if len(high) < k_period:
            raise ValueError(f"Insufficient data: need {k_period} bars, got {len(high)}")
        
        # Calculate %K
        k_values = np.zeros_like(close)
        k_values[:k_period-1] = np.nan
        
        for i in range(k_period-1, len(close)):
            highest_high = np.max(high[i-k_period+1:i+1])
            lowest_low = np.min(low[i-k_period+1:i+1])
            
            if highest_high == lowest_low:
                k_values[i] = 50  # Avoid division by zero
            else:
                k_values[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D (smoothed %K)
        d_values = self.calculate_sma(k_values, d_period)
        
        return {
            'k': k_values,
            'd': d_values
        }
    
    def calculate_williams_r(self, high: Union[List[float], np.ndarray], 
                           low: Union[List[float], np.ndarray], 
                           close: Union[List[float], np.ndarray], 
                           period: int = 14) -> np.ndarray:
        """
        Calculate Williams %R
        
        Args:
            high: High price array
            low: Low price array
            close: Close price array
            period: Williams %R period (default 14)
            
        Returns:
            numpy array of Williams %R values (-100 to 0)
        """
        high = np.array(high, dtype=float)
        low = np.array(low, dtype=float)
        close = np.array(close, dtype=float)
        
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("High, low, and close arrays must have same length")
        
        if len(high) < period:
            raise ValueError(f"Insufficient data: need {period} bars, got {len(high)}")
        
        williams_r = np.zeros_like(close)
        williams_r[:period-1] = np.nan
        
        for i in range(period-1, len(close)):
            highest_high = np.max(high[i-period+1:i+1])
            lowest_low = np.min(low[i-period+1:i+1])
            
            if highest_high == lowest_low:
                williams_r[i] = -50  # Avoid division by zero
            else:
                williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        
        return williams_r
    
    def calculate_all_indicators(self, ohlc_data: Dict[str, List[float]], 
                               config: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Calculate all technical indicators for given OHLC data
        
        Args:
            ohlc_data: Dictionary with 'open', 'high', 'low', 'close' arrays
            config: Optional configuration for indicator parameters
            
        Returns:
            Dictionary containing all calculated indicators
        """
        if config is None:
            config = {
                'ema_periods': [9, 21, 50],
                'rsi_period': 14,
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bb': {'period': 20, 'std_dev': 2.0},
                'atr_period': 14,
                'stoch': {'k_period': 14, 'd_period': 3},
                'williams_period': 14
            }
        
        # Extract OHLC arrays
        open_prices = np.array(ohlc_data['open'], dtype=float)
        high_prices = np.array(ohlc_data['high'], dtype=float)
        low_prices = np.array(ohlc_data['low'], dtype=float)
        close_prices = np.array(ohlc_data['close'], dtype=float)
        
        results = {}
        
        try:
            # Calculate EMAs
            for period in config['ema_periods']:
                results[f'ema_{period}'] = self.calculate_ema(close_prices, period)
            
            # Calculate RSI
            results['rsi'] = self.calculate_rsi(close_prices, config['rsi_period'])
            
            # Calculate MACD
            macd_data = self.calculate_macd(
                close_prices, 
                config['macd']['fast'], 
                config['macd']['slow'], 
                config['macd']['signal']
            )
            results.update({
                'macd_main': macd_data['macd'],
                'macd_signal': macd_data['signal'],
                'macd_histogram': macd_data['histogram']
            })
            
            # Calculate Bollinger Bands
            bb_data = self.calculate_bollinger_bands(
                close_prices, 
                config['bb']['period'], 
                config['bb']['std_dev']
            )
            results.update({
                'bb_upper': bb_data['upper'],
                'bb_middle': bb_data['middle'],
                'bb_lower': bb_data['lower'],
                'bb_position': bb_data['position'],
                'bb_width': bb_data['width']
            })
            
            # Calculate ATR
            results['atr'] = self.calculate_atr(
                high_prices, low_prices, close_prices, config['atr_period']
            )
            
            # Calculate Stochastic
            stoch_data = self.calculate_stochastic(
                high_prices, low_prices, close_prices,
                config['stoch']['k_period'], config['stoch']['d_period']
            )
            results.update({
                'stoch_k': stoch_data['k'],
                'stoch_d': stoch_data['d']
            })
            
            # Calculate Williams %R
            results['williams_r'] = self.calculate_williams_r(
                high_prices, low_prices, close_prices, config['williams_period']
            )
            
            # Add derived indicators
            results['price_sma_ratio'] = close_prices / results['bb_middle']
            results['atr_normalized'] = results['atr'] / close_prices * 10000  # Normalized to pips
            
            self.logger.info(f"Successfully calculated {len(results)} technical indicators")
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            raise
        
        return results
    
    def get_latest_values(self, indicators: Dict[str, np.ndarray], 
                         lookback: int = 1) -> Dict[str, float]:
        """
        Get latest values from calculated indicators
        
        Args:
            indicators: Dictionary of indicator arrays
            lookback: Number of periods to look back (1 = current, 2 = previous, etc.)
            
        Returns:
            Dictionary of latest indicator values
        """
        latest_values = {}
        
        for name, values in indicators.items():
            if len(values) >= lookback:
                latest_value = values[-lookback]
                if not np.isnan(latest_value):
                    latest_values[name] = float(latest_value)
                else:
                    latest_values[name] = None
            else:
                latest_values[name] = None
        
        return latest_values
    
    def detect_crossovers(self, fast_line: np.ndarray, slow_line: np.ndarray) -> Dict[str, List[int]]:
        """
        Detect crossovers between two indicator lines
        
        Args:
            fast_line: Fast moving line (e.g., EMA 9)
            slow_line: Slow moving line (e.g., EMA 21)
            
        Returns:
            Dictionary with 'bullish' and 'bearish' crossover indices
        """
        bullish_crosses = []
        bearish_crosses = []
        
        for i in range(1, len(fast_line)):
            if (not np.isnan(fast_line[i]) and not np.isnan(slow_line[i]) and
                not np.isnan(fast_line[i-1]) and not np.isnan(slow_line[i-1])):
                
                # Bullish crossover: fast crosses above slow
                if fast_line[i-1] <= slow_line[i-1] and fast_line[i] > slow_line[i]:
                    bullish_crosses.append(i)
                
                # Bearish crossover: fast crosses below slow
                elif fast_line[i-1] >= slow_line[i-1] and fast_line[i] < slow_line[i]:
                    bearish_crosses.append(i)
        
        return {
            'bullish': bullish_crosses,
            'bearish': bearish_crosses
        }
    
    def calculate_indicator_slopes(self, indicators: Dict[str, np.ndarray], 
                                 period: int = 3) -> Dict[str, float]:
        """
        Calculate slopes (trend direction) for indicators
        
        Args:
            indicators: Dictionary of indicator arrays
            period: Period for slope calculation
            
        Returns:
            Dictionary of indicator slopes
        """
        slopes = {}
        
        for name, values in indicators.items():
            if len(values) >= period and not np.isnan(values[-period:]).any():
                # Calculate linear regression slope
                x = np.arange(period)
                y = values[-period:]
                slope = np.polyfit(x, y, 1)[0]
                slopes[f'{name}_slope'] = float(slope)
        
        return slopes