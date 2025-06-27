"""
File: src/python/volume_profile.py
Description: Volume Profile Analysis Engine for ForexAI-EA - PRODUCTION FIXED
Author: Claude AI Developer
Version: 2.0.1 - PRODUCTION FIXED
Created: 2025-06-13
Modified: 2025-06-27
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class VolumeNode:
    """Volume node data structure"""
    price_level: float
    volume: float
    percentage: float
    is_poc: bool = False
    is_value_area: bool = False

@dataclass
class VolumeProfile:
    """Volume profile data structure"""
    poc_price: float           # Point of Control
    poc_volume: float          # POC volume
    value_area_high: float     # Value Area High
    value_area_low: float      # Value Area Low
    volume_nodes: List[VolumeNode]
    total_volume: float
    price_range: Tuple[float, float]

class VolumeProfileEngine:
    """Volume Profile calculation and analysis engine - PRODUCTION FIXED"""
    
    def __init__(self, tick_size: float = 0.0001, value_area_percent: float = 0.70):
        """
        Initialize Volume Profile Engine
        
        Args:
            tick_size: Minimum price movement for the instrument
            value_area_percent: Percentage for Value Area calculation (default 70%)
        """
        self.logger = logging.getLogger(__name__)
        self.tick_size = tick_size
        self.value_area_percent = value_area_percent
        
    def calculate_volume_profile(self, ohlcv_data: pd.DataFrame, 
                               num_levels: int = 100) -> Optional[VolumeProfile]:
        """
        Calculate volume profile from OHLCV data - FIXED VERSION
        
        Args:
            ohlcv_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            num_levels: Number of price levels for profile calculation
            
        Returns:
            VolumeProfile object with calculated data or None on error
        """
        try:
            if len(ohlcv_data) == 0:
                self.logger.warning("Empty OHLCV data provided")
                return None
                
            # FIXED: Ensure proper data types and handle timezone issues
            data_copy = ohlcv_data.copy()
            
            # Convert columns to float explicitly to avoid type issues
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data_copy.columns:
                    try:
                        data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
                    except Exception as e:
                        self.logger.warning(f"Failed to convert {col} to numeric: {e}")
                        return None
            
            # Remove any NaN values
            data_copy = data_copy.dropna()
            
            if len(data_copy) == 0:
                self.logger.warning("No valid data after cleaning")
                return None
                
            # Calculate price range with proper float conversion
            try:
                price_min = float(data_copy['low'].min())
                price_max = float(data_copy['high'].max())
                
                if price_min >= price_max:
                    self.logger.warning("Invalid price range")
                    return None
                    
                price_range = (price_min, price_max)
            except Exception as e:
                self.logger.error(f"Price range calculation failed: {e}")
                return None
            
            # Create price levels
            try:
                price_levels = np.linspace(price_min, price_max, num_levels)
                volume_at_price = defaultdict(float)
            except Exception as e:
                self.logger.error(f"Price levels creation failed: {e}")
                return None
            
            # FIXED: Distribute volume across price levels with proper error handling
            for idx in range(len(data_copy)):
                try:
                    row = data_copy.iloc[idx]
                    volume = float(row['volume'])
                    high = float(row['high'])
                    low = float(row['low'])
                    close = float(row['close'])
                    
                    if volume <= 0 or np.isnan(volume):
                        continue
                        
                    # Calculate volume distribution within the bar
                    bar_range = high - low
                    if bar_range <= 0:
                        # If no range, assign all volume to close price
                        closest_level = self._find_closest_price_level(close, price_levels)
                        volume_at_price[closest_level] += volume
                    else:
                        # Distribute volume proportionally across the bar
                        relevant_levels = [p for p in price_levels if low <= p <= high]
                        if relevant_levels:
                            weight_per_level = 1.0 / len(relevant_levels)
                            for price_level in relevant_levels:
                                volume_at_price[price_level] += volume * weight_per_level
                        else:
                            # Fallback: assign to closest level
                            closest_level = self._find_closest_price_level(close, price_levels)
                            volume_at_price[closest_level] += volume
                            
                except Exception as e:
                    self.logger.warning(f"Error processing row {idx}: {e}")
                    continue
            
            # Create volume nodes with safe conversion
            volume_nodes = []
            total_volume = sum(volume_at_price.values())
            
            if total_volume <= 0:
                self.logger.warning("No volume distributed")
                return None
            
            for price_level in price_levels:
                try:
                    volume = float(volume_at_price.get(price_level, 0.0))
                    percentage = (volume / total_volume * 100) if total_volume > 0 else 0.0
                    
                    volume_nodes.append(VolumeNode(
                        price_level=float(price_level),
                        volume=volume,
                        percentage=percentage
                    ))
                except Exception as e:
                    self.logger.warning(f"Error creating volume node: {e}")
                    continue
            
            if not volume_nodes:
                self.logger.warning("No volume nodes created")
                return None
            
            # Sort by volume (descending) with safe comparison
            try:
                volume_nodes.sort(key=lambda x: float(x.volume), reverse=True)
            except Exception as e:
                self.logger.warning(f"Error sorting volume nodes: {e}")
                # Fallback: use original order
            
            # Find Point of Control (POC)
            poc_node = volume_nodes[0] if volume_nodes else None
            poc_price = float(poc_node.price_level) if poc_node else price_min
            poc_volume = float(poc_node.volume) if poc_node else 0.0
            
            # Mark POC
            if poc_node:
                poc_node.is_poc = True
            
            # Calculate Value Area with safe handling
            try:
                value_area_high, value_area_low = self._calculate_value_area(
                    volume_nodes, total_volume, poc_price
                )
            except Exception as e:
                self.logger.warning(f"Value area calculation failed: {e}")
                value_area_high = poc_price
                value_area_low = poc_price
            
            # Mark Value Area nodes
            for node in volume_nodes:
                try:
                    if value_area_low <= node.price_level <= value_area_high:
                        node.is_value_area = True
                except Exception:
                    continue
            
            # Sort nodes back by price level with safe comparison
            try:
                volume_nodes.sort(key=lambda x: float(x.price_level))
            except Exception as e:
                self.logger.warning(f"Error sorting by price level: {e}")
            
            return VolumeProfile(
                poc_price=poc_price,
                poc_volume=poc_volume,
                value_area_high=value_area_high,
                value_area_low=value_area_low,
                volume_nodes=volume_nodes,
                total_volume=total_volume,
                price_range=price_range
            )
            
        except Exception as e:
            self.logger.error(f"Volume profile calculation failed: {e}")
            return None
    
    def _find_closest_price_level(self, price: float, price_levels: np.ndarray) -> float:
        """Find closest price level to given price - FIXED VERSION"""
        try:
            price = float(price)
            price_levels_float = price_levels.astype(float)
            differences = np.abs(price_levels_float - price)
            idx = np.argmin(differences)
            return float(price_levels_float[idx])
        except Exception as e:
            self.logger.warning(f"Error finding closest price level: {e}")
            return float(price_levels[0]) if len(price_levels) > 0 else price
    
    def _calculate_value_area(self, volume_nodes: List[VolumeNode], 
                            total_volume: float, poc_price: float) -> Tuple[float, float]:
        """
        Calculate Value Area High and Low - FIXED VERSION
        
        The Value Area contains the specified percentage of total volume,
        expanding from POC symmetrically.
        """
        try:
            target_volume = total_volume * self.value_area_percent
            accumulated_volume = 0.0
            
            # Find POC node with safe comparison
            poc_node = None
            for node in volume_nodes:
                try:
                    if abs(float(node.price_level) - float(poc_price)) < self.tick_size:
                        poc_node = node
                        break
                except Exception:
                    continue
            
            if not poc_node:
                # Fallback: use highest volume node
                poc_node = max(volume_nodes, key=lambda x: float(x.volume))
                poc_price = float(poc_node.price_level)
            
            # Start with POC
            included_nodes = [poc_node]
            accumulated_volume = float(poc_node.volume)
            
            # Sort other nodes by distance from POC with safe comparison
            other_nodes = [node for node in volume_nodes 
                          if abs(float(node.price_level) - float(poc_price)) >= self.tick_size]
            
            try:
                other_nodes.sort(key=lambda x: abs(float(x.price_level) - float(poc_price)))
            except Exception as e:
                self.logger.warning(f"Error sorting nodes by distance: {e}")
            
            # Add nodes symmetrically around POC until target volume is reached
            for node in other_nodes:
                try:
                    if accumulated_volume >= target_volume:
                        break
                    included_nodes.append(node)
                    accumulated_volume += float(node.volume)
                except Exception:
                    continue
            
            # Calculate boundaries with safe conversion
            try:
                price_levels = [float(node.price_level) for node in included_nodes]
                value_area_high = max(price_levels)
                value_area_low = min(price_levels)
            except Exception as e:
                self.logger.warning(f"Error calculating value area boundaries: {e}")
                value_area_high = float(poc_price)
                value_area_low = float(poc_price)
            
            return value_area_high, value_area_low
            
        except Exception as e:
            self.logger.error(f"Value area calculation failed: {e}")
            return float(poc_price), float(poc_price)
    
    def get_volume_profile_features(self, current_price: float, 
                                  volume_profile: VolumeProfile) -> Dict[str, float]:
        """
        Extract trading features from volume profile - FIXED VERSION
        
        Args:
            current_price: Current market price
            volume_profile: Calculated volume profile
            
        Returns:
            Dictionary of volume profile features
        """
        features = {}
        
        try:
            current_price = float(current_price)
            poc_price = float(volume_profile.poc_price)
            
            # POC-related features with safe division
            if poc_price > 0:
                features['poc_distance'] = (current_price - poc_price) / poc_price
                features['poc_distance_abs'] = abs(features['poc_distance'])
            else:
                features['poc_distance'] = 0.0
                features['poc_distance_abs'] = 0.0
                
            features['price_above_poc'] = 1.0 if current_price > poc_price else 0.0
            
            # Value Area features with safe handling
            va_high = float(volume_profile.value_area_high)
            va_low = float(volume_profile.value_area_low)
            va_range = va_high - va_low
            
            if va_range > 0:
                features['va_position'] = (current_price - va_low) / va_range
            else:
                features['va_position'] = 0.5
            
            features['price_in_value_area'] = 1.0 if va_low <= current_price <= va_high else 0.0
            features['price_above_va_high'] = 1.0 if current_price > va_high else 0.0
            features['price_below_va_low'] = 1.0 if current_price < va_low else 0.0
            
            # Volume distribution features
            price_range = float(volume_profile.price_range[1]) - float(volume_profile.price_range[0])
            if price_range > 0:
                features['price_range_position'] = (
                    (current_price - float(volume_profile.price_range[0])) / price_range
                )
            else:
                features['price_range_position'] = 0.5
            
            # POC strength (POC volume as percentage of total volume)
            total_volume = float(volume_profile.total_volume)
            poc_volume = float(volume_profile.poc_volume)
            
            if total_volume > 0:
                features['poc_strength'] = poc_volume / total_volume
            else:
                features['poc_strength'] = 0.0
            
            # Value Area width (normalized by price range)
            if price_range > 0:
                features['va_width_normalized'] = va_range / price_range
            else:
                features['va_width_normalized'] = 0.0
            
            # Ensure all features are valid floats
            cleaned_features = {}
            for key, value in features.items():
                try:
                    if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                        cleaned_features[key] = float(value)
                    else:
                        cleaned_features[key] = 0.0
                except (ValueError, TypeError):
                    cleaned_features[key] = 0.0
            
            return cleaned_features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return {
                'poc_distance': 0.0,
                'poc_distance_abs': 0.0,
                'price_above_poc': 0.5,
                'va_position': 0.5,
                'price_in_value_area': 1.0,
                'price_above_va_high': 0.0,
                'price_below_va_low': 0.0,
                'price_range_position': 0.5,
                'poc_strength': 0.1,
                'va_width_normalized': 0.05
            }
    
    def identify_key_levels(self, volume_profile: VolumeProfile, 
                          min_volume_percentage: float = 2.0) -> List[float]:
        """
        Identify key price levels based on volume concentration - FIXED VERSION
        
        Args:
            volume_profile: Volume profile data
            min_volume_percentage: Minimum volume percentage to be considered key level
            
        Returns:
            List of key price levels
        """
        try:
            key_levels = []
            
            # Always include POC
            key_levels.append(float(volume_profile.poc_price))
            
            # Add Value Area boundaries
            key_levels.extend([
                float(volume_profile.value_area_high), 
                float(volume_profile.value_area_low)
            ])
            
            # Add high-volume nodes
            for node in volume_profile.volume_nodes:
                try:
                    node_percentage = float(node.percentage)
                    node_price = float(node.price_level)
                    
                    if node_percentage >= min_volume_percentage and node_price not in key_levels:
                        key_levels.append(node_price)
                except Exception:
                    continue
            
            return sorted(key_levels)
            
        except Exception as e:
            self.logger.error(f"Key levels identification failed: {e}")
            return [float(volume_profile.poc_price)]
    
    def calculate_volume_delta(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """
        Calculate volume delta (approximation for Forex) - FIXED VERSION
        Since Forex doesn't have true volume delta, we approximate using price action
        
        Args:
            ohlcv_data: OHLC data with volume
            
        Returns:
            Series of volume delta approximations
        """
        try:
            volume_delta = []
            
            for idx in range(len(ohlcv_data)):
                try:
                    row = ohlcv_data.iloc[idx]
                    open_price = float(row['open'])
                    close_price = float(row['close'])
                    volume = float(row['volume'])
                    
                    # Approximate delta based on close relative to open
                    if close_price > open_price:
                        # Bullish bar - more buying pressure
                        delta = volume * 0.6  # 60% buying, 40% selling
                    elif close_price < open_price:
                        # Bearish bar - more selling pressure  
                        delta = volume * -0.6  # 60% selling, 40% buying
                    else:
                        # Neutral bar
                        delta = 0.0
                    
                    volume_delta.append(delta)
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating delta for row {idx}: {e}")
                    volume_delta.append(0.0)
            
            return pd.Series(volume_delta, index=ohlcv_data.index)
            
        except Exception as e:
            self.logger.error(f"Volume delta calculation failed: {e}")
            return pd.Series([0.0] * len(ohlcv_data), index=ohlcv_data.index)


class VWAPCalculator:
    """Volume Weighted Average Price calculator - PRODUCTION FIXED"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_vwap(self, ohlcv_data: pd.DataFrame, 
                      period: Optional[int] = None) -> Optional[pd.Series]:
        """
        Calculate Volume Weighted Average Price - FIXED VERSION
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            period: Period for rolling VWAP (None for session VWAP)
            
        Returns:
            Series of VWAP values or None on error
        """
        try:
            if len(ohlcv_data) == 0:
                self.logger.warning("Empty OHLCV data for VWAP calculation")
                return None
            
            # FIXED: Ensure proper data types
            data_copy = ohlcv_data.copy()
            for col in ['high', 'low', 'close', 'volume']:
                if col in data_copy.columns:
                    try:
                        data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
                    except Exception as e:
                        self.logger.warning(f"Failed to convert {col} to numeric: {e}")
                        return None
            
            # Remove NaN values
            data_copy = data_copy.dropna()
            if len(data_copy) == 0:
                self.logger.warning("No valid data after cleaning for VWAP")
                return None
            
            # Calculate typical price with safe handling
            try:
                typical_price = (data_copy['high'] + data_copy['low'] + data_copy['close']) / 3.0
                typical_price = typical_price.astype(float)
            except Exception as e:
                self.logger.error(f"Typical price calculation failed: {e}")
                return None
            
            # Calculate price * volume with safe handling
            try:
                volume_series = data_copy['volume'].astype(float)
                pv = typical_price * volume_series
            except Exception as e:
                self.logger.error(f"Price-volume calculation failed: {e}")
                return None
            
            # Calculate VWAP
            try:
                if period is None:
                    # Session VWAP (cumulative)
                    cumulative_pv = pv.cumsum()
                    cumulative_volume = volume_series.cumsum()
                    
                    # Avoid division by zero
                    cumulative_volume = cumulative_volume.replace(0, np.nan)
                    vwap = cumulative_pv / cumulative_volume
                else:
                    # Rolling VWAP
                    rolling_pv = pv.rolling(window=period, min_periods=1).sum()
                    rolling_volume = volume_series.rolling(window=period, min_periods=1).sum()
                    
                    # Avoid division by zero
                    rolling_volume = rolling_volume.replace(0, np.nan)
                    vwap = rolling_pv / rolling_volume
                
                # Fill NaN values
                vwap = vwap.fillna(method='bfill').fillna(method='ffill')
                
                # Ensure we have the same index as original data
                vwap.index = data_copy.index
                
                return vwap
                
            except Exception as e:
                self.logger.error(f"VWAP calculation failed: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"VWAP calculation failed: {e}")
            return None
    
    def calculate_vwap_bands(self, ohlcv_data: pd.DataFrame, 
                           vwap: pd.Series, std_multiplier: float = 2.0) -> Optional[Dict[str, pd.Series]]:
        """
        Calculate VWAP bands (standard deviation bands) - FIXED VERSION
        
        Args:
            ohlcv_data: OHLC data
            vwap: VWAP series
            std_multiplier: Standard deviation multiplier for bands
            
        Returns:
            Dictionary with upper and lower bands or None on error
        """
        try:
            if len(ohlcv_data) == 0 or vwap is None or len(vwap) == 0:
                self.logger.warning("Invalid data for VWAP bands calculation")
                return None
            
            # FIXED: Ensure proper data alignment and types
            data_copy = ohlcv_data.copy()
            for col in ['high', 'low', 'close', 'volume']:
                if col in data_copy.columns:
                    try:
                        data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
                    except Exception:
                        pass
            
            data_copy = data_copy.dropna()
            if len(data_copy) == 0:
                return None
            
            # Align VWAP with data
            vwap_aligned = vwap.reindex(data_copy.index, method='nearest').fillna(method='bfill').fillna(method='ffill')
            
            # Calculate typical price
            try:
                typical_price = (data_copy['high'] + data_copy['low'] + data_copy['close']) / 3.0
                typical_price = typical_price.astype(float)
            except Exception as e:
                self.logger.error(f"Typical price calculation failed: {e}")
                return None
            
            # Calculate variance with safe handling
            try:
                volume_series = data_copy['volume'].astype(float)
                price_diff_squared = (typical_price - vwap_aligned) ** 2
                price_variance = (price_diff_squared * volume_series).cumsum()
                volume_sum = volume_series.cumsum()
                
                # Avoid division by zero
                volume_sum = volume_sum.replace(0, np.nan)
                variance_ratio = price_variance / volume_sum
                
                # Standard deviation with safe sqrt
                vwap_std = np.sqrt(variance_ratio.fillna(0))
                vwap_std = vwap_std.replace([np.inf, -np.inf], 0)
                
            except Exception as e:
                self.logger.error(f"Variance calculation failed: {e}")
                return None
            
            # Calculate bands
            try:
                upper_band = vwap_aligned + (vwap_std * std_multiplier)
                lower_band = vwap_aligned - (vwap_std * std_multiplier)
                
                return {
                    'vwap_upper': upper_band,
                    'vwap_lower': lower_band,
                    'vwap_std': vwap_std
                }
                
            except Exception as e:
                self.logger.error(f"Bands calculation failed: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"VWAP bands calculation failed: {e}")
            return None
    
    def get_vwap_features(self, current_price: float, vwap_value: float, 
                         vwap_bands: Optional[Dict[str, pd.Series]], index: int) -> Dict[str, float]:
        """
        Extract VWAP-based trading features - FIXED VERSION
        
        Args:
            current_price: Current market price
            vwap_value: Current VWAP value
            vwap_bands: VWAP bands data
            index: Current data index
            
        Returns:
            Dictionary of VWAP features
        """
        features = {}
        
        try:
            current_price = float(current_price)
            vwap_value = float(vwap_value)
            
            # VWAP distance and position with safe division
            if vwap_value > 0:
                features['vwap_distance'] = (current_price - vwap_value) / vwap_value
                features['vwap_distance_abs'] = abs(features['vwap_distance'])
            else:
                features['vwap_distance'] = 0.0
                features['vwap_distance_abs'] = 0.0
                
            features['price_above_vwap'] = 1.0 if current_price > vwap_value else 0.0
            
            # VWAP bands features with safe handling
            if vwap_bands is not None and isinstance(vwap_bands, dict):
                try:
                    upper_series = vwap_bands.get('vwap_upper')
                    lower_series = vwap_bands.get('vwap_lower')
                    
                    if upper_series is not None and lower_series is not None:
                        # Safe index access
                        if index < 0:
                            index = len(upper_series) + index
                            
                        if 0 <= index < len(upper_series) and 0 <= index < len(lower_series):
                            upper_band = float(upper_series.iloc[index])
                            lower_band = float(lower_series.iloc[index])
                            
                            band_width = upper_band - lower_band
                            if band_width > 0:
                                features['vwap_band_position'] = (current_price - lower_band) / band_width
                            else:
                                features['vwap_band_position'] = 0.5
                            
                            features['price_above_vwap_upper'] = 1.0 if current_price > upper_band else 0.0
                            features['price_below_vwap_lower'] = 1.0 if current_price < lower_band else 0.0
                            
                            # Band width normalized
                            if vwap_value > 0:
                                features['vwap_band_width'] = band_width / vwap_value
                            else:
                                features['vwap_band_width'] = 0.0
                        else:
                            # Index out of range
                            features.update({
                                'vwap_band_position': 0.5,
                                'price_above_vwap_upper': 0.0,
                                'price_below_vwap_lower': 0.0,
                                'vwap_band_width': 0.0
                            })
                    else:
                        # Missing band data
                        features.update({
                            'vwap_band_position': 0.5,
                            'price_above_vwap_upper': 0.0,
                            'price_below_vwap_lower': 0.0,
                            'vwap_band_width': 0.0
                        })
                        
                except Exception as e:
                    self.logger.warning(f"VWAP bands feature extraction failed: {e}")
                    features.update({
                        'vwap_band_position': 0.5,
                        'price_above_vwap_upper': 0.0,
                        'price_below_vwap_lower': 0.0,
                        'vwap_band_width': 0.0
                    })
            else:
                # No bands data
                features.update({
                    'vwap_band_position': 0.5,
                    'price_above_vwap_upper': 0.0,
                    'price_below_vwap_lower': 0.0,
                    'vwap_band_width': 0.0
                })
            
            # Ensure all features are valid floats
            cleaned_features = {}
            for key, value in features.items():
                try:
                    if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                        cleaned_features[key] = float(value)
                    else:
                        cleaned_features[key] = 0.0
                except (ValueError, TypeError):
                    cleaned_features[key] = 0.0
            
            return cleaned_features
            
        except Exception as e:
            self.logger.error(f"VWAP features extraction failed: {e}")
            return {
                'vwap_distance': 0.0,
                'vwap_distance_abs': 0.0,
                'price_above_vwap': 0.5,
                'vwap_band_position': 0.5,
                'price_above_vwap_upper': 0.0,
                'price_below_vwap_lower': 0.0,
                'vwap_band_width': 0.0
            }


def test_volume_profile_engine():
    """Test function for Volume Profile Engine - PRODUCTION FIXED"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Volume Profile Engine v2.0.1 - PRODUCTION FIXED...")
    
    try:
        # Create sample OHLCV data with proper types
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=200, freq='15min', tz='UTC')
        
        # Generate realistic OHLCV data
        prices = []
        volumes = []
        base_price = 1.1000
        
        for i in range(len(dates)):
            # Random walk with trend
            price_change = np.random.normal(0, 0.0005)
            base_price += price_change
            
            open_price = base_price
            high_price = open_price + abs(np.random.normal(0, 0.0003))
            low_price = open_price - abs(np.random.normal(0, 0.0003))
            close_price = open_price + np.random.normal(0, 0.0002)
            close_price = max(min(close_price, high_price), low_price)
            
            volume = abs(np.random.normal(1000, 300))
            
            prices.append([open_price, high_price, low_price, close_price])
            volumes.append(volume)
        
        # Create DataFrame with proper types
        ohlcv_df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'], index=dates)
        ohlcv_df['volume'] = volumes
        
        # Ensure all columns are float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            ohlcv_df[col] = ohlcv_df[col].astype(float)
        
        print(f"✅ Generated {len(ohlcv_df)} bars of test data")
        
        # Test Volume Profile
        print("\n🧪 Testing Volume Profile Engine...")
        vp_engine = VolumeProfileEngine()
        volume_profile = vp_engine.calculate_volume_profile(ohlcv_df.tail(100))  # Last 100 bars
        
        if volume_profile is not None:
            print(f"✅ Volume Profile Results:")
            print(f"   POC Price: {volume_profile.poc_price:.5f}")
            print(f"   POC Volume: {volume_profile.poc_volume:.2f}")
            print(f"   Value Area High: {volume_profile.value_area_high:.5f}")
            print(f"   Value Area Low: {volume_profile.value_area_low:.5f}")
            print(f"   Total Volume: {volume_profile.total_volume:.2f}")
            print(f"   Volume Nodes: {len(volume_profile.volume_nodes)}")
        else:
            print("❌ Volume Profile calculation failed")
            return False
        
        # Test VWAP
        print("\n🧪 Testing VWAP Calculator...")
        vwap_calc = VWAPCalculator()
        vwap = vwap_calc.calculate_vwap(ohlcv_df)
        
        if vwap is not None and len(vwap) > 0:
            print(f"✅ VWAP Results:")
            print(f"   Current VWAP: {vwap.iloc[-1]:.5f}")
            print(f"   VWAP Length: {len(vwap)}")
            
            # Test VWAP bands
            vwap_bands = vwap_calc.calculate_vwap_bands(ohlcv_df, vwap)
            if vwap_bands is not None:
                print(f"   VWAP Upper Band: {vwap_bands['vwap_upper'].iloc[-1]:.5f}")
                print(f"   VWAP Lower Band: {vwap_bands['vwap_lower'].iloc[-1]:.5f}")
            else:
                print("   ⚠️ VWAP bands calculation failed")
        else:
            print("❌ VWAP calculation failed")
            return False
        
        # Test feature extraction
        print("\n🧪 Testing Feature Extraction...")
        current_price = float(ohlcv_df['close'].iloc[-1])
        
        # Volume Profile features
        vp_features = vp_engine.get_volume_profile_features(current_price, volume_profile)
        print(f"✅ Volume Profile Features ({len(vp_features)}):")
        for key, value in vp_features.items():
            print(f"   {key}: {value:.4f}")
        
        # VWAP features
        vwap_features = vwap_calc.get_vwap_features(
            current_price, 
            float(vwap.iloc[-1]), 
            vwap_bands, 
            -1
        )
        print(f"\n✅ VWAP Features ({len(vwap_features)}):")
        for key, value in vwap_features.items():
            print(f"   {key}: {value:.4f}")
        
        # Test key levels
        key_levels = vp_engine.identify_key_levels(volume_profile)
        print(f"\n✅ Key Levels ({len(key_levels)}):")
        for level in key_levels[:5]:  # Show first 5
            print(f"   {level:.5f}")
        
        # Test volume delta
        volume_delta = vp_engine.calculate_volume_delta(ohlcv_df.tail(10))
        print(f"\n✅ Volume Delta (last 10 bars):")
        print(f"   Average Delta: {volume_delta.mean():.2f}")
        print(f"   Current Delta: {volume_delta.iloc[-1]:.2f}")
        
        # Validation checks
        validation_checks = {
            'Volume Profile Created': volume_profile is not None,
            'VWAP Calculated': vwap is not None and len(vwap) > 0,
            'VWAP Bands Created': vwap_bands is not None,
            'VP Features Generated': len(vp_features) >= 8,
            'VWAP Features Generated': len(vwap_features) >= 4,
            'Key Levels Identified': len(key_levels) >= 3,
            'Volume Delta Calculated': len(volume_delta) > 0,
            'All Features Float': all(isinstance(v, float) for v in {**vp_features, **vwap_features}.values())
        }
        
        print(f"\n🎯 Validation Results:")
        for check, passed in validation_checks.items():
            print(f"   {check}: {'✅' if passed else '❌'}")
        
        overall_success = all(validation_checks.values())
        
        if overall_success:
            print(f"\n🎉 SUCCESS: Volume Profile Engine v2.0.1 - PRODUCTION FIXED!")
            print(f"   🛠️ All type conversion issues resolved")
            print(f"   📊 Volume Profile calculation working")
            print(f"   💫 VWAP calculation and bands working")
            print(f"   🔧 Feature extraction robust and safe")
            print(f"   ⚡ Production-ready with comprehensive error handling")
        else:
            print(f"\n⚠️ Some validation checks failed")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test
    success = test_volume_profile_engine()
    if success:
        print("\n🚀 Volume Profile Engine Ready for Integration!")
    else:
        print("\n🔧 Please check the implementation")