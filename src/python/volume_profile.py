"""
File: src/python/volume_profile.py
Description: Volume Profile Analysis Engine for ForexAI-EA
Author: Claude AI Developer
Version: 2.0.0
Created: 2025-06-13
Modified: 2025-06-13
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict

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
    """Volume Profile calculation and analysis engine"""
    
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
                               num_levels: int = 100) -> VolumeProfile:
        """
        Calculate volume profile from OHLCV data
        
        Args:
            ohlcv_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            num_levels: Number of price levels for profile calculation
            
        Returns:
            VolumeProfile object with calculated data
        """
        try:
            if len(ohlcv_data) == 0:
                raise ValueError("Empty OHLCV data provided")
                
            # Calculate price range
            price_min = ohlcv_data['low'].min()
            price_max = ohlcv_data['high'].max()
            price_range = (price_min, price_max)
            
            # Create price levels
            price_levels = np.linspace(price_min, price_max, num_levels)
            volume_at_price = defaultdict(float)
            
            # Distribute volume across price levels
            for _, row in ohlcv_data.iterrows():
                volume = row['volume']
                high = row['high']
                low = row['low']
                
                # Calculate volume distribution within the bar
                bar_range = high - low
                if bar_range == 0:
                    # If no range, assign all volume to close price
                    closest_level = self._find_closest_price_level(row['close'], price_levels)
                    volume_at_price[closest_level] += volume
                else:
                    # Distribute volume proportionally across the bar
                    for price_level in price_levels:
                        if low <= price_level <= high:
                            # Simple linear distribution (can be enhanced with TPO logic)
                            weight = 1.0 / len([p for p in price_levels if low <= p <= high])
                            volume_at_price[price_level] += volume * weight
            
            # Create volume nodes
            volume_nodes = []
            total_volume = sum(volume_at_price.values())
            
            for price_level in price_levels:
                volume = volume_at_price[price_level]
                percentage = (volume / total_volume * 100) if total_volume > 0 else 0
                
                volume_nodes.append(VolumeNode(
                    price_level=price_level,
                    volume=volume,
                    percentage=percentage
                ))
            
            # Sort by volume (descending)
            volume_nodes.sort(key=lambda x: x.volume, reverse=True)
            
            # Find Point of Control (POC)
            poc_node = volume_nodes[0] if volume_nodes else None
            poc_price = poc_node.price_level if poc_node else price_min
            poc_volume = poc_node.volume if poc_node else 0
            
            # Mark POC
            if poc_node:
                poc_node.is_poc = True
            
            # Calculate Value Area
            value_area_high, value_area_low = self._calculate_value_area(
                volume_nodes, total_volume, poc_price
            )
            
            # Mark Value Area nodes
            for node in volume_nodes:
                if value_area_low <= node.price_level <= value_area_high:
                    node.is_value_area = True
            
            # Sort nodes back by price level
            volume_nodes.sort(key=lambda x: x.price_level)
            
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
            raise
    
    def _find_closest_price_level(self, price: float, price_levels: np.ndarray) -> float:
        """Find closest price level to given price"""
        idx = np.argmin(np.abs(price_levels - price))
        return price_levels[idx]
    
    def _calculate_value_area(self, volume_nodes: List[VolumeNode], 
                            total_volume: float, poc_price: float) -> Tuple[float, float]:
        """
        Calculate Value Area High and Low
        
        The Value Area contains the specified percentage of total volume,
        expanding from POC symmetrically.
        """
        target_volume = total_volume * self.value_area_percent
        accumulated_volume = 0
        
        # Find POC node
        poc_node = next((node for node in volume_nodes if node.price_level == poc_price), None)
        if not poc_node:
            return poc_price, poc_price
        
        # Start with POC
        included_nodes = [poc_node]
        accumulated_volume = poc_node.volume
        
        # Sort nodes by distance from POC
        other_nodes = [node for node in volume_nodes if node.price_level != poc_price]
        other_nodes.sort(key=lambda x: abs(x.price_level - poc_price))
        
        # Add nodes symmetrically around POC until target volume is reached
        for node in other_nodes:
            if accumulated_volume >= target_volume:
                break
            included_nodes.append(node)
            accumulated_volume += node.volume
        
        # Calculate boundaries
        price_levels = [node.price_level for node in included_nodes]
        value_area_high = max(price_levels)
        value_area_low = min(price_levels)
        
        return value_area_high, value_area_low
    
    def get_volume_profile_features(self, current_price: float, 
                                  volume_profile: VolumeProfile) -> Dict[str, float]:
        """
        Extract trading features from volume profile
        
        Args:
            current_price: Current market price
            volume_profile: Calculated volume profile
            
        Returns:
            Dictionary of volume profile features
        """
        features = {}
        
        # POC-related features
        features['poc_distance'] = (current_price - volume_profile.poc_price) / volume_profile.poc_price
        features['poc_distance_abs'] = abs(features['poc_distance'])
        features['price_above_poc'] = 1.0 if current_price > volume_profile.poc_price else 0.0
        
        # Value Area features
        va_range = volume_profile.value_area_high - volume_profile.value_area_low
        if va_range > 0:
            features['va_position'] = (current_price - volume_profile.value_area_low) / va_range
        else:
            features['va_position'] = 0.5
        
        features['price_in_value_area'] = 1.0 if (
            volume_profile.value_area_low <= current_price <= volume_profile.value_area_high
        ) else 0.0
        
        features['price_above_va_high'] = 1.0 if current_price > volume_profile.value_area_high else 0.0
        features['price_below_va_low'] = 1.0 if current_price < volume_profile.value_area_low else 0.0
        
        # Volume distribution features
        price_range = volume_profile.price_range[1] - volume_profile.price_range[0]
        if price_range > 0:
            features['price_range_position'] = (
                (current_price - volume_profile.price_range[0]) / price_range
            )
        else:
            features['price_range_position'] = 0.5
        
        # POC strength (POC volume as percentage of total volume)
        features['poc_strength'] = (
            volume_profile.poc_volume / volume_profile.total_volume
            if volume_profile.total_volume > 0 else 0
        )
        
        # Value Area width (normalized by price range)
        if price_range > 0:
            features['va_width_normalized'] = va_range / price_range
        else:
            features['va_width_normalized'] = 0
        
        return features
    
    def identify_key_levels(self, volume_profile: VolumeProfile, 
                          min_volume_percentage: float = 2.0) -> List[float]:
        """
        Identify key price levels based on volume concentration
        
        Args:
            volume_profile: Volume profile data
            min_volume_percentage: Minimum volume percentage to be considered key level
            
        Returns:
            List of key price levels
        """
        key_levels = []
        
        # Always include POC
        key_levels.append(volume_profile.poc_price)
        
        # Add Value Area boundaries
        key_levels.extend([volume_profile.value_area_high, volume_profile.value_area_low])
        
        # Add high-volume nodes
        for node in volume_profile.volume_nodes:
            if node.percentage >= min_volume_percentage and node.price_level not in key_levels:
                key_levels.append(node.price_level)
        
        return sorted(key_levels)
    
    def calculate_volume_delta(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """
        Calculate volume delta (approximation for Forex)
        Since Forex doesn't have true volume delta, we approximate using price action
        
        Args:
            ohlcv_data: OHLC data with volume
            
        Returns:
            Series of volume delta approximations
        """
        volume_delta = []
        
        for _, row in ohlcv_data.iterrows():
            open_price = row['open']
            close_price = row['close']
            volume = row['volume']
            
            # Approximate delta based on close relative to open
            if close_price > open_price:
                # Bullish bar - more buying pressure
                delta = volume * 0.6  # 60% buying, 40% selling
            elif close_price < open_price:
                # Bearish bar - more selling pressure  
                delta = volume * -0.6  # 60% selling, 40% buying
            else:
                # Neutral bar
                delta = 0
            
            volume_delta.append(delta)
        
        return pd.Series(volume_delta, index=ohlcv_data.index)


class VWAPCalculator:
    """Volume Weighted Average Price calculator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_vwap(self, ohlcv_data: pd.DataFrame, 
                      period: Optional[int] = None) -> pd.Series:
        """
        Calculate Volume Weighted Average Price
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            period: Period for rolling VWAP (None for session VWAP)
            
        Returns:
            Series of VWAP values
        """
        try:
            # Calculate typical price
            typical_price = (ohlcv_data['high'] + ohlcv_data['low'] + ohlcv_data['close']) / 3
            
            # Calculate price * volume
            pv = typical_price * ohlcv_data['volume']
            
            if period is None:
                # Session VWAP (cumulative)
                cumulative_pv = pv.cumsum()
                cumulative_volume = ohlcv_data['volume'].cumsum()
                vwap = cumulative_pv / cumulative_volume
            else:
                # Rolling VWAP
                rolling_pv = pv.rolling(window=period).sum()
                rolling_volume = ohlcv_data['volume'].rolling(window=period).sum()
                vwap = rolling_pv / rolling_volume
            
            return vwap
            
        except Exception as e:
            self.logger.error(f"VWAP calculation failed: {e}")
            raise
    
    def calculate_vwap_bands(self, ohlcv_data: pd.DataFrame, 
                           vwap: pd.Series, std_multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate VWAP bands (standard deviation bands)
        
        Args:
            ohlcv_data: OHLC data
            vwap: VWAP series
            std_multiplier: Standard deviation multiplier for bands
            
        Returns:
            Dictionary with upper and lower bands
        """
        try:
            typical_price = (ohlcv_data['high'] + ohlcv_data['low'] + ohlcv_data['close']) / 3
            
            # Calculate variance
            price_variance = ((typical_price - vwap) ** 2 * ohlcv_data['volume']).cumsum()
            volume_sum = ohlcv_data['volume'].cumsum()
            
            # Standard deviation
            vwap_std = np.sqrt(price_variance / volume_sum)
            
            # Calculate bands
            upper_band = vwap + (vwap_std * std_multiplier)
            lower_band = vwap - (vwap_std * std_multiplier)
            
            return {
                'vwap_upper': upper_band,
                'vwap_lower': lower_band,
                'vwap_std': vwap_std
            }
            
        except Exception as e:
            self.logger.error(f"VWAP bands calculation failed: {e}")
            raise
    
    def get_vwap_features(self, current_price: float, vwap_value: float, 
                         vwap_bands: Dict[str, pd.Series], index: int) -> Dict[str, float]:
        """
        Extract VWAP-based trading features
        
        Args:
            current_price: Current market price
            vwap_value: Current VWAP value
            vwap_bands: VWAP bands data
            index: Current data index
            
        Returns:
            Dictionary of VWAP features
        """
        features = {}
        
        # VWAP distance and position
        features['vwap_distance'] = (current_price - vwap_value) / vwap_value
        features['vwap_distance_abs'] = abs(features['vwap_distance'])
        features['price_above_vwap'] = 1.0 if current_price > vwap_value else 0.0
        
        # VWAP bands features
        if len(vwap_bands['vwap_upper']) > index:
            upper_band = vwap_bands['vwap_upper'].iloc[index]
            lower_band = vwap_bands['vwap_lower'].iloc[index]
            
            band_width = upper_band - lower_band
            if band_width > 0:
                features['vwap_band_position'] = (current_price - lower_band) / band_width
            else:
                features['vwap_band_position'] = 0.5
            
            features['price_above_vwap_upper'] = 1.0 if current_price > upper_band else 0.0
            features['price_below_vwap_lower'] = 1.0 if current_price < lower_band else 0.0
            
            # Band width normalized
            features['vwap_band_width'] = band_width / vwap_value if vwap_value > 0 else 0
        else:
            features['vwap_band_position'] = 0.5
            features['price_above_vwap_upper'] = 0.0
            features['price_below_vwap_lower'] = 0.0
            features['vwap_band_width'] = 0.0
        
        return features


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Create sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=1000, freq='15min')
    
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
    
    # Create DataFrame
    ohlcv_df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'], index=dates)
    ohlcv_df['volume'] = volumes
    
    print("🧪 Testing Volume Profile Engine...")
    
    # Test Volume Profile
    vp_engine = VolumeProfileEngine()
    volume_profile = vp_engine.calculate_volume_profile(ohlcv_df[-100:])  # Last 100 bars
    
    print(f"✅ Volume Profile Results:")
    print(f"   POC Price: {volume_profile.poc_price:.5f}")
    print(f"   POC Volume: {volume_profile.poc_volume:.2f}")
    print(f"   Value Area High: {volume_profile.value_area_high:.5f}")
    print(f"   Value Area Low: {volume_profile.value_area_low:.5f}")
    print(f"   Total Volume: {volume_profile.total_volume:.2f}")
    
    # Test VWAP
    vwap_calc = VWAPCalculator()
    vwap = vwap_calc.calculate_vwap(ohlcv_df)
    vwap_bands = vwap_calc.calculate_vwap_bands(ohlcv_df, vwap)
    
    print(f"\n✅ VWAP Results:")
    print(f"   Current VWAP: {vwap.iloc[-1]:.5f}")
    print(f"   VWAP Upper Band: {vwap_bands['vwap_upper'].iloc[-1]:.5f}")
    print(f"   VWAP Lower Band: {vwap_bands['vwap_lower'].iloc[-1]:.5f}")
    
    # Test feature extraction
    current_price = ohlcv_df['close'].iloc[-1]
    vp_features = vp_engine.get_volume_profile_features(current_price, volume_profile)
    vwap_features = vwap_calc.get_vwap_features(current_price, vwap.iloc[-1], vwap_bands, -1)
    
    print(f"\n✅ Volume Profile Features:")
    for key, value in vp_features.items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\n✅ VWAP Features:")
    for key, value in vwap_features.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n🎯 Volume Profile Engine Ready for Integration!")