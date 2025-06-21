#!/usr/bin/env python3
"""
Debug Feature Count Script
Find out why Enhanced Feature Engineer only generates 88 features instead of 106+
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/python'))

from enhanced_feature_engineer import EnhancedFeatureEngineer

def create_test_data(bars=300):
    """Create test data similar to test suite"""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=bars, freq='15min', tz='UTC')
    
    prices = []
    volumes = []
    base_price = 1.1000
    
    for i, timestamp in enumerate(dates):
        # Session-based volatility patterns
        hour = timestamp.hour
        
        # Enhanced session modeling
        if 0 <= hour < 8:  # Asian Session
            volatility_mult = 0.7
            volume_mult = 0.6
            trend_strength = 0.3
        elif 8 <= hour < 17:  # London Session
            if 13 <= hour < 17:  # London-NY overlap
                volatility_mult = 1.3
                volume_mult = 1.4
                trend_strength = 1.0
            else:
                volatility_mult = 1.0
                volume_mult = 1.0
                trend_strength = 0.8
        else:  # New York Session
            volatility_mult = 1.1
            volume_mult = 1.2
            trend_strength = 0.9
        
        # Price movement with session patterns
        base_change = np.random.normal(0, 0.0008 * volatility_mult)
        session_bias = 0.00002 * np.sin(i / 50) * trend_strength
        base_price += base_change + session_bias
        
        # Generate OHLC with session characteristics
        open_price = base_price
        high_price = open_price + abs(np.random.normal(0, 0.0005 * volatility_mult))
        low_price = open_price - abs(np.random.normal(0, 0.0005 * volatility_mult))
        close_price = open_price + np.random.normal(0, 0.0003 * volatility_mult)
        close_price = max(min(close_price, high_price), low_price)
        
        # Volume with session patterns
        base_volume = 1000 * volume_mult
        volume = abs(np.random.normal(base_volume, 200))
        
        prices.append([open_price, high_price, low_price, close_price])
        volumes.append(volume)
    
    df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'], index=dates)
    df['volume'] = volumes
    
    return df

def debug_feature_generation():
    """Debug feature generation to find missing features"""
    print("🔍 DEBUGGING FEATURE GENERATION")
    print("=" * 60)
    
    # Create test data
    test_data = create_test_data(300)
    print(f"📊 Test data: {len(test_data)} bars")
    
    # Initialize feature engineer
    fe = EnhancedFeatureEngineer("EURUSD", "M15")
    current_timestamp = datetime.now(timezone.utc)
    
    print("\n🧪 Testing individual feature components...")
    
    # Test each component individually
    try:
        # 1. Technical indicators
        print("\n1. Technical Indicators:")
        technical_features = fe._get_technical_features(test_data)
        tech_count = len(technical_features)
        print(f"   📈 Technical features: {tech_count}")
        if tech_count < 25:
            print("   ⚠️ Technical features below expected 25+")
            print(f"   🔍 Technical features: {list(technical_features.keys())}")
    except Exception as e:
        print(f"   ❌ Technical indicators failed: {e}")
        tech_count = 0

    try:
        # 2. Volume Profile
        print("\n2. Volume Profile:")
        if hasattr(fe, '_get_volume_profile_features'):
            vp_features = fe._get_volume_profile_features(test_data)
            vp_count = len(vp_features)
            print(f"   📊 Volume Profile features: {vp_count}")
            if vp_count < 20:
                print("   ⚠️ Volume Profile features below expected 20+")
        else:
            print("   ❌ Volume Profile method not found")
            vp_count = 0
    except Exception as e:
        print(f"   ❌ Volume Profile failed: {e}")
        vp_count = 0

    try:
        # 3. VWAP
        print("\n3. VWAP:")
        if hasattr(fe, '_get_vwap_features'):
            vwap_features = fe._get_vwap_features(test_data)
            vwap_count = len(vwap_features)
            print(f"   📈 VWAP features: {vwap_count}")
            if vwap_count < 20:
                print("   ⚠️ VWAP features below expected 20+")
        else:
            print("   ❌ VWAP method not found")
            vwap_count = 0
    except Exception as e:
        print(f"   ❌ VWAP failed: {e}")
        vwap_count = 0

    try:
        # 4. SMC
        print("\n4. Smart Money Concepts:")
        if hasattr(fe, '_get_smc_features'):
            smc_features = fe._get_smc_features(test_data)
            smc_count = len(smc_features)
            print(f"   🏢 SMC features: {smc_count}")
            if smc_count < 23:
                print("   ⚠️ SMC features below expected 23+")
        else:
            print("   ❌ SMC method not found")
            smc_count = 0
    except Exception as e:
        print(f"   ❌ SMC failed: {e}")
        smc_count = 0

    try:
        # 5. Session Intelligence
        print("\n5. Session Intelligence:")
        if hasattr(fe, '_get_session_features'):
            session_features = fe._get_session_features(test_data, current_timestamp)
            session_count = len(session_features)
            print(f"   🌍 Session features: {session_count}")
            if session_count < 18:
                print("   ⚠️ Session features below expected 18+")
                print(f"   🔍 Session features: {list(session_features.keys())}")
        else:
            print("   ❌ Session features method not found")
            session_count = 0
    except Exception as e:
        print(f"   ❌ Session features failed: {e}")
        session_count = 0

    try:
        # 6. Advanced/Combination features
        print("\n6. Advanced Features:")
        if hasattr(fe, '_get_advanced_features'):
            advanced_features = fe._get_advanced_features(test_data)
            advanced_count = len(advanced_features)
            print(f"   🔬 Advanced features: {advanced_count}")
        else:
            print("   ❌ Advanced features method not found")
            advanced_count = 0
    except Exception as e:
        print(f"   ❌ Advanced features failed: {e}")
        advanced_count = 0

    # Calculate total expected
    expected_total = tech_count + vp_count + vwap_count + smc_count + session_count + advanced_count
    print(f"\n📊 COMPONENT BREAKDOWN:")
    print(f"   Technical: {tech_count}/25+ expected")
    print(f"   Volume Profile: {vp_count}/20+ expected") 
    print(f"   VWAP: {vwap_count}/20+ expected")
    print(f"   SMC: {smc_count}/23+ expected")
    print(f"   Session: {session_count}/18+ expected")
    print(f"   Advanced: {advanced_count}/10+ expected")
    print(f"   TOTAL EXPECTED: {expected_total}")

    # Now test complete feature generation
    print(f"\n🧪 Testing complete feature generation...")
    try:
        all_features = fe.create_enhanced_features(test_data, current_timestamp)
        actual_total = len(all_features)
        
        # Categorize actual features
        actual_session = len([k for k in all_features.keys() if k.startswith('session_')])
        actual_smc = len([k for k in all_features.keys() if k.startswith('smc_')])
        actual_vp = len([k for k in all_features.keys() if k.startswith('vp_')])
        actual_vwap = len([k for k in all_features.keys() if k.startswith('vwap_')])
        actual_tech = len([k for k in all_features.keys() if any(prefix in k for prefix in ['ema_', 'rsi', 'macd_', 'bb_', 'atr', 'stoch_', 'williams_'])])
        actual_advanced = actual_total - (actual_session + actual_smc + actual_vp + actual_vwap + actual_tech)
        
        print(f"\n📊 ACTUAL BREAKDOWN:")
        print(f"   Technical: {actual_tech}")
        print(f"   Volume Profile: {actual_vp}")
        print(f"   VWAP: {actual_vwap}")
        print(f"   SMC: {actual_smc}")
        print(f"   Session: {actual_session}")
        print(f"   Advanced: {actual_advanced}")
        print(f"   TOTAL ACTUAL: {actual_total}")
        
        # Show difference
        print(f"\n⚖️ COMPARISON:")
        print(f"   Expected: {expected_total}")
        print(f"   Actual: {actual_total}")
        print(f"   Difference: {actual_total - expected_total}")
        
        if actual_total < 106:
            print(f"\n❌ MISSING FEATURES: {106 - actual_total}")
            print("🔍 Feature name analysis:")
            
            # Show some feature names for debugging
            feature_names = list(all_features.keys())
            for category in ['session_', 'smc_', 'vp_', 'vwap_', 'ema_', 'rsi']:
                category_features = [f for f in feature_names if f.startswith(category)]
                print(f"   {category}: {len(category_features)} features")
                if len(category_features) <= 5:
                    print(f"      {category_features}")
                else:
                    print(f"      {category_features[:3]}...{category_features[-2:]}")
        else:
            print(f"\n✅ FEATURE COUNT OK: {actual_total} >= 106")
            
    except Exception as e:
        print(f"❌ Complete feature generation failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function"""
    debug_feature_generation()
    
    print(f"\n🎯 SUMMARY:")
    print("If any component shows 0 features or errors:")
    print("1. Check if the method exists in enhanced_feature_engineer.py")
    print("2. Check for import errors or missing dependencies")
    print("3. Check data format requirements")
    print("4. Check if fallback mechanisms are working")

if __name__ == "__main__":
    main()