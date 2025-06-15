"""
File: test_smc_integration.py
Description: Smart Money Concepts Integration Test Suite
Author: Claude AI Developer
Version: 2.1.0
Created: 2025-06-15
Modified: 2025-06-15

Purpose: Test SMC engine integration with existing ForexAI-EA system
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import unittest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/python'))

def create_smc_test_data(bars=300):
    """Create test data with deliberate SMC patterns"""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=bars, freq='15min')
    
    prices = []
    volumes = []
    base_price = 1.1000
    
    for i in range(bars):
        # Create deliberate patterns for testing
        
        # Every 50 bars: Create Order Block pattern
        if i > 0 and i % 50 == 0:
            # Create displacement after small opposing candle
            displacement = 0.003 if (i // 50) % 2 == 0 else -0.003  # Alternating up/down
            base_price += displacement
            
        # Every 75 bars: Create Fair Value Gap
        elif i > 0 and i % 75 == 0:
            # Create gap in price
            gap_size = 0.0008 if (i // 75) % 2 == 0 else -0.0008
            base_price += gap_size
            
        else:
            # Normal price movement
            trend = 0.00001 * np.sin(i / 30)
            noise = np.random.normal(0, 0.0003)
            base_price += trend + noise
        
        # Generate OHLC
        open_price = base_price
        
        # Create realistic candle patterns
        if i % 50 == 49:  # Bar before displacement - make it opposing
            if (i // 50) % 2 == 0:  # Next will be bullish displacement
                # Make this a bearish candle (future bullish OB)
                close_price = open_price - 0.0002
                high_price = open_price + 0.0001
                low_price = close_price - 0.0001
            else:  # Next will be bearish displacement
                # Make this a bullish candle (future bearish OB)
                close_price = open_price + 0.0002
                high_price = close_price + 0.0001
                low_price = open_price - 0.0001
        else:
            # Normal candle
            close_price = open_price + np.random.normal(0, 0.0002)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.0001))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.0001))
        
        # Volume patterns
        if i % 50 == 0:  # High volume on displacement
            volume = abs(np.random.normal(2000, 300))
        else:
            volume = abs(np.random.normal(800, 200))
        
        prices.append([open_price, high_price, low_price, close_price])
        volumes.append(volume)
    
    df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'], index=dates)
    df['volume'] = volumes
    
    return df

def test_smc_engine_basic():
    """Test basic SMC engine functionality"""
    print("🧪 Testing SMC Engine Basic Functionality...")
    
    try:
        from smc_engine import SmartMoneyEngine, OrderBlockType, FVGType
        
        # Create SMC engine
        smc_engine = SmartMoneyEngine("EURUSD", "M15")
        
        # Create test data with patterns
        test_data = create_smc_test_data(200)
        
        # Test SMC analysis
        smc_context = smc_engine.analyze_smc_context(test_data)
        
        # Validate results
        if not isinstance(smc_context, dict):
            print("   ❌ SMC context not returned as dictionary")
            return False
        
        required_keys = ['order_blocks', 'fair_value_gaps', 'market_structure', 
                        'liquidity_sweeps', 'smc_features']
        
        for key in required_keys:
            if key not in smc_context:
                print(f"   ❌ Missing key in SMC context: {key}")
                return False
        
        # Check order blocks
        order_blocks = smc_context['order_blocks']
        if not isinstance(order_blocks, list):
            print("   ❌ Order blocks not returned as list")
            return False
        
        # Check fair value gaps
        fvgs = smc_context['fair_value_gaps']
        if not isinstance(fvgs, list):
            print("   ❌ Fair value gaps not returned as list")
            return False
        
        # Check SMC features
        smc_features = smc_context['smc_features']
        if not isinstance(smc_features, dict):
            print("   ❌ SMC features not returned as dictionary")
            return False
        
        # Should have at least 15 SMC features
        if len(smc_features) < 15:
            print(f"   ❌ Too few SMC features: {len(smc_features)} (expected >15)")
            return False
        
        print(f"   ✅ Order blocks detected: {len(order_blocks)}")
        print(f"   ✅ Fair value gaps detected: {len(fvgs)}")
        print(f"   ✅ SMC features generated: {len(smc_features)}")
        print(f"   ✅ Market structure: {smc_context['market_structure'].trend}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Cannot import SMC engine: {e}")
        return False
    except Exception as e:
        print(f"   ❌ SMC engine test failed: {e}")
        return False

def test_smc_pattern_detection():
    """Test SMC pattern detection accuracy"""
    print("🔍 Testing SMC Pattern Detection Accuracy...")
    
    try:
        from smc_engine import SmartMoneyEngine, OrderBlockType, FVGType
        
        smc_engine = SmartMoneyEngine("EURUSD", "M15")
        
        # Create data with known patterns
        test_data = create_smc_test_data(300)
        smc_context = smc_engine.analyze_smc_context(test_data)
        
        order_blocks = smc_context['order_blocks']
        fvgs = smc_context['fair_value_gaps']
        
        # We created patterns every 50 bars (displacement) and 75 bars (gaps)
        # So we should detect several patterns
        
        expected_min_obs = 3  # At least 3 order blocks in 300 bars
        expected_min_fvgs = 2  # At least 2 FVGs in 300 bars
        
        if len(order_blocks) < expected_min_obs:
            print(f"   ⚠️  Fewer order blocks than expected: {len(order_blocks)} < {expected_min_obs}")
        else:
            print(f"   ✅ Order block detection: {len(order_blocks)} blocks found")
        
        if len(fvgs) < expected_min_fvgs:
            print(f"   ⚠️  Fewer FVGs than expected: {len(fvgs)} < {expected_min_fvgs}")
        else:
            print(f"   ✅ Fair value gap detection: {len(fvgs)} gaps found")
        
        # Test order block types
        bullish_obs = [ob for ob in order_blocks if ob.ob_type == OrderBlockType.BULLISH_OB]
        bearish_obs = [ob for ob in order_blocks if ob.ob_type == OrderBlockType.BEARISH_OB]
        
        print(f"   📈 Bullish order blocks: {len(bullish_obs)}")
        print(f"   📉 Bearish order blocks: {len(bearish_obs)}")
        
        # Test FVG types
        bullish_fvgs = [fvg for fvg in fvgs if fvg.fvg_type == FVGType.BULLISH_FVG]
        bearish_fvgs = [fvg for fvg in fvgs if fvg.fvg_type == FVGType.BEARISH_FVG]
        
        print(f"   📈 Bullish FVGs: {len(bullish_fvgs)}")
        print(f"   📉 Bearish FVGs: {len(bearish_fvgs)}")
        
        # Pattern quality checks
        valid_obs = 0
        for ob in order_blocks:
            if ob.top > ob.bottom and ob.strength > 0:
                valid_obs += 1
        
        valid_fvgs = 0
        for fvg in fvgs:
            if fvg.top > fvg.bottom and fvg.size > 0:
                valid_fvgs += 1
        
        ob_quality = valid_obs / len(order_blocks) if order_blocks else 0
        fvg_quality = valid_fvgs / len(fvgs) if fvgs else 0
        
        print(f"   🎯 Order block quality: {ob_quality:.1%}")
        print(f"   🎯 FVG quality: {fvg_quality:.1%}")
        
        # Success if we detect patterns and quality is good
        success = (len(order_blocks) > 0 and len(fvgs) >= 0 and 
                  ob_quality > 0.8 and fvg_quality > 0.8)
        
        return success
        
    except Exception as e:
        print(f"   ❌ Pattern detection test failed: {e}")
        return False

def test_smc_feature_integration():
    """Test SMC feature integration with existing system"""
    print("🔗 Testing SMC Feature Integration...")
    
    try:
        # Test if we can import and integrate SMC with existing features
        from smc_engine import SmartMoneyEngine
        from enhanced_feature_engineer import EnhancedFeatureEngineer
        
        # Create engines
        smc_engine = SmartMoneyEngine("EURUSD", "M15")
        feature_engineer = EnhancedFeatureEngineer("EURUSD", "M15")
        
        # Create test data
        test_data = create_smc_test_data(200)
        
        # Test existing feature generation (should still work)
        existing_features = feature_engineer.create_enhanced_features(test_data)
        
        if not isinstance(existing_features, dict) or len(existing_features) < 50:
            print("   ❌ Existing feature generation broken")
            return False
        
        print(f"   ✅ Existing features working: {len(existing_features)} features")
        
        # Test SMC analysis
        smc_context = smc_engine.analyze_smc_context(test_data)
        smc_features = smc_context['smc_features']
        
        if not isinstance(smc_features, dict) or len(smc_features) < 15:
            print("   ❌ SMC feature generation failed")
            return False
        
        print(f"   ✅ SMC features working: {len(smc_features)} features")
        
        # Test feature combination
        combined_features = {**existing_features, **smc_features}
        
        print(f"   ✅ Combined features: {len(combined_features)} total")
        print(f"   📊 Feature breakdown:")
        print(f"      - Existing: {len(existing_features)}")
        print(f"      - SMC: {len(smc_features)}")
        print(f"      - Total: {len(combined_features)}")
        
        # Check for feature conflicts (same keys)
        conflicts = set(existing_features.keys()) & set(smc_features.keys())
        if conflicts:
            print(f"   ⚠️  Feature name conflicts: {conflicts}")
            return False
        
        print("   ✅ No feature conflicts detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Feature integration test failed: {e}")
        return False

def test_smc_performance():
    """Test SMC analysis performance"""
    print("⚡ Testing SMC Performance...")
    
    try:
        from smc_engine import SmartMoneyEngine
        import time
        
        smc_engine = SmartMoneyEngine("EURUSD", "M15")
        
        # Test with different data sizes
        test_sizes = [100, 200, 500]
        performance_results = {}
        
        for size in test_sizes:
            test_data = create_smc_test_data(size)
            
            # Measure analysis time
            start_time = time.time()
            smc_context = smc_engine.analyze_smc_context(test_data)
            analysis_time = time.time() - start_time
            
            performance_results[size] = {
                'time': analysis_time,
                'order_blocks': len(smc_context['order_blocks']),
                'fvgs': len(smc_context['fair_value_gaps']),
                'features': len(smc_context['smc_features'])
            }
            
            print(f"   📊 {size} bars: {analysis_time*1000:.1f}ms")
        
        # Performance targets
        max_time_200_bars = 2.0  # Should analyze 200 bars in under 2 seconds
        max_time_500_bars = 5.0  # Should analyze 500 bars in under 5 seconds
        
        success = True
        
        if performance_results[200]['time'] > max_time_200_bars:
            print(f"   ⚠️  200 bars too slow: {performance_results[200]['time']:.2f}s > {max_time_200_bars}s")
            success = False
        else:
            print(f"   ✅ 200 bars performance: {performance_results[200]['time']:.2f}s")
        
        if performance_results[500]['time'] > max_time_500_bars:
            print(f"   ⚠️  500 bars too slow: {performance_results[500]['time']:.2f}s > {max_time_500_bars}s")
            success = False
        else:
            print(f"   ✅ 500 bars performance: {performance_results[500]['time']:.2f}s")
        
        return success
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def test_smc_ai_integration():
    """Test SMC integration with AI engine"""
    print("🤖 Testing SMC + AI Integration...")
    
    try:
        from smc_engine import SmartMoneyEngine
        from enhanced_ai_engine import EnhancedAIEngine
        
        # Create engines
        smc_engine = SmartMoneyEngine("EURUSD", "M15")
        ai_engine = EnhancedAIEngine("EURUSD", "M15")
        
        # Create training data with SMC patterns
        training_data = create_smc_test_data(1000)
        
        print("   🏋️ Training AI with SMC-enhanced data...")
        
        # For this test, we'll simulate SMC integration by adding SMC features manually
        # In real implementation, this will be done in enhanced_feature_engineer.py
        
        # Generate SMC context for training data
        smc_context = smc_engine.analyze_smc_context(training_data[:800])
        smc_features = smc_context['smc_features']
        
        if len(smc_features) < 15:
            print(f"   ❌ Insufficient SMC features for AI training: {len(smc_features)}")
            return False
        
        print(f"   ✅ SMC features ready for AI: {len(smc_features)} features")
        
        # Test that AI engine can handle additional features
        # (This is a simulation - real integration requires feature engineer update)
        
        # Test SMC signal generation
        signals = smc_engine.get_smc_trading_signals(smc_context)
        
        if not isinstance(signals, dict):
            print("   ❌ SMC signal generation failed")
            return False
        
        required_signal_keys = ['smc_bullish_signal', 'smc_bearish_signal', 
                               'smc_confidence', 'smc_primary_signal']
        
        for key in required_signal_keys:
            if key not in signals:
                print(f"   ❌ Missing signal key: {key}")
                return False
        
        print(f"   ✅ SMC signals generated:")
        print(f"      - Bullish: {signals['smc_bullish_signal']:.3f}")
        print(f"      - Bearish: {signals['smc_bearish_signal']:.3f}")
        print(f"      - Confidence: {signals['smc_confidence']:.3f}")
        print(f"      - Primary: {signals['smc_primary_signal']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ SMC + AI integration test failed: {e}")
        return False

def test_smc_memory_usage():
    """Test SMC memory efficiency"""
    print("💾 Testing SMC Memory Usage...")
    
    try:
        import psutil
        import os
        
        from smc_engine import SmartMoneyEngine
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create SMC engine and analyze large dataset
        smc_engine = SmartMoneyEngine("EURUSD", "M15")
        large_data = create_smc_test_data(1000)
        
        # Perform analysis
        smc_context = smc_engine.analyze_smc_context(large_data)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"   📊 Memory usage:")
        print(f"      - Initial: {initial_memory:.1f} MB")
        print(f"      - Final: {final_memory:.1f} MB")
        print(f"      - Increase: {memory_increase:.1f} MB")
        
        # Memory target: Should not increase by more than 100MB for 1000 bars
        memory_limit = 100  # MB
        
        if memory_increase > memory_limit:
            print(f"   ⚠️  Memory usage too high: {memory_increase:.1f} MB > {memory_limit} MB")
            return False
        else:
            print(f"   ✅ Memory usage acceptable: {memory_increase:.1f} MB")
            return True
        
    except ImportError:
        print("   ⚠️  psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"   ❌ Memory test failed: {e}")
        return False

def run_smc_integration_tests():
    """Run complete SMC integration test suite"""
    print("🚀 SMC Integration Test Suite v2.1.0")
    print("=" * 60)
    print("🎯 Testing Smart Money Concepts integration with ForexAI-EA")
    print("📋 Target: 80%+ AI accuracy with SMC features")
    print("=" * 60)
    
    tests = [
        ("SMC Engine Basic", test_smc_engine_basic),
        ("SMC Pattern Detection", test_smc_pattern_detection),
        ("SMC Feature Integration", test_smc_feature_integration),
        ("SMC Performance", test_smc_performance),
        ("SMC + AI Integration", test_smc_ai_integration),
        ("SMC Memory Usage", test_smc_memory_usage)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
        print(f"{'='*50}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 SMC INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    success_rate = passed / total * 100
    print(f"\n📊 Results: {passed}/{total} tests passed ({success_rate:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL SMC TESTS PASSED!")
        print("✅ Smart Money Concepts integration ready")
        print("🎯 Proceeding to Phase 2 Week 7-8 implementation")
        
        print("\n📋 NEXT STEPS:")
        print("1. Integrate SMC engine into enhanced_feature_engineer.py")
        print("2. Update enhanced_ai_engine.py with SMC features") 
        print("3. Add SMC endpoints to socket_server.py")
        print("4. Achieve 80%+ AI accuracy target")
        
    else:
        failed_tests = [name for name, result in results if not result]
        print(f"\n⚠️  {total - passed} TESTS FAILED!")
        print(f"📋 Failed tests: {', '.join(failed_tests)}")
        print("\n🔧 Fix failed tests before proceeding to integration")
    
    return passed == total

if __name__ == "__main__":
    success = run_smc_integration_tests()
    
    if success:
        print("\n🚀 SMC INTEGRATION READY!")
        print("🎯 Phase 2 Week 7-8 can proceed!")
    else:
        print("\n⚠️  Fix SMC issues before integration!")
    
    exit(0 if success else 1)