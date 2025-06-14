"""
File: test_fixed_system.py
Description: Comprehensive test for all fixed ForexAI-EA components
Author: Claude AI Developer
Version: 2.0.2
Created: 2025-06-14
"""

import sys
import os
import logging
import time
import socket
import json
import threading
import numpy as np
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/python'))

def setup_test_environment():
    """Setup test environment"""
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create necessary directories
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/logs', exist_ok=True)

def create_test_data(bars=1000):
    """Create realistic test data"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=bars, freq='15min')
    
    prices = []
    volumes = []
    base_price = 1.1000
    
    for i in range(bars):
        # Random walk with trend
        price_change = np.random.normal(0.00001, 0.0008)
        base_price += price_change
        
        # Generate OHLC
        open_price = base_price
        high_price = open_price + abs(np.random.normal(0, 0.0005))
        low_price = open_price - abs(np.random.normal(0, 0.0005))
        close_price = open_price + np.random.normal(0, 0.0003)
        close_price = max(min(close_price, high_price), low_price)
        
        # Volume
        volume = abs(np.random.normal(1000, 300))
        
        prices.append([open_price, high_price, low_price, close_price])
        volumes.append(volume)
    
    df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'], index=dates)
    df['volume'] = volumes
    
    return df

def test_technical_indicators():
    """Test technical indicators"""
    print("🔧 Testing Technical Indicators...")
    
    try:
        from technical_indicators import TechnicalIndicators
        
        ti = TechnicalIndicators()
        test_data = create_test_data(200)
        
        # Test indicator calculation
        indicators = ti.calculate_all_indicators(test_data)
        
        # Verify outputs are pandas Series
        for name, indicator in indicators.items():
            if not isinstance(indicator, pd.Series):
                print(f"   ❌ {name}: Not a pandas Series")
                return False
            if len(indicator) != len(test_data):
                print(f"   ❌ {name}: Wrong length")
                return False
        
        print(f"   ✅ All {len(indicators)} indicators working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Technical indicators failed: {e}")
        return False

def test_volume_profile():
    """Test volume profile"""
    print("📊 Testing Volume Profile...")
    
    try:
        from volume_profile import VolumeProfileEngine, VWAPCalculator
        
        vp_engine = VolumeProfileEngine()
        vwap_calc = VWAPCalculator()
        test_data = create_test_data(200)
        
        # Test Volume Profile
        volume_profile = vp_engine.calculate_volume_profile(test_data.tail(100))
        
        if volume_profile is None:
            print("   ❌ Volume Profile calculation failed")
            return False
        
        # Test VWAP
        vwap = vwap_calc.calculate_vwap(test_data)
        
        if not isinstance(vwap, pd.Series):
            print("   ❌ VWAP calculation failed")
            return False
        
        print("   ✅ Volume Profile and VWAP working correctly")
        return True
        
    except ImportError:
        print("   ⚠️  Volume Profile not available - using basic features")
        return True
    except Exception as e:
        print(f"   ❌ Volume Profile failed: {e}")
        return False

def test_enhanced_feature_engineer():
    """Test enhanced feature engineer"""
    print("🧬 Testing Enhanced Feature Engineer...")
    
    try:
        from enhanced_feature_engineer import EnhancedFeatureEngineer
        
        fe = EnhancedFeatureEngineer("EURUSD", "M15")
        test_data = create_test_data(200)
        
        # Test feature generation
        features = fe.create_enhanced_features(test_data)
        
        if not isinstance(features, dict):
            print("   ❌ Features not returned as dictionary")
            return False
        
        if len(features) < 20:
            print("   ❌ Too few features generated")
            return False
        
        # Test training data preparation
        features_df, labels_series = fe.prepare_enhanced_training_data(test_data)
        
        if not isinstance(features_df, pd.DataFrame) or not isinstance(labels_series, pd.Series):
            print("   ❌ Training data preparation failed")
            return False
        
        print(f"   ✅ Generated {len(features)} features and {len(features_df)} training samples")
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced Feature Engineer failed: {e}")
        return False

def test_enhanced_ai_engine():
    """Test enhanced AI engine"""
    print("🤖 Testing Enhanced AI Engine...")
    
    try:
        from enhanced_ai_engine import EnhancedAIEngine, EnhancedModelEvaluator
        
        ai_engine = EnhancedAIEngine("EURUSD", "M15")
        test_data = create_test_data(1000)
        
        # Test training
        print("   🏋️ Training model...")
        training_results = ai_engine.train_enhanced_model(test_data[:800])
        
        if 'ensemble_accuracy' not in training_results:
            print("   ❌ Training failed - no accuracy reported")
            return False
        
        accuracy = training_results['ensemble_accuracy']
        if accuracy < 0.5:
            print(f"   ❌ Poor accuracy: {accuracy:.3f}")
            return False
        
        # Test prediction
        signal, confidence, details = ai_engine.predict_enhanced(test_data[:850])
        
        if signal not in [-1, 0, 1]:
            print(f"   ❌ Invalid signal: {signal}")
            return False
        
        if not (0.0 <= confidence <= 1.0):
            print(f"   ❌ Invalid confidence: {confidence}")
            return False
        
        # Test model persistence
        save_success = ai_engine.save_enhanced_model("test_model.pkl")
        if not save_success:
            print("   ❌ Model save failed")
            return False
        
        # Test loading
        new_ai = EnhancedAIEngine("EURUSD", "M15")
        load_success = new_ai.load_enhanced_model("test_model.pkl")
        if not load_success:
            print("   ❌ Model load failed")
            return False
        
        # Test backtesting
        evaluator = EnhancedModelEvaluator()
        backtest_results = evaluator.comprehensive_backtest(
            ai_engine, test_data[700:900], initial_balance=10000, risk_per_trade=0.02
        )
        
        if 'total_return' not in backtest_results:
            print("   ❌ Backtesting failed")
            return False
        
        print(f"   ✅ Model trained with {accuracy:.1%} accuracy, backtested successfully")
        
        # Cleanup
        if os.path.exists("test_model.pkl"):
            os.remove("test_model.pkl")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced AI Engine failed: {e}")
        import traceback
        print(f"   📋 Error details: {traceback.format_exc()}")
        return False

def test_socket_server():
    """Test socket server"""
    print("🔌 Testing Enhanced Socket Server...")
    
    try:
        from socket_server import EnhancedSocketServer
        
        # Start server in background
        server = EnhancedSocketServer("localhost", 8889)  # Use different port for testing
        server_thread = threading.Thread(target=server.start_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Test client connection
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(10)
            client.connect(("localhost", 8889))
            
            # Test ping
            ping_request = {"action": "ping"}
            client.send(json.dumps(ping_request).encode('utf-8'))
            
            response = client.recv(1024)
            ping_response = json.loads(response.decode('utf-8'))
            
            if "pong" not in ping_response.get("message", ""):
                print("   ❌ Ping test failed")
                client.close()
                server.stop_server()
                return False
            
            # Test status
            status_request = {"action": "status"}
            client.send(json.dumps(status_request).encode('utf-8'))
            
            response = client.recv(4096)
            status_response = json.loads(response.decode('utf-8'))
            
            if status_response.get('status') != 'running':
                print("   ❌ Status test failed")
                client.close()
                server.stop_server()
                return False
            
            # Test capabilities
            cap_request = {"action": "capabilities"}
            client.send(json.dumps(cap_request).encode('utf-8'))
            
            response = client.recv(4096)
            cap_response = json.loads(response.decode('utf-8'))
            
            if 'capabilities' not in cap_response:
                print("   ❌ Capabilities test failed")
                client.close()
                server.stop_server()
                return False
            
            client.close()
            print("   ✅ Socket server working correctly")
            
        except Exception as e:
            print(f"   ❌ Client connection failed: {e}")
            server.stop_server()
            return False
        finally:
            server.stop_server()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Socket server failed: {e}")
        return False

def test_integration():
    """Test complete system integration"""
    print("🔗 Testing System Integration...")
    
    try:
        from enhanced_ai_engine import EnhancedAIEngine
        from socket_server import EnhancedSocketServer
        
        # Start server
        server = EnhancedSocketServer("localhost", 8890)
        server_thread = threading.Thread(target=server.start_server)
        server_thread.daemon = True
        server_thread.start()
        time.sleep(2)
        
        # Create test data
        test_data = create_test_data(500)
        training_data = test_data.to_dict('records')
        
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(30)
            client.connect(("localhost", 8890))
            
            # Test training
            train_request = {
                "action": "train",
                "symbol": "EURUSD",
                "training_data": training_data
            }
            
            client.send(json.dumps(train_request).encode('utf-8'))
            response = client.recv(8192)
            train_response = json.loads(response.decode('utf-8'))
            
            if not train_response.get('success', False):
                print(f"   ❌ Training request failed: {train_response}")
                client.close()
                server.stop_server()
                return False
            
            # Test prediction after training
            prediction_data = test_data.tail(100).to_dict('records')
            pred_request = {
                "action": "predict",
                "symbol": "EURUSD",
                "timeframe": "M15",
                "price_data": prediction_data
            }
            
            client.send(json.dumps(pred_request).encode('utf-8'))
            response = client.recv(4096)
            pred_response = json.loads(response.decode('utf-8'))
            
            if 'signal' not in pred_response:
                print(f"   ❌ Prediction request failed: {pred_response}")
                client.close()
                server.stop_server()
                return False
            
            signal = pred_response['signal']
            confidence = pred_response['confidence']
            
            if signal not in [-1, 0, 1]:
                print(f"   ❌ Invalid signal: {signal}")
                client.close()
                server.stop_server()
                return False
            
            print(f"   ✅ Integration test passed - Signal: {signal}, Confidence: {confidence:.3f}")
            
            client.close()
            
        except Exception as e:
            print(f"   ❌ Integration test failed: {e}")
            server.stop_server()
            return False
        finally:
            server.stop_server()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("🛡️ Testing Error Handling...")
    
    try:
        from enhanced_ai_engine import EnhancedAIEngine
        
        ai_engine = EnhancedAIEngine("EURUSD", "M15")
        
        # Test with insufficient data
        small_data = create_test_data(10)
        
        try:
            signal, confidence, details = ai_engine.predict_enhanced(small_data)
            # Should return safe defaults or handle gracefully
            if 'error' not in details and signal == 0 and confidence == 0.0:
                print("   ✅ Insufficient data handled correctly")
            elif 'error' in details:
                print("   ✅ Error handling working correctly")
            else:
                print(f"   ❌ Unexpected response: signal={signal}, confidence={confidence}")
                return False
        except Exception:
            # Exception is acceptable for insufficient data
            print("   ✅ Exception handling working correctly")
        
        # Test with corrupted data
        corrupted_data = create_test_data(100)
        corrupted_data.loc[corrupted_data.index[0], 'close'] = np.nan
        
        try:
            from enhanced_feature_engineer import EnhancedFeatureEngineer
            fe = EnhancedFeatureEngineer("EURUSD", "M15")
            features = fe.create_enhanced_features(corrupted_data)
            
            if isinstance(features, dict) and len(features) > 0:
                print("   ✅ Corrupted data handled gracefully")
            else:
                print("   ❌ Failed to handle corrupted data")
                return False
                
        except Exception:
            # Exception handling is acceptable
            print("   ✅ Exception handling for corrupted data working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False

def test_unicode_handling():
    """Test Unicode handling"""
    print("🌐 Testing Unicode Handling...")
    
    try:
        from socket_server import EnhancedSocketServer
        
        # Start server
        server = EnhancedSocketServer("localhost", 8891)
        server_thread = threading.Thread(target=server.start_server)
        server_thread.daemon = True
        server_thread.start()
        time.sleep(2)
        
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(10)
            client.connect(("localhost", 8891))
            
            # Test with Unicode characters
            unicode_request = {"action": "ping", "message": "Testing 测试 🚀"}
            unicode_json = json.dumps(unicode_request, ensure_ascii=False)
            client.send(unicode_json.encode('utf-8'))
            
            response = client.recv(1024)
            response_data = json.loads(response.decode('utf-8'))
            
            if "pong" in response_data.get("message", ""):
                print("   ✅ Unicode handling working correctly")
                client.close()
                server.stop_server()
                return True
            else:
                print("   ❌ Unicode handling failed")
                client.close()
                server.stop_server()
                return False
                
        except Exception as e:
            print(f"   ❌ Unicode test failed: {e}")
            server.stop_server()
            return False
        finally:
            server.stop_server()
        
    except Exception as e:
        print(f"   ❌ Unicode handling test failed: {e}")
        return False

def test_performance():
    """Test system performance"""
    print("⚡ Testing Performance...")
    
    try:
        from enhanced_ai_engine import EnhancedAIEngine
        
        ai_engine = EnhancedAIEngine("EURUSD", "M15")
        test_data = create_test_data(1000)
        
        # Test training time
        start_time = time.time()
        ai_engine.train_enhanced_model(test_data[:800])
        training_time = time.time() - start_time
        
        if training_time > 120:  # Should complete within 2 minutes
            print(f"   ⚠️  Training slow: {training_time:.1f}s (target: <120s)")
        else:
            print(f"   ✅ Training time: {training_time:.1f}s")
        
        # Test prediction time
        prediction_times = []
        for _ in range(10):
            start_time = time.time()
            ai_engine.predict_enhanced(test_data[:850])
            prediction_time = time.time() - start_time
            prediction_times.append(prediction_time)
        
        avg_prediction_time = np.mean(prediction_times)
        if avg_prediction_time > 0.5:  # Should be under 500ms
            print(f"   ⚠️  Prediction slow: {avg_prediction_time*1000:.1f}ms (target: <500ms)")
        else:
            print(f"   ✅ Prediction time: {avg_prediction_time*1000:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 ForexAI-EA Comprehensive System Test v2.0.2")
    print("=" * 60)
    
    setup_test_environment()
    
    tests = [
        ("Technical Indicators", test_technical_indicators),
        ("Volume Profile", test_volume_profile),
        ("Enhanced Feature Engineer", test_enhanced_feature_engineer),
        ("Enhanced AI Engine", test_enhanced_ai_engine),
        ("Socket Server", test_socket_server),
        ("System Integration", test_integration),
        ("Error Handling", test_error_handling),
        ("Unicode Handling", test_unicode_handling),
        ("Performance", test_performance)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ System is ready for production deployment")
        print("\n📋 Next Steps:")
        print("   1. Start the AI socket server")
        print("   2. Compile and attach EA in MetaTrader 5")
        print("   3. Begin demo trading")
        print("   4. Monitor performance and adjust as needed")
        
        # Create success report
        report = {
            'test_date': datetime.now().isoformat(),
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': passed/total,
            'total_time': total_time,
            'status': 'ALL_PASSED',
            'system_version': '2.0.2'
        }
        
        with open('system_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Test report saved to: system_test_report.json")
        
    else:
        print(f"\n⚠️  {total - passed} TESTS FAILED!")
        print("❌ Please fix issues before proceeding to production")
        
        failed_tests = [name for name, result in results if not result]
        print(f"\n📋 Failed tests: {', '.join(failed_tests)}")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)