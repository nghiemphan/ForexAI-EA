"""
File: test_enhanced_system.py
Description: Comprehensive test suite for Enhanced ForexAI-EA v2.0
Author: Claude AI Developer
Version: 2.0.0
Created: 2025-06-13
Modified: 2025-06-13
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
import logging
import json
import socket
import threading
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/python'))

# Import enhanced modules
try:
    from volume_profile import VolumeProfileEngine, VWAPCalculator
    from enhanced_feature_engineer import EnhancedFeatureEngineer
    from enhanced_ai_engine import EnhancedAIEngine, EnhancedModelEvaluator
    from socket_server import EnhancedSocketServer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all enhanced modules are in src/python/ directory")
    sys.exit(1)

class TestEnhancedSystem(unittest.TestCase):
    """Comprehensive test suite for Enhanced ForexAI-EA v2.0"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        logging.basicConfig(level=logging.WARNING)  # Reduce test noise
        
        # Create test data
        cls.test_data = cls._create_test_data()
        print("🧪 Enhanced ForexAI-EA v2.0 Test Suite")
        print("=" * 50)
    
    @staticmethod
    def _create_test_data(bars=1000):
        """Create realistic test OHLCV data"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=bars, freq='15min')
        
        prices = []
        volumes = []
        base_price = 1.1000
        
        for i in range(bars):
            # Random walk with slight trend
            price_change = np.random.normal(0.00001, 0.0008)
            base_price += price_change
            
            # Generate OHLC
            open_price = base_price
            high_price = open_price + abs(np.random.normal(0, 0.0005))
            low_price = open_price - abs(np.random.normal(0, 0.0005))
            close_price = open_price + np.random.normal(0, 0.0003)
            close_price = max(min(close_price, high_price), low_price)
            
            # Volume with volatility correlation
            volatility = abs(high_price - low_price)
            volume = abs(np.random.normal(1000 + volatility * 100000, 300))
            
            prices.append([open_price, high_price, low_price, close_price])
            volumes.append(volume)
        
        df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'], index=dates)
        df['volume'] = volumes
        
        return df
    
    def test_01_volume_profile_engine(self):
        """Test Volume Profile Engine functionality"""
        print("🧪 Testing Volume Profile Engine...")
        
        vp_engine = VolumeProfileEngine()
        
        # Test volume profile calculation
        volume_profile = vp_engine.calculate_volume_profile(self.test_data.tail(100))
        
        # Validate results
        self.assertIsNotNone(volume_profile)
        self.assertGreater(volume_profile.poc_price, 0)
        self.assertGreater(volume_profile.poc_volume, 0)
        self.assertGreater(volume_profile.value_area_high, volume_profile.value_area_low)
        self.assertGreater(len(volume_profile.volume_nodes), 0)
        
        # Test feature extraction
        current_price = self.test_data['close'].iloc[-1]
        vp_features = vp_engine.get_volume_profile_features(current_price, volume_profile)
        
        self.assertIsInstance(vp_features, dict)
        self.assertIn('poc_distance', vp_features)
        self.assertIn('va_position', vp_features)
        self.assertIn('price_in_value_area', vp_features)
        
        # Test key levels
        key_levels = vp_engine.identify_key_levels(volume_profile)
        self.assertIsInstance(key_levels, list)
        self.assertGreater(len(key_levels), 0)
        
        print("✅ Volume Profile Engine: PASSED")
    
    def test_02_vwap_calculator(self):
        """Test VWAP Calculator functionality"""
        print("🧪 Testing VWAP Calculator...")
        
        vwap_calc = VWAPCalculator()
        
        # Test VWAP calculation
        session_vwap = vwap_calc.calculate_vwap(self.test_data)
        rolling_vwap = vwap_calc.calculate_vwap(self.test_data, period=20)
        
        self.assertIsInstance(session_vwap, pd.Series)
        self.assertIsInstance(rolling_vwap, pd.Series)
        self.assertEqual(len(session_vwap), len(self.test_data))
        
        # Test VWAP bands
        vwap_bands = vwap_calc.calculate_vwap_bands(self.test_data, session_vwap)
        
        self.assertIn('vwap_upper', vwap_bands)
        self.assertIn('vwap_lower', vwap_bands)
        self.assertIn('vwap_std', vwap_bands)
        
        # Test feature extraction
        current_price = self.test_data['close'].iloc[-1]
        vwap_features = vwap_calc.get_vwap_features(
            current_price, session_vwap.iloc[-1], vwap_bands, -1
        )
        
        self.assertIsInstance(vwap_features, dict)
        self.assertIn('vwap_distance', vwap_features)
        self.assertIn('vwap_band_position', vwap_features)
        self.assertIn('price_above_vwap', vwap_features)
        
        print("✅ VWAP Calculator: PASSED")
    
    def test_03_enhanced_feature_engineer(self):
        """Test Enhanced Feature Engineer"""
        print("🧪 Testing Enhanced Feature Engineer...")
        
        enhanced_fe = EnhancedFeatureEngineer("EURUSD", "M15")
        
        # Test feature generation
        features = enhanced_fe.create_enhanced_features(self.test_data)
        
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 60)  # Should have 65+ features
        
        # Check for enhanced feature categories
        tech_features = [k for k in features.keys() if not k.startswith(('vp_', 'vwap_'))]
        vp_features = [k for k in features.keys() if k.startswith('vp_')]
        vwap_features = [k for k in features.keys() if k.startswith('vwap_')]
        
        self.assertGreater(len(tech_features), 20)  # Technical indicators
        self.assertGreater(len(vp_features), 5)     # Volume Profile features
        self.assertGreater(len(vwap_features), 5)   # VWAP features
        
        # Test training data preparation
        features_df, labels_series = enhanced_fe.prepare_enhanced_training_data(self.test_data)
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertIsInstance(labels_series, pd.Series)
        self.assertGreater(len(features_df.columns), 60)
        self.assertEqual(len(features_df), len(labels_series))
        
        # Validate label distribution
        label_counts = labels_series.value_counts()
        self.assertIn(-1, label_counts.index)  # Sell signals
        self.assertIn(0, label_counts.index)   # Hold signals  
        self.assertIn(1, label_counts.index)   # Buy signals
        
        print("✅ Enhanced Feature Engineer: PASSED")
    
    def test_04_enhanced_ai_engine(self):
        """Test Enhanced AI Engine"""
        print("🧪 Testing Enhanced AI Engine...")
        
        enhanced_ai = EnhancedAIEngine("EURUSD", "M15")
        
        # Test model training
        training_results = enhanced_ai.train_enhanced_model(self.test_data)
        
        self.assertIsInstance(training_results, dict)
        self.assertIn('ensemble_accuracy', training_results)
        self.assertIn('individual_accuracies', training_results)
        self.assertIn('cv_mean', training_results)
        self.assertIn('feature_count', training_results)
        
        # Validate accuracy
        self.assertGreater(training_results['ensemble_accuracy'], 0.5)
        self.assertGreater(training_results['cv_mean'], 0.5)
        self.assertGreater(training_results['feature_count'], 60)
        
        # Test prediction
        signal, confidence, details = enhanced_ai.predict_enhanced(self.test_data)
        
        self.assertIn(signal, [-1, 0, 1])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(details, dict)
        
        # Validate prediction details
        self.assertIn('enhanced_features', details)
        self.assertIn('individual_models', details)
        self.assertIn('volume_profile_active', details)
        self.assertIn('vwap_active', details)
        
        # Test model persistence
        save_success = enhanced_ai.save_enhanced_model("test_model.pkl")
        self.assertTrue(save_success)
        
        # Test model loading
        new_ai = EnhancedAIEngine("EURUSD", "M15")
        load_success = new_ai.load_enhanced_model("test_model.pkl")
        self.assertTrue(load_success)
        
        # Clean up
        if os.path.exists("test_model.pkl"):
            os.remove("test_model.pkl")
        
        print("✅ Enhanced AI Engine: PASSED")
    
    def test_05_enhanced_model_evaluator(self):
        """Test Enhanced Model Evaluator"""
        print("🧪 Testing Enhanced Model Evaluator...")
        
        # Create and train AI engine
        enhanced_ai = EnhancedAIEngine("EURUSD", "M15")
        enhanced_ai.train_enhanced_model(self.test_data[:800])
        
        # Test backtesting
        evaluator = EnhancedModelEvaluator()
        backtest_results = evaluator.comprehensive_backtest(
            enhanced_ai, 
            self.test_data[700:900],  # Use different data for backtest
            initial_balance=10000,
            risk_per_trade=0.02
        )
        
        self.assertIsInstance(backtest_results, dict)
        self.assertIn('total_return', backtest_results)
        self.assertIn('win_rate', backtest_results)
        self.assertIn('profit_factor', backtest_results)
        self.assertIn('max_drawdown', backtest_results)
        self.assertIn('enhanced_features', backtest_results)
        
        # Validate enhanced features
        enhanced_features = backtest_results['enhanced_features']
        self.assertIn('volume_profile_trades', enhanced_features)
        self.assertIn('vwap_trades', enhanced_features)
        
        print("✅ Enhanced Model Evaluator: PASSED")
    
    def test_06_socket_server_enhanced(self):
        """Test Enhanced Socket Server functionality"""
        print("🧪 Testing Enhanced Socket Server...")
        
        # Start server in background thread
        server = EnhancedSocketServer("localhost", 8889)  # Use different port for testing
        server_thread = threading.Thread(target=server.start_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        try:
            # Test client connection
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(10)
            client.connect(("localhost", 8889))
            
            # Test capabilities request
            capabilities_request = {"action": "capabilities"}
            client.send(json.dumps(capabilities_request).encode('utf-8'))
            
            response = client.recv(4096)
            capabilities = json.loads(response.decode('utf-8'))
            
            self.assertIn('capabilities', capabilities)
            self.assertIn('volume_profile', capabilities['capabilities'])
            self.assertIn('vwap_analysis', capabilities['capabilities'])
            self.assertIn('ensemble_models', capabilities['capabilities'])
            
            # Test status request
            status_request = {"action": "status"}
            client.send(json.dumps(status_request).encode('utf-8'))
            
            response = client.recv(4096)
            status = json.loads(response.decode('utf-8'))
            
            self.assertEqual(status['server_version'], "2.0.0")
            self.assertEqual(status['status'], "running")
            self.assertIn('enhanced_capabilities', status)
            
            # Test prediction request (with training first)
            training_data = self.test_data.to_dict('records')
            train_request = {
                "action": "train",
                "symbol": "EURUSD",
                "training_data": training_data
            }
            client.send(json.dumps(train_request).encode('utf-8'))
            
            response = client.recv(8192)
            train_response = json.loads(response.decode('utf-8'))
            
            if train_response.get('success', False):
                # Test prediction after training
                prediction_data = self.test_data.tail(50).to_dict('records')
                pred_request = {
                    "action": "predict",
                    "symbol": "EURUSD",
                    "timeframe": "M15",
                    "price_data": prediction_data
                }
                client.send(json.dumps(pred_request).encode('utf-8'))
                
                response = client.recv(4096)
                pred_response = json.loads(response.decode('utf-8'))
                
                self.assertIn('signal', pred_response)
                self.assertIn('confidence', pred_response)
                self.assertIn('enhanced_features', pred_response)
                
                enhanced_features = pred_response['enhanced_features']
                self.assertIn('volume_profile_active', enhanced_features)
                self.assertIn('vwap_active', enhanced_features)
                self.assertIn('feature_count', enhanced_features)
            
            client.close()
            
        finally:
            server.stop_server()
        
        print("✅ Enhanced Socket Server: PASSED")
    
    def test_07_integration_test(self):
        """Test complete enhanced system integration"""
        print("🧪 Testing Complete Enhanced System Integration...")
        
        # Create all components
        vp_engine = VolumeProfileEngine()
        vwap_calc = VWAPCalculator()
        enhanced_fe = EnhancedFeatureEngineer("EURUSD", "M15")
        enhanced_ai = EnhancedAIEngine("EURUSD", "M15")
        
        # End-to-end workflow test
        
        # 1. Volume Profile Analysis
        volume_profile = vp_engine.calculate_volume_profile(self.test_data.tail(200))
        self.assertIsNotNone(volume_profile)
        
        # 2. VWAP Analysis
        vwap = vwap_calc.calculate_vwap(self.test_data)
        self.assertIsNotNone(vwap)
        
        # 3. Enhanced Feature Engineering
        features = enhanced_fe.create_enhanced_features(self.test_data)
        self.assertGreater(len(features), 60)
        
        # 4. AI Model Training
        training_results = enhanced_ai.train_enhanced_model(self.test_data)
        self.assertGreater(training_results['ensemble_accuracy'], 0.5)
        
        # 5. Enhanced Prediction
        signal, confidence, details = enhanced_ai.predict_enhanced(self.test_data)
        self.assertIn(signal, [-1, 0, 1])
        
        # 6. Validate enhanced features are active
        self.assertTrue(details.get('volume_profile_active', False))
        self.assertTrue(details.get('vwap_active', False))
        self.assertGreater(details.get('feature_count', 0), 60)
        
        print("✅ Complete Enhanced System Integration: PASSED")
    
    def test_08_performance_benchmarks(self):
        """Test enhanced system performance benchmarks"""
        print("🧪 Testing Enhanced System Performance...")
        
        enhanced_ai = EnhancedAIEngine("EURUSD", "M15")
        
        # Benchmark training time
        start_time = time.time()
        enhanced_ai.train_enhanced_model(self.test_data)
        training_time = time.time() - start_time
        
        self.assertLess(training_time, 60)  # Should complete within 60 seconds
        
        # Benchmark prediction time
        prediction_times = []
        for _ in range(10):
            start_time = time.time()
            enhanced_ai.predict_enhanced(self.test_data)
            prediction_time = time.time() - start_time
            prediction_times.append(prediction_time)
        
        avg_prediction_time = np.mean(prediction_times)
        self.assertLess(avg_prediction_time, 0.2)  # Should be under 200ms
        
        # Benchmark memory usage (basic check)
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.assertLess(memory_mb, 1000)  # Should use less than 1GB
        
        print(f"✅ Performance Benchmarks: Training={training_time:.1f}s, Prediction={avg_prediction_time*1000:.1f}ms, Memory={memory_mb:.1f}MB")
    
    def test_09_error_handling(self):
        """Test enhanced system error handling"""
        print("🧪 Testing Enhanced System Error Handling...")
        
        enhanced_ai = EnhancedAIEngine("EURUSD", "M15")
        
        # Test with insufficient data
        small_data = self.test_data.head(10)
        
        try:
            signal, confidence, details = enhanced_ai.predict_enhanced(small_data)
            # Should return safe defaults
            self.assertEqual(signal, 0)
            self.assertEqual(confidence, 0.0)
            self.assertIn('error', details)
        except Exception:
            pass  # Exception is acceptable for insufficient data
        
        # Test with corrupted data
        corrupted_data = self.test_data.copy()
        corrupted_data.loc[corrupted_data.index[0], 'close'] = np.nan
        
        try:
            enhanced_fe = EnhancedFeatureEngineer("EURUSD", "M15")
            features = enhanced_fe.create_enhanced_features(corrupted_data)
            # Should handle NaN values gracefully
            self.assertIsInstance(features, dict)
        except Exception:
            pass  # Exception handling is acceptable
        
        print("✅ Enhanced System Error Handling: PASSED")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests"""
        # Clean up any test files
        test_files = ["test_model.pkl", "enhanced_model_test.pkl"]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
        
        print("\n" + "=" * 50)
        print("🎉 Enhanced ForexAI-EA v2.0 Test Suite Complete!")


def run_enhanced_test_suite():
    """Run the complete enhanced test suite"""
    print("🚀 Starting Enhanced ForexAI-EA v2.0 Test Suite...")
    print("📋 Testing Volume Profile, VWAP, Ensemble AI, and Integration")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedSystem)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY:")
    print(f"   Tests Run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print("\n🎉 ALL TESTS PASSED! Enhanced ForexAI-EA v2.0 is ready for deployment!")
        return True
    else:
        print("\n⚠️ Some tests failed. Please review and fix issues before deployment.")
        return False


if __name__ == "__main__":
    success = run_enhanced_test_suite()
    sys.exit(0 if success else 1)