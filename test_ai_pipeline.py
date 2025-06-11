# test_ai_pipeline.py
"""
Complete AI Pipeline Test for ForexAI-EA Project
Tests technical indicators, feature engineering, and AI model
Author: Claude AI Developer
Version: 1.0.0
Created: 2025-06-11
"""

import sys
import os
import time
import numpy as np
import json
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

from technical_indicators import TechnicalIndicators
from feature_engineer import FeatureEngineer
from ai_engine import AITradingEngine, ModelEvaluator

def create_realistic_forex_data(n_bars: int = 1000, symbol: str = "EURUSD") -> dict:
    """Create realistic forex OHLC data for testing"""
    print(f"Generating {n_bars} bars of {symbol} test data...")
    
    np.random.seed(42)
    
    # Starting price
    if symbol == "EURUSD":
        base_price = 1.1000
    elif symbol == "GBPUSD":
        base_price = 1.3000
    elif symbol == "USDJPY":
        base_price = 110.00
    else:
        base_price = 1.0000
    
    prices = [base_price]
    
    # Generate more realistic price movement
    for i in range(n_bars):
        # Trend component (slow changes)
        trend = 0.00005 * np.sin(i / 200) if symbol != "USDJPY" else 0.005 * np.sin(i / 200)
        
        # Random walk component
        volatility = 0.0008 if symbol != "USDJPY" else 0.08
        noise = np.random.normal(0, volatility)
        
        # Mean reversion component
        mean_reversion = -0.001 * (prices[-1] - base_price) / base_price
        
        # Occasional volatility spikes
        if np.random.random() < 0.05:  # 5% chance of volatility spike
            noise *= 3
        
        price_change = trend + noise + mean_reversion
        new_price = prices[-1] * (1 + price_change)
        
        # Ensure price doesn't go negative
        if new_price > 0:
            prices.append(new_price)
        else:
            prices.append(prices[-1])
    
    # Generate OHLC from price series
    ohlc_data = {
        'open': [],
        'high': [],
        'low': [],
        'close': prices[1:]  # Remove first price
    }
    
    for i in range(len(ohlc_data['close'])):
        close = ohlc_data['close'][i]
        open_price = prices[i]
        
        # Generate realistic high/low with proper spread
        if symbol == "USDJPY":
            spread_factor = 0.05  # Larger moves for JPY pairs
        else:
            spread_factor = 0.0005  # Typical for major pairs
        
        body_size = abs(close - open_price)
        wick_size = max(body_size * 0.3, spread_factor)
        
        high = max(open_price, close) + np.random.uniform(0, wick_size)
        low = min(open_price, close) - np.random.uniform(0, wick_size)
        
        ohlc_data['open'].append(open_price)
        ohlc_data['high'].append(high)
        ohlc_data['low'].append(low)
    
    print(f"✅ Generated {len(ohlc_data['close'])} bars")
    print(f"   Price range: {min(ohlc_data['low']):.5f} - {max(ohlc_data['high']):.5f}")
    
    return ohlc_data

def test_technical_indicators():
    """Test technical indicators calculation"""
    print("\n" + "="*60)
    print("🔧 TESTING TECHNICAL INDICATORS")
    print("="*60)
    
    try:
        # Create test data
        test_data = create_realistic_forex_data(500, "EURUSD")
        
        # Initialize indicators engine
        indicators_engine = TechnicalIndicators()
        
        # Test individual indicators
        print("\n📊 Testing individual indicators...")
        
        # Test EMA
        ema_9 = indicators_engine.calculate_ema(test_data['close'], 9)
        ema_21 = indicators_engine.calculate_ema(test_data['close'], 21)
        print(f"✅ EMA calculation: EMA9={ema_9[-1]:.5f}, EMA21={ema_21[-1]:.5f}")
        
        # Test RSI
        rsi = indicators_engine.calculate_rsi(test_data['close'])
        print(f"✅ RSI calculation: {rsi[-1]:.2f}")
        
        # Test MACD
        macd_data = indicators_engine.calculate_macd(test_data['close'])
        print(f"✅ MACD calculation: Main={macd_data['macd'][-1]:.6f}, Signal={macd_data['signal'][-1]:.6f}")
        
        # Test Bollinger Bands
        bb_data = indicators_engine.calculate_bollinger_bands(test_data['close'])
        print(f"✅ Bollinger Bands: Upper={bb_data['upper'][-1]:.5f}, Lower={bb_data['lower'][-1]:.5f}")
        
        # Test ATR
        atr = indicators_engine.calculate_atr(test_data['high'], test_data['low'], test_data['close'])
        print(f"✅ ATR calculation: {atr[-1]:.6f}")
        
        # Test all indicators together
        print("\n📈 Testing complete indicator suite...")
        all_indicators = indicators_engine.calculate_all_indicators(test_data)
        
        print(f"✅ Complete indicator calculation: {len(all_indicators)} indicators")
        for name, values in list(all_indicators.items())[:5]:  # Show first 5
            if not np.isnan(values[-1]):
                print(f"   {name}: {values[-1]:.6f}")
        
        return True, all_indicators, test_data
        
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        return False, None, None

def test_feature_engineering(indicators_data, ohlc_data):
    """Test feature engineering"""
    print("\n" + "="*60)
    print("🧬 TESTING FEATURE ENGINEERING")
    print("="*60)
    
    try:
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Test feature creation
        print("\n🔬 Creating features from indicators...")
        features = feature_engineer.create_features(ohlc_data, indicators_data)
        
        print(f"✅ Feature creation: {len(features)} features generated")
        
        # Show sample features
        print("\n📋 Sample features:")
        feature_items = list(features.items())
        for i, (name, value) in enumerate(feature_items[:10]):  # Show first 10
            print(f"   {name}: {value:.6f}")
        
        # Test label generation
        print("\n🏷️  Testing label generation...")
        atr_values = indicators_data.get('atr', np.ones(len(ohlc_data['close'])) * 0.001)
        labels = feature_engineer.generate_labels(ohlc_data, atr_values)
        
        # Count label distribution
        label_counts = {-1: 0, 0: 0, 1: 0}
        for label in labels:
            label_counts[label] += 1
        
        print(f"✅ Label generation: {len(labels)} labels")
        print(f"   Distribution - Buy: {label_counts[1]}, Hold: {label_counts[0]}, Sell: {label_counts[-1]}")
        
        return True, features
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False, None

def test_ai_engine():
    """Test AI engine training and prediction"""
    print("\n" + "="*60)
    print("🤖 TESTING AI ENGINE")
    print("="*60)
    
    try:
        # Create training data
        print("\n📚 Creating training dataset...")
        training_data = []
        for symbol in ["EURUSD", "GBPUSD"]:
            data = create_realistic_forex_data(1500, symbol)
            training_data.append(data)
        
        # Initialize AI engine
        print("\n🧠 Initializing AI engine...")
        ai_engine = AITradingEngine()
        
        # Train model
        print("\n🎯 Training AI model...")
        start_time = time.time()
        performance = ai_engine.train_model(training_data)
        training_time = time.time() - start_time
        
        print(f"✅ Model training completed in {training_time:.2f} seconds")
        print(f"   Training accuracy: {performance['train_accuracy']:.3f}")
        print(f"   Test accuracy: {performance['test_accuracy']:.3f}")
        print(f"   CV score: {performance['cv_mean']:.3f} ± {performance['cv_std']:.3f}")
        print(f"   Features used: {performance['feature_count']}")
        
        # Test prediction
        print("\n🔮 Testing predictions...")
        test_data = create_realistic_forex_data(200, "EURUSD")
        
        # Make multiple predictions
        predictions = []
        prediction_times = []
        
        for i in range(5):
            # Create data slice for prediction
            end_idx = 100 + i * 10
            pred_data = {
                'open': test_data['open'][:end_idx],
                'high': test_data['high'][:end_idx],
                'low': test_data['low'][:end_idx],
                'close': test_data['close'][:end_idx]
            }
            
            start_time = time.time()
            signal, confidence, metadata = ai_engine.predict(pred_data)
            pred_time = time.time() - start_time
            
            predictions.append((signal, confidence))
            prediction_times.append(pred_time)
            
            signal_name = {-1: "SELL", 0: "HOLD", 1: "BUY"}[signal]
            print(f"   Prediction {i+1}: {signal_name} (confidence: {confidence:.3f}) - {pred_time*1000:.1f}ms")
        
        avg_pred_time = np.mean(prediction_times) * 1000
        print(f"✅ Average prediction time: {avg_pred_time:.1f}ms")
        
        # Test model info
        model_info = ai_engine.get_model_info()
        print(f"✅ Model info retrieved: {model_info['status']}")
        
        # Test feature importance
        feature_importance = ai_engine.get_feature_importance(10)
        print(f"\n🏆 Top 5 most important features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
            print(f"   {i+1}. {feature}: {importance:.4f}")
        
        return True, ai_engine
        
    except Exception as e:
        print(f"❌ AI engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_model_persistence(ai_engine):
    """Test model saving and loading"""
    print("\n" + "="*60)
    print("💾 TESTING MODEL PERSISTENCE")
    print("="*60)
    
    try:
        # Save model
        print("\n💾 Saving model...")
        save_success = ai_engine.save_model()
        
        if save_success:
            print("✅ Model saved successfully")
        else:
            print("❌ Model save failed")
            return False
        
        # Create new AI engine and load model
        print("\n📂 Loading model in new instance...")
        new_ai_engine = AITradingEngine()
        load_success = new_ai_engine.load_model()
        
        if load_success:
            print("✅ Model loaded successfully")
        else:
            print("❌ Model load failed")
            return False
        
        # Test prediction with loaded model
        print("\n🔮 Testing prediction with loaded model...")
        test_data = create_realistic_forex_data(100, "EURUSD")
        
        signal, confidence, metadata = new_ai_engine.predict(test_data)
        signal_name = {-1: "SELL", 0: "HOLD", 1: "BUY"}[signal]
        print(f"✅ Loaded model prediction: {signal_name} (confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Model persistence test failed: {e}")
        return False

def test_backtesting(ai_engine):
    """Test model backtesting"""
    print("\n" + "="*60)
    print("📊 TESTING BACKTESTING")
    print("="*60)
    
    try:
        # Create backtest data
        print("\n📈 Creating backtest dataset...")
        backtest_data = [create_realistic_forex_data(1000, "EURUSD")]
        
        # Initialize evaluator
        evaluator = ModelEvaluator(ai_engine)
        
        # Run backtest
        print("\n🧮 Running backtest...")
        start_time = time.time()
        results = evaluator.backtest_model(backtest_data, initial_balance=10000)
        backtest_time = time.time() - start_time
        
        # Display results
        metrics = results['performance_metrics']
        
        print(f"✅ Backtest completed in {backtest_time:.2f} seconds")
        print(f"\n📊 Backtest Results:")
        print(f"   Total trades: {metrics.get('total_trades', 0)}")
        print(f"   Win rate: {metrics.get('win_rate', 0):.1%}")
        print(f"   Total return: {metrics.get('total_return', 0):.2f}%")
        print(f"   Profit factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"   Max drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"   Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Final balance: ${metrics.get('final_balance', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtesting test failed: {e}")
        return False

def run_complete_test():
    """Run complete AI pipeline test"""
    print("🚀 ForexAI-EA Complete AI Pipeline Test")
    print("="*80)
    
    start_time = time.time()
    test_results = {}
    
    # Test 1: Technical Indicators
    success, indicators_data, ohlc_data = test_technical_indicators()
    test_results['technical_indicators'] = success
    
    if not success:
        print("\n❌ Technical indicators test failed - aborting pipeline test")
        return False
    
    # Test 2: Feature Engineering
    success, features = test_feature_engineering(indicators_data, ohlc_data)
    test_results['feature_engineering'] = success
    
    if not success:
        print("\n❌ Feature engineering test failed - aborting pipeline test")
        return False
    
    # Test 3: AI Engine
    success, ai_engine = test_ai_engine()
    test_results['ai_engine'] = success
    
    if not success:
        print("\n❌ AI engine test failed - aborting pipeline test")
        return False
    
    # Test 4: Model Persistence
    success = test_model_persistence(ai_engine)
    test_results['model_persistence'] = success
    
    # Test 5: Backtesting
    success = test_backtesting(ai_engine)
    test_results['backtesting'] = success
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
        if not passed:
            all_passed = False
    
    print(f"\n⏱️  Total test time: {total_time:.2f} seconds")
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! AI Pipeline is ready for production.")
        print("\n📋 Next steps:")
        print("   1. Start the AI socket server: python src/python/socket_server.py")
        print("   2. Run communication tests: python test_communication.py")
        print("   3. Attach EA in MetaTrader 5")
        print("   4. Monitor live trading performance")
        
        # Save test results
        test_report = {
            'test_date': datetime.now().isoformat(),
            'total_time': total_time,
            'results': test_results,
            'status': 'ALL_PASSED'
        }
        
        with open('test_results.json', 'w') as f:
            json.dump(test_report, f, indent=2)
        
        print(f"\n💾 Test report saved to: test_results.json")
        
    else:
        print("\n⚠️  SOME TESTS FAILED! Please fix issues before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    # Run the complete test
    success = run_complete_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)