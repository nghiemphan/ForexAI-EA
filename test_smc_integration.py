"""
File: test_smc_integration_complete.py
Description: Complete SMC Integration Test for Phase 2 Week 7-8
Author: Claude AI Developer
Version: 1.0.0
Created: June 15, 2025
"""

import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

def setup_logging():
    """Setup logging for test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def generate_comprehensive_test_data(n_samples=500):
    """Generate comprehensive test data for SMC testing"""
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
    
    # Generate realistic EURUSD data with complex patterns for SMC
    prices = []
    volumes = []
    base_price = 1.1000
    
    for i in range(n_samples):
        # Multiple trend components for realistic SMC patterns
        long_trend = 0.000008 * np.sin(i / 120)    # Long-term cycle
        medium_trend = 0.000015 * np.sin(i / 40)    # Medium-term cycle  
        short_trend = 0.000005 * np.sin(i / 15)     # Short-term cycle
        noise = np.random.normal(0, 0.0007)         # Market noise
        
        # Simulate institutional moves (SMC patterns)
        if i % 50 == 0:  # Periodic institutional activity
            institutional_move = np.random.choice([-0.002, 0.002]) * np.random.uniform(0.5, 1.5)
        else:
            institutional_move = 0
        
        price_change = long_trend + medium_trend + short_trend + noise + institutional_move
        base_price += price_change
        
        # Realistic OHLC with institutional characteristics
        open_price = base_price
        
        # Create realistic highs/lows with potential liquidity sweeps
        if np.random.random() < 0.1:  # 10% chance of liquidity sweep
            high_price = open_price + abs(np.random.normal(0, 0.0008)) * 2  # Larger spike
            low_price = open_price - abs(np.random.normal(0, 0.0003))
        else:
            high_price = open_price + abs(np.random.normal(0, 0.0004))
            low_price = open_price - abs(np.random.normal(0, 0.0004))
        
        close_price = open_price + np.random.normal(0, 0.0003)
        close_price = max(min(close_price, high_price), low_price)
        
        # Volume with institutional characteristics
        base_volume = 1000
        volatility_factor = abs(high_price - low_price) / open_price
        time_factor = 1 + 0.4 * np.sin(2 * np.pi * (i % 96) / 96)  # Daily pattern
        
        if institutional_move != 0:  # Higher volume during institutional moves
            volume_multiplier = np.random.uniform(2.0, 4.0)
        else:
            volume_multiplier = np.random.uniform(0.7, 1.5)
        
        volume = abs(np.random.normal(
            base_volume * time_factor * (1 + volatility_factor * 100) * volume_multiplier, 
            200
        ))
        
        prices.append([open_price, high_price, low_price, close_price])
        volumes.append(volume)
    
    ohlcv_df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'], index=dates)
    ohlcv_df['volume'] = volumes
    
    return ohlcv_df

def test_enhanced_feature_engineer():
    """Test enhanced feature engineer with SMC integration"""
    logger = logging.getLogger(__name__)
    
    print("🧪 Testing Enhanced Feature Engineer with SMC Integration")
    print("=" * 60)
    
    try:
        # Import enhanced feature engineer
        from enhanced_feature_engineer import EnhancedFeatureEngineer
        
        # Initialize feature engineer
        feature_engineer = EnhancedFeatureEngineer("EURUSD", "M15")
        
        # Generate test data
        print("📊 Generating comprehensive test data...")
        test_data = generate_comprehensive_test_data(500)
        print(f"✅ Generated {len(test_data)} bars of test data")
        
        # Test feature generation
        print("\n🔧 Testing feature generation...")
        start_time = time.time()
        features = feature_engineer.create_enhanced_features(test_data)
        generation_time = (time.time() - start_time) * 1000
        
        # Analyze features
        total_features = len(features)
        smc_features = [k for k in features.keys() if k.startswith('smc_')]
        vp_features = [k for k in features.keys() if k.startswith('vp_')]
        vwap_features = [k for k in features.keys() if k.startswith('vwap_')]
        technical_features = [k for k in features.keys() if any(x in k for x in ['ema', 'rsi', 'macd', 'bb_', 'atr', 'stoch', 'williams'])]
        
        print(f"✅ Feature Generation Results:")
        print(f"   Total Features: {total_features}")
        print(f"   📊 Technical: {len(technical_features)} features")
        print(f"   📈 Volume Profile: {len(vp_features)} features") 
        print(f"   💫 VWAP: {len(vwap_features)} features")
        print(f"   🏢 SMC: {len(smc_features)} features")
        print(f"   ⚡ Generation Time: {generation_time:.1f}ms")
        
        # Validate targets
        target_achieved = total_features >= 88 and len(smc_features) >= 20
        performance_met = generation_time < 200
        
        print(f"\n🎯 Target Validation:")
        print(f"   {'✅' if total_features >= 88 else '❌'} Total Features: {total_features}/88+ (target)")
        print(f"   {'✅' if len(smc_features) >= 20 else '❌'} SMC Features: {len(smc_features)}/20+ (target)")
        print(f"   {'✅' if performance_met else '❌'} Performance: {generation_time:.1f}ms/200ms (target)")
        print(f"   {'✅' if target_achieved and performance_met else '❌'} Overall: {'SUCCESS' if target_achieved and performance_met else 'NEEDS WORK'}")
        
        # Test SMC integration validation
        print(f"\n🏢 Testing SMC Integration Validation...")
        validation_results = feature_engineer.validate_smc_integration(test_data)
        
        print(f"✅ SMC Integration Validation:")
        print(f"   SMC Available: {validation_results['smc_integration']['smc_available']}")
        print(f"   SMC Features Generated: {validation_results['smc_integration']['smc_features_generated']}")
        print(f"   SMC Target Achieved: {validation_results['smc_integration']['smc_target_achieved']}")
        print(f"   Total Target Achieved: {validation_results['targets']['total_target_achieved']}")
        
        # Display sample SMC features
        if smc_features:
            print(f"\n🏢 Sample SMC Features Generated:")
            for i, feature in enumerate(sorted(smc_features)[:8]):
                print(f"   {i+1:2d}. {feature}: {features[feature]:.4f}")
            if len(smc_features) > 8:
                print(f"   ... and {len(smc_features) - 8} more SMC features")
        
        # Test training data preparation
        print(f"\n📚 Testing Training Data Preparation...")
        try:
            features_df, labels_series = feature_engineer.prepare_enhanced_training_data(test_data)
            
            training_smc_features = [col for col in features_df.columns if col.startswith('smc_')]
            
            print(f"✅ Training Data Preparation:")
            print(f"   Samples: {len(features_df)}")
            print(f"   Total Features: {len(features_df.columns)}")
            print(f"   SMC Features: {len(training_smc_features)}")
            print(f"   Label Distribution: {dict(labels_series.value_counts())}")
            print(f"   Unique Labels: {len(set(labels_series))}")
            
            training_success = len(features_df.columns) >= 88 and len(training_smc_features) >= 20
            print(f"   Training Ready: {'✅ YES' if training_success else '❌ NO'}")
            
        except Exception as e:
            print(f"❌ Training data preparation failed: {e}")
            training_success = False
        
        # Overall assessment
        overall_success = target_achieved and performance_met and training_success
        
        print(f"\n🎯 Enhanced Feature Engineer Assessment:")
        print(f"   {'✅' if target_achieved else '❌'} Feature Generation: {'SUCCESS' if target_achieved else 'FAILED'}")
        print(f"   {'✅' if performance_met else '❌'} Performance: {'SUCCESS' if performance_met else 'FAILED'}")
        print(f"   {'✅' if training_success else '❌'} Training Ready: {'SUCCESS' if training_success else 'FAILED'}")
        print(f"   {'🏆' if overall_success else '🔧'} Overall Status: {'READY FOR AI TRAINING' if overall_success else 'NEEDS FIXES'}")
        
        return overall_success, {
            'total_features': total_features,
            'smc_features': len(smc_features),
            'generation_time': generation_time,
            'training_ready': training_success,
            'features_sample': features
        }
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure enhanced_feature_engineer.py is in the src/python directory")
        return False, {}
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False, {}

def test_enhanced_ai_engine(feature_results):
    """Test enhanced AI engine with SMC features"""
    print("\n🤖 Testing Enhanced AI Engine with SMC Integration")
    print("=" * 60)
    
    try:
        # Import enhanced AI engine
        from enhanced_ai_engine import EnhancedAIEngine, EnhancedModelEvaluator
        
        # Initialize AI engine
        ai_engine = EnhancedAIEngine("EURUSD", "M15")
        
        # Generate training data
        print("📊 Generating training data for AI...")
        training_data = generate_comprehensive_test_data(2000)  # More data for training
        print(f"✅ Generated {len(training_data)} bars for AI training")
        
        # Train enhanced model
        print(f"\n🔧 Training Enhanced AI Model with SMC...")
        start_time = time.time()
        
        training_results = ai_engine.train_enhanced_model(
            training_data,
            validation_split=0.2,
            enable_smc_features=True
        )
        
        training_time = time.time() - start_time
        
        print(f"✅ AI Training Results:")
        print(f"   Training Time: {training_time:.1f} seconds")
        print(f"   Ensemble Accuracy: {training_results['ensemble_accuracy']:.4f}")
        print(f"   Cross-validation: {training_results['cv_mean']:.4f} ± {training_results['cv_std']:.4f}")
        print(f"   Total Features: {training_results['total_features']}")
        print(f"   SMC Features: {training_results['smc_features']}")
        print(f"   Training Samples: {training_results['training_samples']}")
        print(f"   Validation Samples: {training_results['validation_samples']}")
        
        # Check accuracy target
        accuracy_target_met = training_results['ensemble_accuracy'] >= 0.80
        smc_integration_success = training_results['smc_features'] >= 20
        
        print(f"\n🎯 AI Training Validation:")
        print(f"   {'✅' if accuracy_target_met else '❌'} Accuracy Target: {training_results['ensemble_accuracy']:.1%}/80%+")
        print(f"   {'✅' if smc_integration_success else '❌'} SMC Integration: {training_results['smc_features']}/20+")
        print(f"   {'✅' if training_results.get('target_achieved', False) else '❌'} Overall Target: {'ACHIEVED' if training_results.get('target_achieved', False) else 'IN PROGRESS'}")
        
        # Test prediction
        print(f"\n🔮 Testing Enhanced Prediction...")
        test_prediction_data = training_data[:1800]  # Use subset for prediction
        
        signal, confidence, details = ai_engine.predict_enhanced(test_prediction_data)
        
        print(f"✅ Prediction Results:")
        print(f"   Signal: {signal}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Feature Count: {details['feature_count']}")
        print(f"   SMC Active: {details.get('smc_active', False)}")
        print(f"   Volume Profile Active: {details.get('volume_profile_active', False)}")
        print(f"   VWAP Active: {details.get('vwap_active', False)}")
        
        # Test backtesting
        print(f"\n📈 Testing Enhanced Backtesting...")
        evaluator = EnhancedModelEvaluator()
        
        backtest_results = evaluator.comprehensive_backtest(
            ai_engine,
            training_data[1800:2000],  # Use last 200 bars for backtest
            initial_balance=10000,
            risk_per_trade=0.015,
            enable_smc_analysis=True
        )
        
        print(f"✅ Backtest Results:")
        print(f"   Total Return: {backtest_results['total_return']:.4f}")
        print(f"   Win Rate: {backtest_results['win_rate']:.4f}")
        print(f"   Profit Factor: {backtest_results['profit_factor']:.4f}")
        print(f"   Max Drawdown: {backtest_results['max_drawdown']:.4f}")
        print(f"   Total Trades: {backtest_results['total_trades']}")
        print(f"   SMC Enhanced Trades: {backtest_results.get('smc_enhanced_trades', 0)}")
        
        # Overall AI assessment
        ai_success = (accuracy_target_met and smc_integration_success and 
                     confidence > 0.5 and backtest_results['total_trades'] > 0)
        
        print(f"\n🎯 Enhanced AI Engine Assessment:")
        print(f"   {'✅' if accuracy_target_met else '❌'} Accuracy: {'SUCCESS' if accuracy_target_met else 'NEEDS IMPROVEMENT'}")
        print(f"   {'✅' if smc_integration_success else '❌'} SMC Integration: {'SUCCESS' if smc_integration_success else 'INCOMPLETE'}")
        print(f"   {'✅' if confidence > 0.5 else '❌'} Prediction Quality: {'GOOD' if confidence > 0.5 else 'POOR'}")
        print(f"   {'✅' if backtest_results['total_trades'] > 0 else '❌'} Backtesting: {'WORKING' if backtest_results['total_trades'] > 0 else 'FAILED'}")
        print(f"   {'🏆' if ai_success else '🔧'} Overall Status: {'READY FOR PRODUCTION' if ai_success else 'NEEDS WORK'}")
        
        return ai_success, {
            'accuracy': training_results['ensemble_accuracy'],
            'smc_features': training_results['smc_features'],
            'training_time': training_time,
            'backtest_results': backtest_results,
            'prediction_confidence': confidence
        }
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure enhanced_ai_engine.py is in the src/python directory")
        return False, {}
    except Exception as e:
        print(f"❌ AI Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def main():
    """Main test function for complete SMC integration"""
    logger = setup_logging()
    
    print("🚀 ForexAI-EA Phase 2 Week 7-8 Complete Integration Test")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Objective: Validate SMC integration and 80%+ AI accuracy")
    print()
    
    overall_start_time = time.time()
    
    # Test 1: Enhanced Feature Engineer
    print("🧪 TEST 1: Enhanced Feature Engineer with SMC")
    feature_success, feature_results = test_enhanced_feature_engineer()
    
    if not feature_success:
        print("\n❌ Feature Engineer test failed. Cannot proceed to AI testing.")
        return False
    
    # Test 2: Enhanced AI Engine
    print("\n🧪 TEST 2: Enhanced AI Engine with SMC")
    ai_success, ai_results = test_enhanced_ai_engine(feature_results)
    
    # Overall assessment
    total_time = time.time() - overall_start_time
    overall_success = feature_success and ai_success
    
    print(f"\n🎯 PHASE 2 WEEK 7-8 FINAL ASSESSMENT")
    print("=" * 70)
    print(f"⏱️  Total Test Time: {total_time:.1f} seconds")
    print()
    
    print(f"📊 Feature Engineering:")
    if feature_results:
        print(f"   Total Features: {feature_results['total_features']}/88+ {'✅' if feature_results['total_features'] >= 88 else '❌'}")
        print(f"   SMC Features: {feature_results['smc_features']}/20+ {'✅' if feature_results['smc_features'] >= 20 else '❌'}")
        print(f"   Performance: {feature_results['generation_time']:.1f}ms/200ms {'✅' if feature_results['generation_time'] < 200 else '❌'}")
    
    print(f"\n🤖 AI Engine:")
    if ai_results:
        print(f"   Accuracy: {ai_results['accuracy']:.1%}/80%+ {'✅' if ai_results['accuracy'] >= 0.80 else '❌'}")
        print(f"   SMC Features: {ai_results['smc_features']}/20+ {'✅' if ai_results['smc_features'] >= 20 else '❌'}")
        print(f"   Training Time: {ai_results['training_time']:.1f}s")
        print(f"   Prediction Confidence: {ai_results['prediction_confidence']:.4f}")
    
    print(f"\n🏆 PHASE 2 WEEK 7-8 STATUS:")
    if overall_success:
        print(f"   🎉 SUCCESS: All objectives achieved!")
        print(f"   ✅ SMC Integration: Complete")
        print(f"   ✅ 88+ Features: Achieved")
        print(f"   ✅ 80%+ Accuracy: {'Achieved' if ai_results and ai_results['accuracy'] >= 0.80 else 'In Progress'}")
        print(f"   ✅ Performance: Excellent")
        print(f"   🚀 Status: READY FOR PHASE 3")
    else:
        print(f"   🔧 IN PROGRESS: Some objectives need work")
        print(f"   {'✅' if feature_success else '❌'} Feature Engineering: {'Complete' if feature_success else 'Needs work'}")
        print(f"   {'✅' if ai_success else '❌'} AI Engine: {'Complete' if ai_success else 'Needs work'}")
        print(f"   🔧 Status: CONTINUE DEVELOPMENT")
    
    print(f"\n💡 Next Steps:")
    if overall_success:
        print(f"   1. Deploy to demo environment")
        print(f"   2. Begin Phase 3: Advanced session analysis")
        print(f"   3. Prepare for live trading validation")
    else:
        print(f"   1. Fix identified issues")
        print(f"   2. Re-run integration tests")
        print(f"   3. Achieve 80%+ accuracy target")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)