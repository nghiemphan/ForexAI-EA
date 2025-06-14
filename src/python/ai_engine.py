"""
File: src/python/enhanced_ai_engine.py
Description: Enhanced AI Engine with Volume Profile and VWAP Integration
Author: Claude AI Developer
Version: 2.0.0
Created: 2025-06-13
Modified: 2025-06-13
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

# Import our enhanced modules
from enhanced_feature_engineer import EnhancedFeatureEngineer
from volume_profile import VolumeProfileEngine, VWAPCalculator

class EnhancedAIEngine:
    """Enhanced AI Engine with Volume Profile, VWAP, and Ensemble Models"""
    
    def __init__(self, symbol: str = "EURUSD", timeframe: str = "M15"):
        """
        Initialize Enhanced AI Engine
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
        """
        self.logger = logging.getLogger(__name__)
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Initialize feature engineer
        self.feature_engineer = EnhancedFeatureEngineer(symbol, timeframe)
        
        # Model ensemble
        self.ensemble_model = None
        self.feature_scaler = RobustScaler()
        self.feature_columns = None
        self.model_trained = False
        
        # Performance tracking
        self.prediction_history = []
        self.accuracy_tracker = []
        self.confidence_threshold = 0.65
        
        # Enhanced model configuration
        self.model_config = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'class_weight': 'balanced'
            },
            'xgboost': {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'mlogloss'
            },
            'logistic': {
                'random_state': 42,
                'class_weight': 'balanced',
                'max_iter': 1000
            }
        }
        
    def train_enhanced_model(self, ohlcv_data: pd.DataFrame, 
                           validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train enhanced ensemble model with Volume Profile features
        
        Args:
            ohlcv_data: Historical OHLCV data
            validation_split: Percentage of data for validation
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            self.logger.info("🚀 Starting enhanced AI model training...")
            
            # Prepare enhanced training data
            features_df, labels_series = self.feature_engineer.prepare_enhanced_training_data(ohlcv_data)
            
            if len(features_df) < 100:
                raise ValueError("Insufficient training data (need at least 100 samples)")
            
            # Store feature columns for future use
            self.feature_columns = features_df.columns.tolist()
            self.logger.info(f"📊 Training with {len(features_df)} samples and {len(self.feature_columns)} features")
            
            # Split data
            split_idx = int(len(features_df) * (1 - validation_split))
            X_train = features_df.iloc[:split_idx]
            y_train = labels_series.iloc[:split_idx]
            X_val = features_df.iloc[split_idx:]
            y_val = labels_series.iloc[split_idx:]
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            # Create ensemble models
            models = self._create_ensemble_models()
            
            # Train individual models
            trained_models = []
            model_scores = {}
            
            for name, model in models.items():
                self.logger.info(f"🔧 Training {name} model...")
                model.fit(X_train_scaled, y_train)
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val_scaled)
                accuracy = accuracy_score(y_val, val_predictions)
                model_scores[name] = accuracy
                
                trained_models.append((name, model))
                self.logger.info(f"✅ {name} validation accuracy: {accuracy:.4f}")
            
            # Create voting ensemble
            voting_models = [(name, model) for name, model in trained_models]
            self.ensemble_model = VotingClassifier(
                estimators=voting_models,
                voting='soft'  # Use probability-based voting
            )
            
            # Train ensemble
            self.logger.info("🔄 Training ensemble model...")
            self.ensemble_model.fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            ensemble_val_predictions = self.ensemble_model.predict(X_val_scaled)
            ensemble_accuracy = accuracy_score(y_val, ensemble_val_predictions)
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.ensemble_model, X_train_scaled, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            # Feature importance (from Random Forest)
            rf_model = None
            for name, model in trained_models:
                if name == 'random_forest':
                    rf_model = model
                    break
            
            feature_importance = {}
            if rf_model and hasattr(rf_model, 'feature_importances_'):
                importance_scores = rf_model.feature_importances_
                feature_importance = dict(zip(self.feature_columns, importance_scores))
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True))
            
            # Training results
            results = {
                'ensemble_accuracy': ensemble_accuracy,
                'individual_accuracies': model_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'feature_count': len(self.feature_columns),
                'feature_importance': feature_importance,
                'label_distribution': y_train.value_counts().to_dict()
            }
            
            self.model_trained = True
            
            # Log results
            self.logger.info("🎯 Enhanced AI Model Training Complete!")
            self.logger.info(f"   📊 Ensemble Accuracy: {ensemble_accuracy:.4f}")
            self.logger.info(f"   📈 Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            self.logger.info(f"   🔥 Feature Count: {len(self.feature_columns)}")
            
            # Log top features
            if feature_importance:
                top_features = list(feature_importance.keys())[:10]
                self.logger.info(f"   ⭐ Top Features: {', '.join(top_features[:5])}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced model training failed: {e}")
            raise
    
    def _create_ensemble_models(self) -> Dict[str, Any]:
        """Create individual models for ensemble"""
        models = {}
        
        # Random Forest (primary model)
        models['random_forest'] = RandomForestClassifier(**self.model_config['random_forest'])
        
        # XGBoost (gradient boosting)
        models['xgboost'] = xgb.XGBClassifier(**self.model_config['xgboost'])
        
        # Logistic Regression (linear model)
        models['logistic'] = LogisticRegression(**self.model_config['logistic'])
        
        return models
    
    def predict_enhanced(self, ohlcv_data: pd.DataFrame) -> Tuple[int, float, Dict[str, Any]]:
        """
        Make enhanced prediction with Volume Profile context
        
        Args:
            ohlcv_data: Current OHLCV data
            
        Returns:
            Tuple of (signal, confidence, prediction_details)
        """
        try:
            if not self.model_trained or self.ensemble_model is None:
                raise ValueError("Model not trained. Call train_enhanced_model() first.")
            
            # Generate enhanced features
            features = self.feature_engineer.create_enhanced_features(ohlcv_data)
            
            # Convert to DataFrame with correct column order
            features_df = pd.DataFrame([features])[self.feature_columns]
            
            # Handle missing features
            features_df = features_df.fillna(0)
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features_df)
            
            # Get prediction probabilities
            prediction_probs = self.ensemble_model.predict_proba(features_scaled)[0]
            
            # Get class prediction
            predicted_class = self.ensemble_model.predict(features_scaled)[0]
            
            # Calculate confidence (max probability)
            confidence = max(prediction_probs)
            
            # Get individual model predictions for analysis
            individual_predictions = {}
            for name, model in self.ensemble_model.named_estimators_.items():
                try:
                    pred_probs = model.predict_proba(features_scaled)[0]
                    pred_class = model.predict(features_scaled)[0]
                    individual_predictions[name] = {
                        'class': int(pred_class),
                        'confidence': float(max(pred_probs)),
                        'probabilities': pred_probs.tolist()
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to get prediction from {name}: {e}")
            
            # Enhanced signal filtering
            filtered_signal = self._apply_enhanced_filters(
                predicted_class, confidence, features, individual_predictions
            )
            
            # Prediction details
            prediction_details = {
                'raw_signal': int(predicted_class),
                'filtered_signal': int(filtered_signal),
                'confidence': float(confidence),
                'probabilities': {
                    'sell': float(prediction_probs[0]),   # -1 class
                    'hold': float(prediction_probs[1]),   # 0 class  
                    'buy': float(prediction_probs[2])     # 1 class
                },
                'individual_models': individual_predictions,
                'feature_count': len(features),
                'volume_profile_active': any(k.startswith('vp_') for k in features.keys()),
                'vwap_active': any(k.startswith('vwap_') for k in features.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
            # Track prediction
            self._track_prediction(filtered_signal, confidence, prediction_details)
            
            return filtered_signal, confidence, prediction_details
            
        except Exception as e:
            self.logger.error(f"Enhanced prediction failed: {e}")
            # Return neutral signal with low confidence
            return 0, 0.0, {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _apply_enhanced_filters(self, raw_signal: int, confidence: float, 
                              features: Dict[str, float], 
                              individual_predictions: Dict[str, Any]) -> int:
        """
        Apply enhanced filtering logic using Volume Profile and VWAP context
        
        Args:
            raw_signal: Raw model prediction
            confidence: Prediction confidence
            features: Current features
            individual_predictions: Individual model predictions
            
        Returns:
            Filtered signal
        """
        try:
            # Base confidence filter
            if confidence < self.confidence_threshold:
                return 0  # Hold if confidence too low
            
            # Model consensus filter
            model_signals = [pred['class'] for pred in individual_predictions.values()]
            consensus_score = sum(1 for signal in model_signals if signal == raw_signal) / len(model_signals)
            
            if consensus_score < 0.6:  # At least 60% agreement
                return 0  # Hold if no consensus
            
            # Volume Profile filters
            vp_filters_passed = self._check_volume_profile_filters(raw_signal, features)
            if not vp_filters_passed:
                return 0
            
            # VWAP filters
            vwap_filters_passed = self._check_vwap_filters(raw_signal, features)
            if not vwap_filters_passed:
                return 0
            
            # Market structure filters
            structure_filters_passed = self._check_market_structure_filters(raw_signal, features)
            if not structure_filters_passed:
                return 0
            
            # Risk management filters
            risk_filters_passed = self._check_risk_filters(raw_signal, features)
            if not risk_filters_passed:
                return 0
            
            return raw_signal
            
        except Exception as e:
            self.logger.warning(f"Filter application failed: {e}")
            return 0  # Conservative: return hold signal on filter error
    
    def _check_volume_profile_filters(self, signal: int, features: Dict[str, float]) -> bool:
        """Check Volume Profile based filters"""
        try:
            # POC alignment filter
            poc_distance = abs(features.get('vp_poc_distance', 0))
            if poc_distance > 0.005:  # More than 0.5% from POC
                if signal == 1 and features.get('vp_price_above_poc', 0) == 0:
                    return False  # Don't buy below POC when far from it
                if signal == -1 and features.get('vp_price_above_poc', 0) == 1:
                    return False  # Don't sell above POC when far from it
            
            # Value Area filter
            in_value_area = features.get('vp_price_in_value_area', 0)
            if in_value_area == 0:  # Outside value area
                # Be more conservative
                if abs(signal) == 1:  # Only take strong signals outside VA
                    poc_strength = features.get('vp_poc_strength', 0)
                    if poc_strength < 0.05:  # Weak POC
                        return False
            
            # Volume Profile strength filter
            profile_strength = features.get('vp_profile_strength', 0)
            if profile_strength < 0.02:  # Very weak volume profile
                return False  # Don't trade on weak volume data
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Volume Profile filter check failed: {e}")
            return True  # Allow signal if filter check fails
    
    def _check_vwap_filters(self, signal: int, features: Dict[str, float]) -> bool:
        """Check VWAP based filters"""
        try:
            # VWAP trend alignment
            vwap_slope = features.get('vwap_slope', 0)
            
            if signal == 1:  # Buy signal
                # Check if VWAP is trending up or price is above VWAP
                vwap_bullish = (
                    vwap_slope > 0.0001 or 
                    features.get('vwap_session_price_above_vwap', 0) == 1
                )
                if not vwap_bullish:
                    return False
            
            elif signal == -1:  # Sell signal
                # Check if VWAP is trending down or price is below VWAP
                vwap_bearish = (
                    vwap_slope < -0.0001 or 
                    features.get('vwap_session_price_above_vwap', 0) == 0
                )
                if not vwap_bearish:
                    return False
            
            # VWAP band filter
            band_position = features.get('vwap_session_vwap_band_position', 0.5)
            
            if signal == 1 and band_position > 0.8:  # Don't buy at upper band
                return False
            if signal == -1 and band_position < 0.2:  # Don't sell at lower band
                return False
            
            # Multi-timeframe VWAP alignment
            vwap_alignment = features.get('vwap_alignment', 0.5)
            
            if signal == 1 and vwap_alignment < 0.5:  # Buy only with bullish alignment
                return False
            if signal == -1 and vwap_alignment > 0.5:  # Sell only with bearish alignment
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"VWAP filter check failed: {e}")
            return True
    
    def _check_market_structure_filters(self, signal: int, features: Dict[str, float]) -> bool:
        """Check market structure filters"""
        try:
            # Momentum alignment
            momentum_bullish = features.get('momentum_bullish', 0)
            multi_tf_bullish = features.get('multi_timeframe_bullish', 0.5)
            
            if signal == 1:  # Buy signal
                # Need bullish momentum and structure
                if momentum_bullish == 0 and multi_tf_bullish < 0.6:
                    return False
            
            elif signal == -1:  # Sell signal
                # Need bearish momentum and structure
                if momentum_bullish == 1 and multi_tf_bullish > 0.4:
                    return False
            
            # Support/Resistance proximity
            near_resistance = features.get('near_resistance', 0)
            near_support = features.get('near_support', 0)
            
            if signal == 1 and near_resistance == 1:  # Don't buy near resistance
                return False
            if signal == -1 and near_support == 1:  # Don't sell near support
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Market structure filter check failed: {e}")
            return True
    
    def _check_risk_filters(self, signal: int, features: Dict[str, float]) -> bool:
        """Check risk management filters"""
        try:
            # Volatility regime filter
            volatility_regime = features.get('volatility_regime', 1.0)
            
            # Don't trade in extremely high volatility
            if volatility_regime >= 2.0:
                return False
            
            # Be more conservative in low volatility (squeeze)
            if volatility_regime == 0.0 and abs(signal) == 1:
                # Only allow trades with very high confidence in low vol
                # This would need to be checked at higher level
                pass
            
            # Technical indicator confluence
            rsi = features.get('rsi', 50)
            
            # Don't buy in extreme overbought
            if signal == 1 and rsi > 80:
                return False
                
            # Don't sell in extreme oversold
            if signal == -1 and rsi < 20:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Risk filter check failed: {e}")
            return True
    
    def _track_prediction(self, signal: int, confidence: float, details: Dict[str, Any]):
        """Track prediction for performance analysis"""
        try:
            prediction_record = {
                'timestamp': datetime.now(),
                'signal': signal,
                'confidence': confidence,
                'raw_signal': details.get('raw_signal', signal),
                'feature_count': details.get('feature_count', 0),
                'volume_profile_active': details.get('volume_profile_active', False),
                'vwap_active': details.get('vwap_active', False)
            }
            
            self.prediction_history.append(prediction_record)
            
            # Keep only last 1000 predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
                
        except Exception as e:
            self.logger.warning(f"Prediction tracking failed: {e}")
    
    def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get current model performance statistics"""
        try:
            if not self.prediction_history:
                return {'error': 'No predictions recorded yet'}
            
            recent_predictions = self.prediction_history[-100:]  # Last 100 predictions
            
            # Signal distribution
            signals = [p['signal'] for p in recent_predictions]
            signal_counts = {-1: 0, 0: 0, 1: 0}
            for signal in signals:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
            
            # Confidence statistics
            confidences = [p['confidence'] for p in recent_predictions]
            
            # Filter effectiveness
            raw_signals = [p['raw_signal'] for p in recent_predictions]
            filtered_signals = [p['signal'] for p in recent_predictions]
            filter_changes = sum(1 for i in range(len(raw_signals)) 
                               if raw_signals[i] != filtered_signals[i])
            
            stats = {
                'total_predictions': len(self.prediction_history),
                'recent_predictions': len(recent_predictions),
                'signal_distribution': {
                    'sell': signal_counts[-1],
                    'hold': signal_counts[0],
                    'buy': signal_counts[1]
                },
                'confidence_stats': {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                },
                'filter_effectiveness': {
                    'filter_rate': filter_changes / len(recent_predictions),
                    'signals_filtered': filter_changes
                },
                'feature_usage': {
                    'volume_profile_usage': sum(1 for p in recent_predictions 
                                               if p.get('volume_profile_active', False)) / len(recent_predictions),
                    'vwap_usage': sum(1 for p in recent_predictions 
                                     if p.get('vwap_active', False)) / len(recent_predictions)
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Performance stats calculation failed: {e}")
            return {'error': str(e)}
    
    def save_enhanced_model(self, filepath: str) -> bool:
        """Save the enhanced model and scaler"""
        try:
            if not self.model_trained:
                raise ValueError("No trained model to save")
            
            model_data = {
                'ensemble_model': self.ensemble_model,
                'feature_scaler': self.feature_scaler,
                'feature_columns': self.feature_columns,
                'model_config': self.model_config,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'confidence_threshold': self.confidence_threshold,
                'training_timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Enhanced model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            return False
    
    def load_enhanced_model(self, filepath: str) -> bool:
        """Load the enhanced model and scaler"""
        try:
            model_data = joblib.load(filepath)
            
            self.ensemble_model = model_data['ensemble_model']
            self.feature_scaler = model_data['feature_scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_config = model_data.get('model_config', self.model_config)
            self.confidence_threshold = model_data.get('confidence_threshold', 0.65)
            
            self.model_trained = True
            
            training_time = model_data.get('training_timestamp', 'Unknown')
            self.logger.info(f"Enhanced model loaded from {filepath} (trained: {training_time})")
            return True
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return False


# Enhanced Model Evaluator for Volume Profile models
class EnhancedModelEvaluator:
    """Enhanced model evaluation with Volume Profile context"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_backtest(self, ai_engine: EnhancedAIEngine, 
                             ohlcv_data: pd.DataFrame,
                             initial_balance: float = 10000,
                             risk_per_trade: float = 0.02) -> Dict[str, Any]:
        """
        Comprehensive backtesting with Volume Profile context
        
        Args:
            ai_engine: Trained enhanced AI engine
            ohlcv_data: Historical data for backtesting
            initial_balance: Starting balance
            risk_per_trade: Risk percentage per trade
            
        Returns:
            Detailed backtesting results
        """
        try:
            self.logger.info("🧪 Starting enhanced backtesting...")
            
            balance = initial_balance
            position = 0  # 0 = no position, 1 = long, -1 = short
            entry_price = 0
            trades = []
            equity_curve = []
            
            # Start backtesting from bar 200 (need history for features)
            start_idx = 200
            
            for i in range(start_idx, len(ohlcv_data) - 1):
                current_data = ohlcv_data.iloc[:i+1]
                current_price = current_data['close'].iloc[-1]
                next_price = ohlcv_data['close'].iloc[i+1]
                
                # Get AI prediction
                try:
                    signal, confidence, details = ai_engine.predict_enhanced(current_data)
                except Exception as e:
                    signal, confidence = 0, 0.0
                    continue
                
                # Process signal
                if position == 0:  # No position
                    if signal != 0 and confidence > 0.7:  # Higher threshold for backtest
                        # Calculate position size
                        risk_amount = balance * risk_per_trade
                        
                        # Simple position sizing (could be enhanced)
                        position_size = risk_amount / (current_price * 0.01)  # 1% stop loss
                        
                        position = signal
                        entry_price = current_price
                        
                        trade_record = {
                            'entry_time': i,
                            'entry_price': entry_price,
                            'signal': signal,
                            'confidence': confidence,
                            'position_size': position_size,
                            'volume_profile_active': details.get('volume_profile_active', False),
                            'vwap_active': details.get('vwap_active', False)
                        }
                        trades.append(trade_record)
                
                elif position != 0:  # Have position
                    # Check for exit conditions
                    should_exit = False
                    exit_reason = ""
                    
                    # Opposite signal
                    if signal != 0 and signal != position:
                        should_exit = True
                        exit_reason = "opposite_signal"
                    
                    # Stop loss / Take profit (simple)
                    pnl_pct = (next_price - entry_price) / entry_price * position
                    if pnl_pct <= -0.02:  # 2% stop loss
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif pnl_pct >= 0.04:  # 4% take profit
                        should_exit = True
                        exit_reason = "take_profit"
                    
                    if should_exit:
                        # Close position
                        trade = trades[-1]
                        trade['exit_time'] = i
                        trade['exit_price'] = next_price
                        trade['exit_reason'] = exit_reason
                        
                        # Calculate P&L
                        pnl = (next_price - entry_price) * position * trade['position_size']
                        trade['pnl'] = pnl
                        trade['pnl_pct'] = pnl_pct
                        
                        balance += pnl
                        position = 0
                        entry_price = 0
                
                # Record equity
                equity_curve.append({
                    'time': i,
                    'balance': balance,
                    'price': current_price
                })
                
                if i % 100 == 0:
                    self.logger.info(f"Processed {i - start_idx} bars...")
            
            # Calculate performance metrics
            completed_trades = [t for t in trades if 'exit_price' in t]
            
            if not completed_trades:
                return {'error': 'No completed trades in backtest period'}
            
            # Performance calculations
            total_trades = len(completed_trades)
            winning_trades = len([t for t in completed_trades if t['pnl'] > 0])
            losing_trades = total_trades - winning_trades
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(t['pnl'] for t in completed_trades)
            total_return = (balance - initial_balance) / initial_balance
            
            winning_pnls = [t['pnl'] for t in completed_trades if t['pnl'] > 0]
            losing_pnls = [t['pnl'] for t in completed_trades if t['pnl'] <= 0]
            
            avg_win = np.mean(winning_pnls) if winning_pnls else 0
            avg_loss = np.mean(losing_pnls) if losing_pnls else 0
            
            profit_factor = abs(sum(winning_pnls) / sum(losing_pnls)) if losing_pnls else float('inf')
            
            # Drawdown calculation
            equity_values = [e['balance'] for e in equity_curve]
            peak = equity_values[0]
            max_drawdown = 0
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Enhanced metrics with Volume Profile context
            vp_trades = [t for t in completed_trades if t.get('volume_profile_active', False)]
            vwap_trades = [t for t in completed_trades if t.get('vwap_active', False)]
            
            results = {
                'total_return': total_return,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_drawdown': max_drawdown,
                'final_balance': balance,
                'enhanced_features': {
                    'volume_profile_trades': len(vp_trades),
                    'vwap_trades': len(vwap_trades),
                    'vp_win_rate': len([t for t in vp_trades if t['pnl'] > 0]) / len(vp_trades) if vp_trades else 0,
                    'vwap_win_rate': len([t for t in vwap_trades if t['pnl'] > 0]) / len(vwap_trades) if vwap_trades else 0
                },
                'trades': completed_trades,
                'equity_curve': equity_curve
            }
            
            self.logger.info("✅ Enhanced backtesting complete!")
            self.logger.info(f"   📊 Total Return: {total_return:.4f}")
            self.logger.info(f"   📈 Win Rate: {win_rate:.4f}")
            self.logger.info(f"   💰 Profit Factor: {profit_factor:.4f}")
            self.logger.info(f"   📉 Max Drawdown: {max_drawdown:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced backtesting failed: {e}")
            raise


if __name__ == "__main__":
    # Testing Enhanced AI Engine
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Enhanced AI Engine...")
    
    # Create sample data (more comprehensive)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=2000, freq='15min')
    
    # Generate realistic EURUSD data with trends
    prices = []
    volumes = []
    base_price = 1.1000
    trend = 0.00001  # Small upward trend
    
    for i in range(len(dates)):
        # Add trend and noise
        price_change = trend + np.random.normal(0, 0.0008)
        base_price += price_change
        
        # OHLC generation
        open_price = base_price
        high_price = open_price + abs(np.random.normal(0, 0.0005))
        low_price = open_price - abs(np.random.normal(0, 0.0005))
        close_price = open_price + np.random.normal(0, 0.0003)
        close_price = max(min(close_price, high_price), low_price)
        
        # Volume with some correlation to volatility
        volatility = abs(high_price - low_price)
        volume = abs(np.random.normal(1000 + volatility * 100000, 300))
        
        prices.append([open_price, high_price, low_price, close_price])
        volumes.append(volume)
    
    ohlcv_df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'], index=dates)
    ohlcv_df['volume'] = volumes
    
    print(f"✅ Generated {len(ohlcv_df)} bars of test data")
    
    # Test Enhanced AI Engine
    enhanced_ai = EnhancedAIEngine("EURUSD", "M15")
    
    # Train model
    print("\n🧪 Training enhanced AI model...")
    training_results = enhanced_ai.train_enhanced_model(ohlcv_df[:1500])  # Use first 1500 bars for training
    
    print(f"✅ Training Results:")
    print(f"   📊 Ensemble Accuracy: {training_results['ensemble_accuracy']:.4f}")
    print(f"   📈 Cross-validation: {training_results['cv_mean']:.4f} ± {training_results['cv_std']:.4f}")
    print(f"   🔥 Features: {training_results['feature_count']}")
    
    # Test prediction
    print("\n🧪 Testing enhanced prediction...")
    test_data = ohlcv_df[:1600]  # Use data up to bar 1600
    signal, confidence, details = enhanced_ai.predict_enhanced(test_data)
    
    print(f"✅ Prediction Results:")
    print(f"   📊 Signal: {signal}")
    print(f"   📈 Confidence: {confidence:.4f}")
    print(f"   🔥 Volume Profile Active: {details['volume_profile_active']}")
    print(f"   ⚡ VWAP Active: {details['vwap_active']}")
    
    # Test backtesting
    print("\n🧪 Testing enhanced backtesting...")
    evaluator = EnhancedModelEvaluator()
    backtest_results = evaluator.comprehensive_backtest(
        enhanced_ai, 
        ohlcv_df[1500:1800],  # Use bars 1500-1800 for backtest
        initial_balance=10000,
        risk_per_trade=0.02
    )
    
    print(f"✅ Backtest Results:")
    print(f"   📊 Total Return: {backtest_results['total_return']:.4f}")
    print(f"   📈 Win Rate: {backtest_results['win_rate']:.4f}")
    print(f"   💰 Profit Factor: {backtest_results['profit_factor']:.4f}")
    print(f"   📉 Max Drawdown: {backtest_results['max_drawdown']:.4f}")
    
    # Test model persistence
    print("\n🧪 Testing model save/load...")
    save_success = enhanced_ai.save_enhanced_model("test_enhanced_model.pkl")
    print(f"✅ Model saved: {save_success}")
    
    # Create new instance and load
    new_ai = EnhancedAIEngine("EURUSD", "M15")
    load_success = new_ai.load_enhanced_model("test_enhanced_model.pkl")
    print(f"✅ Model loaded: {load_success}")
    
    print(f"\n🎯 Enhanced AI Engine Ready for Phase 2 Integration!")
    print(f"   🚀 Volume Profile: Operational")
    print(f"   💪 VWAP Analysis: Multi-timeframe ready")
    print(f"   ⚡ Ensemble Models: 3-model voting system")
    print(f"   🧠 Enhanced Filtering: VP + VWAP + Structure")
    print(f"   📊 Performance Tracking: Real-time monitoring")