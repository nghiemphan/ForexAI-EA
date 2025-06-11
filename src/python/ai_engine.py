# src/python/ai_engine.py
"""
AI Trading Engine for ForexAI-EA Project
Machine Learning model for forex trading predictions
Author: Claude AI Developer
Version: 1.0.0
Created: 2025-06-11
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
import joblib
import os
from datetime import datetime
import json

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Local imports
from technical_indicators import TechnicalIndicators
from feature_engineer import FeatureEngineer

class AITradingEngine:
    """
    Main AI Trading Engine with machine learning capabilities
    Handles model training, prediction, and performance monitoring
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.indicators_engine = TechnicalIndicators()
        self.feature_engineer = FeatureEngineer()
        
        # ML components
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.model_performance: Dict = {}
        
        # Configuration
        self.config = self._load_config(config_path)
        
        # Model paths
        self.model_dir = self.config.get('model_dir', 'data/models')
        self.model_path = os.path.join(self.model_dir, 'forex_ai_model.joblib')
        self.scaler_path = os.path.join(self.model_dir, 'feature_scaler.joblib')
        self.metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration for AI engine"""
        default_config = {
            'model_type': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'confidence_threshold': 0.7,
            'retrain_threshold': 1000,  # Retrain after N new samples
            'model_dir': 'data/models'
        }
        
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config.get('ai_engine', {}))
        
        return default_config
    
    def create_model(self) -> RandomForestClassifier:
        """Create and configure the ML model"""
        model = RandomForestClassifier(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split'],
            min_samples_leaf=self.config['min_samples_leaf'],
            random_state=self.config['random_state'],
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1  # Use all CPU cores
        )
        
        self.logger.info(f"Created {self.config['model_type']} model with parameters: {model.get_params()}")
        return model
    
    def train_model(self, historical_data: List[Dict], 
                   optimize_hyperparameters: bool = False) -> Dict:
        """
        Train the AI model on historical data
        
        Args:
            historical_data: List of OHLC data dictionaries
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Dictionary with training results and performance metrics
        """
        self.logger.info("Starting model training...")
        
        # Prepare training data
        features_df, labels_series = self.feature_engineer.prepare_training_data(historical_data)
        
        if len(features_df) < 100:
            raise ValueError(f"Insufficient training data: {len(features_df)} samples")
        
        # Store feature names
        self.feature_names = list(features_df.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels_series,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=labels_series
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimize hyperparameters if requested
        if optimize_hyperparameters:
            self.model = self._optimize_hyperparameters(X_train_scaled, y_train)
        else:
            self.model = self.create_model()
        
        # Train model
        self.logger.info("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)
        
        # Calculate performance metrics
        performance = {
            'train_accuracy': accuracy_score(y_train, train_predictions),
            'test_accuracy': accuracy_score(y_test, test_predictions),
            'feature_count': len(self.feature_names),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'class_distribution': labels_series.value_counts().to_dict(),
            'training_date': datetime.now().isoformat()
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=self.config['cv_folds'], scoring='accuracy'
        )
        performance['cv_mean'] = np.mean(cv_scores)
        performance['cv_std'] = np.std(cv_scores)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            performance['feature_importance'] = dict(sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:20])  # Top 20 features
        
        # Classification report
        performance['classification_report'] = classification_report(
            y_test, test_predictions, output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_predictions)
        performance['confusion_matrix'] = cm.tolist()
        
        self.model_performance = performance
        
        # Save model
        self.save_model()
        
        # Log results
        self.logger.info(f"Model training completed:")
        self.logger.info(f"  Train Accuracy: {performance['train_accuracy']:.3f}")
        self.logger.info(f"  Test Accuracy: {performance['test_accuracy']:.3f}")
        self.logger.info(f"  CV Score: {performance['cv_mean']:.3f} ± {performance['cv_std']:.3f}")
        
        return performance
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Optimize model hyperparameters using grid search"""
        self.logger.info("Optimizing hyperparameters...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        base_model = RandomForestClassifier(
            random_state=self.config['random_state'],
            class_weight='balanced',
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=3, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def predict(self, ohlc_data: Dict[str, List[float]]) -> Tuple[int, float, Dict]:
        """Make trading prediction for current market data"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        try:
            # Create features
            features = self.feature_engineer.create_features(ohlc_data)
            
            # Ensure we have all required features in correct order
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            # Convert to DataFrame with proper feature names
            import pandas as pd
            X = pd.DataFrame([feature_vector], columns=self.feature_names)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Calculate confidence (max probability)
            confidence = np.max(probabilities)
            
            # Get class probabilities
            classes = self.model.classes_
            class_probs = dict(zip(classes, probabilities))
            
            # CONVERT ALL NUMPY TYPES TO PYTHON TYPES (FIX HERE)
            prediction = int(prediction)  # Convert numpy.int64 to int
            confidence = float(confidence)  # Convert numpy.float64 to float
            
            # Convert class probabilities to regular Python types
            class_probs_clean = {}
            for key, value in class_probs.items():
                class_probs_clean[int(key)] = float(value)
            
            # Prepare metadata with clean types
            metadata = {
                'class_probabilities': class_probs_clean,
                'feature_count': int(len(feature_vector)),
                'model_performance': float(self.model_performance.get('test_accuracy', 0)),
                'prediction_time': datetime.now().isoformat()
            }
            
            # Apply confidence threshold
            if confidence < self.config['confidence_threshold']:
                prediction = 0  # Hold if confidence too low
                metadata['confidence_filter'] = True
            else:
                metadata['confidence_filter'] = False
            
            self.logger.debug(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
            
            return prediction, confidence, metadata
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return 0, 0.0, {'error': str(e)}
    
    def save_model(self) -> bool:
        """Save trained model, scaler, and metadata"""
        try:
            if self.model is None:
                self.logger.warning("No model to save")
                return False
            
            # Save model
            joblib.dump(self.model, self.model_path)
            
            # Save scaler
            if self.scaler is not None:
                joblib.dump(self.scaler, self.scaler_path)
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'model_performance': self.model_performance,
                'config': self.config,
                'save_date': datetime.now().isoformat()
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved to {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load trained model, scaler, and metadata"""
        try:
            if not os.path.exists(self.model_path):
                self.logger.info("No saved model found")
                return False
            
            # Load model
            self.model = joblib.load(self.model_path)
            
            # Load scaler
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names', [])
                    self.model_performance = metadata.get('model_performance', {})
            
            self.logger.info(f"Model loaded from {self.model_path}")
            self.logger.info(f"Model performance: {self.model_performance.get('test_accuracy', 'N/A')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        if self.model is None:
            return {'status': 'No model loaded'}
        
        info = {
            'status': 'Model loaded',
            'model_type': type(self.model).__name__,
            'feature_count': len(self.feature_names),
            'performance': self.model_performance,
            'config': self.config
        }
        
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        
        return info
    
    def validate_features(self, features: Dict[str, float]) -> bool:
        """Validate that features contain required data"""
        if not self.feature_names:
            return False
        
        required_features = set(self.feature_names)
        available_features = set(features.keys())
        
        missing_features = required_features - available_features
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
            return False
        
        return True
    
    def retrain_if_needed(self, new_data: List[Dict]) -> bool:
        """Check if model needs retraining and retrain if necessary"""
        if not new_data:
            return False
        
        # Check if we have enough new data for retraining
        total_new_samples = sum(len(data.get('close', [])) for data in new_data)
        
        if total_new_samples >= self.config['retrain_threshold']:
            self.logger.info(f"Retraining model with {total_new_samples} new samples...")
            try:
                self.train_model(new_data)
                return True
            except Exception as e:
                self.logger.error(f"Retraining failed: {e}")
                return False
        
        return False
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Sort by importance and return top N
        sorted_importance = dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n])
        
        return sorted_importance
    
    def analyze_prediction_confidence(self, confidence_history: List[float]) -> Dict:
        """Analyze prediction confidence over time"""
        if not confidence_history:
            return {}
        
        confidences = np.array(confidence_history)
        
        analysis = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'low_confidence_pct': np.mean(confidences < self.config['confidence_threshold']) * 100
        }
        
        return analysis


class ModelEvaluator:
    """
    Separate class for model evaluation and backtesting
    """
    
    def __init__(self, ai_engine: AITradingEngine):
        self.ai_engine = ai_engine
        self.logger = logging.getLogger(__name__)
    
    def backtest_model(self, test_data: List[Dict], 
                      initial_balance: float = 10000,
                      risk_per_trade: float = 0.02) -> Dict:
        """
        Backtest the AI model on historical data
        
        Args:
            test_data: Historical OHLC data for testing
            initial_balance: Starting balance for backtest
            risk_per_trade: Risk percentage per trade
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info("Starting model backtest...")
        
        results = {
            'trades': [],
            'equity_curve': [],
            'performance_metrics': {}
        }
        
        balance = initial_balance
        equity = initial_balance
        open_position = None
        
        for data_chunk in test_data:
            close_prices = data_chunk['close']
            high_prices = data_chunk['high']
            low_prices = data_chunk['low']
            
            for i in range(50, len(close_prices)):  # Start after enough data for indicators
                current_data = {
                    'open': data_chunk['open'][:i+1],
                    'high': data_chunk['high'][:i+1],
                    'low': data_chunk['low'][:i+1],
                    'close': data_chunk['close'][:i+1]
                }
                
                current_price = close_prices[i]
                
                try:
                    # Get AI prediction
                    signal, confidence, metadata = self.ai_engine.predict(current_data)
                    
                    # Close existing position if signal changes or stop/target hit
                    if open_position:
                        position_pnl = self._check_position_exit(
                            open_position, current_price, high_prices[i], low_prices[i]
                        )
                        
                        if position_pnl is not None:
                            # Close position
                            balance += position_pnl
                            equity = balance
                            
                            results['trades'].append({
                                'entry_price': open_position['entry_price'],
                                'exit_price': current_price,
                                'direction': open_position['direction'],
                                'pnl': position_pnl,
                                'duration': i - open_position['entry_bar']
                            })
                            
                            open_position = None
                    
                    # Open new position if signal and no current position
                    if signal != 0 and open_position is None and confidence >= 0.7:
                        position_size = self._calculate_position_size(
                            balance, current_price, risk_per_trade
                        )
                        
                        open_position = {
                            'direction': signal,
                            'entry_price': current_price,
                            'entry_bar': i,
                            'size': position_size,
                            'stop_loss': self._calculate_stop_loss(current_price, signal),
                            'take_profit': self._calculate_take_profit(current_price, signal)
                        }
                    
                    # Record equity
                    if open_position:
                        unrealized_pnl = self._calculate_unrealized_pnl(
                            open_position, current_price
                        )
                        current_equity = balance + unrealized_pnl
                    else:
                        current_equity = balance
                    
                    results['equity_curve'].append({
                        'bar': i,
                        'price': current_price,
                        'balance': balance,
                        'equity': current_equity,
                        'signal': signal,
                        'confidence': confidence
                    })
                
                except Exception as e:
                    self.logger.error(f"Backtest error at bar {i}: {e}")
                    continue
        
        # Calculate performance metrics
        results['performance_metrics'] = self._calculate_performance_metrics(
            results, initial_balance
        )
        
        self.logger.info(f"Backtest completed: {len(results['trades'])} trades")
        return results
    
    def _check_position_exit(self, position: Dict, current_price: float, 
                           high: float, low: float) -> Optional[float]:
        """Check if position should be closed and return P&L"""
        direction = position['direction']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        size = position['size']
        
        if direction == 1:  # Long position
            if low <= stop_loss:
                # Stop loss hit
                return size * (stop_loss - entry_price)
            elif high >= take_profit:
                # Take profit hit
                return size * (take_profit - entry_price)
        elif direction == -1:  # Short position
            if high >= stop_loss:
                # Stop loss hit
                return size * (entry_price - stop_loss)
            elif low <= take_profit:
                # Take profit hit
                return size * (entry_price - take_profit)
        
        return None
    
    def _calculate_position_size(self, balance: float, price: float, 
                               risk_pct: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = balance * risk_pct
        # Simplified position sizing - in real implementation would use ATR
        stop_distance = price * 0.005  # 0.5% stop loss
        return risk_amount / stop_distance
    
    def _calculate_stop_loss(self, entry_price: float, direction: int) -> float:
        """Calculate stop loss level"""
        stop_distance = entry_price * 0.005  # 0.5% stop
        if direction == 1:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def _calculate_take_profit(self, entry_price: float, direction: int) -> float:
        """Calculate take profit level"""
        profit_distance = entry_price * 0.015  # 1.5% target
        if direction == 1:
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance
    
    def _calculate_unrealized_pnl(self, position: Dict, current_price: float) -> float:
        """Calculate unrealized P&L for open position"""
        direction = position['direction']
        entry_price = position['entry_price']
        size = position['size']
        
        if direction == 1:
            return size * (current_price - entry_price)
        else:
            return size * (entry_price - current_price)
    
    def _calculate_performance_metrics(self, results: Dict, initial_balance: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        trades = results['trades']
        equity_curve = results['equity_curve']
        
        if not trades:
            return {'error': 'No trades to analyze'}
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t['pnl'] for t in trades)
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average metrics
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Drawdown calculation
        equity_values = [e['equity'] for e in equity_curve]
        max_drawdown = self._calculate_max_drawdown(equity_values)
        
        # Return metrics
        final_balance = equity_values[-1] if equity_values else initial_balance
        total_return = (final_balance - initial_balance) / initial_balance * 100
        
        # Sharpe ratio (simplified)
        returns = np.diff(equity_values) / equity_values[:-1] if len(equity_values) > 1 else [0]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': final_balance
        }
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd * 100  # Return as percentage


# Example usage and testing functions
def create_sample_data(n_bars: int = 1000) -> Dict:
    """Create sample OHLC data for testing"""
    np.random.seed(42)
    
    # Generate random walk price data
    base_price = 1.1000
    price_changes = np.random.normal(0, 0.001, n_bars)
    close_prices = [base_price]
    
    for change in price_changes:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(new_price)
    
    close_prices = close_prices[1:]  # Remove first element
    
    # Generate OHLC from close prices
    ohlc_data = {
        'open': [],
        'high': [],
        'low': [],
        'close': close_prices
    }
    
    for i, close in enumerate(close_prices):
        if i == 0:
            open_price = close
        else:
            open_price = close_prices[i-1]
        
        # Add some random variation for high/low
        high = max(open_price, close) + abs(np.random.normal(0, 0.0005))
        low = min(open_price, close) - abs(np.random.normal(0, 0.0005))
        
        ohlc_data['open'].append(open_price)
        ohlc_data['high'].append(high)
        ohlc_data['low'].append(low)
    
    return ohlc_data


def test_ai_engine():
    """Test function for AI engine"""
    print("Testing AI Trading Engine...")
    
    # Create AI engine
    ai_engine = AITradingEngine()
    
    # Create sample data
    sample_data = create_sample_data(2000)
    training_data = [sample_data]
    
    # Train model
    print("Training model...")
    performance = ai_engine.train_model(training_data)
    print(f"Training completed with accuracy: {performance['test_accuracy']:.3f}")
    
    # Test prediction
    test_data = {
        'open': sample_data['open'][-100:],
        'high': sample_data['high'][-100:],
        'low': sample_data['low'][-100:],
        'close': sample_data['close'][-100:]
    }
    
    signal, confidence, metadata = ai_engine.predict(test_data)
    print(f"Prediction: {signal}, Confidence: {confidence:.3f}")
    
    # Test backtest
    print("Running backtest...")
    evaluator = ModelEvaluator(ai_engine)
    backtest_results = evaluator.backtest_model([sample_data])
    
    metrics = backtest_results['performance_metrics']
    print(f"Backtest results:")
    print(f"  Total trades: {metrics.get('total_trades', 0)}")
    print(f"  Win rate: {metrics.get('win_rate', 0):.1%}")
    print(f"  Total return: {metrics.get('total_return', 0):.1f}%")
    
    return ai_engine


if __name__ == "__main__":
    test_ai_engine()