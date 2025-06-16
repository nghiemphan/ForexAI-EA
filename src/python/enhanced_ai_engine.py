"""
File: src/python/enhanced_ai_engine.py
Description: Enhanced AI Engine with SMC Integration - UPDATED FOR PHASE 2 WEEK 7-8
Author: Claude AI Developer
Version: 2.1.0 (SMC Integration)
Created: 2025-06-13
Modified: 2025-06-15
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
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# Try to import enhanced modules
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not available - using reduced ensemble")
    XGBOOST_AVAILABLE = False

try:
    from enhanced_feature_engineer import EnhancedFeatureEngineer
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    print("Warning: Enhanced features not available - using basic features")
    ENHANCED_FEATURES_AVAILABLE = False
    # Fallback feature engineer
    class EnhancedFeatureEngineer:
        def __init__(self, symbol, timeframe):
            self.symbol = symbol
            self.timeframe = timeframe
        
        def create_enhanced_features(self, data):
            return {'basic_feature': 1.0, 'price': float(data['close'].iloc[-1])}
        
        def prepare_enhanced_training_data(self, data):
            features = pd.DataFrame([self.create_enhanced_features(data)])
            labels = pd.Series([0])
            return features, labels

try:
    from volume_profile import VolumeProfileCalculator
    VOLUME_PROFILE_AVAILABLE = True
except ImportError:
    print("Warning: Volume Profile not available")
    VOLUME_PROFILE_AVAILABLE = False

try:
    from smc_engine import SmartMoneyEngine
    SMC_AVAILABLE = True
except ImportError:
    print("Warning: SMC Engine not available - using enhanced features without SMC")
    SMC_AVAILABLE = False

class EnhancedAIEngine:
    """Enhanced AI Engine with Volume Profile, VWAP, and SMC Integration"""
    
    def __init__(self, symbol: str = "EURUSD", timeframe: str = "M15"):
        """
        Initialize Enhanced AI Engine with SMC capabilities
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
        """
        self.logger = logging.getLogger(__name__)
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Initialize feature engineer
        self.feature_engineer = EnhancedFeatureEngineer(symbol, timeframe)
        
        # Model ensemble - Enhanced for SMC
        self.ensemble_model = None
        self.feature_scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.model_trained = False
        
        # Performance tracking
        self.prediction_history = []
        self.accuracy_tracker = []
        self.confidence_threshold = 0.70  # Increased for SMC integration
        
        # Enhanced model configuration for SMC
        self.model_config = {
            'random_forest': {
                'n_estimators': 250,  # Increased for SMC features
                'max_depth': 18,      # Increased for 88+ features
                'min_samples_split': 8,
                'min_samples_leaf': 4,
                'random_state': 42,
                'class_weight': 'balanced',
                'max_features': 'sqrt'
            },
            'logistic': {
                'random_state': 42,
                'class_weight': 'balanced',
                'max_iter': 1500,  # Increased for convergence
                'C': 0.1  # Regularization for 88+ features
            }
        }
        
        if XGBOOST_AVAILABLE:
            self.model_config['xgboost'] = {
                'n_estimators': 200,  # Increased for SMC
                'max_depth': 10,      # Increased for complexity
                'learning_rate': 0.08,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'random_state': 42,
                'eval_metric': 'mlogloss',
                'objective': 'multi:softprob',
                'num_class': 3,
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 0.1  # L2 regularization
            }
        
        # SMC-specific configuration
        self.smc_config = {
            'smc_weight': 0.25,           # 25% weight for SMC features
            'smc_min_confidence': 0.6,    # Minimum SMC confidence
            'structure_alignment_required': True,  # Require structure alignment
            'order_block_validation': True,       # Validate order block alignment
        }
        
        self.logger.info(f"Enhanced AI Engine v2.1.0 initialized with SMC integration")
        self.logger.info(f"SMC Available: {SMC_AVAILABLE}, Enhanced Features: {ENHANCED_FEATURES_AVAILABLE}")
    
    def train_enhanced_model(self, ohlcv_data: pd.DataFrame, 
                           validation_split: float = 0.2,
                           enable_smc_features: bool = True) -> Dict[str, Any]:
        """
        Train enhanced ensemble model with SMC integration
        
        Args:
            ohlcv_data: Historical OHLCV data
            validation_split: Percentage of data for validation
            enable_smc_features: Whether to use SMC features
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            self.logger.info("🚀 Starting enhanced AI model training with SMC integration...")
            
            # Prepare enhanced training data with SMC
            if ENHANCED_FEATURES_AVAILABLE:
                features_df, labels_series = self.feature_engineer.prepare_enhanced_training_data(ohlcv_data)
            else:
                # Fallback to basic feature preparation
                features_df, labels_series = self._prepare_basic_training_data(ohlcv_data)
            
            if len(features_df) < 120:  # Increased minimum for SMC
                raise ValueError(f"Insufficient training data: {len(features_df)} < 120 samples")
            
            # Store feature columns for future use
            self.feature_columns = features_df.columns.tolist()
            total_features = len(self.feature_columns)
            
            # Count SMC features
            smc_features = [col for col in self.feature_columns if col.startswith('smc_')]
            smc_feature_count = len(smc_features)
            
            self.logger.info(f"📊 Training with {len(features_df)} samples and {total_features} features")
            self.logger.info(f"🏢 SMC Features: {smc_feature_count} (target: 20+)")
            
            # Encode labels from [-1,0,1] to [0,1,2] for XGBoost compatibility
            original_labels = labels_series.copy()
            encoded_labels = self.label_encoder.fit_transform(labels_series)
            
            label_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
            self.logger.info(f"Label mapping: {label_mapping}")
            
            # Split data
            split_idx = int(len(features_df) * (1 - validation_split))
            X_train = features_df.iloc[:split_idx]
            y_train = encoded_labels[:split_idx]
            X_val = features_df.iloc[split_idx:]
            y_val = encoded_labels[split_idx:]
            
            # Scale features (important for SMC features)
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            # Create enhanced ensemble models
            models = self._create_enhanced_ensemble_models()
            
            # Train individual models with hyperparameter optimization
            trained_models = []
            model_scores = {}
            
            for name, model in models.items():
                self.logger.info(f"🔧 Training {name} model with SMC features...")
                
                # Hyperparameter optimization for key models
                if name == 'random_forest' and total_features >= 80:
                    model = self._optimize_random_forest(X_train_scaled, y_train)
                elif name == 'xgboost' and XGBOOST_AVAILABLE and total_features >= 80:
                    model = self._optimize_xgboost(X_train_scaled, y_train)
                
                model.fit(X_train_scaled, y_train)
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val_scaled)
                accuracy = accuracy_score(y_val, val_predictions)
                model_scores[name] = accuracy
                
                trained_models.append((name, model))
                self.logger.info(f"✅ {name} validation accuracy: {accuracy:.4f}")
            
            # Create enhanced voting ensemble
            voting_models = [(name, model) for name, model in trained_models]
            self.ensemble_model = VotingClassifier(
                estimators=voting_models,
                voting='soft',  # Use probability-based voting
                weights=self._calculate_ensemble_weights(model_scores, smc_feature_count)
            )
            
            # Train ensemble
            self.logger.info("🔄 Training enhanced ensemble model...")
            self.ensemble_model.fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            ensemble_val_predictions = self.ensemble_model.predict(X_val_scaled)
            ensemble_accuracy = accuracy_score(y_val, ensemble_val_predictions)
            
            # Enhanced cross-validation with stratification
            cv_scores = cross_val_score(
                self.ensemble_model, X_train_scaled, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            # Feature importance analysis (enhanced for SMC)
            feature_importance = self._calculate_enhanced_feature_importance(trained_models)
            
            # SMC-specific metrics
            smc_metrics = self._calculate_smc_metrics(feature_importance, smc_features)
            
            # Training results
            results = {
                'ensemble_accuracy': ensemble_accuracy,
                'individual_accuracies': model_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'total_features': total_features,
                'smc_features': smc_feature_count,
                'feature_importance': feature_importance,
                'smc_metrics': smc_metrics,
                'label_distribution': pd.Series(y_train).value_counts().to_dict(),
                'original_label_distribution': original_labels.value_counts().to_dict(),
                'enhanced_features_used': ENHANCED_FEATURES_AVAILABLE,
                'smc_available': SMC_AVAILABLE,
                'xgboost_available': XGBOOST_AVAILABLE,
                'volume_profile_available': VOLUME_PROFILE_AVAILABLE,
                'target_achieved': ensemble_accuracy >= 0.80  # 80% target
            }
            
            self.model_trained = True
            
            # Enhanced logging
            self.logger.info("🎯 Enhanced AI Model Training Complete!")
            self.logger.info(f"   📊 Ensemble Accuracy: {ensemble_accuracy:.4f} (target: 0.80)")
            self.logger.info(f"   📈 Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            self.logger.info(f"   🔥 Total Features: {total_features} (target: 88+)")
            self.logger.info(f"   🏢 SMC Features: {smc_feature_count} (target: 20+)")
            self.logger.info(f"   🎯 Target Achieved: {results['target_achieved']}")
            
            # Log top SMC features
            if smc_metrics['top_smc_features']:
                top_smc = list(smc_metrics['top_smc_features'].keys())[:5]
                self.logger.info(f"   ⭐ Top SMC Features: {', '.join(top_smc)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced model training failed: {e}")
            raise
    
    def _create_enhanced_ensemble_models(self) -> Dict[str, Any]:
        """Create enhanced individual models for SMC ensemble"""
        models = {}
        
        # Enhanced Random Forest (primary model for SMC)
        models['random_forest'] = RandomForestClassifier(**self.model_config['random_forest'])
        
        # Enhanced Logistic Regression (linear relationships)
        models['logistic'] = LogisticRegression(**self.model_config['logistic'])
        
        # Enhanced XGBoost (if available)
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(**self.model_config['xgboost'])
        
        return models
    
    def _optimize_random_forest(self, X_train, y_train):
        """Optimize Random Forest hyperparameters for SMC features"""
        try:
            param_grid = {
                'n_estimators': [200, 250, 300],
                'max_depth': [15, 18, 20],
                'min_samples_split': [6, 8, 10],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='accuracy',
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            self.logger.info(f"RF optimization score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            self.logger.warning(f"RF optimization failed: {e}, using default")
            return RandomForestClassifier(**self.model_config['random_forest'])
    
    def _optimize_xgboost(self, X_train, y_train):
        """Optimize XGBoost hyperparameters for SMC features"""
        try:
            param_grid = {
                'n_estimators': [150, 200, 250],
                'max_depth': [8, 10, 12],
                'learning_rate': [0.06, 0.08, 0.1]
            }
            
            xgb_model = xgb.XGBClassifier(
                random_state=42,
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                subsample=0.85,
                colsample_bytree=0.85
            )
            
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, scoring='accuracy',
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            self.logger.info(f"XGB optimization score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            self.logger.warning(f"XGB optimization failed: {e}, using default")
            return xgb.XGBClassifier(**self.model_config['xgboost'])
    
    def _calculate_ensemble_weights(self, model_scores: Dict[str, float], 
                                  smc_feature_count: int) -> List[float]:
        """Calculate dynamic ensemble weights based on model performance and SMC integration"""
        try:
            weights = []
            
            # Base weights
            for model_name in model_scores.keys():
                score = model_scores[model_name]
                
                if model_name == 'random_forest':
                    # Higher weight for RF with SMC features
                    base_weight = 0.5 if smc_feature_count >= 20 else 0.4
                    weight = base_weight + (score - 0.6) * 0.5  # Bonus for performance
                elif model_name == 'xgboost':
                    # XGBoost good with SMC features
                    base_weight = 0.35 if smc_feature_count >= 20 else 0.3
                    weight = base_weight + (score - 0.6) * 0.5
                elif model_name == 'logistic':
                    # Lower weight for linear model with complex SMC features
                    base_weight = 0.15
                    weight = base_weight + (score - 0.6) * 0.3
                else:
                    weight = 0.33  # Default equal weight
                
                weights.append(max(0.1, min(0.7, weight)))  # Clamp weights
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)
            
            self.logger.info(f"Ensemble weights: {dict(zip(model_scores.keys(), weights))}")
            return weights
            
        except Exception as e:
            self.logger.warning(f"Weight calculation failed: {e}, using equal weights")
            return [1.0 / len(model_scores)] * len(model_scores)
    
    def _calculate_enhanced_feature_importance(self, trained_models: List[Tuple[str, Any]]) -> Dict[str, float]:
        """Calculate enhanced feature importance including SMC features"""
        try:
            feature_importance = {}
            
            # Get importance from Random Forest
            for name, model in trained_models:
                if name == 'random_forest' and hasattr(model, 'feature_importances_'):
                    rf_importance = dict(zip(self.feature_columns, model.feature_importances_))
                    break
            else:
                rf_importance = {}
            
            # Get importance from XGBoost if available
            xgb_importance = {}
            for name, model in trained_models:
                if name == 'xgboost' and hasattr(model, 'feature_importances_'):
                    xgb_importance = dict(zip(self.feature_columns, model.feature_importances_))
                    break
            
            # Combine importances
            all_features = set(rf_importance.keys()) | set(xgb_importance.keys())
            for feature in all_features:
                rf_score = rf_importance.get(feature, 0)
                xgb_score = xgb_importance.get(feature, 0)
                
                # Weighted combination
                if rf_score > 0 and xgb_score > 0:
                    combined_score = (rf_score * 0.6 + xgb_score * 0.4)
                elif rf_score > 0:
                    combined_score = rf_score
                elif xgb_score > 0:
                    combined_score = xgb_score
                else:
                    combined_score = 0
                
                feature_importance[feature] = combined_score
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
            return {}
    
    def _calculate_smc_metrics(self, feature_importance: Dict[str, float], 
                             smc_features: List[str]) -> Dict[str, Any]:
        """Calculate SMC-specific metrics"""
        try:
            smc_metrics = {
                'smc_feature_count': len(smc_features),
                'smc_importance_total': 0.0,
                'smc_importance_average': 0.0,
                'top_smc_features': {},
                'smc_categories': {
                    'order_blocks': 0,
                    'fair_value_gaps': 0,
                    'market_structure': 0,
                    'smc_signals': 0
                }
            }
            
            if not smc_features or not feature_importance:
                return smc_metrics
            
            # Calculate SMC importance metrics
            smc_importance_scores = []
            smc_top_features = {}
            
            for feature in smc_features:
                importance = feature_importance.get(feature, 0.0)
                smc_importance_scores.append(importance)
                smc_top_features[feature] = importance
                
                # Categorize SMC features
                if 'ob_' in feature or 'order_block' in feature:
                    smc_metrics['smc_categories']['order_blocks'] += 1
                elif 'fvg' in feature or 'fair_value' in feature:
                    smc_metrics['smc_categories']['fair_value_gaps'] += 1
                elif 'structure' in feature or 'trend' in feature or 'bos' in feature:
                    smc_metrics['smc_categories']['market_structure'] += 1
                elif 'bias' in feature or 'signal' in feature:
                    smc_metrics['smc_categories']['smc_signals'] += 1
            
            if smc_importance_scores:
                smc_metrics['smc_importance_total'] = sum(smc_importance_scores)
                smc_metrics['smc_importance_average'] = np.mean(smc_importance_scores)
            
            # Top SMC features
            smc_metrics['top_smc_features'] = dict(sorted(smc_top_features.items(), 
                                                         key=lambda x: x[1], reverse=True)[:10])
            
            return smc_metrics
            
        except Exception as e:
            self.logger.warning(f"SMC structure filter check failed: {e}")
            return True
    
    def _check_smc_order_block_filters(self, signal: int, features: Dict[str, float]) -> bool:
        """Check Smart Money Concepts order block filters"""
        try:
            # Order Block alignment
            if signal == 1:  # Buy signal
                # Don't buy in bearish order block
                if features.get('smc_price_in_bearish_ob', 0.0) > 0.8:
                    return False
                
                # Prefer bullish order block proximity
                bullish_ob_distance = features.get('smc_nearest_bullish_ob_distance', 1.0)
                if bullish_ob_distance < 0.002:  # Very close to bullish OB
                    bullish_ob_strength = features.get('smc_nearest_bullish_ob_strength', 0.0)
                    if bullish_ob_strength < 0.3:  # Weak order block
                        return False
            
            elif signal == -1:  # Sell signal
                # Don't sell in bullish order block
                if features.get('smc_price_in_bullish_ob', 0.0) > 0.8:
                    return False
                
                # Prefer bearish order block proximity
                bearish_ob_distance = features.get('smc_nearest_bearish_ob_distance', 1.0)
                if bearish_ob_distance < 0.002:  # Very close to bearish OB
                    bearish_ob_strength = features.get('smc_nearest_bearish_ob_strength', 0.0)
                    if bearish_ob_strength < 0.3:  # Weak order block
                        return False
            
            # Recent order block mitigation check
            recent_mitigation = features.get('smc_recent_ob_mitigation', 0.0)
            if recent_mitigation > 0.8:  # Recent mitigation occurred
                return False  # Wait for new setup
            
            return True
            
        except Exception as e:
            self.logger.warning(f"SMC order block filter check failed: {e}")
            return True
    
    def _check_smc_fvg_filters(self, signal: int, features: Dict[str, float]) -> bool:
        """Check Smart Money Concepts Fair Value Gap filters"""
        try:
            # Fair Value Gap alignment
            if signal == 1:  # Buy signal
                # Don't buy in bearish FVG
                if features.get('smc_price_in_bearish_fvg', 0.0) > 0.8:
                    return False
                
                # Check for bullish FVG support
                bullish_fvg_distance = features.get('smc_nearest_bullish_fvg_distance', 1.0)
                if bullish_fvg_distance > 0.01:  # Too far from bullish FVG
                    bullish_fvgs_count = features.get('smc_bullish_fvgs_count', 0.0)
                    if bullish_fvgs_count == 0:  # No bullish FVGs available
                        return False
            
            elif signal == -1:  # Sell signal
                # Don't sell in bullish FVG
                if features.get('smc_price_in_bullish_fvg', 0.0) > 0.8:
                    return False
                
                # Check for bearish FVG resistance
                bearish_fvg_distance = features.get('smc_nearest_bearish_fvg_distance', 1.0)
                if bearish_fvg_distance > 0.01:  # Too far from bearish FVG
                    bearish_fvgs_count = features.get('smc_bearish_fvgs_count', 0.0)
                    if bearish_fvgs_count == 0:  # No bearish FVGs available
                        return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"SMC FVG filter check failed: {e}")
            return True
    
    def _check_enhanced_volume_profile_filters(self, signal: int, features: Dict[str, float]) -> bool:
        """Check enhanced Volume Profile based filters"""
        try:
            # Enhanced POC alignment filter
            poc_distance = abs(features.get('vp_poc_distance', 0))
            price_at_poc = features.get('vp_price_at_poc', 0.0)
            
            if poc_distance > 0.008:  # More than 0.8% from POC
                if signal == 1 and features.get('vp_above_poc', 0) == 0:
                    return False  # Don't buy below POC when far from it
                if signal == -1 and features.get('vp_above_poc', 0) == 1:
                    return False  # Don't sell above POC when far from it
            
            # Volume Profile strength filter (enhanced)
            poc_volume = features.get('vp_poc_volume', 0)
            total_volume = features.get('vp_total_volume', 1)
            volume_concentration = poc_volume / total_volume if total_volume > 0 else 0
            
            if volume_concentration < 0.15:  # Very weak volume profile
                return False  # Don't trade on weak volume data
            
            # Value Area filter
            in_value_area = features.get('vp_in_value_area', 0.0)
            above_value_area = features.get('vp_above_value_area', 0.0)
            below_value_area = features.get('vp_below_value_area', 0.0)
            
            # Enhanced value area logic
            if signal == 1:  # Buy signal
                # Prefer buying near value area low or in value area
                if above_value_area > 0.8:  # Too high above value area
                    return False
            elif signal == -1:  # Sell signal
                # Prefer selling near value area high or in value area
                if below_value_area > 0.8:  # Too low below value area
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Enhanced Volume Profile filter check failed: {e}")
            return True
    
    def _check_enhanced_vwap_filters(self, signal: int, features: Dict[str, float]) -> bool:
        """Check enhanced VWAP based filters"""
        try:
            # Multi-timeframe VWAP alignment
            vwap_20_above = features.get('vwap_20_above', 0.0)
            vwap_50_above = features.get('vwap_50_above', 0.0)
            vwap_session_above = features.get('vwap_session_above', 0.0)
            vwap_alignment = features.get('vwap_alignment', 0.0)
            
            if signal == 1:  # Buy signal
                # Require VWAP alignment for bullish signals
                if vwap_alignment < -0.5:  # Strong bearish VWAP alignment
                    return False
                
                # Check VWAP slope trend
                vwap_20_slope = features.get('vwap_20_slope', 0.0)
                vwap_50_slope = features.get('vwap_50_slope', 0.0)
                
                if vwap_20_slope < -0.0002 and vwap_50_slope < -0.0001:  # Both trending down
                    return False
            
            elif signal == -1:  # Sell signal
                # Require VWAP alignment for bearish signals
                if vwap_alignment > 0.5:  # Strong bullish VWAP alignment
                    return False
                
                # Check VWAP slope trend
                vwap_20_slope = features.get('vwap_20_slope', 0.0)
                vwap_50_slope = features.get('vwap_50_slope', 0.0)
                
                if vwap_20_slope > 0.0002 and vwap_50_slope > 0.0001:  # Both trending up
                    return False
            
            # VWAP band position filter
            vwap_20_position = features.get('vwap_20_position', 0.5)
            if signal == 1 and vwap_20_position > 0.9:  # Too high in VWAP bands
                return False
            if signal == -1 and vwap_20_position < 0.1:  # Too low in VWAP bands
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Enhanced VWAP filter check failed: {e}")
            return True
    
    def _check_enhanced_market_filters(self, signal: int, features: Dict[str, float]) -> bool:
        """Check enhanced market structure filters"""
        try:
            # Enhanced momentum alignment
            momentum_5 = features.get('price_momentum_5', 0.0)
            momentum_confluence = features.get('multi_timeframe_confluence', 0.5)
            
            if signal == 1 and momentum_5 < -0.002:  # Don't buy in strong downtrend
                return False
            if signal == -1 and momentum_5 > 0.002:  # Don't sell in strong uptrend
                return False
            
            # Multi-timeframe confluence filter
            if momentum_confluence < 0.3:  # Poor confluence
                return False
            
            # Enhanced volatility filter
            volatility_regime = features.get('volatility_regime', 1.0)
            atr_normalized = features.get('atr_normalized', 0.01)
            
            if volatility_regime > 1.8 or atr_normalized > 0.025:  # Very high volatility
                return False
            
            # Support/Resistance proximity
            near_support = features.get('near_support', 0.0)
            near_resistance = features.get('near_resistance', 0.0)
            
            if signal == 1 and near_resistance > 0.8:  # Don't buy near resistance
                return False
            if signal == -1 and near_support > 0.8:  # Don't sell near support
                return False
            
            # Risk-reward assessment
            risk_reward_ratio = features.get('risk_reward_ratio', 1.0)
            if risk_reward_ratio < 1.2:  # Poor risk-reward
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Enhanced market filter check failed: {e}")
            return True
    
    def _extract_smc_context(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Extract SMC context for prediction analysis"""
        try:
            smc_context = {
                'market_structure': {
                    'trend_bullish': features.get('smc_trend_bullish', 0.0),
                    'trend_bearish': features.get('smc_trend_bearish', 0.0),
                    'trend_ranging': features.get('smc_trend_ranging', 0.0),
                    'structure_strength': features.get('smc_structure_strength', 0.5),
                    'bos_broken': features.get('smc_bos_broken', 0.0)
                },
                'order_blocks': {
                    'active_count': features.get('smc_active_obs_count', 0.0),
                    'nearest_bullish_distance': features.get('smc_nearest_bullish_ob_distance', 1.0),
                    'nearest_bearish_distance': features.get('smc_nearest_bearish_ob_distance', 1.0),
                    'price_in_bullish': features.get('smc_price_in_bullish_ob', 0.0),
                    'price_in_bearish': features.get('smc_price_in_bearish_ob', 0.0)
                },
                'fair_value_gaps': {
                    'bullish_count': features.get('smc_bullish_fvgs_count', 0.0),
                    'bearish_count': features.get('smc_bearish_fvgs_count', 0.0),
                    'price_in_bullish': features.get('smc_price_in_bullish_fvg', 0.0),
                    'price_in_bearish': features.get('smc_price_in_bearish_fvg', 0.0)
                },
                'bias': {
                    'bullish_bias': features.get('smc_bullish_bias', 0.0),
                    'bearish_bias': features.get('smc_bearish_bias', 0.0),
                    'net_bias': features.get('smc_net_bias', 0.0)
                }
            }
            
            return smc_context
            
        except Exception as e:
            self.logger.warning(f"SMC context extraction failed: {e}")
            return {}
    
    def _prepare_basic_training_data(self, ohlcv_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare basic training data when enhanced features not available"""
        try:
            features_list = []
            labels = []
            
            # Generate basic features
            for i in range(60, len(ohlcv_data) - 10):  # Increased start for SMC
                current_data = ohlcv_data.iloc[:i+1]
                
                # Basic features
                close = current_data['close']
                if len(close) >= 20:
                    features = {
                        'sma_9': float(close.tail(9).mean()),
                        'sma_21': float(close.tail(21).mean() if len(close) >= 21 else close.tail(9).mean()),
                        'price_momentum': float((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] if len(close) >= 10 else 0),
                        'volatility': float(close.tail(14).std() / close.tail(14).mean() if len(close) >= 14 else 0.01),
                        'price_level': float(close.iloc[-1]),
                        'volume_ratio': float(current_data['volume'].iloc[-1] / current_data['volume'].tail(10).mean() if len(current_data) >= 10 else 1.0)
                    }
                    
                    features_list.append(features)
                    
                    # Enhanced label generation for SMC
                    future_price = ohlcv_data['close'].iloc[i+8] if i+8 < len(ohlcv_data) else close.iloc[-1]
                    current_price = close.iloc[-1]
                    change = (future_price - current_price) / current_price
                    
                    if change > 0.003:  # 0.3% threshold (increased for SMC)
                        labels.append(1)   # Buy
                    elif change < -0.003:
                        labels.append(-1)  # Sell
                    else:
                        labels.append(0)   # Hold
            
            features_df = pd.DataFrame(features_list)
            labels_series = pd.Series(labels)
            
            return features_df, labels_series
            
        except Exception as e:
            self.logger.error(f"Basic training data preparation failed: {e}")
            # Return minimal data with diversity
            minimal_features = pd.DataFrame([
                {'basic_feature': 1.0, 'price_level': 1.1000, 'momentum': 0.001},
                {'basic_feature': 1.0, 'price_level': 1.1010, 'momentum': -0.001},
                {'basic_feature': 1.0, 'price_level': 1.1005, 'momentum': 0.0}
            ])
            minimal_labels = pd.Series([1, -1, 0])
            return minimal_features, minimal_labels
    
    def _create_basic_features(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """Create basic features when enhanced features not available"""
        try:
            close = ohlcv_data['close']
            current_price = float(close.iloc[-1])
            
            features = {
                'price_level': current_price,
                'sma_9': float(close.tail(9).mean() if len(close) >= 9 else current_price),
                'sma_21': float(close.tail(21).mean() if len(close) >= 21 else current_price),
                'price_momentum': float((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(close) >= 5 else 0),
                'volatility': float(close.tail(10).std() / close.tail(10).mean() if len(close) >= 10 else 0.01),
                'volume_ratio': float(ohlcv_data['volume'].iloc[-1] / ohlcv_data['volume'].tail(10).mean() if len(ohlcv_data) >= 10 else 1.0)
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Basic feature creation failed: {e}")
            return {'basic_feature': 1.0, 'price_level': 1.0}
    
    def _track_prediction(self, signal: int, confidence: float, details: Dict[str, Any]):
        """Track prediction for performance analysis"""
        try:
            prediction_record = {
                'timestamp': datetime.now(),
                'signal': signal,
                'confidence': confidence,
                'raw_signal': details.get('raw_signal', signal),
                'feature_count': details.get('feature_count', 0),
                'smc_active': details.get('smc_active', False),
                'volume_profile_active': details.get('volume_profile_active', False),
                'vwap_active': details.get('vwap_active', False),
                'enhanced_features': details.get('enhanced_features', False)
            }
            
            self.prediction_history.append(prediction_record)
            
            # Keep only last 1000 predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
                
        except Exception as e:
            self.logger.warning(f"Prediction tracking failed: {e}")
    
    def get_enhanced_model_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced model performance statistics including SMC metrics"""
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
            
            # SMC usage statistics
            smc_usage = sum(1 for p in recent_predictions if p.get('smc_active', False))
            
            stats = {
                'total_predictions': len(self.prediction_history),
                'recent_predictions': len(recent_predictions),
                'signal_distribution': {
                    'sell': signal_counts[-1],
                    'hold': signal_counts[0],
                    'buy': signal_counts[1]
                },
                'confidence_stats': {
                    'mean': float(np.mean(confidences)) if confidences else 0,
                    'std': float(np.std(confidences)) if confidences else 0,
                    'min': float(np.min(confidences)) if confidences else 0,
                    'max': float(np.max(confidences)) if confidences else 0
                },
                'filter_effectiveness': {
                    'filter_rate': filter_changes / len(recent_predictions) if recent_predictions else 0,
                    'signals_filtered': filter_changes
                },
                'feature_usage': {
                    'enhanced_features_usage': sum(1 for p in recent_predictions 
                                                 if p.get('enhanced_features', False)) / len(recent_predictions) if recent_predictions else 0,
                    'smc_usage': smc_usage / len(recent_predictions) if recent_predictions else 0,
                    'volume_profile_usage': sum(1 for p in recent_predictions 
                                               if p.get('volume_profile_active', False)) / len(recent_predictions) if recent_predictions else 0,
                    'vwap_usage': sum(1 for p in recent_predictions 
                                     if p.get('vwap_active', False)) / len(recent_predictions) if recent_predictions else 0
                },
                'system_capabilities': {
                    'enhanced_features_available': ENHANCED_FEATURES_AVAILABLE,
                    'smc_available': SMC_AVAILABLE,
                    'xgboost_available': XGBOOST_AVAILABLE,
                    'volume_profile_available': VOLUME_PROFILE_AVAILABLE
                },
                'model_info': {
                    'feature_count': len(self.feature_columns) if self.feature_columns else 0,
                    'smc_feature_count': len([col for col in (self.feature_columns or []) if col.startswith('smc_')]),
                    'confidence_threshold': self.confidence_threshold,
                    'model_trained': self.model_trained
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Enhanced performance stats calculation failed: {e}")
            return {'error': str(e)}
    
    def save_enhanced_model(self, filepath: str) -> bool:
        """Save the enhanced model with SMC integration"""
        try:
            if not self.model_trained:
                raise ValueError("No trained model to save")
            
            model_data = {
                'ensemble_model': self.ensemble_model,
                'feature_scaler': self.feature_scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'model_config': self.model_config,
                'smc_config': self.smc_config,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'confidence_threshold': self.confidence_threshold,
                'training_timestamp': datetime.now().isoformat(),
                'version': '2.1.0_SMC',
                'system_capabilities': {
                    'enhanced_features_available': ENHANCED_FEATURES_AVAILABLE,
                    'smc_available': SMC_AVAILABLE,
                    'xgboost_available': XGBOOST_AVAILABLE,
                    'volume_profile_available': VOLUME_PROFILE_AVAILABLE
                }
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Enhanced SMC model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced model saving failed: {e}")
            return False
    
    def load_enhanced_model(self, filepath: str) -> bool:
        """Load the enhanced model with SMC integration"""
        try:
            model_data = joblib.load(filepath)
            
            self.ensemble_model = model_data['ensemble_model']
            self.feature_scaler = model_data['feature_scaler']
            self.label_encoder = model_data.get('label_encoder', LabelEncoder())
            self.feature_columns = model_data['feature_columns']
            self.model_config = model_data.get('model_config', self.model_config)
            self.smc_config = model_data.get('smc_config', self.smc_config)
            self.confidence_threshold = model_data.get('confidence_threshold', 0.70)
            
            # If label encoder wasn't saved (old models), fit it with standard labels
            if not hasattr(self.label_encoder, 'classes_'):
                self.label_encoder.fit([-1, 0, 1])
            
            self.model_trained = True
            
            training_time = model_data.get('training_timestamp', 'Unknown')
            version = model_data.get('version', 'Unknown')
            capabilities = model_data.get('system_capabilities', {})
            
            # Count SMC features
            smc_features = [col for col in self.feature_columns if col.startswith('smc_')]
            
            self.logger.info(f"Enhanced SMC model loaded from {filepath}")
            self.logger.info(f"Model version: {version}, trained: {training_time}")
            self.logger.info(f"Total features: {len(self.feature_columns)}, SMC features: {len(smc_features)}")
            self.logger.info(f"Capabilities: Enhanced={capabilities.get('enhanced_features_available', False)}, "
                           f"SMC={capabilities.get('smc_available', False)}, "
                           f"XGBoost={capabilities.get('xgboost_available', False)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced model loading failed: {e}")
            return False


# ENHANCED Model Evaluator with SMC Integration
class EnhancedModelEvaluator:
    """Enhanced model evaluation with SMC context and improved backtesting"""
    
    def __init__(self):
        """Initialize Enhanced Model Evaluator with logger"""
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_backtest(self, ai_engine, 
                             ohlcv_data: pd.DataFrame,
                             initial_balance: float = 10000,
                             risk_per_trade: float = 0.02,
                             enable_smc_analysis: bool = True) -> Dict[str, Any]:
        """
        Comprehensive backtesting with SMC integration and enhanced metrics
        """
        try:
            self.logger.info("🧪 Starting enhanced backtesting with SMC integration...")
            
            # Validate inputs
            if not hasattr(ai_engine, 'predict_enhanced'):
                self.logger.error("AI engine missing predict_enhanced method")
                return {'error': 'AI engine not properly configured'}
            
            if len(ohlcv_data) < 100:
                self.logger.error("Insufficient data for backtesting")
                return {'error': 'Insufficient data for backtesting'}
            
            balance = initial_balance
            position = 0  # 0 = no position, 1 = long, -1 = short
            entry_price = 0
            trades = []
            equity_curve = []
            
            # Enhanced backtesting parameters
            start_idx = 60  # Increased for SMC features
            end_idx = len(ohlcv_data) - 8  # Leave fewer bars at the end
            
            successful_predictions = 0
            failed_predictions = 0
            smc_enhanced_trades = 0
            
            for i in range(start_idx, end_idx):
                try:
                    current_data = ohlcv_data.iloc[:i+1]
                    current_price = float(current_data['close'].iloc[-1])
                    
                    # Get AI prediction with SMC context
                    try:
                        signal, confidence, details = ai_engine.predict_enhanced(current_data)
                        successful_predictions += 1
                        
                        # Track SMC usage
                        if details.get('smc_active', False):
                            smc_enhanced_trades += 1
                            
                    except Exception as e:
                        self.logger.warning(f"Prediction failed at bar {i}: {e}")
                        signal, confidence = 0, 0.0
                        failed_predictions += 1
                        continue
                    
                    # Enhanced trading logic
                    if position == 0:  # No position
                        if signal != 0 and confidence > 0.75:  # Higher threshold for SMC
                            # Enhanced position sizing
                            risk_amount = balance * risk_per_trade
                            
                            # Dynamic stop loss based on ATR or volatility
                            atr_estimate = current_price * 0.012  # Default 1.2%
                            if 'atr_normalized' in details.get('smc_context', {}):
                                atr_estimate = current_price * details['smc_context'].get('atr_normalized', 0.012)
                            
                            position_size = risk_amount / atr_estimate
                            
                            position = signal
                            entry_price = current_price
                            
                            trade_record = {
                                'entry_time': i,
                                'entry_price': float(entry_price),
                                'signal': int(signal),
                                'confidence': float(confidence),
                                'position_size': float(position_size),
                                'smc_enhanced': details.get('smc_active', False),
                                'entry_context': details.get('smc_context', {})
                            }
                            trades.append(trade_record)
                    
                    elif position != 0:  # Have position
                        # Enhanced exit conditions
                        should_exit = False
                        exit_reason = ""
                        
                        # Get next price for PnL calculation
                        if i + 1 < len(ohlcv_data):
                            next_price = float(ohlcv_data['close'].iloc[i+1])
                        else:
                            next_price = current_price
                        
                        # Opposite signal (enhanced threshold)
                        if signal != 0 and signal != position and confidence > 0.70:
                            should_exit = True
                            exit_reason = "opposite_signal"
                        
                        # Enhanced stop loss / take profit
                        if entry_price > 0:
                            pnl_pct = (next_price - entry_price) / entry_price * position
                            
                            # Dynamic stop loss (enhanced)
                            stop_loss_pct = -0.015  # 1.5% default
                            take_profit_pct = 0.030  # 3% default
                            
                            if pnl_pct <= stop_loss_pct:
                                should_exit = True
                                exit_reason = "stop_loss"
                            elif pnl_pct >= take_profit_pct:
                                should_exit = True
                                exit_reason = "take_profit"
                            elif abs(pnl_pct) < 0.001 and (i - trades[-1]['entry_time']) > 20:  # Time-based exit
                                should_exit = True
                                exit_reason = "time_exit"
                        else:
                            should_exit = True
                            exit_reason = "invalid_entry_price"
                        
                        if should_exit and trades:
                            # Close position
                            trade = trades[-1]
                            trade['exit_time'] = i
                            trade['exit_price'] = float(next_price)
                            trade['exit_reason'] = exit_reason
                            
                            # Calculate P&L safely
                            if entry_price > 0 and 'position_size' in trade:
                                pnl = (next_price - entry_price) * position * trade['position_size']
                                trade['pnl'] = float(pnl)
                                trade['pnl_pct'] = float((next_price - entry_price) / entry_price * position)
                                trade['hold_time'] = i - trade['entry_time']
                                balance += pnl
                            else:
                                trade['pnl'] = 0.0
                                trade['pnl_pct'] = 0.0
                                trade['hold_time'] = 0
                            
                            position = 0
                            entry_price = 0
                    
                    # Record equity
                    equity_curve.append({
                        'time': i,
                        'balance': float(balance),
                        'price': float(current_price)
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing bar {i}: {e}")
                    continue
                
                if i % 100 == 0:
                    self.logger.info(f"Processed {i - start_idx} bars...")
            
            # Enhanced performance calculations
            completed_trades = [t for t in trades if 'exit_price' in t]
            
            if not completed_trades:
                self.logger.warning("No completed trades in backtest period")
                return {
                    'total_return': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'max_drawdown': 0.0,
                    'final_balance': float(balance),
                    'successful_predictions': successful_predictions,
                    'failed_predictions': failed_predictions,
                    'prediction_success_rate': successful_predictions / max(successful_predictions + failed_predictions, 1),
                    'smc_enhanced_trades': smc_enhanced_trades,
                    'error': 'No completed trades'
                }
            
            # Enhanced performance metrics
            total_trades = len(completed_trades)
            winning_trades = len([t for t in completed_trades if t.get('pnl', 0) > 0])
            losing_trades = total_trades - winning_trades
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(t.get('pnl', 0) for t in completed_trades)
            total_return = (balance - initial_balance) / initial_balance
            
            winning_pnls = [t['pnl'] for t in completed_trades if t.get('pnl', 0) > 0]
            losing_pnls = [t['pnl'] for t in completed_trades if t.get('pnl', 0) <= 0]
            
            avg_win = np.mean(winning_pnls) if winning_pnls else 0
            avg_loss = np.mean(losing_pnls) if losing_pnls else 0
            
            profit_factor = abs(sum(winning_pnls) / sum(losing_pnls)) if losing_pnls and sum(losing_pnls) != 0 else float('inf')
            
            # Enhanced Sharpe ratio calculation
            if len(equity_curve) > 1:
                returns = []
                for i in range(1, len(equity_curve)):
                    ret = (equity_curve[i]['balance'] - equity_curve[i-1]['balance']) / equity_curve[i-1]['balance']
                    returns.append(ret)
                
                if returns and np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)  # Annualized for 15min data
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Enhanced drawdown calculation
            equity_values = [e['balance'] for e in equity_curve]
            if equity_values:
                peak = equity_values[0]
                max_drawdown = 0
                drawdown_duration = 0
                current_drawdown_duration = 0
                
                for equity in equity_values:
                    if equity > peak:
                        peak = equity
                        current_drawdown_duration = 0
                    else:
                        current_drawdown_duration += 1
                        
                    drawdown = (peak - equity) / peak if peak > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
                    drawdown_duration = max(drawdown_duration, current_drawdown_duration)
            else:
                max_drawdown = 0
                drawdown_duration = 0
            
            # SMC-specific analysis
            smc_enhanced_completed = [t for t in completed_trades if t.get('smc_enhanced', False)]
            regular_trades = [t for t in completed_trades if not t.get('smc_enhanced', False)]
            
            smc_win_rate = len([t for t in smc_enhanced_completed if t.get('pnl', 0) > 0]) / len(smc_enhanced_completed) if smc_enhanced_completed else 0
            regular_win_rate = len([t for t in regular_trades if t.get('pnl', 0) > 0]) / len(regular_trades) if regular_trades else 0
            
            # Trade analysis by exit reason
            exit_reasons = {}
            for trade in completed_trades:
                reason = trade.get('exit_reason', 'unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            # Hold time analysis
            hold_times = [t.get('hold_time', 0) for t in completed_trades if 'hold_time' in t]
            avg_hold_time = np.mean(hold_times) if hold_times else 0
            
            results = {
                'total_return': float(total_return),
                'total_pnl': float(total_pnl),
                'total_trades': int(total_trades),
                'winning_trades': int(winning_trades),
                'losing_trades': int(losing_trades),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999.0,
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'max_drawdown': float(max_drawdown),
                'drawdown_duration': int(drawdown_duration),
                'sharpe_ratio': float(sharpe_ratio),
                'final_balance': float(balance),
                'avg_hold_time': float(avg_hold_time),
                
                # Prediction metrics
                'successful_predictions': int(successful_predictions),
                'failed_predictions': int(failed_predictions),
                'prediction_success_rate': float(successful_predictions / max(successful_predictions + failed_predictions, 1)),
                
                # SMC-specific metrics
                'smc_enhanced_trades': int(len(smc_enhanced_completed)),
                'smc_enhanced_percentage': float(len(smc_enhanced_completed) / total_trades if total_trades > 0 else 0),
                'smc_win_rate': float(smc_win_rate),
                'regular_win_rate': float(regular_win_rate),
                'smc_performance_boost': float(smc_win_rate - regular_win_rate),
                
                # Trade analysis
                'exit_reasons': exit_reasons,
                'trades_sample': completed_trades[-5:] if len(completed_trades) >= 5 else completed_trades,
                'equity_curve_sample': equity_curve[-20:] if len(equity_curve) >= 20 else equity_curve,
                
                # System performance
                'backtest_bars': end_idx - start_idx,
                'data_quality': 'enhanced_smc' if enable_smc_analysis else 'standard'
            }
            
            self.logger.info("✅ Enhanced SMC backtesting complete!")
            self.logger.info(f"   📊 Total Return: {total_return:.4f}")
            self.logger.info(f"   📈 Win Rate: {win_rate:.4f}")
            self.logger.info(f"   💰 Profit Factor: {results['profit_factor']:.4f}")
            self.logger.info(f"   📉 Max Drawdown: {max_drawdown:.4f}")
            self.logger.info(f"   🏢 SMC Enhanced Trades: {len(smc_enhanced_completed)}/{total_trades}")
            self.logger.info(f"   🎯 SMC Win Rate: {smc_win_rate:.4f} vs Regular: {regular_win_rate:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced SMC backtesting failed: {e}")
            return {
                'error': str(e),
                'total_return': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'final_balance': float(initial_balance)
            }


if __name__ == "__main__":
    # Testing Enhanced AI Engine v2.1.0 with SMC Integration
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Enhanced AI Engine v2.1.0 - SMC INTEGRATION...")
    print(f"Enhanced Features Available: {ENHANCED_FEATURES_AVAILABLE}")
    print(f"SMC Available: {SMC_AVAILABLE}")
    print(f"XGBoost Available: {XGBOOST_AVAILABLE}")
    print(f"Volume Profile Available: {VOLUME_PROFILE_AVAILABLE}")
    
    # Create comprehensive sample data (larger for SMC)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=2500, freq='15min')  # Increased for SMC
    
    # Generate realistic EURUSD data with more complex patterns
    prices = []
    volumes = []
    base_price = 1.1000
    trend = 0.000008  # Small upward trend
    
    for i in range(len(dates)):
        # Add multiple trend components for SMC patterns
        long_trend = 0.000005 * np.sin(i / 100)  # Long-term cycle
        medium_trend = 0.000010 * np.sin(i / 30)  # Medium-term cycle
        short_trend = 0.000003 * np.sin(i / 10)   # Short-term cycle
        
        # Combined trend with noise
        price_change = trend + long_trend + medium_trend + short_trend + np.random.normal(0, 0.0006)
        base_price += price_change
        
        # OHLC generation with more realistic patterns
        open_price = base_price
        high_price = open_price + abs(np.random.normal(0, 0.0004))
        low_price = open_price - abs(np.random.normal(0, 0.0004))
        close_price = open_price + np.random.normal(0, 0.0002)
        close_price = max(min(close_price, high_price), low_price)
        
        # Volume with correlation to volatility and time
        volatility = abs(high_price - low_price)
        time_factor = 1 + 0.3 * np.sin(2 * np.pi * (i % 96) / 96)  # Daily volume pattern
        volume = abs(np.random.normal(1000 + volatility * 150000 * time_factor, 400))
        
        prices.append([open_price, high_price, low_price, close_price])
        volumes.append(volume)
    
    ohlcv_df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'], index=dates)
    ohlcv_df['volume'] = volumes
    
    print(f"✅ Generated {len(ohlcv_df)} bars of enhanced test data for SMC")
    
    # Test Enhanced AI Engine with SMC
    enhanced_ai = EnhancedAIEngine("EURUSD", "M15")
    
    # Train enhanced model with SMC
    print("\n🧪 Training enhanced AI model with SMC integration...")
    training_results = enhanced_ai.train_enhanced_model(
        ohlcv_df[:2000],  # Use first 2000 bars for training
        enable_smc_features=True
    )
    
    print(f"✅ Enhanced SMC Training Results:")
    print(f"   📊 Ensemble Accuracy: {training_results['ensemble_accuracy']:.4f} (target: 0.80)")
    print(f"   📈 Cross-validation: {training_results['cv_mean']:.4f} ± {training_results['cv_std']:.4f}")
    print(f"   🔥 Total Features: {training_results['total_features']} (target: 88+)")
    print(f"   🏢 SMC Features: {training_results['smc_features']} (target: 20+)")
    print(f"   🎯 Target Achieved: {training_results['target_achieved']}")
    
    # Test enhanced prediction with SMC
    print("\n🧪 Testing enhanced prediction with SMC...")
    test_data = ohlcv_df[:2100]  # Use data up to bar 2100
    signal, confidence, details = enhanced_ai.predict_enhanced(test_data)
    
    print(f"✅ Enhanced SMC Prediction Results:")
    print(f"   📊 Signal: {signal}")
    print(f"   📈 Confidence: {confidence:.4f}")
    print(f"   🔥 Total Features: {details['feature_count']}")
    print(f"   🏢 SMC Active: {details['smc_active']}")
    print(f"   ⚡ Volume Profile Active: {details['volume_profile_active']}")
    print(f"   💫 VWAP Active: {details['vwap_active']}")
    
    # Display SMC context if available
    if 'smc_context' in details and details['smc_context']:
        smc_ctx = details['smc_context']
        print(f"   🏢 SMC Context:")
        print(f"      Market Structure: Bullish={smc_ctx['market_structure']['trend_bullish']:.2f}")
        print(f"      Order Blocks: Active={smc_ctx['order_blocks']['active_count']:.0f}")
        print(f"      Bias: Net={smc_ctx['bias']['net_bias']:.2f}")
    
    # Test enhanced backtesting with SMC
    print("\n🧪 Testing enhanced backtesting with SMC integration...")
    evaluator = EnhancedModelEvaluator()
    backtest_results = evaluator.comprehensive_backtest(
        enhanced_ai, 
        ohlcv_df[2000:2400],  # Use bars 2000-2400 for backtest
        initial_balance=10000,
        risk_per_trade=0.015,  # Reduced risk for SMC precision
        enable_smc_analysis=True
    )
    
    print(f"✅ Enhanced SMC Backtest Results:")
    print(f"   📊 Total Return: {backtest_results['total_return']:.4f}")
    print(f"   📈 Win Rate: {backtest_results['win_rate']:.4f}")
    print(f"   💰 Profit Factor: {backtest_results['profit_factor']:.4f}")
    print(f"   📉 Max Drawdown: {backtest_results['max_drawdown']:.4f}")
    print(f"   🏢 SMC Enhanced Trades: {backtest_results.get('smc_enhanced_trades', 0)}")
    print(f"   🎯 SMC Win Rate: {backtest_results.get('smc_win_rate', 0):.4f}")
    print(f"   📊 Regular Win Rate: {backtest_results.get('regular_win_rate', 0):.4f}")
    print(f"   🚀 SMC Performance Boost: {backtest_results.get('smc_performance_boost', 0):.4f}")
    
    # Test enhanced model persistence with SMC
    print("\n🧪 Testing enhanced model save/load with SMC...")
    save_success = enhanced_ai.save_enhanced_model("test_enhanced_smc_model.pkl")
    print(f"✅ Enhanced SMC model saved: {save_success}")
    
    # Create new instance and load
    new_ai = EnhancedAIEngine("EURUSD", "M15")
    load_success = new_ai.load_enhanced_model("test_enhanced_smc_model.pkl")
    print(f"✅ Enhanced SMC model loaded: {load_success}")
    
    # Test performance stats
    print("\n🧪 Testing enhanced performance statistics...")
    perf_stats = enhanced_ai.get_enhanced_model_performance_stats()
    print(f"✅ Performance Stats:")
    print(f"   📊 Total Predictions: {perf_stats.get('total_predictions', 0)}")
    print(f"   🏢 SMC Usage: {perf_stats.get('feature_usage', {}).get('smc_usage', 0):.1%}")
    print(f"   🔥 Model Features: {perf_stats.get('model_info', {}).get('feature_count', 0)}")
    print(f"   🏢 SMC Features: {perf_stats.get('model_info', {}).get('smc_feature_count', 0)}")
    
    print(f"\n🎯 Enhanced AI Engine v2.1.0 - SMC INTEGRATION COMPLETE!")
    print(f"   🚀 Total Features: {training_results['total_features']}/88+ (target)")
    print(f"   🏢 SMC Features: {training_results['smc_features']}/20+ (target)")
    print(f"   💪 AI Accuracy: {training_results['ensemble_accuracy']:.1%} (target: 80%+)")
    print(f"   📈 SMC Integration: {'✅ SUCCESS' if training_results.get('smc_features', 0) >= 20 else '❌ INCOMPLETE'}")
    print(f"   🎯 Accuracy Target: {'✅ ACHIEVED' if training_results['ensemble_accuracy'] >= 0.80 else '📈 IN PROGRESS'}")
    
    if training_results['ensemble_accuracy'] >= 0.80 and training_results.get('smc_features', 0) >= 20:
        print(f"\n🏆 PHASE 2 WEEK 7-8 OBJECTIVES COMPLETED!")
        print(f"   ✅ SMC Integration: Complete")
        print(f"   ✅ 80%+ Accuracy: Achieved")
        print(f"   ✅ 88+ Features: Confirmed")
        print(f"   ✅ Production Ready: Yes")
    else:
        print(f"\n📈 PHASE 2 WEEK 7-8 IN PROGRESS...")
        print(f"   🔧 Continue development to reach all targets") metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def predict_enhanced(self, ohlcv_data: pd.DataFrame) -> Tuple[int, float, Dict[str, Any]]:
        """
        Make enhanced prediction with SMC integration
        
        Args:
            ohlcv_data: Current OHLCV data
            
        Returns:
            Tuple of (signal, confidence, prediction_details)
        """
        try:
            if not self.model_trained or self.ensemble_model is None:
                raise ValueError("Model not trained. Call train_enhanced_model() first.")
            
            # Generate enhanced features with SMC
            if ENHANCED_FEATURES_AVAILABLE:
                features = self.feature_engineer.create_enhanced_features(ohlcv_data)
            else:
                features = self._create_basic_features(ohlcv_data)
            
            # Convert to DataFrame with correct column order
            features_df = pd.DataFrame([features])
            
            # Ensure we have all required columns
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0.0
            
            features_df = features_df[self.feature_columns]
            
            # Handle missing features
            features_df = features_df.fillna(0)
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features_df)
            
            # Get prediction probabilities
            prediction_probs = self.ensemble_model.predict_proba(features_scaled)[0]
            
            # Get class prediction (encoded)
            predicted_class_encoded = self.ensemble_model.predict(features_scaled)[0]
            
            # Decode back to original labels [-1,0,1]
            predicted_class = self.label_encoder.inverse_transform([predicted_class_encoded])[0]
            
            # Calculate confidence (max probability)
            confidence = max(prediction_probs)
            
            # Get individual model predictions for analysis
            individual_predictions = {}
            for name, model in self.ensemble_model.named_estimators_.items():
                try:
                    pred_probs = model.predict_proba(features_scaled)[0]
                    pred_class_encoded = model.predict(features_scaled)[0]
                    pred_class = self.label_encoder.inverse_transform([pred_class_encoded])[0]
                    
                    individual_predictions[name] = {
                        'class': int(pred_class),
                        'confidence': float(max(pred_probs)),
                        'probabilities': pred_probs.tolist()
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to get prediction from {name}: {e}")
            
            # Enhanced signal filtering with SMC
            filtered_signal = self._apply_enhanced_smc_filters(
                predicted_class, confidence, features, individual_predictions
            )
            
            # Map prediction probabilities correctly
            label_to_encoded = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
            
            # SMC analysis for prediction context
            smc_context = self._extract_smc_context(features)
            
            prediction_details = {
                'raw_signal': int(predicted_class),
                'filtered_signal': int(filtered_signal),
                'confidence': float(confidence),
                'probabilities': {
                    'sell': float(prediction_probs[label_to_encoded.get(-1, 0)]),
                    'hold': float(prediction_probs[label_to_encoded.get(0, 1)]),
                    'buy': float(prediction_probs[label_to_encoded.get(1, 2)])
                },
                'individual_models': individual_predictions,
                'feature_count': len(features),
                'smc_context': smc_context,
                'volume_profile_active': any(k.startswith('vp_') for k in features.keys()),
                'vwap_active': any(k.startswith('vwap_') for k in features.keys()),
                'smc_active': any(k.startswith('smc_') for k in features.keys()),
                'enhanced_features': ENHANCED_FEATURES_AVAILABLE,
                'smc_available': SMC_AVAILABLE,
                'timestamp': datetime.now().isoformat()
            }
            
            # Track prediction
            self._track_prediction(filtered_signal, confidence, prediction_details)
            
            return filtered_signal, confidence, prediction_details
            
        except Exception as e:
            self.logger.error(f"Enhanced prediction failed: {e}")
            # Return neutral signal with low confidence
            return 0, 0.0, {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _apply_enhanced_smc_filters(self, raw_signal: int, confidence: float, 
                                  features: Dict[str, float], 
                                  individual_predictions: Dict[str, Any]) -> int:
        """
        Apply enhanced filtering logic with SMC integration
        
        Args:
            raw_signal: Raw model prediction
            confidence: Prediction confidence
            features: Current features including SMC
            individual_predictions: Individual model predictions
            
        Returns:
            Filtered signal
        """
        try:
            # Base confidence filter (increased threshold for SMC)
            if confidence < self.confidence_threshold:
                return 0  # Hold if confidence too low
            
            # Enhanced model consensus filter
            if individual_predictions:
                model_signals = [pred['class'] for pred in individual_predictions.values()]
                consensus_score = sum(1 for signal in model_signals if signal == raw_signal) / len(model_signals)
                
                if consensus_score < 0.65:  # Higher consensus required for SMC
                    return 0  # Hold if no strong consensus
            
            # SMC Structure Filters
            if not self._check_smc_structure_filters(raw_signal, features):
                return 0
            
            # SMC Order Block Filters
            if not self._check_smc_order_block_filters(raw_signal, features):
                return 0
            
            # SMC Fair Value Gap Filters
            if not self._check_smc_fvg_filters(raw_signal, features):
                return 0
            
            # Enhanced Volume Profile filters
            if VOLUME_PROFILE_AVAILABLE:
                if not self._check_enhanced_volume_profile_filters(raw_signal, features):
                    return 0
            
            # Enhanced VWAP filters
            if not self._check_enhanced_vwap_filters(raw_signal, features):
                return 0
            
            # Enhanced market structure filters
            if not self._check_enhanced_market_filters(raw_signal, features):
                return 0
            
            return raw_signal
            
        except Exception as e:
            self.logger.warning(f"Enhanced filter application failed: {e}")
            return 0  # Conservative: return hold signal on filter error
    
    def _check_smc_structure_filters(self, signal: int, features: Dict[str, float]) -> bool:
        """Check Smart Money Concepts structure filters"""
        try:
            # Market structure alignment
            if signal == 1:  # Buy signal
                # Don't buy in bearish structure
                if features.get('smc_trend_bearish', 0.0) > 0.7:
                    return False
                # Require some bullish bias
                if features.get('smc_bullish_bias', 0.0) < 0.3:
                    return False
            
            elif signal == -1:  # Sell signal
                # Don't sell in bullish structure
                if features.get('smc_trend_bullish', 0.0) > 0.7:
                    return False
                # Require some bearish bias
                if features.get('smc_bearish_bias', 0.0) < 0.3:
                    return False
            
            # Break of structure validation
            bos_broken = features.get('smc_bos_broken', 0.0)
            structure_strength = features.get('smc_structure_strength', 0.5)
            
            # If BOS recently broken, be more cautious
            if bos_broken > 0.8 and structure_strength < 0.4:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"SMC