"""
File: src/python/socket_server.py
Description: Enhanced Socket Server v2.2.0 - Session-Aware AI Integration
Author: Claude AI Developer
Version: 2.2.0 - SESSION ENHANCED (Based on v2.1.0)
Created: 2025-06-15
Modified: 2025-06-15
Target: 106+ features with session intelligence and 80%+ AI accuracy
"""

import socket
import threading
import json
import time
import logging
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import traceback
import argparse

# Enhanced imports for v2.2.0
try:
    from enhanced_feature_engineer import EnhancedFeatureEngineer
    from enhanced_ai_engine import EnhancedAIEngine, SessionAwarePrediction
    ENHANCED_MODULES_AVAILABLE = True
    print("✅ Enhanced modules v2.2.0 imported successfully")
except ImportError as e:
    print(f"Warning: Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False
    # Create dummy classes for fallback
    class EnhancedFeatureEngineer:
        def __init__(self, symbol, timeframe): pass
        def create_enhanced_features(self, data, **kwargs): return {}
        def prepare_enhanced_training_data(self, data): return pd.DataFrame(), pd.Series()
    
    class EnhancedAIEngine:
        def __init__(self, symbol, timeframe): pass
        def train_session_enhanced_model(self, data): return {}
        def predict_session_aware(self, data, **kwargs): 
            return type('SessionAwarePrediction', (), {
                'signal': 0, 'confidence': 0.0, 'session_context': {},
                'technical_confidence': 0.0, 'session_confidence': 0.0,
                'smc_confidence': 0.0, 'volume_confidence': 0.0
            })()
        def save_session_enhanced_model(self, path): return False
        def load_session_enhanced_model(self, path): return False
        def get_session_performance_stats(self): return {}

@dataclass
class ServerStats:
    """Enhanced server statistics for session-aware AI"""
    total_requests: int = 0
    prediction_requests: int = 0
    session_enhanced_predictions: int = 0
    feature_generation_requests: int = 0
    model_training_requests: int = 0
    session_analysis_requests: int = 0
    smc_analysis_requests: int = 0
    avg_response_time: float = 0.0
    session_response_time: float = 0.0
    feature_count_avg: float = 0.0
    session_feature_count_avg: float = 0.0
    accuracy_target: float = 0.80
    version: str = "2.2.0"
    uptime_start: datetime = None
    errors: int = 0
    warnings: int = 0

class EnhancedSocketServer:
    """
    Enhanced Socket Server v2.2.0 - Session-Aware AI Trading System
    
    Features:
    - Session-enhanced predictions with 106+ features
    - Real-time session analysis and timing optimization
    - Advanced SMC integration from v2.1.0
    - Enhanced risk management with session context
    - Comprehensive performance monitoring
    - 80%+ accuracy target capability
    """
    
    def __init__(self, host: str = "localhost", port: int = 8888, 
                 symbol: str = "EURUSD", timeframe: str = "M15"):
        """
        Initialize Enhanced Socket Server v2.2.0
        
        Args:
            host: Server hostname
            port: Server port
            symbol: Trading symbol
            timeframe: Chart timeframe
        """
        self.host = host
        self.port = port
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Enhanced server configuration for v2.2.0
        self.server_config = {
            'max_connections': 10,
            'timeout': 30,
            'buffer_size': 8192 * 4,  # Increased for 106+ features
            'chunk_size': 1024 * 50,  # 50KB chunks for large data
            'max_data_size': 1024 * 1024 * 50,  # 50MB max data
            'session_analysis_enabled': True,
            'enhanced_logging': True,
            'performance_monitoring': True
        }
        
        # Initialize enhanced components
        if ENHANCED_MODULES_AVAILABLE:
            self.feature_engineer = EnhancedFeatureEngineer(symbol, timeframe)
            self.ai_engine = EnhancedAIEngine(symbol, timeframe)
            self.logger = self._setup_enhanced_logging()
            self.logger.info("✅ Enhanced Socket Server v2.2.0 initialized with session intelligence")
        else:
            self.feature_engineer = None
            self.ai_engine = None
            self.logger = self._setup_basic_logging()
            self.logger.warning("⚠️ Running in fallback mode without enhanced features")
        
        # Server state management
        self.socket = None
        self.running = False
        self.clients = []
        self.price_data = {}
        self.model_trained = False
        
        # Enhanced statistics tracking
        self.stats = ServerStats(uptime_start=datetime.now())
        self.response_times = []
        self.session_response_times = []
        self.feature_counts = []
        self.session_feature_counts = []
        
        # Session-specific caching for performance
        self.session_cache = {
            'last_session_analysis': None,
            'last_session_timestamp': None,
            'session_features_cache': {},
            'optimal_windows_cache': {}
        }
        
        # Enhanced command handlers mapping
        self.command_handlers = {
            # Core prediction (enhanced with session)
            'predict': self._handle_session_prediction_request,
            'get_prediction': self._handle_session_prediction_request,
            
            # Session-specific endpoints (NEW in v2.2.0)
            'session_analysis': self._handle_session_analysis_request,
            'session_optimal_windows': self._handle_session_timing_request,
            'session_risk_analysis': self._handle_session_risk_request,
            'session_performance': self._handle_session_performance_request,
            
            # Enhanced feature generation
            'generate_features': self._handle_enhanced_feature_request,
            'get_features': self._handle_enhanced_feature_request,
            
            # Model management (enhanced)
            'train_model': self._handle_enhanced_training_request,
            'save_model': self._handle_model_save_request,
            'load_model': self._handle_model_load_request,
            'model_info': self._handle_model_info_request,
            
            # SMC analysis (from v2.1.0)
            'smc_analysis': self._handle_smc_analysis_request,
            'order_blocks': self._handle_order_blocks_request,
            'fair_value_gaps': self._handle_fvg_request,
            'market_structure': self._handle_market_structure_request,
            
            # System management
            'server_stats': self._handle_server_stats_request,
            'health_check': self._handle_health_check_request,
            'capabilities': self._handle_capabilities_request,
            'ping': self._handle_ping_request,
            
            # Data management
            'update_data': self._handle_data_update_request,
            'get_data_info': self._handle_data_info_request
        }
        
        self.logger.info(f"Enhanced Socket Server v2.2.0 ready with {len(self.command_handlers)} endpoints")
    
    def _setup_enhanced_logging(self) -> logging.Logger:
        """Setup enhanced logging for v2.2.0"""
        logger = logging.getLogger(f"EnhancedSocketServer_{self.symbol}")
        
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            
            # Console handler with enhanced format
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [v2.2.0] %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler for session analysis
            try:
                os.makedirs('logs', exist_ok=True)
                file_handler = logging.FileHandler(f'logs/enhanced_server_v220_{self.symbol}.log')
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - [v2.2.0] %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not setup file logging: {e}")
        
        return logger
    
    def _setup_basic_logging(self) -> logging.Logger:
        """Setup basic logging for fallback mode"""
        logger = logging.getLogger(f"BasicSocketServer_{self.symbol}")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def start(self) -> bool:
        """Start the enhanced socket server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(self.server_config['max_connections'])
            
            self.running = True
            self.stats.uptime_start = datetime.now()
            
            self.logger.info(f"🚀 Enhanced Socket Server v2.2.0 started on {self.host}:{self.port}")
            self.logger.info(f"   📊 Features: 106+ with session intelligence")
            self.logger.info(f"   🎯 Target Accuracy: {self.stats.accuracy_target:.0%}")
            self.logger.info(f"   🌍 Session Analysis: {'Enabled' if self.server_config['session_analysis_enabled'] else 'Disabled'}")
            self.logger.info(f"   📡 Max Connections: {self.server_config['max_connections']}")
            
            while self.running:
                try:
                    client_socket, address = self.socket.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        self.logger.error(f"Error accepting client: {e}")
                        self.stats.errors += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the enhanced socket server"""
        self.running = False
        
        # Close client connections
        for client in self.clients[:]:
            try:
                client.close()
            except:
                pass
        self.clients.clear()
        
        # Close server socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        self.logger.info("Enhanced Socket Server v2.2.0 stopped")
    
    def _handle_client(self, client_socket: socket.socket, address: Tuple[str, int]) -> None:
        """Handle client connection with enhanced features"""
        self.clients.append(client_socket)
        client_id = f"{address[0]}:{address[1]}"
        
        self.logger.info(f"New client connected: {client_id}")
        
        try:
            client_socket.settimeout(self.server_config['timeout'])
            
            while self.running:
                try:
                    # Enhanced data receiving with chunking support
                    data = self._receive_enhanced_data(client_socket)
                    
                    if not data:
                        break
                    
                    # Process request with enhanced features
                    response = self._process_enhanced_request(data, client_id)
                    
                    # Send response with chunking if needed
                    self._send_enhanced_response(client_socket, response)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"Error handling client {client_id}: {e}")
                    self.stats.errors += 1
                    break
        
        except Exception as e:
            self.logger.error(f"Client handler error for {client_id}: {e}")
            self.stats.errors += 1
        
        finally:
            try:
                client_socket.close()
                if client_socket in self.clients:
                    self.clients.remove(client_socket)
                self.logger.info(f"Client disconnected: {client_id}")
            except:
                pass
    
    def _receive_enhanced_data(self, client_socket: socket.socket) -> Optional[Dict[str, Any]]:
        """Enhanced data receiving with support for large payloads"""
        try:
            # First, receive the header with data size
            header_data = client_socket.recv(1024).decode('utf-8', errors='replace')
            
            if not header_data:
                return None
            
            # Parse header to get data size
            if header_data.startswith('SIZE:'):
                try:
                    size_info = header_data.split('SIZE:')[1].split('\n')[0]
                    expected_size = int(size_info)
                    
                    if expected_size > self.server_config['max_data_size']:
                        self.logger.warning(f"Data size {expected_size} exceeds limit {self.server_config['max_data_size']}")
                        return None
                    
                    # Receive data in chunks
                    received_data = b''
                    remaining_size = expected_size
                    
                    while remaining_size > 0:
                        chunk_size = min(self.server_config['chunk_size'], remaining_size)
                        chunk = client_socket.recv(chunk_size)
                        
                        if not chunk:
                            break
                        
                        received_data += chunk
                        remaining_size -= len(chunk)
                    
                    if len(received_data) == expected_size:
                        return json.loads(received_data.decode('utf-8', errors='replace'))
                    else:
                        self.logger.warning(f"Received {len(received_data)} bytes, expected {expected_size}")
                        return None
                        
                except (ValueError, json.JSONDecodeError) as e:
                    self.logger.error(f"Error parsing chunked data: {e}")
                    return None
            else:
                # Standard small request
                try:
                    return json.loads(header_data)
                except json.JSONDecodeError:
                    # Might be incomplete, receive more
                    additional_data = client_socket.recv(self.server_config['buffer_size'])
                    full_data = header_data + additional_data.decode('utf-8', errors='replace')
                    return json.loads(full_data)
                    
        except Exception as e:
            self.logger.error(f"Enhanced data receive error: {e}")
            return None
    
    def _send_enhanced_response(self, client_socket: socket.socket, response: Dict[str, Any]) -> None:
        """Enhanced response sending with chunking support"""
        try:
            response_json = json.dumps(response, ensure_ascii=False)
            response_bytes = response_json.encode('utf-8', errors='replace')
            
            # If response is large, use chunked sending
            if len(response_bytes) > self.server_config['chunk_size']:
                # Send size header first
                header = f"SIZE:{len(response_bytes)}\n"
                client_socket.send(header.encode('utf-8'))
                
                # Send data in chunks
                sent_bytes = 0
                while sent_bytes < len(response_bytes):
                    chunk_end = min(sent_bytes + self.server_config['chunk_size'], len(response_bytes))
                    chunk = response_bytes[sent_bytes:chunk_end]
                    client_socket.send(chunk)
                    sent_bytes += len(chunk)
            else:
                # Send normally for small responses
                client_socket.send(response_bytes)
                
        except Exception as e:
            self.logger.error(f"Enhanced response send error: {e}")
    
    def _process_enhanced_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Process request with enhanced v2.2.0 features"""
        start_time = time.time()
        
        try:
            self.stats.total_requests += 1
            
            command = request.get('command', 'unknown')
            
            # Log enhanced request info
            if self.server_config['enhanced_logging']:
                self.logger.debug(f"Processing enhanced request: {command} from {client_id}")
            
            # Route to appropriate handler
            if command in self.command_handlers:
                response = self.command_handlers[command](request, client_id)
            else:
                response = {
                    'success': False,
                    'error': f'Unknown command: {command}',
                    'available_commands': list(self.command_handlers.keys()),
                    'version': self.stats.version
                }
            
            # Track response time
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            # Keep only recent response times for averaging
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            self.stats.avg_response_time = np.mean(self.response_times)
            
            # Add metadata to response
            response['response_time'] = response_time
            response['timestamp'] = datetime.now().isoformat()
            response['server_version'] = self.stats.version
            
            return response
            
        except Exception as e:
            self.logger.error(f"Enhanced request processing error: {e}")
            self.stats.errors += 1
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc() if self.server_config['enhanced_logging'] else None,
                'response_time': time.time() - start_time,
                'server_version': self.stats.version
            }
    
    # ========== SESSION-ENHANCED PREDICTION HANDLERS ==========
    
    def _handle_session_prediction_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Enhanced prediction with session intelligence"""
        try:
            self.stats.prediction_requests += 1
            
            if not ENHANCED_MODULES_AVAILABLE:
                return self._handle_fallback_prediction(request)
                
            if not self.model_trained:
                return {
                    'success': False,
                    'error': 'Model not trained. Use train_model command first.',
                    'suggestion': 'Send train_model request with historical data'
                }
            
            # Extract request data
            symbol = request.get('symbol', self.symbol)
            data = request.get('data', [])
            current_timestamp = request.get('timestamp')
            
            if not data:
                return {
                    'success': False,
                    'error': 'No price data provided',
                    'required_format': 'Array of OHLCV data'
                }
            
            # Convert to DataFrame
            try:
                if isinstance(data[0], dict):
                    ohlcv_df = pd.DataFrame(data)
                else:
                    ohlcv_df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
                
                # Ensure proper data types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    ohlcv_df[col] = pd.to_numeric(ohlcv_df[col], errors='coerce')
                
                # Handle timestamp
                if current_timestamp:
                    try:
                        current_timestamp = pd.to_datetime(current_timestamp)
                        if current_timestamp.tz is None:
                            current_timestamp = current_timestamp.replace(tzinfo=timezone.utc)
                    except:
                        current_timestamp = datetime.now(timezone.utc)
                else:
                    current_timestamp = datetime.now(timezone.utc)
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Data conversion failed: {e}',
                    'data_sample': str(data[:2]) if len(data) > 0 else 'No data'
                }
            
            # Make session-aware prediction
            start_time = time.time()
            prediction = self.ai_engine.predict_session_aware(ohlcv_df, current_timestamp=current_timestamp)
            prediction_time = time.time() - start_time
            
            # Track session prediction statistics
            if hasattr(prediction, 'session_context') and prediction.session_context.get('session_enhanced', False):
                self.stats.session_enhanced_predictions += 1
                self.session_response_times.append(prediction_time)
                
                if len(self.session_response_times) > 50:
                    self.session_response_times = self.session_response_times[-50:]
                self.stats.session_response_time = np.mean(self.session_response_times)
            
            # Prepare enhanced response
            response = {
                'success': True,
                'symbol': symbol,
                'prediction': {
                    'signal': prediction.signal,
                    'confidence': float(prediction.confidence),
                    'filtered_signal': prediction.filtered_signal,
                    'signal_text': self._get_signal_text(prediction.signal),
                    'confidence_level': self._get_confidence_level(prediction.confidence)
                },
                'session_analysis': {
                    'session_name': prediction.session_context.get('session_name', 'Unknown'),
                    'activity_score': prediction.session_context.get('activity_score', 0.8),
                    'optimal_window': prediction.session_context.get('optimal_window', False),
                    'volatility_regime': prediction.session_context.get('volatility_regime', 0.5),
                    'liquidity_level': prediction.session_context.get('liquidity_level', 0.8),
                    'session_enhanced': prediction.session_context.get('session_enhanced', False),
                    'risk_multiplier': prediction.session_context.get('risk_multiplier', 1.0)
                },
                'component_analysis': {
                    'technical_confidence': float(prediction.technical_confidence),
                    'volume_confidence': float(prediction.volume_confidence),
                    'smc_confidence': float(prediction.smc_confidence),
                    'session_confidence': float(prediction.session_confidence)
                },
                'ensemble_weights': prediction.ensemble_weights,
                'timestamp': current_timestamp.isoformat(),
                'prediction_time': prediction_time,
                'version': '2.2.0'
            }
            
            # Add session recommendations
            response['session_recommendations'] = self._get_session_recommendations(prediction)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Session prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_available': True
            }
    
    def _handle_session_analysis_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """NEW: Handle session analysis request"""
        try:
            self.stats.session_analysis_requests += 1
            
            if not ENHANCED_MODULES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Enhanced modules not available',
                    'version': '2.2.0'
                }
            
            # Extract parameters
            current_timestamp = request.get('timestamp')
            if current_timestamp:
                try:
                    current_timestamp = pd.to_datetime(current_timestamp)
                    if current_timestamp.tz is None:
                        current_timestamp = current_timestamp.replace(tzinfo=timezone.utc)
                except:
                    current_timestamp = datetime.now(timezone.utc)
            else:
                current_timestamp = datetime.now(timezone.utc)
            
            # Create dummy OHLCV data for session analysis
            dummy_data = pd.DataFrame({
                'open': [1.1000], 'high': [1.1010], 'low': [1.0990], 
                'close': [1.1005], 'volume': [1000]
            }, index=[current_timestamp])
            
            # Generate session features
            session_features = self.feature_engineer._get_session_features(dummy_data, current_timestamp)
            
            # Enhanced session analysis
            current_hour = current_timestamp.hour
            session_info = self.feature_engineer._identify_trading_session(current_hour)
            timing_info = self.feature_engineer._calculate_session_timing(current_hour, current_timestamp)
            
            return {
                'success': True,
                'session_analysis': {
                    'current_session': {
                        'name': self.feature_engineer._get_session_name(session_info['session_id']),
                        'id': int(session_info['session_id']),
                        'activity_score': float(session_info['activity_score']),
                        'volatility_multiplier': float(session_info['volatility_multiplier']),
                        'liquidity_level': float(session_info.get('liquidity_level', 0.8)),
                        'institutional_active': float(session_info.get('institution_active', 0.5))
                    },
                    'timing': {
                        'progress': float(timing_info['progress']),
                        'remaining': float(timing_info['remaining']),
                        'optimal_window': bool(timing_info['optimal']),
                        'momentum_phase': float(timing_info.get('momentum_phase', 0.5))
                    },
                    'session_features': session_features,
                    'feature_count': len(session_features),
                    'enhanced_features': len([k for k in session_features.keys() if 'enhanced' in k or 'efficiency' in k or 'correlation' in k])
                },
                'timestamp': current_timestamp.isoformat(),
                'version': '2.2.0'
            }
            
        except Exception as e:
            self.logger.error(f"Session analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    def _handle_session_timing_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """NEW: Handle session timing and optimal windows request"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Calculate optimal windows for next 24 hours
            optimal_windows = []
            
            for hour_offset in range(24):
                future_time = current_time.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=hour_offset)
                hour = future_time.hour
                
                session_info = self.feature_engineer._identify_trading_session(hour) if ENHANCED_MODULES_AVAILABLE else {'session_id': 1, 'activity_score': 0.8}
                timing_info = self.feature_engineer._calculate_session_timing(hour, future_time) if ENHANCED_MODULES_AVAILABLE else {'optimal': False, 'progress': 0.5}
                
                session_name = {0: 'Asian', 1: 'London', 2: 'New York'}.get(session_info['session_id'], 'Unknown')
                
                optimal_windows.append({
                    'time': future_time.isoformat(),
                    'hour': hour,
                    'session': session_name,
                    'activity_score': float(session_info['activity_score']),
                    'optimal': bool(timing_info['optimal']),
                    'session_progress': float(timing_info.get('progress', 0.5)),
                    'momentum_phase': float(timing_info.get('momentum_phase', 0.5)) if ENHANCED_MODULES_AVAILABLE else 0.5
                })
            
            # Find best trading windows
            best_windows = sorted(optimal_windows, key=lambda x: x['activity_score'], reverse=True)[:6]
            optimal_only = [w for w in optimal_windows if w['optimal']]
            
            return {
                'success': True,
                'optimal_windows': {
                    'next_24_hours': optimal_windows,
                    'best_windows': best_windows,
                    'optimal_only': optimal_only,
                    'next_optimal': optimal_only[0] if optimal_only else None,
                    'current_session_optimal': any(w['optimal'] and w['hour'] == current_time.hour for w in optimal_windows)
                },
                'session_recommendations': {
                    'high_activity_sessions': ['London', 'New York'],
                    'overlap_periods': ['London-NY (13:00-17:00 UTC)'],
                    'avoid_periods': ['Asian Late (06:00-08:00 UTC)', 'NY Late (22:00-24:00 UTC)']
                },
                'timestamp': current_time.isoformat(),
                'version': '2.2.0'
            }
            
        except Exception as e:
            self.logger.error(f"Session timing analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    def _handle_session_risk_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """NEW: Handle session risk analysis request"""
        try:
            current_timestamp = request.get('timestamp')
            if current_timestamp:
                try:
                    current_timestamp = pd.to_datetime(current_timestamp)
                    if current_timestamp.tz is None:
                        current_timestamp = current_timestamp.replace(tzinfo=timezone.utc)
                except:
                    current_timestamp = datetime.now(timezone.utc)
            else:
                current_timestamp = datetime.now(timezone.utc)
            
            current_hour = current_timestamp.hour
            
            if ENHANCED_MODULES_AVAILABLE:
                # Create dummy data for risk analysis
                dummy_data = pd.DataFrame({
                    'open': [1.1000, 1.1005, 1.1010], 
                    'high': [1.1015, 1.1020, 1.1025], 
                    'low': [1.0995, 1.1000, 1.1005], 
                    'close': [1.1005, 1.1010, 1.1015], 
                    'volume': [1000, 1100, 1200]
                }, index=pd.date_range(current_timestamp, periods=3, freq='15min'))
                
                risk_factors = self.feature_engineer._calculate_session_risk_factors(current_hour, dummy_data)
                session_info = self.feature_engineer._identify_trading_session(current_hour)
            else:
                risk_factors = {
                    'risk_multiplier': 1.0,
                    'news_risk': 0.5,
                    'correlation_risk': 0.5,
                    'gap_risk': 0.3
                }
                session_info = {'session_id': 1, 'activity_score': 0.8}
            
            # Session-specific risk analysis
            session_name = {0: 'Asian', 1: 'London', 2: 'New York'}.get(session_info['session_id'], 'Unknown')
            
            risk_analysis = {
                'current_session': {
                    'name': session_name,
                    'base_risk_multiplier': float(risk_factors['risk_multiplier']),
                    'activity_level': float(session_info['activity_score'])
                },
                'risk_factors': {
                    'news_risk': {
                        'level': float(risk_factors['news_risk']),
                        'description': self._get_news_risk_description(risk_factors['news_risk'])
                    },
                    'correlation_risk': {
                        'level': float(risk_factors['correlation_risk']),
                        'description': self._get_correlation_risk_description(risk_factors['correlation_risk'])
                    },
                    'gap_risk': {
                        'level': float(risk_factors['gap_risk']),
                        'description': self._get_gap_risk_description(risk_factors['gap_risk'])
                    }
                },
                'recommended_adjustments': {
                    'position_size_multiplier': float(1.0 / risk_factors['risk_multiplier']),
                    'stop_loss_adjustment': float(risk_factors['risk_multiplier']),
                    'max_positions': self._get_recommended_max_positions(session_info, risk_factors),
                    'risk_level': self._get_overall_risk_level(risk_factors)
                },
                'session_recommendations': self._get_session_risk_recommendations(session_name, risk_factors)
            }
            
            return {
                'success': True,
                'risk_analysis': risk_analysis,
                'timestamp': current_timestamp.isoformat(),
                'version': '2.2.0'
            }
            
        except Exception as e:
            self.logger.error(f"Session risk analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    def _handle_session_performance_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """NEW: Handle session performance statistics request"""
        try:
            if ENHANCED_MODULES_AVAILABLE and hasattr(self.ai_engine, 'get_session_performance_stats'):
                ai_stats = self.ai_engine.get_session_performance_stats()
            else:
                ai_stats = {}
            
            # Server session statistics
            uptime = datetime.now() - self.stats.uptime_start
            
            session_performance = {
                'server_stats': {
                    'uptime_hours': float(uptime.total_seconds() / 3600),
                    'total_requests': self.stats.total_requests,
                    'prediction_requests': self.stats.prediction_requests,
                    'session_enhanced_predictions': self.stats.session_enhanced_predictions,
                    'session_enhancement_rate': (
                        self.stats.session_enhanced_predictions / max(1, self.stats.prediction_requests)
                    ),
                    'avg_response_time': self.stats.avg_response_time,
                    'session_response_time': self.stats.session_response_time,
                    'feature_count_avg': np.mean(self.feature_counts) if self.feature_counts else 0,
                    'session_feature_count_avg': np.mean(self.session_feature_counts) if self.session_feature_counts else 0
                },
                'ai_performance': ai_stats,
                'session_analysis': {
                    'session_requests': self.stats.session_analysis_requests,
                    'smc_requests': self.stats.smc_analysis_requests,
                    'feature_requests': self.stats.feature_generation_requests,
                    'training_requests': self.stats.model_training_requests
                },
                'system_health': {
                    'errors': self.stats.errors,
                    'warnings': self.stats.warnings,
                    'error_rate': self.stats.errors / max(1, self.stats.total_requests),
                    'model_trained': self.model_trained,
                    'cache_size': len(self.session_cache.get('session_features_cache', {}))
                }
            }
            
            return {
                'success': True,
                'performance': session_performance,
                'targets': {
                    'accuracy_target': self.stats.accuracy_target,
                    'feature_target': 106,
                    'session_feature_target': 18,
                    'response_time_target': 0.2
                },
                'version': '2.2.0'
            }
            
        except Exception as e:
            self.logger.error(f"Session performance request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    # ========== ENHANCED FEATURE GENERATION HANDLERS ==========
    
    def _handle_enhanced_feature_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle enhanced feature generation with session intelligence"""
        try:
            self.stats.feature_generation_requests += 1
            
            if not ENHANCED_MODULES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Enhanced feature engineering not available',
                    'version': '2.2.0'
                }
            
            # Extract request data
            data = request.get('data', [])
            current_timestamp = request.get('timestamp')
            
            if not data:
                return {
                    'success': False,
                    'error': 'No price data provided',
                    'required_format': 'Array of OHLCV data'
                }
            
            # Convert to DataFrame
            try:
                if isinstance(data[0], dict):
                    ohlcv_df = pd.DataFrame(data)
                else:
                    ohlcv_df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
                
                # Ensure proper data types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    ohlcv_df[col] = pd.to_numeric(ohlcv_df[col], errors='coerce')
                
                # Handle timestamp
                if current_timestamp:
                    try:
                        current_timestamp = pd.to_datetime(current_timestamp)
                        if current_timestamp.tz is None:
                            current_timestamp = current_timestamp.replace(tzinfo=timezone.utc)
                    except:
                        current_timestamp = datetime.now(timezone.utc)
                else:
                    current_timestamp = datetime.now(timezone.utc)
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Data conversion failed: {e}',
                    'version': '2.2.0'
                }
            
            # Generate enhanced features with session intelligence
            start_time = time.time()
            features = self.feature_engineer.create_enhanced_features(
                ohlcv_df, current_timestamp=current_timestamp
            )
            generation_time = time.time() - start_time
            
            # Track feature statistics
            total_features = len(features)
            session_features = len([k for k in features.keys() if k.startswith('session_')])
            smc_features = len([k for k in features.keys() if k.startswith('smc_')])
            technical_features = len([k for k in features.keys() if any(prefix in k for prefix in ['ema_', 'rsi', 'macd_', 'bb_', 'atr'])])
            vp_features = len([k for k in features.keys() if k.startswith('vp_')])
            vwap_features = len([k for k in features.keys() if k.startswith('vwap_')])
            
            self.feature_counts.append(total_features)
            self.session_feature_counts.append(session_features)
            
            # Keep recent statistics
            if len(self.feature_counts) > 100:
                self.feature_counts = self.feature_counts[-100:]
                self.session_feature_counts = self.session_feature_counts[-100:]
            
            # Categorize features for response
            feature_categories = {
                'session': {k: v for k, v in features.items() if k.startswith('session_')},
                'smc': {k: v for k, v in features.items() if k.startswith('smc_')},
                'technical': {k: v for k, v in features.items() if any(prefix in k for prefix in ['ema_', 'rsi', 'macd_', 'bb_', 'atr'])},
                'volume_profile': {k: v for k, v in features.items() if k.startswith('vp_')},
                'vwap': {k: v for k, v in features.items() if k.startswith('vwap_')},
                'advanced': {k: v for k, v in features.items() if k not in set().union(*[cat.keys() for cat in [
                    {k: v for k, v in features.items() if k.startswith('session_')},
                    {k: v for k, v in features.items() if k.startswith('smc_')},
                    {k: v for k, v in features.items() if any(prefix in k for prefix in ['ema_', 'rsi', 'macd_', 'bb_', 'atr'])},
                    {k: v for k, v in features.items() if k.startswith('vp_')},
                    {k: v for k, v in features.items() if k.startswith('vwap_')}
                ]])}
            }
            
            return {
                'success': True,
                'features': {
                    'all_features': features,
                    'feature_categories': feature_categories,
                    'feature_counts': {
                        'total': total_features,
                        'session': session_features,
                        'smc': smc_features,
                        'technical': technical_features,
                        'volume_profile': vp_features,
                        'vwap': vwap_features,
                        'advanced': len(feature_categories['advanced'])
                    },
                    'targets_achieved': {
                        'total_features': total_features >= 106,
                        'session_features': session_features >= 18,
                        'smc_features': smc_features >= 20
                    }
                },
                'generation_time': generation_time,
                'timestamp': current_timestamp.isoformat(),
                'version': '2.2.0'
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced feature generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    # ========== MODEL TRAINING HANDLERS ==========
    
    def _handle_enhanced_training_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle enhanced model training with session intelligence"""
        try:
            self.stats.model_training_requests += 1
            
            if not ENHANCED_MODULES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Enhanced AI engine not available',
                    'version': '2.2.0'
                }
            
            # Extract training data
            data = request.get('data', [])
            training_params = request.get('training_params', {})
            
            if not data:
                return {
                    'success': False,
                    'error': 'No training data provided',
                    'required_format': 'Array of OHLCV data'
                }
            
            # Convert to DataFrame
            try:
                if isinstance(data[0], dict):
                    ohlcv_df = pd.DataFrame(data)
                else:
                    ohlcv_df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
                
                # Ensure proper data types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    ohlcv_df[col] = pd.to_numeric(ohlcv_df[col], errors='coerce')
                
                # Add timestamp index if not present
                if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
                    ohlcv_df.index = pd.date_range(
                        start=datetime.now(timezone.utc), 
                        periods=len(ohlcv_df), 
                        freq='15min'
                    )
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Training data conversion failed: {e}',
                    'version': '2.2.0'
                }
            
            # Train enhanced model with session intelligence
            self.logger.info(f"🚀 Starting enhanced model training with {len(ohlcv_df)} samples...")
            
            start_time = time.time()
            training_results = self.ai_engine.train_session_enhanced_model(
                ohlcv_df,
                validation_split=training_params.get('validation_split', 0.2),
                hyperparameter_optimization=training_params.get('hyperparameter_optimization', True)
            )
            training_time = time.time() - start_time
            
            if training_results.get('ensemble_accuracy', 0) > 0:
                self.model_trained = True
                self.logger.info(f"✅ Enhanced model training completed: {training_results.get('ensemble_accuracy', 0):.4f} accuracy")
            
            return {
                'success': True,
                'training_results': training_results,
                'training_time': training_time,
                'model_trained': self.model_trained,
                'targets_achieved': {
                    'accuracy_target': training_results.get('target_achieved', False),
                    'feature_target': training_results.get('feature_target_achieved', False),
                    'session_target': training_results.get('session_target_achieved', False)
                },
                'version': '2.2.0'
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced model training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time if 'start_time' in locals() else 0,
                'version': '2.2.0'
            }
    
    def _handle_model_save_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle model save request"""
        try:
            if not ENHANCED_MODULES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Enhanced AI engine not available',
                    'version': '2.2.0'
                }
            
            if not self.model_trained:
                return {
                    'success': False,
                    'error': 'No trained model to save',
                    'version': '2.2.0'
                }
            
            filepath = request.get('filepath', f'models/enhanced_model_{self.symbol}_v220.pkl')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            success = self.ai_engine.save_session_enhanced_model(filepath)
            
            return {
                'success': success,
                'filepath': filepath,
                'message': 'Model saved successfully' if success else 'Model save failed',
                'version': '2.2.0'
            }
            
        except Exception as e:
            self.logger.error(f"Model save failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    def _handle_model_load_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle model load request"""
        try:
            if not ENHANCED_MODULES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Enhanced AI engine not available',
                    'version': '2.2.0'
                }
            
            filepath = request.get('filepath', f'models/enhanced_model_{self.symbol}_v220.pkl')
            
            if not os.path.exists(filepath):
                return {
                    'success': False,
                    'error': f'Model file not found: {filepath}',
                    'version': '2.2.0'
                }
            
            success = self.ai_engine.load_session_enhanced_model(filepath)
            
            if success:
                self.model_trained = True
                self.logger.info(f"✅ Enhanced model loaded from {filepath}")
            
            return {
                'success': success,
                'filepath': filepath,
                'model_trained': self.model_trained,
                'message': 'Model loaded successfully' if success else 'Model load failed',
                'version': '2.2.0'
            }
            
        except Exception as e:
            self.logger.error(f"Model load failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    def _handle_model_info_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle model information request"""
        try:
            if not ENHANCED_MODULES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Enhanced AI engine not available',
                    'version': '2.2.0'
                }
            
            model_info = {
                'model_trained': self.model_trained,
                'ai_engine_version': '2.2.0',
                'feature_engineer_version': '2.2.0',
                'session_intelligence': True,
                'target_features': 106,
                'session_features_target': 18,
                'accuracy_target': 0.80
            }
            
            if self.model_trained and hasattr(self.ai_engine, 'get_session_performance_stats'):
                model_info.update(self.ai_engine.get_session_performance_stats())
            
            return {
                'success': True,
                'model_info': model_info,
                'version': '2.2.0'
            }
            
        except Exception as e:
            self.logger.error(f"Model info request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    # ========== SMC ANALYSIS HANDLERS (from v2.1.0) ==========
    
    def _handle_smc_analysis_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle SMC analysis request (from v2.1.0)"""
        try:
            self.stats.smc_analysis_requests += 1
            
            if not ENHANCED_MODULES_AVAILABLE:
                return {
                    'success': False,
                    'error': 'SMC analysis not available',
                    'version': '2.2.0'
                }
            
            symbol = request.get('symbol', self.symbol)
            data = request.get('data', [])
            
            if not data:
                return {
                    'success': False,
                    'error': 'No price data provided for SMC analysis',
                    'version': '2.2.0'
                }
            
            # Convert data and perform SMC analysis
            try:
                if isinstance(data[0], dict):
                    ohlcv_df = pd.DataFrame(data)
                else:
                    ohlcv_df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
                
                if hasattr(self.feature_engineer, 'smc_engine') and self.feature_engineer.smc_engine:
                    smc_context = self.feature_engineer.smc_engine.analyze_smc_context(ohlcv_df)
                    
                    return {
                        'success': True,
                        'symbol': symbol,
                        'smc_analysis': smc_context,
                        'feature_count': len(smc_context.get('smc_features', {})),
                        'version': '2.2.0'
                    }
                else:
                    return {
                        'success': False,
                        'error': 'SMC engine not available',
                        'version': '2.2.0'
                    }
                    
            except Exception as e:
                return {
                    'success': False,
                    'error': f'SMC analysis failed: {e}',
                    'version': '2.2.0'
                }
                
        except Exception as e:
            self.logger.error(f"SMC analysis request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    def _handle_order_blocks_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle order blocks request"""
        try:
            # Simplified order blocks response for v2.2.0
            return {
                'success': True,
                'order_blocks': {
                    'bullish_blocks': [],
                    'bearish_blocks': [],
                    'active_blocks': 0,
                    'message': 'Order blocks analysis available in full SMC analysis'
                },
                'version': '2.2.0'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    def _handle_fvg_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle Fair Value Gaps request"""
        try:
            # Simplified FVG response for v2.2.0
            return {
                'success': True,
                'fair_value_gaps': {
                    'bullish_fvgs': [],
                    'bearish_fvgs': [],
                    'active_fvgs': 0,
                    'message': 'FVG analysis available in full SMC analysis'
                },
                'version': '2.2.0'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    def _handle_market_structure_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle market structure request"""
        try:
            # Simplified market structure response for v2.2.0
            return {
                'success': True,
                'market_structure': {
                    'trend': 'Unknown',
                    'structure_breaks': [],
                    'message': 'Market structure analysis available in full SMC analysis'
                },
                'version': '2.2.0'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    # ========== SYSTEM HANDLERS ==========
    
    def _handle_server_stats_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle server statistics request"""
        try:
            uptime = datetime.now() - self.stats.uptime_start
            
            return {
                'success': True,
                'server_stats': asdict(self.stats),
                'uptime': {
                    'total_seconds': uptime.total_seconds(),
                    'hours': uptime.total_seconds() / 3600,
                    'days': uptime.days
                },
                'performance': {
                    'avg_response_time': self.stats.avg_response_time,
                    'session_response_time': self.stats.session_response_time,
                    'feature_count_avg': np.mean(self.feature_counts) if self.feature_counts else 0,
                    'session_feature_count_avg': np.mean(self.session_feature_counts) if self.session_feature_counts else 0,
                    'error_rate': self.stats.errors / max(1, self.stats.total_requests)
                },
                'version': '2.2.0'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    def _handle_health_check_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle health check request"""
        try:
            health_status = {
                'server_running': self.running,
                'enhanced_modules': ENHANCED_MODULES_AVAILABLE,
                'model_trained': self.model_trained,
                'active_clients': len(self.clients),
                'memory_usage': 'unknown',  # Could add psutil for memory info
                'version': '2.2.0',
                'status': 'healthy'
            }
            
            # Determine overall health
            if not ENHANCED_MODULES_AVAILABLE:
                health_status['status'] = 'degraded'
                health_status['warning'] = 'Enhanced modules not available'
            elif self.stats.errors > self.stats.total_requests * 0.1:
                health_status['status'] = 'degraded'
                health_status['warning'] = 'High error rate detected'
            
            return {
                'success': True,
                'health': health_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'health': 'unhealthy'
            }
    
    def _handle_capabilities_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle capabilities request"""
        try:
            capabilities = {
                'version': '2.2.0',
                'enhanced_features': ENHANCED_MODULES_AVAILABLE,
                'session_intelligence': ENHANCED_MODULES_AVAILABLE,
                'feature_count_target': 106,
                'session_features_target': 18,
                'accuracy_target': 0.80,
                'available_commands': list(self.command_handlers.keys()),
                'new_features_v220': [
                    'session_analysis',
                    'session_optimal_windows', 
                    'session_risk_analysis',
                    'session_performance',
                    'enhanced_predictions',
                    '106+ features support',
                    '18+ session features'
                ],
                'maintained_features': [
                    'smc_analysis',
                    'enhanced_features',
                    'model_training',
                    'volume_profile',
                    'vwap_analysis'
                ]
            }
            
            return {
                'success': True,
                'capabilities': capabilities
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    def _handle_ping_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle ping request"""
        return {
            'success': True,
            'pong': True,
            'timestamp': datetime.now().isoformat(),
            'version': '2.2.0',
            'client_id': client_id
        }
    
    def _handle_data_update_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle data update request"""
        try:
            symbol = request.get('symbol', self.symbol)
            data = request.get('data', [])
            
            if not data:
                return {
                    'success': False,
                    'error': 'No data provided',
                    'version': '2.2.0'
                }
            
            # Store data for future use
            self.price_data[symbol] = data
            
            return {
                'success': True,
                'symbol': symbol,
                'data_points': len(data),
                'message': f'Data updated for {symbol}',
                'version': '2.2.0'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    def _handle_data_info_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle data info request"""
        try:
            data_info = {}
            
            for symbol, data in self.price_data.items():
                if data:
                    data_info[symbol] = {
                        'data_points': len(data),
                        'latest_price': data[-1].get('close', 'unknown') if isinstance(data[-1], dict) else 'unknown',
                        'data_type': type(data[0]).__name__ if data else 'unknown'
                    }
            
            return {
                'success': True,
                'stored_data': data_info,
                'total_symbols': len(self.price_data),
                'version': '2.2.0'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    # ========== FALLBACK HANDLERS ==========
    
    def _handle_fallback_prediction(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback prediction handler when enhanced modules not available"""
        try:
            import random
            
            # Generate mock prediction
            signal = random.choice([-1, 0, 1])
            confidence = random.uniform(0.5, 0.8)
            
            return {
                'success': True,
                'prediction': {
                    'signal': signal,
                    'confidence': confidence,
                    'signal_text': self._get_signal_text(signal),
                    'confidence_level': self._get_confidence_level(confidence)
                },
                'session_analysis': {
                    'session_name': 'London',
                    'activity_score': 0.8,
                    'optimal_window': False,
                    'session_enhanced': False
                },
                'component_analysis': {
                    'technical_confidence': 0.6,
                    'volume_confidence': 0.5,
                    'smc_confidence': 0.5,
                    'session_confidence': 0.4
                },
                'warning': 'Using fallback prediction - enhanced modules not available',
                'version': '2.2.0'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'version': '2.2.0'
            }
    
    # ========== UTILITY METHODS ==========
    
    def _get_signal_text(self, signal: int) -> str:
        """Convert signal to text"""
        signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        return signal_map.get(signal, 'UNKNOWN')
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence to level"""
        if confidence >= 0.8:
            return 'HIGH'
        elif confidence >= 0.65:
            return 'MEDIUM'
        elif confidence >= 0.5:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def _get_session_recommendations(self, prediction) -> Dict[str, Any]:
        """Get session-specific recommendations"""
        try:
            session_context = prediction.session_context
            recommendations = {
                'trading_recommendations': [],
                'risk_adjustments': [],
                'timing_suggestions': []
            }
            
            # Trading recommendations based on session
            if session_context.get('optimal_window', False):
                recommendations['trading_recommendations'].append('Optimal trading window - consider increased position size')
            
            if session_context.get('session_enhanced', False):
                recommendations['trading_recommendations'].append('Session-enhanced signal - higher confidence')
            
            if session_context.get('liquidity_level', 0.8) > 0.9:
                recommendations['trading_recommendations'].append('High liquidity - good for larger positions')
            
            # Risk adjustments
            risk_multiplier = session_context.get('risk_multiplier', 1.0)
            if risk_multiplier > 1.1:
                recommendations['risk_adjustments'].append(f'Increase stop loss by {((risk_multiplier - 1) * 100):.1f}%')
            elif risk_multiplier < 0.9:
                recommendations['risk_adjustments'].append(f'Decrease stop loss by {((1 - risk_multiplier) * 100):.1f}%')
            
            # Timing suggestions
            session_name = session_context.get('session_name', 'Unknown')
            if session_name == 'Asian':
                recommendations['timing_suggestions'].append('Asian session - consider range trading strategies')
            elif session_name == 'London':
                recommendations['timing_suggestions'].append('London session - good for trend following')
            elif session_name == 'New York':
                recommendations['timing_suggestions'].append('NY session - watch for momentum trades')
            
            return recommendations
            
        except Exception as e:
            return {
                'trading_recommendations': ['Session analysis unavailable'],
                'risk_adjustments': ['Use standard risk management'],
                'timing_suggestions': ['Monitor market conditions']
            }
    
    def _get_news_risk_description(self, news_risk: float) -> str:
        """Get news risk description"""
        if news_risk > 0.8:
            return 'High news risk - major events likely'
        elif news_risk > 0.6:
            return 'Moderate news risk - some events possible'
        elif news_risk > 0.4:
            return 'Low news risk - minimal events expected'
        else:
            return 'Very low news risk - quiet period'
    
    def _get_correlation_risk_description(self, correlation_risk: float) -> str:
        """Get correlation risk description"""
        if correlation_risk > 0.8:
            return 'High correlation - pairs moving together'
        elif correlation_risk > 0.6:
            return 'Moderate correlation - some pair alignment'
        elif correlation_risk > 0.4:
            return 'Low correlation - pairs mostly independent'
        else:
            return 'Very low correlation - pairs diverging'
    
    def _get_gap_risk_description(self, gap_risk: float) -> str:
        """Get gap risk description"""
        if gap_risk > 0.6:
            return 'High gap risk - session opens may have gaps'
        elif gap_risk > 0.4:
            return 'Moderate gap risk - small gaps possible'
        elif gap_risk > 0.2:
            return 'Low gap risk - minimal gaps expected'
        else:
            return 'Very low gap risk - smooth session transitions'
    
    def _get_recommended_max_positions(self, session_info: Dict[str, float], 
                                     risk_factors: Dict[str, float]) -> int:
        """Get recommended maximum positions for current session"""
        base_positions = 4
        activity_score = session_info.get('activity_score', 0.8)
        risk_multiplier = risk_factors.get('risk_multiplier', 1.0)
        
        # Adjust based on session activity
        if activity_score > 0.9:
            adjusted_positions = base_positions + 1  # High activity = more opportunities
        elif activity_score < 0.6:
            adjusted_positions = base_positions - 1  # Low activity = fewer positions
        else:
            adjusted_positions = base_positions
        
        # Adjust based on risk
        if risk_multiplier > 1.2:
            adjusted_positions = max(1, adjusted_positions - 1)  # High risk = fewer positions
        
        return max(1, min(6, adjusted_positions))
    
    def _get_overall_risk_level(self, risk_factors: Dict[str, float]) -> str:
        """Get overall risk level assessment"""
        risk_score = (
            risk_factors.get('risk_multiplier', 1.0) * 0.4 +
            risk_factors.get('news_risk', 0.5) * 0.3 +
            risk_factors.get('correlation_risk', 0.5) * 0.2 +
            risk_factors.get('gap_risk', 0.3) * 0.1
        )
        
        if risk_score > 1.2:
            return 'HIGH'
        elif risk_score > 1.0:
            return 'ELEVATED'
        elif risk_score > 0.8:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _get_session_risk_recommendations(self, session_name: str, 
                                        risk_factors: Dict[str, float]) -> List[str]:
        """Get session-specific risk recommendations"""
        recommendations = []
        
        risk_multiplier = risk_factors.get('risk_multiplier', 1.0)
        news_risk = risk_factors.get('news_risk', 0.5)
        
        # Session-specific recommendations
        if session_name == 'Asian':
            recommendations.append('Asian session: Use smaller position sizes')
            if risk_multiplier > 1.1:
                recommendations.append('Increased volatility expected - use wider stops')
        elif session_name == 'London':
            recommendations.append('London session: Prime trading time')
            if news_risk > 0.7:
                recommendations.append('High news risk - monitor economic calendar')
        elif session_name == 'New York':
            recommendations.append('NY session: Watch for momentum moves')
            if risk_multiplier > 1.2:
                recommendations.append('High volatility period - adjust position sizes')
        
        # General risk recommendations
        if news_risk > 0.8:
            recommendations.append('Avoid trading 30 minutes before/after major news')
        
        if risk_factors.get('correlation_risk', 0.5) > 0.8:
            recommendations.append('High correlation - avoid multiple positions in correlated pairs')
        
        return recommendations


def main():
    """Main function for running the Enhanced Socket Server v2.2.0"""
    parser = argparse.ArgumentParser(description='Enhanced Socket Server v2.2.0 - Session-Aware AI')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8888, help='Server port')
    parser.add_argument('--symbol', default='EURUSD', help='Trading symbol')
    parser.add_argument('--timeframe', default='M15', help='Chart timeframe')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - [v2.2.0] %(message)s'
    )
    
    # Create and start server
    server = EnhancedSocketServer(
        host=args.host,
        port=args.port,
        symbol=args.symbol,
        timeframe=args.timeframe
    )
    
    print(f"🚀 Starting Enhanced Socket Server v2.2.0...")
    print(f"   📊 Features: 106+ with session intelligence")
    print(f"   🎯 Target Accuracy: 80%+")
    print(f"   🌍 Session Analysis: Enabled")
    print(f"   📡 Address: {args.host}:{args.port}")
    print(f"   💱 Symbol: {args.symbol} ({args.timeframe})")
    print(f"   🔧 Enhanced Modules: {'Available' if ENHANCED_MODULES_AVAILABLE else 'Fallback Mode'}")
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Enhanced Socket Server v2.2.0...")
        server.stop()
        print("✅ Server stopped successfully")


if __name__ == "__main__":
    main()