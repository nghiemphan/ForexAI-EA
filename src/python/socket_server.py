# socket_server.py
"""
Updated Socket Server for ForexAI-EA Project with Real AI Engine
Handles communication between Python AI Engine and MQL5 EA
Author: Claude AI Developer
Version: 2.0.0
Updated: 2025-06-11
"""

import socket
import threading
import json
import logging
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import signal
import sys
import numpy as np

# Import our AI components
from ai_engine import AITradingEngine, ModelEvaluator
from technical_indicators import TechnicalIndicators
from feature_engineer import FeatureEngineer

class AISocketServer:
    """
    Enhanced socket server with real AI engine integration
    Processes trading requests and returns AI predictions
    """
    
    def __init__(self, host: str = "localhost", port: int = 8888):
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.is_running = False
        self.client_connections = []
        
        # AI Engine components
        self.ai_engine: Optional[AITradingEngine] = None
        self.indicators_engine = TechnicalIndicators()
        self.feature_engineer = FeatureEngineer()
        
        # Performance tracking
        self.prediction_count = 0
        self.prediction_times = []
        self.last_predictions = []
        
        # Setup logging
        self.setup_logging()
        
        # Initialize AI engine
        self.initialize_ai_engine()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info(f"AISocketServer initialized on {host}:{port}")
    
    def setup_logging(self):
        """Configure logging system"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Full path to log file
        log_file = os.path.join(log_dir, 'ai_server.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_ai_engine(self):
        """Initialize the AI engine with model loading"""
        try:
            self.logger.info("Initializing AI engine...")
            
            # Create AI engine (without config path for now)
            self.ai_engine = AITradingEngine()
            
            # Check if model exists and is loaded
            model_info = self.ai_engine.get_model_info()
            
            if model_info['status'] == 'Model loaded':
                self.logger.info(f"AI model loaded successfully:")
                self.logger.info(f"  Model type: {model_info.get('model_type', 'Unknown')}")
                self.logger.info(f"  Features: {model_info.get('feature_count', 0)}")
                self.logger.info(f"  Accuracy: {model_info.get('performance', {}).get('test_accuracy', 'N/A')}")
            else:
                self.logger.warning("No trained model found. Will use training data to create model...")
                self._train_initial_model()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize AI engine: {e}")
            self.ai_engine = None
    
    def _train_initial_model(self):
        """Train initial model with sample data if no model exists"""
        try:
            self.logger.info("Training initial AI model...")
            
            # Generate sample training data for demonstration
            sample_data = self._generate_sample_training_data()
            
            # Train the model
            performance = self.ai_engine.train_model([sample_data])
            
            self.logger.info(f"Initial model trained with accuracy: {performance['test_accuracy']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to train initial model: {e}")
    
    def _generate_sample_training_data(self, n_bars: int = 2000) -> Dict:
        """Generate sample OHLC data for initial training"""
        np.random.seed(42)
        
        # Generate realistic forex price data
        base_price = 1.1000
        prices = [base_price]
        
        # Add trend and noise
        for i in range(n_bars):
            # Add trend component
            trend = 0.0001 * np.sin(i / 100)
            # Add random walk
            noise = np.random.normal(0, 0.0005)
            # Add mean reversion
            mean_reversion = -0.1 * (prices[-1] - base_price)
            
            price_change = trend + noise + mean_reversion
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
        
        # Generate OHLC from prices
        ohlc_data = {
            'open': [],
            'high': [],
            'low': [],
            'close': prices[1:]  # Remove first price
        }
        
        for i in range(len(ohlc_data['close'])):
            close = ohlc_data['close'][i]
            open_price = prices[i] if i < len(prices) else close
            
            # Generate high/low with some spread
            spread = abs(np.random.normal(0, 0.0003))
            high = max(open_price, close) + spread
            low = min(open_price, close) - spread
            
            ohlc_data['open'].append(open_price)
            ohlc_data['high'].append(high)
            ohlc_data['low'].append(low)
        
        return ohlc_data
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop_server()
        sys.exit(0)
    
    def start_server(self):
        """Start the socket server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.is_running = True
            self.logger.info(f"AI Socket Server started on {self.host}:{self.port}")
            
            if self.ai_engine is None:
                self.logger.warning("AI engine not available - server running in fallback mode")
            
            # Accept connections loop
            while self.is_running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    self.logger.info(f"New connection from {client_address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.is_running:
                        self.logger.error(f"Socket error: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
        finally:
            self.stop_server()
    
    def handle_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle individual client connection"""
        self.client_connections.append(client_socket)
        
        try:
            while self.is_running:
                # Set timeout for socket operations
                client_socket.settimeout(30.0)
                
                # Receive data from client
                data = client_socket.recv(4096)
                if not data:
                    break
                
                # Process request
                response = self.process_request(data.decode('utf-8'))
                
                # Send response back to client
                client_socket.send(response.encode('utf-8'))
                
        except socket.timeout:
            self.logger.warning(f"Client {client_address} timed out")
        except socket.error as e:
            self.logger.error(f"Client {client_address} error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error with client {client_address}: {e}")
        finally:
            try:
                client_socket.close()
                if client_socket in self.client_connections:
                    self.client_connections.remove(client_socket)
                self.logger.info(f"Client {client_address} disconnected")
            except:
                pass
    
    def process_request(self, request_data: str) -> str:
        """
        Process incoming request from MQL5 EA
        Expected format: JSON string with request type and data
        """
        try:
            # Parse JSON request
            request = json.loads(request_data)
            request_type = request.get('type', 'unknown')
            
            self.logger.info(f"Processing request: {request_type}")
            
            # Route request to appropriate handler
            if request_type == 'ping':
                response = self.handle_ping(request)
            elif request_type == 'prediction':
                response = self.handle_prediction_request(request)
            elif request_type == 'health_check':
                response = self.handle_health_check(request)
            elif request_type == 'model_info':
                response = self.handle_model_info_request(request)
            elif request_type == 'performance':
                response = self.handle_performance_request(request)
            else:
                response = {
                    'status': 'error',
                    'message': f'Unknown request type: {request_type}',
                    'timestamp': datetime.now().isoformat()
                }
            
            return json.dumps(response)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON received: {e}")
            return json.dumps({
                'status': 'error',
                'message': 'Invalid JSON format',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return json.dumps({
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    def handle_ping(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping request for connection testing"""
        return {
            'status': 'success',
            'message': 'pong',
            'server_time': datetime.now().isoformat(),
            'ai_engine_status': 'available' if self.ai_engine else 'unavailable',
            'timestamp': datetime.now().isoformat()
        }
    
    def handle_prediction_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle AI prediction request with real AI engine
        """
        try:
            start_time = time.time()
            
            # Extract request data
            symbol = request.get('symbol', 'UNKNOWN')
            timeframe = request.get('timeframe', 'M15')
            prices = request.get('prices', [])
            
            # Validate input
            if not prices or len(prices) < 50:
                return {
                    'status': 'error',
                    'message': f'Insufficient price data for prediction (need 50+, got {len(prices)})',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Prepare OHLC data (assuming prices are close prices)
            ohlc_data = self._prepare_ohlc_data(prices)
            
            # Use real AI engine if available
            if self.ai_engine is not None:
                try:
                    signal, confidence, metadata = self.ai_engine.predict(ohlc_data)
                    
                    # Track performance
                    prediction_time = time.time() - start_time
                    self.prediction_times.append(prediction_time)
                    self.prediction_count += 1
                    
                    # Store last prediction for monitoring
                    self.last_predictions.append({
                        'symbol': symbol,
                        'signal': signal,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Keep only last 100 predictions
                    if len(self.last_predictions) > 100:
                        self.last_predictions = self.last_predictions[-100:]
                    
                    response = {
                        'status': 'success',
                        'prediction': signal,
                        'confidence': confidence,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'prediction_time_ms': prediction_time * 1000,
                        'metadata': metadata,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.logger.info(f"AI Prediction: {signal} (Confidence: {confidence:.3f}) for {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"AI prediction failed: {e}")
                    # Fallback to safe prediction
                    response = {
                        'status': 'success',
                        'prediction': 0,  # Hold signal as fallback
                        'confidence': 0.5,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'error': str(e),
                        'fallback_mode': True,
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                # Fallback mode without AI engine
                response = {
                    'status': 'success',
                    'prediction': 0,  # Conservative hold signal
                    'confidence': 0.5,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'fallback_mode': True,
                    'message': 'AI engine unavailable - using fallback',
                    'timestamp': datetime.now().isoformat()
                }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Prediction request error: {e}")
            return {
                'status': 'error',
                'message': f'Prediction error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_ohlc_data(self, prices: List[float]) -> Dict[str, List[float]]:
        """
        Prepare OHLC data from price list
        For real implementation, this should receive actual OHLC data
        """
        # For now, simulate OHLC from close prices
        # In production, the EA should send actual OHLC data
        
        ohlc_data = {
            'open': [],
            'high': [],
            'low': [],
            'close': prices
        }
        
        for i, close in enumerate(prices):
            if i == 0:
                open_price = close
            else:
                open_price = prices[i-1]
            
            # Simulate high/low based on close prices
            # Add small random variation to simulate realistic OHLC
            price_range = abs(close - open_price) * 0.3
            high = max(open_price, close) + price_range * 0.5
            low = min(open_price, close) - price_range * 0.5
            
            ohlc_data['open'].append(open_price)
            ohlc_data['high'].append(high)
            ohlc_data['low'].append(low)
        
        return ohlc_data
    
    def handle_health_check(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comprehensive health check request"""
        # Calculate average prediction time
        avg_prediction_time = 0
        if self.prediction_times:
            recent_times = self.prediction_times[-50:]  # Last 50 predictions
            avg_prediction_time = np.mean(recent_times) * 1000  # Convert to ms
        
        # Get AI engine status
        ai_status = 'unavailable'
        model_info = {}
        if self.ai_engine:
            ai_status = 'available'
            model_info = self.ai_engine.get_model_info()
        
        return {
            'status': 'success',
            'server_status': 'healthy',
            'ai_engine_status': ai_status,
            'uptime_seconds': time.time(),
            'active_connections': len(self.client_connections),
            'total_predictions': self.prediction_count,
            'avg_prediction_time_ms': avg_prediction_time,
            'model_info': model_info,
            'timestamp': datetime.now().isoformat()
        }
    
    def handle_model_info_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model information request"""
        if self.ai_engine is None:
            return {
                'status': 'error',
                'message': 'AI engine not available',
                'timestamp': datetime.now().isoformat()
            }
        
        model_info = self.ai_engine.get_model_info()
        
        # Add feature importance if available
        feature_importance = self.ai_engine.get_feature_importance(15)
        
        return {
            'status': 'success',
            'model_info': model_info,
            'feature_importance': feature_importance,
            'timestamp': datetime.now().isoformat()
        }
    
    def handle_performance_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance statistics request"""
        # Calculate confidence statistics
        confidence_stats = {}
        if self.last_predictions:
            confidences = [p['confidence'] for p in self.last_predictions]
            confidence_stats = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        
        # Signal distribution
        signal_distribution = {-1: 0, 0: 0, 1: 0}
        if self.last_predictions:
            for pred in self.last_predictions:
                signal_distribution[pred['signal']] += 1
        
        return {
            'status': 'success',
            'prediction_count': self.prediction_count,
            'confidence_stats': confidence_stats,
            'signal_distribution': signal_distribution,
            'recent_predictions': self.last_predictions[-10:],  # Last 10 predictions
            'timestamp': datetime.now().isoformat()
        }
    
    def stop_server(self):
        """Stop the server gracefully"""
        self.is_running = False
        
        # Close all client connections
        for client_socket in self.client_connections[:]:
            try:
                client_socket.close()
            except:
                pass
        self.client_connections.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        self.logger.info("AI Socket Server stopped")


def main():
    """Main function to start the server"""
    print("=" * 60)
    print("ForexAI-EA Socket Server v2.0.0 (Real AI Engine)")
    print("=" * 60)
    
    # Create and start server
    server = AISocketServer(host="localhost", port=8888)
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server.stop_server()
        print("Server shutdown complete")


if __name__ == "__main__":
    main()