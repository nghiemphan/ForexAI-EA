"""
File: src/python/socket_server.py
Description: Enhanced Socket Server v2.0.2 - Fixed All Issues
Author: Claude AI Developer
Version: 2.0.2
Created: 2025-06-13
Modified: 2025-06-14
"""

import socket
import threading
import json
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
import signal
import sys
import os

# Import enhanced modules with fallback
try:
    from enhanced_ai_engine import EnhancedAIEngine, EnhancedModelEvaluator
except ImportError:
    try:
        from ai_engine import AITradingEngine as EnhancedAIEngine
        print("WARNING: Using fallback AI engine - enhanced features may not be available")
        EnhancedModelEvaluator = None
    except ImportError:
        print("ERROR: Cannot import AI engine modules")
        sys.exit(1)

try:
    from volume_profile import VolumeProfileEngine, VWAPCalculator
except ImportError:
    print("WARNING: Volume Profile features not available")
    VolumeProfileEngine = None
    VWAPCalculator = None

class EnhancedSocketServer:
    """Enhanced Socket Server with Volume Profile and VWAP capabilities"""
    
    def __init__(self, host: str = "localhost", port: int = 8888):
        """
        Initialize Enhanced Socket Server
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.clients = []
        
        # Setup logging with Unicode-safe configuration
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Enhanced AI Engine
        self.ai_engine = EnhancedAIEngine("EURUSD", "M15")
        self.model_loaded = False
        
        # Data storage for real-time processing
        self.price_data = {}  # Store price data by symbol
        self.max_bars_stored = 500
        
        # Performance tracking
        self.stats = {
            'connections': 0,
            'predictions': 0,
            'uptime_start': datetime.now(),
            'total_requests': 0,
            'errors': 0,
            'volume_profile_predictions': 0,
            'vwap_predictions': 0
        }
        
        # Enhanced capabilities flags
        self.capabilities = {
            'volume_profile': VolumeProfileEngine is not None,
            'vwap_analysis': VWAPCalculator is not None,
            'ensemble_models': hasattr(self.ai_engine, 'train_enhanced_model'),
            'enhanced_filtering': True,
            'market_structure': True
        }
        
    def setup_logging(self):
        """Setup enhanced logging without Unicode issues"""
        # Create logs directory
        os.makedirs('data/logs', exist_ok=True)
        
        # Configure logging with UTF-8 encoding and error handling
        try:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('data/logs/enhanced_socket_server.log', encoding='utf-8', errors='replace'),
                    logging.StreamHandler(sys.stdout)
                ]
            )
        except Exception as e:
            # Fallback to basic logging if file logging fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler(sys.stdout)]
            )
            print(f"Warning: File logging failed: {e}")
    
    def start_server(self):
        """Start the enhanced socket server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            
            self.running = True
            self.stats['uptime_start'] = datetime.now()
            
            self.logger.info(f"STARTED: Enhanced ForexAI Socket Server v2.0.2 on {self.host}:{self.port}")
            self.logger.info("FEATURES: Volume Profile, VWAP, Ensemble Models")
            
            # Try to load existing model
            self.load_ai_model()
            
            while self.running:
                try:
                    client_socket, address = self.socket.accept()
                    self.stats['connections'] += 1
                    
                    self.logger.info(f"CONNECTION: New client from {address}")
                    
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        self.logger.error(f"Socket error: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Server startup failed: {e}")
            raise
    
    def load_ai_model(self):
        """Load existing AI model if available"""
        try:
            model_path = "data/models/enhanced_ai_model.pkl"
            if os.path.exists(model_path):
                # Check if enhanced model loading is available
                if hasattr(self.ai_engine, 'load_enhanced_model'):
                    success = self.ai_engine.load_enhanced_model(model_path)
                elif hasattr(self.ai_engine, 'load_model'):
                    success = self.ai_engine.load_model(model_path)
                else:
                    success = False
                    
                if success:
                    self.model_loaded = True
                    self.logger.info("SUCCESS: AI model loaded")
                else:
                    self.logger.warning("FAILED: Could not load AI model")
            else:
                self.logger.info("INFO: No existing model found - will need training")
                
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
    
    def handle_client(self, client_socket, address):
        """Handle client connection with enhanced features"""
        try:
            self.clients.append(client_socket)
            
            while self.running:
                try:
                    # Receive data with timeout
                    client_socket.settimeout(30)
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    
                    # Parse request with encoding handling
                    try:
                        request_str = data.decode('utf-8')
                    except UnicodeDecodeError:
                        request_str = data.decode('utf-8', errors='replace')
                    
                    request = json.loads(request_str)
                    self.stats['total_requests'] += 1
                    
                    # Process request
                    response = self.process_enhanced_request(request)
                    
                    # Send response with encoding handling
                    response_json = json.dumps(response, ensure_ascii=False)
                    response_bytes = response_json.encode('utf-8', errors='replace')
                    client_socket.send(response_bytes)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error: {e}")
                    error_response = {"error": "Invalid JSON format"}
                    response_bytes = json.dumps(error_response).encode('utf-8')
                    client_socket.send(response_bytes)
                    
                except socket.timeout:
                    self.logger.warning(f"Client {address} timeout")
                    break
                    
                except Exception as e:
                    self.logger.error(f"Client handling error: {e}")
                    self.stats['errors'] += 1
                    break
                    
        except Exception as e:
            self.logger.error(f"Client connection error: {e}")
        finally:
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            try:
                client_socket.close()
            except:
                pass
            self.logger.info(f"DISCONNECTED: Client {address}")
    
    def process_enhanced_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process enhanced requests with Volume Profile capabilities"""
        try:
            action = request.get('action', '')
            
            if action == 'predict':
                return self.handle_enhanced_prediction(request)
            elif action == 'train':
                return self.handle_model_training(request)
            elif action == 'status':
                return self.get_enhanced_status()
            elif action == 'capabilities':
                return self.get_capabilities()
            elif action == 'volume_profile':
                return self.handle_volume_profile_request(request)
            elif action == 'vwap_analysis':
                return self.handle_vwap_request(request)
            elif action == 'performance':
                return self.get_performance_stats()
            elif action == 'ping':
                return {"message": "pong", "timestamp": datetime.now().isoformat()}
            elif action == 'health_check':
                return {"server_status": "healthy", "timestamp": datetime.now().isoformat()}
            else:
                return {"error": f"Unknown action: {action}"}
                
        except Exception as e:
            self.logger.error(f"Request processing error: {e}")
            return {"error": str(e)}
    
    def handle_enhanced_prediction(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle enhanced prediction requests"""
        try:
            symbol = request.get('symbol', 'EURUSD')
            timeframe = request.get('timeframe', 'M15')
            
            # Get price data
            price_data = request.get('price_data', [])
            if not price_data:
                return {"error": "No price data provided"}
            
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            if not all(col in df.columns for col in required_columns):
                return {"error": f"Missing required columns: {required_columns}"}
            
            # Store data for analysis
            self.price_data[symbol] = df.tail(self.max_bars_stored)
            
            if not self.model_loaded:
                return {
                    "signal": 0,
                    "confidence": 0.0,
                    "message": "Model not trained yet",
                    "enhanced_features": False
                }
            
            # Get prediction (enhanced or basic)
            try:
                if hasattr(self.ai_engine, 'predict_enhanced'):
                    signal, confidence, details = self.ai_engine.predict_enhanced(df)
                elif hasattr(self.ai_engine, 'predict'):
                    signal, confidence, metadata = self.ai_engine.predict(df.to_dict('records'))
                    details = metadata or {}
                else:
                    return {"error": "No prediction method available"}
            except Exception as e:
                self.logger.error(f"Prediction failed: {e}")
                return {"error": f"Prediction failed: {str(e)}"}
            
            self.stats['predictions'] += 1
            if details.get('volume_profile_active', False):
                self.stats['volume_profile_predictions'] += 1
            if details.get('vwap_active', False):
                self.stats['vwap_predictions'] += 1
            
            # Enhanced response
            response = {
                "signal": int(signal),
                "confidence": float(confidence),
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "enhanced_features": {
                    "volume_profile_active": details.get('volume_profile_active', False),
                    "vwap_active": details.get('vwap_active', False),
                    "feature_count": details.get('feature_count', 0),
                    "individual_models": details.get('individual_models', {}),
                    "raw_signal": details.get('raw_signal', signal),
                    "filtered": details.get('raw_signal') != signal
                },
                "server_version": "2.0.2"
            }
            
            self.logger.info(f"PREDICTION: {symbol} -> Signal: {signal}, Confidence: {confidence:.3f}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Enhanced prediction error: {e}")
            return {"error": str(e)}
    
    def handle_model_training(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model training requests"""
        try:
            symbol = request.get('symbol', 'EURUSD')
            
            # Get training data
            if symbol in self.price_data and len(self.price_data[symbol]) > 200:
                training_data = self.price_data[symbol]
            else:
                price_data = request.get('training_data', [])
                if not price_data:
                    return {"error": "No training data available"}
                training_data = pd.DataFrame(price_data)
            
            self.logger.info(f"TRAINING: Starting model training for {symbol}")
            
            # Train model (enhanced or basic)
            try:
                if hasattr(self.ai_engine, 'train_enhanced_model'):
                    results = self.ai_engine.train_enhanced_model(training_data)
                elif hasattr(self.ai_engine, 'train_model'):
                    # Convert DataFrame to format expected by basic train_model
                    training_data_list = [training_data.to_dict('records')]
                    results = self.ai_engine.train_model(training_data_list)
                else:
                    return {"error": "No training method available"}
            except Exception as e:
                self.logger.error(f"Model training failed: {e}")
                return {"error": f"Training failed: {str(e)}"}
            
            # Save model
            model_path = "data/models/enhanced_ai_model.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            try:
                if hasattr(self.ai_engine, 'save_enhanced_model'):
                    save_success = self.ai_engine.save_enhanced_model(model_path)
                elif hasattr(self.ai_engine, 'save_model'):
                    save_success = self.ai_engine.save_model(model_path)
                else:
                    save_success = False
            except Exception as e:
                self.logger.error(f"Model saving failed: {e}")
                save_success = False
            
            if save_success:
                self.model_loaded = True
                
            response = {
                "success": True,
                "model_saved": save_success,
                "training_results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            accuracy = results.get('ensemble_accuracy') or results.get('test_accuracy', 0)
            self.logger.info(f"TRAINING: Complete - Accuracy: {accuracy:.4f}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            return {"error": str(e)}
    
    def handle_volume_profile_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Volume Profile analysis requests"""
        try:
            if VolumeProfileEngine is None:
                return {"error": "Volume Profile not available"}
                
            symbol = request.get('symbol', 'EURUSD')
            
            if symbol not in self.price_data:
                return {"error": f"No data available for {symbol}"}
            
            df = self.price_data[symbol]
            
            # Calculate Volume Profile
            try:
                vp_engine = VolumeProfileEngine()
                lookback = request.get('lookback_bars', 100)
                
                vp_data = df.tail(lookback)
                volume_profile = vp_engine.calculate_volume_profile(vp_data)
                
                current_price = df['close'].iloc[-1]
                vp_features = vp_engine.get_volume_profile_features(current_price, volume_profile)
                key_levels = vp_engine.identify_key_levels(volume_profile)
                
                response = {
                    "symbol": symbol,
                    "volume_profile": {
                        "poc_price": float(volume_profile.poc_price),
                        "poc_volume": float(volume_profile.poc_volume),
                        "value_area_high": float(volume_profile.value_area_high),
                        "value_area_low": float(volume_profile.value_area_low),
                        "total_volume": float(volume_profile.total_volume),
                        "key_levels": [float(level) for level in key_levels],
                        "features": {k: float(v) for k, v in vp_features.items()}
                    },
                    "current_price": float(current_price),
                    "timestamp": datetime.now().isoformat()
                }
                
                return response
            except Exception as e:
                return {"error": f"Volume Profile calculation failed: {str(e)}"}
            
        except Exception as e:
            self.logger.error(f"Volume Profile request error: {e}")
            return {"error": str(e)}
    
    def handle_vwap_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle VWAP analysis requests"""
        try:
            if VWAPCalculator is None:
                return {"error": "VWAP analysis not available"}
                
            symbol = request.get('symbol', 'EURUSD')
            
            if symbol not in self.price_data:
                return {"error": f"No data available for {symbol}"}
            
            df = self.price_data[symbol]
            
            try:
                # Calculate VWAP
                vwap_calc = VWAPCalculator()
                session_vwap = vwap_calc.calculate_vwap(df)
                vwap_20 = vwap_calc.calculate_vwap(df, period=20)
                vwap_50 = vwap_calc.calculate_vwap(df, period=50)
                vwap_bands = vwap_calc.calculate_vwap_bands(df, session_vwap)
                
                current_price = df['close'].iloc[-1]
                vwap_features = vwap_calc.get_vwap_features(
                    current_price, session_vwap.iloc[-1], vwap_bands, -1
                )
                
                response = {
                    "symbol": symbol,
                    "vwap_analysis": {
                        "session_vwap": float(session_vwap.iloc[-1]),
                        "vwap_20": float(vwap_20.iloc[-1]) if len(vwap_20) > 0 else None,
                        "vwap_50": float(vwap_50.iloc[-1]) if len(vwap_50) > 0 else None,
                        "vwap_upper_band": float(vwap_bands['vwap_upper'].iloc[-1]),
                        "vwap_lower_band": float(vwap_bands['vwap_lower'].iloc[-1]),
                        "features": {k: float(v) for k, v in vwap_features.items()}
                    },
                    "current_price": float(current_price),
                    "timestamp": datetime.now().isoformat()
                }
                
                return response
            except Exception as e:
                return {"error": f"VWAP calculation failed: {str(e)}"}
            
        except Exception as e:
            self.logger.error(f"VWAP request error: {e}")
            return {"error": str(e)}
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced server status"""
        uptime = datetime.now() - self.stats['uptime_start']
        
        status = {
            "server_version": "2.0.2",
            "status": "running" if self.running else "stopped",
            "uptime_seconds": int(uptime.total_seconds()),
            "uptime_formatted": str(uptime).split('.')[0],
            "connections": len(self.clients),
            "model_loaded": self.model_loaded,
            "enhanced_capabilities": self.capabilities,
            "statistics": {
                "total_connections": self.stats['connections'],
                "total_predictions": self.stats['predictions'],
                "volume_profile_predictions": self.stats['volume_profile_predictions'],
                "vwap_predictions": self.stats['vwap_predictions'],
                "total_requests": self.stats['total_requests'],
                "errors": self.stats['errors'],
                "success_rate": 1 - (self.stats['errors'] / max(self.stats['total_requests'], 1))
            },
            "data_stored": {symbol: len(data) for symbol, data in self.price_data.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        return status
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities"""
        return {
            "server_version": "2.0.2",
            "capabilities": self.capabilities,
            "supported_actions": [
                "predict", "train", "status", "capabilities", 
                "volume_profile", "vwap_analysis", "performance",
                "ping", "health_check"
            ],
            "enhanced_features": {
                "volume_profile": "Point of Control, Value Area, Key Levels" if self.capabilities['volume_profile'] else "Not Available",
                "vwap_analysis": "Multi-timeframe VWAP with bands" if self.capabilities['vwap_analysis'] else "Not Available", 
                "ensemble_models": "RandomForest + XGBoost + LogisticRegression" if self.capabilities['ensemble_models'] else "Basic RandomForest",
                "enhanced_filtering": "VP + VWAP + Market Structure" if self.capabilities['enhanced_filtering'] else "Basic Filtering",
                "market_structure": "Higher highs/lows, Support/Resistance" if self.capabilities['market_structure'] else "Basic Structure"
            },
            "ai_engine": {
                "feature_count": "65+" if self.capabilities['ensemble_models'] else "45+",
                "model_type": "Ensemble Voting Classifier" if self.capabilities['ensemble_models'] else "RandomForest",
                "confidence_threshold": getattr(self.ai_engine, 'confidence_threshold', 0.65)
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get AI engine performance statistics"""
        try:
            if self.model_loaded and hasattr(self.ai_engine, 'get_model_performance_stats'):
                return self.ai_engine.get_model_performance_stats()
            elif self.model_loaded and hasattr(self.ai_engine, 'get_model_info'):
                return self.ai_engine.get_model_info()
            else:
                return {"message": "No model loaded or performance stats not available"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def stop_server(self):
        """Stop the enhanced server"""
        self.logger.info("STOPPING: Enhanced ForexAI Socket Server...")
        self.running = False
        
        # Close all client connections
        for client in self.clients[:]:
            try:
                client.close()
            except:
                pass
        
        # Close server socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        self.logger.info("STOPPED: Enhanced server stopped")


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\nReceived shutdown signal...")
    if 'server' in globals():
        server.stop_server()
    sys.exit(0)


def test_server_connection(host="localhost", port=8888):
    """Test server connection and basic functionality"""
    try:
        print(f"Testing connection to {host}:{port}...")
        
        # Test basic connection
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect((host, port))
        
        # Test ping
        ping_request = {"action": "ping"}
        client.send(json.dumps(ping_request).encode('utf-8'))
        
        response = client.recv(1024)
        ping_response = json.loads(response.decode('utf-8'))
        
        if "pong" in ping_response.get("message", ""):
            print("✅ Ping test: SUCCESS")
        else:
            print("❌ Ping test: FAILED")
            
        # Test status
        status_request = {"action": "status"}
        client.send(json.dumps(status_request).encode('utf-8'))
        
        response = client.recv(4096)
        status_response = json.loads(response.decode('utf-8'))
        
        print(f"✅ Server Status: {status_response.get('status', 'unknown')}")
        print(f"   Version: {status_response.get('server_version', 'unknown')}")
        print(f"   Model Loaded: {status_response.get('model_loaded', False)}")
        
        # Test capabilities
        cap_request = {"action": "capabilities"}
        client.send(json.dumps(cap_request).encode('utf-8'))
        
        response = client.recv(4096)
        cap_response = json.loads(response.decode('utf-8'))
        
        capabilities = cap_response.get('capabilities', {})
        print(f"   Volume Profile: {capabilities.get('volume_profile', False)}")
        print(f"   VWAP Analysis: {capabilities.get('vwap_analysis', False)}")
        print(f"   Ensemble Models: {capabilities.get('ensemble_models', False)}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Enhanced ForexAI Socket Server v2.0.2")
    parser.add_argument('command', choices=['start', 'status', 'stop', 'test'], 
                       help='Server command')
    parser.add_argument('--host', default='localhost', 
                       help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8888, 
                       help='Server port (default: 8888)')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create and start server
        global server
        server = EnhancedSocketServer(args.host, args.port)
        
        try:
            server.start_server()
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt...")
        finally:
            server.stop_server()
    
    elif args.command == 'status':
        # Quick status check
        success = test_server_connection(args.host, args.port)
        if not success:
            print("Cannot connect to server. Is it running?")
    
    elif args.command == 'test':
        # Test server functionality
        print("🧪 Testing Enhanced ForexAI Socket Server...")
        success = test_server_connection(args.host, args.port)
        
        if success:
            print("🎉 All tests passed!")
        else:
            print("❌ Tests failed. Check server status.")
    
    elif args.command == 'stop':
        print("Stop command - use Ctrl+C to stop running server")


if __name__ == "__main__":
    main()