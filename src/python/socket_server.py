"""
File: src/python/socket_server.py
Description: Enhanced Socket Server v2.0 - Fixed Unicode Issues
Author: Claude AI Developer
Version: 2.0.1
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

# Import enhanced modules
try:
    from enhanced_ai_engine import EnhancedAIEngine, EnhancedModelEvaluator
    from volume_profile import VolumeProfileEngine, VWAPCalculator
except ImportError:
    # Fallback imports for existing system
    try:
        from ai_engine import AITradingEngine as EnhancedAIEngine
        print("WARNING: Using fallback AI engine - enhanced features may not be available")
    except ImportError:
        print("ERROR: Cannot import AI engine modules")
        sys.exit(1)

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
            'volume_profile': True,
            'vwap_analysis': True,
            'ensemble_models': True,
            'enhanced_filtering': True,
            'market_structure': True
        }
        
    def setup_logging(self):
        """Setup enhanced logging without Unicode issues"""
        # Create logs directory
        os.makedirs('data/logs', exist_ok=True)
        
        # Configure logging with UTF-8 encoding
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/logs/enhanced_socket_server.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def start_server(self):
        """Start the enhanced socket server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            
            self.running = True
            self.stats['uptime_start'] = datetime.now()
            
            self.logger.info(f"STARTED: Enhanced ForexAI Socket Server v2.0 on {self.host}:{self.port}")
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
                success = self.ai_engine.load_enhanced_model(model_path)
                if success:
                    self.model_loaded = True
                    self.logger.info("SUCCESS: Enhanced AI model loaded")
                else:
                    self.logger.warning("FAILED: Could not load enhanced AI model")
            else:
                self.logger.info("INFO: No existing enhanced model found - will need training")
                
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
    
    def handle_client(self, client_socket, address):
        """Handle client connection with enhanced features"""
        try:
            self.clients.append(client_socket)
            
            while self.running:
                try:
                    # Receive data
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    
                    # Parse request
                    request = json.loads(data.decode('utf-8'))
                    self.stats['total_requests'] += 1
                    
                    # Process request
                    response = self.process_enhanced_request(request)
                    
                    # Send response
                    response_json = json.dumps(response)
                    client_socket.send(response_json.encode('utf-8'))
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error: {e}")
                    error_response = {"error": "Invalid JSON format"}
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
                    
                except Exception as e:
                    self.logger.error(f"Client handling error: {e}")
                    self.stats['errors'] += 1
                    break
                    
        except Exception as e:
            self.logger.error(f"Client connection error: {e}")
        finally:
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            client_socket.close()
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
            
            # Get enhanced prediction
            signal, confidence, details = self.ai_engine.predict_enhanced(df)
            
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
                "server_version": "2.0.1"
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
            
            self.logger.info(f"TRAINING: Starting enhanced model training for {symbol}")
            
            # Train enhanced model
            results = self.ai_engine.train_enhanced_model(training_data)
            
            # Save model
            model_path = "data/models/enhanced_ai_model.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            save_success = self.ai_engine.save_enhanced_model(model_path)
            
            if save_success:
                self.model_loaded = True
                
            response = {
                "success": True,
                "model_saved": save_success,
                "training_results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"TRAINING: Complete - Accuracy: {results.get('ensemble_accuracy', 0):.4f}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            return {"error": str(e)}
    
    def handle_volume_profile_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Volume Profile analysis requests"""
        try:
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
            except:
                return {"error": "Volume Profile calculation not available - using basic analysis"}
            
        except Exception as e:
            self.logger.error(f"Volume Profile request error: {e}")
            return {"error": str(e)}
    
    def handle_vwap_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle VWAP analysis requests"""
        try:
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
            except:
                return {"error": "VWAP calculation not available - using basic analysis"}
            
        except Exception as e:
            self.logger.error(f"VWAP request error: {e}")
            return {"error": str(e)}
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced server status"""
        uptime = datetime.now() - self.stats['uptime_start']
        
        status = {
            "server_version": "2.0.1",
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
            "server_version": "2.0.1",
            "capabilities": self.capabilities,
            "supported_actions": [
                "predict", "train", "status", "capabilities", 
                "volume_profile", "vwap_analysis", "performance"
            ],
            "enhanced_features": {
                "volume_profile": "Point of Control, Value Area, Key Levels",
                "vwap_analysis": "Multi-timeframe VWAP with bands",
                "ensemble_models": "RandomForest + XGBoost + LogisticRegression",
                "enhanced_filtering": "VP + VWAP + Market Structure",
                "market_structure": "Higher highs/lows, Support/Resistance"
            },
            "ai_engine": {
                "feature_count": "65+",
                "model_type": "Ensemble Voting Classifier",
                "confidence_threshold": getattr(self.ai_engine, 'confidence_threshold', 0.65)
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get AI engine performance statistics"""
        try:
            if self.model_loaded and hasattr(self.ai_engine, 'get_model_performance_stats'):
                return self.ai_engine.get_model_performance_stats()
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
            client.close()
        
        # Close server socket
        if self.socket:
            self.socket.close()
        
        self.logger.info("STOPPED: Enhanced server stopped")


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\nReceived shutdown signal...")
    if 'server' in globals():
        server.stop_server()
    sys.exit(0)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Enhanced ForexAI Socket Server v2.0")
    parser.add_argument('command', choices=['start', 'status', 'stop'], 
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
        try:
            import socket as sock
            client = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            client.settimeout(5)
            client.connect((args.host, args.port))
            
            request = {"action": "status"}
            client.send(json.dumps(request).encode('utf-8'))
            
            response = client.recv(4096)
            status = json.loads(response.decode('utf-8'))
            
            print("Enhanced ForexAI Server Status:")
            print(f"   Status: {status.get('status', 'unknown')}")
            print(f"   Version: {status.get('server_version', 'unknown')}")
            print(f"   Uptime: {status.get('uptime_formatted', 'unknown')}")
            print(f"   Connections: {status.get('connections', 0)}")
            print(f"   Model Loaded: {status.get('model_loaded', False)}")
            print(f"   Total Predictions: {status.get('statistics', {}).get('total_predictions', 0)}")
            print(f"   Volume Profile Predictions: {status.get('statistics', {}).get('volume_profile_predictions', 0)}")
            
            client.close()
            
        except Exception as e:
            print(f"Cannot connect to server: {e}")
    
    elif args.command == 'stop':
        print("Stop command - use Ctrl+C to stop running server")


if __name__ == "__main__":
    main()