# socket_server.py
"""
Basic Socket Server for ForexAI-EA Project
Handles communication between Python AI Engine and MQL5 EA
Author: Claude AI Developer
Version: 1.0.0
Created: 2025-06-08
"""

import socket
import threading
import json
import logging
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional
import signal
import sys

class AISocketServer:
    """
    Socket server for handling communication with MQL5 EA
    Processes trading requests and returns AI predictions
    """
    
    def __init__(self, host: str = "localhost", port: int = 8888):
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.is_running = False
        self.client_connections = []
        
        # Setup logging
        self.setup_logging()
        
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
            'timestamp': datetime.now().isoformat()
        }
    
    def handle_prediction_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle AI prediction request
        For now, returns mock prediction - will be replaced with actual AI model
        """
        try:
            # Extract request data
            symbol = request.get('symbol', 'UNKNOWN')
            timeframe = request.get('timeframe', 'M15')
            prices = request.get('prices', [])
            
            # Validate input
            if not prices or len(prices) < 10:
                return {
                    'status': 'error',
                    'message': 'Insufficient price data for prediction',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Mock AI prediction (will be replaced with actual model)
            prediction = self.mock_ai_prediction(symbol, prices)
            
            return {
                'status': 'success',
                'prediction': prediction,
                'confidence': 0.75,  # Mock confidence
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Prediction error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def mock_ai_prediction(self, symbol: str, prices: list) -> int:
        """
        Mock AI prediction function
        Returns: -1 (Sell), 0 (Hold), 1 (Buy)
        This will be replaced with actual AI model
        """
        # Simple mock logic based on price trend
        if len(prices) >= 5:
            recent_prices = prices[-5:]
            trend = sum([1 if recent_prices[i] > recent_prices[i-1] else -1 
                        for i in range(1, len(recent_prices))])
            
            if trend >= 2:
                return 1  # Buy signal
            elif trend <= -2:
                return -1  # Sell signal
            else:
                return 0  # Hold
        
        return 0  # Default hold
    
    def handle_health_check(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check request"""
        return {
            'status': 'success',
            'server_status': 'healthy',
            'uptime_seconds': time.time(),
            'active_connections': len(self.client_connections),
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
    print("=" * 50)
    print("ForexAI-EA Socket Server v1.0.0")
    print("=" * 50)
    
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