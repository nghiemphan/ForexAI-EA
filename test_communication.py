# test_communication.py
"""
Communication Test Script for ForexAI-EA Project
Tests socket communication between Python AI Server and MQL5 EA
Author: Claude AI Developer
Version: 1.0.0
Created: 2025-06-08
"""

import socket
import json
import time
import threading
from datetime import datetime

class CommunicationTester:
    """Test suite for socket communication"""
    
    def __init__(self, host="localhost", port=8888):
        self.host = host
        self.port = port
        self.test_results = []
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 60)
        print("ForexAI-EA Communication Test Suite")
        print("=" * 60)
        
        # Test 1: Server availability
        self.test_server_availability()
        
        # Test 2: Basic ping/pong
        self.test_ping_pong()
        
        # Test 3: Prediction request
        self.test_prediction_request()
        
        # Test 4: Health check
        self.test_health_check()
        
        # Test 5: Error handling
        self.test_error_handling()
        
        # Test 6: Concurrent connections
        self.test_concurrent_connections()
        
        # Print results
        self.print_test_results()
    
    def test_server_availability(self):
        """Test if AI server is running and accepting connections"""
        print("\n🔍 Test 1: Server Availability")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            
            if result == 0:
                self.log_test("Server Availability", True, "Server is running and accepting connections")
            else:
                self.log_test("Server Availability", False, f"Cannot connect to server (Error: {result})")
                
        except Exception as e:
            self.log_test("Server Availability", False, f"Connection test failed: {e}")
    
    def test_ping_pong(self):
        """Test basic ping/pong communication"""
        print("\n🔍 Test 2: Ping/Pong Communication")
        
        try:
            response = self.send_request({
                "type": "ping",
                "timestamp": datetime.now().isoformat()
            })
            
            if response and "pong" in response.get("message", ""):
                self.log_test("Ping/Pong", True, "Ping/pong communication successful")
            else:
                self.log_test("Ping/Pong", False, f"Invalid ping response: {response}")
                
        except Exception as e:
            self.log_test("Ping/Pong", False, f"Ping test failed: {e}")
    
    def test_prediction_request(self):
        """Test AI prediction request"""
        print("\n🔍 Test 3: Prediction Request")
        
        try:
            # Sample price data for testing - INCREASE TO 100 PRICES
            sample_prices = []
            base_price = 1.1234
            
            # Generate 100 realistic price points
            for i in range(100):
                # Add small random variations
                variation = (i * 0.00001) + (0.0001 * (i % 10 - 5) / 10)
                price = base_price + variation
                sample_prices.append(round(price, 5))
            
            response = self.send_request({
                "type": "prediction",
                "symbol": "EURUSD",
                "timeframe": "M15",
                "prices": sample_prices,  # Now 100 prices instead of 15
                "timestamp": datetime.now().isoformat()
            })
            
            if response and response.get("status") == "success":
                prediction = response.get("prediction")
                confidence = response.get("confidence")
                
                if prediction in [-1, 0, 1] and 0 <= confidence <= 1:
                    self.log_test("Prediction Request", True, 
                                f"Valid prediction: {prediction} (confidence: {confidence:.3f})")
                else:
                    self.log_test("Prediction Request", False, 
                                f"Invalid prediction format: pred={prediction}, conf={confidence}")
            else:
                self.log_test("Prediction Request", False, f"Prediction failed: {response}")
                
        except Exception as e:
            self.log_test("Prediction Request", False, f"Prediction test failed: {e}")
    
    def test_health_check(self):
        """Test health check endpoint"""
        print("\n🔍 Test 4: Health Check")
        
        try:
            response = self.send_request({
                "type": "health_check",
                "timestamp": datetime.now().isoformat()
            })
            
            if response and response.get("server_status") == "healthy":
                self.log_test("Health Check", True, "Server health check passed")
            else:
                self.log_test("Health Check", False, f"Health check failed: {response}")
                
        except Exception as e:
            self.log_test("Health Check", False, f"Health check test failed: {e}")
    
    def test_error_handling(self):
        """Test server error handling"""
        print("\n🔍 Test 5: Error Handling")
        
        try:
            # Test with invalid JSON
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            sock.send(b"invalid json")
            
            response = sock.recv(1024).decode('utf-8')
            sock.close()
            
            parsed_response = json.loads(response)
            if parsed_response.get("status") == "error":
                self.log_test("Error Handling", True, "Server properly handles invalid JSON")
            else:
                self.log_test("Error Handling", False, f"Unexpected response: {response}")
                
        except Exception as e:
            self.log_test("Error Handling", False, f"Error handling test failed: {e}")
    
    def test_concurrent_connections(self):
        """Test multiple concurrent connections"""
        print("\n🔍 Test 6: Concurrent Connections")
        
        def worker_thread(thread_id, results):
            try:
                response = self.send_request({
                    "type": "ping",
                    "thread_id": thread_id,
                    "timestamp": datetime.now().isoformat()
                })
                results[thread_id] = response is not None and "pong" in response.get("message", "")
            except:
                results[thread_id] = False
        
        # Test with 5 concurrent connections
        threads = []
        results = {}
        
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i, results))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        success_count = sum(results.values())
        if success_count == 5:
            self.log_test("Concurrent Connections", True, "All 5 concurrent connections successful")
        else:
            self.log_test("Concurrent Connections", False, 
                        f"Only {success_count}/5 concurrent connections successful")
    
    def send_request(self, request_data, timeout=10):
        """Send request to AI server and return response"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((self.host, self.port))
            
            # Send request
            request_json = json.dumps(request_data)
            sock.send(request_json.encode('utf-8'))
            
            # Receive response
            response_data = sock.recv(4096).decode('utf-8')
            sock.close()
            
            return json.loads(response_data)
            
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    def log_test(self, test_name, success, message):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now()
        }
        self.test_results.append(result)
        print(f"   {status}: {message}")
    
    def print_test_results(self):
        """Print final test results summary"""
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.test_results if r["success"])
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print()
        
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test']}: {result['message']}")
        
        print("\n" + "=" * 60)
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Communication system is ready.")
            print("✅ You can now proceed with EA testing in MetaTrader 5")
        else:
            print("⚠️  Some tests failed. Please check the AI server and configuration.")
            print("❌ Fix issues before proceeding with EA testing")
        
        return passed == total

def quick_ai_test():
    """Quick test for AI prediction with sufficient data"""
    import socket
    import json
    from datetime import datetime
    
    # Generate 100 price points
    sample_prices = [1.1000 + (i * 0.00001) for i in range(100)]
    
    request = {
        "type": "prediction",
        "symbol": "EURUSD", 
        "timeframe": "M15",
        "prices": sample_prices,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("localhost", 8888))
        sock.send(json.dumps(request).encode('utf-8'))
        
        response = sock.recv(4096).decode('utf-8')
        sock.close()
        
        result = json.loads(response)
        
        if result.get("status") == "success":
            print(f"✅ AI Prediction Success!")
            print(f"   Signal: {result.get('prediction')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   Time: {result.get('prediction_time_ms', 0):.1f}ms")
            return True
        else:
            print(f"❌ AI Prediction Failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Starting ForexAI-EA Communication Tests...")
    print("Make sure the AI Socket Server is running before starting tests.")
    
    # Wait for user confirmation
    input("Press Enter when AI server is running...")
    
    # Run tests
    tester = CommunicationTester()
    success = tester.run_all_tests()
    
    
    # Provide next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    
    if success:
        print("1. ✅ Open MetaTrader 5")
        print("2. ✅ Compile and attach ForexAI_EA_v1.mq5")
        print("3. ✅ Check EA logs for connection confirmation")
        print("4. ✅ Monitor EA behavior on demo account")
        print("5. ✅ Report any issues for resolution")
    else:
        print("1. ❌ Check AI server is running (python socket_server.py)")
        print("2. ❌ Verify firewall allows port 8888")
        print("3. ❌ Check network connectivity")
        print("4. ❌ Review server logs for errors")
        print("5. ❌ Re-run tests after fixing issues")
    
    print("\n📧 Contact developer if you need assistance!")

if __name__ == "__main__":
    main()
    print("🧪 Quick AI Prediction Test")
    quick_ai_test()

# Additional utility functions for testing

def quick_ping_test(host="localhost", port=8888):
    """Quick ping test function"""
    try:
        request = {"type": "ping", "timestamp": datetime.now().isoformat()}
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))
        sock.send(json.dumps(request).encode('utf-8'))
        
        response = sock.recv(1024).decode('utf-8')
        sock.close()
        
        parsed = json.loads(response)
        return "pong" in parsed.get("message", "")
        
    except:
        return False

def stress_test_server(host="localhost", port=8888, num_requests=100):
    """Stress test the server with multiple rapid requests"""
    print(f"\n🔥 Stress Testing Server with {num_requests} requests...")
    
    success_count = 0
    start_time = time.time()
    
    for i in range(num_requests):
        if quick_ping_test(host, port):
            success_count += 1
        
        if (i + 1) % 10 == 0:
            print(f"   Completed {i + 1}/{num_requests} requests...")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n📊 Stress Test Results:")
    print(f"   Total Requests: {num_requests}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {num_requests - success_count}")
    print(f"   Success Rate: {(success_count/num_requests)*100:.1f}%")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Requests/Second: {num_requests/duration:.1f}")
    
    return success_count == num_requests

def latency_test(host="localhost", port=8888, num_tests=10):
    """Test communication latency"""
    print(f"\n⚡ Testing Communication Latency ({num_tests} tests)...")
    
    latencies = []
    
    for i in range(num_tests):
        start_time = time.time()
        
        if quick_ping_test(host, port):
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            print(f"   Test {i+1}: {latency:.1f}ms")
        else:
            print(f"   Test {i+1}: FAILED")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"\n📊 Latency Results:")
        print(f"   Average: {avg_latency:.1f}ms")
        print(f"   Minimum: {min_latency:.1f}ms")
        print(f"   Maximum: {max_latency:.1f}ms")
        
        if avg_latency < 50:
            print("   ✅ Excellent latency (<50ms)")
        elif avg_latency < 100:
            print("   ✅ Good latency (<100ms)")
        else:
            print("   ⚠️  High latency (>100ms) - consider optimization")
    else:
        print("   ❌ All latency tests failed")