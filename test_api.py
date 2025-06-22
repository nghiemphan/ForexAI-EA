import socket
import json

def test_simple_protocol():
    """Test with simple JSON (no SIZE header)"""
    try:
        # Test 1: Direct JSON
        print("🔍 Test 1: Direct JSON Protocol")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('127.0.0.1', 8888))
        
        request = {"command": "health_check"}
        json_data = json.dumps(request)
        print(f"Sending: {json_data}")
        
        sock.send(json_data.encode('utf-8'))
        response = sock.recv(4096)
        print(f"Response: {response.decode('utf-8', errors='replace')}")
        sock.close()
        
    except Exception as e:
        print(f"❌ Direct JSON failed: {e}")
    
    try:
        # Test 2: With SIZE header
        print("\n🔍 Test 2: SIZE Header Protocol")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('127.0.0.1', 8888))
        
        request = {"command": "health_check"}
        json_data = json.dumps(request)
        json_bytes = json_data.encode('utf-8')
        
        # Send with SIZE header
        header = f"SIZE:{len(json_bytes)}\n"
        print(f"Sending header: {header.strip()}")
        print(f"Sending data: {json_data}")
        
        sock.send(header.encode('utf-8'))
        sock.send(json_bytes)
        
        response = sock.recv(4096)
        print(f"Response: {response.decode('utf-8', errors='replace')}")
        sock.close()
        
    except Exception as e:
        print(f"❌ SIZE header failed: {e}")

if __name__ == "__main__":
    test_simple_protocol()