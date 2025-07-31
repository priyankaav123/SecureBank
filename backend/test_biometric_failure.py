import requests
import json

# Test biometric authentication failure logging
API_BASE = 'http://localhost:5000/api'

def test_biometric_failure():
    """Test that biometric authentication failures are properly logged"""
    
    # Test 1: Invalid credentials
    print("Test 1: Testing invalid credentials...")
    response = requests.post(f'{API_BASE}/auth/biometric-login', json={
        'email': 'test@example.com',
        'password': 'wrongpassword',
        'keystrokeData': [{'key': 'a', 'dwellTime': 100, 'flightTime': 50}]
    })
    
    print(f"Response status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 2: Valid credentials but simulate biometric mismatch
    print("\nTest 2: Testing with existing user but potential biometric mismatch...")
    response = requests.post(f'{API_BASE}/auth/biometric-login', json={
        'email': 'priyankaavijay04@gmail.com',
        'password': 'test123',  # This might be wrong password
        'keystrokeData': [{'key': 'a', 'dwellTime': 100, 'flightTime': 50}]
    })
    
    print(f"Response status: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == '__main__':
    test_biometric_failure()