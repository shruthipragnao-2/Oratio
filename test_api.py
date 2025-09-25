#!/usr/bin/env python3
"""
Simple test script for Oratio Backend API
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_signup():
    """Test user signup"""
    try:
        data = {"email": "test@example.com", "password": "testpass123"}
        response = requests.post(f"{BASE_URL}/auth/signup", json=data)
        print(f"Signup: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Signup failed: {e}")
        return False

def test_login():
    """Test user login"""
    try:
        data = {"email": "test@example.com", "password": "testpass123"}
        response = requests.post(f"{BASE_URL}/auth/login", json=data)
        print(f"Login: {response.status_code} - {response.json()}")
        if response.status_code == 200:
            return response.json().get("access_token")
        return None
    except Exception as e:
        print(f"Login failed: {e}")
        return None

def test_analyze(token):
    """Test text analysis"""
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        data = {"text": "The crazy old man couldn't understand simple tech."}
        response = requests.post(f"{BASE_URL}/analyze", json=data, headers=headers)
        print(f"Analyze: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Found {result['summary']['biased_count']} biased spans")
            print(f"Bias score: {result['summary']['score']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Analyze failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Oratio Backend API...")
    print("=" * 40)
    
    # Test health
    if not test_health():
        print("❌ Health check failed")
        exit(1)
    
    # Test signup
    if not test_signup():
        print("❌ Signup failed")
        exit(1)
    
    # Test login
    token = test_login()
    if not token:
        print("❌ Login failed")
        exit(1)
    
    # Test analyze
    if not test_analyze(token):
        print("❌ Analyze failed")
        exit(1)
    
    print("=" * 40)
    print("✅ All tests passed!")
