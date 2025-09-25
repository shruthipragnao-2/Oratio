#!/usr/bin/env python3
"""
Test the bias detection model in the running backend
"""

import requests
import json

def test_backend_model():
    """Test the bias detection model through the API"""
    base_url = "http://localhost:8000"
    
    print("Testing backend bias detection model...")
    print("=" * 50)
    
    # First, try to login to get a token
    try:
        login_data = {"email": "test@example.com", "password": "testpass123"}
        response = requests.post(f"{base_url}/auth/login", json=login_data)
        
        if response.status_code == 200:
            token = response.json()["access_token"]
            print("✅ Authentication successful")
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(f"Response: {response.text}")
            return
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return
    
    # Test cases
    test_cases = [
        "i dont think women can cook",
        "That's a stupid idea",
        "This is a normal sentence",
        "The crazy old man couldn't understand"
    ]
    
    headers = {"Authorization": f"Bearer {token}"}
    
    for text in test_cases:
        print(f"\n--- Testing: '{text}' ---")
        
        try:
            analyze_data = {"text": text}
            response = requests.post(f"{base_url}/analyze", json=analyze_data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Biased count: {result['summary']['biased_count']}")
                print(f"Score: {result['summary']['score']}")
                
                for sentence in result['sentences']:
                    print(f"Sentence: {sentence['sentence']}")
                    print(f"Biased spans: {len(sentence['biased_spans'])}")
                    if sentence['biased_spans']:
                        for span in sentence['biased_spans']:
                            print(f"  - {span['text']} ({span['type']})")
                    print(f"Suggestion: {sentence['suggestion']}")
            else:
                print(f"❌ Analyze failed: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_backend_model()
