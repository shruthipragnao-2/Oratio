#!/usr/bin/env python3
"""
Test script for bias detection
"""

import asyncio
from main import AnalyzeRequest, User, analyze_text
from transformers import pipeline

async def test_bias_detection():
    """Test the bias detection functionality"""
    print("Testing bias detection...")
    
    # Initialize the model
    import main
    try:
        main.bias_classifier = pipeline('text-classification', model='unitary/toxic-bert')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        main.bias_classifier = None
    
    # Test cases
    test_cases = [
        "i think women cant cook",
        "The crazy old man couldn't understand simple tech.",
        "This is a normal sentence with no bias.",
        "That's a stupid idea."
    ]
    
    for text in test_cases:
        print(f"\n--- Testing: '{text}' ---")
        
        request = AnalyzeRequest(text=text)
        user = User(id=1, email='test@example.com')
        
        try:
            result = await analyze_text(request, user)
            print(f"Biased count: {result.summary['biased_count']}")
            print(f"Score: {result.summary['score']}")
            
            for sentence in result.sentences:
                print(f"Sentence: {sentence.sentence}")
                print(f"Biased spans: {len(sentence.biased_spans)}")
                if sentence.biased_spans:
                    for span in sentence.biased_spans:
                        print(f"  - {span.text} ({span.type})")
                print(f"Suggestion: {sentence.suggestion}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_bias_detection())
