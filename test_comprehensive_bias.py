#!/usr/bin/env python3
"""
Test comprehensive bias detection with multiple models
"""

import asyncio
from main import AnalyzeRequest, User, analyze_text

async def test_comprehensive_bias():
    """Test comprehensive bias detection"""
    print("Testing Comprehensive Bias Detection...")
    print("=" * 60)
    
    # Test cases covering different types of bias
    test_cases = [
        # Gender bias
        ("i dont think women can cook", "gender_bias"),
        ("men are better at math", "gender_bias"),
        ("girls are bad at sports", "gender_bias"),
        
        # Ableist bias
        ("That's a stupid idea", "ableist"),
        ("He's crazy", "ableist"),
        ("Don't be dumb", "ableist"),
        
        # Ageist bias
        ("Old people can't use technology", "ageist"),
        ("Boomers don't understand", "ageist"),
        
        # Toxic language
        ("You're an idiot", "toxic"),
        ("This is terrible", "negative_sentiment"),
        
        # Neutral text
        ("This is a normal sentence", "neutral"),
        ("I like pizza", "neutral")
    ]
    
    for text, expected_bias in test_cases:
        print(f"\n--- Testing: '{text}' (Expected: {expected_bias}) ---")
        
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
                
                # Check if detection matches expectation
                if expected_bias == "neutral":
                    if len(sentence.biased_spans) == 0:
                        print("✅ Correctly identified as neutral")
                    else:
                        print("❌ Incorrectly flagged as biased")
                else:
                    if len(sentence.biased_spans) > 0:
                        print("✅ Correctly identified bias")
                    else:
                        print("❌ Failed to detect bias")
                        
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_comprehensive_bias())
