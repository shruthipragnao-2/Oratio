#!/usr/bin/env python3
"""
Test model scores for different texts
"""

from transformers import pipeline

def test_model_scores():
    classifier = pipeline('text-classification', model='unitary/toxic-bert')
    
    test_texts = [
        'i think women cant cook',
        'The crazy old man couldn\'t understand simple tech.',
        'This is a normal sentence with no bias.',
        'That\'s a stupid idea.'
    ]
    
    for text in test_texts:
        result = classifier(text)
        print(f'Text: "{text}"')
        print(f'Label: {result[0]["label"]}, Score: {result[0]["score"]:.3f}')
        print()

if __name__ == "__main__":
    test_model_scores()
