"""
Text preprocessing module for Marathi language.

This module provides functions for cleaning and preprocessing Marathi text
from social media, including URL removal, mention handling, and whitespace normalization.
"""

import re
from typing import List, Optional


def preprocess_marathi_text(text: str) -> str:
    """
    Preprocess Marathi text for sentiment analysis.
    
    Applies the following transformations:
    1. Remove URLs (http/https)
    2. Remove @mentions
    3. Extract text from hashtags (keep word, remove #)
    4. Normalize whitespace
    5. Preserve Devanagari characters (U+0900 to U+097F)
    
    Args:
        text: Input Marathi text string
        
    Returns:
        Preprocessed text string
        
    Examples:
        >>> preprocess_marathi_text("@user हे चांगले आहे! #movie http://example.com")
        'हे चांगले आहे! movie'
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs (http and https)
    text = re.sub(r'http(s)?://\S+', '', text)
    
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    
    # Keep text after hashtags (remove # but keep word)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Normalize whitespace (multiple spaces to single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


def batch_preprocess(texts: List[str]) -> List[str]:
    """
    Preprocess a batch of texts.
    
    Args:
        texts: List of input text strings
        
    Returns:
        List of preprocessed text strings
    """
    return [preprocess_marathi_text(text) for text in texts]


def remove_special_characters(text: str, keep_devanagari: bool = True) -> str:
    """
    Remove special characters from text.
    
    Args:
        text: Input text
        keep_devanagari: If True, preserve Devanagari script characters
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    if keep_devanagari:
        # Keep Devanagari (U+0900-U+097F), alphanumeric, and basic punctuation
        text = re.sub(r'[^\u0900-\u097F\w\s.,!?।]', '', text)
    else:
        # Keep only alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
    
    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalize text by handling common variations.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase (for non-Devanagari characters)
    # Note: Devanagari doesn't have case, so this mainly affects English words
    
    # Remove extra punctuation
    text = re.sub(r'([।.!?]){2,}', r'\1', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def clean_marathi_text_advanced(
    text: str,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_hashtags: bool = False,
    remove_special_chars: bool = False,
    normalize: bool = True
) -> str:
    """
    Advanced text cleaning with configurable options.
    
    Args:
        text: Input text
        remove_urls: Remove URLs
        remove_mentions: Remove @mentions
        remove_hashtags: Remove hashtags completely (if False, keeps text)
        remove_special_chars: Remove special characters
        normalize: Apply normalization
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http(s)?://\S+', '', text)
    
    # Remove mentions
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)
    
    # Handle hashtags
    if remove_hashtags:
        text = re.sub(r'#\w+', '', text)
    else:
        text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove special characters
    if remove_special_chars:
        text = remove_special_characters(text, keep_devanagari=True)
    
    # Normalize
    if normalize:
        text = normalize_text(text)
    
    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_text_statistics(text: str) -> dict:
    """
    Get statistics about a text string.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with statistics
    """
    if not isinstance(text, str):
        return {
            'length': 0,
            'word_count': 0,
            'devanagari_chars': 0,
            'has_urls': False,
            'has_mentions': False,
            'has_hashtags': False
        }
    
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'devanagari_chars': len(re.findall(r'[\u0900-\u097F]', text)),
        'has_urls': bool(re.search(r'http(s)?://\S+', text)),
        'has_mentions': bool(re.search(r'@\w+', text)),
        'has_hashtags': bool(re.search(r'#\w+', text))
    }


if __name__ == "__main__":
    """Test preprocessing functions."""
    
    # Test cases
    test_texts = [
        " होता होता राहीलेला  निवडणूक मारो मर्ज़ीभई",
        "मुंबईतील घाटकोपरमध्ये धुळवड खेळून घरी परतलेलं दाम्पत्य बाथरुममध्ये मृतावस्थेत आढळलं! ..//-…",
        "@user हे चांगले आहे! #movie http://example.com",
        "खरा लखोबा तर हा बोबडाच आहे    "
    ]
    
    print("Testing Marathi Text Preprocessing\n")
    print("=" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Original:  {text[:60]}...")
        
        cleaned = preprocess_marathi_text(text)
        print(f"Cleaned:   {cleaned[:60]}...")
        
        stats = get_text_statistics(text)
        print(f"Stats:     Length={stats['length']}, Words={stats['word_count']}, "
              f"Devanagari={stats['devanagari_chars']}")
        print(f"           URLs={stats['has_urls']}, Mentions={stats['has_mentions']}, "
              f"Hashtags={stats['has_hashtags']}")
    
    print("\n" + "=" * 70)
    print("✅ All preprocessing tests completed!")
