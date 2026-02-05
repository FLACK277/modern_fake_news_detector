"""
Text preprocessing and validation utilities
Custom implementations for content sanitization
"""

import re
from typing import Tuple


class ContentSanitizer:
    """Handles text cleaning operations with regex patterns"""
    
    # Compile patterns once for performance
    HYPERLINK_PATTERN = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    SPECIAL_CHAR_PATTERN = re.compile(r'[^\w\s\.,!?\-\']', flags=re.UNICODE)
    EXCESS_WHITESPACE_PATTERN = re.compile(r'\s+')
    
    @classmethod
    def purify_content(cls, raw_text: str) -> str:
        """
        Multi-stage text purification pipeline
        Removes web artifacts and normalizes spacing
        """
        # Stage 1: Strip hyperlinks
        no_urls = cls.HYPERLINK_PATTERN.sub(' ', raw_text)
        
        # Stage 2: Eliminate HTML markup
        no_html = cls.HTML_TAG_PATTERN.sub(' ', no_urls)
        
        # Stage 3: Filter special characters (keep basic punctuation)
        normalized = cls.SPECIAL_CHAR_PATTERN.sub(' ', no_html)
        
        # Stage 4: Collapse multiple spaces
        compacted = cls.EXCESS_WHITESPACE_PATTERN.sub(' ', normalized)
        
        return compacted.strip()


class ContentValidator:
    """Validates content meets quality thresholds"""
    
    MIN_WORD_COUNT = 3
    MIN_CHAR_LENGTH = 10
    MAX_CHAR_LENGTH = 10000
    
    @classmethod
    def assess_content_quality(cls, content: str) -> Tuple[bool, str]:
        """
        Evaluates if content meets minimum quality standards
        Returns: (is_valid, error_message)
        """
        if not content or not content.strip():
            return False, "Content is empty or contains only whitespace"
        
        char_count = len(content)
        if char_count < cls.MIN_CHAR_LENGTH:
            return False, f"Content too short: {char_count} chars (minimum {cls.MIN_CHAR_LENGTH})"
        
        if char_count > cls.MAX_CHAR_LENGTH:
            return False, f"Content exceeds limit: {char_count} chars (maximum {cls.MAX_CHAR_LENGTH})"
        
        word_list = content.split()
        if len(word_list) < cls.MIN_WORD_COUNT:
            return False, f"Insufficient words: {len(word_list)} (minimum {cls.MIN_WORD_COUNT})"
        
        return True, "Content validated successfully"


def prepare_text_for_analysis(raw_input: str) -> str:
    """
    Convenience function for complete text preparation
    Combines sanitization and returns clean text
    """
    clean_text = ContentSanitizer.purify_content(raw_input)
    return clean_text


def validate_input_text(text_content: str) -> Tuple[bool, str]:
    """
    Convenience function for validation
    Wraps validator class for easier usage
    """
    return ContentValidator.assess_content_quality(text_content)
