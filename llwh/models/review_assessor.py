"""Review Assessment Module for analyzing and evaluating text reviews."""

from __future__ import division
import re
import datetime
import io
from typing import Dict, List, Any, Optional


class ReviewAssessor:
    """
    Comprehensive review assessment system for analyzing text reviews,
    conversations, and model outputs.
    
    Provides quality scoring, sentiment analysis, coherence checking,
    and factual consistency assessment.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Review Assessor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self.get_default_config()
        self.assessment_history = []
        
    @staticmethod
    def get_default_config():
        """Get default configuration."""
        return {
            'min_length': 10,
            'max_length': 10000,
            'quality_weights': {
                'length': 0.2,
                'sentiment': 0.3,
                'coherence': 0.3,
                'clarity': 0.2
            }
        }
    
    def assess_review(self, text, include_details=True):
        """
        Perform comprehensive assessment of a review.
        
        Args:
            text: Review text to assess
            include_details: Whether to include detailed analysis
            
        Returns:
            assessment: Dictionary containing assessment results
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Perform individual assessments
        quality_score = self.assess_quality(text)
        sentiment_score = self.assess_sentiment(text)
        coherence_score = self.assess_coherence(text)
        clarity_score = self.assess_clarity(text)
        
        # Calculate overall score
        weights = self.config['quality_weights']
        overall_score = (
            quality_score * weights['length'] +
            sentiment_score * weights['sentiment'] +
            coherence_score * weights['coherence'] +
            clarity_score * weights['clarity']
        )
        
        assessment = {
            'overall_score': round(overall_score, 2),
            'quality_score': round(quality_score, 2),
            'sentiment_score': round(sentiment_score, 2),
            'coherence_score': round(coherence_score, 2),
            'clarity_score': round(clarity_score, 2),
            'word_count': len(text.split()),
            'char_count': len(text),
            'recommendation': self._get_recommendation(overall_score)
        }
        
        if include_details:
            assessment['details'] = {
                'positive_indicators': self._extract_positive_indicators(text),
                'negative_indicators': self._extract_negative_indicators(text),
                'key_phrases': self._extract_key_phrases(text),
                'structure_analysis': self._analyze_structure(text)
            }
        
        # Store in history
        self.assessment_history.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'score': overall_score,
            'timestamp': self._get_timestamp()
        })
        
        return assessment
    
    def assess_quality(self, text):
        """
        Assess overall text quality.
        
        Args:
            text: Text to assess
            
        Returns:
            score: Quality score (0-100)
        """
        # Check length appropriateness
        word_count = len(text.split())
        
        if word_count < self.config['min_length']:
            length_score = (word_count / self.config['min_length']) * 50
        elif word_count > self.config['max_length']:
            length_score = max(0, 100 - (word_count - self.config['max_length']) / 100)
        else:
            length_score = 100
        
        # Check for proper capitalization
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        capitalized = sum(1 for s in sentences if s and s[0].isupper())
        capitalization_score = (capitalized / max(len(sentences), 1)) * 100
        
        # Check for punctuation
        has_punctuation = any(c in text for c in '.!?')
        punctuation_score = 100 if has_punctuation else 50
        
        # Combine scores
        quality_score = (length_score * 0.4 + 
                        capitalization_score * 0.3 + 
                        punctuation_score * 0.3)
        
        return min(100, max(0, quality_score))
    
    def assess_sentiment(self, text):
        """
        Assess sentiment of the text.
        
        Args:
            text: Text to assess
            
        Returns:
            score: Sentiment score (0-100, 50 is neutral)
        """
        text_lower = text.lower()
        
        # Positive words
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful',
            'fantastic', 'outstanding', 'superb', 'brilliant', 'perfect',
            'love', 'like', 'enjoy', 'pleased', 'happy', 'satisfied'
        ]
        
        # Negative words
        negative_words = [
            'bad', 'poor', 'terrible', 'awful', 'horrible',
            'disappointing', 'worst', 'hate', 'dislike', 'unhappy',
            'unsatisfied', 'problem', 'issue', 'fail', 'broken'
        ]
        
        # Count occurrences
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        
        # Calculate sentiment score
        total = positive_count + negative_count
        if total == 0:
            return 50  # Neutral
        
        sentiment_ratio = positive_count / total
        sentiment_score = sentiment_ratio * 100
        
        return sentiment_score
    
    def assess_coherence(self, text):
        """
        Assess text coherence and flow.
        
        Args:
            text: Text to assess
            
        Returns:
            score: Coherence score (0-100)
        """
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) == 0:
            return 0
        
        # Check sentence length variation (good indicator of natural flow)
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 1:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            variation_score = min(100, 100 - (variance / 10))
        else:
            variation_score = 70
        
        # Check for transition words
        transition_words = [
            'however', 'therefore', 'moreover', 'furthermore',
            'additionally', 'consequently', 'meanwhile', 'nevertheless',
            'although', 'because', 'thus', 'hence'
        ]
        
        text_lower = text.lower()
        transition_count = sum(1 for word in transition_words if word in text_lower)
        transition_score = min(100, (transition_count / max(len(sentences), 1)) * 200)
        
        # Check for paragraph structure (double newlines)
        has_paragraphs = '\n\n' in text or '\r\n\r\n' in text
        structure_score = 100 if has_paragraphs else 70
        
        # Combine scores
        coherence_score = (variation_score * 0.4 + 
                          transition_score * 0.3 + 
                          structure_score * 0.3)
        
        return min(100, max(0, coherence_score))
    
    def assess_clarity(self, text):
        """
        Assess text clarity and readability.
        
        Args:
            text: Text to assess
            
        Returns:
            score: Clarity score (0-100)
        """
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not words or not sentences:
            return 0
        
        # Average word length (shorter is generally clearer)
        avg_word_length = sum(len(w) for w in words) / len(words)
        word_length_score = max(0, 100 - (avg_word_length - 5) * 10)
        
        # Average sentence length (moderate is best)
        avg_sentence_length = len(words) / len(sentences)
        if 10 <= avg_sentence_length <= 20:
            sentence_length_score = 100
        elif avg_sentence_length < 10:
            sentence_length_score = 70 + (avg_sentence_length / 10) * 30
        else:
            sentence_length_score = max(0, 100 - (avg_sentence_length - 20) * 2)
        
        # Check for jargon/complexity (simplified)
        complex_words = [w for w in words if len(w) > 12]
        complexity_ratio = len(complex_words) / len(words)
        complexity_score = max(0, 100 - complexity_ratio * 200)
        
        # Combine scores
        clarity_score = (word_length_score * 0.3 + 
                        sentence_length_score * 0.4 + 
                        complexity_score * 0.3)
        
        return min(100, max(0, clarity_score))
    
    def assess_factual_consistency(self, text, reference_texts=None):
        """
        Assess factual consistency against reference texts.
        
        Args:
            text: Text to assess
            reference_texts: Optional list of reference texts
            
        Returns:
            consistency_score: Consistency score (0-100)
        """
        if not reference_texts:
            # Without references, check for internal consistency
            return self._check_internal_consistency(text)
        
        # Check consistency with references
        text_lower = text.lower()
        consistency_scores = []
        
        for ref in reference_texts:
            ref_lower = ref.lower()
            
            # Extract key terms from both
            text_terms = set(re.findall(r'\b\w{4,}\b', text_lower))
            ref_terms = set(re.findall(r'\b\w{4,}\b', ref_lower))
            
            # Calculate overlap
            if ref_terms:
                overlap = len(text_terms & ref_terms) / len(ref_terms)
                consistency_scores.append(overlap * 100)
        
        if consistency_scores:
            return sum(consistency_scores) / len(consistency_scores)
        else:
            return 50  # Neutral if no valid comparison
    
    def _check_internal_consistency(self, text):
        """Check internal consistency of text."""
        # Look for contradictory statements (simplified)
        text_lower = text.lower()
        
        # Check for negation patterns
        positive_statements = text_lower.count('is good') + text_lower.count('works well')
        negative_statements = text_lower.count('is bad') + text_lower.count('does not work')
        
        if positive_statements > 0 and negative_statements > 0:
            # Mixed signals reduce consistency
            ratio = min(positive_statements, negative_statements) / max(positive_statements, negative_statements)
            return 100 - (ratio * 30)
        
        return 85  # Default high consistency
    
    def _extract_positive_indicators(self, text):
        """Extract positive indicators from text."""
        text_lower = text.lower()
        indicators = []
        
        positive_patterns = [
            'recommend', 'excellent', 'great', 'amazing',
            'perfect', 'wonderful', 'fantastic', 'outstanding'
        ]
        
        for pattern in positive_patterns:
            if pattern in text_lower:
                indicators.append(pattern)
        
        return indicators[:5]  # Return top 5
    
    def _extract_negative_indicators(self, text):
        """Extract negative indicators from text."""
        text_lower = text.lower()
        indicators = []
        
        negative_patterns = [
            'not recommend', 'terrible', 'awful', 'poor',
            'worst', 'horrible', 'disappointing', 'useless'
        ]
        
        for pattern in negative_patterns:
            if pattern in text_lower:
                indicators.append(pattern)
        
        return indicators[:5]  # Return top 5
    
    def _extract_key_phrases(self, text):
        """Extract key phrases from text."""
        # Simple extraction based on sentence fragments
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Take first sentence and longest sentence as key phrases
        key_phrases = []
        
        if sentences:
            key_phrases.append(sentences[0])
            
            if len(sentences) > 1:
                longest = max(sentences[1:], key=len)
                if longest not in key_phrases:
                    key_phrases.append(longest)
        
        return key_phrases[:3]
    
    def _analyze_structure(self, text):
        """Analyze text structure."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        paragraphs = text.split('\n\n')
        
        return {
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_sentence_length': len(text.split()) / max(len(sentences), 1),
            'has_introduction': len(sentences) > 0,
            'has_conclusion': len(sentences) > 2
        }
    
    def _get_recommendation(self, score):
        """Get recommendation based on score."""
        if score >= 80:
            return 'Excellent - Highly recommended'
        elif score >= 60:
            return 'Good - Recommended with minor improvements'
        elif score >= 40:
            return 'Fair - Needs improvement'
        else:
            return 'Poor - Significant revision needed'
    
    def _get_timestamp(self):
        """Get current timestamp (simplified)."""
        return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    
    def batch_assess(self, texts):
        """
        Assess multiple reviews in batch.
        
        Args:
            texts: List of text reviews to assess
            
        Returns:
            assessments: List of assessment results
        """
        assessments = []
        
        for text in texts:
            try:
                assessment = self.assess_review(text, include_details=False)
                assessments.append(assessment)
            except Exception as e:
                assessments.append({
                    'error': str(e),
                    'overall_score': 0
                })
        
        return assessments
    
    def compare_reviews(self, text1, text2):
        """
        Compare two reviews.
        
        Args:
            text1: First review text
            text2: Second review text
            
        Returns:
            comparison: Comparison results
        """
        assessment1 = self.assess_review(text1, include_details=False)
        assessment2 = self.assess_review(text2, include_details=False)
        
        return {
            'review1': assessment1,
            'review2': assessment2,
            'score_difference': abs(assessment1['overall_score'] - 
                                   assessment2['overall_score']),
            'better_review': 1 if assessment1['overall_score'] > 
                                  assessment2['overall_score'] else 2,
            'similarity': self._calculate_similarity(text1, text2)
        }
    
    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0
        
        return round((intersection / union) * 100, 2)
    
    def get_assessment_statistics(self):
        """
        Get statistics from assessment history.
        
        Returns:
            stats: Dictionary of statistics
        """
        if not self.assessment_history:
            return {
                'total_assessments': 0,
                'average_score': 0,
                'highest_score': 0,
                'lowest_score': 0
            }
        
        scores = [a['score'] for a in self.assessment_history]
        
        return {
            'total_assessments': len(self.assessment_history),
            'average_score': round(sum(scores) / len(scores), 2),
            'highest_score': round(max(scores), 2),
            'lowest_score': round(min(scores), 2)
        }
    
    def export_assessment_report(self, filepath, assessments=None):
        """
        Export assessment report to file.
        
        Args:
            filepath: Path to save report
            assessments: Optional specific assessments to export
        """
        if assessments is None:
            assessments = self.assessment_history
        
        with io.open(filepath, 'w', encoding='utf-8') as f:
            f.write(u"Review Assessment Report\n")
            f.write(u"=" * 50 + u"\n\n")
            
            stats = self.get_assessment_statistics()
            f.write(u"Statistics:\n")
            f.write(u"  Total Assessments: {}\n".format(stats['total_assessments']))
            f.write(u"  Average Score: {}\n".format(stats['average_score']))
            f.write(u"  Highest Score: {}\n".format(stats['highest_score']))
            f.write(u"  Lowest Score: {}\n\n".format(stats['lowest_score']))
            
            f.write(u"Individual Assessments:\n")
            f.write(u"-" * 50 + u"\n")
            
            for i, assessment in enumerate(assessments, 1):
                f.write(u"\nAssessment #{}\n".format(i))
                f.write(u"  Text: {}\n".format(assessment.get('text', 'N/A')))
                f.write(u"  Score: {}\n".format(assessment.get('score', 'N/A')))
                f.write(u"  Timestamp: {}\n".format(assessment.get('timestamp', 'N/A')))
    
    def clear_history(self):
        """Clear assessment history."""
        self.assessment_history = []
