import re
from typing import Dict, Any

class EmotionDetector:
    """
    Simple emotion detection system for user messages
    """
    
    def __init__(self):
        # Emotion keywords and patterns
        self.emotion_patterns = {
            'happy': ['happy', 'joy', 'excited', 'great', 'awesome', 'wonderful', 'love', 'amazing', 'fantastic', 'excellent'],
            'sad': ['sad', 'depressed', 'down', 'upset', 'crying', 'miserable', 'heartbroken', 'disappointed'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'rage', 'hate'],
            'anxious': ['anxious', 'worried', 'nervous', 'scared', 'afraid', 'panic', 'stress', 'overwhelmed'],
            'confused': ['confused', 'lost', 'unclear', 'puzzled', 'bewildered', 'perplexed'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'wow', 'incredible'],
            'neutral': ['okay', 'fine', 'normal', 'usual', 'regular']
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            'very': 1.5,
            'extremely': 2.0,
            'really': 1.3,
            'quite': 1.2,
            'somewhat': 0.8,
            'slightly': 0.6,
            'a bit': 0.7
        }
    
    def detect_emotion(self, text: str) -> Dict[str, Any]:
        """
        Detect emotion in text with confidence score
        """
        if not text:
            return {'emotion': 'neutral', 'confidence': 0.0, 'details': {}}
        
        text_lower = text.lower()
        emotion_scores = {}
        
        # Check for emotion keywords
        for emotion, keywords in self.emotion_patterns.items():
            score = 0
            matched_words = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    base_score = 1.0
                    
                    # Check for intensity modifiers
                    for modifier, multiplier in self.intensity_modifiers.items():
                        if modifier in text_lower and keyword in text_lower:
                            # Check if modifier appears near the keyword
                            if self._words_are_close(text_lower, modifier, keyword):
                                base_score *= multiplier
                    
                    score += base_score
                    matched_words.append(keyword)
            
            if score > 0:
                emotion_scores[emotion] = {
                    'score': score,
                    'matched_words': matched_words
                }
        
        # Determine primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores.keys(), key=lambda x: emotion_scores[x]['score'])
            max_score = emotion_scores[primary_emotion]['score']
            
            # Calculate confidence based on score and word count
            word_count = len(text.split())
            confidence = min(1.0, max_score / max(1, word_count * 0.1))
            
            return {
                'emotion': primary_emotion,
                'confidence': confidence,
                'details': emotion_scores
            }
        
        # Check for punctuation-based emotion indicators
        if '!' in text:
            if any(word in text_lower for word in ['great', 'awesome', 'amazing', 'yes']):
                return {'emotion': 'happy', 'confidence': 0.6, 'details': {'punctuation': 'exclamation'}}
            else:
                return {'emotion': 'surprised', 'confidence': 0.5, 'details': {'punctuation': 'exclamation'}}
        
        if '?' in text:
            return {'emotion': 'confused', 'confidence': 0.4, 'details': {'punctuation': 'question'}}
        
        # Default to neutral
        return {'emotion': 'neutral', 'confidence': 0.3, 'details': {}}
    
    def _words_are_close(self, text: str, word1: str, word2: str, max_distance: int = 3) -> bool:
        """
        Check if two words appear close to each other in text
        """
        words = text.split()
        
        try:
            pos1 = next(i for i, word in enumerate(words) if word1 in word)
            pos2 = next(i for i, word in enumerate(words) if word2 in word)
            return abs(pos1 - pos2) <= max_distance
        except StopIteration:
            return False
    
    def get_emotion_insights(self, text: str) -> Dict[str, Any]:
        """
        Get detailed emotion analysis with insights
        """
        emotion_result = self.detect_emotion(text)
        
        insights = {
            'primary_emotion': emotion_result['emotion'],
            'confidence': emotion_result['confidence'],
            'emotional_intensity': self._calculate_intensity(text),
            'emotional_stability': self._assess_stability(text),
            'recommendations': self._get_response_recommendations(emotion_result['emotion'])
        }
        
        return insights
    
    def _calculate_intensity(self, text: str) -> str:
        """
        Calculate emotional intensity based on text features
        """
        text_lower = text.lower()
        
        # Count intensity indicators
        intensity_score = 0
        
        # Caps lock
        if any(c.isupper() for c in text):
            intensity_score += 1
        
        # Multiple exclamation marks
        if '!!' in text:
            intensity_score += 2
        elif '!' in text:
            intensity_score += 1
        
        # Strong words
        strong_words = ['very', 'extremely', 'really', 'absolutely', 'completely', 'totally']
        for word in strong_words:
            if word in text_lower:
                intensity_score += 1
        
        # Determine intensity level
        if intensity_score >= 3:
            return 'high'
        elif intensity_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _assess_stability(self, text: str) -> str:
        """
        Assess emotional stability based on text patterns
        """
        emotion_result = self.detect_emotion(text)
        details = emotion_result.get('details', {})
        
        # Check for mixed emotions
        if len(details) > 2:
            return 'unstable'
        elif len(details) == 2:
            return 'mixed'
        else:
            return 'stable'
    
    def _get_response_recommendations(self, emotion: str) -> Dict[str, str]:
        """
        Get recommendations for responding to specific emotions
        """
        recommendations = {
            'happy': {
                'tone': 'enthusiastic and supportive',
                'approach': 'Share in their positivity and build on their good mood'
            },
            'sad': {
                'tone': 'empathetic and gentle',
                'approach': 'Offer comfort and understanding, avoid being overly cheerful'
            },
            'angry': {
                'tone': 'calm and understanding',
                'approach': 'Acknowledge their frustration and help find solutions'
            },
            'anxious': {
                'tone': 'reassuring and patient',
                'approach': 'Provide calm guidance and break down complex issues'
            },
            'confused': {
                'tone': 'clear and helpful',
                'approach': 'Provide step-by-step explanations and clarifications'
            },
            'surprised': {
                'tone': 'informative and engaging',
                'approach': 'Build on their curiosity and provide interesting details'
            },
            'neutral': {
                'tone': 'friendly and professional',
                'approach': 'Maintain balanced, helpful communication'
            }
        }
        
        return recommendations.get(emotion, recommendations['neutral'])