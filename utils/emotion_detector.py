import re

class EmotionDetector:
    """
    Simple emotion detection based on text analysis
    """
    
    # Emotion keyword mappings
    EMOTION_KEYWORDS = {
        "joy": [
            "happy", "joy", "joyful", "excited", "thrilled", "delighted", "pleased",
            "cheerful", "glad", "elated", "euphoric", "blissful", "content", "satisfied",
            "amazing", "wonderful", "fantastic", "great", "excellent", "awesome"
        ],
        "sadness": [
            "sad", "unhappy", "depressed", "miserable", "sorrowful", "melancholy",
            "down", "blue", "gloomy", "dejected", "despair", "grief", "heartbroken",
            "disappointed", "discouraged", "hopeless", "lonely", "terrible", "awful"
        ],
        "anger": [
            "angry", "mad", "furious", "rage", "irritated", "annoyed", "frustrated",
            "outraged", "livid", "hostile", "resentful", "bitter", "indignant",
            "aggravated", "infuriated", "hate", "disgusted", "stupid", "ridiculous"
        ],
        "fear": [
            "afraid", "scared", "fearful", "terrified", "anxious", "worried", "nervous",
            "panic", "frightened", "alarmed", "concerned", "apprehensive", "uneasy",
            "stressed", "tense", "overwhelmed", "paranoid", "phobic"
        ],
        "surprise": [
            "surprised", "shocked", "amazed", "astonished", "stunned", "bewildered",
            "confused", "perplexed", "puzzled", "baffled", "unexpected", "sudden",
            "wow", "oh", "really", "incredible", "unbelievable"
        ],
        "disgust": [
            "disgusted", "revolted", "repulsed", "sick", "nauseous", "gross",
            "awful", "terrible", "horrible", "nasty", "vile", "repugnant"
        ],
        "love": [
            "love", "adore", "cherish", "treasure", "devoted", "affectionate",
            "caring", "tender", "passionate", "romantic", "fond", "attached"
        ],
        "trust": [
            "trust", "confident", "sure", "certain", "believe", "faith", "reliable",
            "dependable", "secure", "comfortable", "safe"
        ],
        "anticipation": [
            "excited", "eager", "looking forward", "anticipating", "expecting",
            "hopeful", "optimistic", "enthusiastic", "impatient", "can't wait"
        ]
    }
    
    # Emotion intensifiers
    INTENSIFIERS = [
        "very", "extremely", "incredibly", "absolutely", "totally", "completely",
        "really", "quite", "rather", "pretty", "so", "too", "highly", "deeply"
    ]
    
    # Emotion diminishers
    DIMINISHERS = [
        "slightly", "somewhat", "a bit", "a little", "kind of", "sort of",
        "barely", "hardly", "scarcely", "not very", "not really"
    ]
    
    @classmethod
    def detect_emotion(cls, text: str) -> str:
        """
        Detect the primary emotion in text
        """
        if not text:
            return "neutral"
        
        text_lower = text.lower()
        
        # Count emotion matches
        emotion_scores = {}
        
        for emotion, keywords in cls.EMOTION_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of each keyword
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
                if count > 0:
                    # Check for intensifiers/diminishers near the keyword
                    intensity_modifier = cls._get_intensity_modifier(text_lower, keyword)
                    score += count * intensity_modifier
            
            if score > 0:
                emotion_scores[emotion] = score
        
        # Return the emotion with the highest score
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            return primary_emotion
        
        return "neutral"
    
    @classmethod
    def detect_emotions_with_confidence(cls, text: str) -> dict:
        """
        Detect emotions with confidence scores
        """
        if not text:
            return {"neutral": 1.0}
        
        text_lower = text.lower()
        emotion_scores = {}
        total_score = 0
        
        for emotion, keywords in cls.EMOTION_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
                if count > 0:
                    intensity_modifier = cls._get_intensity_modifier(text_lower, keyword)
                    score += count * intensity_modifier
            
            if score > 0:
                emotion_scores[emotion] = score
                total_score += score
        
        # Convert to confidence scores (percentages)
        if total_score > 0:
            confidence_scores = {}
            for emotion, score in emotion_scores.items():
                confidence_scores[emotion] = round(score / total_score, 3)
            return confidence_scores
        
        return {"neutral": 1.0}
    
    @classmethod
    def _get_intensity_modifier(cls, text: str, keyword: str) -> float:
        """
        Get intensity modifier based on nearby intensifiers/diminishers
        """
        # Find the position of the keyword
        keyword_pos = text.find(keyword)
        if keyword_pos == -1:
            return 1.0
        
        # Look for intensifiers/diminishers in a window around the keyword
        window_size = 20  # characters before and after
        start_pos = max(0, keyword_pos - window_size)
        end_pos = min(len(text), keyword_pos + len(keyword) + window_size)
        context = text[start_pos:end_pos]
        
        # Check for intensifiers
        for intensifier in cls.INTENSIFIERS:
            if intensifier in context:
                return 1.5  # Boost the score
        
        # Check for diminishers
        for diminisher in cls.DIMINISHERS:
            if diminisher in context:
                return 0.5  # Reduce the score
        
        return 1.0  # No modification
    
    @classmethod
    def get_emotional_tone(cls, text: str) -> str:
        """
        Get overall emotional tone (positive, negative, neutral)
        """
        emotions = cls.detect_emotions_with_confidence(text)
        
        positive_emotions = ["joy", "love", "trust", "anticipation"]
        negative_emotions = ["sadness", "anger", "fear", "disgust"]
        
        positive_score = sum(emotions.get(emotion, 0) for emotion in positive_emotions)
        negative_score = sum(emotions.get(emotion, 0) for emotion in negative_emotions)
        
        if positive_score > negative_score and positive_score > 0.3:
            return "positive"
        elif negative_score > positive_score and negative_score > 0.3:
            return "negative"
        else:
            return "neutral"
    
    @classmethod
    def analyze_emotional_complexity(cls, text: str) -> dict:
        """
        Analyze the emotional complexity of text
        """
        emotions = cls.detect_emotions_with_confidence(text)
        
        # Remove neutral if other emotions are present
        if len(emotions) > 1 and "neutral" in emotions:
            del emotions["neutral"]
        
        emotional_complexity = {
            "primary_emotion": max(emotions, key=emotions.get) if emotions else "neutral",
            "emotion_count": len(emotions),
            "emotional_intensity": max(emotions.values()) if emotions else 0,
            "emotional_diversity": len(emotions) / len(cls.EMOTION_KEYWORDS),
            "emotions_detected": emotions,
            "overall_tone": cls.get_emotional_tone(text)
        }
        
        return emotional_complexity
