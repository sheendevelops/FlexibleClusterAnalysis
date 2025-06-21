import re
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class ChatGPTOptimizer:
    """
    Optimization strategies to make AISapien as effective as ChatGPT
    """
    
    def __init__(self):
        self.response_templates = self._load_response_templates()
        self.conversation_context = {}
        
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """
        Load response templates optimized for different scenarios
        """
        return {
            "greeting": [
                "Hello! I'm AISapien, your ethical AI assistant. How can I help you today?",
                "Hi there! I'm here to assist you with thoughtful, well-reasoned responses. What would you like to explore?",
                "Welcome! I'm AISapien, combining ethical reasoning, logical analysis, and personalized assistance. What can I do for you?"
            ],
            "clarification": [
                "I want to make sure I understand correctly. Could you clarify...",
                "To provide the most helpful response, could you tell me more about...",
                "I'd like to better understand your specific needs. Could you elaborate on..."
            ],
            "explanation": [
                "Let me break this down into clear steps:",
                "Here's a comprehensive explanation:",
                "I'll explain this from multiple perspectives:"
            ],
            "problem_solving": [
                "Let's approach this systematically:",
                "I'll analyze this problem using multiple frameworks:",
                "Here's a structured approach to solving this:"
            ],
            "empathy": [
                "I understand this might be challenging for you.",
                "It sounds like this is important to you, and I want to help.",
                "I can see why this would be concerning."
            ]
        }
    
    def optimize_response_structure(self, raw_response: str, context: Dict[str, Any]) -> str:
        """
        Optimize response structure to match ChatGPT's effectiveness
        """
        # Analyze context for optimization strategy
        user_emotion = context.get('emotion', 'neutral')
        user_intent = self._detect_user_intent(context.get('user_message', ''))
        conversation_history = context.get('conversation_history', [])
        
        # Apply ChatGPT-style optimizations
        optimized_response = self._apply_structure_optimization(
            raw_response, user_emotion, user_intent, conversation_history
        )
        
        return optimized_response
    
    def _detect_user_intent(self, user_message: str) -> str:
        """
        Detect user intent to optimize response approach
        """
        message_lower = user_message.lower()
        
        intent_patterns = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', '?'],
            'request_help': ['help', 'assist', 'support', 'guidance', 'advice'],
            'explanation': ['explain', 'clarify', 'understand', 'tell me about'],
            'problem_solving': ['problem', 'issue', 'solve', 'fix', 'troubleshoot'],
            'information': ['information', 'details', 'facts', 'data'],
            'creative': ['create', 'generate', 'write', 'design', 'brainstorm'],
            'analysis': ['analyze', 'compare', 'evaluate', 'assess', 'review'],
            'conversation': ['chat', 'talk', 'discuss', 'conversation']
        }
        
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in message_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores.keys(), key=lambda x: intent_scores[x])
        else:
            return 'general'
    
    def _apply_structure_optimization(self, response: str, emotion: str, 
                                    intent: str, history: List[Dict]) -> str:
        """
        Apply ChatGPT-style structural optimizations
        """
        # 1. Add empathetic opening if needed
        if emotion in ['sad', 'frustrated', 'anxious', 'angry']:
            empathy_opener = self._get_empathy_opener(emotion)
            response = f"{empathy_opener}\n\n{response}"
        
        # 2. Improve structure based on intent
        if intent == 'explanation':
            response = self._structure_as_explanation(response)
        elif intent == 'problem_solving':
            response = self._structure_as_problem_solving(response)
        elif intent == 'question':
            response = self._structure_as_answer(response)
        
        # 3. Add follow-up questions for engagement
        follow_up = self._generate_follow_up(intent, response)
        if follow_up:
            response = f"{response}\n\n{follow_up}"
        
        # 4. Optimize formatting
        response = self._optimize_formatting(response)
        
        return response
    
    def _get_empathy_opener(self, emotion: str) -> str:
        """
        Get appropriate empathetic opening based on emotion
        """
        empathy_map = {
            'sad': "I understand you're going through a difficult time.",
            'frustrated': "I can sense your frustration, and I want to help.",
            'anxious': "I understand this might be causing you some worry.",
            'angry': "I can see this is really bothering you.",
            'confused': "I understand this can be confusing."
        }
        
        return empathy_map.get(emotion, "I'm here to help you with this.")
    
    def _structure_as_explanation(self, response: str) -> str:
        """
        Structure response as a clear explanation
        """
        # Add clear headers and organization
        if len(response.split('\n')) < 3:
            # Simple response - add structure
            sentences = response.split('. ')
            if len(sentences) > 3:
                structured = "Here's a clear explanation:\n\n"
                
                # Group sentences into logical sections
                for i, sentence in enumerate(sentences, 1):
                    structured += f"{i}. {sentence.strip()}\n"
                
                return structured
        
        return response
    
    def _structure_as_problem_solving(self, response: str) -> str:
        """
        Structure response as problem-solving approach
        """
        if "step" not in response.lower():
            # Add step-by-step structure
            sentences = response.split('. ')
            if len(sentences) > 2:
                structured = "Here's a systematic approach to solve this:\n\n"
                
                for i, sentence in enumerate(sentences, 1):
                    if sentence.strip():
                        structured += f"**Step {i}:** {sentence.strip()}\n\n"
                
                return structured
        
        return response
    
    def _structure_as_answer(self, response: str) -> str:
        """
        Structure response as a direct answer
        """
        # Ensure direct answer comes first
        if not response.strip().startswith(("Yes", "No", "The answer", "It is", "It depends")):
            # Add a direct opener
            return f"To answer your question directly:\n\n{response}"
        
        return response
    
    def _generate_follow_up(self, intent: str, response: str) -> Optional[str]:
        """
        Generate appropriate follow-up questions
        """
        follow_ups = {
            'explanation': "Would you like me to elaborate on any particular aspect?",
            'problem_solving': "Would you like help implementing any of these steps?",
            'question': "Does this answer your question, or would you like me to clarify anything?",
            'information': "Is there any specific aspect you'd like me to explore further?",
            'analysis': "Would you like me to analyze this from a different perspective?",
            'creative': "Would you like me to develop any of these ideas further?"
        }
        
        return follow_ups.get(intent)
    
    def _optimize_formatting(self, response: str) -> str:
        """
        Optimize text formatting for readability
        """
        # Add markdown formatting
        response = re.sub(r'\n([0-9]+\.)', r'\n**\1**', response)  # Bold numbered lists
        response = re.sub(r'\n(-|\*) ', r'\n• ', response)  # Clean bullet points
        
        # Ensure proper spacing
        response = re.sub(r'\n{3,}', '\n\n', response)  # Remove excessive line breaks
        
        # Add emphasis to key terms
        response = self._add_emphasis_to_key_terms(response)
        
        return response.strip()
    
    def _add_emphasis_to_key_terms(self, text: str) -> str:
        """
        Add emphasis to key terms for better readability
        """
        key_terms = [
            'important', 'crucial', 'essential', 'key', 'note that',
            'remember', 'consider', 'however', 'therefore', 'in conclusion'
        ]
        
        for term in key_terms:
            pattern = rf'\b{re.escape(term)}\b'
            replacement = f'**{term}**'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def enhance_with_contextual_memory(self, response: str, user_id: str, 
                                     conversation_context: Dict[str, Any]) -> str:
        """
        Enhance response with contextual memory like ChatGPT
        """
        # Store conversation context
        if user_id not in self.conversation_context:
            self.conversation_context[user_id] = {
                'topics_discussed': [],
                'user_preferences': {},
                'interaction_style': 'formal',
                'expertise_level': 'general'
            }
        
        user_context = self.conversation_context[user_id]
        
        # Update context with current conversation
        current_topic = self._extract_topic(conversation_context.get('user_message', ''))
        if current_topic and current_topic not in user_context['topics_discussed']:
            user_context['topics_discussed'].append(current_topic)
        
        # Adjust response based on context
        if len(user_context['topics_discussed']) > 1:
            # Reference previous topics when relevant
            related_topics = self._find_related_topics(current_topic, user_context['topics_discussed'])
            if related_topics:
                context_reference = f"Building on our previous discussion about {', '.join(related_topics)}, "
                response = f"{context_reference}{response.lower()[0]}{response[1:]}"
        
        return response
    
    def _extract_topic(self, message: str) -> Optional[str]:
        """
        Extract main topic from user message
        """
        topics = {
            'technology': ['ai', 'computer', 'software', 'programming', 'tech'],
            'health': ['health', 'medical', 'doctor', 'medicine', 'wellness'],
            'education': ['learn', 'study', 'school', 'education', 'knowledge'],
            'business': ['business', 'work', 'job', 'career', 'company'],
            'personal': ['personal', 'life', 'relationship', 'family', 'emotion'],
            'science': ['science', 'research', 'theory', 'experiment', 'data']
        }
        
        message_lower = message.lower()
        for topic, keywords in topics.items():
            if any(keyword in message_lower for keyword in keywords):
                return topic
        
        return None
    
    def _find_related_topics(self, current_topic: str, previous_topics: List[str]) -> List[str]:
        """
        Find related topics from previous conversations
        """
        topic_relationships = {
            'technology': ['science', 'business'],
            'health': ['science', 'personal'],
            'education': ['personal', 'business'],
            'business': ['technology', 'education'],
            'personal': ['health', 'education'],
            'science': ['technology', 'health']
        }
        
        related = topic_relationships.get(current_topic, [])
        return [topic for topic in previous_topics if topic in related]
    
    def apply_advanced_reasoning(self, response: str, model_insights: Dict[str, str]) -> str:
        """
        Apply advanced reasoning techniques like ChatGPT
        """
        # Integrate insights from all three models
        conscience_insight = model_insights.get('conscience', '')
        logic_insight = model_insights.get('logic', '')
        personality_insight = model_insights.get('personality', '')
        
        # Create a more sophisticated synthesis
        enhanced_response = self._synthesize_advanced_reasoning(
            response, conscience_insight, logic_insight, personality_insight
        )
        
        return enhanced_response
    
    def _synthesize_advanced_reasoning(self, base_response: str, conscience: str, 
                                     logic: str, personality: str) -> str:
        """
        Synthesize advanced reasoning from multiple perspectives
        """
        # If we have insights from multiple models, create a nuanced response
        if conscience and logic and personality:
            synthesis = f"""Based on a comprehensive analysis considering ethical implications, logical reasoning, and your personal context:

{base_response}

**Ethical Considerations:** {conscience}

**Logical Analysis:** {logic}

**Personalized Perspective:** {personality}

This multi-faceted approach ensures that my response is not only accurate but also ethically sound and tailored to your specific needs."""

            return synthesis
        
        return base_response
    
    def optimize_for_engagement(self, response: str, user_profile: Dict[str, Any]) -> str:
        """
        Optimize response for user engagement like ChatGPT
        """
        engagement_level = user_profile.get('engagement_level', 'medium')
        communication_style = user_profile.get('communication_style', 'balanced')
        
        if engagement_level == 'high':
            # Add more interactive elements
            response = self._add_interactive_elements(response)
        elif engagement_level == 'low':
            # Keep it concise and direct
            response = self._make_concise(response)
        
        if communication_style == 'casual':
            response = self._make_casual(response)
        elif communication_style == 'formal':
            response = self._make_formal(response)
        
        return response
    
    def _add_interactive_elements(self, response: str) -> str:
        """
        Add interactive elements to increase engagement
        """
        # Add thought-provoking questions
        interactive_additions = [
            "\n\nWhat aspects of this interest you most?",
            "\n\nHave you considered how this might apply to your specific situation?",
            "\n\nI'd be curious to hear your thoughts on this approach."
        ]
        
        # Randomly add one interactive element
        import random
        return response + random.choice(interactive_additions)
    
    def _make_concise(self, response: str) -> str:
        """
        Make response more concise for users who prefer brevity
        """
        sentences = response.split('. ')
        # Keep only the most essential sentences
        essential_sentences = sentences[:max(2, len(sentences)//2)]
        return '. '.join(essential_sentences) + '.'
    
    def _make_casual(self, response: str) -> str:
        """
        Adjust tone to be more casual
        """
        # Replace formal phrases with casual ones
        casual_replacements = {
            'Therefore': 'So',
            'However': 'But',
            'Nevertheless': 'Still',
            'Furthermore': 'Also',
            'In conclusion': 'Bottom line'
        }
        
        for formal, casual in casual_replacements.items():
            response = response.replace(formal, casual)
        
        return response
    
    def _make_formal(self, response: str) -> str:
        """
        Adjust tone to be more formal
        """
        # Ensure proper grammar and formal language
        response = response.replace(" 'cause ", " because ")
        response = response.replace(" gonna ", " going to ")
        response = response.replace(" wanna ", " want to ")
        
        return response