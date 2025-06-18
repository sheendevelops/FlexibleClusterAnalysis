import json
import os
from typing import Dict, List, Any
from utils.openai_helper import OpenAIHelper

class PersonalityModel:
    """
    The Personality Model learns and remembers individual users, their preferences,
    communication styles, and provides personalized interactions.
    """
    
    def __init__(self):
        self.openai_helper = OpenAIHelper()
        self.users_file = "data/user_profiles.json"
        self.interactions_file = "data/user_interactions.json"
        self.load_user_data()
    
    def load_user_data(self):
        """Load user profiles and interaction history"""
        try:
            with open(self.users_file, 'r') as f:
                self.user_profiles = json.load(f)
        except FileNotFoundError:
            self.user_profiles = {}
            self.save_user_profiles()
        
        try:
            with open(self.interactions_file, 'r') as f:
                self.interactions = json.load(f)
        except FileNotFoundError:
            self.interactions = {}
            self.save_interactions()
    
    def save_user_profiles(self):
        """Save user profiles to file"""
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        with open(self.users_file, 'w') as f:
            json.dump(self.user_profiles, f, indent=2)
    
    def save_interactions(self):
        """Save interaction history to file"""
        os.makedirs(os.path.dirname(self.interactions_file), exist_ok=True)
        with open(self.interactions_file, 'w') as f:
            json.dump(self.interactions, f, indent=2)
    
    def create_or_update_user_profile(self, user_id: str, message: str, context: str = "") -> Dict[str, Any]:
        """
        Create or update a user's personality profile based on their interactions
        """
        try:
            # Get existing profile or create new one
            existing_profile = self.user_profiles.get(user_id, {
                "preferences": {},
                "communication_style": {},
                "interests": [],
                "needs": [],
                "personality_traits": {},
                "interaction_history": [],
                "last_updated": None
            })
            
            # Analyze the current message for personality insights
            prompt = f"""
            Analyze this user message for personality insights and preferences:
            
            User Message: {message}
            Context: {context}
            
            Existing Profile: {json.dumps(existing_profile.get('personality_traits', {}), indent=2)}
            
            Based on this interaction, provide insights in JSON format:
            - communication_style: object with style preferences (formal/informal, detailed/brief, etc.)
            - interests: list of topics the user seems interested in
            - preferences: object with any preferences mentioned or implied
            - personality_traits: object with observed traits (analytical, creative, practical, etc.)
            - needs: list of what the user might need help with
            - emotional_tone: string describing the user's emotional state
            """
            
            insights = self.openai_helper.get_structured_response(prompt)
            
            # Update profile with new insights
            self.merge_profile_insights(user_id, insights, existing_profile)
            
            # Record this interaction
            self.record_interaction(user_id, message, insights)
            
            return self.user_profiles[user_id]
            
        except Exception as e:
            # Fallback: create basic profile
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "preferences": {},
                    "communication_style": {"tone": "friendly"},
                    "interests": [],
                    "needs": [],
                    "personality_traits": {},
                    "interaction_history": [],
                    "last_updated": self.openai_helper.get_current_timestamp(),
                    "error": str(e)
                }
                self.save_user_profiles()
            
            return self.user_profiles[user_id]
    
    def merge_profile_insights(self, user_id: str, insights: Dict[str, Any], existing_profile: Dict[str, Any]):
        """
        Intelligently merge new insights with existing profile
        """
        try:
            # Update communication style
            existing_style = existing_profile.get("communication_style", {})
            new_style = insights.get("communication_style", {})
            existing_style.update(new_style)
            
            # Add new interests (avoid duplicates)
            existing_interests = set(existing_profile.get("interests", []))
            new_interests = set(insights.get("interests", []))
            combined_interests = list(existing_interests.union(new_interests))
            
            # Update preferences
            existing_prefs = existing_profile.get("preferences", {})
            new_prefs = insights.get("preferences", {})
            existing_prefs.update(new_prefs)
            
            # Update personality traits
            existing_traits = existing_profile.get("personality_traits", {})
            new_traits = insights.get("personality_traits", {})
            existing_traits.update(new_traits)
            
            # Add new needs
            existing_needs = set(existing_profile.get("needs", []))
            new_needs = set(insights.get("needs", []))
            combined_needs = list(existing_needs.union(new_needs))
            
            # Update profile
            self.user_profiles[user_id] = {
                "preferences": existing_prefs,
                "communication_style": existing_style,
                "interests": combined_interests,
                "needs": combined_needs,
                "personality_traits": existing_traits,
                "interaction_history": existing_profile.get("interaction_history", []),
                "last_updated": self.openai_helper.get_current_timestamp(),
                "last_emotional_tone": insights.get("emotional_tone", "neutral")
            }
            
            self.save_user_profiles()
            
        except Exception as e:
            print(f"Error merging profile insights: {e}")
    
    def record_interaction(self, user_id: str, message: str, insights: Dict[str, Any]):
        """
        Record an interaction for future reference
        """
        try:
            if user_id not in self.interactions:
                self.interactions[user_id] = []
            
            interaction = {
                "message": message,
                "insights": insights,
                "timestamp": self.openai_helper.get_current_timestamp()
            }
            
            self.interactions[user_id].append(interaction)
            
            # Keep only last 50 interactions per user to manage storage
            if len(self.interactions[user_id]) > 50:
                self.interactions[user_id] = self.interactions[user_id][-50:]
            
            self.save_interactions()
            
        except Exception as e:
            print(f"Error recording interaction: {e}")
    
    def get_personalized_response_style(self, user_id: str) -> Dict[str, Any]:
        """
        Get the preferred response style for a specific user
        """
        try:
            profile = self.user_profiles.get(user_id, {})
            
            return {
                "communication_style": profile.get("communication_style", {"tone": "friendly"}),
                "interests": profile.get("interests", []),
                "preferences": profile.get("preferences", {}),
                "personality_traits": profile.get("personality_traits", {}),
                "current_needs": profile.get("needs", [])
            }
            
        except Exception:
            return {
                "communication_style": {"tone": "friendly"},
                "interests": [],
                "preferences": {},
                "personality_traits": {},
                "current_needs": []
            }
    
    def generate_personalized_response(self, user_id: str, query: str, context: str = "") -> str:
        """
        Generate a response personalized to the user's style and preferences
        """
        try:
            profile = self.user_profiles.get(user_id, {})
            response_style = self.get_personalized_response_style(user_id)
            
            # Get recent interaction context
            recent_interactions = self.interactions.get(user_id, [])[-5:]  # Last 5 interactions
            interaction_context = ""
            if recent_interactions:
                interaction_context = "\nRecent conversation context:\n"
                for interaction in recent_interactions:
                    interaction_context += f"- User said: {interaction['message'][:100]}...\n"
            
            prompt = f"""
            Provide a personalized response for this user based on their profile:
            
            User Query: {query}
            Context: {context}
            
            User Profile:
            - Communication Style: {json.dumps(response_style['communication_style'])}
            - Interests: {', '.join(response_style['interests'])}
            - Preferences: {json.dumps(response_style['preferences'])}
            - Personality Traits: {json.dumps(response_style['personality_traits'])}
            - Current Needs: {', '.join(response_style['current_needs'])}
            {interaction_context}
            
            Guidelines for personalization:
            1. Match their preferred communication style
            2. Reference their interests when relevant
            3. Consider their personality traits in your tone
            4. Address their specific needs
            5. Build on previous conversations naturally
            6. Be consistent with their preferences
            
            Provide a response that feels tailored specifically to this individual.
            """
            
            response = self.openai_helper.get_text_response(prompt)
            return response
            
        except Exception as e:
            return f"I'd be happy to help you with that. However, I'm having some difficulty accessing your personalization settings right now: {str(e)}. Let me know how you'd prefer me to respond!"
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive insights about a user
        """
        try:
            profile = self.user_profiles.get(user_id, {})
            interactions = self.interactions.get(user_id, [])
            
            # Calculate interaction statistics
            total_interactions = len(interactions)
            recent_activity = len([i for i in interactions if self.is_recent_interaction(i.get('timestamp', ''))])
            
            return {
                "profile_completeness": self.calculate_profile_completeness(profile),
                "total_interactions": total_interactions,
                "recent_activity": recent_activity,
                "primary_interests": profile.get("interests", [])[:5],
                "communication_preferences": profile.get("communication_style", {}),
                "identified_needs": profile.get("needs", []),
                "personality_summary": profile.get("personality_traits", {}),
                "last_interaction": profile.get("last_updated", "Never"),
                "engagement_level": self.calculate_engagement_level(interactions)
            }
            
        except Exception as e:
            return {
                "profile_completeness": 0,
                "total_interactions": 0,
                "recent_activity": 0,
                "primary_interests": [],
                "communication_preferences": {},
                "identified_needs": [],
                "personality_summary": {},
                "last_interaction": "Never",
                "engagement_level": "Unknown",
                "error": str(e)
            }
    
    def calculate_profile_completeness(self, profile: Dict[str, Any]) -> int:
        """
        Calculate how complete a user's profile is (0-100%)
        """
        try:
            fields = ['preferences', 'communication_style', 'interests', 'needs', 'personality_traits']
            completed_fields = 0
            
            for field in fields:
                if profile.get(field) and len(profile[field]) > 0:
                    completed_fields += 1
            
            return int((completed_fields / len(fields)) * 100)
            
        except Exception:
            return 0
    
    def is_recent_interaction(self, timestamp: str) -> bool:
        """
        Check if an interaction happened recently (within last 7 days)
        """
        try:
            from datetime import datetime, timedelta
            
            if not timestamp:
                return False
            
            interaction_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            recent_threshold = datetime.now() - timedelta(days=7)
            
            return interaction_date > recent_threshold
            
        except Exception:
            return False
    
    def calculate_engagement_level(self, interactions: List[Dict]) -> str:
        """
        Calculate user engagement level based on interaction patterns
        """
        try:
            if not interactions:
                return "New User"
            
            total_interactions = len(interactions)
            recent_interactions = len([i for i in interactions if self.is_recent_interaction(i.get('timestamp', ''))])
            
            if total_interactions >= 20 and recent_interactions >= 5:
                return "Highly Engaged"
            elif total_interactions >= 10 and recent_interactions >= 2:
                return "Regularly Engaged"
            elif total_interactions >= 5:
                return "Moderately Engaged"
            else:
                return "New or Casual User"
                
        except Exception:
            return "Unknown"
    
    def get_personality_summary(self) -> Dict[str, Any]:
        """
        Get summary of all users and personality model capabilities
        """
        return {
            "total_users": len(self.user_profiles),
            "total_interactions": sum(len(interactions) for interactions in self.interactions.values()),
            "active_users": len([uid for uid, profile in self.user_profiles.items() 
                               if self.is_recent_interaction(profile.get("last_updated", ""))]),
            "capabilities": [
                "User preference learning",
                "Communication style adaptation",
                "Interest tracking",
                "Personalized responses",
                "Interaction history",
                "Engagement analysis"
            ]
        }
