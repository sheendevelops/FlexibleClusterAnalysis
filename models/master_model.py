import json
import os
from datetime import datetime, time
from typing import Dict, List, Any
from models.conscience_model import ConscienceModel
from models.logic_model import LogicModel
from models.personality_model import PersonalityModel
from utils.ollama_helper import OllamaHelper
from utils.content_filter import ContentFilter
from utils.emotion_detector import EmotionDetector
from utils.chatgpt_optimization import ChatGPTOptimizer

class AISapienMaster:
    """
    The Master AISapien model that orchestrates the conscience, logic, and personality models.
    It makes final decisions based on the synthesis of all three models, prioritizing humanity's betterment.
    """
    
    def __init__(self):
        self.conscience_model = ConscienceModel()
        self.logic_model = LogicModel()
        self.personality_model = PersonalityModel()
        self.llm_helper = OllamaHelper()
        self.content_filter = ContentFilter()
        self.chatgpt_optimizer = ChatGPTOptimizer()
        
        self.decisions_file = "data/master_decisions.json"
        self.skills_file = "data/skills_acquired.json"
        self.current_user = "default_user"
        
        self.load_decision_history()
        self.load_skills()
        
        # Core principles for decision making
        self.core_principles = [
            "Prioritize human wellbeing and dignity",
            "Seek truth and accuracy",
            "Promote fairness and justice",
            "Consider long-term consequences",
            "Respect individual autonomy",
            "Foster human growth and learning",
            "Protect vulnerable populations",
            "Encourage ethical behavior"
        ]
    
    def load_decision_history(self):
        """Load master decision history"""
        try:
            with open(self.decisions_file, 'r') as f:
                self.decision_history = json.load(f)
        except FileNotFoundError:
            self.decision_history = []
            self.save_decision_history()
    
    def save_decision_history(self):
        """Save master decision history"""
        os.makedirs(os.path.dirname(self.decisions_file), exist_ok=True)
        with open(self.decisions_file, 'w') as f:
            json.dump(self.decision_history, f, indent=2)
    
    def load_skills(self):
        """Load acquired skills"""
        try:
            with open(self.skills_file, 'r') as f:
                self.skills = json.load(f)
        except FileNotFoundError:
            self.skills = {
                "technical_skills": [],
                "knowledge_domains": [],
                "communication_skills": [],
                "analytical_skills": [],
                "creative_skills": []
            }
            self.save_skills()
    
    def save_skills(self):
        """Save acquired skills"""
        os.makedirs(os.path.dirname(self.skills_file), exist_ok=True)
        with open(self.skills_file, 'w') as f:
            json.dump(self.skills, f, indent=2)
    
    def set_current_user(self, user_id: str):
        """Set the current user for personalized interactions"""
        self.current_user = user_id
    
    def process_chat_message(self, message: str, user_id: str) -> Dict[str, Any]:
        """
        Process a chat message through all models and provide integrated response
        """
        try:
            # Set current user
            self.set_current_user(user_id)
            
            # Content filtering first
            filter_result = self.content_filter.check_content(message)
            if not filter_result['safe']:
                return {
                    "message": f"I notice this topic might be concerning: {filter_result['reason']}. Could you confirm if you'd like me to proceed with this discussion? I want to make sure I'm being helpful while staying focused on positive contributions.",
                    "model_insights": {
                        "content_filter": filter_result['reason']
                    }
                }
            
            # Detect emotion
            emotion = EmotionDetector.detect_emotion(message)
            
            # Update user personality profile
            self.personality_model.create_or_update_user_profile(user_id, message)
            
            # Get insights from all three models
            model_insights = self.get_model_insights(message, user_id)
            
            # Synthesize response using master decision-making
            final_response = self.synthesize_response(message, model_insights, user_id, emotion)
            
            # Record this decision
            self.record_decision(message, model_insights, final_response, user_id)
            
            # Check if this interaction taught us something new
            self.learn_from_interaction(message, final_response)
            
            return {
                "message": final_response,
                "model_insights": model_insights,
                "detected_emotion": emotion,
                "time_awareness": self.get_time_of_day()
            }
            
        except Exception as e:
            return {
                "message": f"I apologize, but I'm experiencing some difficulty processing your message right now. Error: {str(e)}. Could you please try rephrasing your question?",
                "error": str(e)
            }
    
    def get_model_insights(self, message: str, user_id: str) -> Dict[str, str]:
        """
        Get insights from all three models
        """
        insights = {}
        
        try:
            # Conscience model insight (focus on ethics and humanity)
            conscience_prompt = f"From an ethical and humanitarian perspective, how should I respond to: '{message}'"
            insights['conscience'] = self.conscience_model.provide_ethical_guidance(conscience_prompt, f"User: {user_id}")
        except Exception as e:
            insights['conscience'] = f"Conscience model unavailable: {str(e)}"
        
        try:
            # Logic model insight (focus on efficiency and reasoning)
            logic_prompt = f"From a logical and efficiency perspective, what's the best way to address: '{message}'"
            insights['logic'] = self.logic_model.provide_logical_solution(logic_prompt)
        except Exception as e:
            insights['logic'] = f"Logic model unavailable: {str(e)}"
        
        try:
            # Personality model insight (focus on personalization)
            insights['personality'] = self.personality_model.generate_personalized_response(user_id, message)
        except Exception as e:
            insights['personality'] = f"Personality model unavailable: {str(e)}"
        
        return insights
    
    def synthesize_response(self, message: str, model_insights: Dict[str, str], user_id: str, emotion: str) -> str:
        """
        Synthesize insights from all models into a final response prioritizing humanity's betterment
        """
        try:
            # Get user's response style preferences
            user_style = self.personality_model.get_personalized_response_style(user_id)
            
            # Consider time of day and human factors
            time_context = self.get_contextual_time_advice()
            
            prompt = f"""
            As AISapien, a master AI focused on humanity's betterment, synthesize these insights into a response:
            
            User Message: {message}
            Detected Emotion: {emotion}
            Time Context: {time_context}
            
            Model Insights:
            - Conscience (Ethics/Humanity): {model_insights.get('conscience', 'N/A')}
            - Logic (Efficiency/Reasoning): {model_insights.get('logic', 'N/A')}
            - Personality (Personal): {model_insights.get('personality', 'N/A')}
            
            User Preferences:
            - Communication Style: {user_style.get('communication_style', {})}
            - Interests: {user_style.get('interests', [])}
            - Current Needs: {user_style.get('current_needs', [])}
            
            Core Principles to prioritize:
            {chr(10).join([f"- {principle}" for principle in self.core_principles])}
            
            Instructions:
            1. Prioritize human wellbeing and ethical considerations
            2. Incorporate logical efficiency where it aligns with ethics
            3. Personalize the response to the user's style and needs
            4. Consider the user's emotional state
            5. Provide actionable, helpful guidance
            6. Be thoughtful about timing and human factors
            7. Show empathy and understanding
            8. Offer constructive, positive perspectives
            
            Provide a response that represents the best synthesis of all models while prioritizing humanity's betterment.
            """
            
            response = self.llm_helper.get_text_response(prompt)
            
            # Apply ChatGPT-style optimizations
            optimized_response = self.chatgpt_optimizer.optimize_response_structure(
                response, {
                    'emotion': emotion,
                    'user_message': message,
                    'conversation_history': []
                }
            )
            
            # Apply contextual memory enhancement
            enhanced_response = self.chatgpt_optimizer.enhance_with_contextual_memory(
                optimized_response, user_id, {
                    'user_message': message,
                    'emotion': emotion
                }
            )
            
            # Apply advanced reasoning synthesis
            final_response = self.chatgpt_optimizer.apply_advanced_reasoning(
                enhanced_response, model_insights
            )
            
            return final_response
            
        except Exception as e:
            # Fallback response
            return f"I appreciate you sharing that with me. While I'm having some technical difficulties right now ({str(e)}), I want you to know that I'm here to help. Could you tell me more about what you're looking for, and I'll do my best to assist you?"
    
    def record_decision(self, input_message: str, model_insights: Dict[str, str], final_response: str, user_id: str):
        """
        Record a decision for learning and future reference
        """
        try:
            decision = {
                "timestamp": self.llm_helper.get_current_timestamp(),
                "user_id": user_id,
                "input_message": input_message,
                "model_insights": model_insights,
                "final_response": final_response,
                "decision_factors": self.core_principles,
                "time_of_day": self.get_time_of_day()
            }
            
            self.decision_history.append(decision)
            
            # Keep only last 1000 decisions to manage storage
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-1000:]
            
            self.save_decision_history()
            
        except Exception as e:
            print(f"Error recording decision: {e}")
    
    def learn_from_interaction(self, message: str, response: str):
        """
        Learn new skills or knowledge from an interaction
        """
        try:
            prompt = f"""
            Analyze this interaction to identify any new skills or knowledge that should be recorded:
            
            User Message: {message}
            AI Response: {response}
            
            Identify any:
            - New technical skills demonstrated
            - New knowledge domains explored
            - Communication techniques used
            - Analytical methods applied
            - Creative approaches taken
            
            Respond with JSON containing:
            - technical_skills: list of technical skills used/learned
            - knowledge_domains: list of knowledge areas touched
            - communication_skills: list of communication techniques
            - analytical_skills: list of analytical methods
            - creative_skills: list of creative approaches
            """
            
            new_skills = self.llm_helper.get_structured_response(prompt)
            
            # Add new skills to our collection
            for category, skills in new_skills.items():
                if category in self.skills and isinstance(skills, list):
                    for skill in skills:
                        if skill not in self.skills[category]:
                            self.skills[category].append(skill)
            
            self.save_skills()
            
        except Exception as e:
            print(f"Error learning from interaction: {e}")
    
    def learn_from_text(self, text: str, source: str = "Unknown") -> Dict[str, Any]:
        """
        Learn from provided text content (articles, documents, etc.)
        """
        try:
            # Check content appropriateness
            filter_result = self.content_filter.check_content(text[:1000])  # Check first 1000 chars
            if not filter_result['safe']:
                return {
                    "success": False,
                    "message": f"Content appears to contain concerning material: {filter_result['reason']}. Would you like me to proceed anyway?"
                }
            
            # Extract key insights from the text
            prompt = f"""
            Analyze this text for learning opportunities:
            
            Source: {source}
            Text: {text[:2000]}...  (truncated for analysis)
            
            Extract insights in JSON format:
            - key_concepts: list of important concepts
            - ethical_insights: insights related to human wellbeing
            - logical_patterns: logical or analytical patterns
            - practical_applications: how this knowledge could be applied
            - knowledge_domain: what field this knowledge belongs to
            - potential_skills: skills that could be developed from this knowledge
            """
            
            insights = self.llm_helper.get_structured_response(prompt)
            
            # Store knowledge in appropriate models
            if insights.get('ethical_insights'):
                # Update conscience model
                self.conscience_model.knowledge['ethical_cases'].append({
                    "source": source,
                    "insights": insights['ethical_insights'],
                    "timestamp": self.llm_helper.get_current_timestamp()
                })
                self.conscience_model.save_knowledge()
            
            if insights.get('logical_patterns'):
                # Update logic model
                self.logic_model.knowledge['analysis_cases'].append({
                    "source": source,
                    "patterns": insights['logical_patterns'],
                    "timestamp": self.llm_helper.get_current_timestamp()
                })
                self.logic_model.save_knowledge()
            
            # Update skills
            if insights.get('potential_skills'):
                for skill in insights['potential_skills']:
                    if skill not in self.skills.get('knowledge_domains', []):
                        self.skills['knowledge_domains'].append(skill)
            
            if insights.get('knowledge_domain'):
                if insights['knowledge_domain'] not in self.skills.get('knowledge_domains', []):
                    self.skills['knowledge_domains'].append(insights['knowledge_domain'])
            
            self.save_skills()
            
            return {
                "success": True,
                "message": f"Successfully learned from {source}. Acquired knowledge in: {insights.get('knowledge_domain', 'general knowledge')}",
                "insights": insights
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error learning from text: {str(e)}"
            }
    
    def get_time_of_day(self) -> str:
        """
        Get current time of day for context-aware responses
        """
        try:
            current_time = datetime.now().time()
            
            if time(5, 0) <= current_time < time(12, 0):
                return "Morning"
            elif time(12, 0) <= current_time < time(17, 0):
                return "Afternoon"
            elif time(17, 0) <= current_time < time(21, 0):
                return "Evening"
            else:
                return "Night"
                
        except Exception:
            return "Unknown"
    
    def get_contextual_time_advice(self) -> str:
        """
        Get advice based on time of day and human patterns
        """
        time_of_day = self.get_time_of_day()
        
        advice_map = {
            "Morning": "This is a great time for planning and starting new tasks. Energy levels are typically high.",
            "Afternoon": "Good time for collaborative work and tackling complex problems. Consider taking breaks as needed.",
            "Evening": "Time to wind down. Consider lighter activities and reflection on the day's progress.",
            "Night": "Rest is important for wellbeing. Consider if this task is urgent or can wait until tomorrow."
        }
        
        return advice_map.get(time_of_day, "Consider your energy levels and wellbeing when planning activities.")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        """
        try:
            conscience_summary = self.conscience_model.get_conscience_summary()
            logic_summary = self.logic_model.get_logic_summary()
            personality_summary = self.personality_model.get_personality_summary()
            
            all_skills = []
            for skill_list in self.skills.values():
                all_skills.extend(skill_list)
            
            return {
                "knowledge_areas": len(self.skills.get('knowledge_domains', [])),
                "skills_count": len(all_skills),
                "users_count": personality_summary.get('total_users', 0),
                "decisions_made": len(self.decision_history),
                "recent_skills": all_skills,
                "time_awareness": self.get_time_of_day(),
                "model_status": {
                    "conscience": conscience_summary,
                    "logic": logic_summary,
                    "personality": personality_summary
                }
            }
            
        except Exception as e:
            return {
                "knowledge_areas": 0,
                "skills_count": 0,
                "users_count": 0,
                "decisions_made": 0,
                "recent_skills": [],
                "time_awareness": "Unknown",
                "error": str(e)
            }
    
    def get_knowledge_summary(self) -> Dict[str, List[str]]:
        """
        Get summary of all acquired knowledge
        """
        try:
            return {
                "technical_skills": self.skills.get('technical_skills', []),
                "knowledge_domains": self.skills.get('knowledge_domains', []),
                "communication_skills": self.skills.get('communication_skills', []),
                "analytical_skills": self.skills.get('analytical_skills', []),
                "creative_skills": self.skills.get('creative_skills', [])
            }
        except Exception:
            return {}
    
    def get_acquired_skills(self) -> List[str]:
        """
        Get all acquired skills as a flat list
        """
        try:
            all_skills = []
            for skill_list in self.skills.values():
                all_skills.extend(skill_list)
            return all_skills
        except Exception:
            return []
    
    def reset_knowledge_base(self):
        """
        Reset all knowledge bases (use with caution)
        """
        try:
            # Reset skills
            self.skills = {
                "technical_skills": [],
                "knowledge_domains": [],
                "communication_skills": [],
                "analytical_skills": [],
                "creative_skills": []
            }
            self.save_skills()
            
            # Reset decision history
            self.decision_history = []
            self.save_decision_history()
            
            # Reset model knowledge
            self.conscience_model.knowledge = {
                "ethical_cases": [],
                "moral_principles": self.conscience_model.ethical_principles,
                "human_impact_assessments": []
            }
            self.conscience_model.save_knowledge()
            
            self.logic_model.knowledge = {
                "analysis_cases": [],
                "logical_frameworks": self.logic_model.logical_frameworks,
                "optimization_strategies": [],
                "performance_metrics": []
            }
            self.logic_model.save_knowledge()
            
        except Exception as e:
            print(f"Error resetting knowledge base: {e}")
