import json
import os
from typing import Dict, List, Any
from utils.openai_helper import OpenAIHelper
from utils.database_helper import db

class ConscienceModel:
    """
    The Conscience Model focuses on human ethics, empathy, and humanitarian values.
    It evaluates decisions based on their impact on human wellbeing and ethical considerations.
    """
    
    def __init__(self):
        self.openai_helper = OpenAIHelper()
        self.load_ethical_principles()
    
    def load_ethical_principles(self):
        """Load ethical principles from database"""
        try:
            principles = db.find_many('ethical_principles', order_by='id')
            self.ethical_principles = [p['principle'] for p in principles]
        except Exception as e:
            # Fallback to default principles if database fails
            self.ethical_principles = [
                "Human dignity and respect",
                "Fairness and justice",
                "Compassion and empathy",
                "Non-maleficence (do no harm)",
                "Autonomy and freedom",
                "Truth and honesty",
                "Protection of vulnerable populations",
                "Environmental responsibility"
            ]
    
    def analyze_ethical_implications(self, scenario: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze a scenario for ethical implications and humanitarian impact
        """
        try:
            prompt = f"""
            As an ethical advisor focused on human wellbeing and humanitarian values, 
            analyze the following scenario for ethical implications:
            
            Scenario: {scenario}
            Context: {context}
            
            Please evaluate based on these ethical principles:
            {', '.join(self.ethical_principles)}
            
            Provide your analysis in JSON format with:
            - ethical_score: number from 1-10 (10 being most ethical)
            - humanitarian_impact: string describing impact on human wellbeing
            - ethical_concerns: list of potential ethical issues
            - recommendations: list of suggestions for ethical improvement
            - affected_groups: list of groups that might be affected
            """
            
            response = self.openai_helper.get_structured_response(prompt)
            
            # Store this analysis in database
            try:
                db.insert_one('conscience_knowledge', {
                    'scenario': scenario,
                    'context': context,
                    'analysis': response,
                    'ethical_score': response.get('ethical_score', 5),
                    'humanitarian_impact': response.get('humanitarian_impact', 'Unknown'),
                    'source': 'ethical_analysis'
                })
            except Exception as e:
                print(f"Error storing ethical analysis: {e}")
            
            return response
            
        except Exception as e:
            return {
                "ethical_score": 5,
                "humanitarian_impact": f"Unable to analyze due to error: {str(e)}",
                "ethical_concerns": ["Analysis failed"],
                "recommendations": ["Seek human ethical guidance"],
                "affected_groups": ["Unknown"]
            }
    
    def provide_ethical_guidance(self, question: str, user_context: str = "") -> str:
        """
        Provide ethical guidance for a specific question or dilemma
        """
        try:
            # Include relevant past cases in the context
            relevant_cases = self.get_relevant_ethical_cases(question)
            cases_context = ""
            if relevant_cases:
                cases_context = "\n\nRelevant past ethical considerations:\n"
                for case in relevant_cases[-3:]:  # Use last 3 relevant cases
                    cases_context += f"- {case['analysis'].get('humanitarian_impact', 'N/A')}\n"
            
            prompt = f"""
            As a compassionate ethical advisor prioritizing human wellbeing and dignity,
            provide guidance for this question:
            
            Question: {question}
            User Context: {user_context}
            {cases_context}
            
            Ethical Principles to consider:
            {', '.join(self.ethical_principles)}
            
            Please provide:
            1. A compassionate response that prioritizes human wellbeing
            2. Consideration of how this affects vulnerable populations
            3. Long-term humanitarian implications
            4. Suggestions that align with human dignity and respect
            
            Focus on empathy, care, and the betterment of humanity.
            """
            
            response = self.openai_helper.get_text_response(prompt)
            return response
            
        except Exception as e:
            return f"I'm having difficulty providing ethical guidance right now: {str(e)}. However, I encourage you to consider the impact on human wellbeing and dignity in whatever decision you're facing."
    
    def get_relevant_ethical_cases(self, query: str) -> List[Dict]:
        """
        Find ethical cases relevant to the current query
        """
        try:
            # Search for relevant cases in database
            relevant_cases = db.search_text('conscience_knowledge', 'scenario', query, limit=5)
            return relevant_cases
        except Exception:
            return []
    
    def assess_human_impact(self, action: str, context: str = "") -> Dict[str, Any]:
        """
        Assess the potential impact of an action on human wellbeing
        """
        try:
            prompt = f"""
            Assess the human impact of this action:
            
            Action: {action}
            Context: {context}
            
            Consider:
            - Direct effects on individuals
            - Impact on communities
            - Long-term consequences for human wellbeing
            - Effects on vulnerable populations
            - Emotional and psychological impacts
            
            Respond in JSON format with:
            - impact_score: number from -10 to 10 (-10 very harmful, 10 very beneficial)
            - affected_populations: list of groups affected
            - short_term_effects: string describing immediate impacts
            - long_term_effects: string describing future impacts
            - mitigation_strategies: list of ways to reduce negative impacts
            """
            
            response = self.openai_helper.get_structured_response(prompt)
            
            # Store assessment
            self.knowledge["human_impact_assessments"].append({
                "action": action,
                "context": context,
                "assessment": response,
                "timestamp": self.openai_helper.get_current_timestamp()
            })
            self.save_knowledge()
            
            return response
            
        except Exception as e:
            return {
                "impact_score": 0,
                "affected_populations": ["Unknown"],
                "short_term_effects": f"Unable to assess: {str(e)}",
                "long_term_effects": "Uncertain",
                "mitigation_strategies": ["Proceed with caution", "Seek expert guidance"]
            }
    
    def update_ethical_principles(self, new_principle: str):
        """
        Add a new ethical principle to consider
        """
        if new_principle not in self.ethical_principles:
            self.ethical_principles.append(new_principle)
            self.knowledge["moral_principles"] = self.ethical_principles
            self.save_knowledge()
    
    def get_conscience_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conscience model's knowledge and focus areas
        """
        return {
            "ethical_principles": len(self.ethical_principles),
            "cases_analyzed": len(self.knowledge.get("ethical_cases", [])),
            "impact_assessments": len(self.knowledge.get("human_impact_assessments", [])),
            "focus_areas": [
                "Human dignity and wellbeing",
                "Ethical decision making",
                "Vulnerable population protection",
                "Long-term humanitarian impact"
            ]
        }
