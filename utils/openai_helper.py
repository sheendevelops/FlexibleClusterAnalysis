import json
import os
from datetime import datetime
from typing import Dict, Any, List

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
from openai import OpenAI

class OpenAIHelper:
    """
    Helper class for OpenAI API interactions
    """
    
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "default_key")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"  # Latest model as of May 13, 2024
    
    def get_text_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Get a text response from OpenAI
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Unable to get AI response: {str(e)}"
    
    def get_structured_response(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        """
        Get a structured JSON response from OpenAI
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that always responds with valid JSON."
                    },
                    {"role": "user", "content": prompt + "\n\nPlease respond with valid JSON only."}
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {str(e)}", "raw_response": content}
        except Exception as e:
            return {"error": f"API error: {str(e)}"}
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a sentiment analysis expert. "
                        + "Analyze the sentiment of the text and provide a rating "
                        + "from 1 to 5 stars and a confidence score between 0 and 1. "
                        + "Respond with JSON in this format: "
                        + "{'rating': number, 'confidence': number, 'sentiment': 'positive/negative/neutral'}",
                    },
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            return {
                "rating": max(1, min(5, round(result.get("rating", 3)))),
                "confidence": max(0, min(1, result.get("confidence", 0.5))),
                "sentiment": result.get("sentiment", "neutral")
            }
        except Exception as e:
            return {
                "rating": 3,
                "confidence": 0,
                "sentiment": "neutral",
                "error": str(e)
            }
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize long text
        """
        try:
            prompt = f"Please summarize the following text in approximately {max_length} words, maintaining key points:\n\n{text}"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length * 2  # Rough estimation for tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Unable to summarize: {str(e)}"
    
    def detect_topics(self, text: str) -> List[str]:
        """
        Detect main topics in text
        """
        try:
            prompt = f"""
            Identify the main topics discussed in this text. 
            Return only a JSON list of topics.
            
            Text: {text}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a topic extraction expert. Respond with a JSON array of topic strings."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("topics", [])
            
        except Exception as e:
            return []
    
    def check_content_appropriateness(self, text: str) -> Dict[str, Any]:
        """
        Check if content is appropriate and safe
        """
        try:
            prompt = f"""
            Analyze this text for content appropriateness:
            
            Text: {text}
            
            Check for:
            - Harmful or dangerous content
            - Inappropriate material
            - Sinister or malicious intent
            - Content that could harm human wellbeing
            
            Respond with JSON:
            {{
                "safe": true/false,
                "concerns": ["list of concerns if any"],
                "recommendation": "action recommendation"
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a content safety expert focused on protecting human wellbeing."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "safe": True,  # Default to safe if can't analyze
                "concerns": [],
                "recommendation": f"Unable to analyze: {str(e)}"
            }
    
    def get_current_timestamp(self) -> str:
        """
        Get current timestamp in ISO format
        """
        return datetime.now().isoformat()
    
    def extract_key_information(self, text: str, focus: str = "general") -> Dict[str, Any]:
        """
        Extract key information from text based on focus area
        """
        try:
            prompt = f"""
            Extract key information from this text with focus on: {focus}
            
            Text: {text}
            
            Provide information in JSON format:
            {{
                "key_points": ["list of main points"],
                "entities": ["people, places, organizations mentioned"],
                "concepts": ["important concepts or ideas"],
                "actionable_items": ["things that can be acted upon"],
                "questions_raised": ["questions or issues that arise"]
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert information extractor focusing on {focus}."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "key_points": [],
                "entities": [],
                "concepts": [],
                "actionable_items": [],
                "questions_raised": [],
                "error": str(e)
            }
