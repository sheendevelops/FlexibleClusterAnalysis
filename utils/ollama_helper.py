import json
import os
from datetime import datetime
from typing import Dict, Any, List
import ollama

class OllamaHelper:
    """
    Helper class for Ollama local LLM interactions
    """
    
    def __init__(self):
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.1")  # Default to llama3.1
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        
        # Initialize Ollama client
        self.client = ollama.Client(host=self.host)
        
        # Test connection and model availability
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check if the specified model is available"""
        try:
            models = self.client.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.model not in available_models:
                print(f"Warning: Model '{self.model}' not found. Available models: {available_models}")
                if available_models:
                    self.model = available_models[0]
                    print(f"Using model: {self.model}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama server: {e}")
    
    def get_text_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Get a text response from Ollama
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": 1000  # Max tokens to generate
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Unable to get AI response: {str(e)}"
    
    def get_structured_response(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        """
        Get a structured JSON response from Ollama
        """
        try:
            # Add explicit JSON formatting instruction
            json_prompt = f"""{prompt}

IMPORTANT: Respond with valid JSON only. Do not include any text before or after the JSON object."""
            
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that always responds with valid JSON objects only."
                    },
                    {"role": "user", "content": json_prompt}
                ],
                options={
                    "temperature": temperature,
                    "num_predict": 1000
                }
            )
            
            content = response['message']['content'].strip()
            
            # Try to extract JSON if response contains extra text
            if not content.startswith('{'):
                # Look for JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    content = content[start_idx:end_idx]
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {str(e)}", "raw_response": content}
        except Exception as e:
            return {"error": f"API error: {str(e)}"}
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using Ollama
        """
        try:
            prompt = f"""Analyze the sentiment of the following text and provide a rating from 1 to 5 stars and a confidence score between 0 and 1.

Text: {text}

Respond with JSON in this format:
{{"rating": number, "confidence": number, "sentiment": "positive/negative/neutral"}}"""
            
            response = self.get_structured_response(prompt)
            
            if "error" not in response:
                return {
                    "rating": max(1, min(5, round(response.get("rating", 3)))),
                    "confidence": max(0, min(1, response.get("confidence", 0.5))),
                    "sentiment": response.get("sentiment", "neutral")
                }
            else:
                return {
                    "rating": 3,
                    "confidence": 0,
                    "sentiment": "neutral",
                    "error": response["error"]
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
        Summarize long text using Ollama
        """
        try:
            prompt = f"Please summarize the following text in approximately {max_length} words, maintaining key points:\n\n{text}"
            return self.get_text_response(prompt)
        except Exception as e:
            return f"Unable to summarize: {str(e)}"
    
    def detect_topics(self, text: str) -> List[str]:
        """
        Detect main topics in text using Ollama
        """
        try:
            prompt = f"""Identify the main topics discussed in this text. Return only a JSON array of topic strings.

Text: {text}

Respond with JSON format: {{"topics": ["topic1", "topic2", "topic3"]}}"""
            
            response = self.get_structured_response(prompt)
            return response.get("topics", [])
            
        except Exception as e:
            return []
    
    def check_content_appropriateness(self, text: str) -> Dict[str, Any]:
        """
        Check if content is appropriate and safe using Ollama
        """
        try:
            prompt = f"""Analyze this text for content appropriateness. Check for harmful, dangerous, inappropriate, sinister, or malicious content that could harm human wellbeing.

Text: {text}

Respond with JSON:
{{
    "safe": true/false,
    "concerns": ["list of concerns if any"],
    "recommendation": "action recommendation"
}}"""
            
            response = self.get_structured_response(prompt)
            
            if "error" not in response:
                return response
            else:
                return {
                    "safe": True,  # Default to safe if can't analyze
                    "concerns": [],
                    "recommendation": f"Unable to analyze: {response['error']}"
                }
            
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
        Extract key information from text based on focus area using Ollama
        """
        try:
            prompt = f"""Extract key information from this text with focus on: {focus}

Text: {text}

Provide information in JSON format:
{{
    "key_points": ["list of main points"],
    "entities": ["people, places, organizations mentioned"],
    "concepts": ["important concepts or ideas"],
    "actionable_items": ["things that can be acted upon"],
    "questions_raised": ["questions or issues that arise"]
}}"""
            
            response = self.get_structured_response(prompt)
            
            if "error" not in response:
                return response
            else:
                return {
                    "key_points": [],
                    "entities": [],
                    "concepts": [],
                    "actionable_items": [],
                    "questions_raised": [],
                    "error": response["error"]
                }
            
        except Exception as e:
            return {
                "key_points": [],
                "entities": [],
                "concepts": [],
                "actionable_items": [],
                "questions_raised": [],
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        """
        try:
            models = self.client.list()
            current_model = None
            
            for model in models['models']:
                if model['name'] == self.model:
                    current_model = model
                    break
            
            return {
                "model_name": self.model,
                "host": self.host,
                "available": current_model is not None,
                "model_info": current_model
            }
        except Exception as e:
            return {
                "model_name": self.model,
                "host": self.host,
                "available": False,
                "error": str(e)
            }