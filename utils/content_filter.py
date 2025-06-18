import re
from typing import Dict, Any, List

class ContentFilter:
    """
    Content filtering system to identify potentially harmful or inappropriate content
    """
    
    def __init__(self):
        # Keywords that might indicate concerning content
        self.concerning_keywords = [
            # Violence and harm
            "violence", "weapon", "bomb", "explosive", "kill", "murder", "assault",
            "terrorism", "terrorist", "attack", "destroy", "harm", "hurt", "injure",
            
            # Illegal activities
            "illegal", "drug dealing", "trafficking", "fraud", "scam", "theft",
            "hacking", "cybercrime", "money laundering",
            
            # Hate and discrimination
            "hate speech", "discrimination", "racism", "sexism", "extremism",
            "supremacy", "genocide", "ethnic cleansing",
            
            # Self-harm
            "suicide", "self-harm", "cutting", "overdose",
            
            # Explicit content (mild detection)
            "explicit", "pornography", "adult content",
            
            # Dangerous activities
            "dangerous experiment", "unsafe practice", "reckless behavior"
        ]
        
        # Patterns that might indicate problematic content
        self.concerning_patterns = [
            r"how to (?:make|build|create) (?:bomb|weapon|explosive)",
            r"kill (?:someone|people|yourself)",
            r"harm (?:others|children|animals)",
            r"illegal (?:activities|drugs|weapons)",
            r"hack (?:into|someone|system)",
            r"steal (?:money|data|information)"
        ]
        
        # Context that might make concerning content acceptable (educational, etc.)
        self.acceptable_contexts = [
            "educational", "academic", "research", "historical", "documentary",
            "safety", "prevention", "awareness", "medical", "psychological",
            "literary", "fictional", "movie", "book", "story", "novel"
        ]
    
    def check_content(self, text: str) -> Dict[str, Any]:
        """
        Check if content is safe or potentially concerning
        """
        try:
            text_lower = text.lower()
            
            # Check for concerning keywords
            found_keywords = []
            for keyword in self.concerning_keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
            
            # Check for concerning patterns
            found_patterns = []
            for pattern in self.concerning_patterns:
                if re.search(pattern, text_lower):
                    found_patterns.append(pattern)
            
            # Check for acceptable contexts
            acceptable_context = any(context in text_lower for context in self.acceptable_contexts)
            
            # Determine if content is safe
            has_concerns = len(found_keywords) > 0 or len(found_patterns) > 0
            
            if has_concerns and not acceptable_context:
                return {
                    "safe": False,
                    "reason": f"Contains potentially concerning content: {', '.join(found_keywords[:3])}",
                    "found_keywords": found_keywords[:5],  # Limit to prevent overwhelming
                    "found_patterns": found_patterns[:3],
                    "recommendation": "Human review recommended"
                }
            elif has_concerns and acceptable_context:
                return {
                    "safe": True,
                    "reason": "Contains concerning keywords but appears to be in educational/acceptable context",
                    "found_keywords": found_keywords[:5],
                    "context_detected": True,
                    "recommendation": "Proceed with educational focus"
                }
            else:
                return {
                    "safe": True,
                    "reason": "No concerning content detected",
                    "found_keywords": [],
                    "found_patterns": [],
                    "recommendation": "Safe to proceed"
                }
                
        except Exception as e:
            # Default to requiring review if filtering fails
            return {
                "safe": False,
                "reason": f"Unable to analyze content: {str(e)}",
                "recommendation": "Manual review required due to analysis error"
            }
    
    def check_url_safety(self, url: str) -> Dict[str, Any]:
        """
        Basic URL safety check
        """
        try:
            url_lower = url.lower()
            
            # Known safe domains (educational, news, etc.)
            safe_domains = [
                "wikipedia.org", "edu", "gov", "bbc.com", "reuters.com",
                "nationalgeographic.com", "smithsonianmag.com", "nature.com",
                "sciencedirect.com", "pubmed.ncbi.nlm.nih.gov"
            ]
            
            # Potentially concerning domains or patterns
            concerning_patterns = [
                r"\.onion$",  # Tor hidden services
                r"darkweb", r"deepweb",
                r"illegal", r"underground",
                r"hack", r"crack", r"exploit"
            ]
            
            # Check if it's a known safe domain
            is_safe_domain = any(domain in url_lower for domain in safe_domains)
            
            # Check for concerning patterns
            has_concerning_pattern = any(re.search(pattern, url_lower) for pattern in concerning_patterns)
            
            if is_safe_domain:
                return {
                    "safe": True,
                    "reason": "Known safe/educational domain",
                    "recommendation": "Safe to proceed"
                }
            elif has_concerning_pattern:
                return {
                    "safe": False,
                    "reason": "URL contains potentially concerning patterns",
                    "recommendation": "Human review recommended"
                }
            else:
                return {
                    "safe": True,
                    "reason": "No obvious safety concerns detected",
                    "recommendation": "Proceed with caution and verify content"
                }
                
        except Exception as e:
            return {
                "safe": False,
                "reason": f"Unable to analyze URL: {str(e)}",
                "recommendation": "Manual review required"
            }
    
    def add_concerning_keyword(self, keyword: str):
        """
        Add a new concerning keyword to the filter
        """
        if keyword.lower() not in self.concerning_keywords:
            self.concerning_keywords.append(keyword.lower())
    
    def add_acceptable_context(self, context: str):
        """
        Add a new acceptable context to the filter
        """
        if context.lower() not in self.acceptable_contexts:
            self.acceptable_contexts.append(context.lower())
    
    def get_filter_stats(self) -> Dict[str, int]:
        """
        Get statistics about the filter configuration
        """
        return {
            "concerning_keywords": len(self.concerning_keywords),
            "concerning_patterns": len(self.concerning_patterns),
            "acceptable_contexts": len(self.acceptable_contexts)
        }
