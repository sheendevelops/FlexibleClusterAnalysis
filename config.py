import os
from typing import Literal

# LLM Backend Configuration
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()  # "ollama" or "openai"

# Ollama Configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

# OpenAI Configuration (fallback)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# Database Configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "")

def get_llm_helper():
    """
    Factory function to get the appropriate LLM helper based on configuration
    """
    if LLM_BACKEND == "openai" and OPENAI_API_KEY:
        from utils.openai_helper import OpenAIHelper
        return OpenAIHelper()
    else:
        # Default to Ollama for local operation
        from utils.ollama_helper import OllamaHelper
        return OllamaHelper()

def get_config_info():
    """
    Get current configuration information
    """
    return {
        "llm_backend": LLM_BACKEND,
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "openai_model": OPENAI_MODEL,
        "openai_configured": bool(OPENAI_API_KEY),
        "database_configured": bool(DATABASE_URL)
    }