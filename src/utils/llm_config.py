"""
LLM configuration and utilities for the financial multi-agent system.
"""

import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    """Configuration class for LLM settings."""
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        google_api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.google_api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
    
    def create_llm(self) -> BaseChatModel:
        """Create and return a configured LLM instance."""
        kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "google_api_key": self.google_api_key
        }
        
        # Only add max_output_tokens if specified (Gemini uses different parameter name)
        if self.max_tokens:
            kwargs["max_output_tokens"] = self.max_tokens
            
        return ChatGoogleGenerativeAI(**kwargs)


# Default configurations for different use cases
CLASSIFIER_LLM_CONFIG = LLMConfig(
    model_name="gemini-2.5-flash",
    temperature=0.0,
)

RAG_LLM_CONFIG = LLMConfig(
    model_name="gemini-2.5-flash",
    temperature=0.1,
)

CHATBOT_LLM_CONFIG = LLMConfig(
    model_name="gemini-2.5-flash",
    temperature=0.2,
)


def get_classifier_llm() -> BaseChatModel:
    """Get LLM instance for classification tasks."""
    return CLASSIFIER_LLM_CONFIG.create_llm()

def get_rag_llm() -> BaseChatModel:
    """Get LLM instance for RAG tasks."""
    return RAG_LLM_CONFIG.create_llm()


def get_chatbot_llm() -> BaseChatModel:
    """Get LLM instance for chatbot tasks."""
    return CHATBOT_LLM_CONFIG.create_llm()
