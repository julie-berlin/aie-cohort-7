"""Production model utilities for OpenAI integration."""

import os
from typing import Optional

from langchain_openai import ChatOpenAI


def get_openai_model(
    model_name: Optional[str] = None, 
    temperature: float = 0.1,
    max_tokens: Optional[int] = None
) -> ChatOpenAI:
    """Get a configured OpenAI model instance.
    
    Args:
        model_name: Model name to use. Defaults to env var OPENAI_MODEL or "gpt-4.1-mini"
        temperature: Sampling temperature 
        max_tokens: Maximum tokens to generate
        
    Returns:
        Configured ChatOpenAI instance
    """
    name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    
    kwargs = {
        "model": name,
        "temperature": temperature,
    }
    
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
        
    return ChatOpenAI(**kwargs)

