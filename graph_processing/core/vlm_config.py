"""
VLM Configuration Module for Vision Pipeline
Supports multiple VLM providers including Gemma3-27b and GLM 4.1
"""

import os
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class VLMProvider(Enum):
    """Supported VLM providers"""
    GEMMA3_27B = "gemma3-27b"
    GLM_4_1 = "glm-4.1"
    GPT4O = "gpt-4o"
    GPT4O_VISION = "gpt-4o-2024-11-20"


@dataclass
class VLMConfig:
    """Configuration for a VLM provider"""
    provider: VLMProvider
    api_url: str
    api_key: str
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.1
    supports_structured_output: bool = True
    timeout: int = 120  # seconds


class VLMManager:
    """Manager for VLM providers with fallback support"""
    
    def __init__(self):
        self.configs: Dict[VLMProvider, VLMConfig] = {}
        self.current_provider: Optional[VLMProvider] = None
        self._load_configs()
    
    def _load_configs(self):
        """Load VLM configurations from environment variables"""
        
        # Gemma3-27b configuration
        self.configs[VLMProvider.GEMMA3_27B] = VLMConfig(
            provider=VLMProvider.GEMMA3_27B,
            api_url=os.getenv("GEMMA_API_URL", "http://77.234.216.102:17640/19002/v1"),
            api_key=os.getenv("GEMMA_API_KEY", "sk-t72jJMWW1YVjjJILvbDSDrnRBLdyu"),
            model_name="gemma3-27b",
            supports_structured_output=False  # May not support OpenAI's structured output
        )
        
        # GLM 4.1 configuration
        glm_port = os.getenv("GLM_PORT", "2")
        self.configs[VLMProvider.GLM_4_1] = VLMConfig(
            provider=VLMProvider.GLM_4_1,
            api_url=os.getenv("GLM_API_URL", f"http://77.234.216.102:{glm_port}/v1"),
            api_key=os.getenv("GLM_API_KEY", ""),  # Need to set this
            model_name="glm-4.1",
            supports_structured_output=False
        )
        
        # GPT-4o configuration (fallback)
        if os.getenv("OPENAI_API_KEY"):
            self.configs[VLMProvider.GPT4O] = VLMConfig(
                provider=VLMProvider.GPT4O,
                api_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model_name="gpt-4o",
                supports_structured_output=True
            )
            
            self.configs[VLMProvider.GPT4O_VISION] = VLMConfig(
                provider=VLMProvider.GPT4O_VISION,
                api_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model_name="gpt-4o-2024-11-20",
                supports_structured_output=True
            )
    
    def get_client(self, provider: Optional[VLMProvider] = None):
        """Get OpenAI-compatible client for specified provider"""
        from openai import OpenAI
        
        if provider is None:
            provider = self.current_provider or VLMProvider.GEMMA3_27B
        
        if provider not in self.configs:
            raise ValueError(f"Provider {provider} not configured")
        
        config = self.configs[provider]
        
        # Check if API key is set for non-default providers
        if not config.api_key and provider != VLMProvider.GPT4O:
            logger.warning(f"API key not set for {provider.value}, using without authentication")
        
        client = OpenAI(
            base_url=config.api_url,
            api_key=config.api_key or "dummy-key"  # Some servers don't require auth
        )
        
        self.current_provider = provider
        logger.info(f"Using VLM provider: {provider.value} at {config.api_url}")
        
        return client, config
    
    def get_fallback_providers(self) -> list[VLMProvider]:
        """Get list of providers for fallback strategy"""
        providers = []
        
        # Priority order: Gemma3 -> GLM 4.1 -> GPT-4o
        if VLMProvider.GEMMA3_27B in self.configs:
            providers.append(VLMProvider.GEMMA3_27B)
        if VLMProvider.GLM_4_1 in self.configs:
            providers.append(VLMProvider.GLM_4_1)
        if VLMProvider.GPT4O in self.configs:
            providers.append(VLMProvider.GPT4O)
        
        return providers
    
    def execute_with_fallback(self, func, *args, **kwargs):
        """Execute function with automatic fallback to other providers"""
        providers = self.get_fallback_providers()
        last_error = None
        
        for provider in providers:
            try:
                logger.info(f"Trying provider: {provider.value}")
                client, config = self.get_client(provider)
                
                # Pass client and config to the function
                result = func(client, config, *args, **kwargs)
                logger.info(f"Success with provider: {provider.value}")
                return result
                
            except Exception as e:
                logger.error(f"Failed with provider {provider.value}: {str(e)}")
                last_error = e
                continue
        
        if last_error:
            raise RuntimeError(f"All providers failed. Last error: {str(last_error)}")
        else:
            raise RuntimeError("No providers available")


# Singleton instance
vlm_manager = VLMManager()


def get_vlm_client(provider: Optional[VLMProvider] = None):
    """Convenience function to get VLM client"""
    return vlm_manager.get_client(provider)


def test_vlm_connection(provider: VLMProvider):
    """Test connection to a VLM provider"""
    try:
        client, config = vlm_manager.get_client(provider)
        
        # Simple test query
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "user", "content": "Say 'Hello, connection successful!' in 5 words or less"}
            ],
            max_tokens=50,
            temperature=0
        )
        
        result = response.choices[0].message.content
        logger.info(f"Test successful for {provider.value}: {result}")
        return True, result
        
    except Exception as e:
        logger.error(f"Test failed for {provider.value}: {str(e)}")
        return False, str(e)


if __name__ == "__main__":
    # Test connections when module is run directly
    logging.basicConfig(level=logging.INFO)
    
    print("Testing VLM connections...")
    for provider in VLMProvider:
        if provider in vlm_manager.configs:
            success, message = test_vlm_connection(provider)
            print(f"{provider.value}: {'✓' if success else '✗'} - {message}")