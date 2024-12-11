from enum import Enum
from pydantic import SecretStr
from utils.utils import get_api_key
from typing import Optional
from routers.chat_openai import ChatOpenAIProvider
from routers.chat_openrouter import ChatOpenRouter
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class ChatProvider(Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    LLAMA = "llama"

class ChatConfig:
    """Configuration class for chat parameters"""
    def __init__(self, 
                 provider: ChatProvider = ChatProvider.OPENROUTER,
                 openrouter_model: str = "gryphe/mythomax-l2-13b:free",
                 openai_model: str = "gpt-4o-mini",
                 llama_model: str = "cloud-sambanova-llama-3-405b-instruct",
                 system_prompt_path: str = "templates/system_prompt.md",
                 max_history: int = 10,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        
        # Load environment variables if not already loaded
        if not os.getenv('OPENAI_API_KEY'):
            load_dotenv()
            
        self.provider = provider
        self.openrouter_model = openrouter_model
        self.openai_model = openai_model
        self.llama_model = llama_model
        self.system_prompt_path = system_prompt_path
        self.max_history = max_history
        self.api_key = api_key
        self.base_url = base_url
        self.input_tokens = 0
        self.output_tokens = 0

    def get_api_key(self) -> SecretStr:
        """Get the appropriate API key based on provider"""
        if self.api_key:
            return SecretStr(self.api_key)
            
        # Get from environment variables
        api_key_map = {
            ChatProvider.OPENROUTER: 'OPENROUTER_API_KEY',
            ChatProvider.OPENAI: 'OPENAI_API_KEY',
            ChatProvider.LLAMA: 'PARASAIL_API_KEY'
        }
        env_key = os.getenv(api_key_map[self.provider])
        if not env_key:
            raise ValueError(f"Missing API key for provider {self.provider}")
        return SecretStr(env_key)

    def get_base_url(self) -> Optional[str]:
        """Get the base URL if needed"""
        if self.base_url:
            return self.base_url
        if self.provider == ChatProvider.LLAMA:
            base_url = os.getenv('PARASAIL_BASE_URL')
            if not base_url:
                raise ValueError("Missing PARASAIL_BASE_URL in environment variables")
            return base_url
        return None

    def get_model_name(self) -> str:
        """Get the appropriate model name based on provider"""
        if self.provider == ChatProvider.OPENROUTER:
            return self.openrouter_model
        elif self.provider == ChatProvider.LLAMA:
            return self.llama_model
        return self.openai_model

    def get_chat_provider(self, **kwargs):
        """Get the appropriate chat provider instance based on configuration"""
        model_name = self.get_model_name()
        api_key = self.get_api_key()
        base_url = self.get_base_url()

        if self.provider == ChatProvider.LLAMA:
            return ChatOpenAIProvider(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                **kwargs
            )
        elif self.provider == ChatProvider.OPENAI:
            return ChatOpenAIProvider(
                model_name=model_name,
                api_key=api_key,
                **kwargs
            )
        elif self.provider == ChatProvider.OPENROUTER:
            return ChatOpenRouter(
                model_name=model_name,
                api_key=api_key,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_token_costs(self) -> dict:
        """Get the cost per 1K tokens for the current model"""
        costs = {
            ChatProvider.OPENAI: {
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-4o": {"input": 0.005, "output": 0.015},
                "o1-mini": {"input": 0.003, "output": 0.012},
                "o1": {"input": 0.015, "output": 0.06}
            },
            ChatProvider.LLAMA: {"input": 0.0, "output": 0.0},  # Free
            ChatProvider.OPENROUTER: {"input": 0.001, "output": 0.002}  # Example costs
        }
        return costs.get(self.provider, {"input": 0.0, "output": 0.0})