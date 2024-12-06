from enum import Enum
from pydantic import SecretStr
from utils.utils import get_api_key
from typing import Optional
from routers.chat_openai import ChatOpenAIProvider
from routers.chat_openrouter import ChatOpenRouter

class ChatProvider(Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"

class ChatConfig:
    """Configuration class for chat parameters"""
    def __init__(self, 
                 provider: ChatProvider = ChatProvider.OPENROUTER,
                 openrouter_model: str = "gryphe/mythomax-l2-13b:free",
                 openai_model: str = "gpt-4o-mini",
                 system_prompt_path: str = "templates/system_prompt.md",
                 max_history: int = 10,
                 api_key: Optional[str] = None):
        
        self.provider = provider
        self.openrouter_model = openrouter_model
        self.openai_model = openai_model
        self.system_prompt_path = system_prompt_path
        self.max_history = max_history
        self.api_key = api_key
    def get_model_name(self) -> str:
        """Get the appropriate model name based on provider"""
        return self.openrouter_model if self.provider == ChatProvider.OPENROUTER else self.openai_model

    def get_api_key(self) -> SecretStr:
        """Get the appropriate API key based on provider"""
        api_key_env = 'OPENROUTER_API_KEY' if self.provider == ChatProvider.OPENROUTER else 'OPENAI_API_KEY'
        return SecretStr(get_api_key(api_key_env)) 
    
    def get_chat_provider(self, **kwargs):
        """Get the appropriate chat provider instance based on configuration"""
        model_name = self.get_model_name()
        api_key = self.get_api_key() if self.api_key is None else SecretStr(self.api_key)

        if self.provider == ChatProvider.OPENAI:
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