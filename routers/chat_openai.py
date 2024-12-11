from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, ClassVar
from pydantic import SecretStr
from .base_chat_provider import BaseChatProvider
from langchain.schema import BaseMessage

class ChatOpenAIProvider(BaseChatProvider, ChatOpenAI):
    """OpenAI-specific chat provider implementation"""
    
    SUPPORTED_MODELS: ClassVar[List[str]] = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
        "cloud-sambanova-llama-3-405b-instruct"
    ]
    
    def __init__(
        self,
        model_name: str,
        api_key: SecretStr,
        **kwargs: Any
    ):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Supported models: {', '.join(self.SUPPORTED_MODELS)}")
            
        super().__init__(
            model=model_name,
            api_key=api_key,
            **kwargs
        )
    
    async def agenerate_with_retry(self, messages: List[List[BaseMessage]], *args, **kwargs):
        return await self.agenerate(messages=messages, *args, **kwargs)
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        return cls.SUPPORTED_MODELS
    
    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "api_base": self.client.api_base,
            "timeout": self.request_timeout
        }
