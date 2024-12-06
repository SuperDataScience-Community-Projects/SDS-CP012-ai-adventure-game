from dotenv import load_dotenv
import os
from dataclasses import dataclass
from typing import Optional

# Load environment variables
load_dotenv()

@dataclass
class GameConfig:
    """Configuration for the adventure game"""
    model_name: str = 'claude-3-opus-20240229' #"claude-3-haiku-20240307"
    max_history: int = 10
    temperature: float = 0.7
    system_prompt: str = """You are an immersive storyteller for a text-based fantasy adventure game. 
    Create engaging narratives with vivid descriptions and meaningful choices for the player. Use a dynamic type of engagement.
    Generate unique and original content, avoiding common fantasy tropes."""
    
    @property
    def api_key(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY")

# Default configuration
config = GameConfig()