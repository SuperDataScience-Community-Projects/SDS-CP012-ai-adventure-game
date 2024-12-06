import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from notebooks.magid_bugazia.config import config

from anthropic import Anthropic
from typing import List, Dict

class StreamlitGameEngine:
    def __init__(self):
        self.client = Anthropic(api_key=config.api_key)
        self.conversation_history: List[Dict] = []
        self.player_info: Dict = {
            "inventory": ["basic supplies"],
            "health": 100,
            "location": "",
            "character_type": "",
        }
        
    def initialize_character(self, name: str, character_type: str) -> str:
        """Initialize a new character and return the starting scene"""
        character_types = {
            "1": ("Mystic Sage", ["Staff of Wisdom", "Spellbook"]),
            "2": ("Warrior Knight", ["Enchanted Sword", "Shield"]),
            "3": ("Shadow Runner", ["Lockpicks", "Smoke Bombs"]),
            "4": ("Nature Warden", ["Druid's Charm", "Healing Herbs"])
        }
        
        if character_type not in character_types:
            raise ValueError("Invalid character type selected")
            
        self.player_info["name"] = name
        char_type, items = character_types[character_type]
        self.player_info["character_type"] = char_type
        self.player_info["inventory"].extend(items)
        
        return self._generate_starting_scene()
    
    def _generate_starting_scene(self) -> str:
        """Generate the initial game scene"""
        prompt = f"""
        You are {self.player_info['name']}, a {self.player_info['character_type']}.
        In your possession you have: {', '.join(self.player_info['inventory'])}.
        
        Create a unique starting location and initial situation that would be interesting
        for a {self.player_info['character_type']}. Describe the scene vividly and present
        3-4 meaningful choices that showcase the character's abilities.
        """
        return self._get_ai_response(prompt)
    
    def process_action(self, action: str) -> str:
        """Process a player action and return the result"""
        if action.lower() == "end adventure":
            return self._generate_ending()
            
        context = f"""
        Character: {self.player_info['name']} - {self.player_info['character_type']}
        Current inventory: {', '.join(self.player_info['inventory'])}
        Health: {self.player_info['health']}%
        {self.player_info['name']} attempts to: {action}
        """
        return self._get_ai_response(context)
    
    def _generate_ending(self) -> str:
        """Generate the game ending summary"""
        prompt = f"""
        Create a brief but epic summary of {self.player_info['name']}'s adventure as a 
        {self.player_info['character_type']}. Reference their key actions and decisions 
        from the conversation history.
        """
        return self._get_ai_response(prompt)
    
    def _get_ai_response(self, prompt: str) -> str:
        """Get response from Claude with game-specific system prompt"""
        system_prompt = """
        You are running an immersive fantasy adventure game. For each response:
        1. Provide vivid descriptions of the environment and situation
        2. React to the player's choices meaningfully
        3. Always end with 3-4 clear choices for the player
        4. Keep track of their inventory and abilities
        5. Create interesting challenges that suit their character type
        6. Never mention that you are an AI or that this is a game
        """
        
        message = self.client.messages.create(
            model=config.model_name,
            max_tokens=1024,
            temperature=0.8,
            system=system_prompt,
            messages=[
                *self.conversation_history,
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Store conversation for context
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": message.content[0].text})
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
            
        return message.content[0].text