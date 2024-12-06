from anthropic import Anthropic
from notebooks.magid_bugazia.config import config
from typing import List, Dict

class AdventureGame:
    def __init__(self):
        self.client = Anthropic(api_key=config.api_key)
        self.conversation_history: List[Dict] = []
        self.player_info: Dict = {
            "inventory": ["basic supplies"],
            "health": 100,
            "location": "",
            "character_type": "",
        }
        
    def start_game(self):
        """Initialize the game and present character creation options"""
        print("\n=== Welcome to the Adventure Game! ===\n")
        name = input("Enter your character's name: ").strip()
        self.player_info["name"] = name
        
        initial_prompt = f"""
        Welcome, {name}! Please choose your character type:
        
        1. The Mystic Sage - Master of ancient magic and forgotten lore
           Starting items: Staff of Wisdom, Spellbook
           
        2. The Warrior Knight - Skilled in combat and leadership
           Starting items: Enchanted Sword, Shield
           
        3. The Shadow Runner - Expert in stealth and deception
           Starting items: Lockpicks, Smoke Bombs
           
        4. The Nature Warden - Keeper of natural balance and animal bonds
           Starting items: Druid's Charm, Healing Herbs
        
        Enter your choice (1-4):
        """
        print(initial_prompt)
        
        choice = input().strip()
        self._handle_character_selection(choice)
    
    def _handle_character_selection(self, choice: str):
        """Process character selection and start the adventure"""
        character_types = {
            "1": ("Mystic Sage", ["Staff of Wisdom", "Spellbook"]),
            "2": ("Warrior Knight", ["Enchanted Sword", "Shield"]),
            "3": ("Shadow Runner", ["Lockpicks", "Smoke Bombs"]),
            "4": ("Nature Warden", ["Druid's Charm", "Healing Herbs"])
        }
        
        if choice not in character_types:
            print("Invalid choice. Please select 1-4.")
            return self.start_game()
            
        char_type, items = character_types[choice]
        self.player_info["character_type"] = char_type
        self.player_info["inventory"].extend(items)
        self._start_adventure()
    
    def _start_adventure(self):
        """Begin the main adventure after character selection"""
        adventure_prompt = f"""
        You are {self.player_info['name']}, a {self.player_info['character_type']}.
        In your possession you have: {', '.join(self.player_info['inventory'])}.
        
        Create a unique starting location and initial situation that would be interesting
        for a {self.player_info['character_type']}. Describe the scene vividly and present
        3-4 meaningful choices that showcase the character's abilities.
        """
        
        response = self._get_ai_response(adventure_prompt)
        print("\n" + response + "\n")
        self._game_loop()
    
    def _get_ai_response(self, prompt: str) -> str:
        """Get response from Claude"""
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
                # note list of user vs. ai history, AI will reply to latest prompt of user
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
    
    def _game_loop(self):
        """Main game loop"""
        while True:
            action = input("\nWhat would you like to do? (type 'help' for commands, 'inventory' to check items, 'status' for character info, or 'quit' to exit): ").strip().lower()
            
            if action == 'quit':
                self._end_game()
                break
            
            if action == 'help':
                self._show_help()
                continue
                
            if action == 'inventory':
                self._show_inventory()
                continue
            
            if action == 'status':
                self._show_status()
                continue

             # Enhanced context for AI response
            context = f"""
            Character: {self.player_info['name']} - {self.player_info['character_type']}
            Current inventory: {', '.join(self.player_info['inventory'])}
            Health: {self.player_info['health']}%
            {self.player_info['name']} attempts to: {action}
            """
                
            response = self._get_ai_response(context)
            print("\n" + response + "\n")
    
    def _show_help(self):
        """Display available commands"""
        print("\nAvailable Commands:")
        print("- 'inventory': Check your items")
        print("- 'status': View your character info")
        print("- 'help': Show this help message")
        print("- 'quit': End the game")
        print("- Any other input will be treated as an action for your character")
    
    def _show_inventory(self):
        """Display inventory contents"""
        print("\nInventory:")
        for item in self.player_info["inventory"]:
            print(f"- {item}")
    
    def _show_status(self):
        """Display character status"""
        print("\nCharacter Status:")
        print(f"Name: {self.player_info['name']}")
        print(f"Class: {self.player_info['character_type']}")
        print(f"Health: {self.player_info['health']}%")
    
    def _end_game(self):
        """Handle game ending with a summary"""
        final_prompt = f"""
        Create a brief but epic summary of {self.player_info['name']}'s adventure as a 
        {self.player_info['character_type']}. Reference their key actions and decisions 
        from the conversation history.
        """
        summary = self._get_ai_response(final_prompt)
        print("\nYour Adventure Summary:")
        print(summary)
        print("\nThanks for playing! Farewell, brave adventurer.")

if __name__ == "__main__":
    game = AdventureGame()
    game.start_game()