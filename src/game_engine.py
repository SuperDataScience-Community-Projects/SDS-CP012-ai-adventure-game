from typing import List, Optional
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import logging
from .config import ChatConfig
import json
from langchain.callbacks import get_openai_callback

class GameEngine:
    def __init__(self, config: ChatConfig):
        self.config = config
        self.messages: List[BaseMessage] = []
        self.storyteller = config.get_chat_provider()

        # initialize state message
        self.state_message = None
        
        # Initialize chains
        self._setup_chains()
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _setup_chains(self):
        """Setup the various processing chains"""
        # Character options chain
        character_prompt = ChatPromptTemplate.from_messages([
            ("system", self._load_prompt(self.config.system_prompt_path)),
            ("human", self._load_prompt("templates/character_setting_setup.md"))
        ])
        self.character_chain = character_prompt | self.storyteller | StrOutputParser()
        
        # Story continuation chain
        story_prompt = ChatPromptTemplate.from_messages([
            ("system", self._load_prompt(self.config.system_prompt_path)),
            ("human", "Previous conversation:\n{history}\n\nCurrent state:\n{state_message}\n\nCurrent input:\n{user_input}")
        ])
        self.story_chain = story_prompt | self.storyteller | StrOutputParser()

        # State extraction chain
        state_prompt = ChatPromptTemplate.from_messages([
            ("system", self._load_prompt(self.config.system_prompt_path)),
            ("human", "{story_text} \n Extract the current state of the story.")
        ])
        self.state_chain = state_prompt | self.storyteller | StrOutputParser()

    def _load_prompt(self, path: str) -> str:
        """Load prompt from file"""
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except FileNotFoundError as e:
            logging.error(f"Prompt file not found: {path}")
            raise

    def _format_conversation_history(self, skip_system: bool = True, start_idx: int = 1) -> str:
        """Format conversation history into a string.
        
        Args:
            skip_system: Whether to skip the system message (default: True)
            start_idx: Starting index for messages to include (default: 1)
            
        Returns:
            Formatted conversation history string
        """
        messages_to_format = self.messages[start_idx:] if skip_system else self.messages
        return "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages_to_format
        ])

    def initialize_game(self, character_selection: Optional[str] = None):
        """Setup initial game state and prompts"""
        # Get character options using the character chain
        options_text = self.character_chain.invoke({})
        
        # Store initial messages
        self.messages = [
            SystemMessage(content=self._load_prompt(self.config.system_prompt_path)),
            HumanMessage(content=self._load_prompt("templates/character_setting_setup.md")),
            AIMessage(content=options_text)
        ]
        
        # If no character selection provided, just return the options
        if not character_selection:
            return {
                "options": options_text,
                "initial_story": ""
            }
        
        # Always add the character selection to messages
        self.messages.extend([
            HumanMessage(content=character_selection)
        ])

        
        # Only proceed with story generation if character_selection isn't "Start the adventure!"
        if character_selection != "Start the adventure!":
            # Add start command and generate initial story
            self.messages.append(HumanMessage(content="Start the adventure with the selected character and setting!"))
            
            # Use the new utility method
            history = self._format_conversation_history(start_idx=1)  # Skip system message
            
            # Generate initial story response
            initial_story = self.story_chain.invoke({
                "history": history,
                "state_message": self.state_message,
                "user_input": self.messages[-1].content
            })
            self.messages.append(AIMessage(content=initial_story))
            
            return {
                "options": options_text,
                "initial_story": initial_story
            }
        
        # If it's just the initial "Start the adventure!" command, return only options
        return {
            "options": options_text,
            "initial_story": ""
        }

    def process_turn(self, user_input: str) -> str:
        """Process a single game turn (UI version)"""
        try:
            # Add user input to messages
            self.messages.append(HumanMessage(content=user_input))

            # Track tokens using callback
            with get_openai_callback() as cb:
                # Generate story continuation with history
                history = self._format_conversation_history(skip_system=True)
                story_text = self.story_chain.invoke({
                    "history": history,
                    "state_message": self.state_message,
                    "user_input": self.messages[-1].content
                })
                
                # Extract the current state from the story text
                current_state = self.state_chain.invoke({
                    "story_text": history + "\n\n" + story_text
                })

                # Update token counts
                self.total_input_tokens += cb.prompt_tokens
                self.total_output_tokens += cb.completion_tokens

            #print(f"\n#########################\nCurrent state: {current_state}\n#########################\n")
            
            # update state message
            self.state_message = current_state
            
            # Add AI response to messages
            self.messages.append(AIMessage(content=story_text))
            
            # Maintain conversation history
            if len(self.messages) > self.config.max_history:
                # Keep system message and at least the character selection messages
                min_messages_to_keep = 4  # system + character setup + selection + initial story
                keep_count = max(min_messages_to_keep, self.config.max_history)
                self.messages = [self.messages[0]] + self.messages[-keep_count:]
            
            return story_text
            
        except Exception as e:
            logging.error(f"Error processing turn: {str(e)}", exc_info=True)
            raise

    def get_token_stats(self) -> dict:
        """Get token usage statistics"""
        costs = self.config.get_token_costs()
        input_cost = (self.total_input_tokens / 1000) * costs["input"]
        output_cost = (self.total_output_tokens / 1000) * costs["output"]
        
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost": round(input_cost + output_cost, 4)
        }

    async def run_game_loop(self):
        """Main game loop (Terminal version)"""
        try:
            # Initialize game
            game_init = self.initialize_game()
            print(game_init["options"])
            
            # Get character selection
            character_selection = input("Choose your character and setting: ")
            game_init = self.initialize_game(character_selection)
            print("\nStarting adventure...\n")
            print(game_init["initial_story"])
            
            while True:
                # Get player input
                user_input = input("\nWhat would you like to do? (or type 'quit' to end): ")
                
                if user_input.lower() == 'quit':
                    print("\nThanks for playing!")

                    # save the messages to a file
                    with open("messages.json", "w") as f:
                        json.dump([{"content": message.content} for message in self.messages], f)
                    break
                
                # Process turn and display result
                story_text = self.process_turn(user_input)
                print("\n" + story_text)
                
        except Exception as e:
            print(f"An error occurred: {e}")
            logging.error(f"Error in game loop: {str(e)}", exc_info=True)