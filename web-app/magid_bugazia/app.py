import streamlit as st
from app_engine import StreamlitGameEngine

class StreamlitGameInterface:
    def __init__(self):
        if 'game_engine' not in st.session_state:
            st.session_state.game_engine = StreamlitGameEngine()
    
    def render_interface(self):
        st.title("Welcome to the Adventure Game!")
        
        game = st.session_state.game_engine
        
        if 'game_started' not in st.session_state:
            self._handle_character_creation(game)
        else:
            self._render_game_interface(game)
    
    def _handle_character_creation(self, game):
        name = st.text_input("Enter your character's name:", key="character_name")
        
        st.write("Choose your character type:")
        character_types = {
            "1": ("The Mystic Sage - Master of ancient magic and forgotten lore", "Staff of Wisdom, Spellbook"),
            "2": ("The Warrior Knight - Skilled in combat and leadership", "Enchanted Sword, Shield"),
            "3": ("The Shadow Runner - Expert in stealth and deception", "Lockpicks, Smoke Bombs"),
            "4": ("The Nature Warden - Keeper of natural balance and animal bonds", "Druid's Charm, Healing Herbs")
        }
        
        selected_type = st.selectbox(
            "Select your character class:",
            options=["1", "2", "3", "4"],
            format_func=lambda x: character_types[x][0].split(" - ")[0]
        )

        if selected_type:
            desc, items = character_types[selected_type]
            st.write(f"Description: {desc}")
            st.write(f"Starting items: {items}")
        
        if st.button("Start Adventure"):
            if name.strip():
                try:
                    starting_scene = game.initialize_character(name, selected_type)
                    st.session_state.game_started = True
                    st.session_state.current_scene = starting_scene
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating character: {str(e)}")
            else:
                st.error("Please enter a character name.")
    
    def _render_game_interface(self, game):
        with st.sidebar:
            st.subheader("Character Information")
            st.write(f"Name: {game.player_info['name']}")
            st.write(f"Class: {game.player_info['character_type']}")
            st.write(f"Health: {game.player_info['health']}%")
            
            st.subheader("Inventory")
            for item in game.player_info["inventory"]:
                st.write(f"• {item}")
            
            if st.button("Show Help"):
                st.session_state.current_scene = self._format_help_text()
            
            if st.button("End Adventure"):
                ending_scene = game.process_action("end adventure")
                st.session_state.current_scene = ending_scene
                if st.button("Return to Character Creation"):
                    del st.session_state.game_started
                    st.rerun()
        
        if hasattr(st.session_state, 'current_scene'):
            st.markdown(st.session_state.current_scene)
        
        action = st.text_input("What would you like to do?", key="action_input")
        if st.button("Take Action"):
            if action.strip():
                try:
                    response = game.process_action(action)
                    st.session_state.current_scene = response
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing action: {str(e)}")
    
    def _format_help_text(self):
        return """
        **Available Commands:**
        - Type any action you want to perform
        - Check your inventory in the sidebar
        - View your character status in the sidebar
        - End your adventure using the sidebar button
        
        **Tips:**
        - Be specific in your actions
        - Use your character's unique abilities
        - Pay attention to your inventory items
        - Consider the environment in your choices
        """

def main():
    st.set_page_config(
        page_title="AI Adventure Game",
        layout="wide",
        page_icon="🎮"
    )
    
    interface = StreamlitGameInterface()
    interface.render_interface()

if __name__ == "__main__":
    main()