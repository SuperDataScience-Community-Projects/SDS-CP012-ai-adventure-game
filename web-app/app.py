import streamlit as st
from src.game_engine import GameEngine
from src.config import ChatConfig, ChatProvider
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
import logging
from typing import List

# Configure logging to write to a file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('game_debug.log'),
        logging.StreamHandler()  # This will also show logs in terminal
    ]
)
 
# Add a test log message to verify logging is working
logging.debug("Streamlit app started")

# Page config
st.set_page_config(
    page_title="AI Text Adventure Game",
    page_icon="🎮",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  
if "game_engine" not in st.session_state:
    config = ChatConfig(
        provider=ChatProvider.OPENAI,
        max_history=10
    )
    st.session_state.game_engine = GameEngine(config)
if "game_active" not in st.session_state:
    st.session_state.game_active = False

# Custom CSS
st.markdown("""
    <style>
    .message-container {
        padding: 10px;
        margin: 5px;
        border-radius: 15px;
    }
    .ai-message {
        background-color: #2b313e;
        margin-right: 20%;
    }
    .user-message {
        background-color: #0e4da4;
        margin-left: 20%;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🎮 AI Text Adventure Game")

# Sidebar with game controls
with st.sidebar:
    st.header("Game Controls")
    if st.button("New Game"):
        logging.debug("=== Starting New Game ===")
        # Reset game engine with fresh config
        config = ChatConfig(
            provider=ChatProvider.OPENAI,
            max_history=30
        )
        st.session_state.game_engine = GameEngine(config)
        
        # Initialize game to show character options only
        game_init = st.session_state.game_engine.initialize_game()
        
        # Reset UI state and show only the character selection prompt
        st.session_state.messages = [
            AIMessage(content=game_init["options"])
        ]
        st.session_state.game_active = True

# Message display area
message_container = st.container()
with message_container:
    for message in st.session_state.messages:
        is_ai = isinstance(message, AIMessage)
        div_class = "ai-message" if is_ai else "user-message"
        with st.container():
            st.markdown(f"""
                <div class="message-container {div_class}">
                    {message.content}
                </div>
            """, unsafe_allow_html=True)

# Game input form
if st.session_state.game_active:
    with st.form(key="user_input_form", clear_on_submit=True):
        user_input = st.text_input("Your response:")
        submit = st.form_submit_button("Send")
        
        if submit and user_input:
            logging.debug(f"Processing turn with input: {user_input}")
            
            # Add user message to UI
            st.session_state.messages.append(HumanMessage(content=user_input)) # type: ignore
            
            # Process turn using game engine
            try:
                ai_response = st.session_state.game_engine.process_turn(user_input)
                st.session_state.messages.append(AIMessage(content=ai_response))
                
                # Keep UI messages in sync with max history
                if len(st.session_state.messages) > st.session_state.game_engine.config.max_history:
                    st.session_state.messages = st.session_state.messages[-(st.session_state.game_engine.config.max_history):]
                
            except Exception as e:
                logging.error(f"Error processing turn: {str(e)}", exc_info=True)
                st.error("An error occurred while processing your input. Please try again.")
            
            st.rerun()
else:
    st.info("Click 'New Game' in the sidebar to start your adventure!")

