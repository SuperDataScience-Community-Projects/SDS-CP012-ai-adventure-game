import streamlit as st
from src.game_engine import GameEngine
from src.config import ChatConfig, ChatProvider
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
import logging
from typing import List
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

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
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None
if "game_active" not in st.session_state:
    st.session_state.game_active = False
if "use_free_version" not in st.session_state:
    st.session_state.use_free_version = False
if "turn_counter" not in st.session_state:
    st.session_state.turn_counter = 0

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

# Sidebar with game controls and API key input
with st.sidebar:
    st.header("Game Controls")

    # Add welcome message
    st.markdown("""
    Welcome to this game, let an AI be your storyteller! 
    While you'll see options in each message, feel free to respond however you want - 
    your adventure, your choices! 🎲✨
    """)
    
    # Add toggle for free version
    use_free_version = st.toggle("Use Free Version (Llama 3.3 405B)", value=st.session_state.use_free_version)
    st.session_state.use_free_version = use_free_version
    
    # Modify API key input section
    if not st.session_state.use_free_version:
        api_key = st.text_input("Enter your OpenAI API key:", 
                               value=os.getenv('OPENAI_API_KEY', ''),
                               type="password")
        if api_key:
            st.session_state.openai_api_key = api_key
    
    # Modify New Game button condition
    can_start_game = (st.session_state.use_free_version or st.session_state.openai_api_key)
    if can_start_game:
        if st.button("New Game"):
            logging.debug("=== Starting New Game ===")
            # Initialize game engine with appropriate config
            config = ChatConfig(
                provider=ChatProvider.LLAMA if st.session_state.use_free_version else ChatProvider.OPENAI,
                max_history=30,
                api_key=None if st.session_state.use_free_version else st.session_state.openai_api_key,
                base_url=os.getenv('PARASAIL_BASE_URL') if st.session_state.use_free_version else None
            )
            st.session_state.game_engine = GameEngine(config)
            
            # Initialize game to show character options only
            game_init = st.session_state.game_engine.initialize_game()
            
            # Reset UI state and show only the character selection prompt
            st.session_state.messages = [
                AIMessage(content=game_init["options"])
            ]
            st.session_state.game_active = True
            
            # Reset turn counter when starting new game
            st.session_state.turn_counter = 0
    else:
        st.info("Please either enable the free version or enter your OpenAI API key to start the game.")

    # Display turn counter in sidebar
    if st.session_state.game_active:
        st.metric("Turn", st.session_state.turn_counter)
        
        # Add token statistics
        if hasattr(st.session_state.game_engine, "get_token_stats"):
            stats = st.session_state.game_engine.get_token_stats()
            st.write("### Token Usage")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Input Tokens", stats["input_tokens"])
                st.metric("Output Tokens", stats["output_tokens"])
            with col2:
                st.metric("Total Tokens", stats["total_tokens"])
                st.metric("Est. Cost ($)", stats["estimated_cost"])

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
if st.session_state.game_active and "game_engine" in st.session_state:
    with st.form(key="user_input_form", clear_on_submit=True):
        user_input = st.text_input("Your response:")
        submit = st.form_submit_button("Send")
        
        if submit and user_input:
            logging.debug(f"Processing turn with input: {user_input}")
            
            # Increment turn counter
            st.session_state.turn_counter += 1
            
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

