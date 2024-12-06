import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Function to interact with the model
def get_response(prompt):
    response = model.invoke(prompt)
    return response.content

# Streamlit application
def main():
    st.title("Fantasy Adventure Game")
    st.write("Welcome to the Fantasy Adventure Game! 🌟")
    st.write("You are embarking on a unique journey where your choices shape the story.")

    # Initialize game summary
    if "game_summary" not in st.session_state:
        st.session_state["game_summary"] = {"actions": []}

    # Input form for player action
    with st.form("action_form"):
        player_action = st.text_input(
            "Describe your next action or decision, and the Dungeon Master will guide your journey:"
        )
        submitted = st.form_submit_button("Submit Action")

    if submitted and player_action:
        if player_action.lower() == "end":
            st.write("You have chosen to end your journey.")
            st.write("Summarizing your adventure...")

            # Generate a game summary prompt
            actions = st.session_state["game_summary"]["actions"]
            prompt = (
                f"Summarize the adventure of a player who made the following actions: "
                f"{', '.join(actions)}. Make the summary exciting, immersive, and reflective."
            )
            summary = get_response(prompt)
            st.write("### Adventure Summary:")
            st.write(summary)

            # Clear session state for a new game
            st.session_state["game_summary"] = {"actions": []}
        else:
            # Save action in summary
            st.session_state["game_summary"]["actions"].append(player_action)

            # Request model response based on action
            prompt = f"As the Dungeon Master, narrate what happens after the player decides to: {player_action}. Make it immersive and exciting."
            response = get_response(prompt)
            st.write("### Dungeon Master's Response:")
            st.write(response)

    elif submitted:
        st.warning("Please enter an action or type 'end' to conclude the game.")

if __name__ == "__main__":
    main()
