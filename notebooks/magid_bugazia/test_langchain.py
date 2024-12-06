from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_langchain_anthropic():
    """Test LangChain with Anthropic's Claude"""
    try:
        # Initialize ChatAnthropic
        chat = ChatAnthropic(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-opus-20240229",
            temperature=0.7
        )
        
        # Create messages for a simple game scenario
        messages = [
            SystemMessage(content="""You are a game master for a fantasy text adventure. 
            Create immersive narratives and respond to player actions with vivid descriptions.
            Avoid common fantasy tropes and names. 
            Each world you create should have distinctive features that set it apart 
            from typical medieval fantasy settings."""),
            
            HumanMessage(content="""Create a unique fantasy world with original 
            geographical features we haven't seen before. The world should have an 
            unusual name and at least one completely unique characteristic that 
            players wouldn't find in other fantasy games. Describe the starting 
            location in this world.""")
        ]
        
        # Get response
        response = chat.invoke(messages)
        
        print("LangChain-Anthropic Connection Successful!")
        print("\nGame Introduction:")
        print(response.content)
        
        # Test conversation continuation
        messages.append(AIMessage(content=response.content))
        messages.append(HumanMessage(content="What dangers might I encounter in this area?"))
        
        follow_up = chat.invoke(messages)
        print("\nResponse to follow-up question:")
        print(follow_up.content)
        
        return True
        
    except Exception as e:
        print(f"Error testing LangChain with Anthropic: {str(e)}")
        return False

if __name__ == "__main__":
    test_langchain_anthropic()