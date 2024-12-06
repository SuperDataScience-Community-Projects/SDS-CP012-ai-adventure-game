from anthropic import Anthropic
from config import config

def test_anthropic_connection():
    """Test connection to Anthropic API"""
    try:
        client = Anthropic(api_key=config.api_key)
        
        # Test API with a simple message
        message = client.messages.create(
            model=config.model_name,
            max_tokens=1024,
            temperature=config.temperature,
            system=config.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": "Create a brief introduction for our fantasy adventure game. \
                        This world should not be a typical medieval fantasy setting. Describe its most striking characteristic."
                }
            ]
        )
        
        print("API Connection Successful!")
        print(f"Response: \n{message.content[0].text}\n")
        return True
        
    except Exception as e:
        print(f"Error connecting to Anthropic API: {str(e)}")
        return False

if __name__ == "__main__":
    test_anthropic_connection()