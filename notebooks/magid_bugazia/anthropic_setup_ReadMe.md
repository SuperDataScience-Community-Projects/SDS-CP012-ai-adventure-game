# AI Adventure Game Implementation

This is my implementation of the AI Adventure Game project using Anthropic's Claude API.

## Additional Requirements

In addition to the main project requirements, this implementation uses the Anthropic API. Install it using:



``` terminal
pip install anthropic
```

## Setup

Install requirements:
   ```bash
   pip install -r requirements.txt
   pip install anthropic
   ```
Create a `.env` file with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```
Run the test script:
   ```bash
   python test_LLM.api.py
   ```

## Implementation Notes

This implementation uses Claude 3 for generating game content and handling player interactions. The choice of Anthropic's API over OpenAI was made to explore alternative AI models and their capabilities in interactive storytelling.