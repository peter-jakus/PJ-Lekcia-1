import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def count_characters(text: str) -> Dict[str, Any]:
    """
    Tool function to count characters in text
    Returns detailed character analysis
    """
    return {
        "total_characters": len(text),
        "characters_without_spaces": len(text.replace(" ", "")),
        "spaces": text.count(" "),
        "words": len(text.split()),
        "lines": len(text.split("\n"))
    }

def call_llm_with_tool(user_prompt: str) -> str:
    """
    Main function that:
    1. Calls LLM to generate text
    2. Uses tool to analyze the generated text
    3. Returns final response with analysis
    """
    # Step 1: Generate text based on user prompt
    print("🤖 Generating text...")
    messages = [
        {"role": "system", "content": "You are a creative writer. Generate interesting text based on user requests."},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        generated_text = response.choices[0].message.content
        print(f"✅ Generated text:\n{generated_text}\n")
        # Step 2: Use tool to count characters
        print("🔧 Analyzing text with character counter tool...")
        char_analysis = count_characters(generated_text)
        print(f"✅ Character analysis: {char_analysis}\n")
        # Step 3: Send results back to LLM for final response
        print("🤖 Creating final response...")
        final_messages = [
            {"role": "system", "content": "You are an assistant that provides text analysis. Present the results in a user-friendly way."},
            {"role": "user", "content": f"""
                I asked you to write: \"{user_prompt}\"
                \nYou generated this text: \"{generated_text}\"
                \nThe character analysis tool returned: {json.dumps(char_analysis, indent=2)}\n\nPlease provide a summary of the generated text and its character analysis.
                """}
        ]
        final_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=final_messages,
            max_tokens=300,
            temperature=0.3
        )
        return final_response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """
    Interactive main function
    """
    print("=== LLM Text Generator with Character Counter ===\n")
    while True:
        user_input = input("Enter your text generation prompt (or 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        if not user_input.strip():
            print("Please enter a valid prompt.\n")
            continue
        print(f"\n--- Processing: '{user_input}' ---")
        result = call_llm_with_tool(user_input)
        print("🎉 Final Result:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        print()

if __name__ == "__main__":
    main()

# Example usage for testing without interactive mode:
def test_example():
    """
    Test function with example
    """
    example_prompt = "Write a short poem about programming"
    result = call_llm_with_tool(example_prompt)
    print("Test Result:", result)

