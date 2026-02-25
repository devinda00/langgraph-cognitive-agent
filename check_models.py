# langgraph_cognitive_arch/check_models.py
import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio

async def main():
    """
    This script lists all generative models available to your API key
    that support the 'embedContent' method.
    """
    try:
        # Load the environment variables from .env file
        load_dotenv()

        # Configure the API key
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY not found in .env file.")
            return
            
        genai.configure(api_key=api_key)

        print("--- Checking available models for 'embedContent' ---")
        
        found_models = False
        for m in genai.list_models():
          if 'embedContent' in m.supported_generation_methods:
            print(f"Found compatible model: {m.name}")
            found_models = True
            
        if not found_models:
            print("\nNo models supporting 'embedContent' were found for your API key.")
            print("This might be due to project configuration or permissions issues.")
        else:
            print("\nPlease use one of the model names listed above in the permanent_knowledge.py file.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your API key is correct and that the 'Generative Language API' is enabled in your Google Cloud project.")

if __name__ == "__main__":
    asyncio.run(main())
