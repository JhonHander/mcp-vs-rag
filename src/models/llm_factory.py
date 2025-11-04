from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()


def create_llm(model_name: str):
    """
    Factory function to create LLM instances.

    Supported models:
    - "gpt-5": OpenAI GPT-5 (released August 2025)
    - "gemini-2.5-pro": Google Gemini 2.5 Pro
    """

    if model_name == "gpt-5":
        return ChatOpenAI(
            model="gpt-5",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model_name == "gemini-2.5-pro":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Use 'gpt-5' or 'gemini-2.5-pro'")


def get_available_models():
    """Get list of available models."""
    return ["gpt-5", "gemini-2.5-pro"]
